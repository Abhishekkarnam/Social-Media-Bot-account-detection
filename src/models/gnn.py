from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor, edge_index=None, edge_type=None) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


def _mean_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.layout != torch.strided:
        return torch.sparse.mm(edge_index, x)
    result = torch.zeros_like(x)
    if edge_index.numel() == 0:
        return result
    source, target = edge_index
    result.index_add_(0, target, x[source])
    counts = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
    counts.index_add_(0, target, torch.ones((target.numel(), 1), dtype=x.dtype, device=x.device))
    return result / counts.clamp_min_(1)


def normalized_adjacency(edge_index: torch.Tensor, num_nodes: int, edge_mask=None,
                         edge_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Create A[target, source] row-normalized for sparse mean aggregation."""
    if edge_mask is not None:
        edge_index = edge_index[:, edge_mask]
        if edge_weight is not None: edge_weight = edge_weight[edge_mask]
    if edge_index.numel() == 0:
        indices = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        values = torch.empty((0,), dtype=torch.float32, device=edge_index.device)
        return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()
    source, target = edge_index
    weights = (torch.ones(source.numel(), dtype=torch.float32, device=edge_index.device)
               if edge_weight is None else edge_weight.to(device=edge_index.device, dtype=torch.float32))
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("edge weights must be finite and non-negative")
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    degree.index_add_(0, target, weights)
    values = weights / degree[target].clamp_min_(1e-12)
    return torch.sparse_coo_tensor(torch.stack((target, source)), values, (num_nodes, num_nodes)).coalesce()


class GraphSAGE(nn.Module):
    """Full-batch mean GraphSAGE; suitable for small/medium graphs, not million-node GPU loads."""
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.self1 = nn.Linear(in_dim, hidden_dim)
        self.neighbor1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.self2 = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type=None) -> torch.Tensor:
        h = torch.relu(self.self1(x) + self.neighbor1(_mean_aggregate(x, edge_index)))
        h = self.dropout(h)
        h = torch.relu(self.self2(h) + self.neighbor2(_mean_aggregate(h, edge_index)))
        return self.out(self.dropout(h)).squeeze(-1)


class RelationalGNN(nn.Module):
    """Two-layer relational mean aggregator with a separate transform for each edge type."""
    def __init__(self, in_dim: int, num_relations: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        if num_relations < 1:
            raise ValueError("num_relations must be positive")
        self.num_relations = num_relations
        self.self1 = nn.Linear(in_dim, hidden_dim)
        self.rel1 = nn.ModuleList(nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(num_relations))
        self.self2 = nn.Linear(hidden_dim, hidden_dim)
        self.rel2 = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_relations))
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _rel_aggregate(x, edge_index, edge_type, relation, num_nodes):
        mask = edge_type == relation
        e = edge_index[:, mask]
        if e.numel() == 0:
            return torch.zeros_like(x)
        source, target = e
        result = torch.zeros_like(x)
        result.index_add_(0, target, x[source])
        counts = torch.zeros((num_nodes, 1), dtype=x.dtype, device=x.device)
        counts.index_add_(0, target, torch.ones((target.numel(), 1), dtype=x.dtype, device=x.device))
        return result / counts.clamp_min_(1)

    def _layer(self, x, edge_index, edge_type, self_linear, relation_linears):
        h = self_linear(x)
        for relation, linear in enumerate(relation_linears):
            h = h + linear(self._rel_aggregate(x, edge_index, edge_type, relation, x.shape[0]))
        return torch.relu(h)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        if isinstance(edge_index, (tuple, list)):
            if len(edge_index) != self.num_relations:
                raise ValueError("Expected one normalized adjacency matrix per relation")
            h = self.self1(x)
            for adjacency, linear in zip(edge_index, self.rel1):
                h = h + linear(torch.sparse.mm(adjacency, x))
            h = self.dropout(torch.relu(h))
            h2 = self.self2(h)
            for adjacency, linear in zip(edge_index, self.rel2):
                h2 = h2 + linear(torch.sparse.mm(adjacency, h))
            return self.out(self.dropout(torch.relu(h2))).squeeze(-1)


class DatasetAwareClassifier(nn.Module):
    """Dataset-specific feature encoders with a shared bot/human prediction head."""
    def __init__(self, input_dims: dict[str, int], relation_counts: dict[str, int],
                 model: str = "mlp", hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        if model not in {"mlp", "graphsage", "rgcn"}:
            raise ValueError("model must be mlp, graphsage or rgcn")
        self.model_kind = model
        self.encoders = nn.ModuleDict({name: nn.Sequential(nn.Linear(size, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
                                       for name, size in input_dims.items()})
        self.shared_self = nn.Linear(hidden_dim, hidden_dim)
        self.shared_neighbor = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rel_self = nn.ModuleDict({name: nn.Linear(hidden_dim, hidden_dim) for name in input_dims})
        self.rel_layers = nn.ModuleDict({name: nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False)
                                                              for _ in range(relation_counts.get(name, 0)))
                                         for name in input_dims})
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, dataset: str, x: torch.Tensor, adjacency=None) -> torch.Tensor:
        if dataset not in self.encoders:
            raise ValueError(f"No trained feature encoder for dataset {dataset!r}")
        h = self.encoders[dataset](x)
        if self.model_kind == "graphsage" and adjacency is not None:
            h = torch.relu(self.shared_self(h) + self.shared_neighbor(torch.sparse.mm(adjacency, h)))
        elif self.model_kind == "rgcn" and adjacency is not None:
            if len(adjacency) != len(self.rel_layers[dataset]):
                raise ValueError("Relation adjacency count does not match dataset encoder")
            out = self.rel_self[dataset](h)
            for matrix, transform in zip(adjacency, self.rel_layers[dataset]):
                out = out + transform(torch.sparse.mm(matrix, h))
            h = torch.relu(out)
        return self.head(h).squeeze(-1)
        if edge_type is None or edge_type.numel() != edge_index.shape[1]:
            raise ValueError("RelationalGNN requires one edge type per edge")
        if edge_type.numel() and (int(edge_type.min()) < 0 or int(edge_type.max()) >= self.num_relations):
            raise ValueError("edge_type contains relation outside configured range")
        h = self._layer(x, edge_index, edge_type, self.self1, self.rel1)
        h = self.dropout(h)
        h = self._layer(h, edge_index, edge_type, self.self2, self.rel2)
        return self.out(self.dropout(h)).squeeze(-1)
