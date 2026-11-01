"""Schema-validating inference for single-dataset and dataset-aware checkpoints."""
from _bootstrap import PROJECT

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.models import GraphSAGE, MLP, RelationalGNN, DatasetAwareClassifier, normalized_adjacency


def main():
    p = argparse.ArgumentParser(description="Predict bot probabilities with a saved checkpoint.")
    p.add_argument("--model-dir", type=Path, required=True); p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--id-column", default="account_id")
    p.add_argument("--dataset", help="Required for a multi-dataset checkpoint")
    p.add_argument("--edges", type=Path, help="Optional graph edges CSV with source,target[,relation_id]")
    p.add_argument("--device", default="cpu"); args = p.parse_args()
    checkpoint = torch.load(args.model_dir / "model.pt", map_location="cpu", weights_only=True)
    all_preprocessing = joblib.load(args.model_dir / "preprocessing.joblib")
    multi = "dataset_names" in checkpoint
    if multi:
        if args.dataset not in checkpoint["dataset_names"]:
            raise ValueError(f"Choose a trained dataset encoder from {checkpoint['dataset_names']}")
        dataset = args.dataset; preprocessing = all_preprocessing[dataset]
        feature_names = preprocessing["feature_names"]; input_dim = checkpoint["input_dims"][dataset]
        graph_required = checkpoint["graph_has_edges"][dataset]
    else:
        dataset = checkpoint["dataset"]; preprocessing = all_preprocessing
        feature_names = checkpoint["feature_names"]; input_dim = checkpoint["input_dim"]
        graph_required = checkpoint["graph_has_edges"]
    input_edges = None; input_types = None; input_weights = None; identifiers = None; frame = None
    if args.input.suffix.lower() == ".pt":
        obj = torch.load(args.input, map_location="cpu", weights_only=True)
        matrix = obj["x"].detach().cpu().numpy().astype(np.float32)
        identifiers = obj.get("node_ids", [str(i) for i in range(len(matrix))])
        input_edges = obj.get("edge_index")
        input_types = obj.get("edge_type")
        input_weights = obj.get("edge_weight")
    else:
        frame = pd.read_csv(args.input, low_memory=False)
        missing = [c for c in feature_names if c not in frame.columns]
        if missing: raise ValueError(f"Missing required model features ({len(missing)}): {missing[:20]}")
        matrix = frame[feature_names].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
        if args.id_column in frame: identifiers = frame[args.id_column].astype(str).tolist()
    matrix = preprocessing["imputer"].transform(matrix)
    matrix = preprocessing["scaler"].transform(matrix).astype(np.float32)
    if matrix.shape[1] != input_dim: raise ValueError(f"Expected {input_dim} features, received {matrix.shape[1]}")
    device = torch.device(args.device); kind = checkpoint["model"]
    edge_index = (input_edges.long() if input_edges is not None else torch.empty((2, 0), dtype=torch.long))
    edge_type = (input_types.long() if input_types is not None else torch.zeros(edge_index.shape[1], dtype=torch.long))
    edge_weight = (input_weights.float() if input_weights is not None else torch.ones(edge_index.shape[1], dtype=torch.float32))
    if args.edges:
        if frame is None: raise ValueError("--edges CSV is only supported with CSV feature input; tensor input already carries graph edges")
        edf = pd.read_csv(args.edges, low_memory=False)
        if not {"source", "target"}.issubset(edf.columns): raise ValueError("Edges CSV requires source and target columns")
        if args.id_column in frame:
            id_map = {str(v): i for i, v in enumerate(frame[args.id_column])}
            src = edf["source"].astype(str).map(id_map); dst = edf["target"].astype(str).map(id_map)
        else:
            src = pd.to_numeric(edf["source"], errors="coerce"); dst = pd.to_numeric(edf["target"], errors="coerce")
        valid = src.notna() & dst.notna()
        pairs = np.vstack((src[valid].to_numpy(dtype=np.int64), dst[valid].to_numpy(dtype=np.int64)))
        if pairs.size and (pairs.min() < 0 or pairs.max() >= len(frame)): raise ValueError("Edge endpoint outside input rows")
        edge_index = torch.from_numpy(pairs)
        relation_col = next((c for c in ("relation_id", "edge_type") if c in edf.columns), None)
        edge_type = torch.from_numpy(pd.to_numeric(edf.loc[valid, relation_col], errors="raise").to_numpy(dtype=np.int64)) if relation_col else torch.zeros(edge_index.shape[1], dtype=torch.long)
        weight_col = next((c for c in ("edge_weight", "weight") if c in edf.columns), None)
        edge_weight = torch.from_numpy(pd.to_numeric(edf.loc[valid, weight_col], errors="raise").to_numpy(dtype=np.float32)) if weight_col else torch.ones(edge_index.shape[1], dtype=torch.float32)
    elif graph_required and kind in {"graphsage", "rgcn"} and input_edges is None:
        raise ValueError("This checkpoint used graph edges; pass --edges to preserve its graph input")
    if edge_index.numel() and kind == "rgcn":
        rel_count = checkpoint["relation_counts"][dataset] if multi else checkpoint["num_relations"]
        if edge_type.numel() != edge_index.shape[1] or int(edge_type.max()) >= rel_count:
            raise ValueError("R-GNN inference requires valid relation_id/edge_type values for every edge")
        adjacency = tuple(normalized_adjacency(edge_index, len(matrix), edge_type == r, edge_weight).to(device) for r in range(rel_count))
    elif kind == "graphsage":
        adjacency = normalized_adjacency(edge_index, len(matrix), edge_weight=edge_weight).to(device)
    else: adjacency = None
    if multi:
        model = DatasetAwareClassifier(checkpoint["input_dims"], checkpoint["relation_counts"], kind,
                                       checkpoint["hidden_dim"], checkpoint["dropout"])
    elif kind == "mlp": model = MLP(input_dim, checkpoint["hidden_dim"], checkpoint["dropout"])
    elif kind == "graphsage": model = GraphSAGE(input_dim, checkpoint["hidden_dim"], checkpoint["dropout"])
    else: model = RelationalGNN(input_dim, checkpoint["num_relations"], checkpoint["hidden_dim"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"]); model.to(device).eval()
    x = torch.from_numpy(matrix).to(device)
    with torch.no_grad():
        logits = model(dataset, x, adjacency) if multi else model(x, adjacency if adjacency is not None else edge_index.to(device), edge_type.to(device))
        probabilities = torch.sigmoid(logits).cpu().numpy()
    result = pd.DataFrame({"bot_probability": probabilities, "predicted_class_at_0.5": (probabilities >= 0.5).astype(int),
                           "interpretation": "model prediction; not a verified bot label"})
    if identifiers is not None: result.insert(0, args.id_column, identifiers)
    args.output.parent.mkdir(parents=True, exist_ok=True); result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} model predictions to {args.output}")


if __name__ == "__main__": main()
