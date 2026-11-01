try:
    from _bootstrap import PROJECT
except ModuleNotFoundError:
    from scripts._bootstrap import PROJECT

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)


def load_graph(name: str, root: Path):
    if name == "mgtab_large":
        source = root / "data/raw/mgtab_large/MGStBot-large"
        features = torch.load(source / "large_features.pt", map_location="cpu", weights_only=True, mmap=True)
        edges = torch.load(source / "large_edge_index0.pt", map_location="cpu", weights_only=True, mmap=True)
        if features.ndim != 2 or edges.shape[0] != 2 or int(edges.max()) >= features.shape[0]:
            raise ValueError("MGTAB-large features and edge indices failed shape validation")
        return features, edges, {"dataset": "MGTAB-large", "relation_id": 0, "relation": "followers",
                                 "labels_used": False, "node_order": "official MGTAB-large tensor row order"}
    if name == "snap":
        source = root / "data/processed/snap"
        nf = pd.read_csv(source / "node_features.csv", low_memory=False)
        ids = nf["node_id"].astype(str).tolist(); idmap = {v: i for i, v in enumerate(ids)}
        x = torch.tensor(np.log1p(nf[["in_degree", "out_degree", "total_degree"]].to_numpy(dtype=np.float32)), dtype=torch.float32)
        x = torch.cat((x, torch.tensor(nf[["reciprocal_out_fraction"]].to_numpy(dtype=np.float32)),), dim=1)
        ef = pd.read_csv(source / "edges.csv", low_memory=False)
        src = ef["source"].astype(str).map(idmap).to_numpy(dtype=np.int64)
        dst = ef["target"].astype(str).map(idmap).to_numpy(dtype=np.int64)
        edges = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
        return x, edges, {"dataset": "SNAP ego-Twitter", "labels_used": False,
                          "node_order": "processed node_features.csv order", "feature_transform": "log1p on degree features"}
    raise ValueError("dataset must be mgtab_large or snap")


def main():
    p = argparse.ArgumentParser(description="Memory-bounded graph link reconstruction pretraining; no labels are loaded.")
    p.add_argument("--dataset", choices=["mgtab_large", "snap"], required=True)
    p.add_argument("--steps", type=int, default=1000); p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=64); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    started = time.time(); features, edges, metadata = load_graph(args.dataset, PROJECT)
    if features.ndim != 2 or edges.ndim != 2 or edges.shape[0] != 2 or edges.numel() == 0:
        raise ValueError("Expected a feature matrix and non-empty 2xE edge index")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= features.shape[0]):
        raise ValueError("Edge indices do not align with feature rows")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model = FeatureEncoder(features.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = []
    for step in range(1, args.steps + 1):
        sample = torch.randint(edges.shape[1], (args.batch_size,), dtype=torch.long)
        positive = edges[:, sample].long()
        negative_target = torch.randint(features.shape[0], (args.batch_size,), dtype=torch.long)
        node_ids = torch.cat((positive[0], positive[1], negative_target))
        batch_features = features[node_ids].to(device, dtype=torch.float32)
        z = model(batch_features)
        src = z[:args.batch_size]; dst_pos = z[args.batch_size:2 * args.batch_size]
        dst_neg = z[2 * args.batch_size:]
        pos_score = (src * dst_pos).sum(dim=1)
        neg_score = (src * dst_neg).sum(dim=1)
        logits = torch.cat((pos_score, neg_score))
        labels = torch.cat((torch.ones_like(pos_score), torch.zeros_like(neg_score)))
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"step={step}/{args.steps} link_reconstruction_loss={loss.item():.5f}")
    output = args.output_dir or PROJECT / "outputs" / f"pretrain_{args.dataset}"
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": int(features.shape[1]),
                "hidden_dim": args.hidden_dim, "dataset": args.dataset,
                "objective": "positive observed edge vs uniformly sampled negative pair",
                "uses_labels": False, **metadata}, output / "pretrained_encoder.pt")
    report = {**metadata, "nodes": int(features.shape[0]), "positive_edges": int(edges.shape[1]),
              "feature_dim": int(features.shape[1]), "steps": args.steps, "batch_size": args.batch_size,
              "objective": "sampled graph link reconstruction", "labels_used": False,
              "loss_first": losses[0], "loss_last": losses[-1], "seconds": time.time() - started,
              "device": str(device), "full_graph_on_device": False}
    (output / "pretraining_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
