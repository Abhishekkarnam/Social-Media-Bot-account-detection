try:
    from _bootstrap import PROJECT
except ModuleNotFoundError:
    from scripts._bootstrap import PROJECT

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from scripts.train import score


def resource_estimates(folder: Path, checkpoint: dict) -> dict:
    datasets = checkpoint.get("dataset_names", [checkpoint.get("dataset")])
    result = {}
    for name in datasets:
        if name in {"mgtab", "cresci2015"}:
            data = torch.load(PROJECT / "data/processed" / name / "data.pt", map_location="cpu", weights_only=True, mmap=True)
            rows, dim, edge_count = data["x"].shape[0], data["x"].shape[1], data["edge_index"].shape[1]
        elif name == "cresci2017":
            frame = pd.read_csv(PROJECT / "data/processed/cresci2017/accounts.csv", usecols=["label"])
            rows, dim, edge_count = len(frame), checkpoint.get("input_dim", checkpoint.get("input_dims", {}).get(name)), 0
        else:
            continue
        if dim is None: continue
        result[name] = {"nodes": int(rows), "feature_dim": int(dim),
                        "dense_feature_matrix_bytes_float32": int(rows * dim * 4),
                        "edge_index_bytes_int64": int(edge_count * 2 * 8), "edges": int(edge_count)}
    return result


def refresh(root: Path) -> int:
    changed = 0
    for metrics_path in root.rglob("metrics.json"):
        folder = metrics_path.parent
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        checkpoint_path = folder / "model.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            metrics["approximate_data_bytes"] = resource_estimates(folder, checkpoint)
        single_predictions = folder / "test_predictions.csv"
        if single_predictions.exists():
            frame = pd.read_csv(single_predictions)
            metrics.setdefault("test", {}).update(score(frame["true_label"].to_numpy(), frame["bot_probability"].to_numpy()))
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8"); changed += 1
            continue
        per_dataset = metrics.get("per_dataset_test", {})
        for dataset in list(per_dataset):
            pred_path = folder / f"{dataset}_test_predictions.csv"
            if not pred_path.exists(): continue
            frame = pd.read_csv(pred_path)
            per_dataset[dataset].update(score(frame["true_label"].to_numpy(), frame["bot_probability"].to_numpy()))
        if per_dataset:
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8"); changed += 1
    return changed


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--outputs", type=Path, default=PROJECT / "outputs")
    a = p.parse_args(); print(f"Refreshed per-class metrics in {refresh(a.outputs)} metrics files")
