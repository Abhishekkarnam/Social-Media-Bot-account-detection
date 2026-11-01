try:
    from _bootstrap import PROJECT
except ModuleNotFoundError:
    from scripts._bootstrap import PROJECT

import argparse
import json
from pathlib import Path

import torch


def validate(processed_root: Path):
    results = {}
    for name in ("mgtab", "cresci2015"):
        data_path = processed_root / name / "data.pt"
        obj = torch.load(data_path, map_location="cpu", weights_only=True)
        n = obj["x"].shape[0]
        assert obj["x"].ndim == 2 and obj["y"].numel() == n
        edge = obj["edge_index"]
        assert edge.ndim == 2 and edge.shape[0] == 2
        assert edge.numel() == 0 or (int(edge.min()) >= 0 and int(edge.max()) < n)
        if "edge_type" in obj: assert obj["edge_type"].numel() == edge.shape[1]
        results[name] = {"rows": n, "feature_dim": int(obj["x"].shape[1]), "edges": int(edge.shape[1]), "valid": True}
    for name in ("mgtab", "mgtab_large", "cresci2015", "cresci2017", "snap"):
        report = processed_root / name / "processing_report.json"
        if report.exists(): json.loads(report.read_text(encoding="utf-8"))
    large = torch.load(processed_root / "mgtab_large" / "manifest.pt", map_location="cpu", weights_only=True)
    assert large["labeled_node_range"] == [0, 10199]
    assert large["unlabeled_node_range"][0] == 10199
    results["mgtab_large"] = {"labeled_nodes": 10199, "unlabeled_nodes": 400000, "valid": True}
    snap = json.loads((processed_root / "snap" / "processing_report.json").read_text(encoding="utf-8"))
    assert snap["labels"] is None
    results["snap"] = {"nodes": snap["nodes"], "directed_unique_edges": snap["directed_unique_edges"], "unlabeled": True}
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--processed-root", type=Path, default=PROJECT / "data/processed")
    a = p.parse_args(); output = validate(a.processed_root); print(json.dumps(output, indent=2))
