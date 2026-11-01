try:
    from _bootstrap import PROJECT
except ModuleNotFoundError:
    from scripts._bootstrap import PROJECT

import gzip
import json
import zipfile
from collections import Counter
from pathlib import Path

import torch


def inspect_tensor(path: Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__}
    out = {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": path.stat().st_size}
    if value.ndim == 1 and "label" in path.name.lower():
        out["value_counts"] = {str(int(k)): int(v) for k, v in zip(*torch.unique(value, return_counts=True))}
    if "edge_index" in path.name and value.numel():
        out["min_index"] = int(value.min()); out["max_index"] = int(value.max())
    if value.is_floating_point():
        nans = infs = 0
        for start in range(0, value.shape[0], 65536):
            chunk = value[start:start + 65536]
            nans += int(torch.isnan(chunk).sum()); infs += int(torch.isinf(chunk).sum())
        out["nan_values"] = nans; out["infinite_values"] = infs
    return out


def inspect(root: Path) -> dict:
    datasets = {}
    for name in ("mgtab", "mgtab_large", "cresci2015", "cresci2017", "snap"):
        folder = root / "data/raw" / name
        files = []
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            rec = {"path": str(path.relative_to(root)), "bytes": path.stat().st_size}
            if path.suffix == ".pt":
                try: rec["tensor"] = inspect_tensor(path)
                except Exception as exc: rec["inspection_error"] = str(exc)
            elif path.suffix == ".zip":
                try:
                    with zipfile.ZipFile(path) as zf: rec["archive_members"] = [n for n in zf.namelist() if not n.startswith("__MACOSX")]
                except Exception as exc: rec["inspection_error"] = str(exc)
            elif path.name == "twitter_combined.txt.gz":
                n = 0; nodes = set()
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    for line in stream:
                        cols = line.split()
                        if len(cols) >= 2:
                            n += 1; nodes.add(cols[0]); nodes.add(cols[1])
                rec["edge_lines"] = n; rec["distinct_node_ids"] = len(nodes)
            files.append(rec)
        datasets[name] = {"path": str(folder.relative_to(root)), "exists": folder.exists(), "files": files}
    return {"root": str(root), "datasets": datasets}


def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, default=PROJECT)
    p.add_argument("--output", type=Path, default=PROJECT / "reports/inspection.json")
    a = p.parse_args(); report = inspect(a.root)
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
