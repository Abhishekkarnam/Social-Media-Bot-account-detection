from __future__ import annotations

import json
import io
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


MGTAB_RELATIONS = ["followers", "friends", "mention", "reply", "quoted", "url", "hashtag"]


def _load_tensor(path: Path, mmap: bool = False) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}")
    return value


def _report(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "processing_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_mgtab(source: Path, output: Path) -> dict:
    x = _load_tensor(source / "features.pt")
    y = _load_tensor(source / "labels_bot.pt").long().reshape(-1)
    stance = _load_tensor(source / "labels_stance.pt").long().reshape(-1)
    edge = _load_tensor(source / "edge_index.pt").long()
    rel = _load_tensor(source / "edge_type.pt").long().reshape(-1)
    weight = _load_tensor(source / "edge_weight.pt").reshape(-1)
    if x.ndim != 2 or len(y) != x.shape[0] or len(stance) != x.shape[0]:
        raise ValueError("MGTAB features and label rows do not align")
    if edge.ndim != 2 or edge.shape[0] != 2 or edge.shape[1] != len(rel) or len(rel) != len(weight):
        raise ValueError("MGTAB edge_index, edge_type and edge_weight are inconsistent")
    if edge.numel() and (edge.min() < 0 or edge.max() >= x.shape[0]):
        raise ValueError("MGTAB edge index is outside feature rows")
    if rel.numel() and (rel.min() < 0 or rel.max() >= len(MGTAB_RELATIONS)):
        raise ValueError("Unknown MGTAB relation type")
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x.float(), "y": y, "stance": stance, "edge_index": edge,
                "edge_type": rel, "edge_weight": weight.float(),
                "node_ids": [f"mgtab_row_{i}" for i in range(x.shape[0])],
                "feature_names": [f"feature_{i:03d}" for i in range(x.shape[1])],
                "relations": MGTAB_RELATIONS}, output / "data.pt")
    report = {"dataset": "MGTAB", "source": str(source), "rows": int(x.shape[0]),
              "feature_dim": int(x.shape[1]), "feature_names_semantic": False,
              "label_mapping": {"0": "human", "1": "bot"},
              "label_counts": {str(k): v for k, v in Counter(y.tolist()).items()},
              "edge_count": int(edge.shape[1]), "edge_relation_mapping": dict(enumerate(MGTAB_RELATIONS)),
              "invalid_edge_count": 0, "preserved_edge_weight": True,
              "nan_features": int(torch.isnan(x).sum()), "infinite_features": int(torch.isinf(x).sum())}
    _report(output, report)
    return report


def prepare_mgtab_large(source: Path, output: Path) -> dict:
    """Write an index/label manifest; avoid duplicating the multi-GB source tensors."""
    features_path = source / "large_features.pt"
    labels = _load_tensor(source / "labels_bot.pt").long().reshape(-1)
    stance = _load_tensor(source / "labels_stance.pt").long().reshape(-1)
    # Tensor metadata is verified from the published/reference loader and file metadata.
    # Load features only to check alignment, then release immediately.
    features = _load_tensor(features_path, mmap=True)
    n = int(features.shape[0])
    if features.ndim != 2 or features.shape[1] != 788 or labels.numel() > n or stance.numel() > n:
        raise ValueError("MGTAB-large tensor dimensions are incompatible")
    labeled_count = int(labels.numel())
    if labeled_count != 10199:
        raise ValueError("The published MGTAB-large mapping expects 10,199 leading labeled rows")
    output.mkdir(parents=True, exist_ok=True)
    del features
    edge_inventory = []
    for relation in range(7):
        edge_path = source / f"large_edge_index{relation}.pt"
        weight_path = source / f"large_edge_weight{relation}.pt"
        edges = _load_tensor(edge_path, mmap=True).long()
        weights = _load_tensor(weight_path, mmap=True)
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] != weights.numel():
            raise ValueError(f"MGTAB-large edge/weight mismatch for relation {relation}")
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= n):
            raise ValueError(f"MGTAB-large relation {relation} has out-of-range node indices")
        edge_inventory.append({"relation": MGTAB_RELATIONS[relation], "edges": int(edges.shape[1]),
                              "weight_dtype": str(weights.dtype)})
        del edges, weights
    torch.save({"source_features": str(features_path.resolve()),
                "source_edge_files": [str((source / f"large_edge_index{i}.pt").resolve()) for i in range(7)],
                "labels_bot": labels, "labels_stance": stance,
                "labeled_node_range": [0, labeled_count], "unlabeled_node_range": [labeled_count, n],
                "feature_dim": 788, "relations": MGTAB_RELATIONS,
                "mapping_provenance": "GraphDetec/MGTAB reference Dataset.py uses label count as prefix of feature row count"},
               output / "manifest.pt")
    report = {"dataset": "MGTAB-large", "nodes": n, "labeled_nodes": labeled_count,
              "unlabeled_nodes": n - labeled_count, "label_mapping": {"0": "human", "1": "bot"},
              "node_mapping": "only rows [0,10199) are labeled, as in the official reference loader",
              "graph_processing": "source edge tensors referenced; not concatenated or copied",
              "edge_files": [p.name for p in sorted(source.glob("large_edge_index*.pt"))],
              "edge_inventory": edge_inventory}
    _report(output, report)
    return report


def prepare_cresci15(source: Path, output: Path) -> dict:
    names = ["cat_properties_tensor.pt", "num_properties_tensor.pt", "des_tensor.pt", "tweets_tensor.pt"]
    parts = [_load_tensor(source / name).float() for name in names]
    y = _load_tensor(source / "label.pt").long().reshape(-1)
    x = torch.cat(parts, dim=1)
    edge = _load_tensor(source / "edge_index.pt").long()
    edge_type = _load_tensor(source / "edge_type.pt").long().reshape(-1)
    splits = {name: _load_tensor(source / f"{name}_idx.pt").long().reshape(-1)
              for name in ("train", "val", "test")}
    n = int(x.shape[0])
    if len(y) != n or edge.shape[0] != 2 or edge.shape[1] != edge_type.numel():
        raise ValueError("Cresci-15 tensors do not align")
    if edge.numel() and (edge.min() < 0 or edge.max() >= n):
        raise ValueError("Cresci-15 edge index outside feature rows")
    for key, indices in splits.items():
        if indices.numel() and (indices.min() < 0 or indices.max() >= n):
            raise ValueError(f"Cresci-15 {key} split has invalid row index")
    if set(splits["train"].tolist()) & set(splits["val"].tolist()) or set(splits["train"].tolist()) & set(splits["test"].tolist()) or set(splits["val"].tolist()) & set(splits["test"].tolist()):
        raise ValueError("Cresci-15 official splits overlap")
    offset = 0
    schema = {}
    for name, part in zip(names, parts):
        schema[name.removesuffix("_tensor.pt")] = [offset, offset + part.shape[1]]
        offset += part.shape[1]
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y, "edge_index": edge, "edge_type": edge_type,
                "splits": splits, "node_ids": [f"cresci15_row_{i}" for i in range(n)],
                "feature_group_slices": schema, "feature_names_semantic": False}, output / "data.pt")
    report = {"dataset": "Cresci-15", "nodes": n, "feature_dim": int(x.shape[1]),
              "feature_groups": {k: v for k, v in schema.items()},
              "feature_semantics": "retained tensor groups; individual feature semantics unavailable in local metadata",
              "label_counts": {str(k): v for k, v in Counter(y.tolist()).items()},
              "label_mapping": {"0": "human", "1": "bot"}, "edge_count": int(edge.shape[1]),
              "edge_type_values": sorted(set(edge_type.tolist())),
              "splits": {k: {"count": int(v.numel()), "label_counts":
                             {str(c): int((y[v] == c).sum()) for c in torch.unique(y[v]).tolist()}}
                         for k, v in splits.items()},
              "split_provenance": "original train_idx.pt, val_idx.pt, test_idx.pt; names preserved verbatim"}
    _report(output, report)
    return report


def _numeric_profile_features(frame: pd.DataFrame, id_col: str) -> pd.DataFrame:
    excluded = {id_col.lower(), "screen_name", "name", "description", "location", "url", "lang", "created_at", "time_zone", "profile_image_url", "test_set_1", "test_set_2"}
    out = pd.DataFrame(index=frame.index)
    for col in frame.columns:
        if col.lower() in excluded or "color" in col.lower():
            continue
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if numeric.notna().any():
            out[col.strip().replace(" ", "_")] = numeric
    return out


def prepare_cresci2017(source: Path, output: Path) -> dict:
    groups = {"genuine_accounts": 0, "social_spambots_1": 1, "social_spambots_2": 1,
              "social_spambots_3": 1, "traditional_spambots_1": 1, "traditional_spambots_2": 1,
              "traditional_spambots_3": 1, "traditional_spambots_4": 1, "fake_followers": 1}
    records, failures, group_counts, tweet_members = [], [], {}, {}
    output.mkdir(parents=True, exist_ok=True)
    for group, label in groups.items():
        archives = sorted(source.rglob(f"{group}.csv.zip"))
        if not archives:
            failures.append({"group": group, "reason": "archive not found"})
            continue
        archive = archives[0]
        try:
            with zipfile.ZipFile(archive) as zf:
                user_files = [n for n in zf.namelist() if Path(n).name.lower() == "users.csv"]
                if not user_files:
                    failures.append({"group": group, "archive": archive.name, "reason": "users.csv missing"})
                    continue
                with zf.open(user_files[0]) as stream:
                    raw_users = stream.read()
                profile_path = output / "extracted_accounts" / group / "users.csv"
                profile_path.parent.mkdir(parents=True, exist_ok=True)
                profile_path.write_bytes(raw_users)
                frame = pd.read_csv(io.BytesIO(raw_users), low_memory=False)
                tweet_members[group] = [n for n in zf.namelist() if Path(n).name.lower() == "tweets.csv"]
            lower = {str(c).lower(): c for c in frame.columns}
            id_col = next((lower[c] for c in ("id", "userid", "user_id", "id_str") if c in lower), None)
            if id_col is None:
                failures.append({"group": group, "archive": archive.name, "reason": "account ID field missing"})
                continue
            ids = frame[id_col].astype("string")
            features = _numeric_profile_features(frame, id_col)
            group_counts[group] = int(len(frame))
            for idx, account_id in enumerate(ids):
                rec = {"account_id": str(account_id), "label": label, "source_group": group,
                       "source_file": archive.name}
                for col in features.columns:
                    rec[col] = features.iloc[idx][col]
                records.append(rec)
        except Exception as exc:
            failures.append({"group": group, "archive": archive.name, "reason": str(exc)})
    if not records:
        raise ValueError("No usable Cresci-2017 account profiles were found")
    frame = pd.DataFrame.from_records(records)
    duplicate_mask = frame.duplicated("account_id", keep="first")
    duplicate_count = int(duplicate_mask.sum())
    frame = frame.loc[~duplicate_mask].reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "accounts.csv", index=False)
    report = {"dataset": "Cresci-2017", "accounts": int(len(frame)), "source_groups": group_counts,
              "label_mapping": {"genuine_accounts": 0, "social_spambots_*": 1,
                                "traditional_spambots_*": 1, "fake_followers": 1},
              "label_provenance": "group names from original archive; READ.ME identifies genuine accounts and spambots",
              "account_features": [c for c in frame.columns if c not in {"account_id", "label", "source_group", "source_file"}],
              "tweet_files": "not used as account rows; profile users.csv used",
              "extracted_account_files": {group: f"extracted_accounts/{group}/users.csv" for group in group_counts},
              "tweet_file_members_not_extracted": tweet_members,
              "graph_edges": "not provided as user-user graph in these group archives",
              "duplicate_account_ids_removed": duplicate_count,
              "rows_missing_any_feature": int(frame.drop(columns=["account_id", "label", "source_group", "source_file"], errors="ignore").isna().all(axis=1).sum()),
              "conversion_failures": failures,
              "excluded_crowdflower_annotation_archive": True}
    _report(output, report)
    return report


def prepare_snap(source: Path, output: Path) -> dict:
    import gzip
    graph = next(iter(sorted(source.glob("twitter_combined.txt.gz"))), None)
    if graph is None:
        raise FileNotFoundError(f"twitter_combined.txt.gz not found in {source}")
    nodes, edges = set(), set()
    raw_edge_lines = 0
    with gzip.open(graph, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            cols = line.split()
            if len(cols) < 2:
                continue
            raw_edge_lines += 1
            u, v = cols[0], cols[1]
            nodes.add(u); nodes.add(v); edges.add((u, v))
    # Profile-free structural features. Clustering is omitted: exact per-node clustering
    # at this scale is expensive and not required for a valid unsupervised representation.
    indeg, outdeg = Counter(), Counter()
    reciprocal = 0
    for u, v in edges:
        outdeg[u] += 1; indeg[v] += 1
        reciprocal += int((v, u) in edges)
    ids = sorted(nodes, key=lambda val: (0, int(val)) if val.isdigit() else (1, val))
    rows = []
    for node in ids:
        rows.append({"node_id": node, "in_degree": indeg[node], "out_degree": outdeg[node],
                     "total_degree": indeg[node] + outdeg[node],
                     "reciprocal_out_fraction": 0.0})
    # Calculate reciprocity per node with outgoing neighbors without materializing a second graph.
    out_neighbors = {}
    for u, v in edges:
        out_neighbors.setdefault(u, []).append(v)
    for row in rows:
        node = row["node_id"]
        neighbors = out_neighbors.get(node, ())
        row["reciprocal_out_fraction"] = (sum((v, node) in edges for v in neighbors) / len(neighbors)) if neighbors else 0.0
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "node_features.csv", index=False)
    pd.DataFrame(sorted(edges), columns=["source", "target"]).to_csv(output / "edges.csv", index=False)
    report = {"dataset": "Stanford SNAP ego-Twitter combined", "source_file": str(graph),
              "nodes": len(nodes), "raw_edge_lines": raw_edge_lines,
              "directed_unique_edges": len(edges), "duplicate_edge_lines": raw_edge_lines - len(edges), "labels": None,
              "node_indexing": "original SNAP node IDs preserved in node_id column; sorted internal CSV order",
              "features": ["in_degree", "out_degree", "total_degree", "reciprocal_out_fraction"],
              "self_loops": sum(u == v for u, v in edges), "reciprocal_directed_edge_fraction": reciprocal / len(edges) if edges else 0.0,
              "clustering": "not calculated; use a sampling or scalable graph library if needed"}
    _report(output, report)
    return report
