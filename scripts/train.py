try:
    from _bootstrap import PROJECT
except ModuleNotFoundError:
    from scripts._bootstrap import PROJECT

import argparse
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models import GraphSAGE, MLP, RelationalGNN, DatasetAwareClassifier, normalized_adjacency


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_data(name, root):
    folder = root / name
    if name in {"mgtab", "cresci2015"}:
        obj = torch.load(folder / "data.pt", map_location="cpu", weights_only=True)
        x = obj["x"].float().numpy(); y = obj["y"].numpy().astype(int)
        edge = obj["edge_index"].long(); edge_type = obj.get("edge_type", torch.zeros(edge.shape[1], dtype=torch.long)).long()
        edge_weight = obj.get("edge_weight", torch.ones(edge.shape[1], dtype=torch.float32)).float()
        names = [f"feature_{i:04d}" for i in range(x.shape[1])]
        splits = obj.get("splits")
    elif name == "cresci2017":
        frame = pd.read_csv(folder / "accounts.csv", low_memory=False)
        excluded = {"account_id", "label", "source_group", "source_file"}
        names = []
        values = []
        for col in frame.columns:
            if col in excluded: continue
            numeric = pd.to_numeric(frame[col], errors="coerce")
            if numeric.notna().any():
                names.append(col); values.append(numeric.to_numpy(dtype=np.float32))
        if not values: raise ValueError("Cresci-2017 has no numeric account features")
        x = np.column_stack(values); y = frame["label"].to_numpy(dtype=int)
        edge = torch.empty((2, 0), dtype=torch.long); edge_type = torch.empty(0, dtype=torch.long)
        edge_weight = torch.empty(0, dtype=torch.float32)
        splits = None
    else:
        raise ValueError("Supported supervised datasets: mgtab, cresci2015, cresci2017")
    if x.ndim != 2 or len(x) != len(y) or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Expected aligned features and binary labels encoded human=0, bot=1")
    if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= len(y)):
        raise ValueError("Graph contains node indices outside feature rows")
    if splits:
        train_idx, val_idx, test_idx = (splits[k].numpy() for k in ("train", "val", "test"))
    else:
        all_idx = np.arange(len(y))
        trainval, test_idx = train_test_split(all_idx, test_size=0.15, random_state=42, stratify=y)
        train_idx, val_idx = train_test_split(trainval, test_size=0.176470588, random_state=42, stratify=y[trainval])
    return x, y, edge, edge_type, edge_weight, names, (train_idx, val_idx, test_idx)


def score(y, probabilities):
    pred = probabilities >= 0.5
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        y, pred.astype(int), labels=[0, 1], zero_division=0)
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
            "pr_auc_average_precision": float(average_precision_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
            "confusion_matrix_human0_bot1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "per_class": {"human_0": {"precision": float(class_precision[0]), "recall": float(class_recall[0]),
                                       "f1": float(class_f1[0]), "support": int(class_support[0])},
                          "bot_1": {"precision": float(class_precision[1]), "recall": float(class_recall[1]),
                                    "f1": float(class_f1[1]), "support": int(class_support[1])}}}


def train_multiple(args, dataset_names):
    """Joint training with per-dataset encoders, disjoint graphs and shared binary head."""
    start = time.time(); loaded = {}; preprocessing = {}; relation_counts = {}
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    for name in dataset_names:
        x, y, edge, edge_type, edge_weight, feature_names, split = load_data(name, args.data_root)
        train_i, val_i, test_i = split
        x = np.where(np.isfinite(x), x, np.nan).astype(np.float32)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[train_i])
        x = imputer.transform(x).astype(np.float32)
        scaler = StandardScaler().fit(x[train_i]); x = scaler.transform(x).astype(np.float32)
        edge = edge.to(device); edge_type = edge_type.to(device); edge_weight = edge_weight.to(device)
        rel_count = int(edge_type.max()) + 1 if edge_type.numel() else 0
        if args.model == "graphsage": adjacency = normalized_adjacency(edge, len(y), edge_weight=edge_weight)
        elif args.model == "rgcn":
            adjacency = tuple(normalized_adjacency(edge, len(y), edge_type == rel, edge_weight) for rel in range(rel_count))
            if not adjacency: raise ValueError(f"{name} has no typed graph, so R-GNN cannot use it")
        else: adjacency = None
        loaded[name] = {"x": torch.from_numpy(x).to(device), "y": torch.from_numpy(y.astype(np.float32)).to(device),
                        "edge": adjacency, "edge_type": edge_type, "indices": split, "labels": y,
                        "feature_names": feature_names, "graph_has_edges": bool(edge.numel()),
                        "edge_count": int(edge.shape[1]), "feature_bytes": int(x.nbytes)}
        preprocessing[name] = {"imputer": imputer, "scaler": scaler, "feature_names": feature_names}
        relation_counts[name] = rel_count
    model = DatasetAwareClassifier({n: d["x"].shape[1] for n, d in loaded.items()}, relation_counts,
                                   args.model, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for name, data in loaded.items():
            train_i = torch.as_tensor(data["indices"][0], dtype=torch.long, device=device)
            counts = np.bincount(data["labels"][data["indices"][0]], minlength=2)
            pos_weight = torch.tensor(counts[0] / max(counts[1], 1), dtype=torch.float32, device=device)
            logits = model(name, data["x"], data["edge"])
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(
                logits[train_i], data["y"][train_i], pos_weight=pos_weight))
        train_loss = torch.stack(losses).mean(); train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        model.eval(); val_losses = []
        with torch.no_grad():
            for name, data in loaded.items():
                val_i = torch.as_tensor(data["indices"][1], dtype=torch.long, device=device)
                logits = model(name, data["x"], data["edge"])
                val_losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logits[val_i], data["y"][val_i]))
        val_loss = float(torch.stack(val_losses).mean().item())
        history.append({"epoch": epoch, "train_loss_mean_dataset": float(train_loss.item()), "val_loss_mean_dataset": val_loss})
        print(f"epoch={epoch} train_loss={train_loss.item():.5f} val_loss={val_loss:.5f}")
        if val_loss < best - 1e-6:
            best, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience: break
    if best_state is None: raise RuntimeError("Multi-dataset training did not produce a checkpoint")
    model.load_state_dict(best_state); model.eval(); metrics = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for name, data in loaded.items():
            test_i = data["indices"][2]
            ti = torch.as_tensor(test_i, dtype=torch.long, device=device)
            probabilities = torch.sigmoid(model(name, data["x"], data["edge"])[ti]).cpu().numpy()
            metrics[name] = score(data["labels"][test_i], probabilities)
            pd.DataFrame({"row_index": test_i, "true_label": data["labels"][test_i],
                          "bot_probability": probabilities, "predicted_label": (probabilities >= .5).astype(int)}).to_csv(
                              args.output_dir / f"{name}_test_predictions.csv", index=False)
    torch.save({"state_dict": best_state, "model": args.model, "dataset_names": dataset_names,
                "input_dims": {n: d["x"].shape[1] for n, d in loaded.items()},
                "relation_counts": relation_counts, "graph_has_edges": {n: d["graph_has_edges"] for n, d in loaded.items()},
                "hidden_dim": args.hidden_dim, "dropout": args.dropout,
                "label_mapping": {"0": "human", "1": "bot"}, "seed": args.seed}, args.output_dir / "model.pt")
    joblib.dump(preprocessing, args.output_dir / "preprocessing.joblib")
    (args.output_dir / "metrics.json").write_text(json.dumps({"per_dataset_test": metrics, "history": history,
        "split_sizes": {n: {"train": len(d["indices"][0]), "validation": len(d["indices"][1]), "test": len(d["indices"][2])}
                        for n, d in loaded.items()}, "training_seconds": time.time() - start,
        "device": str(device), "graph_components": "kept disjoint; no cross-dataset edges created",
        "approximate_data_bytes": {n: {"dense_features": d["feature_bytes"], "edge_index": d["edge_count"] * 16,
                                         "edge_count": d["edge_count"]} for n, d in loaded.items()},
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None}, indent=2), encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2)); print(f"Saved artifacts to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train one source-specific model with leakage-safe preprocessing.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=["mgtab", "cresci2015", "cresci2017"])
    group.add_argument("--datasets", help="Comma-separated joint training dataset list")
    parser.add_argument("--model", choices=["mlp", "graphsage", "rgcn"], default="mlp")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=80); parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=64); parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default="auto")
    args = parser.parse_args(); seed_all(args.seed)
    if args.datasets:
        names = list(dict.fromkeys(n.strip() for n in args.datasets.split(",") if n.strip()))
        if len(names) < 2: raise SystemExit("--datasets requires at least two datasets")
        if args.model == "rgcn" and "cresci2017" in names:
            raise SystemExit("Cresci-2017 has no graph; relational joint training requires graph-bearing datasets only")
        if args.output_dir is None:
            args.output_dir = PROJECT / "outputs" / f"multidataset_{'_'.join(names)}_{args.model}"
        train_multiple(args, names); return
    start = time.time()
    x, y, edge, edge_type, edge_weight, feature_names, (train_i, val_i, test_i) = load_data(args.dataset, args.data_root)
    # Fit imputation and scaling on training rows only.
    x = np.where(np.isfinite(x), x, np.nan).astype(np.float32)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[train_i])
    x = imputer.transform(x).astype(np.float32)
    scaler = StandardScaler().fit(x[train_i]); x = scaler.transform(x).astype(np.float32)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    x_t = torch.from_numpy(x).to(device); y_t = torch.from_numpy(y.astype(np.float32)).to(device)
    edge = edge.to(device); edge_type = edge_type.to(device); edge_weight = edge_weight.to(device)
    if args.model == "mlp": model = MLP(x.shape[1], args.hidden_dim, args.dropout)
    elif args.model == "graphsage": model = GraphSAGE(x.shape[1], args.hidden_dim, args.dropout)
    else:
        if not edge_type.numel(): raise ValueError("R-GNN requires typed graph edges")
        model = RelationalGNN(x.shape[1], int(edge_type.max()) + 1, args.hidden_dim, args.dropout)
    if args.model == "graphsage":
        model_edges = normalized_adjacency(edge, len(y), edge_weight=edge_weight)
    elif args.model == "rgcn":
        model_edges = tuple(normalized_adjacency(edge, len(y), edge_type == relation, edge_weight)
                            for relation in range(int(edge_type.max()) + 1))
    else:
        model_edges = edge
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    counts = np.bincount(y[train_i], minlength=2)
    pos_weight = torch.tensor(counts[0] / max(counts[1], 1), dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ti, vi, qi = (torch.as_tensor(i, dtype=torch.long, device=device) for i in (train_i, val_i, test_i))
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model(x_t, model_edges, edge_type)
        loss = criterion(logits[ti], y_t[ti]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x_t, model_edges, edge_type)[vi]
            val_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(val_logits, y_t[vi]).item())
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={loss.item():.5f} val_loss={val_loss:.5f}")
        if val_loss < best - 1e-6:
            best, stale = val_loss, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience: break
    if state is None: raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(state); model.eval()
    with torch.no_grad(): probs = torch.sigmoid(model(x_t, model_edges, edge_type)[qi]).cpu().numpy()
    metrics = score(y[test_i], probs)
    output = args.output_dir or (PROJECT / "outputs" / f"{args.dataset}_{args.model}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {"state_dict": state, "model": args.model, "dataset": args.dataset,
                  "input_dim": int(x.shape[1]), "hidden_dim": args.hidden_dim, "dropout": args.dropout,
                  "num_relations": int(edge_type.max()) + 1 if edge_type.numel() else 0,
                  "graph_has_edges": bool(edge.numel()),
                  "feature_names": feature_names, "label_mapping": {"0": "human", "1": "bot"},
                  "seed": args.seed}
    torch.save(checkpoint, output / "model.pt")
    joblib.dump({"imputer": imputer, "scaler": scaler, "feature_names": feature_names}, output / "preprocessing.joblib")
    (output / "metrics.json").write_text(json.dumps({"test": metrics, "history": history,
        "split_sizes": {"train": len(train_i), "validation": len(val_i), "test": len(test_i)},
        "training_seconds": time.time() - start, "device": str(device),
        "approximate_data_bytes": {"dense_features": int(x.nbytes), "edge_index": int(edge.numel() * 8),
                                    "edge_type": int(edge_type.numel() * 8), "edge_weight": int(edge_weight.numel() * 4)},
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None}, indent=2), encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    pd.DataFrame({"row_index": test_i, "true_label": y[test_i], "bot_probability": probs,
                  "predicted_label": (probs >= 0.5).astype(int)}).to_csv(output / "test_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2)); print(f"Saved artifacts to {output}")


if __name__ == "__main__": main()
