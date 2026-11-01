"""Server-rendered HTML frontend and inference routes for saved model artifacts."""

from __future__ import annotations

import json
import csv
from functools import lru_cache
import io
import threading
from collections import deque
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from scripts._bootstrap import PROJECT
from src.models import DatasetAwareClassifier, GraphSAGE, MLP, RelationalGNN, normalized_adjacency


app = FastAPI(title="Twitter Bot Detection System", version="1.0.0",
              docs_url=None, redoc_url=None, openapi_url=None)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MODEL_CACHE: dict[str, tuple[Any, dict[str, Any], dict[str, Any]]] = {}
MODEL_LOCK = threading.Lock()
RECENT: deque[dict[str, Any]] = deque(maxlen=20000)
MODEL_DIRS = {
    "experiment_mgtab_mlp": ("MGTAB", "mlp"),
    "experiment_mgtab_graphsage": ("MGTAB", "graphsage"),
    "experiment_mgtab_rgcn": ("MGTAB", "rgcn"),
    "experiment_cresci2015_mlp": ("Cresci-15", "mlp"),
    "experiment_cresci2017_mlp": ("Cresci-2017", "mlp"),
}


templates = Jinja2Templates(directory=str(PROJECT / "frontend" / "pages"))
DATASET_NAMES = {"mgtab": "MGTAB", "cresci2015": "Cresci-15", "cresci2017": "Cresci-2017",
                 "mgtab_large": "MGTAB-large", "snap": "SNAP Twitter"}
FEATURE_LABELS = {
    "statuses_count": "Total posts / statuses", "followers_count": "Followers count",
    "friends_count": "Following / friends count", "favourites_count": "Favourites count",
    "listed_count": "Listed count", "default_profile": "Default profile enabled",
    "default_profile_image": "Default profile image", "geo_enabled": "Geo enabled",
    "profile_use_background_image": "Profile uses background image",
    "profile_background_tile": "Profile background tiled", "utc_offset": "UTC offset",
    "is_translator": "Translator flag", "protected": "Protected account", "verified": "Verified account",
}


@lru_cache(maxsize=10)
def _model_metadata(model_id: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if model_id not in MODEL_DIRS:
        raise HTTPException(400, "That trained model is not available.")
    folder = PROJECT / "outputs" / model_id
    try:
        checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        preprocessing = joblib.load(folder / "preprocessing.joblib")
        metrics_path = folder / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        return checkpoint, preprocessing, folder
    except Exception as exc:
        raise HTTPException(503, f"Saved model {model_id} could not be loaded.") from exc


def _get_model(model_id: str):
    with MODEL_LOCK:
        if model_id in MODEL_CACHE:
            return MODEL_CACHE[model_id]
        checkpoint, preprocessing, folder = _model_metadata(model_id)
        kind = checkpoint["model"]
        if "dataset_names" in checkpoint:
            model = DatasetAwareClassifier(checkpoint["input_dims"], checkpoint["relation_counts"], kind,
                                           checkpoint["hidden_dim"], checkpoint["dropout"])
        elif kind == "mlp":
            model = MLP(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["dropout"])
        elif kind == "graphsage":
            model = GraphSAGE(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["dropout"])
        else:
            model = RelationalGNN(checkpoint["input_dim"], checkpoint["num_relations"],
                                  checkpoint["hidden_dim"], checkpoint["dropout"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        MODEL_CACHE[model_id] = (model, checkpoint, preprocessing)
        return MODEL_CACHE[model_id]


def _features_for(checkpoint: dict[str, Any], preprocessing: Any, dataset: str):
    multi = "dataset_names" in checkpoint
    if multi:
        if dataset not in checkpoint["dataset_names"]:
            raise HTTPException(400, f"{dataset} is not supported by that multi-dataset model.")
        config = preprocessing[dataset]
        names = config["feature_names"]
        input_dim = checkpoint["input_dims"][dataset]
        graph_required = checkpoint["graph_has_edges"][dataset]
    else:
        if checkpoint["dataset"] != dataset:
            raise HTTPException(400, "Selected model was trained for a different dataset.")
        config = preprocessing
        names = checkpoint["feature_names"]
        input_dim = checkpoint["input_dim"]
        graph_required = checkpoint["graph_has_edges"]
    return multi, config, names, input_dim, graph_required


def _adjacency(checkpoint, dataset, edge_index, edge_type, edge_weight, n):
    kind = checkpoint["model"]
    if kind == "rgcn":
        rel_count = checkpoint.get("relation_counts", {}).get(dataset, checkpoint.get("num_relations", 0))
        if not edge_type.numel() and edge_index.numel():
            raise HTTPException(400, "This relational model requires a relation type for every graph edge.")
        if edge_type.numel() and (int(edge_type.min()) < 0 or int(edge_type.max()) >= rel_count):
            raise HTTPException(400, "Graph relation IDs are incompatible with this model.")
        return tuple(normalized_adjacency(edge_index, n, edge_type == r, edge_weight) for r in range(rel_count))
    if kind == "graphsage":
        return normalized_adjacency(edge_index, n, edge_weight=edge_weight)
    return None


def _predict(model_id: str, dataset: str, matrix: np.ndarray, threshold: float,
             edge_index: torch.Tensor | None = None, edge_type: torch.Tensor | None = None,
             edge_weight: torch.Tensor | None = None) -> list[dict[str, Any]]:
    model, checkpoint, preprocessing = _get_model(model_id)
    multi, prep, names, input_dim, graph_required = _features_for(checkpoint, preprocessing, dataset)
    if matrix.ndim != 2 or matrix.shape[1] != input_dim:
        raise HTTPException(400, f"Selected model requires exactly {input_dim} features per account.")
    matrix = prep["imputer"].transform(matrix)
    matrix = prep["scaler"].transform(matrix).astype(np.float32)
    n = len(matrix)
    edge_index = edge_index if edge_index is not None else torch.empty((2, 0), dtype=torch.long)
    edge_type = edge_type if edge_type is not None else torch.empty(0, dtype=torch.long)
    edge_weight = edge_weight if edge_weight is not None else torch.ones(edge_index.shape[1], dtype=torch.float32)
    if graph_required and checkpoint["model"] in {"graphsage", "rgcn"} and not edge_index.numel():
        raise HTTPException(400, "This graph model requires its matching graph input. Use Dataset Analysis with MGTAB.")
    adjacency = _adjacency(checkpoint, dataset, edge_index, edge_type, edge_weight, n)
    with torch.inference_mode():
        x = torch.from_numpy(matrix)
        logits = model(dataset, x, adjacency) if multi else model(x, adjacency if adjacency is not None else edge_index, edge_type)
        scores = torch.sigmoid(logits).cpu().numpy()
    output = []
    for i, score in enumerate(scores):
        bot = float(score)
        human = 1.0 - bot
        if bot >= threshold:
            label = "Likely bot"
        elif bot <= 1.0 - threshold:
            label = "Likely human"
        else:
            label = "Uncertain"
        output.append({"bot_probability": bot, "human_probability": human,
                       "classification": label, "model": model_id, "dataset": dataset,
                       "score_note": "Uncalibrated model score; use as a ranking signal, not certainty."})
    return output


@lru_cache(maxsize=1)
def model_catalog():
    result = []
    for model_id, (title, kind) in MODEL_DIRS.items():
        folder = PROJECT / "outputs" / model_id
        if not (folder / "model.pt").exists():
            continue
        checkpoint, _, _ = _model_metadata(model_id)
        metrics_path = folder / "metrics.json"
        try: metrics = json.loads(metrics_path.read_text(encoding="utf-8")).get("test", {})
        except Exception: metrics = {}
        dataset_id = {"MGTAB": "mgtab", "Cresci-15": "cresci2015", "Cresci-2017": "cresci2017"}[title]
        result.append({"id": model_id, "architecture": kind.upper(), "dataset": title,
                       "dataset_id": dataset_id, "input_dim": checkpoint["input_dim"],
                       "feature_names": checkpoint["feature_names"],
                       "graph_required": bool(checkpoint["graph_has_edges"] and kind in {"graphsage", "rgcn"}),
                       "checkpoint": f"outputs/{model_id}/model.pt",
                       "preprocessing": f"outputs/{model_id}/preprocessing.joblib", "metrics": metrics})
    return result


def dataset_catalog():
    result = []
    for dataset_id, title in DATASET_NAMES.items():
        report_path = PROJECT / "data" / "processed" / dataset_id / "processing_report.json"
        try: report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception: report = {}
        if dataset_id == "cresci2017" and report.get("source_groups"):
            human_count = report["source_groups"].get("genuine_accounts", 0)
            bot_count = sum(count for group, count in report["source_groups"].items() if group != "genuine_accounts")
            report["label_counts"] = {"0": human_count, "1": bot_count}
        if dataset_id == "mgtab_large" and not report.get("label_counts"):
            try:
                manifest = torch.load(PROJECT / "data/processed/mgtab_large/manifest.pt", map_location="cpu", weights_only=True)
                labels = manifest["labels_bot"].reshape(-1)
                report["label_counts"] = {str(int(value)): int((labels == value).sum()) for value in torch.unique(labels)}
                del manifest, labels
            except Exception:
                pass
        models = [model for model in model_catalog() if model["dataset_id"] == dataset_id]
        result.append({"id": dataset_id, "name": title, "report": report,
                       "available": bool(report), "labelled": bool(report.get("label_counts") or report.get("label_mapping")),
                       "models": models})
    return result


def base_context(request: Request, active: str, **extra):
    return {"request": request, "active": active, "date": date.today().strftime("%B %d, %Y"),
            "recent_count": len(RECENT), **extra}


def _single_template(request: Request, error=None, prediction=None, values=None):
    values = values or {}
    checkpoint, _, _ = _model_metadata("experiment_cresci2017_mlp")
    schema = checkpoint["feature_names"]
    groups = [
        ("Account profile", ["statuses_count", "followers_count", "friends_count", "favourites_count", "listed_count"]),
        ("Profile settings", ["default_profile", "default_profile_image", "geo_enabled", "profile_use_background_image", "profile_background_tile", "protected", "verified"]),
        ("Locale and account flags", ["utc_offset", "is_translator"]),
    ]
    boolean_fields = {"default_profile", "default_profile_image", "geo_enabled", "profile_use_background_image",
                      "profile_background_tile", "protected", "verified", "is_translator"}
    grouped = []
    for index, (title, keys) in enumerate(groups, 1):
        fields = [{"key": key, "label": FEATURE_LABELS.get(key, key), "boolean": key in boolean_fields,
                   "value": values.get(key, "")} for key in keys if key in schema]
        grouped.append({"title": title, "index": index, "fields": fields})
    context = base_context(request, "single", error=error, prediction=prediction, groups=grouped,
                           model_id="experiment_cresci2017_mlp", account_id=values.get("account_id", ""))
    return templates.TemplateResponse(request, "single-account.html", context)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    recent = list(RECENT)[:8]
    models = model_catalog()
    datasets = [item for item in dataset_catalog() if item["available"]]
    totals = {"all": len(RECENT), "bot": sum(r["classification"] == "Likely bot" for r in RECENT),
              "human": sum(r["classification"] == "Likely human" for r in RECENT),
              "uncertain": sum(r["classification"] == "Uncertain" for r in RECENT)}
    return templates.TemplateResponse(request, "index.html", base_context(request, "dashboard", totals=totals,
        recent=recent, models=models, datasets=datasets))


@app.get("/single", response_class=HTMLResponse)
def single_page(request: Request):
    return _single_template(request)


@app.post("/single", response_class=HTMLResponse)
def single_predict(request: Request, account_id: str = Form(""), threshold: float = Form(70),
                   statuses_count: str = Form(...), followers_count: str = Form(...), friends_count: str = Form(...),
                   favourites_count: str = Form(...), listed_count: str = Form(...), default_profile: str = Form(...),
                   default_profile_image: str = Form(...), geo_enabled: str = Form(...),
                   profile_use_background_image: str = Form(...), profile_background_tile: str = Form(...),
                   utc_offset: str = Form(...), is_translator: str = Form(...), protected: str = Form(...),
                   verified: str = Form(...), model: str = Form("experiment_cresci2017_mlp")):
    values = {"account_id": account_id, "statuses_count": statuses_count, "followers_count": followers_count,
              "friends_count": friends_count, "favourites_count": favourites_count, "listed_count": listed_count,
              "default_profile": default_profile, "default_profile_image": default_profile_image, "geo_enabled": geo_enabled,
              "profile_use_background_image": profile_use_background_image, "profile_background_tile": profile_background_tile,
              "utc_offset": utc_offset, "is_translator": is_translator, "protected": protected, "verified": verified}
    try:
        threshold = float(threshold) / 100
        if not .51 <= threshold <= .99: raise ValueError("Decision threshold must be between 51% and 99%.")
        if model != "experiment_cresci2017_mlp": raise ValueError("This profile form only supports the Cresci-2017 MLP checkpoint.")
        features = {key: float(value) for key, value in values.items() if key != "account_id"}
        if any(not np.isfinite(value) or (key != "utc_offset" and value < 0) for key, value in features.items()):
            raise ValueError("Count and flag values must be finite. Counts cannot be negative.")
        names = _get_model(model)[1]["feature_names"]
        prediction = _predict(model, "cresci2017", np.asarray([[features[name] for name in names]], dtype=np.float32), threshold)[0]
        prediction["account_id"] = account_id.strip() or "Manual entry"
        RECENT.appendleft(prediction.copy())
        return _single_template(request, prediction=prediction, values=values)
    except (ValueError, TypeError, HTTPException) as exc:
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        return _single_template(request, error=str(detail), values=values)


@app.get("/bulk", response_class=HTMLResponse)
def bulk_page(request: Request, error: str = ""):
    return templates.TemplateResponse(request, "bulk-detection.html", base_context(request, "bulk", models=model_catalog(), error=error))


@app.post("/bulk", response_class=HTMLResponse)
def bulk_predict(request: Request, file: UploadFile = File(...), dataset: str = Form(...),
                 model: str = Form(...), threshold: float = Form(70)):
    context = base_context(request, "bulk", models=model_catalog(), error="")
    try:
        if not file.filename or Path(file.filename).suffix.lower() != ".csv": raise ValueError("Upload a CSV file with a .csv extension.")
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES: raise ValueError("CSV uploads are limited to 20 MB.")
        if not content: raise ValueError("The uploaded CSV is empty.")
        threshold = float(threshold) / 100
        if not .51 <= threshold <= .99: raise ValueError("Threshold must be between 51% and 99%.")
        frame = pd.read_csv(io.BytesIO(content), low_memory=False)
        if frame.empty: raise ValueError("The CSV contains no account rows.")
        checkpoint, prep, _ = _model_metadata(model)
        _, _, names, _, graph_required = _features_for(checkpoint, prep, dataset)
        if graph_required and checkpoint["model"] in {"graphsage", "rgcn"}:
            raise ValueError("This graph model requires the prepared graph. Use Dataset Analysis for graph inference.")
        missing = [name for name in names if name not in frame.columns]
        if missing: raise ValueError(f"CSV is missing {len(missing)} required feature column(s), including: {', '.join(missing[:12])}")
        matrix = frame[names].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
        predictions = _predict(model, dataset, matrix, threshold)
        id_column = next((column for column in ("account_id", "username", "screen_name", "id") if column in frame.columns), None)
        for index, prediction in enumerate(predictions):
            prediction["account_id"] = str(frame.iloc[index][id_column]) if id_column else f"row_{index + 1}"
        RECENT.extendleft(reversed([item.copy() for item in predictions]))
        counts = {label: sum(p["classification"] == label for p in predictions) for label in ("Likely bot", "Likely human", "Uncertain")}
        context.update(predictions=predictions[:300], prediction_count=len(predictions), counts=counts,
                       filename=file.filename, required_columns=names, missing_count=int(np.isnan(matrix).sum()))
        return templates.TemplateResponse(request, "bulk-detection.html", context)
    except Exception as exc:
        context["error"] = exc.detail if isinstance(exc, HTTPException) else str(exc)
        return templates.TemplateResponse(request, "bulk-detection.html", context, status_code=400)


@app.get("/datasets", response_class=HTMLResponse)
def dataset_page(request: Request, error: str = "", success: str = ""):
    return templates.TemplateResponse(request, "dataset-analysis.html", base_context(request, "datasets", datasets=dataset_catalog(),
        models=model_catalog(), error=error, success=success))


@app.post("/datasets/predict", response_class=HTMLResponse)
def dataset_predict(request: Request, dataset: str = Form(...), model: str = Form(...), threshold: float = Form(70)):
    try:
        if dataset not in {"mgtab", "cresci2015", "cresci2017"}:
            raise ValueError("No compatible supervised dataset inference route exists for this dataset.")
        threshold = float(threshold) / 100
        if not .51 <= threshold <= .99: raise ValueError("Threshold must be between 51% and 99%.")
        checkpoint, preprocessing, _ = _model_metadata(model)
        _, _, names, _, graph_required = _features_for(checkpoint, preprocessing, dataset)
        if dataset == "cresci2017":
            frame = pd.read_csv(PROJECT / "data/processed/cresci2017/accounts.csv", low_memory=False)
            matrix = frame[names].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
            edges = torch.empty((2, 0), dtype=torch.long); relations = torch.empty(0, dtype=torch.long); weights = torch.empty(0)
            identifiers = frame["account_id"].astype(str).tolist()
        else:
            obj = torch.load(PROJECT / "data" / "processed" / dataset / "data.pt", map_location="cpu", weights_only=True)
            matrix = obj["x"].cpu().numpy().astype(np.float32)
            edges = obj.get("edge_index") if graph_required else torch.empty((2, 0), dtype=torch.long)
            relations = obj.get("edge_type") if graph_required else torch.empty(0, dtype=torch.long)
            weights = obj.get("edge_weight", torch.ones(edges.shape[1])) if graph_required else torch.ones(edges.shape[1])
            identifiers = obj.get("node_ids", [f"row_{i}" for i in range(len(matrix))])
        predictions = _predict(model, dataset, matrix, threshold, edges, relations, weights)
        for index, prediction in enumerate(predictions): prediction["account_id"] = str(identifiers[index])
        RECENT.extendleft(reversed([item.copy() for item in predictions]))
        counts = {label: sum(p["classification"] == label for p in predictions) for label in ("Likely bot", "Likely human", "Uncertain")}
        return templates.TemplateResponse(request, "dataset-analysis.html", base_context(request, "datasets", datasets=dataset_catalog(),
            models=model_catalog(), predictions=predictions[:300], prediction_count=len(predictions), counts=counts,
            selected_dataset=dataset, selected_model=model, success=f"Predicted {len(predictions):,} accounts using the saved checkpoint."))
    except Exception as exc:
        detail = exc.detail if isinstance(exc, HTTPException) else "Could not run inference. Check that the prepared data and selected model are compatible."
        return templates.TemplateResponse(request, "dataset-analysis.html", base_context(request, "datasets", datasets=dataset_catalog(),
            models=model_catalog(), error=detail, selected_dataset=dataset, selected_model=model), status_code=400)


@app.get("/results", response_class=HTMLResponse)
def results_page(request: Request, classification: str = "all", q: str = "", sort: str = "recent"):
    rows = list(RECENT)
    if classification in {"Likely bot", "Likely human", "Uncertain"}: rows = [row for row in rows if row["classification"] == classification]
    if q: rows = [row for row in rows if q.casefold() in f"{row['account_id']} {row['model']} {row['dataset']}".casefold()]
    if sort == "bot_score": rows.sort(key=lambda row: row["bot_probability"], reverse=True)
    totals = {"all": len(RECENT), "bot": sum(r["classification"] == "Likely bot" for r in RECENT),
              "human": sum(r["classification"] == "Likely human" for r in RECENT),
              "uncertain": sum(r["classification"] == "Uncertain" for r in RECENT)}
    return templates.TemplateResponse(request, "results.html", base_context(request, "results", rows=rows[:500], totals=totals,
        classification=classification, query=q, sort=sort))


@app.get("/results/export")
def export_results():
    output = io.StringIO(newline="")
    fields = ["account_id", "dataset", "model", "bot_probability", "human_probability", "classification", "score_note"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader(); writer.writerows(reversed(RECENT)); output.seek(0)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=bot-detection-results.csv"})


@app.get("/models", response_class=HTMLResponse)
def model_page(request: Request):
    return templates.TemplateResponse(request, "models.html", base_context(request, "models", models=model_catalog()))


app.mount("/static", StaticFiles(directory=str(PROJECT / "frontend" / "css")), name="static")
