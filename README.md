# Twitter Bot Detection: Multi-Dataset GNN Pipeline

This repository adapts the datasets actually present under `data/raw/` to a safe processed format, validates graph/label alignment, trains source-specific baselines, and writes reports and checkpoints. Raw files are read only. The current Python is 3.13.5 with PyTorch 2.7.0 CPU, NumPy 2.2.3, pandas 2.3.1, scikit-learn 1.6.1, and joblib 1.4.2. No virtual environment or PyTorch Geometric is required.

## Data findings and limits

- **MGTAB:** 10,199 users, 788 unnamed numeric columns, labels 0=human and 1=bot, 1,700,108 valid edges across seven relations. Edge order follows the published relation list: followers, friends, mention, reply, quoted, URL, hashtag. The numeric columns are retained as positional `feature_NNNN` fields; their semantic names were not present locally.
- **MGTAB-large:** 410,199 rows and 788 columns. The official loader assigns the 10,199 labels to rows `[0, 10199)`, leaving 400,000 rows unlabeled. Preparation writes a manifest and references the large source graph tensors instead of duplicating them.
- **Cresci-15:** 5,301 users and 1,542 joined dimensions: 1 categorical property, 5 numeric properties, 768 description embedding dimensions, and 768 tweet embedding dimensions. Tensor labels are 0=human (1,950) and 1=bot (3,351). Official train/val/test index files are kept in place (3,708/1,058/535); the original reference loader swapped val/test when assigning masks, so this pipeline does not.
- **Cresci-2017:** all nine account groups were found. The `users.csv` profiles yield 14,368 unique accounts; genuine accounts map to 0 and the social spam, traditional spam, and fake-follower groups map to 1. Tweet tables are not treated as account rows. These archives do not provide a user-user graph.
- **SNAP ego-Twitter:** 81,306 node IDs. The gzip contains 2,420,766 edge lines, including 652,617 duplicate lines; this yields the published 1,768,149 unique directed edges. Its IDs are retained, and no bot labels are assigned. The derived feature file contains directed in/out degree, total degree, and outgoing reciprocity. Clustering is omitted.

Source metadata: [MGTAB authors' repository](https://github.com/GraphDetec/MGTAB), [SNAP ego-Twitter](https://snap.stanford.edu/data/ego-Twitter.html), and the original Cresci-2017 `READ.ME` in the raw archive. Feature names absent from those releases are not inferred.

## Project layout

```text
scripts/                 inspection, validation, preparation, training, inference
src/data/adapters.py     dataset-specific, report-producing adapters
src/models/gnn.py        PyTorch MLP, sparse GraphSAGE, relational GNN
 data/raw/               source files (unchanged)
 data/processed/         derived tensors, tables, manifests, reports
 reports/                full raw inventory
 outputs/                checkpoints, metrics, predictions, configs
 tests/                  unit and smoke tests
```

## Inspect, prepare, validate

Run from this project directory in PowerShell:

Run the full inspection, dataset preparation, and validation pipeline with one command from the project folder:

```powershell
python scripts/prepare_all.py
```

The runner stops at the first failed step and prints which step failed. To run the separate unit tests:

```powershell
python -m unittest discover -v
```

Each adapter writes `data/processed/<dataset>/processing_report.json`. The MGTAB-large preparation uses memory-mapped inspection and does not copy the roughly 2.6 GB feature matrix or concatenate its 88 million edges. SNAP processing uses the full directed edge list but does not load it onto a GPU.

## Models and training

All three models use a supervised binary-logit objective (`human=0`, `bot=1`) and class-weighted binary cross entropy. The trainer uses the supplied Cresci-15 split; MGTAB and Cresci-2017 receive deterministic stratified 70/15/15 splits. Imputation and scaling are fit only on training rows. Validation loss controls early stopping, and the test split is evaluated once after selecting the checkpoint.

Graph runs are transductive: the full available node feature matrix and graph topology are visible to message passing during training, while validation/test labels are withheld from the loss. Use these results for a transductive protocol, not an inductive claim about unseen nodes.

GraphSAGE uses normalized sparse adjacency multiplication, avoiding a huge `x[edge_source]` intermediate. The relational GNN keeps each MGTAB edge type in its own sparse adjacency and transform. The MLP ignores graph edges. Cresci-2017 has no user graph, so its GraphSAGE run has no message passing. These are full-batch experiments; the 410k-node, 88-million-edge MGTAB-large graph is not loaded for supervised training.

Example commands (choose one model and dataset per run):

```powershell
python train_multidataset.py --dataset mgtab --model mlp --epochs 80
python train_multidataset.py --dataset mgtab --model graphsage --epochs 80
python train_multidataset.py --dataset mgtab --model rgcn --epochs 80
python train_multidataset.py --dataset cresci2015 --model mlp --epochs 80
python train_multidataset.py --dataset cresci2017 --model mlp --epochs 80
```

The historical `train_multidataset.py` name remains as a compatibility entry point. Each run saves `model.pt`, `preprocessing.joblib`, `metrics.json`, `run_config.json`, and test predictions under `outputs/<dataset>_<model>/`.

For joint training with different feature dimensions, each source gets its own input encoder and shares the bot/human head. Each graph remains an independent component; dataset losses are averaged so a larger source does not automatically dominate. R-GNN relation transforms are dataset-specific because numeric relation codes do not establish matching semantics across sources.

```powershell
python train_multidataset.py --datasets mgtab,cresci2015,cresci2017 --model mlp --epochs 80
python train_multidataset.py --datasets mgtab,cresci2015 --model graphsage --epochs 80
python train_multidataset.py --datasets mgtab,cresci2015 --model rgcn --epochs 80
```

Optional sampled graph link reconstruction pretraining runs on MGTAB-large or SNAP without labels and without placing the full graph on GPU:

```powershell
python scripts/pretrain_graph.py --dataset mgtab_large --steps 1000 --batch-size 512
python scripts/pretrain_graph.py --dataset snap --steps 1000 --batch-size 512
```

These encoders are saved as self-supervised artifacts. No pretrained checkpoint is transferred into supervised training by default; MGTAB-large is semantically closest to MGTAB, but the feature schema and encoder interfaces must be explicitly matched before transfer. SNAP's four structural features are incompatible with account profile features.

A reusable CSV inference command is:

```powershell
python scripts/predict.py --model-dir outputs/cresci2017_mlp --input data/processed/cresci2017/accounts.csv --output outputs/cresci2017_mlp/predictions.csv
```

Inference validates required feature columns and does not call predictions verified labels. CSV-only inference for relational models is not supported because it requires graph edges and relation IDs.

## HTML/CSS dashboard

The dashboard is rendered by FastAPI/Jinja templates and uses only HTML5 and CSS3 in the browser. No JavaScript files or frontend framework are used. Install the additional server requirements from `requirements.txt`, then start the local server from this project directory:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The pages use native form submissions:

- `/` dashboard and current server-session activity
- `/single` manual Cresci-2017 profile inference using its 14 trained numeric features
- `/bulk` CSV inference for supported MLP checkpoints, with exact trained feature headers
- `/datasets` local prepared dataset inference; MGTAB graph checkpoints use saved edges and relation types
- `/results` session predictions, server-side search/filter/sort; `/results/export` downloads CSV
- `/models` saved model checkpoint details and metrics

SNAP has no supervised labels/checkpoint; MGTAB-large has no supervised checkpoint and only 10,199 mapped labels. These are shown as dataset information, not offered as unsupported prediction paths. Prediction result history is held in memory and resets when the local server restarts. See `api_server.py` for server-side route handling and the compatible model allowlist.

The GitHub repository intentionally contains source code only. Raw/processed datasets, prediction outputs, and `.pt`/`.joblib` model artifacts are excluded by `.gitignore`; obtain the datasets from their official sources and train checkpoints locally before expecting inference models to appear in the dashboard. The API accepts both the bundled `outputs/experiment_<dataset>_<model>/` layout and the trainer's default `outputs/<dataset>_<model>/` layout, provided `model.pt` and `preprocessing.joblib` are together.

## Evaluation boundaries

The primary labels are available for MGTAB and Cresci-2017. Their input schemas are not directly compatible: MGTAB has anonymous profile/text dimensions while Cresci-2017 exposes named profile counts and flags. Cresci-15 includes text embedding groups whose source model metadata is absent. Joint training uses source-specific encoders, but that does not constitute a leave-one-dataset-out experiment. No valid cross-dataset evaluation command is available until a shared feature representation is verified; do not treat the joint training split metrics as external transfer results. Cresci-2017's label groups are campaigns, so its row-random split may share campaign characteristics across partitions. No SNAP/MGTAB-large supervised metrics are produced.

## Dependencies

The inspected environment already had the ML dependencies; FastAPI, Uvicorn, Jinja2, and `python-multipart` were added for the server-rendered dashboard. Check versions with `python -c "import torch,numpy,pandas,sklearn,joblib,fastapi,uvicorn,jinja2,multipart; print(torch.__version__,numpy.__version__,pandas.__version__,sklearn.__version__,joblib.__version__,fastapi.__version__,uvicorn.__version__)"`. Do not install PyTorch Geometric or CUDA wheels for this implementation. If a listed library is missing, install only that requirement after checking its Python 3.13 wheel support; avoid `--upgrade` and do not change unrelated packages.

TwiBot-22 can be added as a new `src/data/adapters.py` adapter with a dataset-specific schema, verified label map, official splits, stable account IDs, and graph mapping. It does not require changes to the existing model classes, but cross-dataset use must wait until its feature compatibility with a trained encoder is established.
