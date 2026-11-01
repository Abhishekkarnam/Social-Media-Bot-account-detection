"""Train the supervised models supported by the prepared labeled datasets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from _bootstrap import PROJECT
except ModuleNotFoundError:
    from scripts._bootstrap import PROJECT


TRAINING_RUNS = (
    ("MGTAB · R-GCN", "mgtab", "rgcn"),
    ("Cresci-15 · MLP", "cresci2015", "mlp"),
    ("Cresci-2017 · MLP", "cresci2017", "mlp"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs per dataset (default: 80).")
    parser.add_argument("--patience", type=int, default=6, help="Early-stopping patience (default: 6).")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive integers.")

    train_script = PROJECT / "train_multidataset.py"
    for index, (label, dataset, model) in enumerate(TRAINING_RUNS, start=1):
        command = [sys.executable, str(train_script), "--dataset", dataset, "--model", model,
                   "--epochs", str(args.epochs), "--patience", str(args.patience)]
        print(f"\n[{index}/{len(TRAINING_RUNS)}] Training {label}", flush=True)
        result = subprocess.run(command, cwd=PROJECT)
        if result.returncode:
            print(f"Training stopped: {label} exited with code {result.returncode}.", file=sys.stderr)
            return result.returncode

    print("\nAll supported labeled-dataset training runs completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
