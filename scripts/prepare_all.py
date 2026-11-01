"""Run dataset inspection, all preparation steps, and validation in order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    from _bootstrap import PROJECT
except ModuleNotFoundError:
    from scripts._bootstrap import PROJECT


STEPS = (
    ("Inspect raw datasets", "scripts/inspect_all_datasets.py"),
    ("Prepare MGTAB", "scripts/prepare_mgtab.py"),
    ("Prepare Cresci-15", "scripts/prepare_cresci15.py"),
    ("Prepare Cresci-2017", "scripts/prepare_cresci2017.py"),
    ("Prepare MGTAB-large", "scripts/prepare_mgtab_large.py"),
    ("Prepare SNAP", "scripts/prepare_snap.py"),
    ("Validate prepared datasets", "scripts/validate_datasets.py"),
)


def main() -> int:
    for index, (description, script) in enumerate(STEPS, start=1):
        script_path = PROJECT / Path(script)
        print(f"\n[{index}/{len(STEPS)}] {description}", flush=True)
        result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT)
        if result.returncode:
            print(f"\nStopped: {script} exited with code {result.returncode}.", file=sys.stderr)
            return result.returncode

    print("\nAll dataset preparation steps completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
