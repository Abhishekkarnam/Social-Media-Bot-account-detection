"""Compatibility wrapper for the Cresci-2017 account converter."""
from scripts._bootstrap import PROJECT
from src.data.adapters import prepare_cresci2017

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=PROJECT / "data/raw/cresci2017")
    p.add_argument("--output-dir", type=Path, default=PROJECT / "data/processed/cresci2017")
    a = p.parse_args()
    print(prepare_cresci2017(a.input_dir, a.output_dir))
