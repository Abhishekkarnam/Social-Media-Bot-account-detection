try:
    from _bootstrap import PROJECT
except ModuleNotFoundError:
    from scripts._bootstrap import PROJECT
from src.data.adapters import prepare_cresci15

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=PROJECT / "data/raw/cresci2015/Cresci-15")
    p.add_argument("--output", type=Path, default=PROJECT / "data/processed/cresci2015")
    a = p.parse_args(); print(prepare_cresci15(a.input, a.output))
