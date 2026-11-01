try:
    from _bootstrap import PROJECT
except ModuleNotFoundError:
    from scripts._bootstrap import PROJECT
from src.data.adapters import prepare_mgtab_large

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=PROJECT / "data/raw/mgtab_large/MGStBot-large")
    p.add_argument("--output", type=Path, default=PROJECT / "data/processed/mgtab_large")
    a = p.parse_args(); print(prepare_mgtab_large(a.input, a.output))
