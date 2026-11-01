try:
    from _bootstrap import PROJECT
except ModuleNotFoundError:
    from scripts._bootstrap import PROJECT
from src.data.adapters import prepare_mgtab

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=__import__("pathlib").Path, default=PROJECT / "data/raw/mgtab/MGTAB")
    p.add_argument("--output", type=__import__("pathlib").Path, default=PROJECT / "data/processed/mgtab")
    a = p.parse_args(); print(prepare_mgtab(a.input, a.output))
