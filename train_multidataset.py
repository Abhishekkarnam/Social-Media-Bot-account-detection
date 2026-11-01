"""Compatibility entry point for scripts/train.py.

The previous starter trainer assumed normalized CSVs and fit preprocessing across
all rows. Use the dataset adapters and train-only preprocessing in scripts/train.py.
"""
from scripts.train import main

if __name__ == "__main__":
    main()
