"""Shared loader for the UCI Poker Hand .data CSV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIM = 10


def load_data_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a poker-hand .data file. Returns (raw, labels):
    raw is (n, 10) float64 of S1,C1,...,S5,C5; labels is (n,) int64 class 0-9.
    Uses pandas for speed - the test file is 1,000,000 rows.
    """
    df = pd.read_csv(path, header=None, dtype=np.int64)
    if df.shape[1] != RAW_DIM + 1:
        raise ValueError(f"expected {RAW_DIM + 1} columns in {path}, got {df.shape[1]}")
    rows = df.to_numpy()
    raw = rows[:, :RAW_DIM].astype(np.float64)
    labels = rows[:, RAW_DIM]
    return raw, labels
