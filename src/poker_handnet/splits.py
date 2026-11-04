"""Fixed-seed dev split carved from the training file (integrity ruling #2).

The dev split exists for baseline hyperparameter tuning, per-fit early
stopping / model selection, and Phase 2 error inspection - it is never used
to pick the hand-written net's weights. It IS subtracted from the pool that
every label-budget baseline fit samples from (`scripts/run_baselines.py`'s
`train_pool_idx`): every individual fit (not just the one-time
`tune_hparams` call) early-stops against this split on every call to
`mlp.train_mlp` / `fit_xgb` / `fit_lgb`, so a budget sampled from the full
training file would let that fit's "training" rows overlap the same rows
used to pick when to stop training on them - at the full budget, that
overlap was total. See DECISIONS.md's "baseline budget/dev overlap" entry
for the bug this previously caused and the fix. Because every fit's
pipeline genuinely touches both pools, `scripts/run_baselines.py` reports a
`labels_seen` field (train budget + this split's size) alongside the
train-only `budget` field, rather than pretending the dev rows were never
read.
"""

from __future__ import annotations

import numpy as np

DEV_SEED = 20261103
DEV_FRACTION = 0.2


def dev_split_mask(n_rows: int, dev_fraction: float = DEV_FRACTION, seed: int = DEV_SEED) -> np.ndarray:
    """Boolean mask over row indices [0, n_rows); True marks the dev subset."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    n_dev = int(round(n_rows * dev_fraction))
    mask = np.zeros(n_rows, dtype=bool)
    mask[perm[:n_dev]] = True
    return mask
