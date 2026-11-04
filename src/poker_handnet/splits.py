"""Fixed-seed dev split carved from the training file (integrity ruling #2).

The dev split exists only for baseline hyperparameter tuning and Phase 2
error inspection - it is never used to pick the hand-written net's weights,
and it is not subtracted from the pools that label-budget baseline fits
sample from (those sample from the full training file; the dev split's job
is choosing training *recipes*, such as early-stopping patience or tree
depth, once, not inflating any one fit's effective label budget).
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
