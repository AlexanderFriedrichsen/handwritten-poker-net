"""The fixed, card-order-invariant input encoding shared by the hand-written
net and every trained baseline (integrity ruling: "give the trained
baselines exactly the same encoded input").

Raw UCI features are 10 columns per hand: S1,C1,S2,C2,S3,C3,S4,C4,S5,C5
(suit, rank) x 5 cards, suit in {1..4}, rank in {1..13} (Ace=1 ... King=13).
See data/poker-hand.names (fetched by data/download.py) for the source.

We encode a hand as the element-wise sum, over its 5 cards, of a one-hot
suit vector (4-dim) and a one-hot rank vector (13-dim), concatenated to a
17-dim vector:

    encoded = [suit_count_1..4, rank_count_1..13]

suit_count_s = how many of the 5 cards have suit s
rank_count_r = how many of the 5 cards have rank r

Summing over the 5-card axis is invariant to permuting which raw column
pair is "card 1" vs "card 2" etc, so card order never affects the encoded
vector (and therefore never affects the prediction) - this is the "card
order must not matter" requirement. It is also information-theoretically
sufficient to recover the poker-hand class exactly: see DECISIONS.md for the
combinatorial proof (no label data was used to derive this).
"""

from __future__ import annotations

import numpy as np

N_SUITS = 4
N_RANKS = 13
RAW_DIM = 10
ENCODED_DIM = N_SUITS + N_RANKS

SUIT_NAMES = {1: "Hearts", 2: "Spades", 3: "Diamonds", 4: "Clubs"}
RANK_NAMES = {
    1: "Ace", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
    9: "9", 10: "10", 11: "Jack", 12: "Queen", 13: "King",
}

ENCODED_FEATURE_NAMES = [f"suit_count_{s}" for s in range(1, N_SUITS + 1)] + [
    f"rank_count_{r}" for r in range(1, N_RANKS + 1)
]


def encode(raw: np.ndarray) -> np.ndarray:
    """raw: (..., 10) int array of [S1,C1,S2,C2,S3,C3,S4,C4,S5,C5].
    Returns (..., 17) float64 array [suit_count_1..4, rank_count_1..13].

    Pure vectorized numpy: elementwise comparisons + sum over the 5-card
    axis. No Python-level branching on data values, no per-row loop.
    """
    raw = np.asarray(raw)
    if raw.shape[-1] != RAW_DIM:
        raise ValueError(f"expected last dim {RAW_DIM}, got {raw.shape[-1]}")
    suits = raw[..., 0::2]  # (..., 5) values 1..4
    ranks = raw[..., 1::2]  # (..., 5) values 1..13

    suit_levels = np.arange(1, N_SUITS + 1).reshape((1,) * suits.ndim + (N_SUITS,))
    rank_levels = np.arange(1, N_RANKS + 1).reshape((1,) * ranks.ndim + (N_RANKS,))

    suit_onehot = (suits[..., :, None] == suit_levels).astype(np.float64)  # (...,5,4)
    rank_onehot = (ranks[..., :, None] == rank_levels).astype(np.float64)  # (...,5,13)

    suit_counts = suit_onehot.sum(axis=-2)  # (...,4)
    rank_counts = rank_onehot.sum(axis=-2)  # (...,13)

    return np.concatenate([suit_counts, rank_counts], axis=-1)


def encode_raw_passthrough(raw: np.ndarray) -> np.ndarray:
    """The alternative "standard benchmark" encoding: the raw 10 features,
    unmodified (not order-invariant, not used by the hand-written net, only
    offered as an extra baseline row per the integrity ruling)."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape[-1] != RAW_DIM:
        raise ValueError(f"expected last dim {RAW_DIM}, got {raw.shape[-1]}")
    return raw
