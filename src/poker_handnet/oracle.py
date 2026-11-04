"""Independent rules-based ground-truth oracle, used only by tests.

Deliberately written with ordinary Python control flow (a count table,
sorting, set/range checks on at most 5 small integers) instead of the
ReLU-circuit construction in handnet_spec.py, so that the test suite is
checking the net against a second, structurally different implementation
of the same poker rules - not against itself. Never imported by the net or
by any baseline model.

Design derived the same way as handnet_spec.py (poker rules + UCI's
documented class definitions; see DECISIONS.md), independently re-expressed
here as plain Python rather than linear algebra.
"""

from __future__ import annotations

RAW_DIM = 10


def classify_raw(raw: list[int]) -> int:
    """raw: [S1,C1,S2,C2,S3,C3,S4,C4,S5,C5], suit 1..4, rank 1..13 (Ace=1).
    Returns the UCI class 0..9."""
    if len(raw) != RAW_DIM:
        raise ValueError(f"expected {RAW_DIM} values, got {len(raw)}")
    suits = raw[0::2]
    ranks = raw[1::2]

    is_flush = suits[0] == suits[1] == suits[2] == suits[3] == suits[4]

    counts = [0] * 14  # counts[r] = how many of the 5 cards have rank r (index 0 unused)
    for r in ranks:
        counts[r] += 1

    shape = sorted((c for c in counts if c), reverse=True)

    if shape == [4, 1]:
        return 7  # four of a kind
    if shape == [3, 2]:
        return 6  # full house
    if shape == [3, 1, 1]:
        return 3  # three of a kind
    if shape == [2, 2, 1]:
        return 2  # two pair
    if shape == [2, 1, 1, 1]:
        return 1  # one pair

    assert shape == [1, 1, 1, 1, 1], f"impossible rank-count shape: {shape}"
    distinct_ranks = sorted(r for r in range(1, 14) if counts[r])
    is_broadway = distinct_ranks == [1, 10, 11, 12, 13]
    is_run = (distinct_ranks[-1] - distinct_ranks[0]) == 4  # 5 distinct ints spanning 4 must be consecutive
    is_straight = is_broadway or is_run

    if is_straight and is_flush:
        return 9 if is_broadway else 8
    if is_straight:
        return 4
    if is_flush:
        return 5
    return 0


def classify_batch(raw_batch) -> list[int]:
    """Convenience loop wrapper for an (N, 10) array-like of raw hands."""
    return [classify_raw(list(row)) for row in raw_batch]
