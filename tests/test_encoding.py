from __future__ import annotations

import itertools

import numpy as np

from poker_handnet import encoding


def test_encode_shape_and_known_hand():
    # Hearts A, Hearts 10, Spades 5, Spades 5 would be an invalid duplicate-free
    # deck issue, so use a valid 5-distinct-card hand: H-A, H-10, S-5, D-5, C-King
    raw = np.array([1, 1, 1, 10, 2, 5, 3, 5, 4, 13], dtype=np.float64)
    enc = encoding.encode(raw)
    assert enc.shape == (encoding.ENCODED_DIM,)
    # suit_count: Hearts=2, Spades=1, Diamonds=1, Clubs=1
    np.testing.assert_allclose(enc[:4], [2, 1, 1, 1])
    # rank_count: Ace=1, 5 appears twice, 10 once, King once, rest 0
    expected_rank_counts = np.zeros(13)
    expected_rank_counts[0] = 1  # Ace (rank 1 -> index 0)
    expected_rank_counts[4] = 2  # rank 5 -> index 4
    expected_rank_counts[9] = 1  # rank 10 -> index 9
    expected_rank_counts[12] = 1  # King (rank 13 -> index 12)
    np.testing.assert_allclose(enc[4:], expected_rank_counts)


def test_encode_is_batch_vectorized():
    raw = np.array(
        [
            [1, 1, 1, 10, 2, 5, 3, 5, 4, 13],
            [2, 2, 2, 3, 2, 5, 2, 7, 2, 9],
        ],
        dtype=np.float64,
    )
    enc = encoding.encode(raw)
    assert enc.shape == (2, encoding.ENCODED_DIM)
    for i in range(2):
        np.testing.assert_allclose(enc[i], encoding.encode(raw[i]))


def test_encode_is_card_order_invariant():
    base = [1, 1, 1, 10, 2, 5, 3, 5, 4, 13]
    cards = [tuple(base[2 * i : 2 * i + 2]) for i in range(5)]
    base_enc = encoding.encode(np.array(base, dtype=np.float64))

    rng = np.random.default_rng(0)
    perms = list(itertools.permutations(range(5)))
    sample = [perms[i] for i in rng.choice(len(perms), size=20, replace=False)]
    for perm in sample:
        permuted = []
        for idx in perm:
            permuted.extend(cards[idx])
        enc = encoding.encode(np.array(permuted, dtype=np.float64))
        np.testing.assert_allclose(enc, base_enc)


def test_encode_rejects_wrong_width():
    try:
        encoding.encode(np.zeros(9))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong input width")


def test_raw_passthrough_is_unchanged_and_not_order_invariant():
    raw = np.array([1, 1, 1, 10, 2, 5, 3, 5, 4, 13], dtype=np.float64)
    out = encoding.encode_raw_passthrough(raw)
    np.testing.assert_allclose(out, raw)

    swapped = raw.copy()
    swapped[0:2], swapped[2:4] = raw[2:4].copy(), raw[0:2].copy()
    swapped_out = encoding.encode_raw_passthrough(swapped)
    assert not np.allclose(swapped_out, out)  # raw encoding is order-sensitive, unlike encode()
