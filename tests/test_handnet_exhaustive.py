"""The core correctness test: every one of the 2,598,960 possible unordered
5-card hands from a standard 52-card deck, fed through the hand-written net,
compared against an independently-written rules oracle (oracle.py) that
shares no implementation with the net's ReLU-circuit construction
(handnet_spec.py). No label data is used anywhere in this file - the ground
truth is derived purely from poker rules.

This is deliberately the *entire* combinatorial domain, not a sample: it is
a proof-by-exhaustion that the net is correct on every possible input, not
just a confidence interval over one.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from poker_handnet import encoding, oracle
from poker_handnet.handnet_spec import CLASS_NAMES, build_handnet

# Published UCI combinatorics (poker-hand.names section 10), identical to
# the standard 5-card-poker-hand frequency table - see DECISIONS.md for the
# derivation. Index = UCI class 0..9.
EXPECTED_CLASS_COUNTS = [
    1_302_540,  # 0 nothing
    1_098_240,  # 1 one pair
    123_552,    # 2 two pair
    54_912,     # 3 three of a kind
    10_200,     # 4 straight
    5_108,      # 5 flush
    3_744,      # 6 full house
    624,        # 7 four of a kind
    36,         # 8 straight flush
    4,          # 9 royal flush
]


@pytest.fixture(scope="module")
def all_hands_raw() -> np.ndarray:
    deck = [(s, r) for s in range(1, 5) for r in range(1, 14)]
    assert len(deck) == 52
    suits_of = np.array([c[0] for c in deck])
    ranks_of = np.array([c[1] for c in deck])

    combos = np.array(list(itertools.combinations(range(52), 5)))
    assert combos.shape == (2_598_960, 5)

    raw = np.empty((combos.shape[0], 10), dtype=np.float64)
    raw[:, 0::2] = suits_of[combos]
    raw[:, 1::2] = ranks_of[combos]
    return raw


@pytest.fixture(scope="module")
def net():
    return build_handnet()


@pytest.fixture(scope="module")
def oracle_truth(all_hands_raw) -> np.ndarray:
    return np.array(oracle.classify_batch(all_hands_raw.astype(int)))


@pytest.fixture(scope="module")
def net_predictions(all_hands_raw, net) -> np.ndarray:
    enc = encoding.encode(all_hands_raw)
    logits = net.forward_numpy(enc)
    return np.argmax(logits, axis=1)


def test_oracle_matches_published_uci_class_counts(oracle_truth):
    """Sanity check on the oracle itself (not the net): its class counts
    must match UCI's documented combinatorics table exactly. This was
    checked before any label was read (see DECISIONS.md)."""
    counts = np.bincount(oracle_truth, minlength=10)
    assert counts.sum() == 2_598_960
    assert counts.tolist() == EXPECTED_CLASS_COUNTS


def test_net_matches_oracle_on_every_possible_hand(net_predictions, oracle_truth, all_hands_raw):
    mismatches = np.flatnonzero(net_predictions != oracle_truth)
    if len(mismatches):
        i = mismatches[0]
        raise AssertionError(
            f"{len(mismatches)} / {len(oracle_truth)} hands misclassified; "
            f"first mismatch: raw={all_hands_raw[i].tolist()} "
            f"net={CLASS_NAMES[net_predictions[i]]} "
            f"oracle={CLASS_NAMES[oracle_truth[i]]}"
        )


def test_net_class_counts_match_published_table(net_predictions):
    counts = np.bincount(net_predictions, minlength=10)
    assert counts.tolist() == EXPECTED_CLASS_COUNTS


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_predictions_are_card_order_invariant(net, seed):
    """Sample one hand per class and check every one of the 120 orderings of
    its 5 cards (not just the encoding, but the net's actual prediction)
    yields the identical class."""
    representative_raw = {
        0: [1, 2, 2, 3, 3, 5, 4, 7, 1, 9],       # nothing
        1: [1, 5, 2, 5, 3, 9, 4, 2, 1, 13],      # one pair
        2: [1, 7, 2, 7, 3, 4, 4, 4, 1, 13],      # two pair
        3: [1, 7, 2, 7, 3, 7, 4, 4, 1, 13],      # trips
        4: [1, 1, 2, 2, 3, 3, 4, 4, 1, 5],       # wheel straight
        5: [2, 2, 2, 5, 2, 7, 2, 9, 2, 13],      # flush
        6: [1, 7, 2, 7, 3, 7, 4, 4, 1, 4],       # full house
        7: [1, 1, 2, 1, 3, 1, 4, 1, 1, 13],      # four of a kind
        8: [3, 4, 3, 5, 3, 6, 3, 7, 3, 8],       # straight flush
        9: [1, 1, 1, 10, 1, 11, 1, 12, 1, 13],   # royal flush
    }
    rng = np.random.default_rng(seed)
    perms = list(itertools.permutations(range(5)))
    for cls, raw in representative_raw.items():
        cards = [tuple(raw[2 * i : 2 * i + 2]) for i in range(5)]
        batch = []
        for perm in perms:
            row = []
            for idx in perm:
                row.extend(cards[idx])
            batch.append(row)
        batch = np.array(batch, dtype=np.float64)
        enc = encoding.encode(batch)
        net_built = build_handnet()
        logits = net_built.forward_numpy(enc)
        preds = np.argmax(logits, axis=1)
        assert np.all(preds == cls), (
            f"class {CLASS_NAMES[cls]}: order affected prediction, "
            f"got classes {set(int(p) for p in preds)}"
        )


def test_param_counts_are_small_and_sparse(net):
    """The point of this project is a *small, hand-derived* circuit, not a
    dense net that happens to work - this guards against accidentally
    growing a dense blob (e.g. a bug causing every weight to be filled in)."""
    counts = net.param_counts()
    assert counts["nonzero_weights"] > 0
    assert counts["nonzero_biases"] > 0
    assert counts["nonzero_params_total"] == counts["nonzero_weights"] + counts["nonzero_biases"]
    # generous ceiling: the real net compiles to a few hundred non-zero
    # params (393 as designed); this is not meant to be a tight pin, just a
    # guard against gross regressions.
    assert counts["nonzero_params_total"] < 5000
