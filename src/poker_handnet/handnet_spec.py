"""The hand-written poker-hand classifier, specified as a graph of named,
individually-interpretable nodes (see dag.py for how this compiles to a
plain dense Linear->ReLU->...->Linear stack).

DESIGNED WITH ZERO LABELS SEEN. Every formula below follows only from:
  (a) the UCI documentation (data/poker-hand.names): suit in {1..4}
      = {Hearts,Spades,Diamonds,Clubs}, rank in {1..13} = {Ace,2,...,King},
      and the 10 class definitions (0 nothing .. 9 royal flush), including
      the published combinatorics table (section 10), which is dataset
      *documentation*, not row-level label data; and
  (b) elementary combinatorics of a 52-card deck (one card per rank+suit
      pair), which is poker/counting knowledge, not anything learned from
      this dataset's rows.
See DECISIONS.md for the full derivation, including a from-scratch
cross-check of every number in the UCI combinatorics table (section 10)
against this file's logic, performed before any row of either data file
was opened.

THE CORE COMBINATORIAL FACT this design leans on:
A standard deck has exactly one card of each (rank, suit) pair. So if two
cards share a rank, they cannot share a suit (that would be a duplicate
card). Consequently, any repeated rank (pair/trips/quad) forces >=2 distinct
suits among the 5 cards, which makes a flush (all 5 cards one suit)
impossible. Flush is therefore only possible when all 5 ranks are distinct.
This means the rank-count "shape" of a hand (quad / full house / trips /
two pair / pair / all-distinct - the only 6 ways 5 items can split across
counts bounded at 4) and the straight/flush sub-classification (which only
matters in the all-distinct shape) never collide, so a sum of mutually
exclusive 0/1 indicators is exactly the hand's one-hot class vector. This is
checked exhaustively, by hand, for all 6 shapes x all straight/flush
sub-cases in DECISIONS.md.

CONSTRUCTION BLOCKS:
- A hinge ReLU(x - k) is used as plumbing to build exact-equality
  "bump" detectors: for a bounded integer x in {0,1,2,3,4},
    indicator(x==2) = ReLU(x-1) - 2*ReLU(x-2) + ReLU(x-3)
    indicator(x==3) = ReLU(x-2) - 2*ReLU(x-3)
    indicator(x==4) = ReLU(x-3)
  (each checked by direct substitution for x=0..4 in DECISIONS.md).
- AND of two or more 0/1 indicators a,b,... is ReLU(a+b+...-(n-1)).
- Every node's true value is provably always >= 0 (it's either a count, a
  ReLU, or a provably-{0,1}-valued combination - see the case analysis in
  DECISIONS.md), so applying ReLU everywhere is a safe no-op clamp, never a
  distortion; only the final logit layer is left linear before argmax.
"""

from __future__ import annotations

from .dag import CompiledNet, Graph
from .encoding import ENCODED_FEATURE_NAMES, N_RANKS, N_SUITS, RANK_NAMES, SUIT_NAMES

CLASS_NAMES = [
    "nothing",          # 0
    "one_pair",         # 1
    "two_pair",         # 2
    "three_of_a_kind",  # 3
    "straight",         # 4
    "flush",            # 5
    "full_house",       # 6
    "four_of_a_kind",   # 7
    "straight_flush",   # 8
    "royal_flush",      # 9
]


def build_graph() -> tuple[Graph, list[str], list[str]]:
    g = Graph()

    # ---- Layer 0: inputs (must match encoding.ENCODED_FEATURE_NAMES order) ----
    suit_count = {}
    for s in range(1, N_SUITS + 1):
        name = f"suit_count_{s}"
        suit_count[s] = g.input(name, doc=f"# of the 5 cards that are {SUIT_NAMES[s]}", group="input")

    rank_count = {}
    for r in range(1, N_RANKS + 1):
        name = f"rank_count_{r}"
        rank_count[r] = g.input(name, doc=f"# of the 5 cards with rank {RANK_NAMES[r]}", group="input")

    assert list(suit_count.values()) + list(rank_count.values()) == ENCODED_FEATURE_NAMES

    # ---- Suits: flush detector per suit ----
    # suit_count_s in {0..5}; ReLU(count-4) is 1 iff count==5 (can't exceed 5).
    flush_in_suit = {}
    for s in range(1, N_SUITS + 1):
        name = f"flush_in_suit_{s}"
        flush_in_suit[s] = g.add(
            name,
            {suit_count[s]: 1.0},
            bias=-4.0,
            doc=f"1 iff all 5 cards are {SUIT_NAMES[s]} (flush in that suit)",
            group="flush",
        )

    has_flush = g.add(
        "has_flush",
        {flush_in_suit[s]: 1.0 for s in range(1, N_SUITS + 1)},
        doc="1 iff the hand is a flush: all 5 cards share one suit (at most one suit can reach count 5, so this sum is exactly 0/1)",
        group="flush",
    )

    # ---- Ranks: ramp building blocks, then exact-count detectors ----
    # rank_count_r in {0..4} (a deck has only 4 cards per rank).
    ramp1, ramp2, ramp3 = {}, {}, {}
    for r in range(1, N_RANKS + 1):
        ramp1[r] = g.add(f"rank_{r}_ramp1", {rank_count[r]: 1.0}, bias=-1.0,
                          doc=f"building block ReLU(count({RANK_NAMES[r]})-1)", group="ramp")
        ramp2[r] = g.add(f"rank_{r}_ramp2", {rank_count[r]: 1.0}, bias=-2.0,
                          doc=f"building block ReLU(count({RANK_NAMES[r]})-2)", group="ramp")
        ramp3[r] = g.add(f"rank_{r}_ramp3", {rank_count[r]: 1.0}, bias=-3.0,
                          doc=f"building block ReLU(count({RANK_NAMES[r]})-3)", group="ramp")

    quad_rank, trip_rank, pair_rank, rank_present = {}, {}, {}, {}
    for r in range(1, N_RANKS + 1):
        rn = RANK_NAMES[r]
        quad_rank[r] = g.add(f"quad_rank_{r}", {ramp3[r]: 1.0},
                              doc=f"1 iff exactly 4 of the 5 cards are rank {rn} (four of a kind on {rn})",
                              group="shape")
        trip_rank[r] = g.add(f"trip_rank_{r}", {ramp2[r]: 1.0, ramp3[r]: -2.0},
                              doc=f"1 iff exactly 3 of the 5 cards are rank {rn} (three of a kind on {rn})",
                              group="shape")
        pair_rank[r] = g.add(f"pair_rank_{r}", {ramp1[r]: 1.0, ramp2[r]: -2.0, ramp3[r]: 1.0},
                              doc=f"1 iff exactly 2 of the 5 cards are rank {rn} (a pair of {rn}s)",
                              group="shape")
        rank_present[r] = g.add(f"rank_{r}_present", {rank_count[r]: 1.0, ramp1[r]: -1.0},
                                 doc=f"1 iff at least one card is rank {rn} (count clipped to 0/1)",
                                 group="straight")

    has_quad = g.add("has_quad", {quad_rank[r]: 1.0 for r in range(1, N_RANKS + 1)},
                      doc="1 iff some rank has all 4 of its cards in the hand (four of a kind) - at most one rank can, with only 5 cards",
                      group="shape")
    has_trip = g.add("has_trip", {trip_rank[r]: 1.0 for r in range(1, N_RANKS + 1)},
                      doc="1 iff some rank appears exactly 3 times - at most one rank can, with only 5 cards",
                      group="shape")
    num_pair_ranks = g.add("num_pair_ranks", {pair_rank[r]: 1.0 for r in range(1, N_RANKS + 1)},
                            doc="count of ranks that appear exactly twice: 0, 1 (one pair / full house), or 2 (two pair)",
                            group="shape")

    # ---- Straight windows: 9 "ace-low-or-middle" windows (ranks k..k+4 for
    # k=1..9) plus the separate "ace-high" Broadway run (Ace,10,J,Q,K).
    # Together these are exactly the 10 straight-type rank-sets the UCI
    # combinatorics table (section 10) implies - see DECISIONS.md. AND of 5
    # 0/1 indicators via ReLU(sum-4).
    straight_window = {}
    for k in range(1, 10):
        ranks_in_window = list(range(k, k + 5))
        lo, hi = RANK_NAMES[k], RANK_NAMES[k + 4]
        straight_window[k] = g.add(
            f"straight_window_start_{k}",
            {rank_present[r]: 1.0 for r in ranks_in_window},
            bias=-4.0,
            doc=f"1 iff the hand is exactly the straight {lo}-{hi} (ranks {k}..{k+4} all present)",
            group="straight",
        )

    broadway = g.add(
        "broadway",
        {rank_present[1]: 1.0, rank_present[10]: 1.0, rank_present[11]: 1.0,
         rank_present[12]: 1.0, rank_present[13]: 1.0},
        bias=-4.0,
        doc="1 iff the hand is Ten-Jack-Queen-King-Ace (the one straight where Ace plays high, not low)",
        group="straight",
    )

    straight_window_any = g.add(
        "straight_window_any",
        {straight_window[k]: 1.0 for k in range(1, 10)},
        doc="1 iff the hand is a straight from one of the 9 non-Broadway windows (mutually exclusive, so this sum is 0/1)",
        group="straight",
    )

    has_straight = g.add(
        "has_straight",
        {straight_window_any: 1.0, broadway: 1.0},
        doc="1 iff the hand has 5 consecutively ranked cards, Ace playing low or high as needed",
        group="straight",
    )

    two_pair = g.add("two_pair", {num_pair_ranks: 1.0}, bias=-1.0,
                      doc="1 iff the hand has exactly two different paired ranks (two pair)",
                      group="shape")
    has_pair_any = g.add("has_pair_any", {num_pair_ranks: 1.0, two_pair: -1.0},
                          doc="1 iff at least one rank appears exactly twice (true for one pair, two pair, and full house)",
                          group="shape")

    full_house = g.add("full_house", {has_trip: 1.0, has_pair_any: 1.0}, bias=-1.0,
                        doc="1 iff the hand has both a three-of-a-kind and a separate pair (full house)",
                        group="combo")
    trip_alone = g.add("trip_alone", {has_trip: 1.0, full_house: -1.0},
                        doc="1 iff three-of-a-kind present with no separate pair (plain three of a kind)",
                        group="combo")
    one_pair = g.add("one_pair", {has_pair_any: 1.0, two_pair: -1.0, full_house: -1.0},
                      doc="1 iff exactly one paired rank and nothing else (plain one pair)",
                      group="combo")

    straight_flush_nonroyal = g.add(
        "straight_flush_nonroyal", {straight_window_any: 1.0, has_flush: 1.0}, bias=-1.0,
        doc="1 iff the hand is a flush using one of the 9 non-Broadway straight windows (straight flush, excludes royal)",
        group="combo",
    )
    royal_flush = g.add(
        "royal_flush", {broadway: 1.0, has_flush: 1.0}, bias=-1.0,
        doc="1 iff the hand is the Broadway run and a flush (royal flush)",
        group="combo",
    )
    straight_only = g.add(
        "straight_only", {has_straight: 1.0, straight_flush_nonroyal: -1.0, royal_flush: -1.0},
        doc="1 iff the hand is a straight that is not also a flush",
        group="combo",
    )
    flush_only = g.add(
        "flush_only", {has_flush: 1.0, straight_flush_nonroyal: -1.0, royal_flush: -1.0},
        doc="1 iff the hand is a flush that is not also a straight",
        group="combo",
    )
    distinct_ranks_regime = g.add(
        "distinct_ranks_regime", {has_quad: -1.0, has_trip: -1.0, has_pair_any: -1.0}, bias=1.0,
        doc="1 iff all 5 cards have different ranks (no pair/trips/quad) - the only regime where straight/flush/nothing apply",
        group="combo",
    )
    nothing = g.add(
        "nothing",
        {distinct_ranks_regime: 1.0, has_straight: -1.0, has_flush: -1.0,
         straight_flush_nonroyal: 1.0, royal_flush: 1.0},
        doc="1 iff no pair, straight, or flush of any kind (class 0, 'nothing')",
        group="combo",
    )

    class_indicator = [
        nothing, one_pair, two_pair, trip_alone, straight_only, flush_only,
        full_house, has_quad, straight_flush_nonroyal, royal_flush,
    ]
    output_order = []
    for c, node in enumerate(class_indicator):
        logit_name = f"class_logit_{c}"
        g.add(logit_name, {node: 1.0}, doc=f"raw score for class {c} ({CLASS_NAMES[c]}); argmax over these 10 is the prediction", group="output")
        output_order.append(logit_name)

    return g, list(suit_count.values()) + list(rank_count.values()), output_order


def build_handnet() -> CompiledNet:
    g, input_order, output_order = build_graph()
    return g.compile(input_order, output_order)
