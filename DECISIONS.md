# Decisions log

Dated, append-only running log of design and methodology decisions for the
hand-written poker-hand net project. Earlier entries are not edited except
to append outcomes; corrections get a new entry.

## 2026-11-03 - Integrity rulings (binding, recorded verbatim-in-substance)

These rulings were issued by the captain to resolve ambiguities in
`spec.md` and take precedence over it where more specific:

1. Freeze the weights (commit the construction script + the compiled
   weights file; record the weights file's sha256 and the freezing commit's
   hash in this log) strictly before reading any label. Only after freezing,
   audit label definitions (ace-low straights, royal vs. straight flush,
   etc.) against the training file, counting every labeled row read (a full
   scan counts as ~25k). Any resulting weight change is the *one* allowed
   Phase 2 revision; v1 (zero labels) and v2 (N labels, if different) are
   reported as separate rows in the results table. The 1M-row test file's
   labels are read only once, for final evaluation - never for design or
   tuning.
2. Carve a dev split from the training file with a fixed seed, used for
   baseline hyperparameter tuning and any Phase 2 error inspection.
3. Maintain this file as a dated running log of every design/methodology
   decision, including these rulings themselves.
4. "Pure matrix ops" means a fixed sequence of matmul + bias + elementwise
   activation (ReLU, or a ReLU-built clamp) + argmax, with no data-dependent
   branching, sorting, gathering, or lookup. The input encoding must be a
   fixed, card-order-invariant function of the raw 10 features; document it,
   and give every trained baseline the identical encoded input. Also report
   baselines on the raw 10-feature input as an extra row if cheap (the
   "standard benchmark" encoding).
5. Count non-zero weights and biases separately, and in total, for every
   model in the results table.

## 2026-11-03 - Environment

- Repo-local venv managed by `uv` (`pyproject.toml` pins Python >=3.11;
  `.python-version` pins 3.12 for this machine). `uv sync` installs numpy,
  torch, pandas, scikit-learn, xgboost, lightgbm, matplotlib, pytest, tqdm.
- RTX 5070 (WSL2, CUDA UMD 13.3) used for PyTorch/XGBoost when CUDA
  initializes cleanly; both baseline scripts probe `torch.cuda.is_available()`
  / XGBoost's `device="cuda"` and fall back to CPU silently on failure. The
  dataset and models here are small enough that CPU fallback is never a
  correctness concern, only a speed one.
- Raw UCI data files are never committed (`.gitignore`); `data/download.py`
  is the reproducible source, plus `data/CHECKSUMS.txt` (sha256 + line
  counts) which *is* committed so the exact bytes used are verifiable
  without re-downloading.

## 2026-11-03 - What counts as "reading a label" during Phase 1

Phase 1 design must see zero labels. The boundary question: does reading
UCI's own published documentation (`poker-hand.names`), which includes a
section with the standard combinatorial class-count table, count as
"reading a label"?

Decision: no. That table is dataset *documentation* describing the general,
well-known mathematics of 5-card poker hand frequencies (identical to the
standard combinatorics found in any poker-probability reference) - it
contains no row-level data from either `.data` file. The ruling explicitly
sanctions designing from `poker-hand.names`. Row-level label reading is
limited to: nothing before freezing; a full structural pass at freeze+1
(the Phase 2 audit, which only reads label *definitions* implicit in rows,
not aggregate statistics); and the one-time final test evaluation.

Concretely, during Phase 1, `data/*.data` files were touched only by
`data/download.py`, which performs byte copying, line counts, and a
column *count* check on the first row - never a value read, never loaded
into pandas/numpy. This was a deliberate, enforced discipline, not an
accident: see `data/download.py`'s own docstring.

## 2026-11-03 - Input encoding

17-dim vector: `[suit_count_1..4, rank_count_1..13]`, each the elementwise
sum over the 5 cards of a one-hot (suit or rank) vector. Implemented in
`src/poker_handnet/encoding.py::encode`.

Why this is sufficient (not just convenient) - a proof, not an assumption:

- A standard 52-card deck has exactly one card per (rank, suit) pair. So two
  cards sharing a rank cannot share a suit (that would be a duplicate
  physical card). A repeated rank therefore *forces* >=2 distinct suits
  among the 5 cards, making "all 5 cards one suit" (flush) impossible.
  Hence **flush is only possible when all 5 ranks are distinct**.
- The rank-count multiset of a 5-card hand (ignoring which card is which)
  can only take 6 shapes, since each count is in {0,1,2,3,4} and the counts
  sum to 5: `{4,1}` (quad), `{3,2}` (full house), `{3,1,1}` (trips),
  `{2,2,1}` (two pair), `{2,1,1,1}` (one pair), `{1,1,1,1,1}` (all distinct).
  The first five shapes each map to exactly one UCI class (7, 6, 3, 2, 1)
  regardless of which suits are involved, because they already contain a
  repeat and are therefore (by the point above) never a flush, and a
  hand with any repeated rank can never be 5 *consecutive* ranks either
  (a straight needs 5 distinct, consecutive rank values) - so straight and
  flush sub-classification is irrelevant there.
- Only the `{1,1,1,1,1}` (all-distinct-ranks) shape needs the straight/flush
  sub-split (classes 0, 4, 5, 8, 9), and both "is it a straight" (a function
  of which 5 distinct rank values appear) and "is it a flush" (a function of
  whether a single suit count hits 5) are fully determined by the
  rank-count and suit-count vectors alone, with no dependence on which
  physical card carries which rank/suit.
- Therefore `[suit_count_1..4, rank_count_1..13]` determines the UCI class
  exactly. This is checked exhaustively by hand for all 6 shapes x all
  straight/flush sub-cases below ("Hand-derivation case analysis").
- Summing over the 5-card axis is manifestly invariant to which raw column
  pair is "card 1" vs "card 2", etc, so card order never affects the
  encoding (the "card order must not matter" requirement).
- `encoding.py::encode_raw_passthrough` keeps the unmodified raw 10 features
  available as the alternative "standard benchmark" input, reported as an
  extra baseline row per ruling 4.

## 2026-11-03 - Architecture: graph compiler instead of hand-built matrices

Hand-tracking which of ~300 named quantities must be carried forward
(as identity "passthrough" copies) through up to ~8-10 dense layers, by
hand, is exactly the kind of bookkeeping where an off-by-one hides. Built
`src/poker_handnet/dag.py` instead: a small `Graph` builder (`input()`,
`add(name, {parent: weight, ...}, bias, doc, group)`) plus a compiler that:

1. assigns each node a depth = 1 + max(parent depths);
2. computes how late each node's value is still needed by a consumer;
3. mechanically inserts identity passthrough nodes to bridge any gap;
4. emits one dense `(W, b)` pair per depth, ReLU on every layer except the
   final (linear, since raw logits feed argmax).

This is poker-agnostic and is unit-tested on its own (`tests/test_dag.py`)
independent of any poker logic, so a bug in the poker derivation and a bug
in the plumbing can't be confused for each other.

ReLU is applied on every hidden layer. This is a safe no-op clamp rather
than a distortion only because every node in the poker graph is *proven*
(see the case analysis below) to always be >= 0 - this is the "clamp built
from ReLUs" the spec permits, not a coincidence that happens to work.

## 2026-11-03 - Hand-derivation case analysis (the correctness proof)

Full node-by-node design lives in `src/poker_handnet/handnet_spec.py`
(every node has an inline `doc` string with its poker meaning - this
doubles as the interpretability deliverable). Summary of the construction:

**Exact-count ("bump") detectors.** For a bounded integer `x` in
`{0,1,2,3,4}` (a rank count; at most 4 cards share a rank in a 52-card
deck), define ramps `r1=ReLU(x-1)`, `r2=ReLU(x-2)`, `r3=ReLU(x-3)`. Then:

| x | r1 | r2 | r3 | r1-2r2+r3 | r2-2r3 | r3 |
|---|----|----|----|-----------|--------|----|
| 0 | 0  | 0  | 0  | 0         | 0      | 0  |
| 1 | 0  | 0  | 0  | 0         | 0      | 0  |
| 2 | 1  | 0  | 0  | 1         | 0      | 0  |
| 3 | 2  | 1  | 0  | 0         | 1      | 0  |
| 4 | 3  | 2  | 1  | 0         | 0      | 1  |

So `pair_rank = r1-2r2+r3` is `indicator(x==2)`, `trip_rank = r2-2r3` is
`indicator(x==3)`, `quad_rank = r3` is `indicator(x==4)` - checked by direct
substitution for every possible value of `x`, not just typical ones.
`rank_present = x - r1` is `indicator(x>=1)` (clips 2,3,4 down to 1).

**AND of k binary indicators** `a1..ak` is `ReLU(sum(ai) - (k-1))` (fires
only when every `ai` is 1 since each is in {0,1}).

**Shape indicators** (`has_quad`, `has_trip`, `num_pair_ranks` = sum over
13 ranks of the per-rank bump; `two_pair = ReLU(num_pair_ranks-1)`;
`has_pair_any = num_pair_ranks - two_pair`; `full_house =
ReLU(has_trip+has_pair_any-1)`; `trip_alone = has_trip - full_house`;
`one_pair = has_pair_any - two_pair - full_house`) directly implement the
6-shape case split above: at most one rank can be a trip or a quad (5
cards), and at most two ranks can be pairs (3 pairs would need 6 cards), so
each of these sums is provably in `{0,1,2}` as annotated, never higher.

**Straight detectors.** 9 sliding windows (`straight_window_start_k` for
ranks `k..k+4`, k=1..9) each AND their 5 `rank_present` indicators; plus one
separate `broadway` ANDing `rank_present` for {Ace,10,J,Q,K} (the one
straight where Ace plays high). These 10 rank-sets are exactly the 10
distinct straights in a standard deck (A2345 low straight through the
broadway run), matching the standard count of 10 straight rank-sequences.
`has_straight = straight_window_any + broadway` (mutually exclusive since
they require different rank-presence patterns once ranks are all distinct).

**Flush detector.** `flush_in_suit_s = ReLU(suit_count_s - 4)` is
`indicator(suit_count_s==5)` (suit counts only reach 5 when all 5 cards
share a suit; can't exceed 5). `has_flush = sum_s flush_in_suit_s` (at most
one suit can hit 5 with only 5 cards, so this sum is 0/1).

**Final combination**, using the flush-requires-distinct-ranks fact:

```
straight_flush_nonroyal = AND(straight_window_any, has_flush)
royal_flush             = AND(broadway, has_flush)
straight_only           = has_straight - straight_flush_nonroyal - royal_flush
flush_only              = has_flush    - straight_flush_nonroyal - royal_flush
distinct_ranks_regime   = ReLU(1 - has_quad - has_trip - has_pair_any)
nothing                 = distinct_ranks_regime - has_straight - has_flush
                          + straight_flush_nonroyal + royal_flush
```

**Exhaustive hand-check of all 6 shapes x straight/flush sub-cases**
(performed before writing a line of test code, each verified by substituting
every indicator's value):

- `{4,1}` quad: only `has_quad=1`. All straight/flush terms are 0 (only 2
  distinct ranks present, never enough for any 5-rank window; flush
  impossible by the repeated-rank argument). -> class 7. Correct.
- `{3,2}` full house: `has_trip=1`, `has_pair_any=1` -> `full_house=1`,
  everything else derived from these is 0. -> class 6. Correct.
- `{3,1,1}` trips alone: `has_trip=1`, `has_pair_any=0` -> `trip_alone=1`
  only (3 distinct ranks present, can't fill a straight window). -> class 3.
- `{2,2,1}` two pair: `num_pair_ranks=2` -> `two_pair=1` only (3 distinct
  ranks present). -> class 2.
- `{2,1,1,1}` one pair: `num_pair_ranks=1` -> `one_pair=1` only (4 distinct
  ranks present - even a window-aligned coincidence only sums to 4, never
  reaching the AND's threshold of 5). -> class 1.
- `{1,1,1,1,1}` all distinct, sub-cases on (straight?, flush?):
  - neither: `distinct_ranks_regime=1`, nothing else fires -> `nothing=1`.
    -> class 0.
  - straight (non-broadway window), not flush: `straight_only=1` only.
    -> class 4.
  - broadway, not flush: `straight_only=1` only (broadway without a flush
    has no special UCI class - it's just a straight). -> class 4.
  - flush, not straight: `flush_only=1` only. -> class 5.
  - non-broadway window + flush: `straight_flush_nonroyal=1` only. -> class 8.
  - broadway + flush: `royal_flush=1` only. -> class 9.

Every one of the 6 shapes x applicable sub-cases yields exactly one class
indicator equal to 1 and all nine others equal to 0. This is a full case
analysis over every possible 5-card hand (not sampling), so it is a proof of
correctness given correct arithmetic - independently re-checked
mechanically by the exhaustive enumeration test
(`tests/test_handnet_exhaustive.py`) against an independently-written
oracle (`src/poker_handnet/oracle.py`) that shares no logic with
`handnet_spec.py`.

## 2026-11-03 - Cross-check against UCI's published combinatorics (zero labels read)

`poker-hand.names` documents the standard 5-card poker hand frequency table
(identical to the general combinatorics of a 52-card deck, independent of
any dataset row). Computed independently from first principles and checked
against the hand-derivation above:

| class | UCI name | count (of 2,598,960) |
|---|---|---|
| 9 | Royal flush | 4 |
| 8 | Straight flush | 36 |
| 7 | Four of a kind | 624 |
| 6 | Full house | 3,744 |
| 5 | Flush | 5,108 |
| 4 | Straight | 10,200 |
| 3 | Three of a kind | 54,912 |
| 2 | Two pairs | 123,552 |
| 1 | One pair | 1,098,240 |
| 0 | Nothing | 1,302,540 |

All ten match the standard reference values exactly (sum = 2,598,960 =
`C(52,5)`), which independently confirms the ace-low-straight convention
(10 straight rank-sequences, not 9: A2345 counts) and the
royal-vs-straight-flush split (4 royal + 36 non-royal = 40 total straight
flushes = 10 sequences x 4 suits) used in the hand-derivation above, without
reading a single labeled row of either `.data` file.

## 2026-11-03 - Pre-freeze validation (zero labels read)

Before freezing, the full label-free test suite (`tests/`) was run and
passed (23 tests): `test_dag.py` (compiler, no poker content),
`test_encoding.py` (shape, order-invariance, known-hand spot check),
`test_parity.py` (numpy vs PyTorch forward pass, npz round-trip), and the
key result in `test_handnet_exhaustive.py`:

- Every one of the **2,598,960** possible unordered 5-card hands from a
  standard 52-card deck (the full combinatorial domain, not a sample) was
  fed through the compiled net and compared against `oracle.py`, an
  independently-written, structurally different (plain Python control flow,
  not a ReLU circuit) rules-based classifier. **0 mismatches.**
- The net's resulting class distribution over all 2,598,960 hands is
  `[1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 36, 4]` for
  classes 0-9, matching the UCI-published table above exactly.
- A card-order-permutation check (all 120 orderings of one representative
  hand per class, x3 random seeds) confirmed predictions are order
  invariant, as the encoding's construction guarantees.
- `net.param_counts()`: 335 non-zero weights, 58 non-zero biases, 393
  non-zero params total, across 8 layers.

No labeled row of either `.data` file was opened by any code path exercised
above - `oracle.py` and `handnet_spec.py` are both derived purely from
poker rules and `poker-hand.names` documentation (see earlier entries).

## 2026-11-03 - Weight freeze

Ran `scripts/build_weights.py`, which calls `handnet_spec.build_handnet()`
(poker-rules-derived graph, zero labels read) and serializes it with no
other input:

- Weights file: `weights/handnet_v1.npz`
- sha256: `708d63be05b53b2bb421405af5c6ad2901d8ec69689b299f14eb7e3b49c5d78a`
- Freezing commit hash: recorded in a tiny follow-up commit immediately
  after the freezing commit itself (a commit cannot contain its own hash);
  that follow-up commit also reads zero labels. See the entry below.

This is v1: the zero-labels-seen hand-written net. Everything up to and
including this commit was produced and validated without reading a single
row of either `poker-hand-training-true.data` or `poker-hand-testing.data`.

## 2026-11-03 - Freezing commit hash

The freezing commit (construction script + `weights/handnet_v1.npz` +
tests + this file) is `263f105` ("Freeze hand-written poker-hand net
weights (zero labels read)"). This entry is itself a zero-label commit.

## 2026-11-03 - Phase 2 label-definition audit

Per integrity ruling #1, ran `scripts/audit_labels.py` to read
`data/poker-hand-training-true.data` for the first time (and the first read
of any label anywhere in this project). This is also the first commit in
the project's history that reads label data; it strictly follows the
freeze commits `263f105` / `a1db650`.

- Rows read: 25,010 (full scan of the training file, matching
  `data/CHECKSUMS.txt`'s recorded line count).
- For every row, compared both the frozen net's prediction
  (`weights/handnet_v1.npz`, forward pass on the row's 17-dim encoding) and
  the independent oracle's classification (`oracle.classify_raw` on the raw
  10 features) against UCI's own label column.
- Net vs. UCI label mismatches: **0 / 25,010**.
- Oracle vs. UCI label mismatches: **0 / 25,010**.
- Net vs. oracle mismatches: **0 / 25,010** (expected - already proven
  identical over the full 2,598,960-hand domain before freezing; this is a
  consistency check, not new information).
- Full machine-readable output: `results/label_audit.json`.

Finding: UCI's training-file labels use exactly the straight/flush/royal
conventions (ace-low straight, ace-high straight, royal flush as a
straight flush containing the ace-high broadway run, no wraparound
straights) that `handnet_spec.py` was hand-derived from and that the
pre-freeze exhaustive validation already checked against UCI's published
aggregate class-count table. The audit surfaced no convention mismatch and
no row-level surprise.

**Outcome: the one allowed Phase 2 revision is not exercised.** v1 (the
frozen zero-label weights, sha256
`708d63be05b53b2bb421405af5c6ad2901d8ec69689b299f14eb7e3b49c5d78a`) is
confirmed correct on every training-file label and stands unchanged as the
net used for the rest of this project. Per ruling #1, `RESULTS.md` will
still report this explicitly as "v1 == v2 (no revision triggered)" rather
than silently omitting the v2 row, so the audit step itself is auditable.

## 2026-11-03 - Final test-set evaluation of the hand-written net

Per ruling #1 ("The 1M-row test file's labels are read only once, for
final evaluation - never for design or tuning"), ran
`scripts/eval_test_set.py`: the first and only read of
`data/poker-hand-testing.data`'s labels anywhere in this project, and the
only time this file is used for anything but a baseline's final scoring.

- Rows evaluated: 1,000,000 (full file).
- Accuracy: **1.000000** (1,000,000 / 1,000,000).
- Macro-F1: **1.000000**.
- Per-class recall: 1.000000 on all 10 classes, including the 12
  straight-flush rows and all 3 royal-flush rows in the file.
- Forward pass wall time: 0.78s for 1,000,000 rows (pure numpy).
- Full breakdown: `results/handnet_test_eval.json`.

This is not a surprise - it is the expected consequence of the pre-freeze
exhaustive proof (0 mismatches against the independent oracle across all
2,598,960 possible unordered hands) and the Phase 2 audit (0 mismatches
against UCI's actual training labels): the net was already proven correct
on every hand that exists, including every hand in the test file. Reported
here anyway, in full, because the ruling requires the test file read and
`RESULTS.md` must state this plainly rather than asserting it secondhand
from the combinatorial proof alone.

## 2026-11-03 - Demo gap found on re-reading spec.md: added live neuron activations

Re-read `spec.md` Phase 5 directly before drafting the remaining deliverables,
rather than continuing from my own paraphrase of it. Its exact wording: "a
single self-contained HTML page where the user picks 5 cards and sees the
predicted hand plus **live activations of the named neurons**." The demo as
it stood (built and browser-verified earlier this session) showed only the
final prediction, the 10 output logits, and the 17-dim encoded input - no
hidden-layer activations at all, despite every neuron's name and doc string
already being exported to `demo/weights.json` via `out_names`/`out_docs`
(`export_js.py:20-22`). This was a real gap against an explicit requirement,
not a style choice.

Fixed by extending `demo/forward_pass.js`'s forward pass to return every
intermediate layer's activation vector (`forwardWithActivations`), not just
the final logits, and adding a new panel to `demo/template.html` that
renders all 7 hidden layers' neurons (168 total) grouped by layer, each row
showing its name, live value, and (on hover) its full doc string, with
nonzero ("fired") neurons highlighted. `classify()`'s return value gained an
`activations` field; existing callers (`classIndex`/`logits`/`encoded`) are
unchanged, so `tests/js_parity_runner.js` needed no changes.

Verified live in a real browser (Playwright, same `http://127.0.0.1:8743`
local-server workaround as the earlier verification pass - Playwright still
blocks `file://`, the shipped artifact is unaffected). Royal-flush preset
traces a coherent causal chain: `flush_in_suit_3` and the five Broadway
`rank_count_*__pt1` neurons fire in layer 0, `has_flush`/`rank_*_present`
in layer 1, `broadway`/`has_flush__pt3` in layer 2, `royal_flush` itself
fires in layer 3, and `class_logit_9` propagates through the remaining
layers to the output. "Nothing" preset fires only 12 of 168 hidden neurons
(five rank-presence detectors, `distinct_ranks_regime`, and the `nothing`
concept) - no spurious pair/flush/straight activity. Full pytest suite
(26 tests, including the JS/numpy parity test) still passes. No console
errors in either case.
