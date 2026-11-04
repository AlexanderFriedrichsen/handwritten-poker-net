# handwritten-poker-net

A feedforward neural network that classifies 5-card poker hands into the UCI Poker Hand
dataset's 10 classes, with every weight derived by hand from the rules of poker - no gradient
descent, no label ever seen during construction - benchmarked against generously-tuned trained
baselines (small/large MLP, XGBoost, LightGBM) on the same task and the same input encoding.

**Headline result:** the hand-written net scores 100.0000% test accuracy and 1.0000 macro-F1 on
the full 1,000,000-row UCI test set, using 393 non-zero parameters and zero training labels. No
trained baseline in this project's sweep matches or beats it on any metric, at any label budget.
Full numbers, plots, and caveats are in [`RESULTS.md`](RESULTS.md).

```
pip install uv   # or: see https://docs.astral.sh/uv/
scripts/reproduce_all.sh
```

regenerates every weight, figure, number, and demo file in this repo from scratch in one
command. See [Reproducing everything](#reproducing-everything) below.

## What's actually in this repo

| Deliverable | Where |
|---|---|
| Dataset download (never committed raw) | `data/download.py` -> `data/*.data` |
| Input encoding (shared by the hand-written net and every baseline) | `src/poker_handnet/encoding.py` |
| Hand-written weight construction, one neuron at a time | `src/poker_handnet/handnet_spec.py` |
| Frozen weights | `weights/handnet_v1.npz` |
| Poker-agnostic graph-to-dense-layers compiler | `src/poker_handnet/dag.py` |
| Independent oracle (for validation, not construction) | `src/poker_handnet/oracle.py` |
| Label-definition audit | `scripts/audit_labels.py`, `results/label_audit.json` |
| Final test-set evaluation | `scripts/eval_test_set.py`, `results/handnet_test_eval.json` |
| Trained baselines (MLP small/large, XGBoost, LightGBM) | `scripts/run_baselines.py`, `results/baselines.json` |
| Weight heatmaps + neuron reference | `scripts/plot_weights.py` -> `figures/`, `figures/neuron_reference.md` |
| Accuracy/macro-F1-vs-labels plot | `scripts/plot_results.py` -> `figures/accuracy_vs_labels.png` |
| Self-contained browser demo (live activations) | `scripts/build_demo.py` -> `demo/index.html` |
| Results write-up | `RESULTS.md` |
| Decision log (every non-obvious call, dated, with reasoning) | `DECISIONS.md` |
| Draft X thread (never posted) | `docs/x-thread.md` |

## The hand-written net

The construction has three steps, in order:

1. **Encoding.** A hand of 5 cards is encoded as a 17-dim vector: 4 suit counts + 13 rank
   counts. This is lossy relative to the raw 10 features (2 per card: suit, rank, in whatever
   order the 5 cards were dealt) in exactly the way poker itself is: hand strength does not
   depend on card order, and a standard 52-card deck has exactly one card per (rank, suit) pair,
   so a repeated rank forces at least two distinct suits - meaning every one of UCI's 10 classes
   is fully determined by the rank-count and suit-count vectors alone, with no dependence on
   which physical card carried which value. The proof is in `DECISIONS.md`'s encoding entry; it
   is checked empirically too (`tests/test_handnet_exhaustive.py`) against all 2,598,960 possible
   5-card hands from a standard deck.
2. **Graph, then compile.** `handnet_spec.py` builds a small dataflow graph - one named node per
   human-meaningful quantity ("count of hearts", "rank 7 present", "five consecutive ranks
   starting at rank 10", ...) - using `dag.py`'s `Graph` builder. `dag.py` is poker-agnostic: it
   assigns each node a depth, inserts identity passthrough nodes to bridge any gap between a
   node's depth and the depth its last consumer needs it at, and emits one dense `(W, b)` pair
   per depth with ReLU on every layer except the final linear one (whose 10 outputs feed
   `argmax`, never evaluated at construction time against anything but ReLU's own
   non-negativity proof - see below). The result is a genuine 8-layer feedforward network
   (`17 -> 56 -> 53 -> 14 -> 9 -> 11 -> 15 -> 10 -> 10`), pure matmul + bias + ReLU, no
   if-statements, lookup tables, or Python branching anywhere in the forward pass.
3. **Freeze, then look at a test set only.** The weights were serialized to
   `weights/handnet_v1.npz` (sha256 `708d63be05b53b2bb421405af5c6ad2901d8ec69689b299f14eb7e3b49c5d78a`,
   committed at `263f105`) before any label, training or test, was read anywhere in this
   project. The one allowed revision pass (Phase 2: inspect a labeled dev split, make one fix)
   found zero mismatches over all 25,010 training rows and was not exercised - v1 is the net
   evaluated in `RESULTS.md`.

ReLU is provably not a distortion here, not just empirically harmless: every node in the poker
graph (card counts, rank-present indicators, run-length detectors, and so on) is proven
non-negative by the construction itself, so clipping at zero never discards information ReLU
would otherwise destroy on this specific graph. That proof, and the per-neuron meaning of every
node, is in `src/poker_handnet/handnet_spec.py`'s construction comments and
`figures/neuron_reference.md`.

## "This is just a compiled rule set, not learning"

This is the right objection to raise, and it is correct as far as it goes: the hand-written net
is exactly what it looks like - a decision procedure for 5-card poker hand ranking, written by a
human (well, an AI) who knows the rules, then mechanically compiled into the shape of a neural
network rather than expressed as `if`/`else`. It did not discover that a flush beats a straight
by generalizing from examples; it was told, and the telling is encoded directly in a weight
matrix instead of in Python control flow. Calling that "learning" in the sense of statistical
generalization from data would be dishonest, and this project does not claim otherwise anywhere
in `RESULTS.md` or here.

What it does demonstrate, and what makes it worth building rather than just asserting:

- **The compiled rules really are a neural network, not a network-shaped wrapper around a rule
  engine.** Inference is pure matmul + bias + ReLU + argmax (`src/poker_handnet/dag.py`'s
  `CompiledNet.forward_numpy`, mirrored bit-for-bit in `demo/forward_pass.js` and checked by
  `tests/test_js_parity.py`). There is no branch, lookup table, or sort anywhere at inference
  time - the rules are not hiding behind the matrix multiplication, they *are* the matrix
  multiplication. That is a real, checkable constraint this project holds itself to, not a
  framing choice.
- **It is a meaningful lower bound on what "knowing the rules perfectly" looks like as a
  parameter count and an accuracy ceiling.** 393 non-zero parameters, 100.0000% accuracy, 1.0000
  macro-F1, exact on every one of the 10 classes including the three rarest (12, and 36, and 4
  examples respectively, out of 1,000,000 test rows - see `RESULTS.md`'s rare-class section).
  Every trained baseline in this sweep, including one trained on 500,000 labels (20x the entire
  training file), falls short of that ceiling on macro-F1 and rare-class recall, not just
  nominally but by a wide margin (closest: `mlp_large` at 25,010 labels, 0.9018 macro-F1 vs.
  1.0000; see `RESULTS.md`). That gap is the actual empirical content of this project: not "a
  rule-based system beats a statistical one" in the abstract, but a specific, measured answer to
  "how much labeled data does it take a trained model to reach the ceiling a correct rule set
  reaches with zero," on this exact task, with this exact fair encoding, with generous
  hyperparameter tuning on the trained side.
- **The task was chosen, and the comparison was built, to make this objection easy to raise
  convincingly, not to dodge it.** Poker hand ranking is a closed-form, fully specified function
  of the cards - there is no hidden statistical structure a learned model could uncover that the
  rules don't already capture, which is exactly why "just compile the rules" is *possible* here
  in a way it is not for most real learning tasks (the viral legal-citation-classifier post this
  project is a response to involved messier, less fully specified structure). This project is
  not claiming the hand-written approach generalizes to tasks where the rules are not already
  known and total; it is claiming that *when* the rules are known and total, writing them down
  as a correct, checkable network and then honestly measuring the gap to statistical learning on
  identical footing is more informative than asserting either "learning will always catch up" or
  "rules always win." Here, on this task, the rules win outright, and the margin is reported
  exactly rather than rounded toward whichever story is more flattering.

If the measure of "learning" is "discovered the rule from data," the hand-written net did not
learn anything. If the measure is "a neural network that is provably, checkably correct on the
full domain, expressed with no branching logic at inference time," it earned that the hard way
(see `tests/test_handnet_exhaustive.py`'s exhaustive 2,598,960-hand check) rather than being
handed it. Both of those statements are true at once, and `RESULTS.md` and this section are
written so neither one is left out.

## Reproducing everything

```
scripts/reproduce_all.sh
```

runs, in order: weight construction and freezing (reads no label), the label-definition audit,
the final test-set evaluation, the full baseline sweep (38 rows, ~17 minutes), the weight
heatmap and neuron-reference figures, the accuracy-vs-labels plot, the browser demo build, and
the full test suite (26 tests: exhaustive-hand correctness, card-order invariance, JS/numpy/torch
parity, frozen-weights-match-current-spec, encoding properties). Every number in `RESULTS.md` and
every figure in `figures/` is produced by this one command from source; nothing is hand-edited
after the fact.

## Try the demo

Open `demo/index.html` directly in a browser (no server, no build step - it is fully
self-contained, including the exported weights). Pick 5 cards; it runs the exact same forward
pass as the Python evaluation (`demo/forward_pass.js`, checked bit-identical to
`src/poker_handnet/dag.py`'s numpy forward pass by `tests/test_js_parity.py`) in JavaScript, and
shows the predicted hand alongside every named neuron's live activation for that hand.

## Dataset

UCI Poker Hand (id 158): 10 features (suit + rank for each of 5 cards), 10 classes (0 = nothing
through 9 = royal flush), 25,010 training rows, 1,000,000 test rows. `data/download.py` fetches
the official bundle from archive.ics.uci.edu and writes sha256 checksums to
`data/CHECKSUMS.txt`; raw data files are gitignored and re-downloaded by that script, never
committed. Label-definition conventions (ace-low/ace-high straights, royal flush as a subset of
straight flush) were verified empirically against the training file before being relied on -
see `scripts/audit_labels.py` and `RESULTS.md`'s label-audit section - rather than assumed from
the dataset documentation alone.
