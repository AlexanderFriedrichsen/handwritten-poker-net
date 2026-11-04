# Draft X thread (not posted)

Draft only. 6 posts, numbered for order; each is meant to stand as one post. Written to be
honest first and punchy second - trim for length before posting, but do not cut any caveat to
make it shorter. Links (`[demo]`, `[repo]`) are placeholders for the user to fill in once
published; nothing in this repo has been pushed anywhere, so there is no real link yet.

---

**1/6**
I hand-derived every weight of a neural network from the rules of poker. Zero training labels,
pure matmul + ReLU, no if-statements at inference time. Benchmarked it against trained models on
the UCI Poker Hand dataset (1,000,000 test hands).

Result: 100.0000% accuracy, 1.0000 macro-F1, 393 non-zero parameters.

**2/6**
The net is a real 8-layer feedforward network (17 -> 56 -> 53 -> 14 -> 9 -> 11 -> 15 -> 10 -> 10),
compiled from a small graph where every node has a human name: "count of hearts," "rank 7
present," "five consecutive ranks." The weights were frozen and hashed before a single label was
read. The one allowed revision pass found 0 errors and was never used.

**3/6**
Trained baselines (MLP, XGBoost, LightGBM), same encoding, same dataset, generous tuning, up to
25,010 labels: none matched it. Closest was a 111k-param MLP at 99.90% accuracy, 0.90 macro-F1.
XGBoost/LightGBM hit 99.85-99.87% accuracy but only 0.70-0.74 macro-F1 - accuracy hides how badly
they miss the rare hands. Full table: [repo]/RESULTS.md.

**4/6**
Even giving a baseline 500,000 labels (20x the whole training set) didn't close the gap - it
still missed straight flushes and royal flushes outright, just from how few examples of those
exist anywhere. The hand-written net gets them right every time, by construction, regardless of
how rare they are. Weight heatmaps, one per layer, labeled by neuron meaning: [repo]/figures/.

**5/6**
Built a self-contained demo: pick 5 cards, see the predicted hand and the live activation of
every named neuron, forward pass running in plain JS, checked bit-identical to the Python/numpy
version. No server, no build step, opens from a plain file. Try it: [demo]

**6/6**
Honest caveat, stated up front, not buried: this is a correct, checkable, zero-branching compiled
rule set, not a model that learned poker from data. It didn't discover that a flush beats a
straight; it was told. That's a fair objection and the README doesn't dodge it - the actual
finding is how far trained models still are from a known-correct ceiling, measured exactly
instead of asserted. [repo]/README.md has the full argument, including where this approach would
*not* work (any task whose rules aren't already fully known).
