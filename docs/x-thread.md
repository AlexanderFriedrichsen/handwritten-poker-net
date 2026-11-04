# Draft X thread (not posted)

Draft only. X Premium long-post format: 4 posts, numbered for order, each meant to stand on its
own. Written to be honest first and punchy second - trim for length before posting, but do not
cut any caveat to make it shorter. The repo is public and pushed
(https://github.com/AlexanderFriedrichsen/handwritten-poker-net); nothing in this document has
been posted to X yet.

---

**1/4**
I had Claude write a neural network by hand from the rules of poker: all 393 non-zero
parameters derived by its own reasoning about poker hand rankings, zero training labels,
frozen and hashed before a single label was read anywhere in the project. Pure matmul + ReLU,
no if-statements at inference time. I benchmarked it against trained models (MLP, XGBoost,
LightGBM) on the UCI Poker Hand dataset (1,000,000 test hands), same input encoding, same task.

The hand-written net is exact on every one of the 10 hand classes, every time. The trained
models are not: at the full training budget - every label in the 25,010-row training file,
nothing held back for early stopping - they land at 32.6%, 39.7%, and 70.3% mean recall on the
four rarest hand types (full house through royal flush) for XGBoost, LightGBM, and the best MLP
respectively. Even a version given 500,000 labels, 20x the entire training file, still missed
every held-out straight flush and royal flush in its evaluation slice - all misread as plain
"flush."

**2/4**
The honest objection: this isn't learning, it's a compiled rule set. Claude derived the logic -
"count of hearts," "rank 7 present," "five consecutive ranks starting at rank 10" - as a small
named graph, which a poker-agnostic compiler turns into a real 8-layer feedforward network
(17 -> 56 -> 53 -> 14 -> 9 -> 11 -> 15 -> 10 -> 10). It didn't discover from data that a flush
beats a straight; it was told, and the telling sits in a weight matrix instead of an
if-statement. That's a fair objection and this project doesn't dodge it anywhere.

This exists because of a viral post claiming a model had hand-written the weights of a
legal-citation classifier - a messier, less-than-fully-specified domain. Poker hand ranking is
the opposite: a closed-form, fully known function of 5 cards, with no hidden structure a
learned model could find that the rules don't already capture. That's exactly why "just compile
the rules" is possible here and is not a general recipe - it would not work on any task whose
rules are not already fully known. What's actually being measured isn't "rules beat learning"
in the abstract; it's how far trained models land from a correct, checkable ceiling, on
identical footing, when that ceiling is free because the domain is fully known.

**3/4**
Full results (small disclosed hyperparameter grid, 3 seeds, identical 17-dim encoding for every
model, full writeup and raw numbers in the repo's RESULTS.md):

  Hand-written net      100.0000% acc   1.0000 macro-F1   393 params
  MLP (111k params)      99.89%  acc   0.9084 macro-F1
  XGBoost                99.85%  acc   0.7389 macro-F1
  LightGBM               99.86%  acc   0.7354 macro-F1

No baseline matches or beats the hand-written net on any metric, at any label budget, anywhere
in a 60-row sweep - including a class-balanced reweighting variant, which helps at tiny budgets
but costs macro-F1 at the full one. Accuracy is the misleading column here: XGBoost and
LightGBM both score 0% recall on straight_flush at the full budget, even with every label spent
on training and none held out, despite 12 real straight_flush hands in the 1,000,000-row test
set. Royal flush (3 test examples, the single rarest class in the dataset, four times rarer
than straight_flush) scores 100% recall across all three trained families here - which argues
against "rare just means hard": some rare patterns generalize from nearby examples and some
don't, and a trained model's luck on the rarest classes depends on exactly which few examples
land in its splits. The hand-written net has no equivalent unpredictability - it is correct on
all 2,598,960 possible 5-card hands by construction, checked exhaustively, not by having seen
the right sample.

**4/4**
Also built: a self-contained browser demo, forward pass running in plain JS checked
bit-identical to the Python/numpy version, no server or build step needed - clone the repo and
open demo/index.html directly in a browser. Pick 5 cards, see the predicted hand and the live
activation of every named neuron.

Everything here - the weight construction, the poker-agnostic compiler, the baselines, the
figures, the demo - regenerates from scratch with one command (scripts/reproduce_all.sh). Repo,
full numbers, and the "just a compiled rule set" argument in full:
https://github.com/AlexanderFriedrichsen/handwritten-poker-net
