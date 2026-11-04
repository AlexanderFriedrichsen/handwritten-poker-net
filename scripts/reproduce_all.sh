#!/usr/bin/env bash
# One-command regeneration of every number, figure, and demo file in this
# repo from scratch (spec Phase 4: "a script that regenerates every number
# from scratch with one command"). Run from anywhere; paths are resolved
# relative to this script, not the caller's cwd.
#
# Wall time: ~20 minutes, dominated by scripts/run_baselines.py (~18 min for
# the full 60-row sweep). Set SMOKE=1 to pass --smoke to that script instead
# (tiny budgets/seeds, seconds instead of minutes) for a fast correctness
# check of the pipeline itself, not a real results run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_BASELINES_ARGS=()
if [[ "${SMOKE:-0} " == "1 " ]]; then
  RUN_BASELINES_ARGS+=(--smoke)
  echo "SMOKE=1: running scripts/run_baselines.py --smoke (not a real results run)"
fi

step() { echo; echo "=== $* ==="; }

step "1/9 data: download UCI Poker Hand (id 158) if not already present"
EXPECTED_DATA_FILES=(
  data/poker-hand-training-true.data
  data/poker-hand-testing.data
  data/poker-hand.names
)
if [[ -f "${EXPECTED_DATA_FILES[0]}" && -f "${EXPECTED_DATA_FILES[1]}" && -f "${EXPECTED_DATA_FILES[2]}" ]]; then
  echo "data/*.data already present, skipping download (delete data/*.data to force a re-fetch)"
else
  uv run python data/download.py
fi

step "2/9 weights: construct and freeze the hand-written net (reads no label)"
uv run python scripts/build_weights.py

step "3/9 audit: verify UCI's label conventions against the net and an independent oracle"
uv run python scripts/audit_labels.py

step "4/9 eval: final test-set evaluation of the frozen net (the one read of test labels)"
uv run python scripts/eval_test_set.py

step "5/9 baselines: trained MLP/XGBoost/LightGBM sweep across label budgets and seeds"
uv run python scripts/run_baselines.py "${RUN_BASELINES_ARGS[@]}"

step "6/9 figures: weight heatmaps and neuron reference"
uv run python scripts/plot_weights.py

step "7/9 figures: accuracy/macro-F1 vs. labels-seen plot"
if [[ "${SMOKE:-0} " == "1 " ]]; then
  echo "SMOKE=1: skipping - this plot is defined over the real label budgets" \
       "(512/5000/25010) and the 500k-slice bonus row, neither of which" \
       "--smoke's tiny budgets produce; nothing meaningful to plot here."
else
  uv run python scripts/plot_results.py
fi

step "8/9 demo: build the self-contained browser demo"
uv run python scripts/build_demo.py

step "9/9 tests: full suite"
uv run pytest

echo
echo "Done. Every number in RESULTS.md and every file in figures/ and demo/ was just regenerated from scratch."
