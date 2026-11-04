#!/usr/bin/env python3
"""Phase 4 plot: accuracy and macro-F1 vs. training budget for every trained
baseline (mean +/- std across seeds, log-x), with the frozen hand-written
net drawn as a horizontal reference line at its one point: 100% test
accuracy / 1.0 macro-F1 after seeing zero labels. The class-balanced variant
(mlp_large, xgboost) is drawn as a dashed, hollow-marker line next to its
unweighted counterpart. Also marks the two bonus rows (raw10 encoding; the
500k-test-slice exploratory run) as distinct, clearly-labeled points - they
are not part of the fair label-budget comparison (see DECISIONS.md) and
must not be mistaken for it on sight. The full_budget_no_early_stop rows are
not plotted here (they are single points per family, best read as a table -
see RESULTS.md) but are in results/baselines.json like everything else.

Reads results/baselines.json and results/handnet_test_eval.json; writes
figures/accuracy_vs_labels.png. Writes no new numbers - every value here is
already in those two JSON files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

BASELINES_PATH = REPO_ROOT / "results" / "baselines.json"
HANDNET_EVAL_PATH = REPO_ROOT / "results" / "handnet_test_eval.json"
FIGURES_DIR = REPO_ROOT / "figures"

FAMILIES = ["mlp_small", "mlp_large", "xgboost", "lightgbm"]
# Mirrors run_baselines.py's BALANCED_FAMILIES - the families that have an
# inverse_frequency class-weighted variant to plot alongside the unweighted one.
BALANCED_FAMILIES = {"mlp_large", "xgboost"}
FAMILY_STYLE = {
    "mlp_small": dict(color="#1f77b4", marker="o", label="mlp_small"),
    "mlp_large": dict(color="#ff7f0e", marker="s", label="mlp_large"),
    "xgboost": dict(color="#2ca02c", marker="^", label="xgboost"),
    "lightgbm": dict(color="#d62728", marker="v", label="lightgbm"),
}


def rare_recall_avg(row: dict) -> float:
    vals = row["per_class_recall"][6:10]
    return sum(v if v is not None else 0.0 for v in vals) / len(vals)


def aggregate(rows: list[dict], budgets: list[int], class_weighting: str) -> dict:
    agg: dict[tuple[str, int], dict] = {}
    for fam in FAMILIES:
        for budget in budgets:
            matched = [
                r
                for r in rows
                if r["family"] == fam
                and r["budget"] == budget
                and r["variant"] == "encoded17"
                and r["class_weighting"] == class_weighting
            ]
            if not matched:
                continue
            accs = np.array([r["accuracy"] for r in matched])
            f1s = np.array([r["macro_f1"] for r in matched])
            rares = np.array([rare_recall_avg(r) for r in matched])
            params = np.array([r["param_counts"]["nonzero_params_total"] for r in matched])
            agg[(fam, budget)] = {
                "acc_mean": accs.mean(), "acc_std": accs.std(),
                "f1_mean": f1s.mean(), "f1_std": f1s.std(),
                "rare_mean": rares.mean(), "rare_std": rares.std(),
                "params_mean": params.mean(), "n_seeds": len(matched),
            }
    return agg


def find_bonus(rows: list[dict], variant: str) -> dict:
    matched = [r for r in rows if r["variant"] == variant]
    assert len(matched) == 1, f"expected exactly one {variant} row, found {len(matched)}"
    return matched[0]


def main() -> None:
    data = json.loads(BASELINES_PATH.read_text())
    rows = data["rows"]
    handnet = json.loads(HANDNET_EVAL_PATH.read_text())

    budgets = sorted(
        {r["budget"] for r in rows if r["family"] in FAMILIES and r["variant"] == "encoded17" and r["class_weighting"] == "none"}
    )
    agg = aggregate(rows, budgets, "none")
    agg_bal = aggregate(rows, budgets, "inverse_frequency")
    raw10 = find_bonus(rows, "raw10_bonus")
    slice500k = find_bonus(rows, "exploratory_500k_test_slice")

    fig, (ax_acc, ax_f1) = plt.subplots(1, 2, figsize=(13, 5.5))

    for fam in FAMILIES:
        style = FAMILY_STYLE[fam]
        acc_means = [agg[(fam, b)]["acc_mean"] for b in budgets]
        acc_stds = [agg[(fam, b)]["acc_std"] for b in budgets]
        f1_means = [agg[(fam, b)]["f1_mean"] for b in budgets]
        f1_stds = [agg[(fam, b)]["f1_std"] for b in budgets]
        ax_acc.errorbar(budgets, acc_means, yerr=acc_stds, capsize=4, **style)
        ax_f1.errorbar(budgets, f1_means, yerr=f1_stds, capsize=4, **style)

    for fam in sorted(BALANCED_FAMILIES):
        pts = [b for b in budgets if (fam, b) in agg_bal]
        if not pts:
            continue
        bal_style = dict(FAMILY_STYLE[fam])
        bal_style["label"] = f"{fam} (class-balanced)"
        bal_style["linestyle"] = "--"
        bal_style["markerfacecolor"] = "none"
        acc_means = [agg_bal[(fam, b)]["acc_mean"] for b in pts]
        acc_stds = [agg_bal[(fam, b)]["acc_std"] for b in pts]
        f1_means = [agg_bal[(fam, b)]["f1_mean"] for b in pts]
        f1_stds = [agg_bal[(fam, b)]["f1_std"] for b in pts]
        ax_acc.errorbar(pts, acc_means, yerr=acc_stds, capsize=4, **bal_style)
        ax_f1.errorbar(pts, f1_means, yerr=f1_stds, capsize=4, **bal_style)

    bonus_style = dict(color="#9467bd", s=90, zorder=5, edgecolor="black", linewidth=0.8)
    ax_acc.scatter([raw10["budget"]], [raw10["accuracy"]], marker="X", label="mlp_large, raw10 encoding (bonus)", **bonus_style)
    ax_f1.scatter([raw10["budget"]], [raw10["macro_f1"]], marker="X", label="mlp_large, raw10 encoding (bonus)", **bonus_style)
    ax_acc.scatter([slice500k["budget"]], [slice500k["accuracy"]], marker="*", s=160,
                   label="mlp_large, 500k test-slice (exploratory, not label-fair)",
                   color="#8c564b", zorder=5, edgecolor="black", linewidth=0.8)
    ax_f1.scatter([slice500k["budget"]], [slice500k["macro_f1"]], marker="*", s=160,
                  label="mlp_large, 500k test-slice (exploratory, not label-fair)",
                  color="#8c564b", zorder=5, edgecolor="black", linewidth=0.8)

    handnet_label = f"hand-written net (0 labels seen): acc={handnet['accuracy']:.4f}, macro-F1={handnet['macro_f1']:.4f}"
    for ax, key in ((ax_acc, "accuracy"), (ax_f1, "macro_f1")):
        ax.axhline(handnet[key], color="black", linestyle="--", linewidth=1.5, zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel("training budget, log scale (+ 5,002 fixed dev labels seen via early\nstopping on every fit - see RESULTS.md's labels-seen table)")
        ax.set_ylim(0, 1.03)
        ax.grid(alpha=0.3)

    ax_acc.set_ylabel("test accuracy (1,000,000-row test file)")
    ax_acc.set_title("Accuracy vs. labels seen")
    ax_f1.set_ylabel("macro-F1 (10 classes, equal weight)")
    ax_f1.set_title("Macro-F1 vs. labels seen")

    handles, labels = ax_acc.get_legend_handles_labels()
    handles = [plt.Line2D([], [], color="black", linestyle="--", linewidth=1.5)] + handles
    labels = [handnet_label] + labels
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.suptitle("Hand-written net vs. trained baselines (mean ± std over 3 seeds)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "accuracy_vs_labels.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
