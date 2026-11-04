#!/usr/bin/env python3
"""Phase 3: trained baselines, benchmarked against the frozen hand-written
net at several label budgets and seeds, all on the identical 17-dim
card-order-invariant encoding (integrity ruling #4). Also fits one
raw-10-feature baseline row and one "lots of labels" exploratory row (see
DECISIONS.md for why that last one is reported separately from the fair
comparison).

Hyperparameters are tuned once via tune_hparams(), using the fixed-seed
dev split (ruling #2) on a representative 5,000-label sample, then locked
in and reused for every budget/seed fit below - a real but bounded search,
not a blind grid (see DECISIONS.md).

Every budget is sampled from the non-dev pool only (`train_pool_idx`): the
dev split is used for early stopping / model selection on *every* fit, not
just during tuning, so a budget sampled from the full training file would
let "training" rows overlap the rows used to decide when to stop training
on them (see DECISIONS.md's "baseline budget/dev overlap" entry - this was
a real bug in an earlier version of this script, found by review). Because
every fit's pipeline genuinely touches both the training budget and the dev
split, each row reports `budget` (train rows only), `dev_size` (dev rows
touched by early stopping / selection), and `labels_seen` (their sum) -
`labels_seen` is the number to compare against "25,010 training rows" or
any other everything-the-pipeline-saw figure.

At least mlp_large and one GBDT also get an inverse-frequency class-weighted
variant (`class_weighting="inverse_frequency"`) at every budget/seed, to
test whether class-imbalance handling narrows the macro-F1 / rare-class gap
(see DECISIONS.md).

Run with --smoke for a fast (~seconds) correctness check with tiny
budgets/seeds before trusting the full run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from poker_handnet import encoding, metrics, mlp  # noqa: E402
from poker_handnet.data_io import load_data_file  # noqa: E402
from poker_handnet.splits import dev_split_mask  # noqa: E402

TRAIN_PATH = REPO_ROOT / "data" / "poker-hand-training-true.data"
TEST_PATH = REPO_ROOT / "data" / "poker-hand-testing.data"
RESULTS_PATH = REPO_ROOT / "results" / "baselines.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# "At least mlp_large and one GBDT" (fixes-1.md item 2). Between xgboost and
# lightgbm there is no principled reason to prefer either as "the" GBDT for
# this extra variant; xgboost was picked arbitrarily. Logged per the "if
# anything is ambiguous, pick the more rigorous option and note it"
# instruction - this particular choice has no rigor axis (both are GBDTs
# scored identically elsewhere in this sweep), so it is just disclosed.
BALANCED_FAMILIES = {"mlp_large", "xgboost"}


def sample_budget(n_pool: int, budget: int, seed: int) -> np.ndarray:
    budget = min(budget, n_pool)
    rng = np.random.default_rng(1_000_003 + seed * 7919 + budget)
    return rng.choice(n_pool, size=budget, replace=False)


def compute_class_weights(y: np.ndarray, n_classes: int = metrics.N_CLASSES, max_ratio: float = 20.0) -> np.ndarray:
    """Inverse-frequency class weights computed from THIS fit's own in-budget
    label counts, never from the global/test distribution - that would leak
    information this fit's pipeline does not actually have access to. +1
    Laplace smoothing avoids division by zero: classes 7-9 are thin enough
    (the full 20,008-row non-dev pool has only 4 straight_flush and 3
    royal_flush rows total) that small budgets routinely contain zero
    examples of one or both - see DECISIONS.md. A class with zero in-budget
    examples gets weight 1/(0+1)=1 pre-normalization, same as any other
    absent class; no reweighting scheme can teach a model a class it never
    sees, which is a real, honestly-reported limit on what this variant can
    fix, not a bug in the weighting.

    Raw +1-smoothed inverse-frequency weights span a ~2,513:1 ratio between
    the most- and least-common in-budget class at the full (non-dev-pool)
    budget - measured directly from that pool's class counts
    ([10052, 8425, 964, 405, 76, 47, 27, 5, 4, 3]: (1/4) / (1/10053) -
    see DECISIONS.md. Left uncapped, that is large enough to destabilize
    both gradient descent and tree-split search. `max_ratio=20.0` is not
    derived from a tuning sweep - it is a disclosed judgment call (no
    "right" cap exists without deciding how much training-objective
    distortion is acceptable), picked as a round number well below the
    measured ~2,513:1 so the boost is real but bounded; logged in
    DECISIONS.md per the "pick the more rigorous option if ambiguous, and
    log it" instruction.
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    raw = 1.0 / (counts + 1.0)
    cap = max_ratio * raw.min()
    capped = np.minimum(raw, cap)
    return capped / capped.mean()


def fit_xgb(X, y, Xd=None, yd=None, seed=0, max_depth=3, lr=0.3, sample_weight=None, fixed_rounds=None):
    """Native Booster API, not the sklearn wrapper: num_class is fixed at 10
    as a model hyperparameter regardless of which classes happen to appear
    in a given label budget. This matters because the rarest classes
    (6-9) are so thin that small budgets routinely contain zero examples
    of class 8 or 9 - the sklearn wrapper's `XGBClassifier` infers the
    class set from `y` and hard-errors on exactly this case; the native
    API just leaves that class's logit at baseline, which is the honest
    and correct behavior to measure (see DECISIONS.md / RESULTS.md).

    Returns (booster, fit_iterations) - fit_iterations is the number of
    boosting rounds actually used at prediction time (see predict_xgb and
    metrics.xgboost_param_counts: a booster trained with early stopping
    keeps growing for `early_stopping_rounds` rounds past its best one
    before halting, and both predict() and trees_to_dataframe() include
    those "wasted" rounds unless explicitly told not to).

    sample_weight, when given, is a per-row weight (see
    compute_class_weights) passed straight to the training DMatrix; the dev
    DMatrix is never weighted, so early stopping always selects against the
    natural dev-label distribution regardless of whether training was
    class-weighted. fixed_rounds, when given, skips early stopping
    entirely and trains for exactly that many rounds with no dev set
    involved at all (used for the no-early-stop full-budget row).
    """
    dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)
    params = {
        "max_depth": max_depth,
        "eta": lr,
        "objective": "multi:softmax",
        "num_class": 10,
        "tree_method": "hist",
        "seed": seed,
        "eval_metric": "mlogloss",
        "nthread": -1,
    }
    if fixed_rounds is not None:
        booster = xgb.train(params, dtrain, num_boost_round=fixed_rounds, verbose_eval=False)
        return booster, fixed_rounds

    ddev = xgb.DMatrix(Xd, label=yd)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(ddev, "dev")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return booster, booster.best_iteration + 1


def fit_lgb(X, y, Xd=None, yd=None, seed=0, max_depth=3, lr=0.1, sample_weight=None, fixed_rounds=None):
    """Native Booster API for the same reason as fit_xgb above. Returns
    (booster, fit_iterations); see fit_xgb's docstring for sample_weight /
    fixed_rounds semantics. Unlike xgboost, a LightGBM booster's
    `best_iteration` attribute always exists but is only meaningful (set to
    something other than the 0 sentinel) when early stopping actually ran,
    so fit_iterations is computed here rather than read back from the
    booster by the caller."""
    dtrain = lgb.Dataset(X, label=y, weight=sample_weight)
    params = {
        "max_depth": max_depth,
        "learning_rate": lr,
        "objective": "multiclass",
        "num_class": 10,
        "seed": seed,
        "verbosity": -1,
        "num_threads": -1,
    }
    if fixed_rounds is not None:
        booster = lgb.train(params, dtrain, num_boost_round=fixed_rounds)
        return booster, fixed_rounds

    ddev = lgb.Dataset(Xd, label=yd, reference=dtrain)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[ddev],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return booster, booster.best_iteration


def predict_xgb(booster, X):
    """Limits prediction to `best_iteration + 1` rounds when the booster was
    trained with early stopping - verified empirically that xgboost's
    `Booster.predict()` does NOT do this on its own (it uses every round
    grown, including the early-stopping patience window past the best one;
    see DECISIONS.md). A booster with no `best_iteration` attribute at all
    (trained with `fixed_rounds`, i.e. no early stopping) has nothing to
    truncate."""
    dmat = xgb.DMatrix(X)
    if hasattr(booster, "best_iteration"):
        pred = booster.predict(dmat, iteration_range=(0, booster.best_iteration + 1))
    else:
        pred = booster.predict(dmat)
    return pred.astype(int)


def predict_lgb(booster, X):
    return np.argmax(booster.predict(X), axis=1)


def tune_hparams(X_tune: np.ndarray, y_tune: np.ndarray, Xd: np.ndarray, yd: np.ndarray) -> dict:
    results: dict = {"mlp_lr": {}, "tree": {}}

    for size in ("mlp_small", "mlp_large"):
        best = None
        for lr in (1e-2, 3e-3, 1e-3):
            _, hist = mlp.train_mlp(
                X_tune, y_tune, Xd, yd, size, seed=0, device=DEVICE, lr=lr, max_epochs=1000, patience=40
            )
            if best is None or hist["best_dev_loss"] < best[1]:
                best = (lr, hist["best_dev_loss"])
        results["mlp_lr"][size] = best[0]
        print(f"  tuned {size}: lr={best[0]} dev_loss={best[1]:.4f}")

    predict_fns = {"xgboost": predict_xgb, "lightgbm": predict_lgb}
    for lib, fit_fn in (("xgboost", fit_xgb), ("lightgbm", fit_lgb)):
        best = None
        for max_depth in (3, 6):
            for lr in (0.3, 0.1, 0.05):
                model, _ = fit_fn(X_tune, y_tune, Xd, yd, 0, max_depth, lr)
                pred = predict_fns[lib](model, Xd)
                acc = float((pred == yd).mean())
                if best is None or acc > best[2]:
                    best = (max_depth, lr, acc)
        results["tree"][lib] = {"max_depth": best[0], "learning_rate": best[1]}
        print(f"  tuned {lib}: max_depth={best[0]} lr={best[1]} dev_acc={best[2]:.4f}")

    return results


def eval_row(
    family,
    variant,
    class_weighting,
    budget,
    dev_size,
    seed,
    encoding_name,
    pred,
    y_test,
    param_counts,
    fit_iterations,
    early_stopped,
    fit_seconds,
):
    report = metrics.classification_report_dict(y_test, pred)
    return {
        "family": family,
        "variant": variant,
        "class_weighting": class_weighting,
        "budget": budget,
        "dev_size": dev_size,
        "labels_seen": budget + dev_size,
        "seed": seed,
        "encoding": encoding_name,
        "param_counts": param_counts,
        "fit_iterations": fit_iterations,
        "early_stopped": early_stopped,
        "fit_seconds": fit_seconds,
        **report,
    }


def main() -> None:
    smoke = "--smoke" in sys.argv
    seeds = [0, 1] if smoke else [0, 1, 2]

    t_start = time.time()
    raw_train, y_train_full = load_data_file(TRAIN_PATH)
    raw_test, y_test = load_data_file(TEST_PATH)
    n_train = raw_train.shape[0]
    print(f"train rows: {n_train}, test rows: {raw_test.shape[0]}")

    dev_mask = dev_split_mask(n_train)
    train_pool_idx = np.flatnonzero(~dev_mask)
    n_pool = len(train_pool_idx)
    n_dev = int(dev_mask.sum())
    print(f"non-dev pool: {n_pool} rows, dev split: {n_dev} rows (dev used for early stopping on every fit)")

    budgets = [50, 200] if smoke else [512, 5000, n_pool]

    enc_train = encoding.encode(raw_train)
    enc_test = encoding.encode(raw_test)
    raw10_train = encoding.encode_raw_passthrough(raw_train)
    raw10_test = encoding.encode_raw_passthrough(raw_test)

    X_dev, y_dev = enc_train[dev_mask], y_train_full[dev_mask]

    tune_idx = sample_budget(n_pool, 5000, seed=0)
    tune_idx = train_pool_idx[tune_idx]
    print("tuning hyperparameters on a 5,000-row dev-adjacent sample...")
    hparams = tune_hparams(enc_train[tune_idx], y_train_full[tune_idx], X_dev, y_dev)

    rows = []
    full_budget_iters: dict[str, list[int]] = {"mlp_small": [], "mlp_large": [], "xgboost": [], "lightgbm": []}

    for budget in budgets:
        for seed in seeds:
            idx = train_pool_idx[sample_budget(n_pool, budget, seed)]
            Xb, yb = enc_train[idx], y_train_full[idx]
            print(f"budget={budget} seed={seed} (n={len(idx)}, non-dev pool only)")

            for size in ("mlp_small", "mlp_large"):
                t0 = time.time()
                model, hist = mlp.train_mlp(
                    Xb, yb, X_dev, y_dev, size, seed=seed, device=DEVICE, lr=hparams["mlp_lr"][size]
                )
                pred = mlp.predict(model, enc_test, device=DEVICE)
                dt = time.time() - t0
                counts = metrics.mlp_param_counts(model)
                rows.append(
                    eval_row(
                        size, "encoded17", "none", budget, n_dev, seed, "encoded17", pred, y_test, counts,
                        hist["epochs_run"], True, dt,
                    )
                )
                print(f"  {size}: acc={rows[-1]['accuracy']:.4f} epochs={hist['epochs_run']} ({dt:.1f}s)")
                if budget == n_pool:
                    full_budget_iters[size].append(hist["epochs_run"])

                if size in BALANCED_FAMILIES:
                    t0 = time.time()
                    cw = compute_class_weights(yb)
                    model_b, hist_b = mlp.train_mlp(
                        Xb, yb, X_dev, y_dev, size, seed=seed, device=DEVICE, lr=hparams["mlp_lr"][size],
                        class_weights=cw,
                    )
                    pred_b = mlp.predict(model_b, enc_test, device=DEVICE)
                    dt_b = time.time() - t0
                    counts_b = metrics.mlp_param_counts(model_b)
                    rows.append(
                        eval_row(
                            size, "encoded17", "inverse_frequency", budget, n_dev, seed, "encoded17", pred_b,
                            y_test, counts_b, hist_b["epochs_run"], True, dt_b,
                        )
                    )
                    print(f"  {size}/balanced: acc={rows[-1]['accuracy']:.4f} epochs={hist_b['epochs_run']} ({dt_b:.1f}s)")

            for lib, fit_fn, predict_fn, count_fn in (
                ("xgboost", fit_xgb, predict_xgb, metrics.xgboost_param_counts),
                ("lightgbm", fit_lgb, predict_lgb, metrics.lightgbm_param_counts),
            ):
                t0 = time.time()
                hp = hparams["tree"][lib]
                booster, fit_iters = fit_fn(Xb, yb, X_dev, y_dev, seed, hp["max_depth"], hp["learning_rate"])
                pred = predict_fn(booster, enc_test)
                dt = time.time() - t0
                counts = count_fn(booster)
                rows.append(
                    eval_row(
                        lib, "encoded17", "none", budget, n_dev, seed, "encoded17", pred, y_test, counts,
                        fit_iters, True, dt,
                    )
                )
                print(f"  {lib}: acc={rows[-1]['accuracy']:.4f} ({dt:.1f}s)")
                if budget == n_pool:
                    full_budget_iters[lib].append(fit_iters)

                if lib in BALANCED_FAMILIES:
                    t0 = time.time()
                    cw = compute_class_weights(yb)
                    sw = cw[yb]
                    booster_b, fit_iters_b = fit_fn(
                        Xb, yb, X_dev, y_dev, seed, hp["max_depth"], hp["learning_rate"], sample_weight=sw
                    )
                    pred_b = predict_fn(booster_b, enc_test)
                    dt_b = time.time() - t0
                    counts_b = count_fn(booster_b)
                    rows.append(
                        eval_row(
                            lib, "encoded17", "inverse_frequency", budget, n_dev, seed, "encoded17", pred_b,
                            y_test, counts_b, fit_iters_b, True, dt_b,
                        )
                    )
                    print(f"  {lib}/balanced: acc={rows[-1]['accuracy']:.4f} ({dt_b:.1f}s)")

    if not smoke:
        print("full-budget, no-early-stopping rows (all training rows, fixed iteration count, no dev set touched)...")
        for fam in ("mlp_small", "mlp_large"):
            fixed_epochs = int(round(float(np.mean(full_budget_iters[fam]))))
            t0 = time.time()
            model, hist = mlp.train_mlp(
                enc_train, y_train_full, None, None, fam, seed=0, device=DEVICE, lr=hparams["mlp_lr"][fam],
                max_epochs=fixed_epochs,
            )
            pred = mlp.predict(model, enc_test, device=DEVICE)
            dt = time.time() - t0
            counts = metrics.mlp_param_counts(model)
            rows.append(
                eval_row(
                    fam, "full_budget_no_early_stop", "none", n_train, 0, 0, "encoded17", pred, y_test, counts,
                    hist["epochs_run"], False, dt,
                )
            )
            print(f"  {fam}/no-early-stop: acc={rows[-1]['accuracy']:.4f} fixed_epochs={fixed_epochs} ({dt:.1f}s)")

        for lib, fit_fn, predict_fn, count_fn in (
            ("xgboost", fit_xgb, predict_xgb, metrics.xgboost_param_counts),
            ("lightgbm", fit_lgb, predict_lgb, metrics.lightgbm_param_counts),
        ):
            hp = hparams["tree"][lib]
            fixed_rounds = int(round(float(np.mean(full_budget_iters[lib]))))
            t0 = time.time()
            booster, fit_iters = fit_fn(
                enc_train, y_train_full, seed=0, max_depth=hp["max_depth"], lr=hp["learning_rate"],
                fixed_rounds=fixed_rounds,
            )
            pred = predict_fn(booster, enc_test)
            dt = time.time() - t0
            counts = count_fn(booster)
            rows.append(
                eval_row(
                    lib, "full_budget_no_early_stop", "none", n_train, 0, 0, "encoded17", pred, y_test, counts,
                    fit_iters, False, dt,
                )
            )
            print(f"  {lib}/no-early-stop: acc={rows[-1]['accuracy']:.4f} fixed_rounds={fixed_rounds} ({dt:.1f}s)")

    print("raw-10-feature bonus row (mlp_large, full non-dev training pool, seed 0)...")
    idx_full = train_pool_idx  # every non-dev row - the same pool the main sweep's top budget draws from
    Xd10, yd10 = raw10_train[dev_mask], y_train_full[dev_mask]
    t0 = time.time()
    model, hist = mlp.train_mlp(
        raw10_train[idx_full], y_train_full[idx_full], Xd10, yd10, "mlp_large", seed=0, device=DEVICE,
        lr=hparams["mlp_lr"]["mlp_large"],
    )
    pred = mlp.predict(model, raw10_test, device=DEVICE)
    dt = time.time() - t0
    counts = metrics.mlp_param_counts(model)
    rows.append(
        eval_row(
            "mlp_large", "raw10_bonus", "none", len(idx_full), n_dev, 0, "raw10", pred, y_test, counts,
            hist["epochs_run"], True, dt,
        )
    )
    print(f"  mlp_large/raw10: acc={rows[-1]['accuracy']:.4f} ({dt:.1f}s)")

    if not smoke:
        print("exploratory 500k-slice-of-test bonus row (mlp_large)...")
        n_test = raw_test.shape[0]
        rng = np.random.default_rng(42)
        perm = rng.permutation(n_test)
        half = n_test // 2
        big_train_idx, big_eval_idx = perm[:half], perm[half:]
        internal_dev = big_train_idx[: half // 10]
        internal_fit = big_train_idx[half // 10 :]
        rare_detail = {
            "straight_flush_count": {
                "internal_fit": int((y_test[internal_fit] == 8).sum()),
                "internal_dev": int((y_test[internal_dev] == 8).sum()),
                "big_eval_idx": int((y_test[big_eval_idx] == 8).sum()),
            },
            "royal_flush_count": {
                "internal_fit": int((y_test[internal_fit] == 9).sum()),
                "internal_dev": int((y_test[internal_dev] == 9).sum()),
                "big_eval_idx": int((y_test[big_eval_idx] == 9).sum()),
            },
        }
        t0 = time.time()
        model, hist = mlp.train_mlp(
            enc_test[internal_fit], y_test[internal_fit], enc_test[internal_dev], y_test[internal_dev],
            "mlp_large", seed=0, device=DEVICE, lr=hparams["mlp_lr"]["mlp_large"], max_epochs=200, patience=15,
        )
        pred = mlp.predict(model, enc_test[big_eval_idx], device=DEVICE)
        dt = time.time() - t0
        counts = metrics.mlp_param_counts(model)
        report = metrics.classification_report_dict(y_test[big_eval_idx], pred)
        rows.append(
            {
                "family": "mlp_large",
                "variant": "exploratory_500k_test_slice",
                "class_weighting": "none",
                "budget": len(internal_fit),
                "dev_size": len(internal_dev),
                "labels_seen": len(internal_fit) + len(internal_dev),
                "seed": 0,
                "encoding": "encoded17",
                "param_counts": counts,
                "fit_iterations": hist["epochs_run"],
                "early_stopped": True,
                "fit_seconds": dt,
                "rare_class_split_detail": rare_detail,
                "note": "trains on 90% of one random half of the test file (internal_fit), "
                "early-stops on the other 10% of that same half (internal_dev), evaluates on "
                "the other half of the test file entirely (big_eval_idx, disjoint from both) - "
                "NOT part of the fair label-budget comparison, since it uses test-file labels "
                "for training by construction; reported only to show the asymptotic ceiling a "
                "trained baseline reaches with far more labels than the 25,010-row training "
                "file provides. See rare_class_split_detail for exact straight_flush/"
                "royal_flush counts in each of the three disjoint sets.",
                **report,
            }
        )
        print(f"  mlp_large/500k-slice: acc={rows[-1]['accuracy']:.4f} ({dt:.1f}s)")
        print(f"  rare-class split detail: {rare_detail}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_notes": {
            "budget": "training rows actually fit on (gradient steps / tree splits), always drawn "
            "from the non-dev pool for variant=='encoded17'/'raw10_bonus' rows",
            "dev_size": "rows used for early stopping / best-checkpoint selection in this fit "
            "(0 for variant=='full_budget_no_early_stop', which touches no dev rows at all)",
            "labels_seen": "budget + dev_size: every label this fit's pipeline actually read",
            "class_weighting": "'none' or 'inverse_frequency' (see compute_class_weights)",
            "fit_iterations": "epochs (MLP) or boosting rounds (xgboost/lightgbm) actually used "
            "at prediction time",
            "early_stopped": "whether fit_iterations was chosen by dev-loss early stopping "
            "(True) or fixed in advance (False, full_budget_no_early_stop rows only)",
        },
        "n_train": n_train,
        "n_dev": n_dev,
        "n_pool": n_pool,
        "hparams": hparams,
        "rows": rows,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {RESULTS_PATH}")
    print(f"total wall time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
