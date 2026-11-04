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


def sample_budget(n_pool: int, budget: int, seed: int) -> np.ndarray:
    budget = min(budget, n_pool)
    rng = np.random.default_rng(1_000_003 + seed * 7919 + budget)
    return rng.choice(n_pool, size=budget, replace=False)


def fit_xgb(X, y, Xd, yd, seed, max_depth, lr):
    """Native Booster API, not the sklearn wrapper: num_class is fixed at 10
    as a model hyperparameter regardless of which classes happen to appear
    in a given label budget. This matters because the rarest classes
    (6-9) are so thin that small budgets routinely contain zero examples
    of class 8 or 9 - the sklearn wrapper's `XGBClassifier` infers the
    class set from `y` and hard-errors on exactly this case; the native
    API just leaves that class's logit at baseline, which is the honest
    and correct behavior to measure (see DECISIONS.md / RESULTS.md)."""
    dtrain = xgb.DMatrix(X, label=y)
    ddev = xgb.DMatrix(Xd, label=yd)
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
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(ddev, "dev")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return booster


def fit_lgb(X, y, Xd, yd, seed, max_depth, lr):
    """Native Booster API for the same reason as fit_xgb above."""
    dtrain = lgb.Dataset(X, label=y)
    ddev = lgb.Dataset(Xd, label=yd, reference=dtrain)
    params = {
        "max_depth": max_depth,
        "learning_rate": lr,
        "objective": "multiclass",
        "num_class": 10,
        "seed": seed,
        "verbosity": -1,
        "num_threads": -1,
    }
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[ddev],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return booster


def predict_xgb(booster, X):
    return booster.predict(xgb.DMatrix(X)).astype(int)


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
                model = fit_fn(X_tune, y_tune, Xd, yd, 0, max_depth, lr)
                pred = predict_fns[lib](model, Xd)
                acc = float((pred == yd).mean())
                if best is None or acc > best[2]:
                    best = (max_depth, lr, acc)
        results["tree"][lib] = {"max_depth": best[0], "learning_rate": best[1]}
        print(f"  tuned {lib}: max_depth={best[0]} lr={best[1]} dev_acc={best[2]:.4f}")

    return results


def eval_row(family, variant, budget, seed, encoding_name, pred, y_test, param_counts, fit_seconds):
    report = metrics.classification_report_dict(y_test, pred)
    return {
        "family": family,
        "variant": variant,
        "budget": budget,
        "seed": seed,
        "encoding": encoding_name,
        "param_counts": param_counts,
        "fit_seconds": fit_seconds,
        **report,
    }


def main() -> None:
    smoke = "--smoke" in sys.argv
    budgets = [50, 200] if smoke else [512, 5000, 25010]
    seeds = [0, 1] if smoke else [0, 1, 2]

    t_start = time.time()
    raw_train, y_train_full = load_data_file(TRAIN_PATH)
    raw_test, y_test = load_data_file(TEST_PATH)
    n_train = raw_train.shape[0]
    print(f"train rows: {n_train}, test rows: {raw_test.shape[0]}")

    dev_mask = dev_split_mask(n_train)
    train_pool_idx = np.flatnonzero(~dev_mask)

    enc_train = encoding.encode(raw_train)
    enc_test = encoding.encode(raw_test)
    raw10_train = encoding.encode_raw_passthrough(raw_train)
    raw10_test = encoding.encode_raw_passthrough(raw_test)

    X_dev, y_dev = enc_train[dev_mask], y_train_full[dev_mask]

    tune_idx = sample_budget(len(train_pool_idx), 5000, seed=0)
    tune_idx = train_pool_idx[tune_idx]
    print("tuning hyperparameters on a 5,000-row dev-adjacent sample...")
    hparams = tune_hparams(enc_train[tune_idx], y_train_full[tune_idx], X_dev, y_dev)

    rows = []

    for budget in budgets:
        for seed in seeds:
            idx = sample_budget(n_train, budget, seed)
            Xb, yb = enc_train[idx], y_train_full[idx]
            print(f"budget={budget} seed={seed} (n={len(idx)})")

            for size in ("mlp_small", "mlp_large"):
                t0 = time.time()
                model, hist = mlp.train_mlp(
                    Xb, yb, X_dev, y_dev, size, seed=seed, device=DEVICE, lr=hparams["mlp_lr"][size]
                )
                pred = mlp.predict(model, enc_test, device=DEVICE)
                dt = time.time() - t0
                counts = metrics.mlp_param_counts(model)
                rows.append(eval_row(size, "encoded17", budget, seed, "encoded17", pred, y_test, counts, dt))
                print(f"  {size}: acc={rows[-1]['accuracy']:.4f} epochs={hist['epochs_run']} ({dt:.1f}s)")

            for lib, fit_fn, predict_fn, count_fn in (
                ("xgboost", fit_xgb, predict_xgb, metrics.xgboost_param_counts),
                ("lightgbm", fit_lgb, predict_lgb, metrics.lightgbm_param_counts),
            ):
                t0 = time.time()
                hp = hparams["tree"][lib]
                booster = fit_fn(Xb, yb, X_dev, y_dev, seed, hp["max_depth"], hp["learning_rate"])
                pred = predict_fn(booster, enc_test)
                dt = time.time() - t0
                counts = count_fn(booster)
                rows.append(eval_row(lib, "encoded17", budget, seed, "encoded17", pred, y_test, counts, dt))
                print(f"  {lib}: acc={rows[-1]['accuracy']:.4f} ({dt:.1f}s)")

    print("raw-10-feature bonus row (mlp_large, full training budget, seed 0)...")
    idx_full = sample_budget(n_train, budgets[-1], 0)
    Xd10, yd10 = raw10_train[dev_mask], y_train_full[dev_mask]
    t0 = time.time()
    model, hist = mlp.train_mlp(
        raw10_train[idx_full], y_train_full[idx_full], Xd10, yd10, "mlp_large", seed=0, device=DEVICE,
        lr=hparams["mlp_lr"]["mlp_large"],
    )
    pred = mlp.predict(model, raw10_test, device=DEVICE)
    dt = time.time() - t0
    counts = metrics.mlp_param_counts(model)
    rows.append(eval_row("mlp_large", "raw10_bonus", budgets[-1], 0, "raw10", pred, y_test, counts, dt))
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
                "budget": len(big_train_idx),
                "seed": 0,
                "encoding": "encoded17",
                "param_counts": counts,
                "fit_seconds": dt,
                "note": "trains on half of the test file, evaluates on the other half - "
                "NOT part of the fair label-budget comparison, since it uses test-file "
                "labels for training by construction; reported only to show the "
                "asymptotic ceiling a trained baseline reaches with far more labels "
                "than the 25,010-row training file provides.",
                **report,
            }
        )
        print(f"  mlp_large/500k-slice: acc={rows[-1]['accuracy']:.4f} ({dt:.1f}s)")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"hparams": hparams, "rows": rows}, indent=2) + "\n")
    print(f"wrote {RESULTS_PATH}")
    print(f"total wall time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
