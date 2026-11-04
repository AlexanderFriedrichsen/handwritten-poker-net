"""Shared evaluation metrics so every model (hand-written net and every
trained baseline) is scored the same way, on the same encoded input, for a
fair comparison.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import f1_score

N_CLASSES = 10


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = _confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    accuracy = float(np.trace(cm) / cm.sum())
    macro_f1 = float(f1_score(y_true, y_pred, labels=list(range(N_CLASSES)), average="macro", zero_division=0))
    per_class_support = cm.sum(axis=1)
    per_class_recall = [
        float(cm[i, i] / per_class_support[i]) if per_class_support[i] > 0 else None
        for i in range(N_CLASSES)
    ]
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "per_class_support": per_class_support.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def mlp_param_counts(sequential) -> dict:
    """Count non-zero weight/bias entries in a torch.nn.Sequential of Linears."""
    import torch.nn as nn

    nonzero_weights = 0
    nonzero_biases = 0
    total_weights = 0
    total_biases = 0
    for layer in sequential:
        if isinstance(layer, nn.Linear):
            w = layer.weight.detach().cpu().numpy()
            nonzero_weights += int(np.count_nonzero(w))
            total_weights += w.size
            if layer.bias is not None:
                b = layer.bias.detach().cpu().numpy()
                nonzero_biases += int(np.count_nonzero(b))
                total_biases += b.size
    return {
        "nonzero_weights": nonzero_weights,
        "nonzero_biases": nonzero_biases,
        "nonzero_params_total": nonzero_weights + nonzero_biases,
        "total_weights": total_weights,
        "total_biases": total_biases,
        "total_params": total_weights + total_biases,
    }


def xgboost_param_counts(booster) -> dict:
    """Leaf count (the closest tree analog to a NN's learned scalar weights)
    and total node count, summed across every tree in the ensemble (one set
    of trees per class, for multiclass softmax objective)."""
    df = booster.trees_to_dataframe()
    n_leaves = int((df["Feature"] == "Leaf").sum())
    n_nodes = int(len(df))
    n_trees = int(df["Tree"].nunique())
    return {
        "nonzero_weights": n_leaves,
        "nonzero_biases": 0,
        "nonzero_params_total": n_leaves,
        "total_nodes": n_nodes,
        "n_trees": n_trees,
    }


def lightgbm_param_counts(booster) -> dict:
    df = booster.trees_to_dataframe()
    n_leaves = int(df["node_index"].str.contains("L").sum())
    n_nodes = int(len(df))
    n_trees = int(df["tree_index"].nunique())
    return {
        "nonzero_weights": n_leaves,
        "nonzero_biases": 0,
        "nonzero_params_total": n_leaves,
        "total_nodes": n_nodes,
        "n_trees": n_trees,
    }
