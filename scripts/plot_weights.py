#!/usr/bin/env python3
"""Interpretability artifact: one static heatmap per layer of the frozen
hand-written net, axes labeled with each neuron's actual name (and its doc
string on hover-equivalent, in an accompanying text sidecar), plus a
nonzero-param-per-layer bar chart. Reads no label data - purely a view of
weights/handnet_v1.npz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from poker_handnet.dag import CompiledNet  # noqa: E402

WEIGHTS_PATH = REPO_ROOT / "weights" / "handnet_v1.npz"
FIGURES_DIR = REPO_ROOT / "figures"


def plot_layer(layer, idx: int) -> Path:
    W = layer.W
    out_dim, in_dim = W.shape
    fig_w = max(6.0, in_dim * 0.28)
    fig_h = max(4.0, out_dim * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = np.max(np.abs(W)) or 1.0
    im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(in_dim))
    ax.set_xticklabels(layer.in_names, rotation=90, fontsize=6)
    ax.set_yticks(range(out_dim))
    ax.set_yticklabels(layer.out_names, fontsize=6)
    ax.set_xlabel("input neuron (previous layer)")
    ax.set_ylabel(f"output neuron ({layer.activation})")
    ax.set_title(f"Layer {idx}: {in_dim} -> {out_dim} ({layer.activation})\nweight value")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    out_path = FIGURES_DIR / f"weights_layer_{idx}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_param_counts(net: CompiledNet) -> Path:
    counts = net.param_counts()
    per_layer = counts["per_layer"]
    xs = [p["layer"] for p in per_layer]
    nw = [p["nonzero_weights"] for p in per_layer]
    nb = [p["nonzero_biases"] for p in per_layer]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(xs, nw, label="nonzero weights")
    ax.bar(xs, nb, bottom=nw, label="nonzero biases")
    ax.set_xlabel("layer")
    ax.set_ylabel("count")
    ax.set_title(
        f"Non-zero params per layer (total {counts['nonzero_params_total']}: "
        f"{counts['nonzero_weights']} weights + {counts['nonzero_biases']} biases)"
    )
    ax.legend()
    fig.tight_layout()

    out_path = FIGURES_DIR / "param_counts.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_neuron_docs(net: CompiledNet) -> Path:
    lines = ["# Neuron reference (every node's name -> human-readable meaning)", ""]
    for i, layer in enumerate(net.layers):
        lines.append(f"## Layer {i} outputs ({layer.activation})")
        for name, doc, group in zip(layer.out_names, layer.out_docs, layer.out_groups):
            lines.append(f"- `{name}` [{group}]: {doc}")
        lines.append("")
    out_path = FIGURES_DIR / "neuron_reference.md"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    net = CompiledNet.from_npz(str(WEIGHTS_PATH))

    paths = [plot_layer(layer, i) for i, layer in enumerate(net.layers)]
    paths.append(plot_param_counts(net))
    paths.append(write_neuron_docs(net))

    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
