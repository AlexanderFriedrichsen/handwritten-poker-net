#!/usr/bin/env python3
"""Phase 2 label-definition audit (integrity ruling). This is the FIRST
script in the project allowed to read a label, and only after the weights
are frozen (see DECISIONS.md). Reads every row of the training file exactly
once, and checks both the frozen net's predictions and the independent
oracle's classifications against UCI's own label column - looking
specifically for a systematic convention mismatch (e.g. ace-low straights,
royal vs. straight flush) that the zero-label design might have gotten
wrong despite matching UCI's own aggregate documentation.

Writes results/label_audit.json. Any mismatch found here is the trigger
for the one allowed Phase 2 revision (see DECISIONS.md); if none is found,
v1 and v2 are identical and that is reported plainly, not silently dropped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from poker_handnet import encoding, oracle  # noqa: E402
from poker_handnet.dag import CompiledNet  # noqa: E402

TRAIN_PATH = REPO_ROOT / "data" / "poker-hand-training-true.data"
WEIGHTS_PATH = REPO_ROOT / "weights" / "handnet_v1.npz"


def examples(idxs, raw, uci_label, net_pred, oracle_pred, k=10):
    out = []
    for i in idxs[:k]:
        out.append(
            {
                "row": int(i),
                "raw": raw[i].astype(int).tolist(),
                "uci_label": int(uci_label[i]),
                "net_pred": int(net_pred[i]),
                "oracle_pred": int(oracle_pred[i]),
            }
        )
    return out


def main() -> None:
    rows = np.loadtxt(TRAIN_PATH, delimiter=",", dtype=np.int64)
    n_rows = rows.shape[0]
    assert rows.shape[1] == 11, f"expected 11 columns, got {rows.shape[1]}"

    raw = rows[:, :10].astype(np.float64)
    uci_label = rows[:, 10]

    net = CompiledNet.from_npz(str(WEIGHTS_PATH))
    enc = encoding.encode(raw)
    logits = net.forward_numpy(enc)
    net_pred = np.argmax(logits, axis=1)

    oracle_pred = np.array(oracle.classify_batch(raw.astype(int)))

    net_mismatches = np.flatnonzero(net_pred != uci_label)
    oracle_mismatches = np.flatnonzero(oracle_pred != uci_label)
    net_oracle_mismatches = np.flatnonzero(net_pred != oracle_pred)

    print(f"rows read: {n_rows}")
    print(f"net vs UCI label mismatches: {len(net_mismatches)}")
    print(f"oracle vs UCI label mismatches: {len(oracle_mismatches)}")
    print(f"net vs oracle mismatches (proven 0 over the full domain): {len(net_oracle_mismatches)}")

    result = {
        "rows_read": int(n_rows),
        "net_vs_uci_mismatches": int(len(net_mismatches)),
        "oracle_vs_uci_mismatches": int(len(oracle_mismatches)),
        "net_vs_oracle_mismatches": int(len(net_oracle_mismatches)),
        "net_vs_uci_examples": examples(net_mismatches, raw, uci_label, net_pred, oracle_pred),
    }
    out_path = REPO_ROOT / "results" / "label_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
