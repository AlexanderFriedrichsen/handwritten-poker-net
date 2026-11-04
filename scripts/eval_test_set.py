#!/usr/bin/env python3
"""Final evaluation of the frozen (v1, Phase-2-confirmed) hand-written net
on the full 1,000,000-row test file. Per integrity ruling #1, this is the
ONLY read of the test file's labels for the hand-written net, and it is
read only for this final evaluation - never for design or tuning.

Writes results/handnet_test_eval.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from poker_handnet import encoding, metrics  # noqa: E402
from poker_handnet.data_io import load_data_file  # noqa: E402
from poker_handnet.dag import CompiledNet  # noqa: E402
from poker_handnet.handnet_spec import CLASS_NAMES  # noqa: E402

TEST_PATH = REPO_ROOT / "data" / "poker-hand-testing.data"
WEIGHTS_PATH = REPO_ROOT / "weights" / "handnet_v1.npz"


def main() -> None:
    t0 = time.time()
    raw, labels = load_data_file(TEST_PATH)
    n_rows = raw.shape[0]
    print(f"loaded {n_rows} test rows in {time.time() - t0:.1f}s")

    net = CompiledNet.from_npz(str(WEIGHTS_PATH))
    enc = encoding.encode(raw)
    t1 = time.time()
    logits = net.forward_numpy(enc)
    pred = np.argmax(logits, axis=1)
    print(f"forward pass on {n_rows} rows in {time.time() - t1:.2f}s")

    report = metrics.classification_report_dict(labels, pred)
    counts = net.param_counts()

    print(f"accuracy: {report['accuracy']:.6f}")
    print(f"macro_f1: {report['macro_f1']:.6f}")
    for i, name in enumerate(CLASS_NAMES):
        recall = report["per_class_recall"][i]
        support = report["per_class_support"][i]
        recall_str = "n/a" if recall is None else f"{recall:.6f}"
        print(f"  class {i} ({name}): recall={recall_str} support={support}")

    result = {
        "model": "handnet_v1",
        "weights_path": str(WEIGHTS_PATH.relative_to(REPO_ROOT)),
        "rows_evaluated": n_rows,
        "class_names": CLASS_NAMES,
        "param_counts": counts,
        **report,
    }
    out_path = REPO_ROOT / "results" / "handnet_test_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
