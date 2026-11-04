#!/usr/bin/env python3
"""Compile the hand-derived poker graph (handnet_spec.py) and freeze it to
weights/handnet_v1.npz. This is THE weight-freeze script: per the project's
integrity ruling, its first run (and the resulting weights file) must be
committed strictly before any label is read. Prints the weights file's
sha256 so it can be recorded in DECISIONS.md and the freezing commit
message.

Deliberately reads no data file - it only calls into handnet_spec.py, which
is itself derived from poker rules + UCI's documentation (not row data).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from poker_handnet.handnet_spec import build_handnet  # noqa: E402


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out_path = REPO_ROOT / "weights" / "handnet_v1.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    net = build_handnet()
    net.to_npz(str(out_path))

    counts = net.param_counts()
    digest = sha256_of(out_path)

    print(f"wrote {out_path}")
    print(f"sha256: {digest}")
    print(f"layers: {len(net.layers)}")
    print(f"nonzero_weights={counts['nonzero_weights']} "
          f"nonzero_biases={counts['nonzero_biases']} "
          f"nonzero_params_total={counts['nonzero_params_total']}")


if __name__ == "__main__":
    main()
