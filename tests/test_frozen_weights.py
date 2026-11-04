"""Checks the committed weights/handnet_v1.npz artifact itself (not just the
serialization mechanism) matches what handnet_spec.py currently compiles to.
If this ever fails, either the frozen artifact is stale (re-run
scripts/build_weights.py - and remember that is only allowed as the one
sanctioned Phase 2 revision after freezing, see DECISIONS.md) or it was
hand-edited, which should never happen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from poker_handnet.dag import CompiledNet
from poker_handnet.handnet_spec import build_handnet

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "handnet_v1.npz"


def test_frozen_weights_file_exists():
    assert WEIGHTS_PATH.exists(), f"missing {WEIGHTS_PATH} - run scripts/build_weights.py"


def test_frozen_weights_match_current_spec_compilation():
    frozen = CompiledNet.from_npz(str(WEIGHTS_PATH))
    fresh = build_handnet()

    assert frozen.input_names == fresh.input_names
    assert frozen.output_names == fresh.output_names
    assert len(frozen.layers) == len(fresh.layers)
    for fl, rl in zip(frozen.layers, fresh.layers):
        np.testing.assert_array_equal(fl.W, rl.W)
        np.testing.assert_array_equal(fl.b, rl.b)
        assert fl.activation == rl.activation
        assert fl.out_names == rl.out_names
