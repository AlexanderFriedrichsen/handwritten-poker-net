"""Checks demo/forward_pass.js (the browser demo's forward pass) agrees
bit-for-bit with forward_numpy on random hands, via Node. Requires
demo/weights.json (scripts/build_demo.py) and a `node` binary; skips
cleanly if either is missing rather than failing the whole suite.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from poker_handnet import encoding
from poker_handnet.dag import CompiledNet

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_JSON = REPO_ROOT / "demo" / "weights.json"
WEIGHTS_NPZ = REPO_ROOT / "weights" / "handnet_v1.npz"
RUNNER = REPO_ROOT / "tests" / "js_parity_runner.js"

NODE = shutil.which("node")


def _random_hand(rng: random.Random) -> list[int]:
    deck = [(s, r) for s in range(1, 5) for r in range(1, 14)]
    cards = rng.sample(deck, 5)
    raw: list[int] = []
    for s, r in cards:
        raw.extend([s, r])
    return raw


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.skipif(not WEIGHTS_JSON.exists(), reason="demo/weights.json not built - run scripts/build_demo.py")
def test_js_matches_numpy_forward_pass(tmp_path):
    rng = random.Random(20261103)
    hands = [_random_hand(rng) for _ in range(500)]

    input_path = tmp_path / "hands.json"
    output_path = tmp_path / "results.json"
    input_path.write_text(json.dumps(hands))

    subprocess.run(
        [NODE, str(RUNNER), str(WEIGHTS_JSON), str(input_path), str(output_path)],
        check=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    js_results = json.loads(output_path.read_text())
    assert len(js_results) == len(hands)

    net = CompiledNet.from_npz(str(WEIGHTS_NPZ))
    raw_arr = np.array(hands, dtype=np.float64)
    enc = encoding.encode(raw_arr)
    logits = net.forward_numpy(enc)
    py_pred = np.argmax(logits, axis=1)

    for i, (hand, js_r) in enumerate(zip(hands, js_results)):
        assert js_r["classIndex"] == int(py_pred[i]), f"class mismatch on hand {hand}: js={js_r['classIndex']} py={py_pred[i]}"
        np.testing.assert_allclose(js_r["logits"], logits[i], atol=1e-9)
