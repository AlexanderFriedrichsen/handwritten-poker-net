"""numpy vs PyTorch parity, and npz serialization round-trip fidelity, for
the compiled hand-written net. (JS vs numpy parity is covered separately in
tests/test_js_parity.py once the demo exists.)
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

from poker_handnet import encoding
from poker_handnet.forward_torch import HandnetTorch
from poker_handnet.handnet_spec import build_handnet


def _sample_raw_hands() -> np.ndarray:
    deck = [(s, r) for s in range(1, 5) for r in range(1, 14)]
    rng = np.random.default_rng(42)
    combos = list(itertools.combinations(range(52), 5))
    idx = rng.choice(len(combos), size=2000, replace=False)
    raw = np.empty((len(idx), 10), dtype=np.float64)
    for i, ci in enumerate(idx):
        for j, card_i in enumerate(combos[ci]):
            s, r = deck[card_i]
            raw[i, 2 * j] = s
            raw[i, 2 * j + 1] = r
    return raw


def test_numpy_and_torch_forward_agree():
    net = build_handnet()
    torch_net = HandnetTorch(net)

    raw = _sample_raw_hands()
    enc = encoding.encode(raw)

    numpy_out = net.forward_numpy(enc)
    with torch.no_grad():
        torch_out = torch_net(torch.from_numpy(enc)).numpy()

    np.testing.assert_allclose(numpy_out, torch_out, atol=1e-10)
    np.testing.assert_array_equal(np.argmax(numpy_out, axis=1), np.argmax(torch_out, axis=1))


def test_npz_roundtrip_preserves_forward_pass(tmp_path):
    net = build_handnet()
    raw = _sample_raw_hands()
    enc = encoding.encode(raw)
    before = net.forward_numpy(enc)

    path = tmp_path / "handnet.npz"
    net.to_npz(str(path))

    from poker_handnet.dag import CompiledNet

    reloaded = CompiledNet.from_npz(str(path))
    after = reloaded.forward_numpy(enc)

    np.testing.assert_allclose(before, after)
    assert reloaded.input_names == net.input_names
    assert reloaded.output_names == net.output_names
    for l1, l2 in zip(net.layers, reloaded.layers):
        assert l1.out_docs == l2.out_docs
        assert l1.out_groups == l2.out_groups
