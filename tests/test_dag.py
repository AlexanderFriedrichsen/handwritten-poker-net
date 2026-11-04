"""Unit tests for the graph-to-dense-layers compiler (dag.py), using toy
graphs with no poker content at all. These must pass independently of
whether handnet_spec.py's poker logic is correct - a bug in the compiler
and a bug in the poker derivation should never be confused for each other.
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_handnet.dag import Graph


def test_single_layer_linear_output():
    g = Graph()
    a = g.input("a")
    b = g.input("b")
    c = g.add("c", {a: 2.0, b: -1.0}, bias=0.5)
    net = g.compile([a, b], [c])

    assert len(net.layers) == 1
    assert net.layers[0].activation == "linear"

    x = np.array([[3.0, 4.0], [1.0, 1.0]])
    out = net.forward_numpy(x)
    expected = (2.0 * x[:, 0] - 1.0 * x[:, 1] + 0.5).reshape(-1, 1)
    np.testing.assert_allclose(out, expected)


def test_relu_applied_on_hidden_layers_only():
    g = Graph()
    a = g.input("a")
    hidden = g.add("hidden", {a: 1.0}, bias=-5.0)  # ReLU(a - 5)
    out = g.add("out", {hidden: 1.0}, bias=-100.0)  # linear: (a-5 clamped) - 100
    net = g.compile([a], [out])

    assert len(net.layers) == 2
    assert net.layers[0].activation == "relu"
    assert net.layers[1].activation == "linear"

    x = np.array([[0.0], [10.0]])
    result = net.forward_numpy(x)
    # a=0: ReLU(0-5)=0, then 0-100=-100
    # a=10: ReLU(10-5)=5, then 5-100=-95
    np.testing.assert_allclose(result, [[-100.0], [-95.0]])


def test_passthrough_bridges_a_multi_layer_gap():
    """`a` is consumed directly (depth 1) AND needed again at depth 3, so the
    compiler must carry it forward through an intermediate passthrough."""
    g = Graph()
    a = g.input("a")
    b = g.input("b")
    d1 = g.add("d1", {a: 1.0, b: 1.0})  # depth 1
    d2 = g.add("d2", {d1: 1.0})  # depth 2
    # d3 depends on both d2 (depth 2) and the original `a` (depth 0) - `a`
    # must be bridged across depths 1 and 2 to be available here.
    d3 = g.add("d3", {d2: 1.0, a: 10.0})  # depth 3
    net = g.compile([a, b], [d3])

    assert len(net.layers) == 3
    x = np.array([[2.0, 3.0], [-1.0, 5.0]])
    out = net.forward_numpy(x)
    # Every hidden-layer node is ReLU'd, including passthrough copies of `a`
    # - this is why the real poker spec only ever passes through quantities
    # it has separately proven are non-negative (see handnet_spec.py).
    # a=2,b=3: d1=5, d2=5, a-passthrough=ReLU(2)=2 at every hidden depth,
    #   d3 = d2 + 10*2 = 25.
    # a=-1,b=5: d1=ReLU(4)=4, d2=ReLU(4)=4, a-passthrough=ReLU(-1)=0 at
    #   depth 1 and stays 0, d3 = d2 + 10*0 = 4 (not 4 + 10*(-1)=-6: the
    #   passthrough clamps the negative `a` to 0 before it reaches d3).
    np.testing.assert_allclose(out, [[25.0], [4.0]])


def test_multiple_outputs_preserve_requested_order():
    g = Graph()
    a = g.input("a")
    x = g.add("x", {a: 1.0}, bias=1.0)
    y = g.add("y", {a: 1.0}, bias=2.0)
    net = g.compile([a], [y, x])  # deliberately reversed vs declaration order

    assert net.layers[-1].out_names == ["y", "x"]
    out = net.forward_numpy(np.array([[0.0]]))
    np.testing.assert_allclose(out, [[2.0, 1.0]])


def test_param_counts_nonzero_weights_and_biases():
    g = Graph()
    a = g.input("a")
    b = g.input("b")
    # one zero-weight edge (a*0.0) should not count as a non-zero weight
    c = g.add("c", {a: 1.0, b: 0.0}, bias=0.0)
    out = g.add("out", {c: 1.0}, bias=3.0)
    net = g.compile([a, b], [out])

    counts = net.param_counts()
    assert counts["nonzero_biases"] == 1  # only `out`'s bias is non-zero
    # layer 0: c's edge from a counts (weight 1.0); the explicit 0.0 edge to b
    # is stored as a zero entry in the dense matrix and must NOT be counted.
    assert counts["nonzero_weights"] == 2  # a->c, c->out
    assert counts["nonzero_params_total"] == counts["nonzero_weights"] + counts["nonzero_biases"]


def test_duplicate_name_rejected():
    g = Graph()
    g.input("a")
    with pytest.raises(ValueError):
        g.input("a")


def test_unknown_parent_rejected():
    g = Graph()
    a = g.input("a")
    with pytest.raises(ValueError):
        g.add("b", {"nonexistent": 1.0})
    del a  # silence unused-var lint; `a` only exists to make the graph non-trivial


def test_unknown_output_rejected():
    g = Graph()
    a = g.input("a")
    with pytest.raises(ValueError):
        g.compile([a], ["nonexistent"])


def test_diamond_dependency_merges_at_correct_depth():
    """a -> b, a -> c (both depth 1), out = b + c (depth 2): a non-trivial
    DAG shape (not just a chain), to check depth assignment handles fan-out
    and fan-in correctly."""
    g = Graph()
    a = g.input("a")
    b = g.add("b", {a: 2.0})
    c = g.add("c", {a: 3.0})
    out = g.add("out", {b: 1.0, c: 1.0})
    net = g.compile([a], [out])

    assert len(net.layers) == 2
    result = net.forward_numpy(np.array([[1.0], [-1.0]]))
    # a=1: b=2,c=3,out=5. a=-1: b=-2->ReLU0, c=-3->ReLU0, out=0
    np.testing.assert_allclose(result, [[5.0], [0.0]])
