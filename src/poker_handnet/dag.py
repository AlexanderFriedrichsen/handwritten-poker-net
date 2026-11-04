"""A tiny named-node compiler that turns a hand-specified circuit of linear
combinations + ReLUs into a dense "Linear -> ReLU -> Linear -> ReLU -> ... ->
Linear" stack (a plain feedforward net).

Why this exists: the hand-written poker net (see handnet_spec.py) is designed
as ~300 small, individually-checkable named formulas such as
"pair_rank_7 = ReLU(rank_count_7 - 1) - 2*ReLU(rank_count_7 - 2) + ReLU(rank_count_7 - 3)".
Each node lives at whatever "depth" its dependencies put it at. A node whose
value is needed by a consumer several layers later must be carried forward
through every intermediate layer via an identity ("passthrough") copy -
exactly what a wide hidden layer in a real MLP does implicitly. Doing that
bookkeeping by hand across ~10 layers is exactly the kind of place an
off-by-one would hide, so it is done once, mechanically, here, and unit
tested (tests/test_dag.py) independently of any poker logic.

Every node's activation is ReLU except the final output layer, which is
linear (raw logits feed argmax). This is safe here because every node in the
graph is proven (see handnet_spec.py comments) to always evaluate to a
non-negative number, so ReLU is a no-op clamp, not a distortion - this is the
"clamp built from ReLUs" the project's integrity rulings allow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Node:
    name: str
    parents: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    doc: str = ""
    group: str = ""
    is_input: bool = False


@dataclass
class CompiledLayer:
    """One Linear(+ReLU) layer: y = activation(x @ W.T + b)."""

    W: np.ndarray  # (out_dim, in_dim)
    b: np.ndarray  # (out_dim,)
    activation: str  # "relu" or "linear"
    in_names: list[str]
    out_names: list[str]
    out_docs: list[str]
    out_groups: list[str]


@dataclass
class CompiledNet:
    layers: list[CompiledLayer]
    input_names: list[str]
    output_names: list[str]

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, dtype=np.float64)
        for layer in self.layers:
            h = h @ layer.W.T + layer.b
            if layer.activation == "relu":
                h = np.maximum(h, 0.0)
        return h

    def param_counts(self) -> dict:
        """Non-zero weights and biases, per layer and in total (ruling: count
        weights and biases separately and in total)."""
        per_layer = []
        total_w = total_b = 0
        for i, layer in enumerate(self.layers):
            nw = int(np.count_nonzero(layer.W))
            nb = int(np.count_nonzero(layer.b))
            per_layer.append({"layer": i, "nonzero_weights": nw, "nonzero_biases": nb})
            total_w += nw
            total_b += nb
        return {
            "per_layer": per_layer,
            "nonzero_weights": total_w,
            "nonzero_biases": total_b,
            "nonzero_params_total": total_w + total_b,
        }

    def to_npz(self, path: str) -> None:
        """Serialize every layer's (W, b, activation) plus per-node names/
        docs/groups to a single .npz - this is the frozen weights artifact."""
        arrays = {
            "n_layers": np.array(len(self.layers)),
            "input_names": np.array(self.input_names),
            "output_names": np.array(self.output_names),
        }
        for i, layer in enumerate(self.layers):
            arrays[f"W_{i}"] = layer.W
            arrays[f"b_{i}"] = layer.b
            arrays[f"activation_{i}"] = np.array(layer.activation)
            arrays[f"in_names_{i}"] = np.array(layer.in_names)
            arrays[f"out_names_{i}"] = np.array(layer.out_names)
            arrays[f"out_docs_{i}"] = np.array(layer.out_docs)
            arrays[f"out_groups_{i}"] = np.array(layer.out_groups)
        np.savez(path, **arrays)

    @classmethod
    def from_npz(cls, path: str) -> "CompiledNet":
        data = np.load(path)
        n_layers = int(data["n_layers"].item())
        layers = []
        for i in range(n_layers):
            layers.append(
                CompiledLayer(
                    W=data[f"W_{i}"],
                    b=data[f"b_{i}"],
                    activation=str(data[f"activation_{i}"].item()),
                    in_names=list(data[f"in_names_{i}"]),
                    out_names=list(data[f"out_names_{i}"]),
                    out_docs=list(data[f"out_docs_{i}"]),
                    out_groups=list(data[f"out_groups_{i}"]),
                )
            )
        return cls(
            layers=layers,
            input_names=list(data["input_names"]),
            output_names=list(data["output_names"]),
        )


class Graph:
    """Builder. Nodes must be added in dependency order: a node's parents
    must already exist when it is added. This also makes insertion order a
    valid topological order for free, so the compiler never needs a separate
    topo-sort pass."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self._order: list[str] = []

    def input(self, name: str, doc: str = "", group: str = "input") -> str:
        if name in self.nodes:
            raise ValueError(f"duplicate node name: {name}")
        self.nodes[name] = Node(name=name, is_input=True, doc=doc, group=group)
        self._order.append(name)
        return name

    def add(
        self,
        name: str,
        parents: dict[str, float],
        bias: float = 0.0,
        doc: str = "",
        group: str = "",
    ) -> str:
        if name in self.nodes:
            raise ValueError(f"duplicate node name: {name}")
        for p in parents:
            if p not in self.nodes:
                raise ValueError(f"node {name!r}: unknown parent {p!r}")
        self.nodes[name] = Node(name=name, parents=dict(parents), bias=bias, doc=doc, group=group)
        self._order.append(name)
        return name

    def compile(self, input_order: list[str], output_order: list[str]) -> CompiledNet:
        return _compile_graph(self, input_order, output_order)


def _compile_graph(g: Graph, input_order: list[str], output_order: list[str]) -> CompiledNet:
    for name in input_order:
        if name not in g.nodes or not g.nodes[name].is_input:
            raise ValueError(f"{name!r} is not a declared input node")
    for name in output_order:
        if name not in g.nodes:
            raise ValueError(f"output {name!r} is not a declared node")

    # 1. natural depth = longest path from an input. Insertion order is
    # already topological (enforced at add()-time), so a single left-to-right
    # pass suffices.
    depth: dict[str, int] = {name: 0 for name in input_order}
    for name in g._order:
        node = g.nodes[name]
        if node.is_input:
            depth.setdefault(name, 0)
            continue
        depth[name] = 1 + max(depth[p] for p in node.parents)

    output_depth = max(depth[o] for o in output_order)

    # 2. needed_until[name] = last depth at which `name`'s value must still be
    # directly readable as a layer's input. Every consumer at depth[consumer]
    # reads its parents from layer depth[consumer]-1, so a parent must be
    # extended up to at least that. Every output must be extended up to
    # exactly output_depth (the final layer).
    needed_until: dict[str, int] = dict(depth)
    for name in g._order:
        node = g.nodes[name]
        for p in node.parents:
            needed_until[p] = max(needed_until[p], depth[name] - 1)
    for o in output_order:
        needed_until[o] = max(needed_until[o], output_depth)

    # 3. materialize passthrough chains: for every node whose needed_until
    # exceeds its natural depth, add identity nodes at every intermediate
    # depth. alias[(name, d)] gives the node name that holds `name`'s value
    # at depth d.
    alias: dict[tuple[str, int], str] = {}
    extra_nodes: dict[str, Node] = {}
    extra_depth: dict[str, int] = {}
    for name in list(g._order):
        d0 = depth[name]
        alias[(name, d0)] = name
        prev = name
        for d in range(d0 + 1, needed_until[name] + 1):
            pt_name = f"{name}__pt{d}"
            extra_nodes[pt_name] = Node(
                name=pt_name,
                parents={prev: 1.0},
                bias=0.0,
                doc=f"(passthrough of {name} to layer {d})",
                group="passthrough",
            )
            extra_depth[pt_name] = d
            alias[(name, d)] = pt_name
            alias[(pt_name, d)] = pt_name  # a passthrough is itself resolvable at its own depth
            prev = pt_name

    def resolve(name: str, at_depth: int) -> str:
        return alias[(name, at_depth)]

    all_depth = dict(depth)
    all_depth.update(extra_depth)
    all_nodes = dict(g.nodes)
    all_nodes.update(extra_nodes)

    # 4. group nodes per layer. Layer 0 is exactly input_order. Layers
    # 1..output_depth-1 hold every (original + passthrough) node at that
    # depth, deterministically ordered. The final layer holds exactly the
    # requested outputs, in the requested order.
    max_mid_depth = output_depth - 1
    layer_nodes: dict[int, list[str]] = {0: list(input_order)}
    for d in range(1, max_mid_depth + 1):
        names_at_d = [n for n in g._order if depth.get(n) == d]
        names_at_d += [n for n in extra_nodes if extra_depth[n] == d]
        layer_nodes[d] = names_at_d
    layer_nodes[output_depth] = [resolve(o, output_depth) for o in output_order]

    # 5. emit dense (W, b) per layer.
    layers: list[CompiledLayer] = []
    prev_names = layer_nodes[0]
    for d in range(1, output_depth + 1):
        cur_names = layer_nodes[d]
        idx0 = {n: i for i, n in enumerate(prev_names)}
        W = np.zeros((len(cur_names), len(prev_names)), dtype=np.float64)
        b = np.zeros((len(cur_names),), dtype=np.float64)
        docs, groups = [], []
        for row, name in enumerate(cur_names):
            node = all_nodes[name]
            docs.append(node.doc)
            groups.append(node.group)
            b[row] = node.bias
            for parent_name, weight in node.parents.items():
                resolved = resolve(parent_name, d - 1)
                if resolved not in idx0:
                    raise AssertionError(
                        f"internal error compiling {name!r}: parent "
                        f"{parent_name!r} not available at depth {d - 1}"
                    )
                W[row, idx0[resolved]] += weight
        activation = "relu" if d < output_depth else "linear"
        layers.append(
            CompiledLayer(
                W=W,
                b=b,
                activation=activation,
                in_names=list(prev_names),
                out_names=list(cur_names),
                out_docs=docs,
                out_groups=groups,
            )
        )
        prev_names = cur_names

    if prev_names != [resolve(o, output_depth) for o in output_order]:
        raise AssertionError("final layer does not match requested output order")

    return CompiledNet(layers=layers, input_names=list(input_order), output_names=list(output_order))
