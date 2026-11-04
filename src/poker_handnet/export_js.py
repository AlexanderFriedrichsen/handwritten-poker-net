"""Serialize a CompiledNet to a plain-JSON-able dict, for the browser demo
and for the Node-based JS/numpy parity test to share one source of truth."""

from __future__ import annotations

from .dag import CompiledNet
from .handnet_spec import CLASS_NAMES


def net_to_json_obj(net: CompiledNet) -> dict:
    return {
        "input_names": net.input_names,
        "output_names": net.output_names,
        "class_names": CLASS_NAMES,
        "layers": [
            {
                "W": layer.W.tolist(),
                "b": layer.b.tolist(),
                "activation": layer.activation,
                "in_names": layer.in_names,
                "out_names": layer.out_names,
                "out_docs": layer.out_docs,
            }
            for layer in net.layers
        ],
    }
