"""PyTorch reimplementation of CompiledNet.forward_numpy (dag.py), used to
prove numpy/PyTorch parity on the frozen weights (the same kind of check
the JS demo's forward pass is verified against, just for the other runtime
all the baseline-comparison code uses).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .dag import CompiledNet


class HandnetTorch(nn.Module):
    def __init__(self, compiled: CompiledNet):
        super().__init__()
        self.linears = nn.ModuleList()
        self.activations: list[str] = []
        for layer in compiled.layers:
            lin = nn.Linear(layer.W.shape[1], layer.W.shape[0], bias=True)
            with torch.no_grad():
                lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(layer.W)))
                lin.bias.copy_(torch.from_numpy(np.ascontiguousarray(layer.b)))
            self.linears.append(lin.double())
            self.activations.append(layer.activation)
        self.input_names = list(compiled.input_names)
        self.output_names = list(compiled.output_names)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.to(torch.float64)
        for lin, act in zip(self.linears, self.activations):
            h = lin(h)
            if act == "relu":
                h = torch.relu(h)
        return h

    @classmethod
    def from_npz(cls, path: str) -> "HandnetTorch":
        return cls(CompiledNet.from_npz(path))
