"""Trained MLP baselines. Same op shape as the hand-written net (Linear +
ReLU, stacked, final layer linear logits + argmax) so the comparison is
about *how the weights were obtained*, not about architecture novelty.

"small" is param-count-matched to the hand-written net (393 non-zero
params); "large" clears 100k params - see DECISIONS.md for the arithmetic.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

ARCHS = {
    "mlp_small": (14,),
    "mlp_large": (320, 320),
}


def make_mlp(size: str, in_dim: int, out_dim: int = 10) -> nn.Sequential:
    hidden = ARCHS[size]
    dims = (in_dim, *hidden, out_dim)
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    size: str,
    seed: int,
    device: str = "cpu",
    lr: float = 1e-3,
    max_epochs: int = 3000,
    patience: int = 100,
    weight_decay: float = 1e-5,
) -> tuple[nn.Sequential, dict]:
    torch.manual_seed(seed)
    model = make_mlp(size, in_dim=X_train.shape[1]).to(device).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X_train).double().to(device)
    yt = torch.from_numpy(y_train).long().to(device)
    Xd = torch.from_numpy(X_dev).double().to(device)
    yd = torch.from_numpy(y_dev).long().to(device)

    batch_size = min(512, len(Xt))
    n = len(Xt)

    best_dev_loss = float("inf")
    best_state = None
    epochs_without_improve = 0
    rng = np.random.default_rng(seed)

    for epoch in range(max_epochs):
        perm = rng.permutation(n)
        model.train()
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = Xt[idx]
            yb = yt[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            dev_loss = loss_fn(model(Xd), yd).item()

        if dev_loss < best_dev_loss - 1e-6:
            best_dev_loss = dev_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    history = {"epochs_run": epoch + 1, "best_dev_loss": best_dev_loss}
    return model, history


def predict(model: nn.Sequential, X: np.ndarray, device: str = "cpu", batch_size: int = 100_000) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).double().to(device)
            logits = model(xb)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds)
