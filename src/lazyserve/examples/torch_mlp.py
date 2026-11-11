"""A small PyTorch MLP. Needs the optional extra: ``pip install "lazy-model-server[torch]"``.

torch is imported inside ``load``, so a server that has this model configured
but never calls it does not pay the import cost or the memory for torch.
"""

from __future__ import annotations

from typing import Any


class TorchMLP:
    def __init__(self, net: Any, torch: Any, device: str) -> None:
        self.net = net
        self.torch = torch
        self.device = device

    def predict(self, inputs: Any) -> dict[str, Any]:
        """``inputs`` is a list of rows with ``in_features`` numbers each."""
        torch = self.torch
        x = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        if x.dim() != 2 or x.shape[1] != self.net[0].in_features:
            raise ValueError(f"inputs must have shape (n, {self.net[0].in_features})")
        with torch.inference_mode():
            probs = torch.softmax(self.net(x), dim=1)
        return {"classes": probs.argmax(dim=1).tolist(), "probabilities": probs.tolist()}

    def close(self) -> None:
        # Drop the weights so the registry's gc.collect() and empty_cache() can free them.
        self.net = None


def load(
    in_features: int = 16, hidden: int = 256, classes: int = 4, device: str = "auto", seed: int = 0
) -> TorchMLP:
    import torch
    from torch import nn

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, classes),
    ).to(device)
    net.eval()
    return TorchMLP(net, torch, device)
