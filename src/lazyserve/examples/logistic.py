"""A tiny logistic regression classifier in pure Python.

It trains on a fixed synthetic two-class dataset when it loads, so loading
takes a moment of real work and the model needs no numpy or scikit-learn.
"""

from __future__ import annotations

import math
import random
from typing import Any


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class LogisticModel:
    """Binary logistic regression with a scikit-learn style fit/predict API."""

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features
        self.weights = [0.0] * n_features
        self.bias = 0.0

    def fit(
        self, rows: list[list[float]], labels: list[int], epochs: int = 200, lr: float = 0.1
    ) -> LogisticModel:
        n = len(rows)
        for _ in range(epochs):
            grad_w = [0.0] * self.n_features
            grad_b = 0.0
            for x, y in zip(rows, labels, strict=True):
                err = self._prob(x) - y
                for j, xj in enumerate(x):
                    grad_w[j] += err * xj
                grad_b += err
            self.weights = [w - lr * g / n for w, g in zip(self.weights, grad_w, strict=True)]
            self.bias -= lr * grad_b / n
        return self

    def _prob(self, x: list[float]) -> float:
        return _sigmoid(sum(w * xi for w, xi in zip(self.weights, x, strict=True)) + self.bias)

    def predict_proba(self, rows: list[list[float]]) -> list[float]:
        return [self._prob(self._check_row(x)) for x in rows]

    def predict(self, inputs: Any) -> dict[str, list]:
        """``inputs`` is a list of feature rows, each a list of ``n_features`` numbers."""
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("inputs must be a non-empty list of feature rows")
        probs = self.predict_proba(inputs)
        return {
            "labels": [int(p >= 0.5) for p in probs],
            "probabilities": [round(p, 4) for p in probs],
        }

    def _check_row(self, x: Any) -> list[float]:
        if (
            not isinstance(x, list)
            or len(x) != self.n_features
            or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in x)
        ):
            raise ValueError(f"each row must be a list of {self.n_features} numbers, got {x!r}")
        return x


def make_blobs(n: int = 400, seed: int = 7) -> tuple[list[list[float]], list[int]]:
    """Two Gaussian blobs centred at (-1, -1) and (1, 1)."""
    rng = random.Random(seed)
    rows, labels = [], []
    for i in range(n):
        label = i % 2
        centre = 1.0 if label else -1.0
        rows.append([rng.gauss(centre, 0.8), rng.gauss(centre, 0.8)])
        labels.append(label)
    return rows, labels


def load(n_samples: int = 400, epochs: int = 200, seed: int = 7) -> LogisticModel:
    rows, labels = make_blobs(n_samples, seed)
    return LogisticModel(n_features=2).fit(rows, labels, epochs=epochs)
