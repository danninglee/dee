"""Shallow multilayer perceptron expert."""

from __future__ import annotations

from typing import Iterable, Tuple, Union

import torch
import torch.nn as nn

from dee.predictors.base import BasePredictor, register_predictor


@register_predictor("mlp")
class MLPPredictor(BasePredictor):
    """MLP with batch normalization, matching the paper's nonlinear family.

    Architecture: optional dropout, then ``len(hidden)`` blocks of
    ``Linear -> BatchNorm1d -> LeakyReLU``, then an output ``Linear``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden: Union[Iterable[int], int] = (32,),
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(input_dim)
        if isinstance(hidden, int):
            hidden = (hidden,)
        hidden = tuple(int(h) for h in hidden)
        if not hidden or any(h <= 0 for h in hidden):
            raise ValueError(
                f"hidden must be a non-empty sequence of positive ints, got {hidden!r}"
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")

        self.hidden: Tuple[int, ...] = hidden
        self.dropout = float(dropout)

        dims = [input_dim, *hidden]
        layers: list[nn.Module] = []
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(dims[-1], 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @property
    def uses_weight_decay(self) -> bool:
        return True
