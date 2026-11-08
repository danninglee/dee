"""Linear expert family: unregularized, ridge, lasso and elastic net."""

from __future__ import annotations

import torch
import torch.nn as nn

from dee.predictors.base import BasePredictor, register_predictor


@register_predictor("linear")
class LinearPredictor(BasePredictor):
    """Unregularized linear expert.

    Uses the optimizer's weight decay, matching the paper's unregularized
    baseline member; explicit penalties are reserved for the penalized
    variants below.
    """

    def __init__(self, input_dim: int, **kwargs) -> None:
        super().__init__(input_dim)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    @property
    def uses_weight_decay(self) -> bool:
        return True


class _PenalizedLinear(LinearPredictor):
    """Shared plumbing for linear experts with an explicit penalty."""

    @property
    def uses_weight_decay(self) -> bool:
        # Explicit penalties already regularize the weights; enabling the
        # optimizer weight decay would double-penalize them (paper Sec. 2.2).
        return False

    def _weight(self) -> torch.Tensor:
        return self.linear.weight


@register_predictor("ridge")
class RidgePredictor(_PenalizedLinear):
    """Linear expert with an L2 penalty ``alpha * ||w||^2``."""

    def penalty(self) -> torch.Tensor:
        return (self._weight() ** 2).sum()


@register_predictor("lasso")
class LassoPredictor(_PenalizedLinear):
    """Linear expert with an L1 penalty ``alpha * ||w||_1``."""

    def penalty(self) -> torch.Tensor:
        return self._weight().abs().sum()


@register_predictor("elastic_net")
class ElasticNetPredictor(_PenalizedLinear):
    """Linear expert with an elastic-net penalty.

    ``penalty = l1_ratio * ||w||_1 + (1 - l1_ratio) * ||w||^2``.
    """

    def __init__(self, input_dim: int, l1_ratio: float = 0.5, **kwargs) -> None:
        super().__init__(input_dim, **kwargs)
        if not 0.0 <= float(l1_ratio) <= 1.0:
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio!r}")
        self.l1_ratio = float(l1_ratio)

    def penalty(self) -> torch.Tensor:
        weight = self._weight()
        l1 = weight.abs().sum()
        l2 = (weight ** 2).sum()
        return self.l1_ratio * l1 + (1.0 - self.l1_ratio) * l2
