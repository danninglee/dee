"""Soft decision tree expert."""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn

from dee.predictors.base import BasePredictor, register_predictor


@register_predictor("soft_tree")
class SoftTreePredictor(BasePredictor):
    """Soft decision tree with sigmoid gates and learned leaf values.

    Internal node ``n`` at level ``l`` computes ``g_n(x) = sigmoid(a_n^T x + b_n)``.
    The prediction is ``sum_l P(l | x) * mu_l`` where ``P(l | x)`` is the product
    of branch probabilities along the path from the root to leaf ``l``.

    Leaves are numbered in breadth-first order: at each level the left child
    (gate probability) is appended before the right child (one minus gate).

    Parameters
    ----------
    depth : int
        Tree depth. Default 12 (paper setting), giving ``2**12 = 4096`` leaves.
    max_leaves : int
        Hard cap on the number of leaves. If ``2**depth`` exceeds it, ``depth``
        is reduced to ``floor(log2(max_leaves))`` and a warning is emitted.
    """

    def __init__(
        self,
        input_dim: int,
        depth: int = 12,
        max_leaves: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__(input_dim)
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError(f"depth must be a positive int, got {depth!r}")
        if isinstance(max_leaves, bool) or not isinstance(max_leaves, int) or max_leaves < 2:
            raise ValueError(f"max_leaves must be an int >= 2, got {max_leaves!r}")

        max_depth = int(math.floor(math.log2(max_leaves)))
        if depth > max_depth:
            warnings.warn(
                f"depth={depth} would create {2 ** depth} leaves, exceeding "
                f"max_leaves={max_leaves}; depth truncated to {max_depth}.",
                stacklevel=2,
            )
            depth = max_depth

        self.depth = depth
        self.max_leaves = max_leaves
        self.n_leaves = 2 ** depth

        self.gates = nn.ModuleList(
            [nn.Linear(input_dim, 2 ** level) for level in range(depth)]
        )
        self.leaf_values = nn.Parameter(torch.zeros(self.n_leaves, 1))

    def leaf_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``P(l | x)`` with shape ``(batch, n_leaves)``."""
        probs = x.new_ones(x.size(0), 1)
        for layer in self.gates:
            gate = torch.sigmoid(layer(x))
            probs = torch.cat([probs * gate, probs * (1.0 - gate)], dim=1)
        return probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.leaf_probabilities(x)
        return (probs * self.leaf_values.squeeze(-1)).sum(dim=1)

    @property
    def uses_weight_decay(self) -> bool:
        return True
