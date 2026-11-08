"""Predictor base class and global registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type

import torch
import torch.nn as nn

_REGISTRY: Dict[str, Type["BasePredictor"]] = {}


class BasePredictor(nn.Module, ABC):
    """Base class for a single expert.

    Subclasses must implement :meth:`forward` and may override :meth:`penalty`
    and :attr:`uses_weight_decay`.

    Contract
    --------
    ``forward`` receives a tensor of shape ``(batch, input_dim)`` where
    ``input_dim`` is the size of *this expert's* feature subset. It must return
    a tensor of shape ``(batch,)``.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive int, got {input_dim!r}")
        self.input_dim = input_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions of shape ``(batch,)``."""

    def penalty(self) -> torch.Tensor:
        """Return the explicit regularization term for this expert.

        The training loop adds ``alpha * penalty()`` to the MSE loss. The
        default is zero. Implementations should regularize weights only, not
        biases.
        """
        param = next(self.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        return torch.zeros((), device=device)

    @property
    def uses_weight_decay(self) -> bool:
        """Whether the optimizer should apply the global weight decay."""
        return False

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def register_predictor(name: str):
    """Class decorator that registers a predictor under ``name``."""

    if not isinstance(name, str) or not name:
        raise ValueError("predictor name must be a non-empty string")

    def decorator(cls):
        if not (isinstance(cls, type) and issubclass(cls, BasePredictor)):
            raise TypeError("register_predictor requires a BasePredictor subclass")
        if name in _REGISTRY:
            raise ValueError(f"predictor {name!r} is already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def build_predictor(name: str, input_dim: int, **kwargs) -> BasePredictor:
    """Instantiate a registered predictor."""
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown predictor {name!r}. Available predictors: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](input_dim=input_dim, **kwargs)


def available_predictors() -> list[str]:
    """Return the sorted names of all registered predictors."""
    return sorted(_REGISTRY)
