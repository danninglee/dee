"""Bootstrap row sampling and feature subsampling for the expert pool."""

from __future__ import annotations

import numpy as np


def _check_ratio(ratio: float, name: str) -> None:
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise TypeError(f"{name} must be a float")
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError(f"{name} must satisfy 0 < {name} <= 1, got {ratio!r}")


def bootstrap_indices(
    n_samples: int,
    n_experts: int,
    ratio: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Draw one bootstrap row pool per expert.

    Parameters
    ----------
    n_samples : int
        Number of rows in the training split.
    n_experts : int
        Number of experts to build pools for.
    ratio : float
        Bootstrap pool size as a fraction of ``n_samples``. Must be in ``(0, 1]``.
        Sampling is always with replacement; ``ratio=1.0`` is a full-size
        bootstrap, not a no-op.
    rng : numpy.random.Generator
        Source of randomness. Global RNG state is never touched.

    Returns
    -------
    list of numpy.ndarray
        One integer index array per expert, each of length
        ``max(1, floor(n_samples * ratio))``.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_experts <= 0:
        raise ValueError("n_experts must be positive")
    _check_ratio(ratio, "ratio")

    size = max(1, int(np.floor(n_samples * ratio)))
    return [rng.integers(0, n_samples, size=size) for _ in range(n_experts)]


def feature_subsets(
    n_features: int,
    n_experts: int,
    ratio: float,
    rng: np.random.Generator,
    ensure_coverage: bool = True,
) -> list[np.ndarray]:
    """Draw one feature subset per expert.

    Parameters
    ----------
    n_features : int
        Total number of factors.
    n_experts : int
        Number of experts.
    ratio : float
        Subset size as a fraction of ``n_features``. Must be in ``(0, 1]``.
    rng : numpy.random.Generator
        Source of randomness.
    ensure_coverage : bool
        If True, every feature is guaranteed to appear in at least one subset.
        Uncovered features are appended to subsets in round-robin order, so a
        subset may end up larger than ``max(1, floor(n_features * ratio))``.

    Returns
    -------
    list of numpy.ndarray
        One sorted, duplicate-free integer array per expert.
    """
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if n_experts <= 0:
        raise ValueError("n_experts must be positive")
    _check_ratio(ratio, "ratio")

    size = min(max(1, int(np.floor(n_features * ratio))), n_features)
    subsets = [
        np.sort(rng.choice(n_features, size=size, replace=False))
        for _ in range(n_experts)
    ]

    if ensure_coverage:
        covered = np.zeros(n_features, dtype=bool)
        for subset in subsets:
            covered[subset] = True
        missing = np.flatnonzero(~covered)
        for offset, feature in enumerate(missing):
            target = offset % n_experts
            subsets[target] = np.sort(np.append(subsets[target], feature))

    return subsets
