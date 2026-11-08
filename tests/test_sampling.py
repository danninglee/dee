import numpy as np
import pytest

from dee.sampling import bootstrap_indices, feature_subsets


def test_bootstrap_shapes_and_range():
    rng = np.random.default_rng(0)
    pools = bootstrap_indices(100, 5, 0.8, rng)
    assert len(pools) == 5
    for pool in pools:
        assert len(pool) == 80
        assert pool.min() >= 0 and pool.max() < 100


def test_bootstrap_is_with_replacement():
    rng = np.random.default_rng(0)
    pool = bootstrap_indices(50, 1, 1.0, rng)[0]
    assert len(pool) == 50
    assert len(np.unique(pool)) < 50


def test_bootstrap_deterministic_given_seed():
    a = bootstrap_indices(90, 4, 0.7, np.random.default_rng(7))
    b = bootstrap_indices(90, 4, 0.7, np.random.default_rng(7))
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_bootstrap_rejects_bad_ratio():
    with pytest.raises(ValueError, match="ratio"):
        bootstrap_indices(100, 3, 0.0, np.random.default_rng(0))
    with pytest.raises(ValueError, match="ratio"):
        bootstrap_indices(100, 3, 1.5, np.random.default_rng(0))


def test_feature_subsets_shape_and_uniqueness():
    rng = np.random.default_rng(0)
    subsets = feature_subsets(20, 6, 0.3, rng)
    assert len(subsets) == 6
    for subset in subsets:
        assert len(np.unique(subset)) == len(subset)
        assert subset.min() >= 0 and subset.max() < 20
        assert np.all(np.diff(subset) > 0)


def test_feature_subsets_full_coverage():
    rng = np.random.default_rng(1)
    subsets = feature_subsets(30, 5, 0.2, rng, ensure_coverage=True)
    assert set(np.concatenate(subsets).tolist()) == set(range(30))


def test_feature_subsets_without_coverage_can_miss():
    rng = np.random.default_rng(3)
    subsets = feature_subsets(100, 2, 0.05, rng, ensure_coverage=False)
    covered = set(np.concatenate(subsets).tolist())
    assert len(covered) < 100


def test_feature_subsets_rejects_bad_ratio():
    with pytest.raises(ValueError, match="ratio"):
        feature_subsets(10, 3, 0.0, np.random.default_rng(0))
