import warnings

import pytest
import torch
import torch.nn as nn

from dee.predictors.base import (
    BasePredictor,
    available_predictors,
    build_predictor,
    register_predictor,
)


class ToyPredictor(BasePredictor):
    def __init__(self, input_dim, scale=1.0):
        super().__init__(input_dim)
        self.scale = scale
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1) * self.scale


def _ensure(name, cls):
    if name not in available_predictors():
        register_predictor(name)(cls)


def test_registry_roundtrip():
    _ensure("toy", ToyPredictor)
    assert "toy" in available_predictors()
    model = build_predictor("toy", input_dim=4, scale=2.0)
    assert isinstance(model, ToyPredictor)
    assert model.scale == 2.0
    assert model(torch.zeros(3, 4)).shape == (3,)


def test_duplicate_registration_raises():
    _ensure("toy", ToyPredictor)
    with pytest.raises(ValueError, match="already registered"):
        register_predictor("toy")(ToyPredictor)


def test_unknown_predictor_lists_available():
    with pytest.raises(ValueError, match="Available"):
        build_predictor("does_not_exist", input_dim=3)


def test_default_hooks():
    _ensure("toy_defaults", ToyPredictor)
    model = build_predictor("toy_defaults", input_dim=3)
    assert torch.equal(model.penalty(), torch.zeros(()))
    assert model.uses_weight_decay is False


def test_rejects_non_predictor_class():
    with pytest.raises(TypeError, match="BasePredictor"):
        register_predictor("bad")(dict)


def test_rejects_invalid_input_dim():
    with pytest.raises(ValueError, match="input_dim"):
        ToyPredictor(input_dim=0)


# ---------------------------------------------------------------------------
# Linear family
# ---------------------------------------------------------------------------
from dee.predictors.linear import (  # noqa: E402
    ElasticNetPredictor,
    LassoPredictor,
    LinearPredictor,
    RidgePredictor,
)


@pytest.mark.parametrize(
    "cls,expected_weight_decay",
    [
        (LinearPredictor, True),
        (RidgePredictor, False),
        (LassoPredictor, False),
        (ElasticNetPredictor, False),
    ],
)
def test_linear_family_shapes_and_flags(cls, expected_weight_decay):
    model = cls(input_dim=6)
    assert model(torch.randn(5, 6)).shape == (5,)
    assert model.uses_weight_decay is expected_weight_decay


def test_ridge_penalty_matches_weight_norm():
    model = RidgePredictor(input_dim=4)
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        model.linear.bias.fill_(0.0)
    assert torch.isclose(model.penalty(), torch.tensor(4.0))


def test_lasso_penalty_is_l1_of_weights():
    model = LassoPredictor(input_dim=4)
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1.0, -2.0, 0.0, 3.0]]))
        model.linear.bias.fill_(100.0)
    assert torch.isclose(model.penalty(), torch.tensor(6.0))


def test_elastic_net_penalty_mixes_l1_and_l2():
    model = ElasticNetPredictor(input_dim=2, l1_ratio=0.5)
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1.0, 1.0]]))
        model.linear.bias.fill_(0.0)
    assert torch.isclose(model.penalty(), torch.tensor(2.0))


def test_elastic_net_rejects_bad_l1_ratio():
    with pytest.raises(ValueError, match="l1_ratio"):
        ElasticNetPredictor(input_dim=3, l1_ratio=1.5)


# ---------------------------------------------------------------------------
# MLP and soft tree
# ---------------------------------------------------------------------------
from dee.predictors.mlp import MLPPredictor  # noqa: E402
from dee.predictors.soft_tree import SoftTreePredictor  # noqa: E402


def test_mlp_output_shape_and_weight_decay():
    model = MLPPredictor(input_dim=8, hidden=(16, 8), dropout=0.0)
    assert model(torch.randn(6, 8)).shape == (6,)
    assert model.uses_weight_decay is True


def test_mlp_rejects_invalid_hidden():
    with pytest.raises(ValueError, match="hidden"):
        MLPPredictor(input_dim=8, hidden=())


def test_soft_tree_output_shape():
    model = SoftTreePredictor(input_dim=5, depth=3)
    assert model(torch.randn(7, 5)).shape == (7,)
    assert model.n_leaves == 8


def test_soft_tree_leaf_probabilities_sum_to_one():
    model = SoftTreePredictor(input_dim=5, depth=3)
    with torch.no_grad():
        probs = model.leaf_probabilities(torch.randn(4, 5))
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)


def test_soft_tree_max_leaves_truncates_and_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SoftTreePredictor(input_dim=4, depth=12, max_leaves=16)
    assert model.depth == 4
    assert model.n_leaves == 16
    assert any("max_leaves" in str(w.message) for w in caught)


def test_soft_tree_rejects_bad_depth():
    with pytest.raises(ValueError, match="depth"):
        SoftTreePredictor(input_dim=4, depth=0)
    with pytest.raises(ValueError, match="max_leaves"):
        SoftTreePredictor(input_dim=4, depth=3, max_leaves=1)
