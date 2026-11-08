"""Built-in predictors and the registration interface."""

from dee.predictors.base import (
    BasePredictor,
    available_predictors,
    build_predictor,
    register_predictor,
)
from dee.predictors.linear import (
    ElasticNetPredictor,
    LassoPredictor,
    LinearPredictor,
    RidgePredictor,
)
from dee.predictors.mlp import MLPPredictor
from dee.predictors.soft_tree import SoftTreePredictor

__all__ = [
    "BasePredictor",
    "available_predictors",
    "build_predictor",
    "register_predictor",
    "LinearPredictor",
    "RidgePredictor",
    "LassoPredictor",
    "ElasticNetPredictor",
    "MLPPredictor",
    "SoftTreePredictor",
]
