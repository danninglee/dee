"""Diversified Expert Ensemble (DEE) for cross-sectional return prediction."""

from dee.data import PanelSplit
from dee.ensemble import DiversifiedExpertEnsemble

__all__ = ["PanelSplit", "DiversifiedExpertEnsemble", "__version__"]
__version__ = "0.1.0"
