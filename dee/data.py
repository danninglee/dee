"""Generic panel data container for cross-sectional return prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def _is_numeric(obj) -> bool:
    if isinstance(obj, pd.DataFrame):
        return all(pd.api.types.is_numeric_dtype(dtype) for dtype in obj.dtypes)
    return bool(pd.api.types.is_numeric_dtype(obj))


@dataclass
class PanelSplit:
    """A validated panel of features and labels.

    Parameters
    ----------
    features : pandas.DataFrame
        Index must be a two-level ``MultiIndex`` whose level 0 is the date and
        level 1 is the instrument. Columns are factors.
    labels : pandas.Series, optional
        Forward returns with an index identical to ``features.index``. Required
        for training and evaluation; may be omitted for prediction-only splits.
    name : str
        Optional human-readable split name (e.g. ``"train"``).

    Notes
    -----
    Features must be finite. Labels may contain NaN — these rows are expected
    at the end of a forward-return sample and are removed with :meth:`dropna`.
    """

    features: pd.DataFrame
    labels: Optional[pd.Series] = None
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.features, pd.DataFrame):
            raise TypeError("features must be a pandas.DataFrame")
        if self.labels is not None and not isinstance(self.labels, pd.Series):
            raise TypeError("labels must be a pandas.Series or None")
        if not isinstance(self.features.index, pd.MultiIndex) or self.features.index.nlevels != 2:
            raise ValueError(
                "features.index must be a pandas MultiIndex with exactly two levels "
                "(datetime, instrument)"
            )
        if self.features.index.names == ["instrument", "datetime"]:
            raise ValueError(
                "features.index levels are (instrument, datetime); expected (datetime, instrument). "
                "Call features.swaplevel().sort_index() before constructing PanelSplit."
            )
        if self.labels is not None and not self.labels.index.equals(self.features.index):
            raise ValueError("labels.index must be identical to features.index")
        if not self.features.columns.is_unique:
            raise ValueError("features.columns must be unique")
        if not _is_numeric(self.features):
            raise ValueError("features must contain only numeric columns")
        if self.labels is not None and not _is_numeric(self.labels):
            raise ValueError("labels must be numeric")
        if not np.isfinite(self.features.to_numpy(dtype=float)).all():
            bad = self.features.isna().sum()
            bad = bad[bad > 0].to_dict()
            raise ValueError(
                "features contain NaN or infinite values; call fillna() first. "
                f"Columns with NaN: {bad}"
            )

    def __len__(self) -> int:
        return len(self.features)

    @property
    def dates(self) -> pd.Index:
        """Sorted unique dates in the panel."""
        return self.features.index.get_level_values(0).unique().sort_values()

    @property
    def instruments(self) -> pd.Index:
        """Sorted unique instruments in the panel."""
        return self.features.index.get_level_values(1).unique().sort_values()

    @property
    def has_missing_labels(self) -> bool:
        if self.labels is None:
            return False
        return not np.isfinite(self.labels.to_numpy(dtype=float)).all()

    def dropna(self) -> "PanelSplit":
        """Return a copy with rows whose label is missing removed."""
        if self.labels is None:
            return PanelSplit(self.features.copy(), None, name=self.name)
        mask = np.isfinite(self.labels.to_numpy(dtype=float))
        return PanelSplit(self.features.loc[mask], self.labels.loc[mask], name=self.name)

    def fillna(self, value: float = 0.0) -> "PanelSplit":
        """Return a copy with missing numeric values replaced by ``value``."""
        labels = None if self.labels is None else self.labels.fillna(value)
        return PanelSplit(self.features.fillna(value), labels, name=self.name)

    def subset(self, start, end) -> "PanelSplit":
        """Return the rows whose date lies in ``[start, end]``."""
        dates = self.features.index.get_level_values(0)
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        labels = None if self.labels is None else self.labels.loc[mask]
        return PanelSplit(self.features.loc[mask], labels, name=self.name)
