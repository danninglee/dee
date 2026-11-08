import numpy as np
import pandas as pd
import pytest

from dee.data import PanelSplit


def make_panel(n_dates=5, n_assets=4, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2020-01-01", periods=n_dates), [f"S{i}" for i in range(n_assets)]],
        names=["datetime", "instrument"],
    )
    features = pd.DataFrame(
        rng.normal(size=(len(idx), n_features)),
        index=idx,
        columns=[f"f{i}" for i in range(n_features)],
    )
    labels = pd.Series(rng.normal(size=len(idx)), index=idx, name="label")
    return features, labels


def test_constructs_and_reports_shape():
    features, labels = make_panel()
    split = PanelSplit(features, labels)
    assert len(split) == 20
    assert len(split.dates) == 5
    assert len(split.instruments) == 4


def test_rejects_non_multiindex():
    features, labels = make_panel()
    flat = features.reset_index(drop=True)
    with pytest.raises(ValueError, match="MultiIndex"):
        PanelSplit(flat, pd.Series(labels.to_numpy()))


def test_rejects_mismatched_index():
    features, labels = make_panel()
    with pytest.raises(ValueError, match="index"):
        PanelSplit(features, labels.iloc[:-1])


def test_rejects_nan_features():
    features, labels = make_panel()
    features.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        PanelSplit(features, labels)


def test_allows_missing_labels_then_dropna():
    features, labels = make_panel()
    labels.iloc[0] = np.nan
    split = PanelSplit(features, labels)
    assert split.has_missing_labels
    assert len(split.dropna()) == 19


def test_labels_are_optional():
    features, _ = make_panel()
    split = PanelSplit(features)
    assert split.labels is None
    assert len(split) == 20


def test_fillna_returns_new_split():
    features, labels = make_panel()
    features.iloc[0, 0] = np.nan
    split = PanelSplit(features.fillna(0.0), labels)
    assert not split.features.isna().any().any()


def test_dropna_drops_missing_labels():
    features, labels = make_panel()
    labels.iloc[0] = np.nan
    split = PanelSplit(features, labels).dropna()
    assert len(split) == 19


def test_subset_slices_by_date():
    features, labels = make_panel(n_dates=10)
    split = PanelSplit(features, labels).subset("2020-01-02", "2020-01-06")
    assert split.dates.min() >= pd.Timestamp("2020-01-02")
    assert split.dates.max() <= pd.Timestamp("2020-01-06")


def test_rejects_swapped_level_names():
    features, labels = make_panel()
    swapped = features.swaplevel().sort_index()
    with pytest.raises(ValueError, match="datetime"):
        PanelSplit(swapped, labels.swaplevel().sort_index())
