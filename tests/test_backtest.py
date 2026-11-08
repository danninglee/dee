import numpy as np
import pandas as pd
import pytest

from dee.backtest import topk_backtest


def make_index(n_assets, date="2020-01-01"):
    return pd.MultiIndex.from_product(
        [[pd.Timestamp(date)], [f"S{i}" for i in range(n_assets)]],
        names=["datetime", "instrument"],
    )


def test_selects_top_k_only():
    idx = make_index(4)
    scores = pd.Series([0.1, 0.2, 0.9, 0.8], index=idx)
    returns = pd.Series([0.0, 0.0, 0.10, 0.20], index=idx)
    result = topk_backtest(scores, returns, k=2, periods_per_year=1)
    assert np.isclose(result.daily_returns.iloc[0], 0.15)


def test_metrics_on_constant_returns():
    idx = make_index(2)
    scores = pd.Series([0.5, 0.4], index=idx)
    returns = pd.Series([0.01, 0.01], index=idx)
    result = topk_backtest(scores, returns, k=2, periods_per_year=1)
    assert np.isclose(result.metrics["annual_return"], 0.01)
    assert np.isclose(result.metrics["annual_vol"], 0.0)
    assert np.isnan(result.metrics["sharpe"])


def test_max_drawdown():
    dates = pd.bdate_range("2020-01-01", periods=3)
    idx = pd.MultiIndex.from_product([dates, ["S0"]], names=["datetime", "instrument"])
    scores = pd.Series([1.0, 1.0, 1.0], index=idx)
    returns = pd.Series([0.10, -0.20, 0.05], index=idx)
    result = topk_backtest(scores, returns, k=1, periods_per_year=1)
    assert np.isclose(result.metrics["max_drawdown"], 0.2)


def test_skips_dates_with_too_few_assets():
    idx = make_index(2)
    scores = pd.Series([0.5, 0.4], index=idx)
    returns = pd.Series([0.01, 0.02], index=idx)
    result = topk_backtest(scores, returns, k=5, periods_per_year=1)
    assert len(result.daily_returns) == 0


def test_rejects_bad_k():
    idx = make_index(2)
    scores = pd.Series([0.5, 0.4], index=idx)
    returns = pd.Series([0.01, 0.02], index=idx)
    with pytest.raises(ValueError, match="k"):
        topk_backtest(scores, returns, k=0)
