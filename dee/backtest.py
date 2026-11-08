"""Equal-weight top-K portfolio backtest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dee.metrics import align_scores_returns


@dataclass
class BacktestResult:
    """Container for backtest metrics and the daily return series."""

    metrics: dict
    daily_returns: pd.Series


def topk_backtest(
    scores: pd.Series,
    returns: pd.Series,
    k: int = 50,
    periods_per_year: int = 242,
) -> BacktestResult:
    """Run a daily top-K equal-weight long portfolio.

    Parameters
    ----------
    scores : pandas.Series
        Predicted scores with a ``(datetime, instrument)`` MultiIndex.
    returns : pandas.Series
        **Forward** returns aligned to the same dates as ``scores``. Passing
        contemporaneous returns leaks future information.
    k : int
        Number of names held each day.
    periods_per_year : int
        Annualization factor. 242 trading days by default.

    Returns
    -------
    BacktestResult
        Annualized return, annualized volatility, Sharpe ratio and maximum
        drawdown, plus the daily return series.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive int, got {k!r}")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    frame = align_scores_returns(scores, returns)

    def topk_return(group: pd.DataFrame) -> float:
        if len(group) < k:
            return np.nan
        return float(group.nlargest(k, "score")["ret"].mean())

    daily = frame.groupby(level=0).apply(topk_return).dropna()
    daily.index.name = "datetime"
    daily.name = "daily_return"

    if len(daily) == 0:
        empty = {
            "annual_return": float("nan"),
            "annual_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }
        return BacktestResult(metrics=empty, daily_returns=daily)

    annual_return = float(daily.mean() * periods_per_year)
    annual_vol = (
        float(daily.std(ddof=1) * np.sqrt(periods_per_year)) if len(daily) > 1 else 0.0
    )
    sharpe = float(annual_return / annual_vol) if annual_vol > 0 else float("nan")

    cumulative = (1.0 + daily).cumprod()
    drawdown = (cumulative.cummax() - cumulative) / cumulative.cummax()
    max_drawdown = float(drawdown.max())

    return BacktestResult(
        metrics={
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
        },
        daily_returns=daily,
    )
