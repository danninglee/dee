"""Cross-sectional information coefficient metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_scores_returns(scores: pd.Series, returns: pd.Series) -> pd.DataFrame:
    """Align scores and returns on a two-level ``(datetime, instrument)`` index.

    Rows where either input is missing are dropped. Both inputs must already
    carry the same MultiIndex structure.
    """
    if not isinstance(scores, pd.Series) or not isinstance(returns, pd.Series):
        raise TypeError("scores and returns must be pandas.Series")
    for name, series in (("scores", scores), ("returns", returns)):
        if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
            raise ValueError(
                f"{name}.index must be a MultiIndex with levels (datetime, instrument)"
            )
    return pd.DataFrame({"score": scores, "ret": returns}).dropna()


def _daily_corr(frame: pd.DataFrame, min_obs: int, rank: bool) -> pd.Series:
    if min_obs < 2:
        raise ValueError("min_obs must be at least 2")

    def corr(group: pd.DataFrame) -> float:
        if len(group) < min_obs:
            return np.nan
        left, right = group["score"], group["ret"]
        if rank:
            left, right = left.rank(), right.rank()
        return left.corr(right)

    result = frame.groupby(level=0).apply(corr).dropna()
    result.index.name = "datetime"
    return result


def daily_ic(scores: pd.Series, returns: pd.Series, min_obs: int = 5) -> pd.Series:
    """Per-date cross-sectional Pearson correlation between scores and returns."""
    result = _daily_corr(align_scores_returns(scores, returns), min_obs=min_obs, rank=False)
    result.name = "ic"
    return result


def daily_rank_ic(scores: pd.Series, returns: pd.Series, min_obs: int = 5) -> pd.Series:
    """Per-date cross-sectional Spearman correlation between scores and returns."""
    result = _daily_corr(align_scores_returns(scores, returns), min_obs=min_obs, rank=True)
    result.name = "rank_ic"
    return result


def summarize_ic(daily: pd.Series) -> dict:
    """Summarize a daily IC series as mean, standard deviation and ICIR."""
    daily = pd.Series(daily).dropna()
    if len(daily) == 0:
        return {"mean": float("nan"), "std": float("nan"), "icir": float("nan")}
    mean = float(daily.mean())
    std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    icir = float(mean / std) if std > 0 else float("nan")
    return {"mean": mean, "std": std, "icir": icir}
