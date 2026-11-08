import numpy as np
import pandas as pd
import pytest

from dee.metrics import align_scores_returns, daily_ic, daily_rank_ic, summarize_ic


def make_series(values, date="2020-01-01", prefix="S"):
    idx = pd.MultiIndex.from_product(
        [[pd.Timestamp(date)], [f"{prefix}{i}" for i in range(len(values))]],
        names=["datetime", "instrument"],
    )
    return pd.Series(values, index=idx, dtype=float)


def test_align_drops_missing_rows():
    scores = make_series([1.0, 2.0, np.nan])
    returns = make_series([0.1, 0.2, 0.3])
    assert len(align_scores_returns(scores, returns)) == 2


def test_align_rejects_flat_index():
    with pytest.raises(ValueError, match="MultiIndex"):
        align_scores_returns(pd.Series([1.0, 2.0]), pd.Series([0.1, 0.2]))


def test_ic_is_one_for_perfect_linear_relation():
    scores = make_series([1.0, 2.0, 3.0, 4.0, 5.0])
    returns = make_series([2.0, 4.0, 6.0, 8.0, 10.0])
    ic = daily_ic(scores, returns)
    assert len(ic) == 1
    assert np.isclose(ic.iloc[0], 1.0)


def test_rank_ic_invariant_to_monotone_transform():
    rng = np.random.default_rng(0)
    scores = make_series(rng.normal(size=20))
    returns = make_series(rng.normal(size=20))
    transformed = make_series(np.exp(scores.to_numpy()))
    assert np.isclose(
        daily_rank_ic(scores, returns).iloc[0],
        daily_rank_ic(transformed, returns).iloc[0],
    )


def test_min_obs_skips_small_dates():
    small = make_series([1.0, 2.0])
    returns = make_series([0.1, 0.2])
    assert len(daily_ic(small, returns, min_obs=5)) == 0


def test_summarize_ic():
    stats = summarize_ic(pd.Series([0.1, 0.2, 0.3]))
    assert np.isclose(stats["mean"], 0.2)
    assert np.isclose(stats["std"], 0.1)
    assert np.isclose(stats["icir"], 2.0)


def test_summarize_ic_empty():
    stats = summarize_ic(pd.Series(dtype=float))
    assert np.isnan(stats["mean"])
    assert np.isnan(stats["icir"])
