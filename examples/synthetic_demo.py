"""End-to-end DEE demo on synthetic panel data (CPU friendly)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dee import DiversifiedExpertEnsemble, PanelSplit
from dee.backtest import topk_backtest
from dee.metrics import daily_ic, daily_rank_ic, summarize_ic


_BETA_CACHE = {}


def _true_beta(n_features):
    """Shared ground-truth coefficients so every split has the same signal."""
    if n_features not in _BETA_CACHE:
        rng = np.random.default_rng(12345)
        _BETA_CACHE[n_features] = rng.normal(size=n_features)
    return _BETA_CACHE[n_features]


def make_panel(n_dates, n_assets=80, n_features=12, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    instruments = [f"S{i:03d}" for i in range(n_assets)]
    index = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    features = rng.normal(size=(len(index), n_features))
    signal = features @ _true_beta(n_features)
    signal = signal / signal.std()
    # Realistic daily-return scale: weak signal (0.5%), 2% idiosyncratic noise
    # and a 1% common market factor shared by all names on a given date.
    market = np.repeat(rng.normal(scale=0.01, size=n_dates), n_assets)
    labels = 0.005 * signal + rng.normal(scale=0.02, size=len(index)) + market
    return PanelSplit(
        pd.DataFrame(
            features, index=index, columns=[f"f{i}" for i in range(n_features)]
        ),
        pd.Series(labels, index=index, name="label"),
    )


def main():
    train = make_panel(120, seed=0)
    valid = make_panel(40, seed=1)
    test = make_panel(40, seed=2)

    model = DiversifiedExpertEnsemble(
        predictor_specs={
            "linear": 3,
            "ridge": 1,
            "lasso": 1,
            "elastic_net": 1,
            "mlp": 3,
            "soft_tree": 2,
        },
        feature_subsample_ratio=0.5,
        bootstrap_ratio=0.8789,
        ic_filter_std=1.5,
        lr=1e-2,
        batch_size=1024,
        max_steps=300,
        eval_steps=20,
        early_stop_rounds=10,
        mlp_hidden=(32,),
        soft_tree_depth=4,
        seed=42,
        verbose=True,
    )
    model.fit(train, valid)

    scores = model.predict(test)
    ic = summarize_ic(daily_ic(scores, test.labels))
    rank_ic = summarize_ic(daily_rank_ic(scores, test.labels))
    result = topk_backtest(scores, test.labels, k=20)

    print("\n=== Test metrics ===")
    print(f"IC      : {ic['mean']:.4f} +/- {ic['std']:.4f}")
    print(f"RankIC  : {rank_ic['mean']:.4f} +/- {rank_ic['std']:.4f}")
    print(f"AnnRet  : {result.metrics['annual_return']:.2%}")
    print(f"Sharpe  : {result.metrics['sharpe']:.3f}")
    print(f"MaxDD   : {result.metrics['max_drawdown']:.2%}")
    print("\n=== Experts ===")
    print(model.expert_details().to_string(index=False))


if __name__ == "__main__":
    main()
