import numpy as np
import pandas as pd
import pytest

from dee.data import PanelSplit
from dee.ensemble import DiversifiedExpertEnsemble


def make_split(n_dates, n_assets=30, n_features=6, seed=0, name=""):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    idx = pd.MultiIndex.from_product(
        [dates, [f"S{i}" for i in range(n_assets)]], names=["datetime", "instrument"]
    )
    features = pd.DataFrame(
        rng.normal(size=(len(idx), n_features)),
        index=idx,
        columns=[f"f{i}" for i in range(n_features)],
    )
    labels = pd.Series(rng.normal(size=len(idx)), index=idx, name="label")
    return PanelSplit(features, labels, name=name)


@pytest.fixture()
def small_config():
    return dict(
        predictor_specs={
            "linear": 2,
            "ridge": 1,
            "lasso": 1,
            "elastic_net": 1,
            "mlp": 2,
            "soft_tree": 1,
        },
        feature_subsample_ratio=0.5,
        bootstrap_ratio=0.8,
        ic_filter_std=1.5,
        alpha=1e-3,
        lr=1e-2,
        batch_size=128,
        max_steps=40,
        eval_steps=10,
        early_stop_rounds=3,
        mlp_hidden=(8,),
        soft_tree_depth=3,
        seed=0,
        verbose=False,
    )


def test_fit_predict_end_to_end(small_config):
    train, valid, test = make_split(30, seed=0), make_split(10, seed=1), make_split(10, seed=2)
    model = DiversifiedExpertEnsemble(**small_config)
    model.fit(train, valid)
    scores = model.predict(test)
    assert isinstance(scores, pd.Series)
    assert len(scores) == len(test)
    assert scores.index.equals(test.features.index)
    assert np.isfinite(scores.to_numpy()).all()


def test_expert_details_and_diversity_stats(small_config):
    model = DiversifiedExpertEnsemble(**small_config)
    model.fit(make_split(30, seed=0), make_split(10, seed=1))
    details = model.expert_details()
    assert len(details) == 8
    assert set(details.columns) == {
        "expert_id",
        "type",
        "n_features",
        "valid_rank_ic",
        "active",
    }
    stats = model.diversity_stats()
    assert stats["n_total"] == 8
    assert 1 <= stats["n_active"] <= 8


def test_same_seed_is_deterministic(small_config):
    train, valid, test = make_split(30), make_split(10, seed=1), make_split(10, seed=2)
    a = DiversifiedExpertEnsemble(**small_config).fit(train, valid)
    b = DiversifiedExpertEnsemble(**small_config).fit(train, valid)
    assert np.allclose(a.predict(test).to_numpy(), b.predict(test).to_numpy())


def test_save_and_load_roundtrip(tmp_path, small_config):
    train, valid, test = make_split(30), make_split(10, seed=1), make_split(10, seed=2)
    model = DiversifiedExpertEnsemble(**small_config).fit(train, valid)
    model.save(tmp_path / "dee_model")
    restored = DiversifiedExpertEnsemble.load(tmp_path / "dee_model")
    assert np.allclose(
        model.predict(test).to_numpy(), restored.predict(test).to_numpy()
    )


def test_fit_without_validation_keeps_all_experts(small_config):
    model = DiversifiedExpertEnsemble(**dict(small_config, max_steps=10))
    model.fit(make_split(20), None)
    assert model.expert_details()["active"].all()


def test_at_least_one_expert_survives_filter(small_config):
    model = DiversifiedExpertEnsemble(**dict(small_config, ic_filter_std=1000.0))
    model.fit(make_split(30), make_split(10, seed=1))
    assert model.expert_details()["active"].sum() >= 1


def test_rejects_empty_specs():
    with pytest.raises(ValueError, match="predictor_specs"):
        DiversifiedExpertEnsemble(predictor_specs={})


def test_rejects_unknown_predictor():
    with pytest.raises(ValueError, match="unknown predictor"):
        DiversifiedExpertEnsemble(predictor_specs={"nope": 1})


def test_predict_before_fit_raises(small_config):
    model = DiversifiedExpertEnsemble(**small_config)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(make_split(5))


def test_rejects_missing_training_labels(small_config):
    train = make_split(10)
    train = PanelSplit(train.features, None)
    model = DiversifiedExpertEnsemble(**small_config)
    with pytest.raises(ValueError, match="labels"):
        model.fit(train)
