"""Diversified Expert Ensemble."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dee.data import PanelSplit
from dee.metrics import daily_rank_ic, summarize_ic
from dee.predictors import available_predictors, build_predictor
from dee.sampling import bootstrap_indices, feature_subsets

DEFAULT_PREDICTOR_SPECS = {
    "linear": 10,
    "ridge": 2,
    "lasso": 5,
    "elastic_net": 3,
    "mlp": 15,
    "soft_tree": 15,
}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class DiversifiedExpertEnsemble:
    """Heterogeneous ensemble of linear and nonlinear experts.

    Experts are trained in one synchronous loop. Early stopping monitors the
    ensemble-level validation loss, so the collective determines the training
    horizon rather than any individual member (paper Algorithm 1).

    Parameters
    ----------
    predictor_specs : dict, optional
        Mapping from registered predictor name to the number of experts of
        that type. Defaults to the paper allocation, 10 / 2 / 5 / 3 / 15 / 15.
    predictor_kwargs : dict, optional
        Per-type constructor overrides, e.g. ``{"mlp": {"hidden": (64,)}}``.
    feature_subsample_ratio : float
        Fraction of features each expert observes (paper gamma = 0.5).
    bootstrap_ratio : float
        Bootstrap pool size as a fraction of the training rows (paper beta = 0.8789).
    ic_filter_std : float
        Experts with validation RankIC below ``mean - ic_filter_std * std`` are
        removed (paper kappa = 1.5). Set to a non-positive value to disable.
    alpha : float
        Weight of each expert's explicit penalty term.
    lr, weight_decay, batch_size, max_steps, eval_steps, early_stop_rounds
        Optimization settings, matching the paper configuration.
    dropout : float
        Default dropout passed to predictors that accept it.
    mlp_hidden : tuple
        Default hidden sizes for MLP experts.
    soft_tree_depth, soft_tree_max_leaves
        Default soft-tree shape.
    device : str
        ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"cuda:N"``.
    seed : int
        Seed for type assignment, sampling and batch order.
    verbose : bool
        Print progress to stdout.
    """

    def __init__(
        self,
        predictor_specs: Optional[Dict[str, int]] = None,
        predictor_kwargs: Optional[Dict[str, dict]] = None,
        feature_subsample_ratio: float = 0.5,
        bootstrap_ratio: float = 0.8789,
        ic_filter_std: float = 1.5,
        alpha: float = 1e-3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 8192,
        max_steps: int = 8000,
        eval_steps: int = 20,
        early_stop_rounds: int = 50,
        dropout: float = 0.0,
        mlp_hidden: tuple = (32,),
        soft_tree_depth: int = 12,
        soft_tree_max_leaves: int = 4096,
        device: str = "auto",
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        specs = dict(DEFAULT_PREDICTOR_SPECS if predictor_specs is None else predictor_specs)
        if not specs:
            raise ValueError("predictor_specs must contain at least one predictor type")
        for name, count in specs.items():
            if name not in available_predictors():
                raise ValueError(
                    f"unknown predictor {name!r}. "
                    f"Available predictors: {available_predictors()}"
                )
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"count for predictor {name!r} must be a positive int")
        if not 0.0 < float(feature_subsample_ratio) <= 1.0:
            raise ValueError("feature_subsample_ratio must be in (0, 1]")
        if not 0.0 < float(bootstrap_ratio) <= 1.0:
            raise ValueError("bootstrap_ratio must be in (0, 1]")
        for name, value in (
            ("batch_size", batch_size),
            ("max_steps", max_steps),
            ("eval_steps", eval_steps),
            ("early_stop_rounds", early_stop_rounds),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int")

        self.predictor_specs = specs
        self.predictor_kwargs = dict(predictor_kwargs or {})
        self.feature_subsample_ratio = float(feature_subsample_ratio)
        self.bootstrap_ratio = float(bootstrap_ratio)
        self.ic_filter_std = float(ic_filter_std)
        self.alpha = float(alpha)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_steps = int(max_steps)
        self.eval_steps = int(eval_steps)
        self.early_stop_rounds = int(early_stop_rounds)
        self.dropout = float(dropout)
        self.mlp_hidden = tuple(mlp_hidden)
        self.soft_tree_depth = int(soft_tree_depth)
        self.soft_tree_max_leaves = int(soft_tree_max_leaves)
        self.device = _resolve_device(device)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.n_predictors = int(sum(specs.values()))
        self.predictors_: List[torch.nn.Module] = []
        self.optimizers_: List[torch.optim.Optimizer] = []
        self.expert_types_: List[str] = []
        self.feature_indices_: List[np.ndarray] = []
        self._feature_tensors_: List[torch.Tensor] = []
        self.ic_mask_: Optional[np.ndarray] = None
        self.valid_rank_ic_: List[float] = []
        self.history_: Dict[str, list] = {"step": [], "train_loss": [], "valid_loss": []}
        self.best_step_: Optional[int] = None
        self.fitted_ = False

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _assign_expert_types(self, rng: np.random.Generator) -> None:
        types: List[str] = []
        for name, count in self.predictor_specs.items():
            types.extend([name] * count)
        rng.shuffle(types)
        self.expert_types_ = types

    def _build_experts(self) -> None:
        self.predictors_ = []
        self.optimizers_ = []
        self._feature_tensors_ = []
        for expert_type, feature_idx in zip(self.expert_types_, self.feature_indices_):
            kwargs = dict(self.predictor_kwargs.get(expert_type, {}))
            if expert_type == "mlp":
                kwargs.setdefault("hidden", self.mlp_hidden)
                kwargs.setdefault("dropout", self.dropout)
            elif expert_type == "soft_tree":
                kwargs.setdefault("depth", self.soft_tree_depth)
                kwargs.setdefault("max_leaves", self.soft_tree_max_leaves)
            predictor = build_predictor(expert_type, len(feature_idx), **kwargs).to(self.device)
            decay = self.weight_decay if predictor.uses_weight_decay else 0.0
            optimizer = torch.optim.Adam(predictor.parameters(), lr=self.lr, weight_decay=decay)
            self.predictors_.append(predictor)
            self.optimizers_.append(optimizer)
            self._feature_tensors_.append(torch.as_tensor(feature_idx, dtype=torch.long))

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def fit(
        self, train: PanelSplit, valid: Optional[PanelSplit] = None
    ) -> "DiversifiedExpertEnsemble":
        """Train all experts jointly.

        Parameters
        ----------
        train : PanelSplit
            Training panel. Labels are required and must be free of NaN.
        valid : PanelSplit, optional
            Validation panel. When provided, early stopping and the IC filter
            use it. When omitted, training runs for ``max_steps`` and no expert
            is filtered out.
        """
        self._validate_fit_inputs(train, valid)
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            # Seed PyTorch locally so parameter init and dropout are
            # reproducible without perturbing the caller's global RNG state.
            torch.manual_seed(self.seed)
            self._fit(train, valid)
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        return self

    @staticmethod
    def _validate_fit_inputs(train: PanelSplit, valid: Optional[PanelSplit]) -> None:
        if not isinstance(train, PanelSplit):
            raise TypeError("train must be a PanelSplit")
        if valid is not None and not isinstance(valid, PanelSplit):
            raise TypeError("valid must be a PanelSplit or None")
        if train.labels is None:
            raise ValueError("train.labels is required for fitting")
        if train.has_missing_labels:
            raise ValueError("train contains missing labels; call train.dropna() first")
        if valid is not None:
            if valid.labels is None:
                raise ValueError("valid.labels is required when valid is provided")
            if valid.has_missing_labels:
                raise ValueError("valid contains missing labels; call valid.dropna() first")

    def _fit(self, train: PanelSplit, valid: Optional[PanelSplit]) -> None:
        rng = np.random.default_rng(self.seed)
        self._assign_expert_types(rng)
        self.feature_indices_ = feature_subsets(
            train.features.shape[1],
            self.n_predictors,
            self.feature_subsample_ratio,
            rng,
            ensure_coverage=True,
        )
        self.bootstrap_indices_ = bootstrap_indices(
            len(train), self.n_predictors, self.bootstrap_ratio, rng
        )
        self._build_experts()

        x_train = torch.as_tensor(train.features.to_numpy(dtype=np.float32))
        y_train = torch.as_tensor(train.labels.to_numpy(dtype=np.float32)).reshape(-1)

        if valid is not None:
            x_valid = torch.as_tensor(valid.features.to_numpy(dtype=np.float32))
            y_valid = valid.labels.to_numpy(dtype=np.float32).reshape(-1)

        pool_orders = [rng.permutation(pool) for pool in self.bootstrap_indices_]
        pool_positions = [0] * self.n_predictors
        batch_rngs = [
            np.random.default_rng(self.seed + 1000 + i) for i in range(self.n_predictors)
        ]

        best_loss = float("inf")
        best_state = None
        best_step = None
        stale_rounds = 0
        self.history_ = {"step": [], "train_loss": [], "valid_loss": []}

        for step in range(1, self.max_steps + 1):
            step_losses = []
            for i, (predictor, optimizer) in enumerate(
                zip(self.predictors_, self.optimizers_)
            ):
                pool = self.bootstrap_indices_[i]
                if len(pool) <= self.batch_size:
                    batch = pool[
                        batch_rngs[i].integers(0, len(pool), size=self.batch_size)
                    ]
                else:
                    if pool_positions[i] + self.batch_size > len(pool):
                        pool_orders[i] = batch_rngs[i].permutation(pool)
                        pool_positions[i] = 0
                    start = pool_positions[i]
                    batch = pool_orders[i][start : start + self.batch_size]
                    pool_positions[i] = start + self.batch_size

                batch_tensor = torch.as_tensor(batch, dtype=torch.long)
                x_batch = x_train[batch_tensor][:, self._feature_tensors_[i]].to(self.device)
                y_batch = y_train[batch_tensor].to(self.device)

                predictor.train()
                optimizer.zero_grad()
                prediction = predictor(x_batch)
                loss = F.mse_loss(prediction, y_batch) + self.alpha * predictor.penalty()
                loss.backward()
                optimizer.step()
                step_losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(step_losses))

            if valid is not None and (step % self.eval_steps == 0 or step == self.max_steps):
                valid_predictions = self._predict_experts(x_valid)
                ensemble_pred = np.mean(valid_predictions, axis=0)
                valid_loss = float(np.mean((ensemble_pred - y_valid) ** 2))

                self.valid_rank_ic_ = []
                for prediction in valid_predictions:
                    scores = pd.Series(prediction, index=valid.features.index)
                    stats = summarize_ic(daily_rank_ic(scores, valid.labels))
                    self.valid_rank_ic_.append(
                        0.0 if np.isnan(stats["mean"]) else stats["mean"]
                    )

                self.history_["step"].append(step)
                self.history_["train_loss"].append(train_loss)
                self.history_["valid_loss"].append(valid_loss)

                if self.verbose:
                    print(
                        f"[step {step}] train_loss={train_loss:.6f} "
                        f"valid_loss={valid_loss:.6f}"
                    )

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_step = step
                    stale_rounds = 0
                    best_state = {
                        f"expert_{i}": {
                            key: value.detach().cpu().clone()
                            for key, value in predictor.state_dict().items()
                        }
                        for i, predictor in enumerate(self.predictors_)
                    }
                else:
                    stale_rounds += 1
                    if stale_rounds >= self.early_stop_rounds:
                        if self.verbose:
                            print(f"Early stopping at step {step}")
                        break

        if best_state is not None:
            self.load_state_dict(best_state)
        self.best_step_ = best_step
        self._fit_ic_mask(has_valid=valid is not None)
        self.fitted_ = True

    def _predict_experts(self, x: torch.Tensor) -> List[np.ndarray]:
        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for i, predictor in enumerate(self.predictors_):
                predictor.eval()
                chunks = []
                for start in range(0, len(x), 8192):
                    batch = x[start : start + 8192][:, self._feature_tensors_[i]]
                    chunks.append(predictor(batch.to(self.device)).cpu().numpy())
                outputs.append(np.concatenate(chunks) if chunks else np.empty(0))
        return outputs

    def _fit_ic_mask(self, has_valid: bool) -> None:
        if not has_valid or self.ic_filter_std <= 0 or not self.valid_rank_ic_:
            self.ic_mask_ = np.ones(self.n_predictors, dtype=bool)
            return
        scores = np.asarray(self.valid_rank_ic_, dtype=float)
        threshold = scores.mean() - self.ic_filter_std * (scores.std() + 1e-12)
        mask = scores >= threshold
        if not mask.any():
            mask[int(np.argmax(scores))] = True
        self.ic_mask_ = mask

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def predict(self, split: PanelSplit) -> pd.Series:
        """Return the equal-weight ensemble score for every row of ``split``."""
        if not self.fitted_:
            raise RuntimeError("call fit() before predict()")
        if not isinstance(split, PanelSplit):
            raise TypeError("split must be a PanelSplit")
        x = torch.as_tensor(split.features.to_numpy(dtype=np.float32))
        predictions = self._predict_experts(x)
        matrix = np.stack(predictions, axis=1)
        mask = (
            self.ic_mask_
            if self.ic_mask_ is not None
            else np.ones(self.n_predictors, dtype=bool)
        )
        final = matrix[:, mask].mean(axis=1)
        return pd.Series(final, index=split.features.index, name="score")

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def expert_details(self) -> pd.DataFrame:
        """Per-expert diagnostics as a DataFrame."""
        if not self.predictors_:
            raise RuntimeError("call fit() first")
        rows = []
        for i, expert_type in enumerate(self.expert_types_):
            rows.append(
                {
                    "expert_id": i,
                    "type": expert_type,
                    "n_features": int(len(self.feature_indices_[i])),
                    "valid_rank_ic": float(
                        self.valid_rank_ic_[i]
                        if i < len(self.valid_rank_ic_)
                        else float("nan")
                    ),
                    "active": (
                        bool(self.ic_mask_[i])
                        if self.ic_mask_ is not None
                        else True
                    ),
                }
            )
        return pd.DataFrame(rows)

    def diversity_stats(self) -> dict:
        """Aggregate diversity statistics over experts."""
        mask = (
            self.ic_mask_
            if self.ic_mask_ is not None
            else np.ones(self.n_predictors, dtype=bool)
        )
        scores = np.asarray(self.valid_rank_ic_, dtype=float)
        if len(scores) == 0:
            return {
                "rank_ic_mean": float("nan"),
                "rank_ic_std": float("nan"),
                "rank_ic_min": float("nan"),
                "rank_ic_max": float("nan"),
                "n_active": int(mask.sum()),
                "n_total": self.n_predictors,
            }
        return {
            "rank_ic_mean": float(scores.mean()),
            "rank_ic_std": float(scores.std()),
            "rank_ic_min": float(scores.min()),
            "rank_ic_max": float(scores.max()),
            "n_active": int(mask.sum()),
            "n_total": self.n_predictors,
        }

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _config(self) -> dict:
        return {
            "predictor_specs": self.predictor_specs,
            "predictor_kwargs": self.predictor_kwargs,
            "feature_subsample_ratio": self.feature_subsample_ratio,
            "bootstrap_ratio": self.bootstrap_ratio,
            "ic_filter_std": self.ic_filter_std,
            "alpha": self.alpha,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "eval_steps": self.eval_steps,
            "early_stop_rounds": self.early_stop_rounds,
            "dropout": self.dropout,
            "mlp_hidden": list(self.mlp_hidden),
            "soft_tree_depth": self.soft_tree_depth,
            "soft_tree_max_leaves": self.soft_tree_max_leaves,
            "device": str(self.device),
            "seed": self.seed,
            "verbose": self.verbose,
        }

    def state_dict(self) -> dict:
        """Return per-expert weights keyed by ``expert_{i}``."""
        return {f"expert_{i}": p.state_dict() for i, p in enumerate(self.predictors_)}

    def load_state_dict(self, state: dict) -> None:
        for i, predictor in enumerate(self.predictors_):
            predictor.load_state_dict(state[f"expert_{i}"])

    def save(self, path) -> None:
        """Save configuration, weights and diagnostics to a directory."""
        if not self.fitted_:
            raise RuntimeError("call fit() before save()")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self._config(),
            "expert_types": self.expert_types_,
            "feature_indices": [idx.tolist() for idx in self.feature_indices_],
            "valid_rank_ic": [float(x) for x in self.valid_rank_ic_],
            "ic_mask": self.ic_mask_.tolist(),
            "best_step": self.best_step_,
            "history": self.history_,
        }
        (path / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        torch.save(self.state_dict(), path / "weights.pt")

    @classmethod
    def load(cls, path) -> "DiversifiedExpertEnsemble":
        """Load a model previously written by :meth:`save`."""
        path = Path(path)
        payload = json.loads((path / "config.json").read_text(encoding="utf-8"))
        model = cls(**payload["config"])
        model.expert_types_ = list(payload["expert_types"])
        model.feature_indices_ = [
            np.asarray(idx, dtype=int) for idx in payload["feature_indices"]
        ]
        model.valid_rank_ic_ = [float(x) for x in payload["valid_rank_ic"]]
        model.ic_mask_ = np.asarray(payload["ic_mask"], dtype=bool)
        model.best_step_ = payload["best_step"]
        model.history_ = payload["history"]
        model._build_experts()
        state = torch.load(path / "weights.pt", map_location=model.device, weights_only=True)
        model.load_state_dict(state)
        model.fitted_ = True
        return model
