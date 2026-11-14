<h1 align="center">Diversification Across Predictors: Heterogeneous Ensemble Learning for Cross-Sectional Return Prediction</h1>

<h3 align="center">ICAIF 2026 · The 7th ACM International Conference on AI in Finance</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2611.12389"><img src="https://img.shields.io/badge/Paper-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://github.com/danninglee/dee"><img src="https://img.shields.io/badge/Code-181717?style=for-the-badge&logo=github&logoColor=white" alt="Code"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <a href="https://github.com/danninglee">Danning Li</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)
</p>

_DEE diversifies across predictors: heterogeneous linear and nonlinear experts are
trained on independently bootstrap-sampled rows and random feature views; a
validation-based RankIC filter removes degenerate experts, and equal-weight
averaging reduces prediction variance the way diversifying assets reduces
portfolio risk._

## Intro

Cross-sectional return prediction is a low-signal-to-noise learning problem in
which a single predictor's error is dominated by variance. Ensembles built from
homogeneous models share the same inductive bias, so their errors remain strongly
correlated and the achievable variance reduction is limited. We propose the
**Diversified Expert Ensemble (DEE)**, which transfers the principle of portfolio
diversification from assets to predictors. DEE combines linear experts
(unregularized linear, ridge, lasso, and elastic net) with nonlinear experts
(shallow MLPs with batch normalization and soft decision trees with sigmoid
gates). Every expert is trained on independently bootstrap-sampled rows
(ratio β) and a randomly subsampled feature view (ratio γ), so that expert errors
are weakly correlated by construction; averaging them reduces prediction variance
in the same way that diversifying assets reduces portfolio risk. Before the final
equal-weight aggregation, a validation-based information-coefficient filter
(per-date cross-sectional RankIC, threshold parameter κ) removes degenerate
experts. The implementation follows Algorithm 1 of the paper: all experts are
updated inside one synchronous training loop, and early stopping monitors the
ensemble-level validation loss rather than individual experts. The default
configuration uses 50 experts and reproduces the paper setting.

## News

- **2026-11**: The paper is available on [arXiv](https://arxiv.org/abs/2611.12389), and the reference implementation is open-sourced.

## Quick Start

### Environment

```bash
git clone https://github.com/danninglee/dee.git
cd dee

conda create -n dee python=3.10 -y
conda activate dee
pip install -e .
```

Requirements: Python ≥ 3.9, NumPy, pandas, SciPy, PyTorch. No data source is
bundled.

### Synthetic Demo

```bash
python examples/synthetic_demo.py
```

### Fit and Predict

```python
from dee import DiversifiedExpertEnsemble, PanelSplit

train = PanelSplit(X_train, y_train)   # MultiIndex (datetime, instrument)
valid = PanelSplit(X_valid, y_valid)

model = DiversifiedExpertEnsemble().fit(train, valid)
scores = model.predict(PanelSplit(X_test))
```

`PanelSplit` accepts any pandas objects that follow the documented index
convention.

## Data Format

`PanelSplit` requires:

- a two-level `pandas.MultiIndex` with level 0 = `datetime` and level 1 = `instrument`;
- numeric feature columns;
- a numeric label series with the same index (forward returns).

Feature values must be finite. Labels may contain NaN at the end of the sample;
remove those rows explicitly:

```python
split = PanelSplit(features, labels).dropna()     # drop rows with missing labels
split = PanelSplit(features, labels).fillna(0.0)  # explicit zero fill
```

Prediction-only splits do not need labels: `PanelSplit(X_test)`.

## Built-in Predictors

| Name | Description | Explicit penalty |
|---|---|---|
| `linear` | Unregularized linear expert | — |
| `ridge` | Linear expert with L2 penalty | `‖w‖²` |
| `lasso` | Linear expert with L1 penalty | `‖w‖₁` |
| `elastic_net` | Linear expert mixing L1 and L2 | `ρ‖w‖₁ + (1−ρ)‖w‖²` |
| `mlp` | Shallow MLP with batch normalization | — |
| `soft_tree` | Soft decision tree with sigmoid gates | — |

Experts with an explicit penalty disable the optimizer's weight decay to avoid
double regularization.

## Defining a Custom Predictor

```python
import torch.nn as nn
from dee.predictors import BasePredictor, register_predictor

@register_predictor("tiny_mlp")
class TinyMLP(BasePredictor):
    def __init__(self, input_dim, hidden=16, **kwargs):
        super().__init__(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = DiversifiedExpertEnsemble(
    predictor_specs={"tiny_mlp": 4, "linear": 2},
    predictor_kwargs={"tiny_mlp": {"hidden": 64}},
)
```

See `examples/custom_predictor.py` for a complete runnable example.

## Default Hyperparameters

Defaults reproduce the paper configuration.

| Parameter | Value |
|---|---|
| `predictor_specs` | 10 linear, 2 ridge, 5 lasso, 3 elastic net, 15 MLP, 15 soft tree |
| `feature_subsample_ratio` (γ) | 0.5 |
| `bootstrap_ratio` (β) | 0.8789 |
| `ic_filter_std` (κ) | 1.5 |
| `alpha` | 1e-3 |
| `lr` | 1e-3 |
| `weight_decay` | 1e-4 |
| `batch_size` | 8192 |
| `max_steps` | 8000 |
| `eval_steps` | 20 |
| `early_stop_rounds` | 50 |
| `dropout` | 0.0 |
| `mlp_hidden` | (32,) |
| `soft_tree_depth` | 12 |

The full configuration uses 50 experts and depth-12 trees. For a quick CPU run,
reduce `predictor_specs`, `max_steps` and `soft_tree_depth` as in the examples.
`SoftTreePredictor` additionally accepts `max_leaves` (default 4096) as a hard
cap; a depth that would exceed it is truncated with a warning.

## Evaluation and Backtest

```python
from dee.metrics import daily_ic, daily_rank_ic, summarize_ic
from dee.backtest import topk_backtest

ic = summarize_ic(daily_ic(scores, test_labels))
rank_ic = summarize_ic(daily_rank_ic(scores, test_labels))
result = topk_backtest(scores, test_labels, k=50)
print(result.metrics)
```

`topk_backtest` expects **forward** returns aligned to the same dates as the
scores. Annualization uses 242 trading days by default.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Changes from the Internal Research Code

For transparency and reproducibility, we document that the released code differs
from the internal experiment scripts in three substantive ways:

1. The information-coefficient filter now uses per-date cross-sectional RankIC
   rather than a pooled correlation over the whole validation set.
2. Batch sampling walks a shuffled bootstrap pool once per epoch instead of
   resampling with replacement at every step.
3. All Qlib and MLflow dependencies were removed; data enters through
   `PanelSplit`.

## License

The source code in this repository is released under the **MIT License** — see
[`LICENSE`](LICENSE).

**Intellectual property.** The MIT license covers the software only. The paper,
its figures, and other manuscript materials remain under their respective
copyright terms (arXiv / ACM) and are not relicensed under MIT. If you use DEE,
build upon this implementation, or reproduce results from the paper, please cite
the work below.

## Citation

```bibtex
@inproceedings{li2026diversification,
  title     = {Diversification Across Predictors: Heterogeneous Ensemble Learning
               for Cross-Sectional Return Prediction},
  author    = {Li, Danning},
  booktitle = {Proceedings of the 7th ACM International Conference on AI in Finance},
  year      = {2026},
  note      = {arXiv:2611.12389}
}
```

## Contact

Danning Li — dli381@connect.hkust-gz.edu.cn
