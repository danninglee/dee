"""Register and use a predictor defined outside the dee package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch.nn as nn

from dee import DiversifiedExpertEnsemble
from dee.predictors import BasePredictor, register_predictor

from synthetic_demo import make_panel


@register_predictor("tiny_mlp")
class TinyMLP(BasePredictor):
    """A minimal user-defined expert."""

    def __init__(self, input_dim, hidden=16, **kwargs):
        super().__init__(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    train = make_panel(60, seed=0)
    valid = make_panel(15, seed=1)
    test = make_panel(15, seed=2)

    model = DiversifiedExpertEnsemble(
        predictor_specs={"tiny_mlp": 4, "linear": 2},
        predictor_kwargs={"tiny_mlp": {"hidden": 64}},
        lr=1e-2,
        batch_size=1024,
        max_steps=100,
        eval_steps=20,
        early_stop_rounds=5,
        seed=0,
        verbose=False,
    )
    model.fit(train, valid)
    scores = model.predict(test)
    print(scores.head())
    print(model.expert_details().to_string(index=False))


if __name__ == "__main__":
    main()
