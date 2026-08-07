from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .candidate_features import AVIATION_VARIABLE_CATEGORIES, ECOLE_VARIABLE_FEATURE_NAMES
from .observation import GLOBAL_FEATURE_NAMES


class CandidateQNetwork(nn.Module):
    def __init__(self, hidden_sizes: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        input_size = (
            len(ECOLE_VARIABLE_FEATURE_NAMES)
            + len(GLOBAL_FEATURE_NAMES)
            + len(AVIATION_VARIABLE_CATEGORIES)
        )
        layers: list[nn.Module] = []
        previous = input_size
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous, int(width)), nn.ReLU()))
            previous = int(width)
        layers.append(nn.Linear(previous, 1))
        self.mlp = nn.Sequential(*layers)

        self.register_buffer("variable_mean", torch.zeros(len(ECOLE_VARIABLE_FEATURE_NAMES)))
        self.register_buffer("variable_std", torch.ones(len(ECOLE_VARIABLE_FEATURE_NAMES)))
        self.register_buffer("global_mean", torch.zeros(len(GLOBAL_FEATURE_NAMES)))
        self.register_buffer("global_std", torch.ones(len(GLOBAL_FEATURE_NAMES)))

    def forward(
        self,
        variable_features: torch.Tensor,
        global_features: torch.Tensor,
        category_features: torch.Tensor,
    ) -> torch.Tensor:
        normalized_variables = (variable_features - self.variable_mean) / self.variable_std
        normalized_globals = (global_features - self.global_mean) / self.global_std
        features = torch.cat((normalized_variables, normalized_globals, category_features), dim=1)
        return self.mlp(features).squeeze(1)

    def set_normalization(self, statistics: dict[str, np.ndarray]) -> None:
        with torch.no_grad():
            for name in ("variable_mean", "variable_std", "global_mean", "global_std"):
                destination = getattr(self, name)
                source = torch.as_tensor(statistics[name], dtype=destination.dtype, device=destination.device)
                destination.copy_(source)


def export_torchscript(model: CandidateQNetwork, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(str(path))
