from pathlib import Path

import numpy as np
import torch
from torch import nn

from .candidate_features import AVIATION_VARIABLE_CATEGORIES, ECOLE_VARIABLE_FEATURE_NAMES
from .graph_features import AVIATION_CONSTRAINT_CATEGORIES
from .observation import EDGE_FEATURE_NAMES, EXTENDED_ROW_FEATURE_NAMES, GLOBAL_FEATURE_NAMES


class BipartiteGCNNQNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        distributional_bins: int = 1,
        z_min: float = -1.0,
        z_max: float = 12.0,
        use_aviation_categories: bool = True,
        use_global_features: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.distributional_bins = int(distributional_bins)
        self.use_aviation_categories = bool(use_aviation_categories)
        self.use_global_features = bool(use_global_features)
        if self.distributional_bins <= 0:
            raise ValueError("distributional_bins must be positive")

        variable_width = len(ECOLE_VARIABLE_FEATURE_NAMES) + len(AVIATION_VARIABLE_CATEGORIES)
        row_width = len(EXTENDED_ROW_FEATURE_NAMES) + len(AVIATION_CONSTRAINT_CATEGORIES)
        edge_width = len(EDGE_FEATURE_NAMES)
        global_width = len(GLOBAL_FEATURE_NAMES)

        self.variable_encoder = self._mlp(variable_width, hidden_dim, embedding_dim)
        self.row_encoder = self._mlp(row_width, hidden_dim, embedding_dim)
        self.edge_encoder = self._mlp(edge_width, hidden_dim, embedding_dim)
        self.global_encoder = self._mlp(global_width, hidden_dim, embedding_dim)
        self.variable_to_row = self._mlp(2 * embedding_dim, hidden_dim, embedding_dim)
        self.row_update = self._mlp(2 * embedding_dim, hidden_dim, embedding_dim)
        self.row_to_variable = self._mlp(2 * embedding_dim, hidden_dim, embedding_dim)
        self.variable_update = self._mlp(2 * embedding_dim, hidden_dim, embedding_dim)
        self.q_head = self._mlp(
            2 * embedding_dim,
            hidden_dim,
            self.distributional_bins,
            final_relu=False,
        )

        self.register_buffer("variable_mean", torch.zeros(len(ECOLE_VARIABLE_FEATURE_NAMES)))
        self.register_buffer("variable_std", torch.ones(len(ECOLE_VARIABLE_FEATURE_NAMES)))
        self.register_buffer("row_mean", torch.zeros(len(EXTENDED_ROW_FEATURE_NAMES)))
        self.register_buffer("row_std", torch.ones(len(EXTENDED_ROW_FEATURE_NAMES)))
        self.register_buffer("edge_mean", torch.zeros(len(EDGE_FEATURE_NAMES)))
        self.register_buffer("edge_std", torch.ones(len(EDGE_FEATURE_NAMES)))
        self.register_buffer("global_mean", torch.zeros(len(GLOBAL_FEATURE_NAMES)))
        self.register_buffer("global_std", torch.ones(len(GLOBAL_FEATURE_NAMES)))
        self.register_buffer(
            "z_centers",
            torch.linspace(float(z_min), float(z_max), self.distributional_bins),
        )

    @staticmethod
    def _mlp(
        input_width: int,
        hidden_width: int,
        output_width: int,
        final_relu: bool = True,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Linear(input_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, output_width),
        ]
        if final_relu:
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    @staticmethod
    def _mean_aggregate(
        messages: torch.Tensor,
        indices: torch.Tensor,
        output_size: int,
    ) -> torch.Tensor:
        aggregated = torch.zeros(
            (output_size, messages.size(1)),
            dtype=messages.dtype,
            device=messages.device,
        )
        aggregated.index_add_(0, indices, messages)
        counts = torch.zeros(output_size, dtype=messages.dtype, device=messages.device)
        counts.index_add_(0, indices, torch.ones_like(indices, dtype=messages.dtype))
        return aggregated / counts.clamp_min(1.0).unsqueeze(1)

    def _candidate_logits(
        self,
        row_features: torch.Tensor,
        variable_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        global_features: torch.Tensor,
        variable_categories: torch.Tensor,
        row_categories: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_aviation_categories:
            variable_categories = torch.zeros_like(variable_categories)
            row_categories = torch.zeros_like(row_categories)
        if not self.use_global_features:
            global_features = torch.zeros_like(global_features)
        normalized_variables = (variable_features - self.variable_mean) / self.variable_std
        normalized_rows = (row_features - self.row_mean) / self.row_std
        normalized_edges = (edge_features - self.edge_mean) / self.edge_std
        normalized_global = (global_features - self.global_mean) / self.global_std

        variables = self.variable_encoder(
            torch.cat((normalized_variables, variable_categories), dim=1)
        )
        rows = self.row_encoder(torch.cat((normalized_rows, row_categories), dim=1))
        edges = self.edge_encoder(normalized_edges)
        row_indices = edge_indices[0]
        variable_indices = edge_indices[1]

        variable_messages = self.variable_to_row(
            torch.cat((variables[variable_indices], edges), dim=1)
        )
        row_messages = self._mean_aggregate(
            variable_messages, row_indices, rows.size(0)
        )
        rows = self.row_update(torch.cat((rows, row_messages), dim=1))

        row_messages = self.row_to_variable(
            torch.cat((rows[row_indices], edges), dim=1)
        )
        variable_messages = self._mean_aggregate(
            row_messages, variable_indices, variables.size(0)
        )
        variables = self.variable_update(
            torch.cat((variables, variable_messages), dim=1)
        )

        global_embedding = self.global_encoder(normalized_global.reshape(1, -1))
        candidates = variables[candidate_indices]
        repeated_global = global_embedding.expand(candidates.size(0), -1)
        return self.q_head(torch.cat((candidates, repeated_global), dim=1))

    @torch.jit.export
    def logits(
        self,
        row_features: torch.Tensor,
        variable_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        global_features: torch.Tensor,
        variable_categories: torch.Tensor,
        row_categories: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self._candidate_logits(
            row_features,
            variable_features,
            edge_indices,
            edge_features,
            global_features,
            variable_categories,
            row_categories,
            candidate_indices,
        )

    def forward(
        self,
        row_features: torch.Tensor,
        variable_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        global_features: torch.Tensor,
        variable_categories: torch.Tensor,
        row_categories: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._candidate_logits(
            row_features,
            variable_features,
            edge_indices,
            edge_features,
            global_features,
            variable_categories,
            row_categories,
            candidate_indices,
        )
        if self.distributional_bins == 1:
            return logits.squeeze(1)
        probabilities = torch.softmax(logits, dim=1)
        transformed = (probabilities * self.z_centers).sum(dim=1)
        return -torch.pow(torch.full_like(transformed, 2.0), transformed)

    def set_normalization(self, statistics: dict[str, np.ndarray]) -> None:
        with torch.no_grad():
            for name in (
                "variable_mean",
                "variable_std",
                "row_mean",
                "row_std",
                "edge_mean",
                "edge_std",
                "global_mean",
                "global_std",
            ):
                destination = getattr(self, name)
                source = torch.as_tensor(
                    statistics[name], dtype=destination.dtype, device=destination.device
                )
                destination.copy_(source)


def export_gcnn_torchscript(model: BipartiteGCNNQNetwork, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.jit.script(model).save(str(path))
