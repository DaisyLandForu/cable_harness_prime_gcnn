"""Small aviation-independent one-round MILP bipartite GCNN baseline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ..config import StrictConfigError, load_yaml_mapping
from ..contracts import canonical_json
from ..solver.bipartite_observation import (
    CONSTRAINT_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    MILP_BIPARTITE_V1,
    VARIABLE_FEATURE_NAMES,
    MilpBipartiteState,
)


def _require_keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown or missing:
        raise StrictConfigError(f"{label} fields mismatch: missing={missing}, unknown={unknown}")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise StrictConfigError(f"{label} must be a string-keyed mapping")
    return dict(value)


def load_b0_config(path: Path | str) -> dict[str, Any]:
    raw = load_yaml_mapping(path)
    _require_keys(
        raw,
        {
            "schema_version", "stage", "model_id", "bipartite_schema_id",
            "solver_stack_id", "protocol_id", "split", "split_policy",
            "solver_seed", "model_seed", "device", "cpu_threads",
            "architecture", "snapshot", "gate",
        },
        "S04 B0 config",
    )
    if raw["schema_version"] != 1 or raw["stage"] != "S04":
        raise StrictConfigError("S04 B0 config schema_version/stage mismatch")
    if raw["model_id"] != "b0_milp_gcnn_v1":
        raise StrictConfigError("S04 model_id must remain b0_milp_gcnn_v1")
    if raw["bipartite_schema_id"] != MILP_BIPARTITE_V1.schema_id:
        raise StrictConfigError("S04 config must use milp_bipartite_v1")
    if raw["solver_stack_id"] != "scip804-ecole081-pyscipopt430":
        raise StrictConfigError("S04 config changed the frozen solver stack")
    if raw["protocol_id"] != "P1" or raw["split"] != "train":
        raise StrictConfigError("S04 snapshots must use frozen P1/train")
    if raw["solver_seed"] != 0 or raw["model_seed"] not in {101, 202, 303, 404, 505}:
        raise StrictConfigError("S04 solver/model seed is outside the registered set")
    if raw["device"] != "cpu" or int(raw["cpu_threads"]) != 1:
        raise StrictConfigError("S04 is a one-thread CPU-only stage")

    architecture = _mapping(raw["architecture"], "architecture")
    _require_keys(
        architecture,
        {
            "variable_features", "constraint_features", "edge_features",
            "embedding_dim", "hidden_dim", "message_passing_rounds", "aggregation",
        },
        "architecture",
    )
    expected_widths = {
        "variable_features": len(VARIABLE_FEATURE_NAMES),
        "constraint_features": len(CONSTRAINT_FEATURE_NAMES),
        "edge_features": len(EDGE_FEATURE_NAMES),
    }
    if any(int(architecture[key]) != value for key, value in expected_widths.items()):
        raise StrictConfigError("S04 B0 feature widths must remain 19/5/1")
    if int(architecture["message_passing_rounds"]) != 1 or architecture["aggregation"] != "sum":
        raise StrictConfigError("S04 B0 must use one standard sum-aggregation round")
    if int(architecture["embedding_dim"]) <= 0 or int(architecture["hidden_dim"]) <= 0:
        raise StrictConfigError("S04 architecture widths must be positive")

    snapshot = _mapping(raw["snapshot"], "snapshot")
    _require_keys(
        snapshot,
        {
            "family", "n_nodes", "n_terminals", "generator_seed", "max_states",
            "action_policy", "warmup_iterations", "timed_iterations",
        },
        "snapshot",
    )
    if snapshot["action_policy"] != "minimum_edge_id":
        raise StrictConfigError("S04 snapshot action policy changed")
    for key in ("n_nodes", "n_terminals", "max_states", "timed_iterations"):
        if int(snapshot[key]) <= 0:
            raise StrictConfigError(f"snapshot.{key} must be positive")
    if int(snapshot["warmup_iterations"]) < 0:
        raise StrictConfigError("snapshot.warmup_iterations must be non-negative")

    gate = _mapping(raw["gate"], "gate")
    _require_keys(
        gate,
        {
            "max_full_closure_logit_error", "required_argmax_agreement",
            "required_action_mapping_rate", "require_all_finite", "min_snapshots",
        },
        "gate",
    )
    if float(gate["max_full_closure_logit_error"]) != 1.0e-5:
        raise StrictConfigError("S04 parity threshold must remain 1e-5")
    if float(gate["required_argmax_agreement"]) != 1.0:
        raise StrictConfigError("S04 requires 100% argmax agreement")
    if float(gate["required_action_mapping_rate"]) != 1.0:
        raise StrictConfigError("S04 requires 100% action mapping")
    if gate["require_all_finite"] is not True:
        raise StrictConfigError("S04 must reject non-finite features/logits")
    if int(gate["min_snapshots"]) < 1:
        raise StrictConfigError("S04 min_snapshots must be positive")
    return raw


def config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(config)).encode("utf-8")).hexdigest()


class MilpBipartiteGCNN(nn.Module):
    """One variable→constraint→variable propagation round with no global state."""

    def __init__(self, *, embedding_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("model widths must be positive")
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.variable_encoder = self._mlp(len(VARIABLE_FEATURE_NAMES), hidden_dim, embedding_dim)
        self.constraint_encoder = self._mlp(
            len(CONSTRAINT_FEATURE_NAMES), hidden_dim, embedding_dim
        )
        self.edge_encoder = self._mlp(len(EDGE_FEATURE_NAMES), hidden_dim, embedding_dim)
        self.variable_to_constraint = self._mlp(2 * embedding_dim, hidden_dim, embedding_dim)
        self.constraint_update = self._mlp(2 * embedding_dim, hidden_dim, embedding_dim)
        self.constraint_to_variable = self._mlp(2 * embedding_dim, hidden_dim, embedding_dim)
        self.variable_update = self._mlp(2 * embedding_dim, hidden_dim, embedding_dim)
        self.output = self._mlp(embedding_dim, hidden_dim, 1, final_relu=False)

    @staticmethod
    def _mlp(
        input_dim: int, hidden_dim: int, output_dim: int, *, final_relu: bool = True
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ]
        if final_relu:
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    @staticmethod
    def _sum_aggregate(
        messages: torch.Tensor, indices: torch.Tensor, output_size: int
    ) -> torch.Tensor:
        result = torch.zeros(
            (output_size, messages.shape[1]), dtype=messages.dtype, device=messages.device
        )
        result.index_add_(0, indices, messages)
        return result

    def forward(
        self,
        constraint_features: torch.Tensor,
        variable_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        constraints = self.constraint_encoder(constraint_features)
        variables = self.variable_encoder(variable_features)
        edges = self.edge_encoder(edge_features)
        constraint_indices = edge_indices[0]
        variable_indices = edge_indices[1]

        messages = self.variable_to_constraint(
            torch.cat((variables[variable_indices], edges), dim=1)
        )
        aggregated = self._sum_aggregate(messages, constraint_indices, constraints.shape[0])
        constraints = self.constraint_update(torch.cat((constraints, aggregated), dim=1))

        messages = self.constraint_to_variable(
            torch.cat((constraints[constraint_indices], edges), dim=1)
        )
        aggregated = self._sum_aggregate(messages, variable_indices, variables.shape[0])
        variables = self.variable_update(torch.cat((variables, aggregated), dim=1))
        return self.output(variables[candidate_indices]).squeeze(1)


def state_tensors(
    state: MilpBipartiteState, *, device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state.validate()
    return (
        torch.tensor(np.asarray(state.constraint_features), dtype=torch.float32, device=device),
        torch.tensor(np.asarray(state.variable_features), dtype=torch.float32, device=device),
        torch.tensor(np.asarray(state.edge_indices), dtype=torch.int64, device=device),
        torch.tensor(np.asarray(state.edge_features), dtype=torch.float32, device=device),
        torch.tensor(np.asarray(state.candidate_indices), dtype=torch.int64, device=device),
    )


def score_state(
    model: MilpBipartiteGCNN,
    state: MilpBipartiteState,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    return model(*state_tensors(state, device=device))


def parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
