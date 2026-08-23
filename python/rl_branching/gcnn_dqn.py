from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .gcnn_model import BipartiteGCNNQNetwork
from .graph_features import GraphState
from .graph_replay import PrioritizedBatch
from .prim_bias import apply_prim_bias, stable_argmax_with_scores


def graph_state_tensors(state: GraphState, device: torch.device):
    return (
        torch.tensor(state.row_features, dtype=torch.float32, device=device),
        torch.tensor(state.variable_features, dtype=torch.float32, device=device),
        torch.tensor(state.edge_indices, dtype=torch.long, device=device),
        torch.tensor(state.edge_features, dtype=torch.float32, device=device),
        torch.tensor(state.global_features, dtype=torch.float32, device=device),
        torch.tensor(state.variable_categories, dtype=torch.float32, device=device),
        torch.tensor(state.row_categories, dtype=torch.float32, device=device),
        torch.tensor(state.actions, dtype=torch.long, device=device),
    )


def stable_graph_argmax(
    q_values: np.ndarray,
    state: GraphState,
    *,
    lambda_prim: float = 0.0,
) -> int:
    values = np.asarray(q_values, dtype=np.float64)
    if values.shape != (state.candidate_count,) or not np.isfinite(values).all():
        raise ValueError("graph Q values must be finite and align with candidates")
    if lambda_prim != 0.0 and state.variable_names:
        # solution_value is feature index 8 in ECOLE_VARIABLE_FEATURE_NAMES
        solution_values = state.variable_features[:, 8]
        values, _ = apply_prim_bias(
            values,
            state.candidate_names,
            variable_names=state.variable_names,
            solution_values=solution_values,
            lambda_prim=lambda_prim,
        )
    return stable_argmax_with_scores(values, state.candidate_names, state.actions)


@dataclass(frozen=True)
class GraphUpdateMetrics:
    loss: float
    td_error: float
    q_mean: float
    q_std: float
    gradient_norm: float
    priorities: np.ndarray


class GraphDoubleDQNLearner:
    def __init__(
        self,
        online: BipartiteGCNNQNetwork,
        target: BipartiteGCNNQNetwork,
        device: torch.device,
        learning_rate: float,
        gamma: float,
        gradient_clip: float,
        target_tau: float,
        hl_gauss_sigma: float = 0.75,
    ) -> None:
        self.online = online.to(device)
        self.target = target.to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.device = device
        self.gamma = float(gamma)
        self.gradient_clip = float(gradient_clip)
        self.target_tau = float(target_tau)
        self.hl_gauss_sigma = float(hl_gauss_sigma)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(learning_rate))
        self.gradient_step = 0

    def q_values(self, state: GraphState) -> np.ndarray:
        self.online.eval()
        with torch.no_grad():
            values = self.online(*graph_state_tensors(state, self.device))
        result = values.detach().cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("GCNN Q values contain NaN or Inf")
        return result

    def select_action(
        self,
        state: GraphState,
        epsilon: float,
        rng: np.random.Generator,
        boltzmann_temperature: float = 0.0,
        lambda_prim: float = 0.0,
    ) -> tuple[int, int, float, np.ndarray]:
        q_values = self.q_values(state)
        if rng.random() < epsilon:
            position = int(rng.integers(state.candidate_count))
        elif boltzmann_temperature > 0.0:
            centered = (q_values - q_values.max()) / boltzmann_temperature
            probabilities = np.exp(centered)
            probabilities /= probabilities.sum()
            position = int(rng.choice(state.candidate_count, p=probabilities))
        else:
            position = stable_graph_argmax(q_values, state, lambda_prim=lambda_prim)
        rank = float(1 + np.count_nonzero(q_values > q_values[position] + 1.0e-7))
        return int(state.actions[position]), position, rank, q_values

    def _target_histogram(self, target: torch.Tensor) -> torch.Tensor:
        centers = self.online.z_centers
        minimum = torch.pow(torch.tensor(2.0, device=target.device), centers[0])
        maximum = torch.pow(torch.tensor(2.0, device=target.device), centers[-1])
        transformed = torch.log2(torch.clamp(-target, min=minimum, max=maximum))
        logits = -0.5 * ((centers - transformed) / self.hl_gauss_sigma) ** 2
        return torch.softmax(logits, dim=0)

    def _soft_update(self) -> None:
        with torch.no_grad():
            for target_parameter, online_parameter in zip(
                self.target.parameters(), self.online.parameters()
            ):
                target_parameter.mul_(1.0 - self.target_tau)
                target_parameter.add_(online_parameter, alpha=self.target_tau)

    def update(self, batch: PrioritizedBatch) -> GraphUpdateMetrics:
        predicted_values = []
        target_values = []
        sample_losses = []
        self.online.train()

        for experience in batch.experiences:
            state_tensors = graph_state_tensors(experience.state, self.device)
            q_values = self.online(*state_tensors)
            predicted = q_values[experience.action_position]
            predicted_values.append(predicted)

            with torch.no_grad():
                bootstrap = 0.0
                if experience.bootstrap_mask and experience.next_state is not None:
                    next_tensors = graph_state_tensors(experience.next_state, self.device)
                    online_next = self.online(*next_tensors)
                    next_position = stable_graph_argmax(
                        online_next.detach().cpu().numpy(), experience.next_state
                    )
                    target_next = self.target(*next_tensors)[next_position]
                    bootstrap = (
                        self.gamma**experience.n_steps
                        * float(experience.bootstrap_mask)
                        * target_next
                    )
                target_value = torch.as_tensor(
                    experience.reward, dtype=torch.float32, device=self.device
                ) + bootstrap
            target_values.append(target_value)

            if self.online.distributional_bins == 1:
                sample_losses.append(
                    nn.functional.smooth_l1_loss(predicted, target_value, reduction="none")
                )
            else:
                logits = self.online.logits(*state_tensors)[experience.action_position]
                histogram = self._target_histogram(target_value)
                sample_losses.append(
                    -(histogram * torch.log_softmax(logits, dim=0)).sum()
                )

        weights = torch.as_tensor(batch.weights, dtype=torch.float32, device=self.device)
        if weights.shape[0] != len(sample_losses):
            raise ValueError("PER weights must align with the logical batch")
        batch_size = len(sample_losses)
        self.optimizer.zero_grad(set_to_none=True)
        weighted_losses = []
        for sample_loss, weight in zip(sample_losses, weights):
            weighted = sample_loss * weight / batch_size
            if not torch.isfinite(weighted):
                raise FloatingPointError("GCNN DQN loss is NaN or Inf")
            weighted.backward()
            weighted_losses.append(weighted.detach())
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.gradient_clip
        )
        self.optimizer.step()
        self.gradient_step += 1
        self._soft_update()

        predicted_tensor = torch.stack([value.detach() for value in predicted_values])
        target_tensor = torch.stack(
            [
                value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
                for value in target_values
            ]
        )
        td_errors = (target_tensor - predicted_tensor).abs()
        return GraphUpdateMetrics(
            loss=float(torch.stack(weighted_losses).sum()),
            td_error=float(td_errors.mean()),
            q_mean=float(predicted_tensor.mean()),
            q_std=float(predicted_tensor.std(unbiased=False)),
            gradient_norm=float(gradient_norm),
            priorities=td_errors.detach().cpu().numpy(),
        )
