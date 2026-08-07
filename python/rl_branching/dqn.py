from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .candidate_features import CandidateState
from .candidate_model import CandidateQNetwork
from .replay import ReplayExperience


def _state_tensors(state: CandidateState, device: torch.device):
    count = state.candidate_count
    variables = torch.tensor(state.variable_features, dtype=torch.float32, device=device)
    globals_ = torch.tensor(state.global_features, dtype=torch.float32, device=device)
    globals_ = globals_.unsqueeze(0).expand(count, -1)
    categories = torch.tensor(state.category_features, dtype=torch.float32, device=device)
    return variables, globals_, categories


def stable_argmax_position(
    q_values: np.ndarray,
    state: CandidateState,
    tolerance: float = 1e-7,
) -> int:
    q_values = np.asarray(q_values, dtype=np.float64)
    if q_values.shape != (state.candidate_count,):
        raise ValueError("Q values must align with the candidate state")
    if not np.isfinite(q_values).all():
        raise FloatingPointError("Q values contain NaN or Inf")
    best = float(q_values.max())
    tied = np.flatnonzero(q_values >= best - tolerance)
    return min(
        (int(position) for position in tied),
        key=lambda position: (state.variable_names[position], int(state.actions[position])),
    )


@dataclass(frozen=True)
class UpdateMetrics:
    loss: float
    td_error: float
    q_mean: float
    q_std: float
    gradient_norm: float


class DoubleDQNLearner:
    def __init__(
        self,
        online: CandidateQNetwork,
        target: CandidateQNetwork,
        device: torch.device,
        learning_rate: float,
        gamma: float,
        gradient_clip: float,
        target_update_interval: int,
    ) -> None:
        self.online = online.to(device)
        self.target = target.to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.device = device
        self.gamma = float(gamma)
        self.gradient_clip = float(gradient_clip)
        self.target_update_interval = int(target_update_interval)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(learning_rate))
        self.loss_function = nn.SmoothL1Loss()
        self.gradient_step = 0

    def q_values(self, state: CandidateState) -> np.ndarray:
        self.online.eval()
        with torch.no_grad():
            values = self.online(*_state_tensors(state, self.device))
        return values.detach().cpu().numpy()

    def select_action(
        self,
        state: CandidateState,
        epsilon: float,
        rng: np.random.Generator,
    ) -> tuple[int, int, float, np.ndarray]:
        q_values = self.q_values(state)
        if rng.random() < epsilon:
            position = int(rng.integers(state.candidate_count))
        else:
            position = stable_argmax_position(q_values, state)
        rank = int(1 + np.count_nonzero(q_values > q_values[position] + 1e-7))
        return int(state.actions[position]), position, float(rank), q_values

    def update(self, experiences: Iterable[ReplayExperience]) -> UpdateMetrics:
        batch = list(experiences)
        variable_parts = []
        global_parts = []
        category_parts = []
        selected_indices = []
        offset = 0
        for experience in batch:
            tensors = _state_tensors(experience.state, self.device)
            variable_parts.append(tensors[0])
            global_parts.append(tensors[1])
            category_parts.append(tensors[2])
            selected_indices.append(offset + experience.action_position)
            offset += experience.state.candidate_count

        self.online.train()
        all_q = self.online(
            torch.cat(variable_parts),
            torch.cat(global_parts),
            torch.cat(category_parts),
        )
        selected = torch.as_tensor(selected_indices, dtype=torch.long, device=self.device)
        predicted = all_q[selected]

        targets = []
        with torch.no_grad():
            for experience in batch:
                bootstrap = 0.0
                if experience.bootstrap_mask and experience.next_state is not None:
                    next_tensors = _state_tensors(experience.next_state, self.device)
                    online_next = self.online(*next_tensors)
                    next_position = stable_argmax_position(
                        online_next.detach().cpu().numpy(), experience.next_state
                    )
                    target_next = self.target(*next_tensors)[next_position]
                    bootstrap = (self.gamma**experience.n_steps) * float(experience.bootstrap_mask) * float(target_next)
                targets.append(experience.reward + bootstrap)
            target_values = torch.as_tensor(targets, dtype=torch.float32, device=self.device)

        loss = self.loss_function(predicted, target_values)
        if not torch.isfinite(loss):
            raise FloatingPointError("DQN loss is NaN or Inf")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.gradient_step += 1
        if self.gradient_step % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        td_error = (target_values - predicted.detach()).abs()
        return UpdateMetrics(
            loss=float(loss.detach()),
            td_error=float(td_error.mean()),
            q_mean=float(predicted.detach().mean()),
            q_std=float(predicted.detach().std(unbiased=False)),
            gradient_norm=float(gradient_norm),
        )
