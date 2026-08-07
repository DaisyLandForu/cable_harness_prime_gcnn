from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .candidate_features import CandidateState


@dataclass(frozen=True)
class OneStepExperience:
    state: CandidateState
    action_position: int
    reward: float
    next_state: Optional[CandidateState]
    bootstrap_mask: float


@dataclass(frozen=True)
class ReplayExperience:
    state: CandidateState
    action_position: int
    reward: float
    next_state: Optional[CandidateState]
    bootstrap_mask: float
    n_steps: int


class NStepAccumulator:
    def __init__(self, n_steps: int, gamma: float) -> None:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)
        self._pending: deque[OneStepExperience] = deque()

    def _build(self, horizon: int) -> ReplayExperience:
        steps = list(self._pending)[:horizon]
        reward = sum((self.gamma**index) * step.reward for index, step in enumerate(steps))
        last = steps[-1]
        return ReplayExperience(
            state=steps[0].state,
            action_position=steps[0].action_position,
            reward=float(reward),
            next_state=last.next_state,
            bootstrap_mask=float(last.bootstrap_mask),
            n_steps=horizon,
        )

    def append(self, experience: OneStepExperience) -> list[ReplayExperience]:
        self._pending.append(experience)
        emitted: list[ReplayExperience] = []
        if len(self._pending) >= self.n_steps:
            emitted.append(self._build(self.n_steps))
            self._pending.popleft()
        if experience.bootstrap_mask == 0.0:
            while self._pending:
                emitted.append(self._build(len(self._pending)))
                self._pending.popleft()
        return emitted

    def clear(self) -> None:
        self._pending.clear()


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self._storage: list[ReplayExperience] = []
        self._next = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, experience: ReplayExperience) -> None:
        if len(self._storage) < self.capacity:
            self._storage.append(experience)
        else:
            self._storage[self._next] = experience
        self._next = (self._next + 1) % self.capacity

    def sample(self, batch_size: int) -> list[ReplayExperience]:
        if batch_size > len(self._storage):
            raise ValueError("not enough replay entries")
        indices = self._rng.choice(len(self._storage), size=batch_size, replace=False)
        return [self._storage[int(index)] for index in indices]
