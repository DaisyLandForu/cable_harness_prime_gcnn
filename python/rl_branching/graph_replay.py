from dataclasses import dataclass

import numpy as np

from .replay import ReplayExperience


@dataclass(frozen=True)
class PrioritizedBatch:
    experiences: tuple[ReplayExperience, ...]
    indices: np.ndarray
    weights: np.ndarray


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        seed: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: int = 10000,
        epsilon: float = 1.0e-5,
    ) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("PER alpha must be in [0, 1]")
        if not 0.0 <= beta_start <= 1.0:
            raise ValueError("PER beta_start must be in [0, 1]")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_steps = max(int(beta_steps), 1)
        self.epsilon = float(epsilon)
        self._storage: list[ReplayExperience] = []
        self._priorities = np.zeros(self.capacity, dtype=np.float64)
        self._next = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, experience: ReplayExperience) -> None:
        maximum = float(self._priorities[: len(self._storage)].max(initial=1.0))
        if len(self._storage) < self.capacity:
            self._storage.append(experience)
        else:
            self._storage[self._next] = experience
        self._priorities[self._next] = maximum
        self._next = (self._next + 1) % self.capacity

    def sample(self, batch_size: int, gradient_step: int) -> PrioritizedBatch:
        size = len(self._storage)
        if batch_size > size:
            raise ValueError("not enough replay entries")
        scaled = np.power(self._priorities[:size], self.alpha)
        probabilities = scaled / scaled.sum()
        indices = self._rng.choice(size, size=batch_size, replace=False, p=probabilities)
        beta_fraction = min(max(int(gradient_step), 0) / self.beta_steps, 1.0)
        beta = self.beta_start + beta_fraction * (1.0 - self.beta_start)
        weights = np.power(size * probabilities[indices], -beta)
        weights /= weights.max()
        return PrioritizedBatch(
            experiences=tuple(self._storage[int(index)] for index in indices),
            indices=np.asarray(indices, dtype=np.int64),
            weights=np.asarray(weights, dtype=np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        td_errors = np.asarray(td_errors, dtype=np.float64)
        if indices.shape != td_errors.shape:
            raise ValueError("PER indices and TD errors must align")
        if not np.isfinite(td_errors).all():
            raise FloatingPointError("PER TD errors contain NaN or Inf")
        self._priorities[indices] = np.abs(td_errors) + self.epsilon
