from dataclasses import dataclass

import numpy as np

from .graph_features import GraphState, graph_state_storage_bytes, transition_storage_bytes
from .replay import ReplayExperience

STATE_BYTE_LIMIT = 512 * 1024 * 1024
MEDIUM_COUNT_LIMIT = 224
MEDIUM_BYTE_LIMIT = 3 * 1024 * 1024 * 1024
LARGE_COUNT_LIMIT = 32
LARGE_BYTE_LIMIT = 1 * 1024 * 1024 * 1024
LARGE_SAMPLE_QUOTA = 4
MEDIUM_SAMPLE_QUOTA = 12


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


def _experience_bytes(experience: ReplayExperience) -> int:
    next_state = experience.next_state if isinstance(experience.next_state, GraphState) else None
    if not isinstance(experience.state, GraphState):
        raise TypeError("dual-pool replay requires GraphState tensors")
    return transition_storage_bytes(experience.state, next_state)


def _reject_oversized_state(state: GraphState | None) -> None:
    if state is None:
        return
    nbytes = graph_state_storage_bytes(state)
    if nbytes > STATE_BYTE_LIMIT:
        raise MemoryError(
            f"single graph state is {nbytes} bytes, exceeds 512MiB; "
            "return to candidate chunking"
        )


@dataclass
class DualPoolSnapshot:
    medium_count: int
    large_count: int
    medium_bytes: int
    large_bytes: int
    evictions_by_count: int
    evictions_by_bytes: int
    can_sample_large_quota: bool


class _ByteLimitedPool:
    def __init__(self, count_limit: int, byte_limit: int) -> None:
        self.count_limit = int(count_limit)
        self.byte_limit = int(byte_limit)
        self.storage: list[ReplayExperience] = []
        self.nbytes: list[int] = []
        self.total_bytes = 0
        self.evictions_by_count = 0
        self.evictions_by_bytes = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, experience: ReplayExperience, nbytes: int) -> None:
        if nbytes > self.byte_limit:
            raise MemoryError(
                f"transition is {nbytes} bytes, exceeds pool budget {self.byte_limit}"
            )
        while self.storage and (
            len(self.storage) >= self.count_limit
            or self.total_bytes + nbytes > self.byte_limit
        ):
            if len(self.storage) >= self.count_limit:
                self.evictions_by_count += 1
            else:
                self.evictions_by_bytes += 1
            self.storage.pop(0)
            self.total_bytes -= self.nbytes.pop(0)
        self.storage.append(experience)
        self.nbytes.append(nbytes)
        self.total_bytes += nbytes


class DualPoolGraphReplay:
    def __init__(
        self,
        seed: int,
        medium_count_limit: int = MEDIUM_COUNT_LIMIT,
        medium_byte_limit: int = MEDIUM_BYTE_LIMIT,
        large_count_limit: int = LARGE_COUNT_LIMIT,
        large_byte_limit: int = LARGE_BYTE_LIMIT,
        large_sample_quota: int = LARGE_SAMPLE_QUOTA,
        medium_sample_quota: int = MEDIUM_SAMPLE_QUOTA,
    ) -> None:
        self.medium = _ByteLimitedPool(medium_count_limit, medium_byte_limit)
        self.large = _ByteLimitedPool(large_count_limit, large_byte_limit)
        self.large_sample_quota = int(large_sample_quota)
        self.medium_sample_quota = int(medium_sample_quota)
        self._rng = np.random.default_rng(seed)

    def add(self, experience: ReplayExperience, pool: str) -> int:
        _reject_oversized_state(experience.state)
        if isinstance(experience.next_state, GraphState):
            _reject_oversized_state(experience.next_state)
        nbytes = _experience_bytes(experience)
        target = self.large if pool == "large" else self.medium
        if pool not in {"medium", "large"}:
            raise ValueError(f"unknown replay pool: {pool}")
        target.add(experience, nbytes)
        return nbytes

    def can_hold_large(self, n_transitions: int, typical_bytes: int) -> bool:
        return int(n_transitions) * int(typical_bytes) <= self.large.byte_limit

    def snapshot(self) -> DualPoolSnapshot:
        return DualPoolSnapshot(
            medium_count=len(self.medium),
            large_count=len(self.large),
            medium_bytes=self.medium.total_bytes,
            large_bytes=self.large.total_bytes,
            evictions_by_count=self.medium.evictions_by_count + self.large.evictions_by_count,
            evictions_by_bytes=self.medium.evictions_by_bytes + self.large.evictions_by_bytes,
            can_sample_large_quota=len(self.large) >= self.large_sample_quota,
        )

    def sample_logical_batch(self) -> tuple[ReplayExperience, ...]:
        large_take = (
            self.large_sample_quota
            if len(self.large) >= self.large_sample_quota
            else 0
        )
        medium_take = self.medium_sample_quota + (self.large_sample_quota - large_take)
        if len(self.medium) < medium_take:
            raise ValueError("not enough medium replay entries")
        if large_take and len(self.large) < large_take:
            raise ValueError("not enough large replay entries")
        batch: list[ReplayExperience] = []
        if large_take:
            large_idx = self._rng.choice(len(self.large), size=large_take, replace=False)
            batch.extend(self.large.storage[int(index)] for index in large_idx)
        medium_idx = self._rng.choice(len(self.medium), size=medium_take, replace=False)
        batch.extend(self.medium.storage[int(index)] for index in medium_idx)
        return tuple(batch)
