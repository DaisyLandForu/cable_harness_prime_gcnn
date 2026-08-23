from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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


class DualPoolQuotaUnfillable(ValueError):
    """Raised when the 12:4 or 16:0 logical batch cannot be filled."""


@dataclass(frozen=True)
class ReplayHandle:
    pool: str
    entry_id: int

    def __post_init__(self) -> None:
        if self.pool not in {"medium", "large"}:
            raise ValueError(f"unknown replay pool: {self.pool}")
        if int(self.entry_id) <= 0:
            raise ValueError("replay entry_id must be positive")


@dataclass(frozen=True)
class PrioritizedBatch:
    experiences: tuple[ReplayExperience, ...]
    indices: np.ndarray
    weights: np.ndarray
    handles: tuple[ReplayHandle, ...] = ()


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
        indices = self._rng.choice(size, size=batch_size, replace=True, p=probabilities)
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


def _importance_beta(beta_start: float, beta_steps: int, gradient_step: int) -> float:
    beta_fraction = min(max(int(gradient_step), 0) / max(int(beta_steps), 1), 1.0)
    return float(beta_start) + beta_fraction * (1.0 - float(beta_start))


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
        self.order: list[int] = []
        self.experiences: dict[int, ReplayExperience] = {}
        self.nbytes: dict[int, int] = {}
        self.priorities: dict[int, float] = {}
        self._next_id = 1
        self.total_bytes = 0
        self.evictions_by_count = 0
        self.evictions_by_bytes = 0

    def __len__(self) -> int:
        return len(self.order)

    @property
    def storage(self) -> list[ReplayExperience]:
        return [self.experiences[entry_id] for entry_id in self.order]

    def max_priority(self) -> float:
        if not self.priorities:
            return 1.0
        return float(max(self.priorities.values()))

    def add(self, experience: ReplayExperience, nbytes: int) -> int:
        if nbytes > self.byte_limit:
            raise MemoryError(
                f"transition is {nbytes} bytes, exceeds pool budget {self.byte_limit}"
            )
        while self.order and (
            len(self.order) >= self.count_limit
            or self.total_bytes + nbytes > self.byte_limit
        ):
            if len(self.order) >= self.count_limit:
                self.evictions_by_count += 1
            else:
                self.evictions_by_bytes += 1
            evicted = self.order.pop(0)
            self.total_bytes -= self.nbytes.pop(evicted)
            del self.experiences[evicted]
            del self.priorities[evicted]
        entry_id = self._next_id
        self._next_id += 1
        self.order.append(entry_id)
        self.experiences[entry_id] = experience
        self.nbytes[entry_id] = nbytes
        self.priorities[entry_id] = self.max_priority()
        self.total_bytes += nbytes
        return entry_id

    def sample_ids(
        self,
        count: int,
        rng: np.random.Generator,
        *,
        alpha: float,
        beta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        size = len(self.order)
        if count > size:
            raise ValueError("not enough replay entries")
        if count == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        ids = np.asarray(self.order, dtype=np.int64)
        scaled = np.power(
            np.asarray([self.priorities[int(entry_id)] for entry_id in ids], dtype=np.float64),
            alpha,
        )
        probabilities = scaled / scaled.sum()
        chosen = rng.choice(size, size=count, replace=True, p=probabilities)
        weights = np.power(size * probabilities[chosen], -beta)
        return ids[chosen], weights

    def update_priority(self, entry_id: int, td_error: float, epsilon: float) -> None:
        if entry_id not in self.priorities:
            raise KeyError(f"stale replay handle: {entry_id}")
        if not np.isfinite(td_error):
            raise FloatingPointError("PER TD errors contain NaN or Inf")
        self.priorities[entry_id] = abs(float(td_error)) + float(epsilon)


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
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: int = 10000,
        epsilon: float = 1.0e-5,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("PER alpha must be in [0, 1]")
        if not 0.0 <= beta_start <= 1.0:
            raise ValueError("PER beta_start must be in [0, 1]")
        self.medium = _ByteLimitedPool(medium_count_limit, medium_byte_limit)
        self.large = _ByteLimitedPool(large_count_limit, large_byte_limit)
        self.large_sample_quota = int(large_sample_quota)
        self.medium_sample_quota = int(medium_sample_quota)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_steps = max(int(beta_steps), 1)
        self.epsilon = float(epsilon)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.medium) + len(self.large)

    def add(self, experience: ReplayExperience, pool: str) -> int:
        _reject_oversized_state(experience.state)
        if isinstance(experience.next_state, GraphState):
            _reject_oversized_state(experience.next_state)
        nbytes = _experience_bytes(experience)
        if pool not in {"medium", "large"}:
            raise ValueError(f"unknown replay pool: {pool}")
        target = self.large if pool == "large" else self.medium
        return target.add(experience, nbytes)

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

    def sample_logical_batch(self, gradient_step: int = 0) -> PrioritizedBatch:
        large_take = (
            self.large_sample_quota
            if len(self.large) >= self.large_sample_quota
            else 0
        )
        medium_take = self.medium_sample_quota + (self.large_sample_quota - large_take)
        if len(self.medium) < medium_take:
            raise DualPoolQuotaUnfillable(
                f"quota_unfillable: need {medium_take} medium, have {len(self.medium)}"
            )
        beta = _importance_beta(self.beta_start, self.beta_steps, gradient_step)
        experiences: list[ReplayExperience] = []
        handles: list[ReplayHandle] = []
        weights: list[np.ndarray] = []
        if large_take:
            large_ids, large_weights = self.large.sample_ids(
                large_take, self._rng, alpha=self.alpha, beta=beta
            )
            experiences.extend(self.large.experiences[int(entry_id)] for entry_id in large_ids)
            handles.extend(ReplayHandle("large", int(entry_id)) for entry_id in large_ids)
            weights.append(large_weights)
        medium_ids, medium_weights = self.medium.sample_ids(
            medium_take, self._rng, alpha=self.alpha, beta=beta
        )
        experiences.extend(self.medium.experiences[int(entry_id)] for entry_id in medium_ids)
        handles.extend(ReplayHandle("medium", int(entry_id)) for entry_id in medium_ids)
        weights.append(medium_weights)
        combined = np.concatenate(weights) if weights else np.empty(0, dtype=np.float64)
        if combined.size:
            if not np.isfinite(combined).all():
                raise FloatingPointError("PER importance weights contain NaN or Inf")
            combined = combined / combined.max()
        return PrioritizedBatch(
            experiences=tuple(experiences),
            indices=np.asarray(
                [(handle.pool, handle.entry_id) for handle in handles], dtype=object
            ),
            weights=np.asarray(combined, dtype=np.float32),
            handles=tuple(handles),
        )

    def update_priorities(
        self,
        handles: Sequence[ReplayHandle] | np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        parsed = _parse_handles(handles)
        td_errors = np.asarray(td_errors, dtype=np.float64)
        if len(parsed) != td_errors.size:
            raise ValueError("PER handles and TD errors must align")
        if not np.isfinite(td_errors).all():
            raise FloatingPointError("PER TD errors contain NaN or Inf")
        for handle, error in zip(parsed, td_errors.reshape(-1)):
            pool = self.large if handle.pool == "large" else self.medium
            pool.update_priority(handle.entry_id, float(error), self.epsilon)


def _parse_handles(handles: Sequence[ReplayHandle] | np.ndarray) -> tuple[ReplayHandle, ...]:
    if isinstance(handles, PrioritizedBatch):
        raise TypeError("pass batch.handles, not the batch itself")
    parsed: list[ReplayHandle] = []
    for item in handles:
        if isinstance(item, ReplayHandle):
            parsed.append(item)
            continue
        if isinstance(item, (tuple, list, np.ndarray)) and len(item) == 2:
            parsed.append(ReplayHandle(str(item[0]), int(item[1])))
            continue
        raise TypeError(f"unsupported replay handle: {item!r}")
    return tuple(parsed)
