from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import ecole
import numpy as np

from .config import BBMDPConfig, RewardMode
from .observation import BipartiteObservation, CopiedNodeBipartite


TERMINAL_STATUSES = {"optimal", "infeasible", "unbounded", "inforunbd"}
TRUNCATED_STATUSES = {
    "timelimit",
    "nodelimit",
    "totalnodelimit",
    "stallnodelimit",
    "memlimit",
    "userinterrupt",
    "restartlimit",
    "scip_error",
}


def _immutable_actions(action_set) -> np.ndarray:
    if action_set is None:
        actions = np.empty(0, dtype=np.int64)
    else:
        actions = np.array(action_set, dtype=np.int64, copy=True)
    actions.setflags(write=False)
    return actions


class SolverInformation:
    def before_reset(self, model) -> None:
        return None

    def extract(self, model, done) -> Dict[str, Any]:
        pyscip_model = model.as_pyscipopt()
        current_node = pyscip_model.getCurrentNode()
        parent_node = current_node.getParent() if current_node is not None else None
        open_node_groups = pyscip_model.getOpenNodes()
        open_node_ids = tuple(
            int(node.getNumber())
            for group in open_node_groups
            for node in group
        )
        return {
            "status": str(pyscip_model.getStatus()).lower(),
            "done": bool(done),
            "node_count": int(pyscip_model.getNNodes()),
            "total_node_count": int(pyscip_model.getNTotalNodes()),
            "depth": int(pyscip_model.getDepth()),
            "lp_iterations": int(pyscip_model.getNLPIterations()),
            "solving_time": float(pyscip_model.getSolvingTime()),
            "primal_bound": float(pyscip_model.getPrimalbound()),
            "dual_bound": float(pyscip_model.getDualbound()),
            "gap": float(pyscip_model.getGap()),
            "solution_count": int(pyscip_model.getNSols()),
            "current_node_id": None if current_node is None else int(current_node.getNumber()),
            "parent_node_id": None if parent_node is None else int(parent_node.getNumber()),
            "open_node_ids": open_node_ids,
        }


@dataclass(frozen=True)
class SearchTreeSnapshot:
    visited_node_ids: tuple[int, ...]
    parent_by_node: Dict[int, Optional[int]]
    open_node_ids: tuple[int, ...]


class SearchTreeTracker:
    def __init__(self) -> None:
        self._visited: list[int] = []
        self._parents: Dict[int, Optional[int]] = {}
        self._open: tuple[int, ...] = ()

    def update(self, info: Dict[str, Any]) -> None:
        node_id = info.get("current_node_id")
        if node_id is not None and node_id not in self._parents:
            copied_id = int(node_id)
            self._visited.append(copied_id)
            parent_id = info.get("parent_node_id")
            self._parents[copied_id] = None if parent_id is None else int(parent_id)
        self._open = tuple(int(node_id) for node_id in info.get("open_node_ids", ()))

    def snapshot(self) -> SearchTreeSnapshot:
        return SearchTreeSnapshot(
            visited_node_ids=tuple(self._visited),
            parent_by_node=dict(self._parents),
            open_node_ids=self._open,
        )


@dataclass(frozen=True)
class EnvironmentState:
    observation: Optional[BipartiteObservation]
    action_set: np.ndarray
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


@dataclass(frozen=True)
class Transition:
    observation: BipartiteObservation
    action: int
    action_position: int
    reward: float
    next_observation: Optional[BipartiteObservation]
    next_action_set: np.ndarray
    terminated: bool
    truncated: bool
    bootstrap_mask: float
    info: Dict[str, Any]


class BBMDPBranchingEnv:
    def __init__(self, config: BBMDPConfig):
        self.config = config
        reward_function = (
            -ecole.reward.NNodes()
            if config.reward_mode == RewardMode.NEGATIVE_NODE_INCREMENT
            else ecole.reward.Constant(-1.0)
        )
        self._ecole_env = ecole.environment.Branching(
            observation_function=CopiedNodeBipartite(cache=config.cache_static_features),
            reward_function=reward_function,
            information_function=SolverInformation(),
            scip_params=config.scip_parameters(),
        )
        self._state: Optional[EnvironmentState] = None
        self._search_tree = SearchTreeTracker()
        self._closed = False

    @property
    def current_state(self) -> EnvironmentState:
        if self._state is None:
            raise RuntimeError("environment must be reset before use")
        return self._state

    @property
    def search_tree(self) -> SearchTreeSnapshot:
        return self._search_tree.snapshot()

    def reset(self, instance_path: Path | str) -> EnvironmentState:
        if self._closed:
            raise RuntimeError("environment is closed")
        instance_path = Path(instance_path)
        if not instance_path.is_file():
            raise FileNotFoundError(instance_path)

        self._ecole_env.seed(self.config.seed)
        self._search_tree = SearchTreeTracker()
        try:
            observation, action_set, reset_reward, done, info = self._ecole_env.reset(str(instance_path))
        except Exception as error:
            self._state = self._error_state(error)
            return self._state
        info = dict(info)
        info["reset_reward"] = float(reset_reward)
        terminated, truncated = self._classify_done(bool(done), info.get("status", "unknown"))
        self._state = EnvironmentState(
            observation=observation,
            action_set=_immutable_actions(action_set),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        self._search_tree.update(info)
        self._validate_state(self._state)
        return self._state

    def step(self, action: int) -> Transition:
        state = self.current_state
        if state.terminated or state.truncated or state.observation is None:
            raise RuntimeError("cannot step a finished episode")

        matches = np.flatnonzero(state.action_set == int(action))
        if matches.size != 1:
            raise ValueError(f"action {action} is not in the current action set")
        action_position = int(matches[0])

        try:
            next_observation, next_action_set, reward, done, info = self._ecole_env.step(int(action))
        except Exception as error:
            info = self._error_state(error).info
            next_observation = None
            next_action_set = None
            reward = 0.0
            done = True
        info = dict(info)
        terminated, truncated = self._classify_done(bool(done), info.get("status", "unknown"))
        next_state = EnvironmentState(
            observation=next_observation,
            action_set=_immutable_actions(next_action_set),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        self._search_tree.update(info)
        self._validate_state(next_state)

        if terminated:
            bootstrap_mask = 0.0
        elif truncated and not self.config.bootstrap_on_truncation:
            bootstrap_mask = 0.0
        else:
            bootstrap_mask = 1.0

        transition = Transition(
            observation=state.observation,
            action=int(action),
            action_position=action_position,
            reward=float(reward),
            next_observation=next_observation,
            next_action_set=next_state.action_set,
            terminated=terminated,
            truncated=truncated,
            bootstrap_mask=bootstrap_mask,
            info=info,
        )
        self._state = next_state
        return transition

    def candidate_name(self, action: int) -> str:
        state = self.current_state
        if state.observation is None or action not in state.action_set:
            raise ValueError(f"action {action} is not active")
        return state.observation.variable_names[int(action)]

    def candidate_mapping_is_current(self, action: int) -> bool:
        expected = self.candidate_name(action)
        variables = self._ecole_env.model.as_pyscipopt().getVars(transformed=True)
        return int(action) < len(variables) and str(variables[int(action)].name) == expected

    def scip_parameter(self, name: str):
        return self._ecole_env.model.as_pyscipopt().getParam(name)

    def close(self) -> None:
        self._state = None
        self._ecole_env = None
        self._closed = True

    @staticmethod
    def _error_state(error: Exception) -> EnvironmentState:
        return EnvironmentState(
            observation=None,
            action_set=_immutable_actions(None),
            terminated=False,
            truncated=True,
            info={
                "status": "scip_error",
                "done": True,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )

    @staticmethod
    def _classify_done(done: bool, status: str) -> tuple[bool, bool]:
        if not done:
            return False, False
        status = str(status).lower()
        if status in TRUNCATED_STATUSES:
            return False, True
        if status in TERMINAL_STATUSES:
            return True, False
        return True, False

    @staticmethod
    def _validate_state(state: EnvironmentState) -> None:
        if state.action_set.ndim != 1 or state.action_set.flags.writeable:
            raise ValueError("action set must be an immutable one-dimensional copy")
        if state.observation is None:
            if not (state.terminated or state.truncated):
                raise ValueError("a live state must contain an observation")
            if state.action_set.size:
                raise ValueError("a finished state cannot expose actions")
            return
        state.observation.validate()
        if state.action_set.size == 0:
            raise ValueError("a live branching state must expose actions")
        if state.action_set.min() < 0 or state.action_set.max() >= state.observation.variable_features.shape[0]:
            raise ValueError("action set contains an invalid variable index")
