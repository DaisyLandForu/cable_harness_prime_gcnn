from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

import ecole
import numpy as np

from .config import BBMDPConfig, RewardMode, hybrid_rewards
from .observation import BipartiteObservation, CopiedNodeBipartite
from .scip_profile import (
    assert_production_live_invariants,
    dump_effective_search_params,
    effective_search_params_sha256,
)


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
    def __init__(self) -> None:
        self._first_solution_time: Optional[float] = None

    def before_reset(self, model) -> None:
        self._first_solution_time = None
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
        solution_count = int(pyscip_model.getNSols())
        solving_time = float(pyscip_model.getSolvingTime())
        if solution_count > 0 and self._first_solution_time is None:
            self._first_solution_time = solving_time
        try:
            pdi = float(pyscip_model.getPrimalDualIntegral())
        except Exception:
            pdi = float("nan")
        return {
            "status": str(pyscip_model.getStatus()).lower(),
            "done": bool(done),
            "node_count": int(pyscip_model.getNNodes()),
            "total_node_count": int(pyscip_model.getNTotalNodes()),
            "depth": int(pyscip_model.getDepth()),
            "lp_iterations": int(pyscip_model.getNLPIterations()),
            "solving_time": solving_time,
            "primal_bound": float(pyscip_model.getPrimalbound()),
            "dual_bound": float(pyscip_model.getDualbound()),
            "gap": float(pyscip_model.getGap()),
            "solution_count": solution_count,
            "primal_dual_integral": pdi,
            "first_solution_time": self._first_solution_time,
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
            ecole.reward.Constant(-1.0)
            if config.reward_mode == RewardMode.CONSTANT_MINUS_ONE
            else -ecole.reward.NNodes()
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
        if not (self._state.terminated or self._state.truncated):
            self.assert_live_production_invariants()
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
        node_reward, lp_reward, total_reward = self._step_rewards(state.info, info, float(reward))
        info["node_reward"] = node_reward
        info["lp_reward"] = lp_reward
        info["total_reward"] = total_reward
        info["delta_nodes"] = int(info.get("node_count", 0)) - int(state.info.get("node_count", 0))
        info["delta_lp"] = int(info.get("lp_iterations", 0)) - int(state.info.get("lp_iterations", 0))
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
        if truncated and next_observation is None:
            bootstrap_mask = 0.0

        transition = Transition(
            observation=state.observation,
            action=int(action),
            action_position=action_position,
            reward=float(total_reward),
            next_observation=next_observation,
            next_action_set=next_state.action_set,
            terminated=terminated,
            truncated=truncated,
            bootstrap_mask=bootstrap_mask,
            info=info,
        )
        self._state = next_state
        return transition

    def _step_rewards(
        self,
        previous_info: Dict[str, Any],
        next_info: Dict[str, Any],
        ecole_reward: float,
    ) -> tuple[float, float, float]:
        delta_nodes = int(next_info.get("node_count", 0)) - int(previous_info.get("node_count", 0))
        delta_lp = int(next_info.get("lp_iterations", 0)) - int(previous_info.get("lp_iterations", 0))
        if delta_nodes < 0 or delta_lp < 0:
            delta_nodes = max(0, delta_nodes)
            delta_lp = max(0, delta_lp)
        node_reward, lp_reward, hybrid_total = hybrid_rewards(delta_nodes, delta_lp)
        if self.config.reward_mode == RewardMode.HYBRID_NODE_LP:
            return node_reward, lp_reward, hybrid_total
        if self.config.reward_mode == RewardMode.NEGATIVE_NODE_INCREMENT:
            return node_reward, 0.0, node_reward
        return 0.0, 0.0, float(ecole_reward)

    def mark_live_budget_truncation(self, transition: Transition) -> Transition:
        if transition.next_observation is None or transition.next_action_set.size == 0:
            return transition
        over_nodes = (
            self.config.node_limit > 0
            and int(transition.info.get("node_count", 0)) >= int(self.config.node_limit)
        )
        over_time = float(transition.info.get("solving_time", 0.0)) >= float(self.config.time_limit)
        if not (over_nodes or over_time):
            return transition
        reason = "nodelimit" if over_nodes else "timelimit"
        info = dict(transition.info)
        info["status"] = reason
        info["trainer_truncated"] = True
        truncated = replace(
            transition,
            terminated=False,
            truncated=True,
            bootstrap_mask=1.0,
            info=info,
        )
        self._state = EnvironmentState(
            observation=truncated.next_observation,
            action_set=truncated.next_action_set,
            terminated=False,
            truncated=True,
            info=info,
        )
        return truncated

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

    def effective_search_params_dump(self, *, include_seeds: bool = True) -> str:
        return dump_effective_search_params(self.scip_parameter, include_seeds=include_seeds)

    def effective_search_params_sha256(self, *, include_seeds: bool = True) -> str:
        return effective_search_params_sha256(self.scip_parameter, include_seeds=include_seeds)

    def assert_live_production_invariants(self) -> None:
        assert_production_live_invariants(self.scip_parameter)

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
