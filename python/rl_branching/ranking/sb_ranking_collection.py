"""C1: collect expert ranking states (soft teacher scores + mixture rollouts).

IMPORTANT:
  On this aviation MILP family, Ecole/SCIP `StrongBranchingScores` often returns
  the constant product-score floor (~1e-12) for every candidate (zero LP gains
  within SCIP epsilon). That makes SB hard/soft labels useless (lexicographic
  tie-break only). The collector therefore supports teacher modes:

    - auto: use SB if informative, else Pseudocosts (+ tiny fractionality)
    - pseudocost: always Pseudocosts (+ tiny fractionality)
    - sb: raw SB (will raise if degenerate)
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import ecole
import numpy as np

from rl_branching.candidate_features import extract_candidate_state
from rl_branching.environment import SolverInformation
from rl_branching.observation import CopiedNodeBipartite
from rl_branching.prim_bias import parse_m, parse_y, parse_z

SB_FLOOR = 1.0e-12
_SB_DEGENERATE_TOL = 1.0e-15


def _variable_family(name: str) -> str:
    if parse_z(name) is not None:
        return "z"
    if parse_m(name) is not None:
        return "m"
    if parse_y(name) is not None:
        return "y"
    return "other"


def _stable_expert_position(
    candidate_scores: np.ndarray,
    variable_names: Sequence[str],
    actions: np.ndarray,
) -> int:
    scores = np.asarray(candidate_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("candidate_scores must be a non-empty vector")
    finite = np.isfinite(scores)
    if not finite.any():
        return min(range(scores.size), key=lambda i: (variable_names[i], int(actions[i])))
    best = float(np.max(scores[finite]))
    tied = [int(i) for i in np.flatnonzero(finite & (scores >= best - 1e-12))]
    return min(tied, key=lambda i: (variable_names[i], int(actions[i])))


def _teacher_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    ranks = np.empty(scores.shape[0], dtype=np.int32)
    ranks[order] = np.arange(scores.shape[0], dtype=np.int32)
    return ranks


def _is_degenerate_scores(scores: np.ndarray) -> bool:
    values = np.asarray(scores, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return True
    if np.nanmax(finite) - np.nanmin(finite) <= _SB_DEGENERATE_TOL:
        return True
    # classic vanillafullstrong product floor when both gains ~ 0
    if np.allclose(finite, SB_FLOOR, rtol=0.0, atol=1e-15):
        return True
    return False


def _choose_action_position(
    *,
    expert_position: int,
    n_candidates: int,
    policy: str,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    policy = str(policy).lower()
    if n_candidates <= 0:
        raise ValueError("empty candidate set")
    if policy in ("expert", "sb", "strong_branching", "teacher"):
        return int(expert_position)
    if policy == "random":
        return int(rng.integers(0, n_candidates))
    if policy in ("epsilon_expert", "eps_expert", "mixture"):
        if float(rng.random()) < float(epsilon):
            return int(rng.integers(0, n_candidates))
        return int(expert_position)
    raise ValueError(f"unsupported rollout policy: {policy}")


def _scip_parameters(
    *,
    seed: int,
    time_limit: float,
    node_limit: int,
    protocol: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "parallel/minnthreads": 1,
        "parallel/maxnthreads": 1,
        "lp/threads": 1,
        "randomization/randomseedshift": seed,
        "randomization/permutationseed": seed,
        "randomization/lpseed": seed,
        "limits/time": time_limit,
    }
    if node_limit >= 0:
        params["limits/nodes"] = node_limit
    if protocol == "controlled-bbmdp":
        params.update(
            {
                "nodeselection/dfs/stdpriority": 1_000_000,
                "nodeselection/dfs/memsavepriority": 1_000_000,
                "separating/maxrounds": 0,
                "estimation/restarts/restartpolicy": "n",
                "limits/restarts": 0,
                "presolving/maxrestarts": 0,
            }
        )
    elif protocol != "production-scip":
        raise ValueError(f"unsupported protocol: {protocol}")
    return params


def _fractionality(bipartite, actions: np.ndarray) -> np.ndarray:
    # ECOLE solution_frac is column 9
    feats = np.asarray(bipartite.variable_features, dtype=np.float64)
    if feats.ndim != 2 or feats.shape[1] < 10:
        return np.zeros(actions.shape[0], dtype=np.float64)
    return np.abs(feats[actions, 9])


def _build_teacher_scores(
    *,
    teacher_mode: str,
    sb_scores: np.ndarray,
    pc_scores: np.ndarray,
    actions: np.ndarray,
    bipartite,
) -> tuple[np.ndarray, str]:
    mode = str(teacher_mode).lower()
    sb = np.asarray(sb_scores, dtype=np.float64)[actions]
    pc = np.asarray(pc_scores, dtype=np.float64)[actions]
    frac = _fractionality(bipartite, actions)
    # Stabilize PC ties with fractionality (does not dominate PC scale).
    pc_hybrid = np.nan_to_num(pc, nan=0.0) + 1.0e-6 * frac

    if mode == "sb":
        if _is_degenerate_scores(sb):
            raise RuntimeError(
                "StrongBranchingScores are degenerate (constant ~1e-12). "
                "Refuse teacher_mode=sb; use auto/pseudocost."
            )
        return sb.astype(np.float32), "sb"
    if mode == "pseudocost":
        if _is_degenerate_scores(pc_hybrid):
            raise RuntimeError("Pseudocost teacher also degenerate")
        return pc_hybrid.astype(np.float32), "pseudocost"
    if mode == "auto":
        if not _is_degenerate_scores(sb):
            return sb.astype(np.float32), "sb"
        if _is_degenerate_scores(pc_hybrid):
            raise RuntimeError("Both SB and Pseudocost teachers are degenerate")
        return pc_hybrid.astype(np.float32), "pseudocost_fallback"
    raise ValueError(f"unsupported teacher_mode: {teacher_mode}")


@dataclass(frozen=True)
class RankingJob:
    instance: str
    seed: int
    worker_id: int = 0
    policy: str = "epsilon_expert"


@dataclass(frozen=True)
class EpisodeSummary:
    instance: str
    seed: int
    policy: str
    n_samples: int
    status: str
    nodes: int
    solving_time: float
    wall_time: float
    teacher_used: str = ""
    error: str = ""


def collect_ranking_episode(
    instance: Path | str,
    *,
    seed: int,
    time_limit: float,
    node_limit: int,
    protocol: str = "production-scip",
    max_decisions: int = 120,
    rollout_policy: str = "epsilon_expert",
    epsilon: float = 0.3,
    store_graph: bool = False,
    max_depth_record: int = 10**9,
    teacher_mode: str = "auto",
) -> tuple[list[dict[str, Any]], EpisodeSummary]:
    instance = Path(instance)
    observation_function = {
        "bipartite": CopiedNodeBipartite(cache=True),
        "sb": ecole.observation.StrongBranchingScores(pseudo_candidates=False),
        "pseudocost": ecole.observation.Pseudocosts(),
    }
    env = ecole.environment.Branching(
        observation_function=observation_function,
        reward_function=-ecole.reward.NNodes(),
        information_function=SolverInformation(),
        scip_params=_scip_parameters(
            seed=seed,
            time_limit=time_limit,
            node_limit=node_limit,
            protocol=protocol,
        ),
        pseudo_candidates=False,
    )
    env.seed(seed)
    rng = np.random.default_rng(seed + 17)
    samples: list[dict[str, Any]] = []
    wall0 = time.monotonic()
    status = "unknown"
    nodes = 0
    solving_time = 0.0
    error = ""
    teacher_counter: Dict[str, int] = {}

    try:
        observation, action_set, _reward, done, info = env.reset(str(instance))
        info = dict(info or {})
        status = str(info.get("status", "unknown")).lower()
        nodes = int(info.get("node_count", 0) or 0)
        solving_time = float(info.get("solving_time", 0.0) or 0.0)

        decisions = 0
        while (
            observation is not None
            and action_set is not None
            and not done
            and decisions < max_decisions
        ):
            bipartite = observation["bipartite"]
            sb_scores = observation["sb"]
            pc_scores = observation["pseudocost"]
            actions = np.asarray(action_set, dtype=np.int64)
            if actions.size == 0:
                break
            depth = int(info.get("depth", -1))
            state = extract_candidate_state(bipartite, actions)
            teacher_scores, teacher_used = _build_teacher_scores(
                teacher_mode=teacher_mode,
                sb_scores=sb_scores,
                pc_scores=pc_scores,
                actions=state.actions,
                bipartite=bipartite,
            )
            teacher_counter[teacher_used] = teacher_counter.get(teacher_used, 0) + 1
            expert_position = _stable_expert_position(
                teacher_scores, state.variable_names, state.actions
            )

            if depth > max_depth_record:
                action = int(state.actions[expert_position])
                observation, action_set, _reward, done, info = env.step(action)
                info = dict(info or {})
                status = str(info.get("status", status)).lower()
                nodes = int(info.get("node_count", nodes) or nodes)
                solving_time = float(info.get("solving_time", solving_time) or solving_time)
                decisions += 1
                continue

            ranks = _teacher_ranks(teacher_scores)
            chosen_position = _choose_action_position(
                expert_position=expert_position,
                n_candidates=int(teacher_scores.size),
                policy=rollout_policy,
                epsilon=epsilon,
                rng=rng,
            )
            families = [_variable_family(name) for name in state.variable_names]
            top2_gap = 0.0
            if teacher_scores.size >= 2:
                ordered = np.sort(np.asarray(teacher_scores, dtype=np.float64))[::-1]
                top2_gap = float(ordered[0] - ordered[1])

            sample: dict[str, Any] = {
                "state_id": f"{instance.stem}__seed{seed}__d{decisions}",
                "instance_id": instance.stem,
                "instance": instance.name,
                "seed": int(seed),
                "rollout_policy": str(rollout_policy),
                "epsilon": float(epsilon),
                "teacher_mode": str(teacher_mode),
                "teacher_used": teacher_used,
                "depth": depth,
                "node_count": int(info.get("node_count", -1)),
                "lp_iterations": int(info.get("lp_iterations", -1)),
                "gap": float(info.get("gap", float("nan"))),
                "primal_bound": float(info.get("primal_bound", float("nan"))),
                "dual_bound": float(info.get("dual_bound", float("nan"))),
                "variable_features": np.asarray(state.variable_features, dtype=np.float32),
                "global_features": np.asarray(state.global_features, dtype=np.float32),
                "category_features": np.asarray(state.category_features, dtype=np.float32),
                "variable_names": list(state.variable_names),
                "variable_family": families,
                "actions": np.asarray(state.actions, dtype=np.int64),
                "teacher_scores": np.asarray(teacher_scores, dtype=np.float32),
                "sb_scores_raw": np.asarray(sb_scores, dtype=np.float32)[state.actions],
                "pseudocost_scores_raw": np.asarray(pc_scores, dtype=np.float32)[
                    state.actions
                ],
                "teacher_ranks": ranks,
                "expert_position": int(expert_position),
                "chosen_position": int(chosen_position),
                "followed_expert": bool(chosen_position == expert_position),
                "top1_top2_teacher_gap": top2_gap,
                "n_candidates": int(teacher_scores.size),
            }
            if store_graph:
                sample["graph"] = {
                    "variable_features": np.asarray(
                        bipartite.variable_features, dtype=np.float32
                    ),
                    "row_features": np.asarray(
                        bipartite.extended_row_features
                        if bipartite.extended_row_features is not None
                        else bipartite.row_features,
                        dtype=np.float32,
                    ),
                    "edge_indices": np.asarray(bipartite.edge_indices, dtype=np.int64),
                    "edge_features": np.asarray(bipartite.edge_features, dtype=np.float32)
                    if bipartite.edge_features is not None
                    else np.zeros((0, 1), dtype=np.float32),
                    "global_features": np.asarray(
                        bipartite.global_features, dtype=np.float32
                    ),
                    "variable_names": list(bipartite.variable_names),
                    "row_names": list(bipartite.row_names or []),
                }
            samples.append(sample)

            action = int(state.actions[chosen_position])
            observation, action_set, _reward, done, info = env.step(action)
            info = dict(info or {})
            status = str(info.get("status", status)).lower()
            nodes = int(info.get("node_count", nodes) or nodes)
            solving_time = float(info.get("solving_time", solving_time) or solving_time)
            decisions += 1
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        status = "scip_error"

    dominant_teacher = ""
    if teacher_counter:
        dominant_teacher = max(teacher_counter.items(), key=lambda item: item[1])[0]
    summary = EpisodeSummary(
        instance=instance.name,
        seed=seed,
        policy=str(rollout_policy),
        n_samples=len(samples),
        status=status,
        nodes=nodes,
        solving_time=solving_time,
        wall_time=time.monotonic() - wall0,
        teacher_used=dominant_teacher or ",".join(f"{k}:{v}" for k, v in teacher_counter.items()),
        error=error,
    )
    return samples, summary


def save_shard(path: Path, samples: Sequence[dict[str, Any]], summary: EpisodeSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "c1_branch_ranking_v2",
        "samples": list(samples),
        "summary": {
            "instance": summary.instance,
            "seed": summary.seed,
            "policy": summary.policy,
            "n_samples": summary.n_samples,
            "status": summary.status,
            "nodes": summary.nodes,
            "solving_time": summary.solving_time,
            "wall_time": summary.wall_time,
            "teacher_used": summary.teacher_used,
            "error": summary.error,
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_ranking_job(
    job: RankingJob,
    *,
    output_dir: Path,
    time_limit: float,
    node_limit: int,
    protocol: str,
    max_decisions: int,
    epsilon: float,
    store_graph: bool,
    max_depth_record: int,
    teacher_mode: str,
) -> Dict[str, Any]:
    out = Path(output_dir)
    stem = Path(job.instance).stem
    shard = (
        out
        / "shards"
        / f"worker{job.worker_id}__{stem}__seed{job.seed}__{job.policy}.pkl"
    )
    if shard.is_file():
        with shard.open("rb") as handle:
            existing = pickle.load(handle)
        summary = existing.get("summary", {})
        return {"shard": str(shard), "skipped": True, **summary}

    samples, summary = collect_ranking_episode(
        job.instance,
        seed=job.seed,
        time_limit=time_limit,
        node_limit=node_limit,
        protocol=protocol,
        max_decisions=max_decisions,
        rollout_policy=job.policy,
        epsilon=epsilon,
        store_graph=store_graph,
        max_depth_record=max_depth_record,
        teacher_mode=teacher_mode,
    )
    save_shard(shard, samples, summary)
    return {"shard": str(shard), "skipped": False, **summary.__dict__}
