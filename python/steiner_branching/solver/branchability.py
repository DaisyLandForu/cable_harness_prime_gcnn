"""S03 branchability/resource probe with strict, resumable JSON shards."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time
from typing import Any, Iterable, Mapping

from ..config import StrictConfigError, load_yaml_mapping
from ..contracts import content_sha256
from ..data.generate import GeneratorConfig, SYNTHETIC_FAMILIES, generate_graph
from ..data.split import split_for_synthetic_seed
from ..milp.mcf import build_mcf


EXPECTED_STACK_ID = "scip804-ecole081-pyscipopt430"
FORMAL_BASELINES = ("scip_default", "relpscost", "mostinf")


@dataclass(frozen=True)
class ProbeTask:
    task_id: str
    kind: str
    instance_id: str
    family: str
    bucket_id: str
    n_nodes: int
    n_terminals: int
    generator_seed: int
    baseline: str
    strong_branch: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "instance_id": self.instance_id,
            "family": self.family,
            "bucket_id": self.bucket_id,
            "n_nodes": self.n_nodes,
            "n_terminals": self.n_terminals,
            "generator_seed": self.generator_seed,
            "baseline": self.baseline,
            "strong_branch": self.strong_branch,
        }


def _require_keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown or missing:
        raise StrictConfigError(f"{label} fields mismatch: missing={missing}, unknown={unknown}")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise StrictConfigError(f"{label} must be a string-keyed mapping")
    return dict(value)


def load_s03_config(path: Path | str) -> dict[str, Any]:
    raw = load_yaml_mapping(path)
    _require_keys(
        raw,
        {
            "schema_version", "experiment_id", "stage", "split", "split_policy",
            "solver_stack_id", "formulation_id", "protocol_id", "solver_seed",
            "limits", "controls", "baselines", "primary_baseline", "formal_matrix",
            "worker_ramp", "strong_branch", "aggregation", "gate",
        },
        "S03 config",
    )
    if raw["schema_version"] != 1 or raw["stage"] != "S03":
        raise StrictConfigError("S03 config schema_version/stage mismatch")
    if raw["split"] != "train" or raw["solver_stack_id"] != EXPECTED_STACK_ID:
        raise StrictConfigError("S03 config must use the frozen train split and solver stack")
    if raw["formulation_id"] != "rooted_mcf_v1" or raw["protocol_id"] != "P1":
        raise StrictConfigError("S03 config must use rooted_mcf_v1 under P1")
    if raw["solver_seed"] != 0:
        raise StrictConfigError("S03 pilot solver_seed must be 0")
    if tuple(raw["baselines"]) != FORMAL_BASELINES or raw["primary_baseline"] != "relpscost":
        raise StrictConfigError("S03 baselines or primary baseline changed")

    limits = _mapping(raw["limits"], "limits")
    _require_keys(limits, {"time_seconds", "nodes", "memory_mb", "threads"}, "limits")
    if limits != {"time_seconds": 600, "nodes": 200000, "memory_mb": 8192, "threads": 1}:
        raise StrictConfigError("S03 P1 limits differ from the frozen protocol")
    controls = _mapping(raw["controls"], "controls")
    _require_keys(
        controls,
        {
            "presolving_maxrounds", "separating_maxrounds", "separating_maxroundsroot",
            "heuristics", "restart_limit", "node_selector", "propagation",
        },
        "controls",
    )
    expected_controls = {
        "presolving_maxrounds": 0,
        "separating_maxrounds": 0,
        "separating_maxroundsroot": 0,
        "heuristics": "off",
        "restart_limit": 0,
        "node_selector": "estimate",
        "propagation": "default",
    }
    if controls != expected_controls:
        raise StrictConfigError("S03 P1 controls differ from the frozen protocol")

    matrix = _mapping(raw["formal_matrix"], "formal_matrix")
    _require_keys(matrix, {"families", "buckets", "replicates", "seed_start"}, "formal_matrix")
    if tuple(matrix["families"]) != SYNTHETIC_FAMILIES:
        raise StrictConfigError("formal_matrix must cover the five frozen families in order")
    if not isinstance(matrix["buckets"], list) or not matrix["buckets"]:
        raise StrictConfigError("formal_matrix.buckets must be a non-empty list")
    for bucket in matrix["buckets"]:
        bucket = _mapping(bucket, "formal bucket")
        _require_keys(bucket, {"bucket_id", "n_nodes", "n_terminals"}, "formal bucket")
        if not 2 <= int(bucket["n_terminals"]) <= int(bucket["n_nodes"]):
            raise StrictConfigError("formal bucket has invalid node/terminal counts")
    if int(matrix["replicates"]) < 1:
        raise StrictConfigError("formal_matrix.replicates must be positive")

    ramp = _mapping(raw["worker_ramp"], "worker_ramp")
    _require_keys(
        ramp,
        {"workers", "tasks_per_wave", "n_nodes", "n_terminals", "seed_start", "baseline"},
        "worker_ramp",
    )
    if ramp["workers"] != [1, 3, 6] or ramp["tasks_per_wave"] != [1, 3, 6]:
        raise StrictConfigError("worker ramp must remain 1 -> 3 -> 6")
    if ramp["baseline"] != "relpscost":
        raise StrictConfigError("worker ramp baseline must be relpscost")

    strong = _mapping(raw["strong_branch"], "strong_branch")
    _require_keys(
        strong,
        {
            "baseline", "selected_bucket", "include_all_families_and_replicates",
            "max_states_per_task", "iteration_limit_per_candidate", "candidate_limit",
            "idempotent", "tie_relative_tolerance",
        },
        "strong_branch",
    )
    if strong["baseline"] != "relpscost" or not strong["include_all_families_and_replicates"]:
        raise StrictConfigError("strong-branch selection must cover every family/replicate")
    if int(strong["max_states_per_task"]) < 1 or int(strong["candidate_limit"]) < 0:
        raise StrictConfigError("invalid strong-branch sampling limits")

    aggregation = _mapping(raw["aggregation"], "aggregation")
    _require_keys(
        aggregation,
        {"percentile_method", "missing_expected_strong_states_are_invalid"},
        "aggregation",
    )
    if aggregation != {
        "percentile_method": "nearest_rank",
        "missing_expected_strong_states_are_invalid": True,
    }:
        raise StrictConfigError("S03 aggregation rules changed")
    gate = _mapping(raw["gate"], "gate")
    _require_keys(
        gate,
        {
            "min_instance_fraction_with_5_decisions", "min_nontrivial_median_decisions",
            "min_valid_strong_state_fraction", "max_all_tie_valid_state_fraction",
            "min_action_mapping_rate", "max_p95_worker_rss_mb",
            "max_continuous_flow_variables", "max_p95_build_seconds",
            "max_projected_six_worker_rss_mb",
        },
        "gate",
    )
    return raw


def config_sha256(config: Mapping[str, Any]) -> str:
    return content_sha256(dict(config))


def expand_tasks(config: Mapping[str, Any]) -> tuple[list[ProbeTask], list[ProbeTask]]:
    matrix = config["formal_matrix"]
    strong = config["strong_branch"]
    formal: list[ProbeTask] = []
    next_seed = int(matrix["seed_start"])
    for family in matrix["families"]:
        for bucket in matrix["buckets"]:
            for replicate in range(int(matrix["replicates"])):
                seed = next_seed
                next_seed += 1
                if split_for_synthetic_seed(seed, policy_path=config["split_policy"]) != "train":
                    raise StrictConfigError(f"formal seed {seed} is outside the train split")
                instance_id = (
                    f"{family}-{bucket['bucket_id']}-r{replicate}-s{seed}"
                )
                for baseline in config["baselines"]:
                    task_id = f"formal--{instance_id}--{baseline}"
                    formal.append(
                        ProbeTask(
                            task_id=task_id,
                            kind="formal",
                            instance_id=instance_id,
                            family=str(family),
                            bucket_id=str(bucket["bucket_id"]),
                            n_nodes=int(bucket["n_nodes"]),
                            n_terminals=int(bucket["n_terminals"]),
                            generator_seed=seed,
                            baseline=str(baseline),
                            strong_branch=(
                                baseline == strong["baseline"]
                                and bucket["bucket_id"] == strong["selected_bucket"]
                            ),
                        )
                    )

    ramp_config = config["worker_ramp"]
    ramp: list[ProbeTask] = []
    ramp_count = sum(int(value) for value in ramp_config["tasks_per_wave"])
    for index in range(ramp_count):
        family = matrix["families"][index % len(matrix["families"])]
        seed = int(ramp_config["seed_start"]) + index
        if split_for_synthetic_seed(seed, policy_path=config["split_policy"]) != "train":
            raise StrictConfigError(f"ramp seed {seed} is outside the train split")
        instance_id = f"ramp-{index:02d}-{family}-s{seed}"
        ramp.append(
            ProbeTask(
                task_id=f"ramp--{instance_id}--{ramp_config['baseline']}",
                kind="ramp",
                instance_id=instance_id,
                family=str(family),
                bucket_id="ramp",
                n_nodes=int(ramp_config["n_nodes"]),
                n_terminals=int(ramp_config["n_terminals"]),
                generator_seed=seed,
                baseline=str(ramp_config["baseline"]),
                strong_branch=False,
            )
        )
    if len({task.task_id for task in formal + ramp}) != len(formal) + len(ramp):
        raise StrictConfigError("expanded S03 task IDs are not unique")
    return formal, ramp


def task_sha256(task: ProbeTask, config_digest: str) -> str:
    return content_sha256({"task": task.to_dict(), "config_sha256": config_digest})


def _normalise_variable_name(name: str) -> str:
    value = str(name)
    while value.startswith("t_"):
        value = value[2:]
    return value


class CandidateObserver:
    """State container mixed into a PySCIPOpt Branchrule at runtime."""

    def initialise(self, known_names: Iterable[str]) -> None:
        self.known_names = frozenset(known_names)
        self.node_numbers: set[int] = set()
        self.branch_states = 0
        self.candidates_observed = 0
        self.candidates_mapped = 0
        self.mapping_failures: list[str] = []
        self.root_lp_objective: float | None = None
        self.root_fractional_edges: int | None = None
        self.callback_errors: list[str] = []

    def observe(self) -> None:
        values = self.model.getLPBranchCands()
        candidates = values[0]
        n_priority = int(values[4])
        legal = candidates[:n_priority]
        node = self.model.getCurrentNode()
        node_number = int(node.getNumber()) if node is not None else -1
        if node_number in self.node_numbers:
            return
        self.node_numbers.add(node_number)
        if legal:
            self.branch_states += 1
        mapped = 0
        for variable in legal:
            name = _normalise_variable_name(variable.name)
            if name in self.known_names:
                mapped += 1
            elif len(self.mapping_failures) < 20:
                self.mapping_failures.append(name)
        self.candidates_observed += len(legal)
        self.candidates_mapped += mapped
        if node is not None and int(node.getDepth()) == 0 and self.root_lp_objective is None:
            self.root_lp_objective = float(self.model.getLPObjVal())
            self.root_fractional_edges = len(legal)


def _configure_p1(model: Any, config: Mapping[str, Any], baseline: str) -> dict[str, Any]:
    from pyscipopt import SCIP_PARAMSETTING

    limits = config["limits"]
    model.setParam("limits/time", float(limits["time_seconds"]))
    model.setParam("limits/nodes", int(limits["nodes"]))
    model.setParam("limits/memory", float(limits["memory_mb"]))
    model.setParam("parallel/minnthreads", 1)
    model.setParam("parallel/maxnthreads", 1)
    model.setParam("lp/threads", 1)
    seed = int(config["solver_seed"])
    model.setParam("randomization/randomseedshift", seed)
    model.setParam("randomization/permutationseed", seed)
    model.setParam("randomization/lpseed", seed)
    model.setParam("presolving/maxrounds", 0)
    model.setParam("separating/maxrounds", 0)
    model.setParam("separating/maxroundsroot", 0)
    model.setParam("limits/restarts", 0)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    model.setParam("nodeselection/estimate/stdpriority", 1_000_000)
    if baseline == "relpscost":
        model.setParam("branching/relpscost/priority", 900_000)
    elif baseline == "mostinf":
        model.setParam("branching/mostinf/priority", 900_000)
    elif baseline != "scip_default":
        raise ValueError(f"unsupported S03 baseline: {baseline}")
    keys = (
        "limits/time", "limits/nodes", "limits/memory", "parallel/minnthreads",
        "parallel/maxnthreads", "lp/threads", "randomization/randomseedshift",
        "randomization/permutationseed", "randomization/lpseed",
        "presolving/maxrounds", "separating/maxrounds", "separating/maxroundsroot",
        "limits/restarts", "nodeselection/estimate/stdpriority",
        "branching/relpscost/priority", "branching/mostinf/priority",
    )
    return {key: model.getParam(key) for key in keys}


def _finite_or_none(value: Any) -> float | int | None:
    number = float(value)
    return number if math.isfinite(number) and abs(number) < 1e19 else None


def parse_native_probe_output(stdout: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def parse_fields(line: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        for token in line.split()[1:]:
            key, separator, value = token.partition("=")
            if not separator:
                raise ValueError(f"malformed native probe token: {token}")
            fields[key] = value
        return fields

    states: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    int_fields = {
        "node", "depth", "legal", "evaluated", "mapped", "fully_valid",
        "finite_scores", "lp_errors", "lp_iterations_delta", "sb_lp_iterations_delta",
        "sb_calls_delta", "valid", "all_tie", "nodes", "states", "lp_iterations",
    }
    float_fields = {"score_min", "score_max", "peak_rss_mb"}
    for line in stdout.splitlines():
        if not (line.startswith("S03_STATE ") or line.startswith("S03_FINAL ")):
            continue
        raw = parse_fields(line)
        converted: dict[str, Any] = {}
        for key, value in raw.items():
            if key in int_fields:
                converted[key] = int(value)
            elif key in float_fields:
                converted[key] = float(value)
            else:
                converted[key] = value
        if line.startswith("S03_STATE "):
            converted["valid"] = bool(converted["valid"])
            converted["all_tie"] = bool(converted["all_tie"])
            states.append(converted)
        else:
            final = converted
    if final is None:
        raise ValueError("native probe produced no S03_FINAL record")
    return states, final


def _run_native_strong_probe(
    *, model: Any, config: Mapping[str, Any], model_path: Path, native_probe: Path
) -> dict[str, Any]:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.writeProblem(str(model_path))
    strong = config["strong_branch"]
    limits = config["limits"]
    command = [
        str(native_probe), "--instance", str(model_path),
        "--seed", str(config["solver_seed"]),
        "--max-states", str(strong["max_states_per_task"]),
        "--iteration-limit", str(strong["iteration_limit_per_candidate"]),
        "--candidate-limit", str(strong["candidate_limit"]),
        "--idempotent", "1" if strong["idempotent"] else "0",
        "--tie-tolerance", str(strong["tie_relative_tolerance"]),
        "--time-limit", str(limits["time_seconds"]),
        "--node-limit", str(limits["nodes"]),
        "--memory-limit", str(limits["memory_mb"]),
    ]
    started = time.monotonic()
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        return {
            "status": "native_probe_error",
            "returncode": process.returncode,
            "elapsed_seconds": elapsed,
            "states": [],
            "stderr_tail": process.stderr[-4000:],
        }
    try:
        states, final = parse_native_probe_output(process.stdout)
    except Exception as error:
        return {
            "status": "native_probe_parse_error",
            "returncode": process.returncode,
            "elapsed_seconds": elapsed,
            "states": [],
            "error": f"{type(error).__name__}: {error}",
            "stdout_tail": process.stdout[-4000:],
            "stderr_tail": process.stderr[-4000:],
        }
    return {
        "status": "completed",
        "returncode": process.returncode,
        "elapsed_seconds": elapsed,
        "states": states,
        "final": final,
        "stderr_tail": process.stderr[-4000:],
    }


def run_probe_task(
    task: ProbeTask,
    config: Mapping[str, Any],
    *,
    native_probe: Path,
    task_dir: Path,
) -> dict[str, Any]:
    if os.environ.get("STEINER_SOLVER_STACK_ID") != EXPECTED_STACK_ID:
        raise RuntimeError("S03 tasks must enter through run_with_scip804.sh")
    from pyscipopt import Branchrule, SCIP_RESULT

    class Observer(Branchrule, CandidateObserver):
        def branchexeclp(self, allowaddcons: bool) -> dict[str, Any]:
            del allowaddcons
            try:
                self.observe()
            except Exception as error:  # preserve the solve and make the run fail closed in aggregation
                if len(self.callback_errors) < 20:
                    self.callback_errors.append(f"{type(error).__name__}: {error}")
            return {"result": SCIP_RESULT.DIDNOTRUN}

    task_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    graph = generate_graph(
        GeneratorConfig(
            family=task.family,
            n_nodes=task.n_nodes,
            n_terminals=task.n_terminals,
            seed=task.generator_seed,
        )
    )
    build_started = time.monotonic()
    build = build_mcf(graph, configure_correctness_profile=False, hide_output=True)
    build_seconds = time.monotonic() - build_started
    effective_parameters = _configure_p1(build.model, config, task.baseline)
    observer = Observer()
    observer.initialise(item.variable_name for item in build.metadata.edge_variables)
    build.model.includeBranchrule(
        observer, "steiner_s03_observer", "non-intervening S03 legal-action observer",
        1_000_000, -1, 1.0,
    )

    strong_result: dict[str, Any] | None = None
    if task.strong_branch:
        if not native_probe.is_file():
            raise RuntimeError(f"native strong-branch probe is missing: {native_probe}")
        strong_result = _run_native_strong_probe(
            model=build.model,
            config=config,
            model_path=task_dir / "strong_probe.cip",
            native_probe=native_probe,
        )

    original_variables = build.model.getNVars()
    original_constraints = build.model.getNConss()
    solve_started = time.monotonic()
    build.model.optimize()
    solve_wall_seconds = time.monotonic() - solve_started
    primal = _finite_or_none(build.model.getPrimalbound())
    dual = _finite_or_none(build.model.getDualbound())
    gap = _finite_or_none(build.model.getGap())
    root_lp = _finite_or_none(observer.root_lp_objective) if observer.root_lp_objective is not None else None
    root_gap = None
    if root_lp is not None and primal is not None:
        denominator = min(abs(float(root_lp)), abs(float(primal)))
        if denominator > 0 and float(root_lp) * float(primal) >= 0:
            root_gap = abs(float(primal) - float(root_lp)) / denominator
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    if strong_result and isinstance(strong_result.get("final"), Mapping):
        peak_rss_mb = max(
            peak_rss_mb,
            float(strong_result["final"].get("peak_rss_mb", 0.0)),
        )
    result = {
        "schema_version": 1,
        "task": task.to_dict(),
        "status": str(build.model.getStatus()),
        "classification": (
            "root_solved" if observer.branch_states == 0 and str(build.model.getStatus()) == "optimal"
            else "no_legal_edge_action" if observer.branch_states == 0
            else str(build.model.getStatus())
        ),
        "graph_sha256": graph.graph_sha256,
        "metadata_sha256": build.metadata.sha256,
        "model": {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "terminals": len(graph.terminals),
            "binary_edge_variables": build.counts.binary_edge_variables,
            "continuous_flow_variables": build.counts.continuous_flow_variables,
            "flow_balance_constraints": build.counts.flow_balance_constraints,
            "linking_constraints": build.counts.linking_constraints,
            "original_variables": original_variables,
            "original_constraints": original_constraints,
        },
        "timing": {
            "build_seconds": build_seconds,
            "solve_wall_seconds": solve_wall_seconds,
            "scip_solve_seconds": float(build.model.getSolvingTime()),
            "task_wall_seconds": time.monotonic() - started,
        },
        "solver": {
            "nodes": int(build.model.getNNodes()),
            "lp_iterations": int(build.model.getNLPIterations()),
            "primal_bound": primal,
            "dual_bound": dual,
            "final_gap": gap,
            "root_lp_objective": root_lp,
            "root_gap_to_final_primal": root_gap,
            "root_fractional_edge_variables": observer.root_fractional_edges,
        },
        "branchability": {
            "legal_decisions": observer.branch_states,
            "candidates_observed": observer.candidates_observed,
            "candidates_mapped": observer.candidates_mapped,
            "mapping_failures": observer.mapping_failures,
            "callback_errors": observer.callback_errors,
        },
        "resources": {"peak_rss_mb": peak_rss_mb},
        "effective_parameters": effective_parameters,
        "strong_branch": strong_result,
    }
    return result


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_valid_shard(path: Path, task: ProbeTask, digest: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if raw.get("config_sha256") != digest or raw.get("task_sha256") != task_sha256(task, digest):
        raise RuntimeError(f"stale or mismatched shard must not be reused: {path}")
    return raw


def nearest_rank(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def aggregate_results(
    config: Mapping[str, Any], formal_tasks: list[ProbeTask], ramp_tasks: list[ProbeTask],
    shard_dir: Path,
) -> dict[str, Any]:
    digest = config_sha256(config)

    def collect(tasks: list[ProbeTask]) -> tuple[list[dict[str, Any]], list[str]]:
        found: list[dict[str, Any]] = []
        missing: list[str] = []
        for task in tasks:
            shard = load_valid_shard(shard_dir / f"{task.task_id}.json", task, digest)
            if shard is None:
                missing.append(task.task_id)
            else:
                found.append(shard)
        return found, missing

    formal, missing_formal = collect(formal_tasks)
    ramp, missing_ramp = collect(ramp_tasks)
    primary = [item for item in formal if item["task"]["baseline"] == config["primary_baseline"]]
    decisions = [int(item.get("branchability", {}).get("legal_decisions", 0)) for item in primary]
    nontrivial = [value for value in decisions if value > 0]
    fraction_five = sum(value >= 5 for value in decisions) / len(formal_tasks[::len(config["baselines"])])
    nontrivial_median = float(statistics.median(nontrivial)) if nontrivial else 0.0

    observed = sum(int(item.get("branchability", {}).get("candidates_observed", 0)) for item in formal)
    mapped = sum(int(item.get("branchability", {}).get("candidates_mapped", 0)) for item in formal)
    callback_errors = sum(len(item.get("branchability", {}).get("callback_errors", [])) for item in formal)
    mapping_rate = mapped / observed if observed else 0.0
    rss_values = [float(item["resources"]["peak_rss_mb"]) for item in formal if "resources" in item]
    build_values = [float(item["timing"]["build_seconds"]) for item in formal if "timing" in item]
    p95_rss = nearest_rank(rss_values, 0.95)
    p95_build = nearest_rank(build_values, 0.95)
    max_flow = max(
        (int(item["model"]["continuous_flow_variables"]) for item in formal if "model" in item),
        default=0,
    )

    selected_tasks = [task for task in formal_tasks if task.strong_branch]
    expected_strong = len(selected_tasks) * int(config["strong_branch"]["max_states_per_task"])
    strong_states: list[dict[str, Any]] = []
    strong_failures: dict[str, int] = {}
    for item in primary:
        if not item["task"]["strong_branch"]:
            continue
        strong = item.get("strong_branch")
        state = "missing" if not isinstance(strong, Mapping) else str(strong.get("status", "missing"))
        if state != "completed":
            strong_failures[state] = strong_failures.get(state, 0) + 1
            continue
        strong_states.extend(strong.get("states", []))
    valid_strong = [state for state in strong_states if state.get("valid") is True]
    valid_fraction = len(valid_strong) / expected_strong if expected_strong else 0.0
    all_tie_fraction = (
        sum(state.get("all_tie") is True for state in valid_strong) / len(valid_strong)
        if valid_strong else 1.0
    )
    missing_strong = max(0, expected_strong - len(strong_states))

    gate_config = config["gate"]
    formal_measurable = all(
        item.get("status") != "solver_error"
        and "resources" in item
        and "timing" in item
        and "model" in item
        and "branchability" in item
        for item in formal
    )
    ramp_measurable = all(
        item.get("status") != "solver_error" and "resources" in item for item in ramp
    )
    checks = {
        "complete_formal_matrix": len(missing_formal) == 0 and formal_measurable,
        "worker_ramp_completed": len(missing_ramp) == 0 and ramp_measurable,
        "instance_fraction_with_at_least_5_legal_decisions": (
            fraction_five >= float(gate_config["min_instance_fraction_with_5_decisions"])
        ),
        "nontrivial_instance_legal_decisions_median": (
            nontrivial_median >= float(gate_config["min_nontrivial_median_decisions"])
        ),
        "strong_branch_valid_state_fraction": (
            valid_fraction >= float(gate_config["min_valid_strong_state_fraction"])
        ),
        "strong_branch_all_tie_state_fraction": (
            all_tie_fraction <= float(gate_config["max_all_tie_valid_state_fraction"])
        ),
        "action_to_original_edge_mapping_rate": (
            mapping_rate >= float(gate_config["min_action_mapping_rate"]) and callback_errors == 0
        ),
        "p95_worker_rss_mb": (
            p95_rss is not None and p95_rss <= float(gate_config["max_p95_worker_rss_mb"])
        ),
    }
    projected_six = p95_rss * 6 if p95_rss is not None else None
    triggers = {
        "continuous_flow_variables": max_flow > int(gate_config["max_continuous_flow_variables"]),
        "p95_build_seconds": p95_build is None or p95_build > float(gate_config["max_p95_build_seconds"]),
        "p95_worker_rss_mb": p95_rss is None or p95_rss > float(gate_config["max_p95_worker_rss_mb"]),
        "projected_six_worker_rss_mb": (
            projected_six is None
            or projected_six > float(gate_config["max_projected_six_worker_rss_mb"])
        ),
    }
    overall = all(checks.values()) and not any(triggers.values())
    status_counts: dict[str, int] = {}
    for item in formal:
        status = str(item.get("status", "missing"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema_version": 1,
        "stage": "S03",
        "experiment_id": config["experiment_id"],
        "config_sha256": digest,
        "split": config["split"],
        "protocol_id": config["protocol_id"],
        "solver_seed": config["solver_seed"],
        "expected": {
            "formal_tasks": len(formal_tasks),
            "primary_instances": len(formal_tasks) // len(config["baselines"]),
            "ramp_tasks": len(ramp_tasks),
            "strong_states": expected_strong,
        },
        "completion": {
            "formal_shards": len(formal),
            "ramp_shards": len(ramp),
            "missing_formal_tasks": missing_formal,
            "missing_ramp_tasks": missing_ramp,
            "solver_status_counts": status_counts,
        },
        "measurements": {
            "primary_decisions": decisions,
            "instance_fraction_with_at_least_5_legal_decisions": fraction_five,
            "nontrivial_instance_count": len(nontrivial),
            "nontrivial_instance_legal_decisions_median": nontrivial_median,
            "candidates_observed": observed,
            "candidates_mapped": mapped,
            "callback_errors": callback_errors,
            "action_mapping_rate": mapping_rate,
            "strong_states_observed": len(strong_states),
            "strong_states_missing_or_failed": missing_strong,
            "strong_states_valid": len(valid_strong),
            "strong_valid_state_fraction": valid_fraction,
            "strong_valid_states_all_tie": sum(
                state.get("all_tie") is True for state in valid_strong
            ),
            "strong_all_tie_valid_state_fraction": all_tie_fraction,
            "strong_probe_failure_counts": strong_failures,
            "p95_worker_rss_mb": p95_rss,
            "p95_build_seconds": p95_build,
            "max_continuous_flow_variables": max_flow,
            "projected_six_worker_rss_mb": projected_six,
        },
        "gate": {"checks": checks, "overall_pass": overall},
        "mcf_to_scf_triggers": triggers,
    }
