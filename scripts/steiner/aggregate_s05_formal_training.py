#!/usr/bin/env python3
"""Aggregate all five audited S05 formal seeds and evaluate the offline Gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


REPO = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from steiner_branching.learning.formal_protocol import (  # noqa: E402
    FORMAL_CONFIG_PATH,
    FORMAL_TRAINING_SEEDS,
    FORMAL_V2_MANIFEST,
    FORMAL_V2_SOURCE_MANIFEST_SHA256,
    formal_config_sha256,
    load_formal_v2_selection_seal,
    load_s05_formal_config,
    load_s05_formal_v2_config,
)
from steiner_branching.learning.imitation import atomic_write_json  # noqa: E402
from steiner_branching.learning.teacher_data import file_sha256  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(FORMAL_CONFIG_PATH))
    parser.add_argument("--data-manifest")
    parser.add_argument("--input", action="append", dest="inputs")
    parser.add_argument("--formal-v2", action="store_true")
    return parser.parse_args()


def _holm_adjust(raw_pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw_pvalues, key=raw_pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (total - rank) * raw_pvalues[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def _wilcoxon_greater(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if bool(np.all(array == 0.0)):
        return 1.0
    return float(wilcoxon(array, alternative="greater", zero_method="wilcox").pvalue)


def evaluate_formal_gate(
    config: dict[str, Any], teacher_manifest: dict[str, Any], reports: list[dict[str, Any]]
) -> dict[str, Any]:
    seed_reports = {int(report["training_seed"]): report for report in reports}
    if set(seed_reports) != set(FORMAL_TRAINING_SEEDS):
        raise ValueError("formal aggregation requires exactly all five registered seeds")
    reference_keys = None
    graph_seed_effects: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    graph_seed_model_regrets: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    graph_families: dict[str, str] = {}
    seed_regrets: dict[int, float] = {}
    seed_effects: dict[int, float] = {}
    for seed in FORMAL_TRAINING_SEEDS:
        report = seed_reports[seed]
        if (
            report.get("status") != "completed"
            or report.get("formal_gate_evaluated") is not False
            or report.get("reload_max_absolute_error") != 0.0
            or report.get("reload_state_dict_bit_exact") is not True
        ):
            raise ValueError(f"formal seed {seed} is incomplete or failed reload")
        records = report.get("gate_state_records")
        if not isinstance(records, list) or len(records) != 320:
            raise ValueError(f"formal seed {seed} Gate state matrix is incomplete")
        keys = [str(record["semantic_sha256"]) for record in records]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate formal Gate state identity")
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError("formal seeds evaluated different Gate states/order")
        for record in records:
            graph = str(record["graph_sha256"])
            family = str(record["family"])
            previous = graph_families.setdefault(graph, family)
            if previous != family:
                raise ValueError("formal Gate graph family identity changed")
            effect = float(record["random_regret"]) - float(record["model_regret"])
            if not np.isfinite(effect):
                raise ValueError("formal Gate effect is non-finite")
            graph_seed_effects[graph][seed].append(effect)
            graph_seed_model_regrets[graph][seed].append(float(record["model_regret"]))
        seed_regrets[seed] = float(np.mean([
            np.mean(by_seed[seed]) for by_seed in graph_seed_model_regrets.values()
        ]))
        seed_effects[seed] = float(np.mean([
            np.mean(by_seed[seed]) for by_seed in graph_seed_effects.values()
        ]))
    if len(graph_seed_effects) != 30:
        raise ValueError("formal Gate must contain exactly 30 base graph lineages")
    graph_effects: dict[str, float] = {}
    for graph, by_seed in graph_seed_effects.items():
        if set(by_seed) != set(FORMAL_TRAINING_SEEDS):
            raise ValueError("formal Gate lineage is missing a training seed")
        graph_effects[graph] = float(np.mean([
            np.mean(by_seed[seed]) for seed in FORMAL_TRAINING_SEEDS
        ]))
    family_effects = {
        family: float(np.mean([
            effect for graph, effect in graph_effects.items() if graph_families[graph] == family
        ]))
        for family in sorted(set(graph_families.values()))
    }
    expected_families = set(config["state_selection"]["train_quotas"])
    family_graph_counts = {
        family: sum(value == family for value in graph_families.values())
        for family in expected_families
    }
    if set(family_effects) != expected_families or set(family_graph_counts.values()) != {6}:
        raise ValueError("formal Gate family/lineage balance changed")
    supplemental_vectors = {
        "overall": list(graph_effects.values()),
        **{
            f"family:{family}": [
                effect for graph, effect in graph_effects.items()
                if graph_families[graph] == family
            ]
            for family in sorted(set(graph_families.values()))
        },
    }
    supplemental_raw = {
        label: _wilcoxon_greater(vector)
        for label, vector in supplemental_vectors.items()
    }
    values = np.asarray(list(graph_effects.values()), dtype=np.float64)
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    replicates = int(config["statistics"]["bootstrap_replicates"])
    bootstrap = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        bootstrap[index] = float(np.mean(rng.choice(values, size=values.size, replace=True)))
    lower, upper = np.quantile(bootstrap, [0.025, 0.975], method="linear")
    regrets = np.asarray([seed_regrets[seed] for seed in FORMAL_TRAINING_SEEDS])
    mean_regret = float(np.mean(regrets))
    cv = (
        0.0 if mean_regret == 0.0 and bool(np.all(regrets == 0.0))
        else float(np.std(regrets, ddof=1) / mean_regret)
    )
    checks = {
        "teacher_gate_pass": teacher_manifest.get("teacher_gate", {}).get("status") == "PASS",
        "all_five_training_seeds_complete": True,
        "every_seed_mean_improvement_over_random": all(value > 0.0 for value in seed_effects.values()),
        "every_family_aggregate_mean_improvement_over_random": all(value > 0.0 for value in family_effects.values()),
        "primary_bootstrap_ci_lower_bound_above_zero": float(lower) > 0.0,
        "seed_regret_coefficient_of_variation": cv <= float(config["gate"]["max_seed_regret_coefficient_of_variation"]),
        "checkpoint_reload_exact": all(report["reload_max_absolute_error"] == 0.0 for report in reports),
        "manifest_checksum_reload": True,
        "zero_split_or_role_leakage": teacher_manifest["teacher_gate"]["checks"]["zero_split_or_role_leakage"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "primary_effect": "random_regret_minus_model_regret",
        "primary_mean_effect": float(np.mean(values)),
        "primary_bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_unit": "base_graph_lineage",
        "bootstrap_lineages": len(values),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": int(config["statistics"]["bootstrap_seed"]),
        "seed_mean_regrets": {str(key): value for key, value in seed_regrets.items()},
        "seed_mean_effects": {str(key): value for key, value in seed_effects.items()},
        "seed_regret_mean": mean_regret,
        "seed_regret_sample_std": float(np.std(regrets, ddof=1)),
        "seed_regret_coefficient_of_variation": cv,
        "family_mean_effects": family_effects,
        "graph_mean_effects": graph_effects,
        "supplemental_wilcoxon_holm": {
            "alternative": "random_regret_minus_model_regret_greater_than_zero",
            "raw_pvalues": supplemental_raw,
            "holm_adjusted_pvalues": _holm_adjust(supplemental_raw),
            "gate_relevant": False,
        },
    }


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = (
        load_s05_formal_v2_config(require_activation=True)
        if args.formal_v2
        else load_s05_formal_config(config_path, require_activation=True)
    )
    report_root = resolve_path(config["artifacts"]["report_root"])
    input_paths = (
        [resolve_path(value) for value in args.inputs]
        if args.inputs
        else [report_root / "formal_training_shards" / f"seed-{seed:03d}.json" for seed in FORMAL_TRAINING_SEEDS]
    )
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in input_paths]
    data_manifest_path = (
        resolve_path(args.data_manifest)
        if args.data_manifest
        else (FORMAL_V2_MANIFEST if args.formal_v2 else resolve_path(config["artifacts"]["raw_root"]) / "manifest.json")
    )
    teacher_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout.strip()
    config_digest = formal_config_sha256(config)
    data_digest = file_sha256(data_manifest_path)
    if args.formal_v2:
        seal = load_formal_v2_selection_seal()
        if (
            teacher_manifest.get("schema_version") != 1
            or teacher_manifest.get("experiment_id") != config["experiment_id"]
            or teacher_manifest.get("status") != "completed"
            or teacher_manifest.get("effective_config_sha256") != config_digest
            or teacher_manifest.get("implementation_run_head") != head
            or teacher_manifest.get("source_manifest_sha256") != FORMAL_V2_SOURCE_MANIFEST_SHA256
            or teacher_manifest.get("formal_gate_evaluated") is not False
            or seal.get("implementation_run_head") != head
        ):
            raise ValueError("formal-v2 selection manifest identity/Gate changed")
    elif (
        teacher_manifest.get("schema_version") != 2
        or teacher_manifest.get("experiment_id") != config["experiment_id"]
        or teacher_manifest.get("status") != "completed"
        or teacher_manifest.get("config_sha256") != config_digest
        or teacher_manifest.get("git_commit") != head
        or teacher_manifest.get("formal_gate_evaluated") is not False
    ):
        raise ValueError("formal teacher manifest identity/Gate changed")
    for path, report in zip(input_paths, reports):
        if (
            report.get("experiment_id") != config["experiment_id"]
            or report.get("git_commit") != head
            or report.get("config_sha256") != config_digest
            or report.get("data_manifest_sha256") != data_digest
            or file_sha256(report["checkpoint_manifest"])
            != report.get("checkpoint_manifest_sha256")
        ):
            raise ValueError(f"formal seed report identity mismatch: {path}")
    gate = evaluate_formal_gate(config, teacher_manifest, reports)
    aggregate = {
        "schema_version": 1,
        "stage": "S05",
        "experiment_id": config["experiment_id"],
        "created_at_utc": utc_now(),
        "git_commit": head,
        "config_sha256": config_digest,
        "data_manifest": str(data_manifest_path),
        "data_manifest_sha256": data_digest,
        "seed_report_sha256": {
            str(report["training_seed"]): file_sha256(path)
            for path, report in zip(input_paths, reports)
        },
        "gate": gate,
        "formal_gate_evaluated": True,
        "s06_authorized": False if args.formal_v2 else gate["status"] == "PASS",
    }
    output = report_root / "formal_training_report.json"
    atomic_write_json(output, aggregate)
    summary = report_root / "S05_FORMAL_GATE_SUMMARY.json"
    atomic_write_json(summary, aggregate)
    print(json.dumps({"status": gate["status"], "report": str(output)}, sort_keys=True))
    return 0 if gate["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
