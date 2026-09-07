#!/usr/bin/env python3
"""Select and seal the audited S05 formal-v3 30-lineage Gate without model access."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from steiner_branching.data.generate import SYNTHETIC_FAMILIES  # noqa: E402
from steiner_branching.learning.formal_v3 import (  # noqa: E402
    V3_CANDIDATES_FILE_SHA256,
    V3_CONFIG_FILE_SHA256,
    V3_EXPERIMENT_ID,
    V3_RAW_ROOT,
    V3_SELECTED_MANIFEST,
    V3_SELECTION_SEAL,
    V3_TEACHER_MANIFEST,
    load_v3_candidates,
    load_v3_protocol,
    prior_s05_graph_hashes,
    require_v3_implementation_identity,
)
from steiner_branching.learning.imitation import atomic_write_json  # noqa: E402
from steiner_branching.learning.teacher_data import file_sha256, load_teacher_sample  # noqa: E402


RECORD_FIELDS = {
    "path", "file_sha256", "semantic_sha256", "task_id", "role", "split",
    "family", "bucket_id", "candidate_rank", "generator_seed", "graph_sha256",
    "teacher_seed", "state_index", "state_valid", "all_tie", "candidate_count",
}


def _load_teacher_manifest() -> dict[str, Any]:
    try:
        raw = json.loads(V3_TEACHER_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"formal-v3 teacher manifest is unreadable: {error}") from error
    expected = {
        "schema_version": 1,
        "stage": "S05",
        "experiment_id": V3_EXPERIMENT_ID,
        "status": "completed",
        "protocol_yaml_sha256": V3_CONFIG_FILE_SHA256,
        "candidate_manifest_sha256": V3_CANDIDATES_FILE_SHA256,
        "selection_performed": False,
        "checkpoint_loaded": False,
        "formal_gate_evaluated": False,
        "s06_authorized": False,
        "test_and_final_accessed": False,
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"formal-v3 teacher manifest {key} changed")
    if raw.get("teacher_gate", {}).get("status") != "PASS":
        raise ValueError("formal-v3 teacher quality Gate did not pass")
    checks = raw["teacher_gate"].get("checks", {})
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError("formal-v3 teacher quality checks are incomplete")
    return raw


def _verified_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = {item["graph_sha256"]: item for item in load_v3_candidates()}
    values = manifest.get("shards")
    if not isinstance(values, list):
        raise ValueError("formal-v3 teacher shards must be a list")
    records = []
    seen_semantics: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict) or set(value) != RECORD_FIELDS:
            raise ValueError(f"formal-v3 teacher record schema changed at {index}")
        if value["role"] != "validation_gate_candidate" or value["split"] != "validation_iid":
            raise ValueError("formal-v3 teacher record crosses its role/split")
        candidate = candidates.get(value["graph_sha256"])
        if candidate is None or any(
            value[key] != candidate[key]
            for key in ("family", "bucket_id", "candidate_rank", "generator_seed")
        ):
            raise ValueError("formal-v3 teacher record is not a frozen candidate")
        semantic = str(value["semantic_sha256"])
        if semantic in seen_semantics:
            raise ValueError(f"duplicate formal-v3 semantic SHA: {semantic}")
        seen_semantics.add(semantic)
        sample = load_teacher_sample(
            V3_RAW_ROOT / value["path"],
            expected_file_sha256=str(value["file_sha256"]),
        )
        if any((
            sample.semantic_sha256 != semantic,
            sample.graph_sha256 != value["graph_sha256"],
            sample.family != value["family"],
            sample.teacher_seed != int(value["teacher_seed"]),
            sample.state_index != int(value["state_index"]),
            sample.labels.state_valid != bool(value["state_valid"]),
            sample.state.candidate_count != int(value["candidate_count"]),
        )):
            raise ValueError("formal-v3 teacher shard payload identity changed")
        records.append(dict(value))
    return records


def select_v3_gate_records(
    config: dict[str, Any],
    candidates: tuple[dict[str, Any], ...],
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_by_graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["state_valid"]:
            valid_by_graph[str(record["graph_sha256"])].append(record)
    threshold = int(config["lineage_selection"]["eligibility_rule"][
        "minimum_semantic_unique_valid_states_across_three_teacher_tasks"
    ])
    selected_lineages = []
    for family in SYNTHETIC_FAMILIES:
        eligible = sorted(
            (
                candidate
                for candidate in candidates
                if candidate["family"] == family
                and len(valid_by_graph[candidate["graph_sha256"]]) >= threshold
            ),
            key=lambda item: (
                int(item["candidate_rank"]), str(item["graph_sha256"])
            ),
        )
        if len(eligible) < 6:
            raise ValueError(f"formal-v3 family has fewer than six eligible lineages: {family}")
        selected_lineages.extend(eligible[:6])

    selected = []
    for family in SYNTHETIC_FAMILIES:
        lineages = [item for item in selected_lineages if item["family"] == family]
        ordered = {
            item["graph_sha256"]: sorted(
                valid_by_graph[item["graph_sha256"]],
                key=lambda row: (
                    int(row["state_index"]), int(row["teacher_seed"]),
                    str(row["semantic_sha256"]),
                ),
            )
            for item in lineages
        }
        depth = 0
        family_selected = []
        while len(family_selected) < 64:
            added = False
            for lineage in lineages:
                rows = ordered[lineage["graph_sha256"]]
                if depth < len(rows):
                    family_selected.append(rows[depth])
                    added = True
                    if len(family_selected) == 64:
                        break
            if not added:
                raise ValueError(f"formal-v3 family has fewer than 64 valid states: {family}")
            depth += 1
        selected.extend(family_selected)
    return selected_lineages, selected


def _checkpoint_preflight(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    results = {}
    for seed in config["frozen_models"]["seeds"]:
        root = REPO / f"checkpoints/steiner/s05/s05-teacher-il-formal-v2/seed-{seed}"
        manifest_path = root / "manifest.json"
        model_path = root / "model.pt"
        frozen = config["frozen_models"]["checkpoints"][seed]
        if file_sha256(manifest_path) != frozen["manifest_sha256"]:
            raise ValueError(f"formal-v3 checkpoint manifest changed for seed {seed}")
        if file_sha256(model_path) != frozen["model_sha256"]:
            raise ValueError(f"formal-v3 model payload changed for seed {seed}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any((
            int(manifest.get("training_seed", -1)) != int(seed),
            manifest.get("checkpoint_sha256") != frozen["model_sha256"],
            manifest.get("data_manifest_sha256")
            != config["frozen_models"]["source_selection_manifest_sha256"],
        )):
            raise ValueError(f"formal-v3 checkpoint internal identity changed for seed {seed}")
        results[str(seed)] = {
            "manifest_path": str(manifest_path.relative_to(REPO)),
            "manifest_sha256": frozen["manifest_sha256"],
            "model_path": str(model_path.relative_to(REPO)),
            "model_sha256": frozen["model_sha256"],
        }
    return results


def build_selected_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_v3_protocol(require_activation=True)
    teacher = _load_teacher_manifest()
    implementation_head = str(teacher["implementation_run_head"])
    require_v3_implementation_identity(implementation_head)
    records = _verified_records(teacher)
    candidates = load_v3_candidates()
    selected_lineages, selected = select_v3_gate_records(config, candidates, records)

    family_lineages = Counter(str(item["family"]) for item in selected_lineages)
    family_states = Counter(str(item["family"]) for item in selected)
    selected_graphs = {str(item["graph_sha256"]) for item in selected_lineages}
    represented_graphs = {str(item["graph_sha256"]) for item in selected}
    prior_graphs, old_v2_gate_graphs = prior_s05_graph_hashes()
    checks = {
        "candidate_manifest_checksum": True,
        "every_teacher_shard_checksum": True,
        "teacher_quality_gate_pass": teacher["teacher_gate"]["status"] == "PASS",
        "unique_gate_lineages_30": len(selected_graphs) == 30,
        "six_lineages_per_family": family_lineages
        == Counter({family: 6 for family in SYNTHETIC_FAMILIES}),
        "every_lineage_has_selected_state": represented_graphs == selected_graphs,
        "selected_states_320": len(selected) == 320,
        "sixty_four_states_per_family": family_states
        == Counter({family: 64 for family in SYNTHETIC_FAMILIES}),
        "semantic_sha256_unique": len({item["semantic_sha256"] for item in selected}) == 320,
        "zero_cross_role_lineage": all(
            item["role"] == "validation_gate_candidate" and item["split"] == "validation_iid"
            for item in selected
        ),
        "zero_prior_s05_lineage": not (selected_graphs & prior_graphs),
        "zero_old_v2_gate_lineage": not (selected_graphs & old_v2_gate_graphs),
        "zero_test_or_final_access": teacher["test_and_final_accessed"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"formal-v3 pre-model-access Gate failed: {checks}")
    checkpoints = _checkpoint_preflight(config)
    selected_manifest = {
        "schema_version": 1,
        "stage": "S05",
        "experiment_id": V3_EXPERIMENT_ID,
        "status": "completed",
        "implementation_run_head": implementation_head,
        "protocol_yaml_sha256": V3_CONFIG_FILE_SHA256,
        "candidate_manifest_sha256": V3_CANDIDATES_FILE_SHA256,
        "teacher_manifest_path": str(V3_TEACHER_MANIFEST.relative_to(REPO)),
        "teacher_manifest_sha256": file_sha256(V3_TEACHER_MANIFEST),
        "source_root": str(V3_RAW_ROOT.relative_to(REPO)),
        "lineage_selection": selected_lineages,
        "selection": {"validation_gate": selected},
        "counts": {
            "selected_lineages": 30,
            "selected_states": 320,
            "family_lineages": dict(family_lineages),
            "family_states": dict(family_states),
        },
        "teacher_gate": teacher["teacher_gate"],
        "pre_model_access_gate": {"status": "PASS", "checks": checks},
        "frozen_checkpoints": checkpoints,
        "checkpoint_loaded": False,
        "formal_gate_evaluated": False,
        "s06_authorized": False,
        "test_and_final_accessed": False,
    }
    seal = {
        "schema_version": 1,
        "stage": "S05",
        "experiment_id": V3_EXPERIMENT_ID,
        "implementation_run_head": implementation_head,
        "protocol_yaml_sha256": V3_CONFIG_FILE_SHA256,
        "candidate_manifest_sha256": V3_CANDIDATES_FILE_SHA256,
        "teacher_manifest_path": str(V3_TEACHER_MANIFEST.relative_to(REPO)),
        "teacher_manifest_sha256": file_sha256(V3_TEACHER_MANIFEST),
        "selected_manifest_path": str(V3_SELECTED_MANIFEST.relative_to(REPO)),
        "selected_manifest_sha256": "FILLED_AFTER_ATOMIC_WRITE",
        "selected_lineages": 30,
        "selected_states": 320,
        "family_lineage_counts": {family: 6 for family in SYNTHETIC_FAMILIES},
        "family_state_counts": {family: 64 for family in SYNTHETIC_FAMILIES},
        "pre_model_access_gate": "PASS",
        "checkpoint_loaded": False,
        "formal_gate_evaluated": False,
        "s06_authorized": False,
        "test_and_final_accessed": False,
    }
    return selected_manifest, seal


def main() -> int:
    if V3_SELECTED_MANIFEST.exists() or V3_SELECTION_SEAL.exists():
        raise SystemExit("refusing to overwrite formal-v3 selection evidence")
    selected, seal = build_selected_manifest()
    atomic_write_json(V3_SELECTED_MANIFEST, selected)
    seal["selected_manifest_sha256"] = file_sha256(V3_SELECTED_MANIFEST)
    atomic_write_json(V3_SELECTION_SEAL, seal)
    print(json.dumps({
        "status": "PASS",
        "selected_manifest": str(V3_SELECTED_MANIFEST),
        "selected_manifest_sha256": seal["selected_manifest_sha256"],
        "selection_seal": str(V3_SELECTION_SEAL),
        "counts": selected["counts"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
