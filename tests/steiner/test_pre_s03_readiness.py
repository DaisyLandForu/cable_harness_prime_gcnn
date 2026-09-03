"""Contract tests for the pre-S03 environment and isolation decisions."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
RESOURCE = REPO / "configs/steiner/resource_preflight_20260903.yml"
PROTOCOLS = REPO / "configs/steiner/experiments/protocols_v1.yml"
PROVENANCE = REPO / "configs/steiner/data_provenance_v1.yml"
FINAL_CONTENT_SCRIPT = REPO / "scripts/steiner/lock_final_content.py"


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_resource_assessment_preserves_frozen_limits_and_requires_ramp():
    resource = load_yaml(RESOURCE)
    protocols = load_yaml(PROTOCOLS)
    demand = resource["frozen_protocol_demand"]
    planned = protocols["planned_parallelism"]

    assert resource["cgroup_limits"]["effective_cpu_cores"] == 8.01
    assert resource["cgroup_limits"]["memory_max_mib"] == 65537
    assert resource["host_visibility"]["swap_bytes"] == 0
    assert demand["threads_per_solver"] == protocols["threads_per_solver"] == 1
    assert demand["planned_rollout_workers"] == planned["rollout_workers"] == 6
    assert demand["maximum_p95_rss_per_worker_mb"] == planned["maximum_p95_rss_per_worker_mb"] == 8192
    assert demand["aggregate_worker_budget_mb"] == planned["aggregate_worker_budget_mb"] == 49152
    assert demand["reserved_host_headroom_mb"] == planned["reserved_host_headroom_mb"] == 16384

    s03 = resource["assessment"]["S03"]
    assert s03["status"] == "CONDITIONAL_PASS"
    assert s03["required_ramp_workers"] == [1, 3, 6]
    assert not s03["direct_six_worker_start_allowed"]
    assert resource["assessment"]["S04"] == {
        "status": "PASS_CPU_ONLY",
        "gpu_required": False,
        "rationale": resource["assessment"]["S04"]["rationale"],
    }
    assert resource["repeat_before_formal_run"]["required"]


def test_public_data_policy_is_official_source_checksum_only_and_no_raw_release():
    policy = load_yaml(PROVENANCE)
    assert policy["acquisition"]["official_source_only"]
    assert policy["acquisition"]["checksum_algorithm"] == "SHA-256"
    assert policy["acquisition"]["checksum_mismatch_action"] == "hard_fail_no_substitution"
    assert not policy["acquisition"]["raw_cache_git_tracked"]
    publication = policy["publication"]
    assert not publication["raw_SteinLib_or_DIMACS_redistribution_allowed"]
    assert publication["checksum_is_not_a_license"]
    assert publication["license_resolution_deadline"] == "before_any_public_release"

    script = FINAL_CONTENT_SCRIPT.read_text(encoding="utf-8")
    assert policy["sources"]["SteinLib"]["official_download_template"] in script
    assert policy["sources"]["DIMACS11"]["official_spg_archive"] in script
    assert "data/steiner/raw/" in (REPO / ".gitignore").read_text(encoding="utf-8")


def test_aviation_failures_are_explicitly_deferred_outside_s03():
    backlog = (REPO / "docs/steiner/AVIATION_REGRESSION_BACKLOG.md").read_text(
        encoding="utf-8"
    )
    assert "DEFERRED_SEPARATE_WORKSTREAM" in backlog
    assert "59 passed、4 failed、0 skipped" in backlog
    for failure in (
        "test_parse_z_and_grown_sets",
        "test_prim_variable_features",
        "test_dsu_sixdim_scip_fixture_matches_cpp_extractor",
        "test_scip_tree_help_exposes_remapped_seed_triple",
    ):
        assert failure in backlog
    assert "不在 Steiner S03 中修复" in backlog
    assert "maintenance/aviation-regression-baseline" in backlog
