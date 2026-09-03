"""S00 contract checks; these tests do not run a solver or any final instance."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
PROTOCOLS = REPO / "configs/steiner/experiments/protocols_v1.yml"
SPLITS = REPO / "configs/steiner/splits/split_policy_v1.yml"
FINAL = REPO / "configs/steiner/splits/final_test_v1.yml"
FINAL_CONTENT = REPO / "configs/steiner/splits/final_test_content_v1.json"
ENVIRONMENT = REPO / "configs/steiner/environment.lock.yml"
GIT_GOVERNANCE = REPO / "configs/steiner/git_governance_v1.yml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise AssertionError(f"{path} must contain one YAML mapping")
    return value


def canonical_final_entries(manifest: dict) -> list[str]:
    entries: list[str] = []
    for suite in manifest["suites"]:
        suite_id = suite["suite_id"]
        revision = suite["source_revision"]
        selector = suite["selector"]
        if selector["kind"] == "numeric_sequence":
            for number in range(
                selector["start"], selector["stop"] + 1, selector["step"]
            ):
                path = selector["path_template"].format(number=number)
                entries.append(f"{suite_id}:{path}@{revision}")
        elif selector["kind"] == "complete_families":
            for family in selector["families"]:
                entries.append(f"{suite_id}:family:{family}:ALL@{revision}")
        elif selector["kind"] == "complete_archive":
            entries.append(
                f"{suite_id}:archive:{selector['archive_id']}:ALL@{revision}"
            )
        else:
            raise AssertionError(f"unsupported selector kind: {selector['kind']}")
    return sorted(entries)


class S00ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocols = load_yaml(PROTOCOLS)
        cls.splits = load_yaml(SPLITS)
        cls.final = load_yaml(FINAL)
        cls.environment = load_yaml(ENVIRONMENT)
        cls.git_governance = load_yaml(GIT_GOVERNANCE)

    def test_required_documents_exist(self) -> None:
        required = [
            "plans/STEINER_RL_BRANCHING_MIGRATION_MASTER_PLAN.md",
            "docs/steiner/RESEARCH_CONTRACT.md",
            "docs/steiner/STATUS.md",
            "docs/steiner/adr/0001-formulation.md",
            "docs/steiner/adr/0002-representation.md",
            "docs/steiner/adr/0003-learning.md",
            "docs/steiner/adr/0004-evaluation.md",
            "docs/steiner/adr/0005-single-branch-git-governance.md",
            "docs/steiner/phases/S00/S00_PLAN.md",
            "docs/steiner/phases/S00/S00_CHANGELOG.md",
            "docs/steiner/phases/S00/S00_TEST_REPORT.md",
            "docs/steiner/phases/S00/S00_RESULT_ANALYSIS.md",
            "docs/steiner/phases/S00/S00_AUDIT_PACKET.md",
            "docs/steiner/phases/S00/S00_COMMANDS.txt",
        ]
        missing = [path for path in required if not (REPO / path).is_file()]
        self.assertEqual(missing, [])

    def test_protocol_ids_limits_and_seeds_are_frozen(self) -> None:
        expected = {
            "P0": ("correctness-v1", 60, 10000, 4096),
            "P1": ("controlled-branching-v1", 600, 200000, 8192),
            "P2": ("generic-scip-v1", 600, 200000, 8192),
            "P3": ("scip-jack-external-v1", 1800, -1, 16384),
            "P4": ("scip-jack-branching-hard-v1", 1800, 500000, 16384),
        }
        self.assertEqual(self.protocols["threads_per_solver"], 1)
        for key, values in expected.items():
            profile = self.protocols["protocols"][key]
            self.assertEqual(
                (
                    profile["id"],
                    profile["time_limit_seconds"],
                    profile["node_limit"],
                    profile["memory_limit_mb"],
                ),
                values,
            )
        seeds = self.protocols["seeds"]
        self.assertEqual(seeds["solver_pilot"], [0])
        self.assertEqual(seeds["solver_formal"], [0, 1, 2, 3, 4])
        self.assertEqual(seeds["training_formal"], [101, 202, 303, 404, 505])
        self.assertEqual(seeds["teacher_collection"], [1001, 1002, 1003])
        self.assertEqual(seeds["bootstrap"], 20260902)

    def test_primary_baselines_metrics_and_statistics_are_unambiguous(self) -> None:
        required_baselines = {
            "scip_default",
            "relpscost",
            "random_candidate",
            "most_infeasible",
            "strong_branching_budgeted_subset",
            "B0_milp_gcnn_imitation",
            "RL_from_frozen_IL",
            "RL_from_scratch_ablation",
        }
        self.assertEqual(
            set(self.protocols["baselines"]["required"]), required_baselines
        )
        self.assertEqual(
            self.protocols["metrics"]["primary_order"],
            [
                "solved_rate_maximize",
                "PAR2_minimize",
                "primal_dual_integral_minimize",
            ],
        )
        stats = self.protocols["statistics"]
        self.assertEqual(stats["pairing_unit"], "instance_solver_seed")
        self.assertEqual(stats["bootstrap_unit"], "instance")
        self.assertEqual(stats["bootstrap_replicates"], 10000)
        self.assertEqual(stats["confidence_level"], 0.95)
        self.assertFalse(stats["checkpoint_selection"]["final_test_allowed_for_selection"])
        self.assertGreaterEqual(len(self.protocols["prohibited_comparisons"]), 8)

    def test_action_and_branchability_gates_are_not_relaxed(self) -> None:
        action = self.protocols["action_contract"]["S02_through_S09"]
        self.assertEqual(action["entity_kind"], "EDGE")
        self.assertEqual(action["eligible_variables"], "fractional_binary_stp_x_only")
        self.assertFalse(action["continuous_flow_variables_eligible"])
        self.assertEqual(action["coverage_required"], 1.0)
        formal = self.protocols["action_contract"]["formal_learned_runs"]
        for key in (
            "invalid_actions_allowed",
            "nan_scores_allowed",
            "mapping_failures_allowed",
            "unexpected_fallbacks_allowed",
        ):
            self.assertEqual(formal[key], 0)
        gate = self.protocols["S03_branchability_gate"]
        self.assertEqual(
            gate["instance_fraction_with_at_least_5_legal_decisions_min"], 0.60
        )
        self.assertEqual(gate["nontrivial_instance_legal_decisions_median_min"], 10)
        self.assertEqual(gate["strong_branch_valid_state_fraction_min"], 0.60)
        self.assertEqual(gate["strong_branch_all_tie_state_fraction_max"], 0.40)
        self.assertEqual(gate["action_to_original_edge_mapping_rate_min"], 1.0)
        self.assertEqual(gate["p95_worker_rss_mb_max"], 8192)

    def test_synthetic_seed_ranges_are_disjoint(self) -> None:
        ranges: list[tuple[int, int, str]] = []
        for name, values in self.splits["synthetic_seed_ranges"].items():
            start = values["start_inclusive"]
            end = values["end_inclusive"]
            self.assertLessEqual(start, end)
            ranges.append((start, end, name))
        for index, (start, end, name) in enumerate(ranges):
            for other_start, other_end, other_name in ranges[index + 1 :]:
                self.assertTrue(
                    end < other_start or other_end < start,
                    f"{name} overlaps {other_name}",
                )
        self.assertEqual(self.splits["grouping_unit"], "base_graph_lineage")
        self.assertFalse(self.splits["state_level_split_allowed"])
        self.assertEqual(self.splits["normalization_source"], "train_only")

    def test_public_dev_and_final_assignments_do_not_overlap(self) -> None:
        assignment = self.splits["public_data_assignment"]
        self.assertIn("odd", assignment["PACE_2018_Track1"]["development_selector"])
        self.assertIn("even", assignment["PACE_2018_Track1"]["final_selector"])
        self.assertIn("odd", assignment["PACE_2018_Track2"]["development_selector"])
        self.assertIn("even", assignment["PACE_2018_Track2"]["final_selector"])
        steinlib = assignment["SteinLib_SPG"]
        self.assertTrue(
            set(steinlib["development_families"]).isdisjoint(
                steinlib["final_families"]
            )
        )

    def test_final_manifest_is_sealed_unrun_and_hash_matches(self) -> None:
        self.assertTrue(self.final["sealed"])
        self.assertEqual(self.final["first_allowed_stage"], "S12")
        self.assertEqual(self.final["learning_runs_total"], 0)
        self.assertEqual(self.final["result_artifacts"], [])
        for suite in self.final["suites"]:
            self.assertEqual(suite["status"], "sealed")
            self.assertFalse(suite["tuning_allowed"])
            self.assertEqual(suite["learning_runs"], 0)
        entries = canonical_final_entries(self.final)
        self.assertEqual(len(entries), 106)
        self.assertEqual(len(entries), len(set(entries)))
        canonical = "".join(f"{entry}\n" for entry in entries).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(canonical).hexdigest(),
            self.final["canonical_entries_sha256"],
        )

    def test_final_content_lock_is_byte_only_complete_and_self_consistent(self) -> None:
        reference = self.final["content_lock"]
        self.assertEqual(reference["operation"], "byte_hash_only_no_parse_no_solve")
        self.assertEqual(reference["locked_stage"], "S02")
        content_path = REPO / reference["manifest_path"]
        raw = content_path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), reference["manifest_sha256"])
        lock = json.loads(raw)
        self.assertEqual(lock["operation"], reference["operation"])
        self.assertEqual(lock["learning_runs_total"], 0)
        self.assertEqual(
            lock["canonical_entries_sha256"], self.final["canonical_entries_sha256"]
        )
        expected_counts = {
            "pace2018_track1_even": 50,
            "pace2018_track2_even": 50,
            "steinlib_spg_final_families": 188,
            "dimacs11_official_spg_bundle": 50,
        }
        self.assertEqual(
            {suite["suite_id"]: suite["instance_count"] for suite in lock["suites"]},
            expected_counts,
        )
        self.assertEqual(sum(expected_counts.values()), reference["instance_count"])
        digest = re.compile(r"^[0-9a-f]{64}$")
        self.assertEqual(
            {notice["source_id"] for notice in lock["source_notices"]},
            {"pace2018", "steinlib", "dimacs11"},
        )
        self.assertEqual(
            next(
                notice["notice_kind"]
                for notice in lock["source_notices"]
                if notice["source_id"] == "pace2018"
            ),
            "license",
        )
        for notice in lock["source_notices"]:
            self.assertRegex(notice["sha256"], digest)
            self.assertGreater(notice["size_bytes"], 0)
        for suite in lock["suites"]:
            member_groups = [suite.get("members", [])]
            for archive in suite.get("archives", []):
                self.assertRegex(archive["archive_sha256"], digest)
                member_groups.append(archive["members"])
            if "archive_sha256" in suite:
                self.assertRegex(suite["archive_sha256"], digest)
            for members in member_groups:
                for member in members:
                    self.assertRegex(member["sha256"], digest)
                    self.assertGreater(member["size_bytes"], 0)
        pace_suites = [
            suite for suite in lock["suites"] if suite["suite_id"].startswith("pace2018")
        ]
        for suite in pace_suites:
            self.assertTrue(
                all(int(Path(member["relative_path"]).stem[-3:]) % 2 == 0 for member in suite["members"])
            )
        lock_script = (REPO / "scripts/steiner/lock_final_content.py").read_text(
            encoding="utf-8"
        )
        for forbidden_import in ("pyscipopt", "parse_pace", "parse_steinlib", "build_mcf"):
            self.assertNotIn(forbidden_import, lock_script)

    def test_environment_records_selected_and_conflicting_stacks(self) -> None:
        decision = self.environment["decision"]
        self.assertEqual(decision["stack_id"], "scip804-ecole081-pyscipopt430")
        self.assertTrue(decision["compatibility_gate_required"])
        self.assertEqual(self.environment["solver_stack"]["scip"]["version"], "8.0.4")
        self.assertEqual(self.environment["solver_stack"]["soplex"]["version"], "6.0.4")
        packages = self.environment["python_environment"]["packages"]
        self.assertEqual(packages["ecole"], "0.8.1")
        self.assertEqual(packages["pyscipopt"], "4.3.0")
        self.assertEqual(packages["torch"], "2.5.1+cu121")
        conflict = self.environment["conflicting_default_stack"]
        self.assertEqual(conflict["scip_version"], "9.2.2")
        self.assertFalse(conflict["allowed_for_formal_runs"])
        self.assertGreaterEqual(
            len(self.environment["formal_run_preconditions"]), 4
        )

    def test_single_branch_git_governance_uses_immutable_phase_checkpoints(self) -> None:
        governance = self.git_governance
        self.assertEqual(governance["schema_version"], 1)
        self.assertEqual(governance["active_branch"], "research/steiner-migration")
        self.assertTrue(governance["one_active_migration_branch"])
        self.assertFalse(governance["stage_branch_creation_allowed"])
        self.assertFalse(governance["force_push_allowed"])
        self.assertFalse(governance["rebase_published_history_allowed"])
        self.assertTrue(governance["push_policy"]["fast_forward_only"])
        self.assertFalse(
            governance["audit_identity"]["moving_branch_name_is_sufficient"]
        )
        self.assertEqual(
            set(governance["audit_identity"]["immutable_fields"]),
            {
                "base_sha",
                "content_head_sha",
                "phase_head_sha",
                "substantive_commit_range",
            },
        )
        checkpoints = governance["historical_checkpoints"]
        self.assertEqual(set(checkpoints), {"S00", "S01", "S02"})
        for stage, checkpoint in checkpoints.items():
            self.assertRegex(checkpoint["base_sha"], r"^[0-9a-f]{40}$")
            self.assertRegex(checkpoint["content_head_sha"], r"^[0-9a-f]{40}$")
            self.assertRegex(checkpoint["phase_head_sha"], r"^[0-9a-f]{40}$")
            self.assertEqual(
                checkpoint["local_gate_tag"],
                f"steiner-{stage.lower()}-local-gate-v1",
            )
            self.assertEqual(checkpoint["gpt_audit"], "NOT_RUN")
        master_plan = (
            REPO / "plans/STEINER_RL_BRANCHING_MIGRATION_MASTER_PLAN.md"
        ).read_text(encoding="utf-8")
        research_contract = (REPO / "docs/steiner/RESEARCH_CONTRACT.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("版本：v1.3", master_plan)
        self.assertIn("research/steiner-migration", master_plan)
        self.assertIn("研究契约 v1.3", research_contract)
        self.assertIn("v1.3 冻结候选", research_contract)
        self.assertIn("24.01 核、131,073 MiB", research_contract)
        self.assertIn("research/steiner-migration", research_contract)


if __name__ == "__main__":
    unittest.main()
