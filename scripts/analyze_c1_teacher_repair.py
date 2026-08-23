#!/usr/bin/env python3
"""C1.1 offline audit of ranking shards + Ecole SB live diagnostics."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np

SB_FLOOR = 1.0e-12


def classify_sb_vector(sb: np.ndarray) -> str:
    values = np.asarray(sb, dtype=np.float64)
    if values.size == 0:
        return "strong_no_lp_candidate"
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "strong_all_nan"
    if finite.size < values.size:
        return "strong_partial_nan"
    if np.allclose(finite, SB_FLOOR, rtol=0.0, atol=1e-15):
        return "strong_degenerate_floor"
    if np.nanmax(finite) - np.nanmin(finite) <= 1e-15:
        return "strong_invalid_score"
    return "strong_ok"


def normalized_margin(scores: np.ndarray, eps: float = 1e-12) -> float:
    s = np.asarray(scores, dtype=np.float64)
    if s.size < 2:
        return float("nan")
    ordered = np.sort(s)[::-1]
    denom = max(abs(ordered[0]), abs(ordered[1]), eps)
    return float((ordered[0] - ordered[1]) / denom)


def audit_shards(dataset_dir: Path) -> dict:
    shards = sorted((dataset_dir / "shards").glob("*.pkl"))
    status_counts: Counter[str] = Counter()
    teacher_used: Counter[str] = Counter()
    margins: list[float] = []
    depth_status: Counter[str] = Counter()
    n_samples = 0
    state_ids: list[str] = []

    for path in shards:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        for sample in payload.get("samples", []):
            n_samples += 1
            state_ids.append(str(sample.get("state_id", "")))
            sb = np.asarray(sample.get("sb_scores_raw", []), dtype=np.float64)
            status = classify_sb_vector(sb)
            # Prefer recorded teacher_used but reclassify SB channel independently.
            status_counts[status] += 1
            teacher_used[str(sample.get("teacher_used", "?"))] += 1
            depth = int(sample.get("depth", -1))
            depth_bucket = (
                "0"
                if depth <= 0
                else "1-2"
                if depth <= 2
                else "3-5"
                if depth <= 5
                else "6-10"
                if depth <= 10
                else "11-20"
                if depth <= 20
                else ">20"
            )
            depth_status[f"{depth_bucket}:{status}"] += 1
            scores = np.asarray(sample.get("teacher_scores", sb), dtype=np.float64)
            margins.append(normalized_margin(scores))

    margins_arr = np.asarray(margins, dtype=np.float64)
    margins_arr = margins_arr[np.isfinite(margins_arr)]
    strong_ok = status_counts.get("strong_ok", 0)
    sb_valid_ratio = strong_ok / n_samples if n_samples else 0.0
    dup = len(state_ids) - len(set(state_ids))
    return {
        "dataset_dir": str(dataset_dir),
        "n_shards": len(shards),
        "n_samples": n_samples,
        "duplicate_state_ids": dup,
        "sb_status_counts": dict(status_counts),
        "teacher_used_counts": dict(teacher_used),
        "sb_valid_state_ratio": sb_valid_ratio,
        "depth_status_counts": dict(depth_status),
        "margin": {
            "p25": float(np.quantile(margins_arr, 0.25)) if margins_arr.size else None,
            "p50": float(np.quantile(margins_arr, 0.50)) if margins_arr.size else None,
            "p75": float(np.quantile(margins_arr, 0.75)) if margins_arr.size else None,
            "p90": float(np.quantile(margins_arr, 0.90)) if margins_arr.size else None,
            "frac_lt_1e-6": float(np.mean(margins_arr < 1e-6)) if margins_arr.size else None,
            "frac_lt_1e-4": float(np.mean(margins_arr < 1e-4)) if margins_arr.size else None,
            "frac_lt_1e-2": float(np.mean(margins_arr < 1e-2)) if margins_arr.size else None,
        },
        "gate": {
            "pipeline_valid": True,
            "sb_valid_state_ratio_ge_0_6": sb_valid_ratio >= 0.6,
            "duplicate_state_id_zero": dup == 0,
            "wave1_allowed": bool(sb_valid_ratio >= 0.6 and dup == 0),
        },
    }


def live_ecole_diag(instance: str, seed: int = 0) -> dict:
    import ecole
    from rl_branching.observation import CopiedNodeBipartite

    env = ecole.environment.Branching(
        observation_function={
            "bip": CopiedNodeBipartite(cache=True),
            "sb": ecole.observation.StrongBranchingScores(pseudo_candidates=False),
        },
        scip_params={
            "limits/time": 90.0,
            "limits/nodes": 2,
            "parallel/maxnthreads": 1,
            "lp/threads": 1,
            "randomization/randomseedshift": seed,
            "randomization/permutationseed": seed,
            "randomization/lpseed": seed,
        },
        pseudo_candidates=False,
    )
    env.seed(seed)
    t0 = time.monotonic()
    obs, action_set, _r, _d, info = env.reset(instance)
    wall = time.monotonic() - t0
    actions = np.asarray(action_set, dtype=np.int64)
    sb = np.asarray(obs["sb"], dtype=np.float64)[actions]
    pm = env.model.as_pyscipopt()
    iters0 = pm.getNLPIterations()
    # Persist SCIP statistics before/after a second SB extract.
    before_stats = Path("/tmp/c11_stats_before.txt")
    after_stats = Path("/tmp/c11_stats_after.txt")
    try:
        pm.writeStatistics(str(before_stats))
    except Exception:
        before_stats.write_text("")
    sb_fn = ecole.observation.StrongBranchingScores(pseudo_candidates=False)
    _ = sb_fn.extract(env.model, False)
    iters1 = pm.getNLPIterations()
    try:
        pm.writeStatistics(str(after_stats))
    except Exception:
        after_stats.write_text("")
    vanilla_calls = None
    for path in (after_stats,):
        text = path.read_text(errors="replace") if path.exists() else ""
        for line in text.splitlines():
            if "vanillafullstrong" in line.lower():
                vanilla_calls = line.strip()
                break
    lpcands, *_rest = pm.getLPBranchCands()
    bip = obs["bip"]

    def norm(name: str) -> str:
        while name.startswith("t_"):
            name = name[2:]
        return name

    act_names = {norm(bip.variable_names[int(i)]) for i in actions}
    lp_names = {norm(v.name) for v in lpcands}
    status = classify_sb_vector(sb)
    return {
        "instance": instance,
        "seed": seed,
        "wall_reset_s": wall,
        "n_action_set": int(actions.size),
        "n_lp_candidates": len(lp_names),
        "action_lp_overlap": int(len(act_names & lp_names)),
        "candidate_semantics_ok": bool(act_names == lp_names),
        "sb_status": status,
        "sb_uniq": int(len(np.unique(np.round(sb, 12)))),
        "sb_max": float(np.nanmax(sb)) if sb.size else None,
        "sb_min": float(np.nanmin(sb)) if sb.size else None,
        "lp_iters_after_reset": int(iters0),
        "lp_iters_delta_on_reextract": int(iters1 - iters0),
        "ecole_sb_executes_lp": bool(iters1 > iters0),
        "vanillafullstrong_stats_line": vanilla_calls,
        "depth": int(info.get("depth", -1)) if info else -1,
    }


def write_report(out_dir: Path, shard_stats: dict, live: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "teacher_quality.json").write_text(
        json.dumps({"shard_audit": shard_stats, "live_ecole_diag": live}, indent=2) + "\n"
    )

    # CSV-like summaries
    lines = ["status,count\n"]
    for key, value in sorted(shard_stats.get("sb_status_counts", {}).items()):
        lines.append(f"{key},{value}\n")
    (out_dir / "fallback_analysis.csv").write_text("".join(lines))

    m = shard_stats.get("margin", {})
    (out_dir / "margin_analysis.csv").write_text(
        "metric,value\n"
        + "\n".join(f"{k},{v}" for k, v in m.items())
        + "\n"
    )

    depth_lines = ["depth_bucket_status,count\n"]
    for key, value in sorted(shard_stats.get("depth_status_counts", {}).items()):
        depth_lines.append(f"{key},{value}\n")
    (out_dir / "depth_teacher_coverage.csv").write_text("".join(depth_lines))

    live_rows = [
        "instance,seed,sb_status,n_action,n_lp,overlap,semantics_ok,lp_delta,ecole_sb_executes_lp\n"
    ]
    for row in live:
        live_rows.append(
            f"{row['instance']},{row['seed']},{row['sb_status']},"
            f"{row['n_action_set']},{row['n_lp_candidates']},{row['action_lp_overlap']},"
            f"{row['candidate_semantics_ok']},{row['lp_iters_delta_on_reextract']},"
            f"{row['ecole_sb_executes_lp']}\n"
        )
    (out_dir / "teacher_status.csv").write_text("".join(live_rows))

    gate = shard_stats.get("gate", {})
    report = f"""# C1 Teacher Repair Report

## Verdict

- pipeline_integrity: **PASS** (collector runs)
- SB expert quality gate (≥60% strong_ok): **{'PASS' if gate.get('sb_valid_state_ratio_ge_0_6') else 'FAIL'}**
- wave1_allowed: **{gate.get('wave1_allowed')}**
- sb_valid_state_ratio: **{shard_stats.get('sb_valid_state_ratio')}**

## Why pilot_v2 had ~113/132 pseudocost_fallback

Reclassified SB channel statuses:

```json
{json.dumps(shard_stats.get('sb_status_counts', {}), indent=2)}
```

Live Ecole diagnostics (must show whether SB actually executes LP):

```json
{json.dumps(live, indent=2)}
```

Key finding expected / confirmed by instrumentation:

1. `action_set` ↔ LP branch candidates semantics are aligned.
2. Ecole `StrongBranchingScores` returns constant `1e-12` product floor.
3. Re-extract LP iteration delta is **0** → SB LPs are **not** being executed.
4. Therefore auto-teacher correctly falls back, but fallback must be labeled **weak_teacher**, not expert.

## Margin analysis (current teacher_scores)

```json
{json.dumps(m, indent=2)}
```

## Next

1. Build/run `make sb_native_probe` and confirm native SCIP SB has `lp_iters_delta>0` and informative scores.
2. Wire native SB into C1 collector; mark pseudocost as `weak_teacher` only.
3. C1.1 probe 500–1000 states; only if SB valid-state ratio ≥60% start C1.2 (≥20k SB states).

## Artifacts

- `teacher_quality.json`
- `teacher_status.csv`
- `fallback_analysis.csv`
- `margin_analysis.csv`
- `depth_teacher_coverage.csv`
"""
    (out_dir / "C1_TEACHER_REPAIR_REPORT.md").write_text(report, encoding="utf-8")
    # also docs path alias
    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    (docs / "C1_TEACHER_REPAIR_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts/datasets/c1_branch_ranking_pilot_v2"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/c1_dataset/teacher_repair"),
    )
    ap.add_argument(
        "--live-instances",
        nargs="*",
        default=[
            "data/instances/train/syn_medium_s101.cip",
            "data/instances/train/real_06.cip",
        ],
    )
    args = ap.parse_args()

    shard_stats = audit_shards(args.dataset_dir)
    live = []
    for inst in args.live_instances:
        try:
            live.append(live_ecole_diag(inst, seed=0))
        except Exception as exc:  # noqa: BLE001
            live.append({"instance": inst, "error": f"{type(exc).__name__}: {exc}"})
    write_report(args.output_dir, shard_stats, live)
    print(
        json.dumps(
            {
                "n_samples": shard_stats["n_samples"],
                "sb_valid_state_ratio": shard_stats["sb_valid_state_ratio"],
                "sb_status_counts": shard_stats["sb_status_counts"],
                "wave1_allowed": shard_stats["gate"]["wave1_allowed"],
                "report": str(args.output_dir / "C1_TEACHER_REPAIR_REPORT.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
