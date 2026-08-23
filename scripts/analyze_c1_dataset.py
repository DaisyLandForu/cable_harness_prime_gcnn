#!/usr/bin/env python3
"""Build C1 dataset_stats.json + DATASET_REPORT.md."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def _depth_bucket(depth: int) -> str:
    if depth <= 0:
        return "0"
    if depth <= 2:
        return "1-2"
    if depth <= 5:
        return "3-5"
    if depth <= 10:
        return "6-10"
    return ">10"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    args = ap.parse_args()
    root = args.dataset_dir
    shard_dir = root / "shards"
    shards = sorted(shard_dir.glob("*.pkl"))

    n_samples = 0
    n_empty = 0
    gaps = []
    followed = 0
    depth_buckets: Counter[str] = Counter()
    family_top1: Counter[str] = Counter()
    by_instance: Counter[str] = Counter()
    by_policy: Counter[str] = Counter()
    close_top12 = 0
    informative = 0
    teacher_used_counter: Counter[str] = Counter()
    errors = []

    for path in shards:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        summary = payload.get("summary", {})
        if summary.get("error"):
            errors.append({"shard": str(path), "error": summary["error"]})
        samples = payload.get("samples", [])
        if not samples:
            n_empty += 1
        for sample in samples:
            n_samples += 1
            by_instance[str(sample.get("instance_id", sample.get("instance", "?")))] += 1
            by_policy[str(sample.get("rollout_policy", "?"))] += 1
            depth_buckets[_depth_bucket(int(sample.get("depth", -1)))] += 1
            if sample.get("followed_expert"):
                followed += 1
            gap = float(sample.get("top1_top2_teacher_gap", np.nan))
            if np.isfinite(gap):
                gaps.append(gap)
                if gap < 1e-3:
                    close_top12 += 1
            scores = np.asarray(sample.get("teacher_scores", []), dtype=np.float64)
            if scores.size and (np.nanmax(scores) - np.nanmin(scores) > 1e-15):
                informative += 1
            teacher_used_counter[str(sample.get("teacher_used", "?"))] += 1
            names = sample.get("variable_names") or []
            exp = int(sample.get("expert_position", -1))
            fams = sample.get("variable_family")
            if fams and 0 <= exp < len(fams):
                family_top1[str(fams[exp])] += 1
            elif names and 0 <= exp < len(names):
                family_top1["unknown"] += 1

    gaps_arr = np.asarray(gaps, dtype=np.float64) if gaps else np.asarray([], dtype=np.float64)
    informative_frac = (informative / n_samples) if n_samples else 0.0
    label_gate = informative_frac >= 0.9
    stats = {
        "n_shards": len(shards),
        "n_empty_shards": n_empty,
        "n_samples": n_samples,
        "gate_30k": n_samples >= 30000,
        "gate_label_quality": label_gate,
        "informative_score_frac": informative_frac,
        "teacher_used": dict(teacher_used_counter),
        "followed_expert_frac": (followed / n_samples) if n_samples else 0.0,
        "close_top1_top2_frac": (close_top12 / n_samples) if n_samples else 0.0,
        "teacher_gap": {
            "mean": float(gaps_arr.mean()) if gaps_arr.size else None,
            "p50": float(np.median(gaps_arr)) if gaps_arr.size else None,
            "p10": float(np.quantile(gaps_arr, 0.1)) if gaps_arr.size else None,
        },
        "depth_buckets": dict(depth_buckets),
        "by_instance": dict(by_instance),
        "by_policy": dict(by_policy),
        "expert_top1_family": dict(family_top1),
        "n_errors": len(errors),
        "errors_head": errors[:20],
    }
    (root / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    report = [
        "# C1 Branch Ranking Dataset Report",
        "",
        f"- samples: **{n_samples}**",
        f"- gate ≥30k: **{'PASS' if n_samples >= 30000 else 'FAIL'}**",
        f"- gate label quality (informative scores ≥90%): **{'PASS' if label_gate else 'FAIL'}**",
        f"- informative_score_frac: {informative_frac:.3f}",
        f"- teacher_used: `{dict(teacher_used_counter)}`",
        f"- shards: {len(shards)} (empty={n_empty})",
        f"- followed expert frac: {stats['followed_expert_frac']:.3f}",
        f"- top1/top2 teacher gap tiny (<1e-3) frac: {stats['close_top1_top2_frac']:.3f}",
        "",
        "## Depth buckets",
        "",
        "```",
        json.dumps(dict(depth_buckets), indent=2),
        "```",
        "",
        "## By instance",
        "",
        "```",
        json.dumps(dict(by_instance), indent=2),
        "```",
        "",
        "## By rollout policy",
        "",
        "```",
        json.dumps(dict(by_policy), indent=2),
        "```",
        "",
        "## Notes",
        "",
        "- Soft labels (`teacher_scores` / `teacher_ranks`) are primary; do not treat near-ties as hard negatives.",
        "- If `teacher_used` is mostly `pseudocost_fallback`, SB labels were degenerate on this family.",
        "- Validation/test/transfer instances must not appear here.",
        "- Full bipartite graphs are optional (`store_graph`); wave-1 defaults to candidate ranking features for throughput.",
        "",
    ]
    (root / "DATASET_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_samples": n_samples,
                "gate_30k": n_samples >= 30000,
                "gate_label_quality": label_gate,
                "informative_score_frac": informative_frac,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
