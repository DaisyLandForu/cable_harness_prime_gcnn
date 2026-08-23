#!/usr/bin/env python3
"""Aggregate C0.1 branching decision logs into audit CSVs + report."""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def load_branch_logs(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern, recursive=True))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {p}: {exc}")
            continue
        if df.empty:
            continue
        df["source_file"] = p
        path = Path(p)
        stem = path.name
        # stems look like: production-scip__real_01__gcnn__seed0.branches.csv
        if "method" not in df.columns:
            for token in (
                "root-z-bias",
                "topology-only",
                "full-prim",
                "z-bias",
                "gcnn",
                "rl-gcnn-prim-decode",
                "rl-gcnn-prim-feat",
                "rl-gcnn",
            ):
                if f"__{token}__" in stem or f"__{token}." in stem:
                    df["method"] = token
                    break
        if "instance_id" not in df.columns:
            for part in path.parts:
                if part.startswith("real_"):
                    df["instance_id"] = part
                    break
        # align C0 branchrule column aliases used by report text
        if "prim_score" not in df.columns and "selected_bias" in df.columns:
            df["prim_score"] = df["selected_bias"]
        if "extract_time_seconds" not in df.columns and "graph_extract_time_seconds" in df.columns:
            df["extract_time_seconds"] = df["graph_extract_time_seconds"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", required=True)
    ap.add_argument("--raw-results", default="")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dec = load_branch_logs(args.input_glob)
    if dec.empty:
        (out / "C0_AUDIT_REPORT.md").write_text(
            "# C0 Audit Report\n\n_No branch decision logs found._\n"
            f"Pattern: `{args.input_glob}`\n",
            encoding="utf-8",
        )
        print("No decision logs; wrote empty report.")
        return

    dec.to_csv(out / "decision_logs.csv", index=False)

    # Q / Prim scale analysis
    q_std = "q_std" if "q_std" in dec.columns else None
    lam = "lambda_prim" if "lambda_prim" in dec.columns else None
    prim = "prim_score" if "prim_score" in dec.columns else ("bias_score" if "bias_score" in dec.columns else None)

    scale_rows = []
    group_cols = [c for c in ["instance_id", "method", "depth"] if c in dec.columns]
    if not group_cols:
        group_cols = ["source_file"]

    for keys, g in dec.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_decisions"] = len(g)
        if q_std:
            row["q_std_mean"] = float(g[q_std].mean())
            row["q_std_median"] = float(g[q_std].median())
        if "q_top1_top2_margin" in g.columns:
            row["margin_mean"] = float(g["q_top1_top2_margin"].mean())
        if lam and prim and q_std:
            # effective bias magnitude vs Q scale
            lvals = g[lam].astype(float)
            pvals = g[prim].astype(float).abs()
            qs = g[q_std].astype(float).replace(0, np.nan)
            ratio = (lvals * pvals) / qs
            row["lambda_prim_over_qstd_mean"] = float(ratio.mean())
            row["prim_dominates_q_frac"] = float((ratio > 1.0).mean())
        if "variable_family" in g.columns:
            row["frac_z"] = float((g["variable_family"] == "z").mean())
            row["frac_m"] = float((g["variable_family"] == "m").mean())
            row["frac_y"] = float((g["variable_family"] == "y").mean())
        if "extract_time_seconds" in g.columns:
            row["extract_s_mean"] = float(g["extract_time_seconds"].mean())
        if "inference_time_seconds" in g.columns:
            row["infer_s_mean"] = float(g["inference_time_seconds"].mean())
        if "selection_time_seconds" in g.columns:
            row["select_s_mean"] = float(g["selection_time_seconds"].mean())
        scale_rows.append(row)

    scale = pd.DataFrame(scale_rows)
    scale.to_csv(out / "q_prim_scale_analysis.csv", index=False)

    # instance summary
    inst_rows = []
    for inst, g in dec.groupby(dec["instance_id"] if "instance_id" in dec.columns else [0]):
        r = {"instance_id": inst, "n_decisions": len(g)}
        if "depth" in g.columns:
            r["depth_mean"] = float(g["depth"].mean())
            r["depth_max"] = float(g["depth"].max())
            r["frac_depth0"] = float((g["depth"] == 0).mean())
        if "variable_family" in g.columns:
            r["chosen_z_frac"] = float((g["variable_family"] == "z").mean())
        if q_std and lam and prim:
            lvals = g[lam].astype(float)
            pvals = g[prim].astype(float).abs()
            qs = g[q_std].astype(float).replace(0, np.nan)
            ratio = (lvals * pvals) / qs
            r["prim_dominates_frac"] = float((ratio > 1.0).mean())
            r["lambda_over_qstd_p50"] = float(ratio.median())
        if "inference_time_seconds" in g.columns and "selection_time_seconds" in g.columns:
            overhead = g["inference_time_seconds"].fillna(0) + g.get(
                "extract_time_seconds", pd.Series(0, index=g.index)
            ).fillna(0)
            r["ml_overhead_s_sum"] = float(overhead.sum())
        inst_rows.append(r)
    inst = pd.DataFrame(inst_rows)
    inst.to_csv(out / "instance_summary.csv", index=False)

    # optional join with raw wall times
    wall_note = ""
    if args.raw_results and Path(args.raw_results).exists():
        raw = pd.read_csv(args.raw_results)
        wall_note = f"\nJoined raw results: `{args.raw_results}` ({len(raw)} rows).\n"

    # answers for the audit questions
    answers = []
    if "lambda_prim_over_qstd_mean" in scale.columns:
        overall = float(scale["lambda_prim_over_qstd_mean"].mean())
        dom = float(scale["prim_dominates_q_frac"].mean()) if "prim_dominates_q_frac" in scale else float("nan")
        answers.append(
            f"- λ·|Prim| / Q_std 平均约为 **{overall:.3f}**；Prim 量级超过 Q_std 的决策占比约 **{dom:.1%}**。"
        )
    else:
        answers.append("- 缺少 q_std / lambda / prim 列，无法估计 λ 相对 Q 尺度（检查 branchrule CSV schema）。")

    if "depth" in dec.columns and "instance_id" in dec.columns:
        r01 = dec[dec["instance_id"] == "real_01"]
        if not r01.empty:
            by_d = r01.groupby("depth").size().sort_index()
            top_d = by_d.head(5).to_dict()
            answers.append(f"- real_01 决策 depth 分布（前几档计数）: `{top_d}`。")
            if "variable_family" in r01.columns:
                answers.append(
                    f"- real_01 选中变量族: z={((r01.variable_family=='z').mean()):.1%}, "
                    f"m={((r01.variable_family=='m').mean()):.1%}, y={((r01.variable_family=='y').mean()):.1%}。"
                )
        r05 = dec[dec["instance_id"] == "real_05"]
        if not r05.empty and "depth" in r05.columns:
            answers.append(
                f"- real_05 决策数={len(r05)}, depth_mean={r05.depth.mean():.2f}, depth_max={r05.depth.max()}。"
            )
            if "q_top1_top2_margin" in r05.columns:
                answers.append(
                    f"- real_05 Q top1-top2 margin mean={r05.q_top1_top2_margin.mean():.4f} "
                    f"(极小 margin 可能对应不稳定 argmax)。"
                )

    report = [
        "# C0 Audit Report",
        "",
        "## Scope",
        "",
        f"- Decision rows: **{len(dec)}**",
        f"- Source glob: `{args.input_glob}`",
        wall_note,
        "## Must-answer questions",
        "",
        *answers,
        "",
        "## Artifacts",
        "",
        f"- `{out / 'decision_logs.csv'}`",
        f"- `{out / 'instance_summary.csv'}`",
        f"- `{out / 'q_prim_scale_analysis.csv'}`",
        "",
        "## Next",
        "",
        "- 对照 `C0_PRIM_DECOMPOSITION.md` 的 claim。",
        "- 若 Prim 经常支配 Q（ratio>1），固定 λ=0.5 不可迁移——进入 residual/normalized score 设计。",
        "- 灾难实例按 depth × family 切片后再看。",
        "",
    ]
    (out / "C0_AUDIT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {out / 'C0_AUDIT_REPORT.md'} ({len(dec)} decisions)")


if __name__ == "__main__":
    main()
