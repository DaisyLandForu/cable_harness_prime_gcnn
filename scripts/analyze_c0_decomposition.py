#!/usr/bin/env python3
"""Summarize C0.2 Prim decomposition wall/nodes/gap by method × instance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ["gcnn", "z-bias", "root-z-bias", "full-prim", "topology-only"]
FOCUS = ["real_01", "real_05", "real_08"]


def shifted_geomean(x: np.ndarray, shift: float = 10.0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(x + shift))) - shift)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-results", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.raw_results)
    if "method" not in df.columns and "method_name" in df.columns:
        df = df.rename(columns={"method_name": "method"})

    # normalize column names used across runners
    wall_col = "wall_time" if "wall_time" in df.columns else "solving_time"
    nodes_col = "nodes" if "nodes" in df.columns else "n_nodes"
    gap_col = "gap" if "gap" in df.columns else "final_gap"

    rows = []
    for inst in FOCUS:
        sub_i = df[df["instance_id"] == inst]
        for m in METHODS:
            sub = sub_i[sub_i["method"] == m]
            if sub.empty:
                continue
            rows.append(
                {
                    "instance_id": inst,
                    "method": m,
                    "n_runs": len(sub),
                    "wall_mean": float(sub[wall_col].mean()),
                    "wall_median": float(sub[wall_col].median()),
                    "wall_shifted_geomean": shifted_geomean(sub[wall_col].to_numpy()),
                    "nodes_mean": float(sub[nodes_col].mean()),
                    "gap_mean": float(sub[gap_col].mean()) if gap_col in sub else float("nan"),
                    "optimal_rate": float((sub.get("status", pd.Series(dtype=str)).astype(str).str.contains("optimal", case=False, na=False)).mean())
                    if "status" in sub
                    else float("nan"),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "method_instance_summary.csv", index=False)

    # speedup vs gcnn / vs full-prim on real_01
    findings = []
    for inst in FOCUS:
        base = summary[(summary.instance_id == inst) & (summary.method == "gcnn")]
        prim = summary[(summary.instance_id == inst) & (summary.method == "full-prim")]
        rootz = summary[(summary.instance_id == inst) & (summary.method == "root-z-bias")]
        zb = summary[(summary.instance_id == inst) & (summary.method == "z-bias")]
        topo = summary[(summary.instance_id == inst) & (summary.method == "topology-only")]
        if base.empty:
            continue
        bw = float(base.iloc[0]["wall_shifted_geomean"])
        for name, block in [
            ("z-bias", zb),
            ("root-z-bias", rootz),
            ("full-prim", prim),
            ("topology-only", topo),
        ]:
            if block.empty or not np.isfinite(bw) or bw <= 0:
                continue
            mw = float(block.iloc[0]["wall_shifted_geomean"])
            findings.append(
                {
                    "instance_id": inst,
                    "method": name,
                    "speedup_vs_gcnn": bw / mw if mw > 0 else float("nan"),
                    "wall_sgm": mw,
                    "gcnn_wall_sgm": bw,
                }
            )
    find_df = pd.DataFrame(findings)
    find_df.to_csv(out / "speedup_vs_gcnn.csv", index=False)

    # scientific claim helper
    claim = "INCONCLUSIVE"
    detail = ""
    r01 = find_df[find_df.instance_id == "real_01"] if not find_df.empty else pd.DataFrame()
    if not r01.empty:
        sp_prim = r01[r01.method == "full-prim"]
        sp_root = r01[r01.method == "root-z-bias"]
        sp_topo = r01[r01.method == "topology-only"]
        if not sp_prim.empty and not sp_root.empty:
            p = float(sp_prim.iloc[0]["speedup_vs_gcnn"])
            r = float(sp_root.iloc[0]["speedup_vs_gcnn"])
            t = float(sp_topo.iloc[0]["speedup_vs_gcnn"]) if not sp_topo.empty else float("nan")
            # If root-z recovers most of full-prim gain over gcnn
            if np.isfinite(p) and np.isfinite(r) and p > 1.05:
                frac = (r - 1.0) / (p - 1.0) if p > 1.0 else float("nan")
                if np.isfinite(frac) and frac >= 0.7:
                    claim = "ROOT_Z_FAMILY_PRIOR"
                    detail = (
                        f"On real_01, root-z-bias recovers {frac:.0%} of full-prim's "
                        f"speedup vs gcnn (root-z={r:.3f}×, full-prim={p:.3f}×)."
                    )
                elif np.isfinite(t) and t + 0.05 >= p and (not np.isfinite(r) or r < 1.05):
                    claim = "TOPOLOGY_CONNECTIVITY"
                    detail = (
                        f"topology-only ≈ full-prim ({t:.3f}× vs {p:.3f}×) while root-z "
                        f"does not explain the gain ({r:.3f}×)."
                    )
                else:
                    claim = "MIXED"
                    detail = (
                        f"real_01 speedups: root-z={r:.3f}×, topology={t:.3f}×, "
                        f"full-prim={p:.3f}× vs gcnn."
                    )
            elif np.isfinite(p) and p <= 1.05:
                claim = "NO_PRIM_GAIN_ON_REAL01"
                detail = f"full-prim speedup vs gcnn on real_01 is only {p:.3f}×."

    report = out / "C0_PRIM_DECOMPOSITION.md"
    lines = [
        "# C0 Prim Decomposition Report",
        "",
        "## Claim",
        "",
        f"**{claim}**",
        "",
        detail or "_See tables; insufficient contrast._",
        "",
        "## Protocol",
        "",
        "- Methods: gcnn / z-bias / root-z-bias / full-prim / topology-only",
        "- Focus instances: real_01, real_05, real_08",
        "- real_09 must NOT be used for selecting λ or mode",
        "- Exploratory seeds in config (expand to 5 for final judgment)",
        "",
        "## Method × Instance (shifted-geomean wall)",
        "",
    ]
    if not summary.empty:
        pivot = summary.pivot_table(
            index="instance_id", columns="method", values="wall_shifted_geomean", aggfunc="first"
        )
        lines.append("```")
        lines.append(pivot.to_string())
        lines.append("```")
    else:
        lines.append("_No rows parsed from raw_results.csv_")
    lines += [
        "",
        "## Speedup vs gcnn",
        "",
    ]
    if not find_df.empty:
        lines.append("```")
        lines.append(find_df.to_string(index=False))
        lines.append("```")
    else:
        lines.append("_n/a_")
    lines += [
        "",
        "## Interpretation rules",
        "",
        "1. If `root-z-bias ≈ full-prim` on real_01 → Phase-A gain is mainly **root z-family prior**.",
        "2. If `topology-only ≈ full-prim` and root-z fails → connectivity prior matters.",
        "3. If only `z-bias` (all depths) works → variable-family prior, not Prim growth.",
        "4. Do **not** proceed to hard-mask `both_in` based on these results alone.",
        "",
        "## Artifacts",
        "",
        f"- `{out / 'method_instance_summary.csv'}`",
        f"- `{out / 'speedup_vs_gcnn.csv'}`",
        f"- raw: `{args.raw_results}`",
        "",
        "```json",
        json.dumps({"claim": claim, "detail": detail}, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")
    print(f"CLAIM={claim}")


if __name__ == "__main__":
    main()
