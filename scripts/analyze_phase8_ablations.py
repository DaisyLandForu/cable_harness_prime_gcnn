#!/usr/bin/env python3
"""Summarize phase-8 model, protocol, and depth ablations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEYS = ["protocol", "instance_id", "seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def shifted_geomean(values: pd.Series, shift: float = 1.0) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    clean = clean[np.isfinite(clean) & (clean >= 0)]
    return float(np.exp(np.mean(np.log(clean + shift))) - shift) if clean.size else math.nan


def prepare(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for column in ("wall_time", "solve_time", "nodes", "time_limit", "rl_inference_total", "fallback_count"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["optimal"] = result["status"].eq("optimal")
    result["par2_time"] = np.where(result["optimal"], result["wall_time"], 2.0 * result["time_limit"])
    result["wall_without_inference"] = np.maximum(
        0.0, result["wall_time"] - result["rl_inference_total"].fillna(0.0)
    )
    return result


def aggregate(data: pd.DataFrame, experiment: str) -> pd.DataFrame:
    rows = []
    for (protocol, method), group in data.groupby(["protocol", "method"], sort=False):
        solve_sum = group["solve_time"].sum()
        rows.append(
            {
                "experiment": experiment,
                "protocol": protocol,
                "method": method,
                "runs": len(group),
                "solved": int(group["optimal"].sum()),
                "solved_rate": float(group["optimal"].mean()),
                "shifted_gmean_wall_time": shifted_geomean(group["wall_time"]),
                "shifted_gmean_nodes": shifted_geomean(group["nodes"]),
                "par2": float(group["par2_time"].mean()),
                "median_wall_time": float(group["wall_time"].median()),
                "median_nodes": float(group["nodes"].median()),
                "inference_fraction": (
                    float(group["rl_inference_total"].sum() / solve_sum) if solve_sum > 0 else 0.0
                ),
                "fallback_count": int(group["fallback_count"].sum()),
                "shifted_gmean_wall_without_inference": shifted_geomean(group["wall_without_inference"]),
            }
        )
    return pd.DataFrame(rows)


def paired_contrast(data: pd.DataFrame, baseline: str, candidate: str, contrast: str) -> dict[str, object]:
    base = data[data["method"] == baseline][KEYS + ["par2_time", "nodes", "optimal"]]
    trial = data[data["method"] == candidate][KEYS + ["par2_time", "nodes", "optimal"]]
    paired = base.merge(trial, on=KEYS, suffixes=("_base", "_candidate"))
    ratios = paired["par2_time_base"] / paired["par2_time_candidate"]
    finite = ratios[np.isfinite(ratios) & (ratios > 0)]
    joint = paired[paired["optimal_base"] & paired["optimal_candidate"]]
    node_ratios = joint["nodes_base"] / joint["nodes_candidate"].replace(0, np.nan)
    return {
        "contrast": contrast,
        "baseline": baseline,
        "candidate": candidate,
        "paired_runs": len(paired),
        "paired_time_speedup": float(np.exp(np.log(finite).mean())) if len(finite) else math.nan,
        "wins": int((ratios > 1.0 + 1e-12).sum()),
        "ties": int(np.isclose(ratios, 1.0, rtol=1e-12, atol=1e-12).sum()),
        "losses": int((ratios < 1.0 - 1e-12).sum()),
        "jointly_solved": len(joint),
        "shifted_gmean_node_speedup": shifted_geomean(node_ratios, shift=0.0),
    }


def save_bar(data: pd.DataFrame, value: str, title: str, ylabel: str, path: Path) -> None:
    labels = data["method"].str.replace("rl-gcnn-", "", regex=False).tolist()
    positions = np.arange(len(data))
    plt.figure(figsize=(max(6.5, len(data) * 1.15), 4.2))
    plt.bar(positions, data[value], color="#3A71A8")
    plt.xticks(positions, labels, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = parse_args()
    output = args.output_dir
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    main_data = prepare(pd.read_csv(args.main))
    model_data = prepare(pd.read_csv(args.models))
    depth_data = prepare(pd.read_csv(args.depth))

    main_subset = main_data[main_data["method"].isin(["default", "rl-mlp", "rl-gcnn"])]
    summaries = pd.concat(
        [
            aggregate(main_subset, "architecture-and-protocol"),
            aggregate(model_data, "model-ablation"),
            aggregate(depth_data, "depth-ablation"),
        ],
        ignore_index=True,
    )
    summaries.to_csv(output / "ablation_summary.csv", index=False)

    contrasts = [
        paired_contrast(
            main_subset[main_subset["protocol"] == protocol],
            "rl-mlp",
            "rl-gcnn",
            f"MLP vs GCNN ({protocol})",
        )
        for protocol in sorted(main_subset["protocol"].unique())
    ]
    contrasts.extend(
        [
            paired_contrast(model_data, "rl-gcnn-1step", "rl-gcnn-3step", "one-step vs three-step"),
            paired_contrast(model_data, "rl-gcnn-3step", "rl-gcnn-hlgauss", "scalar vs HL-Gauss"),
            paired_contrast(
                model_data, "rl-gcnn-no-categories", "rl-gcnn-3step", "without vs with variable categories"
            ),
            paired_contrast(model_data, "rl-gcnn-no-global", "rl-gcnn-3step", "without vs with global features"),
        ]
    )
    pd.DataFrame(contrasts).to_csv(output / "ablation_contrasts.csv", index=False)

    model_summary = aggregate(model_data, "model-ablation")
    depth_summary = aggregate(depth_data, "depth-ablation")
    save_bar(
        model_summary,
        "par2",
        "GCNN training and feature ablations",
        "Mean PAR-2 time (s)",
        figures / "ablation_model.png",
    )
    save_bar(
        depth_summary,
        "par2",
        "RL depth and relpscost fallback",
        "Mean PAR-2 time (s)",
        figures / "ablation_depth.png",
    )

    protocol = aggregate(main_subset, "architecture-and-protocol")
    pivot = protocol.pivot(index="method", columns="protocol", values="par2")
    pivot.plot(kind="bar", figsize=(7.2, 4.3), color=["#287271", "#C17C3A"])
    plt.ylabel("Mean PAR-2 time (s)")
    plt.xlabel("")
    plt.title("DFS-controlled vs production SCIP")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures / "ablation_protocol.png", dpi=180, bbox_inches="tight")
    plt.close()

    rl_rows = pd.concat([main_subset, model_data, depth_data], ignore_index=True)
    rl_rows = rl_rows[rl_rows["method"].str.startswith("rl-")]
    inference = rl_rows.groupby("method", sort=False).agg(
        actual=("wall_time", shifted_geomean),
        simulated_without_inference=("wall_without_inference", shifted_geomean),
    )
    inference.plot(kind="bar", figsize=(9.5, 4.5), color=["#3B4A54", "#D1495B"])
    plt.ylabel("Shifted geometric mean wall time (s)")
    plt.xlabel("")
    plt.title("Measured wall time and zero-inference lower bound")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures / "ablation_inference_simulated.png", dpi=180, bbox_inches="tight")
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
