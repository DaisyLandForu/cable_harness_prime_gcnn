#!/usr/bin/env python3
"""Aggregate phase-8 results and generate reproducible figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEYS = ["protocol", "instance_id", "seed"]
METHOD_ORDER = ["default", "relpscost", "random", "mostinf", "strong", "rl-mlp", "rl-gcnn"]
COLORS = {
    "default": "#3B4A54",
    "relpscost": "#287271",
    "random": "#C17C3A",
    "mostinf": "#8E6C88",
    "strong": "#8C2F39",
    "rl-mlp": "#3A71A8",
    "rl-gcnn": "#D1495B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def shifted_geomean(values: pd.Series, shift: float = 1.0) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    clean = clean[np.isfinite(clean) & (clean >= 0)]
    return float(np.exp(np.mean(np.log(clean + shift))) - shift) if clean.size else math.nan


def method_order(methods) -> list[str]:
    values = list(dict.fromkeys(str(item) for item in methods))
    return [item for item in METHOD_ORDER if item in values] + sorted(set(values) - set(METHOD_ORDER))


def add_penalties(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for column in (
        "objective",
        "gap",
        "wall_time",
        "solve_time",
        "nodes",
        "rl_inference_total",
        "rl_inference_mean",
        "rl_inference_max",
        "fallback_count",
        "time_limit",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["optimal"] = result["status"].eq("optimal")
    result["par2_time"] = np.where(
        result["optimal"],
        result["wall_time"],
        2.0 * result["time_limit"],
    )
    result["inference_fraction"] = np.where(
        result["solve_time"] > 0,
        result["rl_inference_total"] / result["solve_time"],
        0.0,
    )
    return result


def bootstrap_speedup(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return math.nan, math.nan
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        sample = rng.choice(values, size=values.size, replace=True)
        estimates[index] = np.exp(np.mean(np.log(sample)))
    return tuple(float(item) for item in np.quantile(estimates, [0.025, 0.975]))


def paired_statistics(data: pd.DataFrame, samples: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for protocol in sorted(data["protocol"].unique()):
        selected = data[data["protocol"] == protocol]
        default = selected[selected["method"] == "default"][KEYS + ["par2_time", "nodes", "optimal"]]
        default = default.rename(
            columns={"par2_time": "default_time", "nodes": "default_nodes", "optimal": "default_optimal"}
        )
        for method in method_order(selected["method"].unique()):
            current = selected[selected["method"] == method][KEYS + ["par2_time", "nodes", "optimal"]]
            current = current.rename(
                columns={"par2_time": "method_time", "nodes": "method_nodes", "optimal": "method_optimal"}
            )
            paired = default.merge(current, on=KEYS, how="inner")
            speedups = (paired["default_time"] / paired["method_time"]).to_numpy(dtype=float)
            finite = speedups[np.isfinite(speedups) & (speedups > 0)]
            geometric = float(np.exp(np.mean(np.log(finite)))) if finite.size else math.nan
            low, high = bootstrap_speedup(finite, samples, rng)
            jointly = paired[paired["default_optimal"] & paired["method_optimal"]].copy()
            node_ratio = np.where(
                jointly["method_nodes"] > 0,
                jointly["default_nodes"] / jointly["method_nodes"],
                np.nan,
            )
            rows.append(
                {
                    "protocol": protocol,
                    "method": method,
                    "paired_runs": len(paired),
                    "paired_speedup": geometric,
                    "paired_speedup_ci_low": low,
                    "paired_speedup_ci_high": high,
                    "wins": int(np.sum(speedups > 1.0 + 1e-12)),
                    "ties": int(np.sum(np.isclose(speedups, 1.0, rtol=1e-12, atol=1e-12))),
                    "losses": int(np.sum(speedups < 1.0 - 1e-12)),
                    "jointly_solved": len(jointly),
                    "geomean_node_speedup_joint": shifted_geomean(pd.Series(node_ratio), shift=0.0),
                }
            )
    return pd.DataFrame(rows)


def rl_vs_random_statistics(data: pd.DataFrame, samples: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(1)
    for protocol in sorted(data["protocol"].unique()):
        selected = data[data["protocol"] == protocol]
        baseline = selected[selected["method"] == "random"][KEYS + ["par2_time", "nodes", "optimal"]]
        baseline = baseline.rename(
            columns={"par2_time": "baseline_time", "nodes": "baseline_nodes", "optimal": "baseline_optimal"}
        )
        for method in ("rl-mlp", "rl-gcnn"):
            current = selected[selected["method"] == method][KEYS + ["par2_time", "nodes", "optimal"]]
            current = current.rename(
                columns={"par2_time": "method_time", "nodes": "method_nodes", "optimal": "method_optimal"}
            )
            paired = baseline.merge(current, on=KEYS, how="inner")
            speedups = (paired["baseline_time"] / paired["method_time"]).to_numpy(dtype=float)
            finite = speedups[np.isfinite(speedups) & (speedups > 0)]
            geometric = float(np.exp(np.mean(np.log(finite)))) if finite.size else math.nan
            low, high = bootstrap_speedup(finite, samples, rng)
            jointly = paired[paired["baseline_optimal"] & paired["method_optimal"]]
            node_ratio = np.where(
                jointly["method_nodes"] > 0,
                jointly["baseline_nodes"] / jointly["method_nodes"],
                np.nan,
            )
            rows.append(
                {
                    "protocol": protocol,
                    "baseline": "random",
                    "method": method,
                    "paired_runs": len(paired),
                    "paired_speedup": geometric,
                    "paired_speedup_ci_low": low,
                    "paired_speedup_ci_high": high,
                    "wins": int(np.sum(speedups > 1.0 + 1e-12)),
                    "ties": int(np.sum(np.isclose(speedups, 1.0, rtol=1e-12, atol=1e-12))),
                    "losses": int(np.sum(speedups < 1.0 - 1e-12)),
                    "jointly_solved": len(jointly),
                    "geomean_node_speedup_joint": shifted_geomean(pd.Series(node_ratio), shift=0.0),
                }
            )
    return pd.DataFrame(rows)


def aggregate_instances(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in data.groupby(["protocol", "instance_id", "split", "size", "method"]):
        protocol, instance_id, split, size, method = key
        finite_gap = group["gap"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "protocol": protocol,
                "instance_id": instance_id,
                "split": split,
                "size": size,
                "method": method,
                "runs": len(group),
                "solved": int(group["optimal"].sum()),
                "solved_rate": float(group["optimal"].mean()),
                "shifted_gmean_wall_time": shifted_geomean(group["wall_time"]),
                "shifted_gmean_nodes": shifted_geomean(group["nodes"]),
                "mean_final_gap": float(finite_gap.mean()) if len(finite_gap) else math.nan,
                "finite_gap_runs": len(finite_gap),
                "par2": float(group["par2_time"].mean()),
                "branch_decisions": int(group["branch_decisions"].sum()),
                "rl_inference_total": float(group["rl_inference_total"].sum()),
                "fallback_count": int(group["fallback_count"].sum()),
            }
        )
    return pd.DataFrame(rows)


def aggregate(data: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    ranked = data.copy()
    ranked["rank"] = ranked.groupby(KEYS)["par2_time"].rank(method="average")
    rows = []
    group_columns = ["protocol", "split", "size", "method"]
    views = [("detail", ranked)]
    overall = ranked.copy()
    overall["split"] = "all"
    overall["size"] = "all"
    views.append(("overall", overall))
    for _, view in views:
        for key, group in view.groupby(group_columns, dropna=False):
            protocol, split, size, method = key
            ranks = group["rank"]
            paired_row = paired[
                (paired["protocol"] == protocol) & (paired["method"] == method)
            ]
            paired_values = paired_row.iloc[0].to_dict() if len(paired_row) else {}
            finite_gap = group["gap"].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "protocol": protocol,
                    "split": split,
                    "size": size,
                    "method": method,
                    "runs": len(group),
                    "solved": int(group["optimal"].sum()),
                    "solved_rate": float(group["optimal"].mean()),
                    "shifted_gmean_wall_time": shifted_geomean(group["wall_time"]),
                    "shifted_gmean_nodes": shifted_geomean(group["nodes"]),
                    "median_wall_time": float(group["wall_time"].median()),
                    "median_nodes": float(group["nodes"].median()),
                    "mean_final_gap": float(finite_gap.mean()) if len(finite_gap) else math.nan,
                    "finite_gap_runs": len(finite_gap),
                    "par2": float(group["par2_time"].mean()),
                    "average_rank": float(ranks.mean()) if len(ranks) else math.nan,
                    "rl_inference_total": float(group["rl_inference_total"].sum()),
                    "rl_inference_fraction": (
                        float(group["rl_inference_total"].sum() / group["solve_time"].sum())
                        if group["solve_time"].sum() > 0
                        else 0.0
                    ),
                    "fallback_count": int(group["fallback_count"].sum()),
                    "paired_speedup": paired_values.get("paired_speedup", math.nan),
                    "paired_speedup_ci_low": paired_values.get("paired_speedup_ci_low", math.nan),
                    "paired_speedup_ci_high": paired_values.get("paired_speedup_ci_high", math.nan),
                    "wins": paired_values.get("wins", 0),
                    "ties": paired_values.get("ties", 0),
                    "losses": paired_values.get("losses", 0),
                }
            )
    return pd.DataFrame(rows)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_training_curves(figures: Path) -> None:
    sources = {
        "MLP": Path("artifacts/models/mlp/training_history.csv"),
        "GCNN": Path("artifacts/models/gcnn/training_history.csv"),
    }
    plt.figure(figsize=(7, 4.2))
    plotted = False
    for name, path in sources.items():
        if not path.is_file():
            continue
        history = pd.read_csv(path)
        validation = history[history["event"] == "validation"].copy()
        if validation.empty:
            continue
        plt.plot(validation["gradient_step"], validation["validation_nodes"], marker="o", label=name)
        plotted = True
    if plotted:
        plt.xlabel("Gradient step")
        plt.ylabel("Validation nodes")
        plt.title("Validation training curve")
        plt.legend()
        plt.grid(alpha=0.25)
    save_figure(figures / "training_curve.png")


def plot_cactus(data: pd.DataFrame, figures: Path) -> None:
    selected = data[(data["protocol"] == "production-scip") & data["optimal"]]
    plt.figure(figsize=(7.2, 4.5))
    for method in method_order(selected["method"].unique()):
        values = np.sort(selected[selected["method"] == method]["wall_time"].dropna().to_numpy())
        if values.size:
            plt.step(values, np.arange(1, values.size + 1), where="post", label=method, color=COLORS.get(method))
    plt.xlabel("Wall time (s)")
    plt.ylabel("Solved runs")
    plt.title("Production SCIP cactus plot")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    save_figure(figures / "cactus_plot.png")


def plot_performance_profile(data: pd.DataFrame, figures: Path) -> None:
    selected = data[data["protocol"] == "production-scip"].copy()
    selected["best"] = selected.groupby(["instance_id", "seed"])["par2_time"].transform("min")
    selected["ratio"] = selected["par2_time"] / selected["best"]
    maximum = max(2.0, float(selected["ratio"].max()))
    thresholds = np.geomspace(1.0, maximum, 120)
    plt.figure(figsize=(7.2, 4.5))
    for method in method_order(selected["method"].unique()):
        ratios = selected[selected["method"] == method]["ratio"].dropna().to_numpy()
        if ratios.size:
            profile = [float(np.mean(ratios <= threshold)) for threshold in thresholds]
            plt.plot(thresholds, profile, label=method, color=COLORS.get(method))
    plt.xscale("log")
    plt.xlabel("Performance ratio to best")
    plt.ylabel("Fraction of runs")
    plt.title("Production SCIP performance profile")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    save_figure(figures / "performance_profile.png")


def paired_with_default(data: pd.DataFrame, method: str) -> pd.DataFrame:
    default = data[data["method"] == "default"].copy()
    current = data[data["method"] == method].copy()
    return default.merge(current, on=KEYS, suffixes=("_default", "_method"))


def plot_speedup_scatter(data: pd.DataFrame, figures: Path) -> None:
    selected = data[data["protocol"] == "production-scip"]
    plt.figure(figsize=(6.4, 5.0))
    for method in ("rl-mlp", "rl-gcnn"):
        paired = paired_with_default(selected, method)
        if paired.empty:
            continue
        speedup = paired["par2_time_default"] / paired["par2_time_method"]
        plt.scatter(paired["par2_time_default"], speedup, alpha=0.75, label=method, color=COLORS[method])
    plt.axhline(1.0, color="#333333", linewidth=1)
    plt.xlabel("Default PAR-2 time (s)")
    plt.ylabel("Wall-time speedup")
    plt.title("Per-run RL speedup vs default")
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure(figures / "wall_time_speedup_scatter.png")


def plot_node_time(data: pd.DataFrame, figures: Path) -> None:
    selected = data[data["protocol"] == "production-scip"]
    plt.figure(figsize=(6.4, 5.0))
    for method in ("rl-mlp", "rl-gcnn"):
        paired = paired_with_default(selected, method)
        paired = paired[paired["optimal_default"] & paired["optimal_method"]]
        if paired.empty:
            continue
        node_reduction = 1.0 - paired["nodes_method"] / paired["nodes_default"].clip(lower=1)
        time_reduction = 1.0 - paired["wall_time_method"] / paired["wall_time_default"]
        plt.scatter(node_reduction, time_reduction, alpha=0.75, label=method, color=COLORS[method])
    plt.axhline(0.0, color="#555555", linewidth=1)
    plt.axvline(0.0, color="#555555", linewidth=1)
    plt.xlabel("Node reduction")
    plt.ylabel("Wall-time reduction")
    plt.title("Node reduction vs time reduction")
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure(figures / "node_vs_time_reduction.png")


def plot_inference(data: pd.DataFrame, figures: Path) -> None:
    selected = data[data["method"].isin(["rl-mlp", "rl-gcnn"])]
    grouped = selected.groupby(["protocol", "method"])[["rl_inference_total", "solve_time"]].sum()
    fractions = (grouped["rl_inference_total"] / grouped["solve_time"]).unstack("method")
    fractions.plot(kind="bar", figsize=(7.0, 4.4), color=[COLORS.get(item) for item in fractions.columns])
    plt.ylabel("Inference / solving time")
    plt.xlabel("Protocol")
    plt.title("RL inference overhead")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.25)
    save_figure(figures / "inference_overhead.png")


def plot_id_transfer(data: pd.DataFrame, figures: Path) -> None:
    selected = data[data["protocol"] == "production-scip"].copy()
    selected["distribution"] = np.where(selected["split"].eq("transfer"), "transfer", "ID")
    rows = []
    for distribution in ("ID", "transfer"):
        subset = selected[selected["distribution"] == distribution]
        for method in ("rl-mlp", "rl-gcnn"):
            paired = paired_with_default(subset, method)
            ratio = paired["par2_time_default"] / paired["par2_time_method"]
            rows.append({"distribution": distribution, "method": method, "speedup": shifted_geomean(ratio, 0.0)})
    frame = pd.DataFrame(rows).pivot(index="distribution", columns="method", values="speedup")
    frame.plot(kind="bar", figsize=(6.8, 4.4), color=[COLORS.get(item) for item in frame.columns])
    plt.axhline(1.0, color="#333333", linewidth=1)
    plt.ylabel("Geometric mean PAR-2 speedup")
    plt.xlabel("")
    plt.title("In-distribution vs transfer")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.25)
    save_figure(figures / "id_vs_transfer.png")


def plot_scale(data: pd.DataFrame, figures: Path) -> None:
    selected = data[data["protocol"] == "production-scip"]
    table = selected.groupby(["size", "method"])["par2_time"].mean().unstack("method")
    columns = [item for item in method_order(table.columns) if item in table.columns]
    table[columns].plot(
        kind="bar",
        figsize=(8.0, 4.6),
        color=[COLORS.get(item) for item in columns],
    )
    plt.ylabel("Mean PAR-2 time (s)")
    plt.xlabel("Instance size")
    plt.title("Performance by instance size")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    save_figure(figures / "performance_by_size.png")


def main() -> None:
    args = parse_args()
    output = args.output_dir
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    data = add_penalties(pd.read_csv(args.input))
    paired = paired_statistics(data, args.bootstrap_samples)
    rl_vs_random = rl_vs_random_statistics(data, args.bootstrap_samples)
    summary = aggregate(data, paired)
    instance_summary = aggregate_instances(data)
    summary.to_csv(output / "summary.csv", index=False)
    paired.to_csv(output / "paired_comparisons.csv", index=False)
    rl_vs_random.to_csv(output / "rl_vs_random.csv", index=False)
    instance_summary.to_csv(output / "instance_summary.csv", index=False)

    plot_training_curves(figures)
    plot_cactus(data, figures)
    plot_performance_profile(data, figures)
    plot_speedup_scatter(data, figures)
    plot_node_time(data, figures)
    plot_inference(data, figures)
    plot_id_transfer(data, figures)
    plot_scale(data, figures)

    metadata = {
        "runs": len(data),
        "protocols": sorted(data["protocol"].unique()),
        "methods": method_order(data["method"].unique()),
        "figures": sorted(path.name for path in figures.glob("*.png")),
        "bootstrap_samples": args.bootstrap_samples,
        "par2_policy": "optimal uses measured wall_time; otherwise 2 * SCIP time_limit",
        "shifted_geomean_shift": 1.0,
    }
    (output / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
