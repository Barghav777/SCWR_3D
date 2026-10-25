from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon, studentized_range


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TARGET_ORDER = [
    "density_kg_m3",
    "temperature_c",
    "turbulence_eddy_dissipation_m2_s3",
    "turbulence_kinetic_energy_m2_s2",
    "velocity_u_m_s",
    "velocity_v_m_s",
    "velocity_w_m_s",
]
PRESSURE_TARGET = "pressure_pa"


def operating_pressure_bar(case_id: str) -> float:
    match = re.search(r"_(\d+(?:\.\d+)?)_bar_", case_id)
    if match is None:
        raise ValueError(f"Could not parse operating pressure from case ID {case_id}.")
    return float(match.group(1))


def summarize_case_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "protocol",
        "model",
        "case_id",
        "target",
        "n_nodes",
        "mae",
        "rmse",
        "mae_over_std",
        "r2",
        "sum_y",
        "sum_y2",
        "sum_sq_err",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Kaggle case-metric file is missing columns: {sorted(missing)}")
    frame = metrics.loc[metrics["protocol"] == "kfold"].copy()
    frame = frame.loc[frame["model"] != "Lattice floor (oracle)"]
    frame = frame.loc[frame["target"] != PRESSURE_TARGET]
    complete = (
        frame.groupby(["model", "target"])["case_id"].nunique().rename("n_cases")
    )
    frame = frame.join(complete, on=["model", "target"])
    frame = frame.loc[frame["n_cases"] == 42].copy()
    if frame.empty:
        raise ValueError("No complete 42-case model/target results were found.")
    frame["pressure_bar"] = frame["case_id"].map(operating_pressure_bar)
    output = []
    rng = np.random.default_rng(2026)
    for (model, target), group in frame.groupby(["model", "target"], sort=True):
        group = group.sort_values("case_id")
        mae = group["mae"].to_numpy(float)
        rmse = group["rmse"].to_numpy(float)
        nmae = group["mae_over_std"].to_numpy(float)
        r2 = group["r2"].to_numpy(float)
        boot_mae = np.empty(2000)
        boot_median_nmae = np.empty(2000)
        for b in range(2000):
            idx = rng.integers(0, len(group), len(group))
            boot_mae[b] = mae[idx].mean()
            boot_median_nmae[b] = np.median(nmae[idx])
        n_nodes = group["n_nodes"].to_numpy(float)
        n = int(n_nodes.sum())
        sum_y = float(group["sum_y"].sum())
        sum_y2 = float(group["sum_y2"].sum())
        sse = float(group["sum_sq_err"].sum())
        pooled_gauge = 1.0 - sse / max(sum_y2 - sum_y**2 / n, 1e-24)
        centered_sst = float(
            (group["sum_y2"] - group["sum_y"] ** 2 / group["n_nodes"]).sum()
        )
        pooled_centered = 1.0 - sse / max(centered_sst, 1e-24)
        output.append(
            {
                "model": model,
                "target": target,
                "n_cases": len(group),
                "case_macro_mae": mae.mean(),
                "case_macro_mae_ci_lo": np.quantile(boot_mae, 0.025),
                "case_macro_mae_ci_hi": np.quantile(boot_mae, 0.975),
                "case_macro_rmse": rmse.mean(),
                "median_nmae": np.median(nmae),
                "q25_nmae": np.quantile(nmae, 0.25),
                "q75_nmae": np.quantile(nmae, 0.75),
                "median_nmae_ci_lo": np.quantile(boot_median_nmae, 0.025),
                "median_nmae_ci_hi": np.quantile(boot_median_nmae, 0.975),
                "median_case_r2": np.median(r2),
                "q25_case_r2": np.quantile(r2, 0.25),
                "q75_case_r2": np.quantile(r2, 0.75),
                "pooled_r2_gauge": pooled_gauge,
                "pooled_r2_case_centered": pooled_centered,
            }
        )
    return pd.DataFrame(output)


def summarize_pressure_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "protocol", "model", "case_id", "pressure_level_abs_error",
        "pressure_level_sq_error", "pressure_residual_mae",
        "pressure_residual_rmse", "pressure_residual_nrmse",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(
            "Kaggle output lacks pressure level/residual metrics. "
            f"Missing columns: {sorted(missing)}. Regenerate and rerun the notebook."
        )
    pressure = metrics.loc[
        (metrics["protocol"] == "kfold")
        & (metrics["target"] == PRESSURE_TARGET)
        & (metrics["model"] != "Lattice floor (oracle)")
    ].copy()
    output = []
    for model, group in pressure.groupby("model", sort=True):
        if group["case_id"].nunique() != 42:
            continue
        output.append(
            {
                "model": model,
                "n_cases": int(group["case_id"].nunique()),
                "case_level_mean_pressure_mae_pa": group[
                    "pressure_level_abs_error"
                ].mean(),
                "case_level_mean_pressure_rmse_pa": np.sqrt(
                    group["pressure_level_sq_error"].mean()
                ),
                "centered_residual_mae_pa": group["pressure_residual_mae"].mean(),
                "centered_residual_rmse_pa": np.sqrt(
                    np.square(group["pressure_residual_rmse"]).mean()
                ),
                "centered_residual_nrmse_median": group[
                    "pressure_residual_nrmse"
                ].median(),
            }
        )
    return pd.DataFrame(output)


def statistics_and_plot(case_metrics: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    complete = case_metrics.loc[
        (case_metrics["protocol"] == "kfold")
        & (case_metrics["model"] != "Lattice floor (oracle)")
    ].copy()
    for target in TARGET_ORDER:
        target_frame = complete.loc[complete["target"] == target]
        matrix = target_frame.pivot(index="case_id", columns="model", values="mae_over_std")
        matrix = matrix.dropna(axis=0, how="any").dropna(axis=1, how="any")
        if matrix.shape[1] < 2 or matrix.shape[0] < 3:
            continue
        values = matrix.to_numpy()
        ranks = np.apply_along_axis(rankdata, 1, values)
        mean_ranks = ranks.mean(axis=0)
        statistic, p_value = friedmanchisquare(*[values[:, i] for i in range(values.shape[1])])
        pd.DataFrame(
            {
                "target": target,
                "model": matrix.columns,
                "n_cases": len(matrix),
                "mean_rank_by_case_nmae": mean_ranks,
                "friedman_statistic": statistic,
                "friedman_p_value": p_value,
            }
        ).to_csv(output_dir / f"friedman_{target}.csv", index=False)

        k, n = values.shape[1], values.shape[0]
        cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
        order = np.argsort(mean_ranks)
        fig, ax = plt.subplots(figsize=(10, max(3.5, 0.36 * k)))
        positions = np.arange(k)
        ordered_ranks = mean_ranks[order]
        ax.scatter(ordered_ranks, positions, color="#176b87", zorder=3)
        for y_pos, model_index in zip(positions, order):
            ax.text(ordered_ranks[np.where(order == model_index)[0][0]] + 0.04, y_pos, matrix.columns[model_index], va="center", fontsize=8)
        ax.axvline(1, color="#777777", linewidth=0.8)
        ax.set_xlim(0.8, k + 1.2)
        ax.set_ylim(-1, k)
        ax.set_yticks([])
        ax.set_xlabel("Mean Friedman rank (lower is better)")
        ax.set_title(f"Case-level nMAE ranks: {target} (Nemenyi CD={cd:.2f}, n={n})")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"critical_difference_{target}.png", dpi=180)
        fig.savefig(output_dir / f"critical_difference_{target}.pdf")
        plt.close(fig)

        reference = matrix.columns[int(np.argmin(mean_ranks))]
        p_raw = []
        comparisons = []
        for candidate in matrix.columns:
            if candidate == reference:
                continue
            result = wilcoxon(
                matrix[reference].to_numpy(),
                matrix[candidate].to_numpy(),
                alternative="two-sided",
                zero_method="wilcox",
            )
            p_raw.append(float(result.pvalue))
            comparisons.append(candidate)
        sorted_indices = np.argsort(p_raw)
        adjusted = np.empty(len(p_raw))
        running = 0.0
        m = len(p_raw)
        for rank, index in enumerate(sorted_indices):
            running = max(running, min(1.0, (m - rank) * p_raw[index]))
            adjusted[index] = running
        pd.DataFrame(
            {
                "target": target,
                "reference_lowest_mean_rank": reference,
                "comparator": comparisons,
                "wilcoxon_p": p_raw,
                "holm_adjusted_p": adjusted,
            }
        ).to_csv(output_dir / f"wilcoxon_holm_{target}.csv", index=False)

    heat = summary.pivot(index="model", columns="target", values="median_nmae")
    heat = heat.reindex(columns=[t for t in TARGET_ORDER if t in heat.columns])
    fig, ax = plt.subplots(figsize=(12, max(6, 0.42 * len(heat))))
    image = ax.imshow(np.log10(np.maximum(heat.to_numpy(float), 1e-6)), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(heat.columns)), [t.replace("_", "\n") for t in heat.columns], fontsize=8)
    ax.set_yticks(np.arange(len(heat.index)), heat.index, fontsize=8)
    ax.set_title("CO₂-candidate OOF median case nMAE by model and target")
    fig.colorbar(image, ax=ax, label="log₁₀ median case nMAE")
    fig.tight_layout()
    fig.savefig(output_dir / "median_nmae_heatmap.png", dpi=180)
    fig.savefig(output_dir / "median_nmae_heatmap.pdf")
    paper_graphs = RESULTS / "graphs"
    paper_graphs.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_graphs / "clean_full_median_nmae_heatmap.png", dpi=220)
    fig.savefig(paper_graphs / "clean_full_median_nmae_heatmap.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Downloaded Kaggle /kaggle/working contents.")
    parser.add_argument("--seed", type=int, choices=(2026, 2027, 2028), required=True)
    args = parser.parse_args()
    source = args.input_dir.resolve()
    metrics_path = source / "all_case_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected Kaggle output {metrics_path}")
    output_dir = RESULTS / "clean_gpu_runs" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(metrics_path)
    raw.to_csv(output_dir / "all_case_metrics.csv", index=False)
    summary = summarize_case_metrics(raw)
    summary.to_csv(output_dir / "model_summary.csv", index=False)
    pressure_summary = summarize_pressure_metrics(raw)
    pressure_summary.to_csv(output_dir / "pressure_summary.csv", index=False)
    statistics_and_plot(raw, summary, output_dir)

    for filename in [
        "run_status.csv",
        "training_history.csv",
        "split_assignments.csv",
        "case_prediction_selection.csv",
        "prediction_maps_main_targets.png",
        "prediction_maps_main_targets.pdf",
        "prediction_maps_supplement_targets.png",
        "prediction_maps_supplement_targets.pdf",
        "kfold_model_comparison.csv",
        "geometry_holdout_relative_mae.csv",
    ]:
        candidate = source / filename
        if candidate.exists():
            destination = output_dir / filename
            if candidate.resolve() != destination.resolve():
                shutil.copy2(candidate, destination)
    print(f"Ingested Kaggle seed {args.seed} outputs from {source}")
    print(f"Complete model/target rows: {len(summary)}")
    print("Field summary:", output_dir / "model_summary.csv")
    print("Pressure summary:", output_dir / "pressure_summary.csv")


if __name__ == "__main__":
    main()
