from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TARGETS = [
    ("density_kg_m3", "Density"),
    ("pressure_pa", "Gauge pressure"),
    ("temperature_c", "Temperature"),
    ("turbulence_eddy_dissipation_m2_s3", "TED"),
    ("turbulence_kinetic_energy_m2_s2", "TKE"),
    ("velocity_u_m_s", "Velocity $u$"),
    ("velocity_v_m_s", "Velocity $v$"),
    ("velocity_w_m_s", "Velocity $w$"),
]
MODEL_ORDER = [
    "Mean baseline",
    "Ridge",
    "Random Forest",
    "Extra Trees",
    "XGBoost",
    "CatBoost",
]
MODEL_COLORS = dict(
    zip(MODEL_ORDER, plt.get_cmap("tab10")(np.linspace(0, 1, len(MODEL_ORDER))))
)


def main() -> None:
    metrics = pd.read_csv(RESULTS / "co2_candidate_case_and_random_node_metrics.csv")
    summary = pd.read_csv(RESULTS / "co2_candidate_cpu_benchmark_summary.csv")
    case_metrics = metrics.loc[metrics["protocol"] == "case_disjoint"]

    fig, axes = plt.subplots(4, 2, figsize=(16, 17), constrained_layout=True)
    for ax, (target, label) in zip(axes.flat, TARGETS):
        panel = case_metrics.loc[case_metrics["target"] == target]
        groups = [
            panel.loc[panel["model"] == model, "nmae"].dropna().to_numpy(float)
            for model in MODEL_ORDER
        ]
        boxes = ax.boxplot(
            groups,
            positions=np.arange(1, len(MODEL_ORDER) + 1),
            widths=0.68,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.25},
            whiskerprops={"linewidth": 0.9},
            capprops={"linewidth": 0.9},
        )
        for box, model in zip(boxes["boxes"], MODEL_ORDER):
            box.set_facecolor(MODEL_COLORS[model])
            box.set_alpha(0.35)
        for position, (model, values) in enumerate(zip(MODEL_ORDER, groups), start=1):
            ax.scatter(
                position + np.linspace(-0.11, 0.11, len(values)),
                values,
                s=13,
                color=MODEL_COLORS[model],
                alpha=0.68,
                edgecolors="none",
                zorder=3,
            )
        ax.set_yscale("log")
        ax.set_ylabel("Case nMAE (log scale)")
        ax.set_title(label)
        ax.set_xticks(range(1, len(MODEL_ORDER) + 1), MODEL_ORDER, rotation=28, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Clean-data case-disjoint errors by target and model",
        fontsize=15,
        fontweight="bold",
    )
    output = RESULTS / "graphs"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "co2_candidate_case_nmae_by_model_target.png", dpi=220)
    fig.savefig(output / "co2_candidate_case_nmae_by_model_target.pdf")
    plt.close(fig)

    common_models = [
        model
        for model in MODEL_ORDER
        if set(summary.loc[summary["protocol"] == "random_node", "model"]) >= {model}
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, (target, label) in zip(axes.flat, TARGETS):
        panel = summary.loc[
            summary["target"].eq(target)
            & summary["protocol"].isin(["case_disjoint", "random_node"])
            & summary["model"].isin(common_models)
        ]
        for model in common_models:
            values = (
                panel.loc[panel["model"] == model]
                .set_index("protocol")["median_nmae"]
                .reindex(["random_node", "case_disjoint"])
            )
            ax.plot(
                [0, 1],
                values.to_numpy(float),
                marker="o",
                linewidth=1.2,
                markersize=4,
                color=MODEL_COLORS[model],
                label=model,
            )
        ax.set_yscale("log")
        ax.set_xticks([0, 1], ["Random-node", "Case-disjoint"], rotation=18)
        ax.set_title(label)
        ax.set_ylabel("Median case nMAE")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=len(common_models),
        frameon=False,
    )
    fig.suptitle(
        "Random-node splits versus case-disjoint validation",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output / "random_node_vs_case_disjoint_nmae.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "random_node_vs_case_disjoint_nmae.pdf", bbox_inches="tight")
    plt.close(fig)

    pressure = case_metrics.loc[
        case_metrics["target"].eq("pressure_pa") & case_metrics["model"].eq("Mean baseline")
    ].drop_duplicates("case_id")
    pressure["power_w"] = pressure["case_id"].str.extract(
        r"_P(\d+(?:\.\d+)?)_"
    )[0].astype(float)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for configuration, group in pressure.groupby(
        pressure["case_id"].str.extract(r"^([A-Z]+)_")[0], sort=True
    ):
        ax.scatter(
            group["power_w"],
            group["case_mean_y"] / 1e6,
            s=36,
            label=configuration,
            alpha=0.85,
        )
    ax.set_xlabel("Filename-derived heater power (W)")
    ax.set_ylabel("Case-mean gauge pressure (MPa)")
    ax.set_title("Case-mean gauge pressure versus heater power")
    ax.grid(alpha=0.25)
    ax.legend(title="Configuration", frameon=False)
    fig.savefig(output / "co2_candidate_case_mean_gauge_pressure_vs_power.png", dpi=220)
    fig.savefig(output / "co2_candidate_case_mean_gauge_pressure_vs_power.pdf")
    plt.close(fig)
    print("Saved candidate-data CPU benchmark and pressure diagnostic figures.")


if __name__ == "__main__":
    main()
