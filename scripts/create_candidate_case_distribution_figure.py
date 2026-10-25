from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA_PATH = ROOT / "mega_dataset_co2_candidate.parquet"
if not DATA_PATH.exists():
    DATA_PATH = ROOT / "mega_dataset_co2_only.parquet"

TARGETS = [
    ("density_kg_m3", "Density (kg m$^{-3}$)", None),
    ("pressure_pa", "Gauge pressure (MPa)", 1e6),
    ("temperature_c", "Temperature ($^\\circ$C)", None),
    (
        "turbulence_eddy_dissipation_m2_s3",
        "TED (m$^2$ s$^{-3}$)",
        None,
    ),
    (
        "turbulence_kinetic_energy_m2_s2",
        "TKE (m$^2$ s$^{-2}$)",
        None,
    ),
    ("velocity_u_m_s", "Velocity $u$ (m s$^{-1}$)", None),
    ("velocity_v_m_s", "Velocity $v$ (m s$^{-1}$)", None),
    ("velocity_w_m_s", "Velocity $w$ (m s$^{-1}$)", None),
]
CONFIGURATIONS = ["HHHC", "HHVC", "VHVC"]
COLORS = {"HHHC": "#4477AA", "HHVC": "#EE6677", "VHVC": "#228833"}


def main() -> None:
    columns = ["case_id", "configuration", *[target for target, _, _ in TARGETS]]
    frame = pd.read_parquet(DATA_PATH, columns=columns)
    summaries = frame.groupby(
        ["case_id", "configuration"], sort=True, observed=True
    ).median(numeric_only=True).reset_index()
    if len(summaries) != 42:
        raise ValueError(f"Expected 42 case summaries, found {len(summaries)}.")

    fig, axes = plt.subplots(4, 2, figsize=(13, 15), constrained_layout=True)
    rng = np.random.default_rng(2026)
    for ax, (target, label, scale) in zip(axes.flat, TARGETS):
        values = summaries[target].to_numpy(float)
        if scale is not None:
            values /= scale
        summaries[target] = values
        grouped = [
            summaries.loc[summaries["configuration"] == configuration, target].dropna().to_numpy()
            for configuration in CONFIGURATIONS
        ]
        boxplot = ax.boxplot(
            grouped,
            positions=np.arange(1, len(CONFIGURATIONS) + 1),
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.3},
            boxprops={"linewidth": 1.0},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )
        for patch, configuration in zip(boxplot["boxes"], CONFIGURATIONS):
            patch.set_facecolor(COLORS[configuration])
            patch.set_alpha(0.28)
        for position, (configuration, group) in enumerate(
            zip(CONFIGURATIONS, grouped), start=1
        ):
            ax.scatter(
                position + rng.uniform(-0.09, 0.09, len(group)),
                group,
                s=21,
                color=COLORS[configuration],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.88,
                zorder=3,
            )
        ax.set_xticks(range(1, len(CONFIGURATIONS) + 1), CONFIGURATIONS)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        if target in {
            "turbulence_eddy_dissipation_m2_s3",
            "turbulence_kinetic_energy_m2_s2",
        }:
            ax.set_yscale("symlog", linthresh=1e-6)
    fig.suptitle(
        "Case-level median CFD fields on CO$_2$-candidate nodes",
        fontsize=15,
        fontweight="bold",
    )
    output = RESULTS / "graphs"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "case_median_target_distributions_co2_candidate.png", dpi=220)
    fig.savefig(output / "case_median_target_distributions_co2_candidate.pdf")
    plt.close(fig)
    print(f"Saved case medians for {len(summaries)} cases from {DATA_PATH}.")


if __name__ == "__main__":
    main()
