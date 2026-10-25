from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from CoolProp.CoolProp import PropsSI


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PARQUET = ROOT / "mega_dataset.parquet"
TARGETS = [
    "density_kg_m3",
    "pressure_pa",
    "temperature_c",
    "turbulence_eddy_dissipation_m2_s3",
    "turbulence_kinetic_energy_m2_s2",
    "velocity_u_m_s",
    "velocity_v_m_s",
    "velocity_w_m_s",
]
EOS_COLUMNS = [
    "case_id",
    "configuration",
    "pressure_bar",
    "pressure_pa",
    "temperature_c",
    "density_kg_m3",
]


def temperature_audit(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id, group in data.groupby("case_id", sort=True):
        match = re.search(r"_T_(-?[\d.]+)$", case_id)
        if match is None:
            raise ValueError(f"Could not parse filename temperature from {case_id}")
        label = float(match.group(1))
        minimum_c = float(group["temperature_c"].min())
        rows.append(
            {
                "case_id": case_id,
                "filename_temperature_parameter": label,
                "minimum_node_temperature_c": minimum_c,
                "minimum_node_temperature_k": minimum_c + 273.15,
                "median_node_temperature_c": float(group["temperature_c"].median()),
                "maximum_node_temperature_c": float(group["temperature_c"].max()),
                "minimum_kelvin_minus_filename_label": minimum_c + 273.15 - label,
                "corrected_filename_temperature_parameter": (
                    305.0 if case_id.endswith("_T_205") else label
                ),
            }
        )
    return pd.DataFrame(rows)


def eos_audit(data: pd.DataFrame, samples_per_case: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = []
    for case_id, group in data.groupby("case_id", sort=True):
        sample = group.sample(min(samples_per_case, len(group)), random_state=2026)
        temperature_k = sample["temperature_c"].to_numpy() + 273.15
        gauge_pressure = sample["pressure_pa"].to_numpy()
        operating_pressure = sample["pressure_bar"].to_numpy() * 1e5
        density = sample["density_kg_m3"].to_numpy()
        co2_density = np.asarray(
            PropsSI(
                "D",
                "T",
                temperature_k,
                "P",
                gauge_pressure + operating_pressure,
                "CO2",
            )
        )
        water_density_1atm = np.full(len(sample), np.nan)
        water_density_gauge_plus_atm = np.full(len(sample), np.nan)
        water_valid = (temperature_k > 273.16) & (temperature_k < 647.0)
        for index in np.flatnonzero(water_valid):
            try:
                water_density_1atm[index] = PropsSI(
                    "D",
                    "T",
                    float(temperature_k[index]),
                    "P",
                    101325.0,
                    "Water",
                )
                water_density_gauge_plus_atm[index] = PropsSI(
                    "D",
                    "T",
                    float(temperature_k[index]),
                    "P",
                    float(max(gauge_pressure[index] + 101325.0, 1e4)),
                    "Water",
                )
            except ValueError:
                continue
        co2_residual = np.abs(density - co2_density) / co2_density
        water_residual = np.abs(density - water_density_1atm) / water_density_1atm
        water_gauge_residual = (
            np.abs(density - water_density_gauge_plus_atm) / water_density_gauge_plus_atm
        )
        samples.append(
            pd.DataFrame(
                {
                    "case_id": case_id,
                    "configuration": sample["configuration"].to_numpy(),
                    "density_cfd_kg_m3": density,
                    "density_co2_eos_kg_m3": co2_density,
                    "relative_error_co2_eos": co2_residual,
                    "density_water_eos_1atm_kg_m3": water_density_1atm,
                    "relative_error_water_eos_1atm": water_residual,
                    "relative_error_water_eos_gauge_plus_1atm": water_gauge_residual,
                    "temperature_k": temperature_k,
                    "pressure_gauge_pa": gauge_pressure,
                    "pressure_abs_co2_pa": gauge_pressure + operating_pressure,
                    "water_eos_1atm_closer": water_residual < co2_residual,
                    "high_density_water_candidate": density > 950.0,
                }
            )
        )
    sample_df = pd.concat(samples, ignore_index=True)
    summary = (
        sample_df.groupby(["case_id", "configuration"], as_index=False)
        .agg(
            sample_nodes=("relative_error_co2_eos", "size"),
            median_relative_error_co2=("relative_error_co2_eos", "median"),
            p95_relative_error_co2=(
                "relative_error_co2_eos",
                lambda values: values.quantile(0.95),
            ),
            fraction_co2_eos_within_5pct=(
                "relative_error_co2_eos",
                lambda values: (values < 0.05).mean(),
            ),
            fraction_water_eos_1atm_within_5pct=(
                "relative_error_water_eos_1atm",
                lambda values: (values < 0.05).mean(),
            ),
            fraction_water_eos_1atm_closer=("water_eos_1atm_closer", "mean"),
            fraction_density_above_950=("high_density_water_candidate", "mean"),
        )
    )
    return sample_df, summary


def write_co2_only(data: pd.DataFrame) -> pd.DataFrame:
    co2 = data.loc[data["density_kg_m3"] <= 950.0].copy()
    co2.loc[co2["case_id"].str.endswith("_T_205"), "temperature_parameter_value"] = 305.0
    output_path = ROOT / "mega_dataset_co2_candidate.parquet"
    co2.to_parquet(output_path, index=False, compression="zstd")
    summary = (
        co2.groupby(["case_id", "configuration", "geometry_class"], as_index=False)
        .agg(
            co2_candidate_nodes=("density_kg_m3", "size"),
            density_min_kg_m3=("density_kg_m3", "min"),
            density_median_kg_m3=("density_kg_m3", "median"),
            density_max_kg_m3=("density_kg_m3", "max"),
            temperature_min_c=("temperature_c", "min"),
            temperature_median_c=("temperature_c", "median"),
            temperature_max_c=("temperature_c", "max"),
            loop_x_extent_m=("loop_x_extent_m", "first"),
            loop_y_extent_m=("loop_y_extent_m", "first"),
            corrected_temperature_parameter=("temperature_parameter_value", "first"),
        )
    )
    original_counts = data.groupby("case_id").size().rename("original_retained_nodes")
    summary = summary.join(original_counts, on="case_id")
    summary["excluded_high_density_nodes"] = (
        summary["original_retained_nodes"] - summary["co2_candidate_nodes"]
    )
    summary["excluded_node_fraction"] = (
        summary["excluded_high_density_nodes"] / summary["original_retained_nodes"]
    )
    summary.to_csv(RESULTS / "mega_dataset_co2_candidate_case_summary.csv", index=False)
    return summary


def plane_audit(data: pd.DataFrame) -> pd.DataFrame:
    large_case = (
        data.loc[data["geometry_class"] == "large", "case_id"].drop_duplicates().sort_values()
    )
    if large_case.empty:
        raise ValueError("No large-loop case exists for the plane audit.")
    case_id = large_case.iloc[0]
    large = data.loc[data["case_id"] == case_id]
    grouped = (
        large.groupby("plane_id", as_index=False)
        .agg(
            unique_nodes=("plane_id", "size"),
            x_min_m=("x_m", "min"),
            x_max_m=("x_m", "max"),
            y_min_m=("y_m", "min"),
            y_max_m=("y_m", "max"),
            z_min_m=("z_m", "min"),
            z_max_m=("z_m", "max"),
        )
    )
    grouped["axis"] = grouped["plane_id"].str.extract(r"Plane_([XY])_", expand=False)
    grouped["station_coordinate_m"] = grouped.apply(
        lambda row: row["x_min_m"] if row["axis"] == "X" else row["y_min_m"], axis=1
    )
    grouped["other_axis_extent_m"] = grouped.apply(
        lambda row: (
            row["y_max_m"] - row["y_min_m"]
            if row["axis"] == "X"
            else row["x_max_m"] - row["x_min_m"]
        ),
        axis=1,
    )
    grouped["cross_section_family"] = grouped["axis"].map(
        {"X": "constant-x cut across y-oriented legs", "Y": "constant-y cut across x-oriented legs"}
    )
    grouped.insert(0, "representative_case_id", case_id)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-case", type=int, default=500)
    args = parser.parse_args()
    if args.samples_per_case < 1:
        raise ValueError("--samples-per-case must be positive.")

    data = pd.read_parquet(PARQUET)
    temperature = temperature_audit(data)
    temperature.to_csv(RESULTS / "temperature_label_audit.csv", index=False)

    fluid_counts = (
        data.assign(high_density_water_candidate=data["density_kg_m3"] > 950.0)
        .groupby(["case_id", "configuration"], as_index=False)
        .agg(
            retained_nodes=("density_kg_m3", "size"),
            high_density_water_candidate_nodes=("high_density_water_candidate", "sum"),
        )
    )
    fluid_counts["high_density_candidate_fraction"] = (
        fluid_counts["high_density_water_candidate_nodes"] / fluid_counts["retained_nodes"]
    )
    fluid_counts.to_csv(RESULTS / "fluid_candidate_counts_by_case.csv", index=False)

    sample, eos_summary = eos_audit(data[EOS_COLUMNS], args.samples_per_case)
    sample.to_csv(RESULTS / "coolprop_eos_node_sample_audit.csv", index=False)
    eos_summary.to_csv(RESULTS / "coolprop_eos_case_summary.csv", index=False)

    plane_audit(data).to_csv(RESULTS / "large_loop_plane_node_audit.csv", index=False)
    co2_summary = write_co2_only(data)
    print(
        f"Temperature audit: {len(temperature)} cases; "
        f"T_205 minimum = {temperature.loc[temperature.case_id.str.endswith('_T_205'), 'minimum_node_temperature_k'].iloc[0]:.6f} K."
    )
    print(
        f"EOS sample: {len(sample):,} nodes. "
        f"Candidate water-like high-density nodes: "
        f"{int(fluid_counts.high_density_water_candidate_nodes.sum()):,}."
    )
    print(
        f"CO2-candidate dataset: {int(co2_summary.co2_candidate_nodes.sum()):,} nodes; "
        f"written to {ROOT / 'mega_dataset_co2_candidate.parquet'}."
    )


if __name__ == "__main__":
    main()
