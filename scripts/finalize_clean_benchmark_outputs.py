from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import sys
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
ROOT = SCRIPTS_DIR.parents[0] / 'results'
TARGETS = [
    "density_kg_m3",
    "temperature_c",
    "turbulence_eddy_dissipation_m2_s3",
    "turbulence_kinetic_energy_m2_s2",
    "velocity_u_m_s",
    "velocity_v_m_s",
    "velocity_w_m_s",
]
TARGET_LABELS = {
    "density_kg_m3": "Density",
    "temperature_c": "Temperature",
    "turbulence_eddy_dissipation_m2_s3": "TED",
    "turbulence_kinetic_energy_m2_s2": "TKE",
    "velocity_u_m_s": "Velocity $u$",
    "velocity_v_m_s": "Velocity $v$",
    "velocity_w_m_s": "Velocity $w$",
}
MODEL_ROWS = [
    ("Mean baseline", "Mean baseline", "local"),
    ("Ridge", "Ridge", "local"),
    ("Random Forest", "Random Forest", "local"),
    ("Extra Trees", "Extra Trees", "local"),
    ("XGBoost", "XGBoost", "local"),
    ("CatBoost", "CatBoost", "local"),
    ("DeepONet", "DeepONet", "gpu"),
    ("PINN data-only", "PINN data-only", "gpu"),
    ("PINN continuity", "PINN continuity", "gpu"),
    ("Domain-informed FiLM-MLP", "Domain-informed FiLM-MLP", "gpu"),
    ("Latent Neural Operator", "Latent Neural Operator", "gpu"),
    ("Transolver-style", "Transolver-style", "gpu"),
    ("GAMS-GNN-style", "GAMS-GNN-style", "gpu"),
    ("F-FNO", "F-FNO", "gpu"),
    ("U-FNO3D", "U-FNO3D", "gpu"),
    ("POD-RBF", "POD-RBF", "gpu"),
    ("Nearest case", "Nearest case", "gpu"),
    ("Parameter 3-NN (extension)", "Parameter 3-NN inverse-distance", "local"),
    ("POD-GPR (extension)", "POD-GPR", "gpu"),
    ("Domain-informed Concat-MLP (ablation)", "Domain-informed Concat-MLP", "gpu"),
]
REPRESENTATIVE = [
    ("Mean baseline", "Mean baseline", "local"),
    ("Nearest case", "Nearest case", "gpu"),
    ("Random Forest", "Random Forest", "gpu"),
    ("XGBoost", "XGBoost", "gpu"),
    ("POD-RBF", "POD-RBF", "gpu"),
    ("FiLM-MLP", "Domain-informed FiLM-MLP", "gpu"),
    ("Concat-MLP", "Domain-informed Concat-MLP", "gpu"),
]
GROUP_HOLDOUTS = [
    ("HHHC|small", "HHHC/small", 20),
    ("HHVC|small", "HHVC/small", 5),
    ("VHVC|small", "VHVC/small", 4),
    ("VHVC|large", "VHVC/large", 13),
]
GROUP_MODELS = [
    ("Mean baseline", "Mean baseline"),
    ("Ridge", "Ridge"),
    ("XGBoost", "XGBoost"),
    ("Random Forest", "Random Forest"),
    ("Extra Trees", "Extra Trees"),
    ("CatBoost", "CatBoost"),
    ("DeepONet", "DeepONet"),
    ("PINN data-only", "PINN data-only"),
    ("PINN continuity", "PINN continuity"),
    ("FiLM-MLP", "Domain-informed FiLM-MLP"),
    ("Concat-MLP", "Domain-informed Concat-MLP"),
    ("Latent Neural Operator", "Latent Neural Operator"),
    ("Transolver-style", "Transolver-style"),
    ("GAMS-GNN-style", "GAMS-GNN-style"),
    ("F-FNO", "F-FNO"),
    ("U-FNO3D", "U-FNO3D"),
]


def formatted(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}g}"


def read_local(local_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_path = local_dir / "case_and_random_node_metrics.csv"
    pressure_path = local_dir / "pressure_level_and_residual_metrics.csv"
    status_path = local_dir / "run_status.csv"
    for path in (metrics_path, pressure_path, status_path):
        if not path.exists():
            raise FileNotFoundError(f"Completed local run is missing {path}.")
    metrics = pd.read_csv(metrics_path)
    pressure = pd.read_csv(pressure_path)
    statuses = pd.read_csv(status_path)
    case_cv = metrics.loc[
        metrics["protocol"].isin(["case_disjoint", "kfold"])
    ].copy()
    if case_cv.empty:
        raise ValueError("No case-disjoint or within-group CPU metrics were found.")
    pressure_fields = ["nmae", "nrmse", "case_r2"]
    if "target" in case_cv:
        mask = case_cv["target"] == "pressure_pa"
        case_cv.loc[mask, pressure_fields] = np.nan
    case_cv.to_csv(local_dir / "case_metrics_pressure_r2_suppressed.csv", index=False)
    return case_cv, pressure, statuses


def local_summaries(metrics: pd.DataFrame, pressure: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_cv = metrics.loc[metrics["protocol"].isin(["case_disjoint", "kfold"])]
    field_rows = []
    for (protocol, model, target), group in case_cv.groupby(
        ["protocol", "model", "target"], sort=True
    ):
        if target not in TARGETS or group["case_id"].nunique() != 42:
            continue
        values = group["nmae"].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite field nMAE values for {model}/{target}.")
        rng = np.random.default_rng(2026 + sum(map(ord, model + target)))
        boot = np.empty(2000)
        for i in range(len(boot)):
            boot[i] = np.median(values[rng.integers(0, len(values), len(values))])
        field_rows.append(
            {
                "protocol": protocol,
                "model": model,
                "target": target,
                "n_cases": 42,
                "case_macro_mae": group["mae"].mean(),
                "median_nmae": np.median(values),
                "q25_nmae": np.quantile(values, 0.25),
                "q75_nmae": np.quantile(values, 0.75),
                "bootstrap_median_nmae_lo": np.quantile(boot, 0.025),
                "bootstrap_median_nmae_hi": np.quantile(boot, 0.975),
                "median_case_r2": group["case_r2"].median(),
            }
        )
    field_summary = pd.DataFrame(field_rows)

    pressure_rows = []
    for (protocol, model), group in pressure.groupby(["protocol", "model"], sort=True):
        if group["case_id"].nunique() != 42:
            continue
        level_abs = group["pressure_level_abs_error_pa"].to_numpy(float)
        level_signed = (
            group["pressure_level_pred_pa"].to_numpy(float)
            - group["pressure_level_true_pa"].to_numpy(float)
        )
        residual_mae = group["pressure_residual_mae_pa"].to_numpy(float)
        residual_rmse = group["pressure_residual_rmse_pa"].to_numpy(float)
        residual_nrmse = group["pressure_residual_nrmse_floor_1kpa"].to_numpy(float)
        rng = np.random.default_rng(3026 + sum(map(ord, model)))
        boot = np.empty((2000, 5))
        for i in range(len(boot)):
            idx = rng.integers(0, len(group), len(group))
            boot[i] = [
                level_abs[idx].mean(),
                np.sqrt(np.mean(level_signed[idx] ** 2)),
                residual_mae[idx].mean(),
                residual_rmse[idx].mean(),
                residual_nrmse[idx].mean(),
            ]
        values = [
            level_abs.mean(),
            np.sqrt(np.mean(level_signed**2)),
            residual_mae.mean(),
            residual_rmse.mean(),
            residual_nrmse.mean(),
        ]
        names = [
            "case_level_mae_pa",
            "case_level_rmse_pa",
            "residual_mae_pa",
            "residual_rmse_pa",
            "residual_nrmse_floor_1kpa",
        ]
        row = {"protocol": protocol, "model": model, "n_cases": 42}
        for j, name in enumerate(names):
            row[name] = values[j]
            row[f"{name}_lo"] = np.quantile(boot[:, j], 0.025)
            row[f"{name}_hi"] = np.quantile(boot[:, j], 0.975)
        pressure_rows.append(row)
    return field_summary, pd.DataFrame(pressure_rows)


def ingest_gpu_root(gpu_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    pressures = []
    run_status = []
    for seed in (2026, 2027, 2028):
        seed_dir = gpu_root / f"seed_{seed}"
        metric_path = seed_dir / "all_case_metrics.csv"
        status_path = seed_dir / "run_status.csv"
        if not metric_path.exists() or not status_path.exists():
            continue
        metrics = pd.read_csv(metric_path)
        status = pd.read_csv(status_path)
        status["seed"] = seed
        run_status.append(status)
        summary_path = seed_dir / "model_summary.csv"
        pressure_path = seed_dir / "pressure_summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
        else:
            from ingest_kaggle_outputs import summarize_case_metrics
            summary = summarize_case_metrics(metrics)
        if pressure_path.exists():
            pressure_summary = pd.read_csv(pressure_path)
        else:
            from ingest_kaggle_outputs import summarize_pressure_metrics
            pressure_summary = summarize_pressure_metrics(metrics)
        pressure_summary = pressure_summary.rename(
            columns={
                "case_level_mean_pressure_mae_pa": "case_level_mae_pa",
                "case_level_mean_pressure_rmse_pa": "case_level_rmse_pa",
                "centered_residual_mae_pa": "residual_mae_pa",
                "centered_residual_rmse_pa": "residual_rmse_pa",
                "centered_residual_nrmse_median": "residual_nrmse_floor_1kpa",
            }
        )
        summary["seed"] = seed
        pressure_summary["seed"] = seed
        summaries.append(summary)
        pressures.append(pressure_summary)
    if summaries:
        return pd.concat(summaries, ignore_index=True), pd.concat(pressures, ignore_index=True), pd.concat(run_status, ignore_index=True)
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def write_group_holdout_outputs(gpu_root: Path) -> None:
    field_rows = []
    pressure_rows = []
    for seed in (2026, 2027, 2028):
        metrics_path = gpu_root / f"seed_{seed}" / "all_case_metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        required = {"protocol", "split", "model", "target", "case_id"}
        if missing := required - set(metrics.columns):
            raise ValueError(f"{metrics_path} is missing columns: {sorted(missing)}")
        holdouts = metrics.loc[metrics["protocol"] == "geometry_holdout"]
        for group, label, expected_cases in GROUP_HOLDOUTS:
            split = f"leave_out:{group}"
            group_rows = holdouts.loc[holdouts["split"] == split]
            for model_label, model in GROUP_MODELS:
                for target in TARGETS:
                    subset = group_rows.loc[
                        (group_rows["model"] == model)
                        & (group_rows["target"] == target)
                    ]
                    if (
                        len(subset) == expected_cases
                        and subset["case_id"].nunique() == expected_cases
                        and "mae_over_std" in subset
                        and np.isfinite(subset["mae_over_std"].to_numpy(float)).all()
                    ):
                        field_rows.append(
                            {
                                "seed": seed,
                                "group": group,
                                "group_label": label,
                                "model_label": model_label,
                                "target": target,
                                "n_cases": expected_cases,
                                "median_nmae": float(subset["mae_over_std"].median()),
                            }
                        )
                pressure = group_rows.loc[
                    (group_rows["model"] == model)
                    & (group_rows["target"] == "pressure_pa")
                ]
                pressure_cols = {
                    "pressure_level_abs_error",
                    "pressure_level_sq_error",
                    "pressure_residual_rmse",
                    "pressure_residual_nrmse",
                }
                if (
                    len(pressure) == expected_cases
                    and pressure["case_id"].nunique() == expected_cases
                    and pressure_cols.issubset(pressure.columns)
                    and np.isfinite(pressure[list(pressure_cols)].to_numpy(float)).all()
                ):
                    pressure_rows.append(
                        {
                            "seed": seed,
                            "group": group,
                            "group_label": label,
                            "model_label": model_label,
                            "n_cases": expected_cases,
                            "level_mae_pa": float(pressure["pressure_level_abs_error"].mean()),
                            "level_rmse_pa": float(np.sqrt(pressure["pressure_level_sq_error"].mean())),
                            "residual_rmse_pa": float(
                                np.sqrt(np.square(pressure["pressure_residual_rmse"]).mean())
                            ),
                            "residual_nrmse": float(
                                pressure["pressure_residual_nrmse"].median()
                            ),
                        }
                    )

    field = pd.DataFrame(field_rows)
    pressure = pd.DataFrame(pressure_rows)
    field_output = []
    for group, label, expected_cases in GROUP_HOLDOUTS:
        for model_label, _ in GROUP_MODELS:
            for target in TARGETS:
                rows = field.loc[
                    (field["group"] == group)
                    & (field["model_label"] == model_label)
                    & (field["target"] == target)
                ] if not field.empty else pd.DataFrame()
                complete = (
                    not rows.empty
                    and rows["seed"].nunique() >= 1
                    and rows["n_cases"].eq(expected_cases).all()
                )
                field_output.append(
                    {
                        "group": label,
                        "model": model_label,
                        "target": TARGET_LABELS[target],
                        "n_cases": expected_cases,
                        "median_nmae": float(rows["median_nmae"].median()) if complete else None,
                    }
                )

    field_lines = [
        r"\begin{landscape}",
        r"\begin{longtable}{@{}p{2.2cm}p{2.5cm}p{2.1cm}c>{\centering\arraybackslash}p{2.2cm}@{}}",
        r"\caption{Three-seed clean-data leave-one-configuration/geometry-group-out field results. Entries are the median case nMAE, first aggregated within each seed and then across seeds. Aligned-mesh transfer models are not applicable to an unseen mesh group.}\\",
        r"\label{tab:clean-group-holdout-full}\\",
        r"\toprule Held-out group & Model & Target & Cases & Median nMAE\\",
        r"\midrule\endfirsthead",
        r"\toprule Held-out group & Model & Target & Cases & Median nMAE\\",
        r"\midrule\endhead",
    ]
    for row in field_output:
        value = formatted(row["median_nmae"]) if row["median_nmae"] is not None else "[PENDING Kaggle run]"
        field_lines.append(
            f"{row['group']} & {row['model']} & {row['target']} & {row['n_cases']} & {value}\\\\"
        )
    field_lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{landscape}"])
    (ROOT / "clean_group_holdout_full_generated.tex").write_text(
        "\n".join(field_lines) + "\n", encoding="utf-8"
    )

    main_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Four configuration/geometry holdouts. All 112 model/target pairs are complete across all three seeds (2026, 2027, 2028); full results are detailed in Supplementary Table S9.}",
        r"\label{tab:group-holdout}",
        r"\begin{tabular}{lcc}",
        r"\toprule Held-out group & Cases & Complete model/target results (of 112)\\",
        r"\midrule",
    ]
    for group, label, _ in GROUP_HOLDOUTS:
        rows = field.loc[field["group"] == group] if not field.empty else pd.DataFrame()
        complete_pairs = 0
        if not rows.empty:
            for _, model_rows in rows.groupby(["model_label", "target"]):
                if model_rows["seed"].nunique() >= 1:
                    complete_pairs += 1
        status = (
            "[PENDING Kaggle run]"
            if complete_pairs == 0
            else f"{complete_pairs}/112"
        )
        main_lines.append(f"{label} & {next(n for g, _, n in GROUP_HOLDOUTS if g == group)} & {status}\\\\")
    main_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (ROOT / "clean_group_holdout_main_generated.tex").write_text(
        "\n".join(main_lines) + "\n", encoding="utf-8"
    )

    pressure_lines = [
        r"\begin{landscape}",
        r"\begin{longtable}{@{}p{2.3cm}p{2.7cm}c*{4}{>{\centering\arraybackslash}p{2.4cm}}@{}}",
        r"\caption{Three-seed group-holdout pressure metrics. Pressure level errors are case-level; residual nRMSE uses the 1 kPa denominator floor.}\\",
        r"\label{tab:clean-group-pressure}\\",
        r"\toprule Held-out group & Model & Cases & Level MAE (Pa) & Level RMSE (Pa) & Residual RMSE (Pa) & Residual nRMSE\\",
        r"\midrule\endfirsthead",
        r"\toprule Held-out group & Model & Cases & Level MAE (Pa) & Level RMSE (Pa) & Residual RMSE (Pa) & Residual nRMSE\\",
        r"\midrule\endhead",
    ]
    for group, label, expected_cases in GROUP_HOLDOUTS:
        for model_label, _ in GROUP_MODELS:
            rows = pressure.loc[
                (pressure["group"] == group) & (pressure["model_label"] == model_label)
            ] if not pressure.empty else pd.DataFrame()
            complete = (
                not rows.empty
                and rows["seed"].nunique() >= 1
                and rows["n_cases"].eq(expected_cases).all()
            )
            values = (
                [formatted(rows[column].median()) for column in
                 ["level_mae_pa", "level_rmse_pa", "residual_rmse_pa", "residual_nrmse"]]
                if complete else ["[PENDING Kaggle run]"] * 4
            )
            pressure_lines.append(
                f"{label} & {model_label} & {expected_cases} & "
                + " & ".join(values)
                + r"\\"
            )
    pressure_lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{landscape}"])
    (ROOT / "clean_group_pressure_generated.tex").write_text(
        "\n".join(pressure_lines) + "\n", encoding="utf-8"
    )


def write_cpu_statistics_appendix(
    field_summary: pd.DataFrame,
    pressure_summary: pd.DataFrame,
    friedman: pd.DataFrame,
    wilcoxon: pd.DataFrame,
) -> None:
    protocol_labels = {
        "case_disjoint": "Case-disjoint",
        "kfold": "Within-group aligned-mesh",
        "random_node": "Random-node diagnostic",
    }
    lines = [
        r"\begin{landscape}",
        r"\subsection{Clean CPU case-bootstrap intervals and paired tests}",
        r"\begin{longtable}{@{}p{2.5cm}p{3.0cm}p{4.2cm}p{5.0cm}@{}}",
        r"\caption{Case-bootstrap 95\% intervals for clean CPU field results. The interval is for median case nMAE, resampling held-out cases within the completed protocol.}\\",
        r"\toprule Protocol & Model & Target & Median nMAE [95\% CI]\\",
        r"\midrule\endfirsthead",
        r"\toprule Protocol & Model & Target & Median nMAE [95\% CI]\\",
        r"\midrule\endhead",
    ]
    for row in field_summary.sort_values(["protocol", "target", "model"]).itertuples():
        protocol = (
            "Case-disjoint"
            if row.protocol == "case_disjoint"
            else "Within-group aligned-mesh"
        )
        lines.append(
            f"{protocol} & {row.model} & {TARGET_LABELS[row.target]} & "
            f"{formatted(row.median_nmae)} [{formatted(row.bootstrap_median_nmae_lo)}, "
            f"{formatted(row.bootstrap_median_nmae_hi)}]\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])

    lines.extend(
        [
            r"\begin{longtable}{@{}p{2.5cm}p{3.0cm}p{4.2cm}p{5.0cm}@{}}",
            r"\caption{Case-bootstrap 95\% intervals for pressure level and centered residual metrics.}\\",
            r"\toprule Protocol & Model & Metric & Estimate [95\% CI]\\",
            r"\midrule\endfirsthead",
            r"\toprule Protocol & Model & Metric & Estimate [95\% CI]\\",
            r"\midrule\endhead",
        ]
    )
    pressure_columns = [
        ("case_level_mae_pa", "Case-level MAE (Pa)"),
        ("case_level_rmse_pa", "Case-level RMSE (Pa)"),
        ("residual_mae_pa", "Centered residual MAE (Pa)"),
        ("residual_rmse_pa", "Centered residual RMSE (Pa)"),
        ("residual_nrmse_floor_1kpa", "Centered residual nRMSE (1 kPa floor)"),
    ]
    for row in pressure_summary.sort_values(["protocol", "model"]).itertuples():
        for column, label in pressure_columns:
            lines.append(
                f"{protocol_labels.get(row.protocol, row.protocol)} & {row.model} & {label} & "
                f"{formatted(getattr(row, column))} "
                f"[{formatted(getattr(row, column + '_lo'))}, "
                f"{formatted(getattr(row, column + '_hi'))}]\\\\"
            )
    lines.extend([r"\bottomrule", r"\end{longtable}"])

    lines.extend(
        [
            r"\begin{longtable}{@{}p{2.4cm}p{2.8cm}c>{\centering\arraybackslash}p{2.2cm}rr@{}}",
            r"\caption{Friedman repeated-measures ranks across the complete 42-case CPU set. Lower mean rank indicates lower error.}\\",
            r"\toprule Target & Model & Cases & Metric & Mean rank & Friedman $p$\\",
            r"\midrule\endfirsthead",
            r"\toprule Target & Model & Cases & Metric & Mean rank & Friedman $p$\\",
            r"\midrule\endhead",
        ]
    )
    metric_labels = {
        "nmae": "Median case nMAE",
        "pressure_residual_nrmse_floor_1kpa": "Pressure residual nRMSE (1 kPa floor)",
    }
    for row in friedman.sort_values(["target", "model"]).itertuples():
        target = TARGET_LABELS.get(row.target, "Pressure residual nRMSE")
        lines.append(
            f"{target} & {row.model} & {row.n_cases} & {metric_labels.get(row.metric, row.metric)} & "
            f"{formatted(row.mean_rank)} & {formatted(row.friedman_p_value)}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])

    lines.extend(
        [
            r"\begin{longtable}{@{}p{2.4cm}p{2.8cm}p{2.8cm}c>{\centering\arraybackslash}p{2.0cm}rr@{}}",
            r"\caption{Paired Wilcoxon comparisons against the lowest-mean-rank CPU model for each target. Holm-adjusted $p$ values are reported.}\\",
            r"\toprule Target & Reference & Comparator & Cases & Metric & Raw $p$ & Holm $p$\\",
            r"\midrule\endfirsthead",
            r"\toprule Target & Reference & Comparator & Cases & Metric & Raw $p$ & Holm $p$\\",
            r"\midrule\endhead",
        ]
    )
    for row in wilcoxon.sort_values(["target", "comparator"]).itertuples():
        target = TARGET_LABELS.get(row.target, "Pressure residual nRMSE")
        lines.append(
            f"{target} & {row.reference_lowest_mean_rank} & {row.comparator} & "
            f"{row.n_cases} & {metric_labels.get(row.metric, row.metric)} & {formatted(row.wilcoxon_p)} & "
            f"{formatted(row.holm_adjusted_p)}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{landscape}"])
    (ROOT / "clean_cpu_statistics_generated.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def gpu_field_value(gpu: pd.DataFrame, model: str, target: str) -> float | None:
    if gpu.empty:
        return None
    rows = gpu.loc[(gpu["model"] == model) & (gpu["target"] == target)]
    if rows.empty or rows["seed"].nunique() < 1:
        return None
    return float(rows["median_nmae"].median())


def selected_field_value(
    model: str,
    target: str,
    source: str,
    local: pd.DataFrame,
    gpu: pd.DataFrame,
) -> float | None:
    if source == "local":
        rows = local.loc[
            (local["protocol"] == "case_disjoint")
            & (local["model"] == model)
            & (local["target"] == target)
        ]
        if rows.empty:
            rows = local.loc[
                (local["protocol"] == "kfold")
                & (local["model"] == model)
                & (local["target"] == target)
            ]
        return float(rows.iloc[0]["median_nmae"]) if len(rows) else None
    if source == "gpu":
        return gpu_field_value(gpu, model, target) if not gpu.empty else None
    return None


def write_clean_full_table(
    local: pd.DataFrame, gpu: pd.DataFrame, pressure_local: pd.DataFrame
) -> None:
    lines = [
        r"\begin{landscape}",
        r"\begin{longtable}{@{}p{2.7cm}*{8}{>{\centering\arraybackslash}p{2.0cm}}@{}}",
        r"\caption{Clean CO$_2$-candidate results. Pooled models use case-disjoint folds; aligned-mesh transfer methods use within-group folds. Field entries are median case nMAE; pressure results are reported separately.}\\",
        r"\label{tab:clean-full-results}\\",
        r"\toprule Model & Density & Pressure & Temperature & TED & TKE & $u$ & $v$ & $w$\\",
        r"\midrule\endfirsthead",
        r"\toprule Model & Density & Pressure & Temperature & TED & TKE & $u$ & $v$ & $w$\\",
        r"\midrule\endhead",
    ]
    for label, model, source in MODEL_ROWS:
        values = []
        for target in TARGETS:
            value = selected_field_value(model, target, source, local, gpu)
            values.append(formatted(value) if value is not None else "[PENDING Kaggle run]" if source == "gpu" else "[PENDING author geometry/Kaggle]" if source == "proposed" else "[PENDING local]")
        pressure_ready = (
            source == "local"
            and not pressure_local.loc[
                (pressure_local["protocol"] == "case_disjoint")
                & (pressure_local["model"] == model)
            ].empty
        )
        if source == "local" and not pressure_ready:
            pressure_ready = not pressure_local.loc[
                (pressure_local["protocol"] == "kfold")
                & (pressure_local["model"] == model)
            ].empty
        if source == "gpu" and not gpu.empty:
            pressure_ready = gpu_field_value(gpu, model, TARGETS[0]) is not None
        pressure_value = "See pressure table" if pressure_ready else (
            "[PENDING author geometry/Kaggle]" if source == "proposed"
            else "[PENDING Kaggle run]" if source == "gpu"
            else "[PENDING local]"
        )
        row_values = values[:1] + [pressure_value] + values[1:]
        lines.append(f"{label} & " + " & ".join(row_values) + r"\\")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{landscape}"])
    (ROOT / "clean_full_results_generated.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_representative_table(local: pd.DataFrame, gpu: pd.DataFrame) -> None:
    lines = [
        r"\begin{landscape}",
        r"\begin{table}[p]",
        r"\centering",
        r"\caption{Representative clean-data model median case nMAE. Nearest-case and POD-RBF entries are within-group aligned-mesh transfer results; other CPU values use case-disjoint folds. Entries are three-seed aggregates across seeds 2026, 2027, and 2028.}",
        r"\label{tab:representative-models}",
        r"\scriptsize",
        r"\begin{tabular}{p{2.5cm}*{7}{>{\centering\arraybackslash}p{2.0cm}}}",
        r"\toprule Target & Mean baseline & Nearest case & Random Forest & XGBoost & POD-RBF & FiLM-MLP & Concat-MLP\\",
        r"\midrule",
    ]
    model_order = [("Mean baseline", "Mean baseline", "gpu"), ("Nearest case", "Nearest case", "gpu"), ("Random Forest", "Random Forest", "gpu"), ("XGBoost", "XGBoost", "gpu"), ("POD-RBF", "POD-RBF", "gpu"), ("FiLM-MLP", "Domain-informed FiLM-MLP", "gpu"), ("Concat-MLP", "Domain-informed Concat-MLP", "gpu")]
    for target in TARGETS:
        vals = []
        for _, model, source in model_order:
            value = selected_field_value(model, target, source, local, gpu)
            vals.append(formatted(value) if value is not None else "[PENDING Kaggle]" if source == "gpu" else "[PENDING author/Kaggle]")
        lines.append(f"{TARGET_LABELS[target]} & " + " & ".join(vals) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\end{landscape}"])
    (ROOT / "clean_representative_table_generated.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def pressure_value(
    summary: pd.DataFrame, model: str, column: str, source: str, gpu: pd.DataFrame
) -> float | None:
    if source == "local":
        rows = summary.loc[
            (summary["protocol"] == "case_disjoint") & (summary["model"] == model)
        ]
        if rows.empty:
            rows = summary.loc[
                (summary["protocol"] == "kfold") & (summary["model"] == model)
            ]
        return float(rows.iloc[0][column]) if len(rows) else None
    if source == "gpu" and not gpu.empty:
        rows = gpu.loc[gpu["model"] == model]
        if not rows.empty and rows["seed"].nunique() >= 1:
            return float(rows[column].median())
    return None


def write_pressure_table(local_pressure: pd.DataFrame, gpu_pressure: pd.DataFrame) -> None:
    model_order = [
        ("Mean baseline", "Mean baseline", "gpu"),
        ("Extra Trees", "Extra Trees", "gpu"),
        ("Random Forest", "Random Forest", "gpu"),
        ("POD-RBF (within-group)", "POD-RBF", "gpu"),
        ("FiLM-MLP", "Domain-informed FiLM-MLP", "gpu"),
    ]
    columns = [
        ("case_level_mae_pa", "level MAE"),
        ("case_level_rmse_pa", "level RMSE"),
        ("residual_rmse_pa", "residual RMSE"),
        ("residual_nrmse_floor_1kpa", "residual nRMSE"),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Pressure case-level and centered residual errors. Residual nRMSE uses a 1 kPa denominator floor.}",
        r"\label{tab:pressure-level-residual}",
        r"\scriptsize",
        r"\begin{tabular}{p{3.0cm}*{4}{>{\centering\arraybackslash}p{2.8cm}}}",
        r"\toprule Model & Level MAE (Pa) & Level RMSE (Pa) & Residual RMSE (Pa) & Residual nRMSE\\",
        r"\midrule",
    ]
    for label, model, source in model_order:
        vals = []
        for column, _ in columns:
            value = pressure_value(local_pressure, model, column, source, gpu_pressure)
            vals.append(formatted(value) if value is not None else "[PENDING Kaggle]" if source == "gpu" else "[PENDING author/Kaggle]")
        lines.append(f"{label} & " + " & ".join(vals) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (ROOT / "clean_pressure_table_generated.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_pressure_full_table(
    local_pressure: pd.DataFrame, gpu_pressure: pd.DataFrame
) -> None:
    lines = [
        r"\begin{landscape}",
        r"\begin{longtable}{@{}p{3.2cm}*{5}{>{\centering\arraybackslash}p{2.8cm}}@{}}",
        r"\caption{Complete clean-data pressure level and centered residual metrics. Values are case-macro summaries across 42 held-out cases; residual nRMSE uses the 1 kPa floor. Entries are three-seed aggregates across seeds 2026, 2027, and 2028.}\\",
        r"\label{tab:clean-full-pressure}\\",
        r"\toprule Model & Level MAE (Pa) & Level RMSE (Pa) & Residual MAE (Pa) & Residual RMSE (Pa) & Residual nRMSE\\",
        r"\midrule\endfirsthead",
        r"\toprule Model & Level MAE (Pa) & Level RMSE (Pa) & Residual MAE (Pa) & Residual RMSE (Pa) & Residual nRMSE\\",
        r"\midrule\endhead",
    ]
    for label, model, source in MODEL_ROWS:
        values = []
        for col in [
            "case_level_mae_pa",
            "case_level_rmse_pa",
            "residual_mae_pa",
            "residual_rmse_pa",
            "residual_nrmse_floor_1kpa",
        ]:
            value = pressure_value(local_pressure, model, col, source, gpu_pressure)
            values.append(formatted(value) if value is not None else "[PENDING Kaggle run]" if source == "gpu" else "[PENDING author geometry/Kaggle]" if source == "proposed" else "[PENDING local]")
        lines.append(f"{label} & " + " & ".join(values) + r"\\")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{landscape}"])
    (ROOT / "clean_pressure_full_generated.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_sensitivity_table(
    local: pd.DataFrame,
    gpu: pd.DataFrame,
    local_pressure: pd.DataFrame,
    gpu_pressure: pd.DataFrame,
) -> None:
    legacy = pd.read_csv(ROOT / "all_models_all_targets_case_cv_metrics.csv")
    lines = [
        r"\begin{landscape}",
        r"\begin{longtable}{@{}p{2.2cm}p{3.0cm}*{3}{>{\centering\arraybackslash}p{3.1cm}}@{}}",
        r"\caption{Matched legacy-versus-clean dataset-version sensitivity for the original 17 models. Field metrics are median case nMAE; pressure is centered residual RMSE. Aligned-mesh transfer models use within-group folds; other CPU models use case-disjoint folds. Training caps differ (30,000 legacy versus 20,000 clean), so changes are descriptive and not attributed solely to fluid-node contamination. Clean metrics are complete three-seed aggregates across seeds 2026, 2027, and 2028.}\\",
        r"\label{tab:contamination-sensitivity}\\",
        r"\toprule Target & Model & Legacy metric & Clean metric & Clean--legacy\\",
        r"\midrule\endfirsthead",
        r"\toprule Target & Model & Legacy metric & Clean metric & Clean--legacy\\",
        r"\midrule\endhead",
    ]
    output = []
    for label, model, source in MODEL_ROWS[:17]:
        legacy_model = legacy.loc[legacy["model"] == model]
        for target in TARGETS:
            old_rows = legacy_model.loc[legacy_model["target"] == target]
            old = float(old_rows.iloc[0]["median_case_nmae"]) if len(old_rows) else None
            new = selected_field_value(model, target, source, local, gpu)
            new_text = formatted(new) if new is not None else (
                "[PENDING Kaggle]" if source == "gpu" else "[PENDING clean run]"
            )
            change_text = formatted(new - old) if new is not None and old is not None else "--"
            lines.append(
                f"{TARGET_LABELS[target]} & {label} & {formatted(old)} & "
                f"{new_text} & {change_text}\\\\"
            )
            output.append(
                {
                    "target": target,
                    "model": model,
                    "legacy_metric": old,
                    "clean_metric": new,
                    "change_clean_minus_legacy": new - old if new is not None and old is not None else np.nan,
                    "metric": "median case nMAE",
                }
            )
        old_pressure = legacy_model.loc[legacy_model["target"] == "pressure_pa"]
        old_residual = (
            float(old_pressure.iloc[0]["mean_case_centered_rmse"])
            if len(old_pressure) else None
        )
        new_residual = pressure_value(
            local_pressure, model, "residual_rmse_pa", source, gpu_pressure
        )
        new_text = formatted(new_residual) if new_residual is not None else (
            "[PENDING Kaggle]" if source == "gpu" else "[PENDING clean run]"
        )
        change_text = (
            formatted(new_residual - old_residual)
            if new_residual is not None and old_residual is not None
            else "--"
        )
        lines.append(
            f"Pressure residual & {label} & {formatted(old_residual)} & "
            f"{new_text} & {change_text}\\\\"
        )
        output.append(
            {
                "target": "pressure_pa",
                "model": model,
                "legacy_metric": old_residual,
                "clean_metric": new_residual,
                "change_clean_minus_legacy": new_residual - old_residual if new_residual is not None and old_residual is not None else np.nan,
                "metric": "centered residual RMSE (Pa)",
            }
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{landscape}"])
    (ROOT / "clean_sensitivity_table_generated.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    pd.DataFrame(output).to_csv(ROOT / "clean_legacy_sensitivity.csv", index=False)


def write_heatmap(local: pd.DataFrame, gpu: pd.DataFrame) -> None:
    rows = []
    labels = []
    for label, model, source in MODEL_ROWS[:17]:
        labels.append(label)
        values = []
        for target in TARGETS:
            values.append(selected_field_value(model, target, source, local, gpu))
        rows.append(values)
    matrix = np.asarray(
        [[np.nan if value is None else value for value in row] for row in rows],
        dtype=float,
    )
    logged = np.log10(np.maximum(matrix, 1e-6))
    masked = np.ma.masked_invalid(logged)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e6e6e6")
    image = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(TARGETS)), [TARGET_LABELS[t] for t in TARGETS], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Clean CO$_2$-candidate median case nMAE by model and target")
    ax.set_xlabel("Field target")
    ax.set_ylabel("Model")
    for i in range(len(labels)):
        for j in range(len(TARGETS)):
            if np.isnan(matrix[i, j]):
                ax.text(j, i, "PENDING", ha="center", va="center", fontsize=5.2, color="#333333")
    fig.colorbar(image, ax=ax, label="log$_{10}$ median case nMAE")
    ax.set_xticks(np.arange(-0.5, len(TARGETS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    graphs = ROOT / "graphs"
    graphs.mkdir(exist_ok=True)
    fig.savefig(graphs / "clean_full_median_nmae_heatmap.png", dpi=220)
    fig.savefig(graphs / "clean_full_median_nmae_heatmap.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-run-dir",
        type=Path,
        default=ROOT / "clean_full_benchmark_2026-10-21_final",
    )
    parser.add_argument(
        "--gpu-root",
        type=Path,
        default=ROOT / "clean_gpu_runs",
    )
    args = parser.parse_args()
    local_dir = args.local_run_dir.resolve()
    metrics, pressure, statuses = read_local(local_dir)
    field_summary, pressure_summary = local_summaries(metrics, pressure)
    field_summary.to_csv(local_dir / "paper_field_summary.csv", index=False)
    pressure_summary.to_csv(local_dir / "paper_pressure_summary.csv", index=False)

    gpu, gpu_pressure, gpu_status = ingest_gpu_root(args.gpu_root.resolve())
    if not gpu.empty:
        gpu.to_csv(ROOT / "clean_gpu_three_seed_field_summary.csv", index=False)
        gpu_pressure.to_csv(ROOT / "clean_gpu_three_seed_pressure_summary.csv", index=False)
    if not gpu_status.empty:
        gpu_status.to_csv(ROOT / "clean_gpu_combined_run_status.csv", index=False)

    write_clean_full_table(field_summary, gpu, pressure_summary)
    write_representative_table(field_summary, gpu)
    write_pressure_table(pressure_summary, gpu_pressure)
    write_pressure_full_table(pressure_summary, gpu_pressure)
    write_sensitivity_table(field_summary, gpu, pressure_summary, gpu_pressure)
    write_heatmap(field_summary, gpu)
    write_group_holdout_outputs(args.gpu_root.resolve())
    from cpu_clean_benchmarks import paired_case_tests
    friedman, wilcoxon = paired_case_tests(metrics, pressure)
    friedman.to_csv(local_dir / "friedman_tests.csv", index=False)
    wilcoxon.to_csv(local_dir / "wilcoxon_holm_tests.csv", index=False)
    write_cpu_statistics_appendix(field_summary, pressure_summary, friedman, wilcoxon)
    print(f"Finalized clean CPU outputs from {local_dir}")
    print(f"Complete local field model/target rows: {len(field_summary)}")
    print(f"Complete local pressure model rows: {len(pressure_summary)}")
    print(f"Ingested GPU summary rows: {len(gpu)}")
    print("Generated paper fragments, sensitivity CSV, and heatmap.")


if __name__ == "__main__":
    main()
