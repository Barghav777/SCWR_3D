"""Build the complete three-seed supplementary benchmark tables (S1-S11) and PDF."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "supplementary"

TARGET_ORDER = [
    ("density_kg_m3", "Density ($\\rho$, kg m$^{-3}$)", "S1"),
    ("temperature_c", "Temperature ($T$, $^\\circ$C)", "S2"),
    ("turbulence_eddy_dissipation_m2_s3", "Turbulence eddy dissipation ($\\varepsilon$, m$^2$ s$^{-3}$)", "S3"),
    ("turbulence_kinetic_energy_m2_s2", "Turbulence kinetic energy ($k$, m$^2$ s$^{-2}$)", "S4"),
    ("velocity_u_m_s", "Horizontal velocity ($u$, m s$^{-1}$)", "S5"),
    ("velocity_v_m_s", "Vertical velocity ($v$, m s$^{-1}$)", "S6"),
    ("velocity_w_m_s", "Transverse velocity ($w$, m s$^{-1}$)", "S7"),
]

ALIGNED_MODELS = {"POD-RBF", "POD-GPR", "Nearest case"}


def fmt(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "--"
    val = float(value)
    if abs(val) > 0 and (abs(val) < 1e-3 or abs(val) >= 1e5):
        return f"{val:.{digits}g}"
    return f"{val:.{digits}g}"


def tex_escape(text: str) -> str:
    return str(text).replace("_", r"\_")


def load_three_seed_summaries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    field_3seed = pd.read_csv(RESULTS / "clean_gpu_three_seed_field_summary.csv")
    press_3seed = pd.read_csv(RESULTS / "clean_gpu_three_seed_pressure_summary.csv")
    raw_frames = []
    for seed in (2026, 2027, 2028):
        df = pd.read_csv(RESULTS / "clean_gpu_runs" / f"seed_{seed}" / "all_case_metrics.csv")
        df["seed"] = seed
        raw_frames.append(df)
    raw_all = pd.concat(raw_frames, ignore_index=True)
    return field_3seed, press_3seed, raw_all


def build_tables_s1_to_s7(field_3seed: pd.DataFrame) -> list[str]:
    blocks = []
    agg = (
        field_3seed.groupby(["model", "target"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            median_nmae_mean=("median_nmae", "mean"),
            median_nmae_sd=("median_nmae", lambda s: s.std(ddof=1)),
            q25_nmae_mean=("q25_nmae", "mean"),
            q75_nmae_mean=("q75_nmae", "mean"),
            ci_lo_mean=("median_nmae_ci_lo", "mean"),
            ci_hi_mean=("median_nmae_ci_hi", "mean"),
            median_r2_mean=("median_case_r2", "mean"),
            q25_r2_mean=("q25_case_r2", "mean"),
            q75_r2_mean=("q75_case_r2", "mean"),
            mae_mean=("case_macro_mae", "mean"),
            rmse_mean=("case_macro_rmse", "mean"),
            pooled_r2_mean=("pooled_r2_gauge", "mean"),
        )
    )
    for target, label, s_id in TARGET_ORDER:
        sub = agg[agg["target"] == target].sort_values("median_nmae_mean")
        rows = []
        for _, r in sub.iterrows():
            m_name = str(r["model"])
            proto = "Within-group aligned" if m_name in ALIGNED_MODELS else "Case-disjoint 5-fold"
            rows.append(
                " & ".join(
                    [
                        tex_escape(m_name),
                        proto,
                        f"{fmt(r['median_nmae_mean'])} $\\pm$ {fmt(r['median_nmae_sd'], 2)}",
                        f"[{fmt(r['q25_nmae_mean'])}, {fmt(r['q75_nmae_mean'])}]",
                        f"[{fmt(r['ci_lo_mean'])}, {fmt(r['ci_hi_mean'])}]",
                        f"{fmt(r['mae_mean'])}",
                        f"{fmt(r['rmse_mean'])}",
                        f"{fmt(r['pooled_r2_mean'])}",
                    ]
                )
                + r" \\"
            )
        blocks.append(
            "\\begin{landscape}\\scriptsize\n"
            "\\begin{longtable}{p{3.8cm}p{2.9cm}rrrrrr}\n"
            f"\\caption{{Table {s_id}: Complete three-seed (2026, 2027, 2028) out-of-fold benchmark metrics for {label} across all 42 primary $\\mathrm{{sCO}}_2$ cases. "
            "Entries report three-seed mean $\\pm$ SD of median case nMAE, mean interquartile range (IQR), mean 95\\% case-bootstrap CI, case-macro MAE, case-macro RMSE, and pooled $R^2$.}\\\\\n"
            "\\toprule Model & Protocol & Median nMAE ($\\pm$SD) & nMAE IQR & 95\\% Bootstrap CI & Macro MAE & Macro RMSE & Pooled $R^2$\\\\\n"
            "\\midrule\\endfirsthead\n"
            "\\toprule Model & Protocol & Median nMAE ($\\pm$SD) & nMAE IQR & 95\\% Bootstrap CI & Macro MAE & Macro RMSE & Pooled $R^2$\\\\\n"
            "\\midrule\\endhead\n"
            + "\n".join(rows)
            + "\n\\bottomrule\\end{longtable}\\end{landscape}\n"
        )
    return blocks


def build_table_s8_pressure(raw_all: pd.DataFrame) -> str:
    p_kfold = raw_all[
        (raw_all["protocol"] == "kfold")
        & (raw_all["target"] == "pressure_pa")
        & (raw_all["model"] != "Lattice floor (oracle)")
    ]
    seed_rows = []
    for (seed, model), grp in p_kfold.groupby(["seed", "model"], sort=True):
        if grp["case_id"].nunique() != 42:
            continue
        n_nodes = grp["n_nodes"].to_numpy(float)
        n = int(n_nodes.sum())
        sum_y = float(grp["sum_y"].sum())
        sum_y2 = float(grp["sum_y2"].sum())
        sse = float(grp["sum_sq_err"].sum())
        pooled_r2 = 1.0 - sse / max(sum_y2 - sum_y**2 / n, 1e-24)
        seed_rows.append(
            {
                "seed": seed,
                "model": model,
                "macro_mae_kpa": grp["mae"].mean() / 1e3,
                "macro_rmse_kpa": grp["rmse"].mean() / 1e3,
                "level_mae_kpa": grp["pressure_level_abs_error"].mean() / 1e3,
                "level_rmse_kpa": np.sqrt(grp["pressure_level_sq_error"].mean()) / 1e3,
                "res_mae_pa": grp["pressure_residual_mae"].mean(),
                "res_rmse_pa": np.sqrt(np.square(grp["pressure_residual_rmse"]).mean()),
                "res_nrmse": grp["pressure_residual_nrmse"].median(),
                "pooled_r2": pooled_r2,
            }
        )
    df_p = pd.DataFrame(seed_rows)
    agg_p = (
        df_p.groupby("model", as_index=False)
        .agg(
            macro_mae_kpa=("macro_mae_kpa", "mean"),
            macro_mae_sd=("macro_mae_kpa", lambda s: s.std(ddof=1)),
            macro_rmse_kpa=("macro_rmse_kpa", "mean"),
            level_mae_kpa=("level_mae_kpa", "mean"),
            level_rmse_kpa=("level_rmse_kpa", "mean"),
            res_mae_pa=("res_mae_pa", "mean"),
            res_rmse_pa=("res_rmse_pa", "mean"),
            res_nrmse=("res_nrmse", "mean"),
            pooled_r2=("pooled_r2", "mean"),
        )
        .sort_values("macro_mae_kpa")
    )
    rows = []
    for _, r in agg_p.iterrows():
        m_name = str(r["model"])
        proto = "Within-group aligned" if m_name in ALIGNED_MODELS else "Case-disjoint 5-fold"
        rows.append(
            " & ".join(
                [
                    tex_escape(m_name),
                    proto,
                    f"{fmt(r['macro_mae_kpa'])} $\\pm$ {fmt(r['macro_mae_sd'], 2)}",
                    f"{fmt(r['macro_rmse_kpa'])}",
                    f"{fmt(r['level_mae_kpa'])}",
                    f"{fmt(r['level_rmse_kpa'])}",
                    f"{fmt(r['res_mae_pa'])}",
                    f"{fmt(r['res_rmse_pa'])}",
                    f"{fmt(r['res_nrmse'])}",
                    f"{fmt(r['pooled_r2'])}",
                ]
            )
            + r" \\"
        )
    return (
        "\\begin{landscape}\\scriptsize\n"
        "\\begin{longtable}{p{3.6cm}p{2.7cm}rrrrrrrr}\n"
        "\\caption{Table S8: Complete three-seed gauge-pressure ($p_{\\mathrm{gauge}}$) decomposition metrics across all 42 cases. "
        "Columns report total node-level case-macro MAE and RMSE (kPa), case-mean offset MAE and RMSE (kPa), centered spatial residual MAE and RMSE (Pa), median centered residual nRMSE (with a 1 kPa denominator floor), and pooled gauge-pressure $R^2$.}\\\\\n"
        "\\toprule Model & Protocol & Macro MAE (kPa) & Macro RMSE (kPa) & Level MAE (kPa) & Level RMSE (kPa) & Resid. MAE (Pa) & Resid. RMSE (Pa) & Resid. nRMSE & Pooled $R^2$\\\\\n"
        "\\midrule\\endfirsthead\n"
        "\\toprule Model & Protocol & Macro MAE (kPa) & Macro RMSE (kPa) & Level MAE (kPa) & Level RMSE (kPa) & Resid. MAE (Pa) & Resid. RMSE (Pa) & Resid. nRMSE & Pooled $R^2$\\\\\n"
        "\\midrule\\endhead\n"
        + "\n".join(rows)
        + "\n\\bottomrule\\end{longtable}\\end{landscape}\n"
    )


def build_table_s9_group_holdouts(raw_all: pd.DataFrame) -> str:
    gh = raw_all[raw_all["protocol"] == "geometry_holdout"].copy()
    groups = [
        ("leave_out:HHHC|small", "HHHC / small ($n=20$)"),
        ("leave_out:HHVC|small", "HHVC / small ($n=5$)"),
        ("leave_out:VHVC|small", "VHVC / small ($n=4$)"),
        ("leave_out:VHVC|large", "VHVC / large ($n=13$)"),
    ]
    targets_short = [
        ("density_kg_m3", "$\\rho$"),
        ("temperature_c", "$T$"),
        ("turbulence_eddy_dissipation_m2_s3", "$\\varepsilon$"),
        ("turbulence_kinetic_energy_m2_s2", "$k$"),
        ("velocity_u_m_s", "$u$"),
        ("velocity_v_m_s", "$v$"),
        ("velocity_w_m_s", "$w$"),
    ]
    rows = []
    for split_key, g_label in groups:
        sub_g = gh[gh["split"] == split_key]
        models_g = sorted(sub_g["model"].unique())
        for m_name in models_g:
            sub_m = sub_g[sub_g["model"] == m_name]
            cells = [g_label, tex_escape(m_name)]
            for t_name, _ in targets_short:
                sub_t = sub_m[sub_m["target"] == t_name]
                seed_meds = sub_t.groupby("seed")["mae_over_std"].median()
                cells.append(fmt(seed_meds.mean()))
            sub_p = sub_m[sub_m["target"] == "pressure_pa"]
            seed_p_mae = sub_p.groupby("seed")["pressure_level_abs_error"].mean() / 1e3
            seed_p_res = sub_p.groupby("seed")["pressure_residual_nrmse"].median()
            cells.append(fmt(seed_p_mae.mean()))
            cells.append(fmt(seed_p_res.mean()))
            rows.append(" & ".join(cells) + r" \\")
    return (
        "\\begin{landscape}\\scriptsize\n"
        "\\begin{longtable}{p{2.8cm}p{3.6cm}rrrrrrrrr}\n"
        "\\caption{Table S9: Complete three-seed leave-one-configuration/geometry-group-out extrapolation metrics across the four structural groups. "
        "Entries report three-seed mean of median case nMAE for the seven spatially varying fields, case-level gauge-pressure MAE (kPa), and centered pressure residual nRMSE (1 kPa floor). Mesh-aligned transfer methods (POD-RBF, POD-GPR, Nearest case) are not applicable across disjoint loop geometries.}\\\\\n"
        "\\toprule Held-out group & Model & $\\rho$ nMAE & $T$ nMAE & $\\varepsilon$ nMAE & $k$ nMAE & $u$ nMAE & $v$ nMAE & $w$ nMAE & $p$ Level MAE (kPa) & $p$ Resid. nRMSE\\\\\n"
        "\\midrule\\endfirsthead\n"
        "\\toprule Held-out group & Model & $\\rho$ nMAE & $T$ nMAE & $\\varepsilon$ nMAE & $k$ nMAE & $u$ nMAE & $v$ nMAE & $w$ nMAE & $p$ Level MAE (kPa) & $p$ Resid. nRMSE\\\\\n"
        "\\midrule\\endhead\n"
        + "\n".join(rows)
        + "\n\\bottomrule\\end{longtable}\\end{landscape}\n"
    )


def build_table_s10_statistics(raw_all: pd.DataFrame) -> str:
    from scipy import stats

    rows = []
    kfold = raw_all[(raw_all["protocol"] == "kfold") & (raw_all["model"] != "Lattice floor (oracle)")].copy()
    case_avg = kfold.groupby(["target", "model", "case_id"], as_index=False).agg(
        {"mae_over_std": "mean", "mae": "mean"}
    )
    all_targets = TARGET_ORDER + [("pressure_pa", "Gauge pressure ($p_{\\mathrm{gauge}}$, Pa)", "S8")]
    film_name = "Domain-informed FiLM-MLP"
    for t_name, label, _ in all_targets:
        mcol = "mae" if t_name == "pressure_pa" else "mae_over_std"
        piv = case_avg[case_avg["target"] == t_name].pivot(index="case_id", columns="model", values=mcol).dropna(axis=1)
        ranks = piv.rank(axis=1, method="average", ascending=True).mean(axis=0).sort_values()
        stat_f, p_f = stats.friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
        p_f_str = "$< 0.001$" if p_f < 0.001 else fmt(p_f)

        film_vec = piv[film_name]
        pvals = {}
        for col in piv.columns:
            if col == film_name:
                continue
            _, p_val = stats.wilcoxon(film_vec, piv[col], alternative="two-sided")
            pvals[col] = float(p_val)
        sorted_cols = sorted(pvals.keys(), key=lambda c: pvals[c])
        holm_p = {}
        running_max = 0.0
        m_cnt = len(sorted_cols)
        for idx, col in enumerate(sorted_cols):
            adj = min(1.0, pvals[col] * (m_cnt - idx))
            running_max = max(running_max, adj)
            holm_p[col] = running_max

        for m_name, mean_rk in ranks.items():
            if m_name == film_name:
                raw_str, holm_str = "-- (FiLM ref.)", "-- (FiLM ref.)"
            else:
                raw_v = pvals[m_name]
                holm_v = holm_p[m_name]
                raw_str = "$< 0.001$" if raw_v < 0.001 else fmt(raw_v)
                holm_str = "$< 0.001$" if holm_v < 0.001 else fmt(holm_v)
            rows.append(
                " & ".join(
                    [
                        label,
                        tex_escape(m_name),
                        fmt(mean_rk),
                        fmt(stat_f),
                        p_f_str,
                        raw_str,
                        holm_str,
                    ]
                )
                + r" \\"
            )
    return (
        "\\begin{landscape}\\scriptsize\n"
        "\\begin{longtable}{p{4.2cm}p{3.6cm}rrrrr}\n"
        "\\caption{Table S10: Exploratory three-seed nonparametric repeated-measures Friedman ranks and paired two-sided Wilcoxon signed-rank tests (with Holm step-down multiplicity adjustment against \\texttt{Domain-informed FiLM-MLP}) across the $N=42$ held-out cases for all 18 evaluated models.}\\\\\n"
        "\\toprule Target field & Model & Mean rank & Friedman $\\chi^2_F$ & Friedman $p$ & Wilcoxon raw $p$ vs.\\ FiLM & Holm $p$ vs.\\ FiLM\\\\\n"
        "\\midrule\\endfirsthead\n"
        "\\toprule Target field & Model & Mean rank & Friedman $\\chi^2_F$ & Friedman $p$ & Wilcoxon raw $p$ vs.\\ FiLM & Holm $p$ vs.\\ FiLM\\\\\n"
        "\\midrule\\endhead\n"
        + "\n".join(rows)
        + "\n\\bottomrule\\end{longtable}\\end{landscape}\n"
    )


def build_table_s11_sensitivity(field_3seed: pd.DataFrame, raw_all: pd.DataFrame) -> str:
    legacy = pd.read_csv(RESULTS / "all_models_all_targets_case_cv_metrics.csv")
    agg_f = field_3seed.groupby(["model", "target"], as_index=False)["median_nmae"].mean()
    p_kfold = raw_all[
        (raw_all["protocol"] == "kfold")
        & (raw_all["target"] == "pressure_pa")
        & (raw_all["model"] != "Lattice floor (oracle)")
    ]
    p_res = (
        p_kfold.groupby(["seed", "model"])["pressure_residual_rmse"]
        .apply(lambda s: float(np.sqrt(np.mean(np.square(s)))))
        .groupby("model")
        .mean()
    )
    model_map = [
        ("Mean baseline", "Mean baseline"),
        ("Ridge", "Ridge"),
        ("Random Forest", "Random Forest"),
        ("Extra Trees", "Extra Trees"),
        ("XGBoost", "XGBoost"),
        ("CatBoost", "CatBoost"),
        ("DeepONet", "DeepONet"),
        ("PINN data-only", "PINN data-only"),
        ("PINN continuity", "PINN continuity"),
        ("Domain-informed FiLM-MLP", "Domain-informed FiLM-MLP"),
        ("Latent Neural Operator", "Latent Neural Operator"),
        ("Transolver-style", "Transolver-style"),
        ("GAMS-GNN-style", "GAMS-GNN-style"),
        ("F-FNO", "F-FNO"),
        ("U-FNO3D", "U-FNO3D"),
        ("POD-RBF", "POD-RBF"),
        ("Nearest case", "Nearest case"),
    ]
    rows = []
    out_csv = []
    for label_m, leg_m in model_map:
        leg_sub = legacy[legacy["model"] == leg_m]
        for t_name, t_label, _ in TARGET_ORDER:
            old_val = float(leg_sub.loc[leg_sub["target"] == t_name, "median_case_nmae"].iloc[0])
            new_val = float(agg_f.loc[(agg_f["model"] == label_m) & (agg_f["target"] == t_name), "median_nmae"].iloc[0])
            diff = new_val - old_val
            rows.append(
                f"{t_label} & {tex_escape(label_m)} & Median case nMAE & {fmt(old_val)} & {fmt(new_val)} & {fmt(diff)} \\\\"
            )
            out_csv.append(
                {
                    "target": t_name,
                    "model": label_m,
                    "metric": "median_case_nmae",
                    "legacy_unfiltered_metric": old_val,
                    "clean_sco2_three_seed_metric": new_val,
                    "difference_clean_minus_legacy": diff,
                }
            )
        old_p = float(leg_sub.loc[leg_sub["target"] == "pressure_pa", "mean_case_centered_rmse"].iloc[0])
        new_p = float(p_res.loc[label_m])
        diff_p = new_p - old_p
        rows.append(
            f"Gauge pressure ($p_{{\\mathrm{{gauge}}}}$) & {tex_escape(label_m)} & Centered residual RMSE (Pa) & {fmt(old_p)} & {fmt(new_p)} & {fmt(diff_p)} \\\\"
        )
        out_csv.append(
            {
                "target": "pressure_pa",
                "model": label_m,
                "metric": "centered_residual_rmse_pa",
                "legacy_unfiltered_metric": old_p,
                "clean_sco2_three_seed_metric": new_p,
                "difference_clean_minus_legacy": diff_p,
            }
        )
    pd.DataFrame(out_csv).to_csv(RESULTS / "clean_legacy_sensitivity.csv", index=False)
    return (
        "\\begin{landscape}\\scriptsize\n"
        "\\begin{longtable}{p{4.0cm}p{3.4cm}p{3.4cm}rrr}\n"
        "\\caption{Table S11: Matched dataset-version sensitivity comparing the legacy unfiltered fluid dataset (6,730,936 nodes including the outer cooling-water annulus) against the clean primary $\\mathrm{sCO}_2$ pipe core dataset (5,451,341 nodes, $\\rho \\le 950$ kg m$^{-3}$) across all 17 benchmark models.}\\\\\n"
        "\\toprule Target field & Model & Metric & Legacy unfiltered & Clean $\\mathrm{sCO}_2$ (3-seed) & Clean $-$ Legacy\\\\\n"
        "\\midrule\\endfirsthead\n"
        "\\toprule Target field & Model & Metric & Legacy unfiltered & Clean $\\mathrm{sCO}_2$ (3-seed) & Clean $-$ Legacy\\\\\n"
        "\\midrule\\endhead\n"
        + "\n".join(rows)
        + "\n\\bottomrule\\end{longtable}\\end{landscape}\n"
    )


def write_supplementary_document(blocks: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "supplementary_tables.tex").write_text("\n".join(blocks), encoding="utf-8")
    doc_tex = r"""\documentclass[10pt]{article}
\usepackage[margin=0.6in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,longtable,pdflscape,array,graphicx}
\graphicspath{{../results/graphs/}{../figures/}{./}}
\begin{document}
\section*{Supplementary Material: Case-Disjoint Benchmark of Surrogate Models for Flow and Thermal Fields in Supercritical CO$_2$ Natural-Circulation Loops}
This supplementary document provides:
\begin{itemize}
\item \textbf{Section S1:} Detailed loop geometry, pipe/wall/jacket radial dimensions, cross-sectional plane sampling counts, plane-weighted mean $\mathrm{CO}_2$ density tests, and CoolProp $(\partial p/\partial T)_\rho$ values across operating groups.
\item \textbf{Figure S1:} Spatial field reconstructions and absolute error maps for the median (rank 21/42) and worst (rank 42/42) held-out cases by out-of-fold case-nMAE rank for density ($\rho$, \texttt{DeepONet}), temperature ($T$, \texttt{PINN continuity}), and vertical velocity ($v$, \texttt{Domain-informed FiLM-MLP}).
\item \textbf{Tables S1--S7:} Complete three-seed (\texttt{2026}, \texttt{2027}, \texttt{2028}) case-disjoint and within-group mesh-aligned field metrics across all evaluated models for density ($\rho$), temperature ($T$), turbulence eddy dissipation ($\varepsilon$), turbulence kinetic energy ($k$), and velocity components ($u, v, w$).
\item \textbf{Table S8:} Complete three-seed gauge-pressure ($p_{\mathrm{gauge}}$) decomposition into case-level mean pressure error and centered spatial residual error.
\item \textbf{Table S9:} Complete three-seed leave-one-configuration/geometry-group-out extrapolation metrics across the four structural groups (\texttt{HHHC/small}, \texttt{HHVC/small}, \texttt{VHVC/small}, and \texttt{VHVC/large}; $16\text{ models} \times 8\text{ fields} = 128$ model-field pairs per group).
\item \textbf{Table S10:} Exploratory nonparametric Friedman repeated-measures ranks and paired Wilcoxon--Holm tests across the 42 cases.
\end{itemize}

\subsection*{S1. Detailed Conjugate Loop Geometry, Plane Sampling, and Thermodynamic Derivative Summary}
\begin{itemize}
\item \textbf{Small-scale loops (\texttt{HHHC}, 20 cases; \texttt{HHVC}, 5 cases; \texttt{VHVC}, 4 cases):} Centerline rectangular loop width $L_x = 0.8936\text{ m}$ ($x_{\mathrm{cl}} \in [0, 0.8936]\text{ m}$) and height $L_y = 1.0000\text{ m}$ ($y_{\mathrm{cl}} \in [0, 1.0000]\text{ m}$). The primary $\mathrm{sCO}_2$ fluid flows inside a circular pipe of inner diameter $D_i = 15.0\text{ mm}$ ($R_i = 7.50\text{ mm}$, $z \in [-7.50, +7.50]\text{ mm}$) surrounded by a $2.83\text{ mm}$ solid stainless-steel wall ($7.50 \le r \le 10.33\text{ mm}$, $\rho = 7969.4\text{ kg m}^{-3}$) and a $0.660\text{ m}$ concentric cooling-water jacket annulus ($10.33 \le r \le 12.96\text{ mm}$, outer jacket wall $12.96 \le r \le 15.65\text{ mm}$). In \texttt{HHHC}, the heater is on the bottom leg ($y_{\mathrm{cl}}=0$) and the cooler jacket spans $x \in [0.1168, 0.7768]\text{ m}$ on the top leg ($y_{\mathrm{cl}}=1.000\text{ m}$). In \texttt{HHVC} and small-loop \texttt{VHVC}, the cooler jacket is on the right vertical leg ($x_{\mathrm{cl}}=0.8936\text{ m}$), spanning $y \in [0.2232, 0.8832]\text{ m}$ and $y \in [0.2792, 0.9392]\text{ m}$, respectively, while the heater is on the bottom leg (\texttt{HHVC}) or lower left vertical leg (\texttt{VHVC}). Each small-loop case is exported on 97 cross-sectional planes (46 constant-$x$ and 51 constant-$y$ planes at $0.02\text{ m}$ nominal spacing), yielding $89,647\text{--}97,264$ primary $\mathrm{sCO}_2$ nodes per case ($\rho \le 950\text{ kg m}^{-3}$) after removing solid wall rows and $7,236\text{--}7,430$ outer cooling-water jacket nodes ($\rho \in [971.4, 1000.7]\text{ kg m}^{-3}$) and deduplicating intersecting plane coordinates.
\item \textbf{Large-scale loop (\texttt{VHVC}, 13 cases):} Centerline width $L_x = 3.0100\text{ m}$ ($x_{\mathrm{cl}} \in [0, 3.0100]\text{ m}$) and height $L_y = 4.0800\text{ m}$ ($y_{\mathrm{cl}} \in [0, 4.0800]\text{ m}$), with inner $\mathrm{sCO}_2$ pipe diameter $D_i = 13.84\text{ mm}$ ($R_i = 6.92\text{ mm}$, $z \in [-6.92, +6.92]\text{ mm}$), $3.58\text{ mm}$ solid stainless-steel wall ($6.92 \le r \le 10.50\text{ mm}$, $\rho = 7971.8\text{ kg m}^{-3}$), and a $1.130\text{ m}$ vertical cooler jacket on the right leg ($x_{\mathrm{cl}}=3.0100\text{ m}$, $y \in [2.220, 3.350]\text{ m}$, cooling-water annulus $10.50 \le r \le 38.93\text{ mm}$, outer jacket wall $38.93 \le r \le 45.80\text{ mm}$). Each large-loop case is sampled on 142 unique cross-sectional planes (61 constant-$x$ and 81 constant-$y$ planes at $0.05\text{ m}$ nominal spacing; raw exports contain 284 plane blocks due to a duplicated plane pass removed during coordinate deduplication), yielding $207,586$ primary $\mathrm{sCO}_2$ nodes per case (and $82,185$ outer cooling-water jacket nodes).
\end{itemize}

\begin{table}[htbp]
\centering
\scriptsize
\caption*{Summary of plane-weighted mean $\mathrm{CO}_2$ density $\bar{\rho}_{\mathrm{plane},c}$, node-mean density $\bar{\rho}_c$, loop-mean temperature $\bar{T}_c$, case-mean gauge pressure $\bar{p}_c$, and CoolProp $(\partial p/\partial T)_\rho$ evaluated at $(\bar{\rho}_c, \bar{T}_c)$ across operating subgroups.}
\begin{tabular}{llrrrrrrr}
\toprule
Group & $(p_{\mathrm{op}}, T_{\mathrm{in}})$ & Cases & Planes & $\bar{\rho}_{\mathrm{plane},c}$ (kg m$^{-3}$) & $\mathrm{SD}(\bar{\rho}_{\mathrm{plane},c})$ & $\bar{T}_c$ ($^\circ$C) & $\bar{p}_c$ (MPa) & CoolProp $(\partial p/\partial T)_\rho$ (MPa K$^{-1}$) \\
\midrule
\texttt{HHHC/small} & 80 bar, 305 K & 3 & 97 & 582.0--612.5 & 16.06 & 34.43--35.85 & 0.360--0.464 & 0.235--0.255 \\
\texttt{HHHC/small} & 90 bar, 300 K & 3 & 97 & 737.7--756.1 & 9.23  & 30.89--33.24 & 0.281--0.553 & 0.393--0.417 \\
\texttt{HHHC/small} & 90 bar, 305 K & 8 & 97 & 671.6--704.0 & 9.79  & 33.42--36.14 & 0.223--0.504 & 0.315--0.349 \\
\texttt{HHHC/small} & 90 bar, 315 K & 3 & 97 & 376.2--389.4 & 6.62  & 45.28--47.52 & 0.365--0.520 & 0.133--0.139 \\
\texttt{HHHC/small} & 100 bar, 305 K & 3 & 97 & 709.1--723.2 & 7.09 & 35.12--37.38 & 0.319--0.631 & 0.362--0.377 \\
\texttt{HHVC/small} & 90 bar, 305 K & 5 & 97 & 681.2--717.8 & 16.66 & 31.92--39.36 & 0.006--1.763 & 0.332--0.365 \\
\texttt{VHVC/small} & 90 bar, 305 K & 4 & 97 & 696.8--712.1 & 6.62  & 32.63--34.59 & 0.131--0.513 & 0.342--0.358 \\
\texttt{VHVC/large} & 85 bar, 283.5 K & 13 & 142 & 161.5--860.8 & 270.83 & 16.48--92.22 & 0.001--0.009 & 0.039--0.596 \\
\bottomrule
\end{tabular}
\end{table}

\newpage
\begin{figure}[htbp]
\centering
\includegraphics[width=0.92\textwidth,height=0.90\textheight,keepaspectratio]{heldout_cfd_prediction_error_maps_median_worst.pdf}
\caption*{Figure S1: Out-of-fold spatial reconstructions and absolute error maps for the median (rank 21/42) and worst (rank 42/42) held-out cases selected by case-nMAE rank (seed 2026) for density $\rho$ (\texttt{DeepONet}), temperature $T$ (\texttt{PINN continuity}), and vertical velocity $v$ (\texttt{Domain-informed FiLM-MLP}).}
\end{figure}
\clearpage

\input{supplementary_tables.tex}
\end{document}
"""
    (OUT / "supplementary_tables_document.tex").write_text(doc_tex, encoding="utf-8")


def main() -> None:
    field_3seed, press_3seed, raw_all = load_three_seed_summaries()
    blocks = []
    blocks.extend(build_tables_s1_to_s7(field_3seed))
    blocks.append(build_table_s8_pressure(raw_all))
    blocks.append(build_table_s9_group_holdouts(raw_all))
    blocks.append(build_table_s10_statistics(raw_all))
    write_supplementary_document(blocks)
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "supplementary_tables_document.tex"],
            cwd=OUT,
            check=True,
        )
    shutil.copy2(OUT / "supplementary_tables_document.pdf", OUT / "supplementary_tables.pdf")
    print("Built supplementary/supplementary_tables_document.pdf and supplementary/supplementary_tables.pdf successfully.")


if __name__ == "__main__":
    main()
