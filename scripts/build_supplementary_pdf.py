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

    # Append Lattice floor (oracle) rows from raw_all if not already in field_3seed
    if "Lattice floor (oracle)" not in field_3seed["model"].unique():
        lf_sub = raw_all[
            (raw_all["protocol"] == "kfold")
            & (raw_all["model"] == "Lattice floor (oracle)")
            & (raw_all["target"] != "pressure_pa")
        ]
        lf_rows = []
        for (seed, target), grp in lf_sub.groupby(["seed", "target"], sort=True):
            rng = np.random.default_rng(2026)
            v = grp["mae_over_std"].to_numpy(float)
            boots = np.median(rng.choice(v, size=(400, len(v)), replace=True), axis=1)
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
            n = int(grp["n_nodes"].sum())
            sum_y, sum_y2, sse = float(grp["sum_y"].sum()), float(grp["sum_y2"].sum()), float(grp["sum_sq_err"].sum())
            pooled_r2 = 1.0 - sse / max(sum_y2 - sum_y**2 / n, 1e-24)
            lf_rows.append(
                {
                    "model": "Lattice floor (oracle)",
                    "target": target,
                    "n_cases": len(grp),
                    "case_macro_mae": float(grp["mae"].mean()),
                    "case_macro_mae_ci_lo": float(grp["mae"].mean()),
                    "case_macro_mae_ci_hi": float(grp["mae"].mean()),
                    "case_macro_rmse": float(grp["rmse"].mean()),
                    "median_nmae": float(np.median(v)),
                    "q25_nmae": float(np.percentile(v, 25)),
                    "q75_nmae": float(np.percentile(v, 75)),
                    "median_nmae_ci_lo": float(ci_lo),
                    "median_nmae_ci_hi": float(ci_hi),
                    "median_case_r2": float(grp["r2"].median()),
                    "q25_case_r2": float(np.percentile(grp["r2"], 25)),
                    "q75_case_r2": float(np.percentile(grp["r2"], 75)),
                    "pooled_r2_gauge": float(pooled_r2),
                    "pooled_r2_case_centered": np.nan,
                    "seed": seed,
                }
            )
        field_3seed = pd.concat([field_3seed, pd.DataFrame(lf_rows)], ignore_index=True)
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
            proto = (
                "Within-group aligned"
                if m_name in ALIGNED_MODELS
                else ("Voxel round-trip ref." if m_name == "Lattice floor (oracle)" else "Case-disjoint 5-fold")
            )
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
            f"\\caption{{Table {s_id}: Complete three-seed (2026, 2027, 2028) out-of-fold benchmark metrics for {label} across all 42 primary $\\mathrm{{sCO}}_2$ cases (19 competing models plus \\texttt{{Lattice floor (oracle)}}). "
            "Entries report three-seed mean $\\pm$ SD of 42-case median nMAE, three-seed mean interquartile range (IQR), three-seed mean of per-seed 95\\% case-bootstrap interval endpoints ($B=400$ case resamples per seed), 42-case arithmetic-mean Macro MAE, 42-case arithmetic-mean Macro RMSE, and node-pooled $R^2$.}\\\\\n"
            "\\toprule Model & Protocol & Median nMAE ($\\pm$SD) & nMAE IQR & Seed-mean 95\\% Boot. & Macro MAE & Macro RMSE & Pooled $R^2$\\\\\n"
            "\\midrule\\endfirsthead\n"
            "\\toprule Model & Protocol & Median nMAE ($\\pm$SD) & nMAE IQR & Seed-mean 95\\% Boot. & Macro MAE & Macro RMSE & Pooled $R^2$\\\\\n"
            "\\midrule\\endhead\n"
            + "\n".join(rows)
            + "\n\\bottomrule\\end{longtable}\\end{landscape}\n"
        )
    return blocks


def build_table_s8_pressure(raw_all: pd.DataFrame) -> str:
    p_kfold = raw_all[
        (raw_all["protocol"] == "kfold")
        & (raw_all["target"] == "pressure_pa")
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
                "mean_res_rmse_pa": grp["pressure_residual_rmse"].mean(),
                "rms_res_rmse_pa": np.sqrt(np.square(grp["pressure_residual_rmse"]).mean()),
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
            mean_res_rmse_pa=("mean_res_rmse_pa", "mean"),
            rms_res_rmse_pa=("rms_res_rmse_pa", "mean"),
            res_nrmse=("res_nrmse", "mean"),
            pooled_r2=("pooled_r2", "mean"),
        )
        .sort_values("macro_mae_kpa")
    )
    rows = []
    for _, r in agg_p.iterrows():
        m_name = str(r["model"])
        proto = (
            "Aligned"
            if m_name in ALIGNED_MODELS
            else ("Voxel ref." if m_name == "Lattice floor (oracle)" else "5-fold")
        )
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
                    f"{fmt(r['mean_res_rmse_pa'])}",
                    f"{fmt(r['rms_res_rmse_pa'])}",
                    f"{fmt(r['res_nrmse'])}",
                    f"{fmt(r['pooled_r2'])}",
                ]
            )
            + r" \\"
        )
    return (
        "\\begin{landscape}\\scriptsize\\setlength{\\tabcolsep}{3.2pt}\n"
        "\\begin{longtable}{p{3.4cm}p{1.3cm}rrrrrrrrr}\n"
        "\\caption{Table S8: Complete three-seed gauge-pressure ($p_{\\mathrm{gauge}}$) decomposition metrics across all 42 cases (19 competing models plus \\texttt{Lattice floor (oracle)}). "
        "Columns report 42-case arithmetic-mean Macro MAE and Macro RMSE (kPa), 42-case arithmetic-mean Level MAE and 42-case root-mean-square Level RMSE (kPa), 42-case arithmetic-mean Residual MAE (Pa), 42-case arithmetic-mean Residual RMSE (Pa), 42-case root-mean-square Residual RMSE (Pa), 42-case median Residual nRMSE (1 kPa floor), and node-pooled $R^2$. Per case $c$, $\\mathrm{RMSE}_c^2 = e_{\\mathrm{level},c}^2 + \\mathrm{RMSE}_{\\mathrm{res},c}^2$ holds identically under \\texttt{ddof=0}.}\\\\\n"
        "\\toprule Model & Scope & Macro MAE & Macro RMSE & Level MAE & Level RMS & Res.\\ MAE & Mean Res. & RMS Res. & Med.\\ Res. & Pooled\\\\\n"
        " & & (kPa) & (kPa) & (kPa) & (kPa) & (Pa) & RMSE (Pa) & RMSE (Pa) & nRMSE & $R^2$\\\\\n"
        "\\midrule\\endfirsthead\n"
        "\\toprule Model & Scope & Macro MAE & Macro RMSE & Level MAE & Level RMS & Res.\\ MAE & Mean Res. & RMS Res. & Med.\\ Res. & Pooled\\\\\n"
        " & & (kPa) & (kPa) & (kPa) & (kPa) & (Pa) & RMSE (Pa) & RMSE (Pa) & nRMSE & $R^2$\\\\\n"
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
    all_targets = [
        (t, f"{lbl.split('(')[0].strip()} (case nMAE)", sid) for t, lbl, sid in TARGET_ORDER
    ] + [("pressure_pa", "Gauge pressure ($p_{\\mathrm{gauge}}$, case MAE in kPa)", "S8")]
    film_name = "Domain-informed FiLM-MLP"
    for t_name, label, _ in all_targets:
        mcol = "mae" if t_name == "pressure_pa" else "mae_over_std"
        scale = 1e3 if t_name == "pressure_pa" else 1.0
        piv = (case_avg[case_avg["target"] == t_name].pivot(index="case_id", columns="model", values=mcol) / scale).dropna(axis=1)
        ranks = piv.rank(axis=1, method="average", ascending=True).mean(axis=0).sort_values()
        stat_f, p_f = stats.friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
        p_f_str = "$< 0.001$" if p_f < 0.001 else fmt(p_f)

        film_vec = piv[film_name]
        pvals = {}
        deltas = {}
        for col in piv.columns:
            if col == film_name:
                continue
            _, p_val = stats.wilcoxon(film_vec, piv[col], alternative="two-sided")
            pvals[col] = float(p_val)
            rng = np.random.default_rng(2026)
            d = (piv[col] - film_vec).to_numpy()
            med = float(np.median(d))
            boots = np.median(rng.choice(d, size=(2000, len(d)), replace=True), axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            deltas[col] = f"${med:+.3f}$ $[{lo:+.3f}, {hi:+.3f}]$"
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
                delta_str, raw_str, holm_str = "-- (FiLM ref.)", "-- (FiLM ref.)", "-- (FiLM ref.)"
            else:
                delta_str = deltas[m_name]
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
                        delta_str,
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
        "\\begin{longtable}{p{4.0cm}p{3.4cm}rcrrrr}\n"
        "\\caption{Table S10: Exploratory three-seed nonparametric repeated-measures Friedman ranks, paired case-level median differences $\\Delta = \\mathrm{Median}_c(e_{\\mathrm{comp},c}-e_{\\mathrm{FiLM},c})$ with 95\\% case-bootstrap confidence intervals ($B=2000$ resamples of the 42 paired case differences), and paired two-sided Wilcoxon signed-rank tests across the $N=42$ held-out cases for all 19 evaluated surrogate models. The tested case-level error $e_c$ is $\\mathrm{nMAE}_c$ on the seven spatially varying fields and case-level MAE (in kPa) on $p_{\\mathrm{gauge}}$. Holm step-down multiplicity adjustment ($18$ comparisons against \\texttt{Domain-informed FiLM-MLP}) is applied separately within each of the eight target fields.}\\\\\n"
        "\\toprule Target field (tested quantity) & Model & Mean rank & Median $\\Delta$ vs.\\ FiLM [95\\% CI] & Friedman $\\chi^2_F$ & Friedman $p$ & Wilcoxon raw $p$ & Holm $p$\\\\\n"
        "\\midrule\\endfirsthead\n"
        "\\toprule Target field (tested quantity) & Model & Mean rank & Median $\\Delta$ vs.\\ FiLM [95\\% CI] & Friedman $\\chi^2_F$ & Friedman $p$ & Wilcoxon raw $p$ & Holm $p$\\\\\n"
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
\makeatletter
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{#2}%
  \ifdim \wd\@tempboxa >\hsize
    #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}
\def\LT@makecaption#1#2#3{%
  \LT@mcol\LT@cols c{\hbox to\z@{\hss\parbox[t]\LTcapwidth{%
    \sbox\@tempboxa{#3}%
    \ifdim\wd\@tempboxa>\hsize
      #3%
    \else
      \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
    \fi
    \endgraf\vskip\baselineskip}%
  \hss}}}
\makeatother
\setlength{\emergencystretch}{3em}
\sloppy
\graphicspath{{../results/graphs/}{../figures/}{./}}
\begin{document}
\section*{Supplementary Material: Case-Disjoint Benchmark of Surrogate Models for Flow and Thermal Fields in Supercritical CO$_2$ Natural-Circulation Loops}
This supplementary document provides:
\begin{itemize}
\item \textbf{Section S1 (Tables S0a--S0b):} Detailed loop geometry, pipe/wall/jacket radial dimensions, cross-sectional plane sampling counts (Table S0a), plane-weighted mean $\mathrm{CO}_2$ density summaries, and CoolProp $(\partial p/\partial T)_\rho$ values across operating subgroups (Table S0b).
\item \textbf{Figure S1:} Spatial field reconstructions and absolute error maps for the median (rank 21/42) and worst (rank 42/42) held-out cases by out-of-fold case-nMAE rank for density ($\rho$, \texttt{DeepONet}), temperature ($T$, \texttt{PINN continuity}), and vertical velocity ($v$, \texttt{Domain-informed FiLM-MLP}).
\item \textbf{Tables S1--S7:} Complete 20-row three-seed (\texttt{2026}, \texttt{2027}, \texttt{2028}) case-disjoint and within-group mesh-aligned field metrics across all 19 competing models plus the \texttt{Lattice floor (oracle)} voxel round-trip reconstruction reference for density ($\rho$), temperature ($T$), turbulence eddy dissipation ($\varepsilon$), turbulence kinetic energy ($k$), and velocity components ($u, v, w$).
\item \textbf{Table S8:} Complete 20-row three-seed gauge-pressure ($p_{\mathrm{gauge}}$) decomposition into case-level mean pressure error and centered spatial residual error (reporting both 42-case arithmetic-mean and 42-case root-mean-square residual RMSE).
\item \textbf{Table S9:} Complete three-seed leave-one-group-out shift metrics across the four structural/regime groups (\texttt{HHHC/small}, \texttt{HHVC/small}, \texttt{VHVC/small}, and \texttt{VHVC/large}; $16\text{ pooled models} \times 8\text{ fields} = 128$ model--field pairs per group).
\item \textbf{Table S10:} Exploratory nonparametric Friedman repeated-measures ranks, paired case-level median differences $\Delta$ with 95\% bootstrap confidence intervals, and within-field Holm-adjusted Wilcoxon signed-rank tests across the 42 cases.
\end{itemize}

\subsection*{S1. Detailed Conjugate Loop Geometry, Plane Sampling, and Thermodynamic Derivative Summary}
\begin{itemize}
\item \textbf{Small-scale loops (\texttt{HHHC}, 20 cases; \texttt{HHVC}, 5 cases; \texttt{VHVC}, 4 cases):} Centerline rectangular loop width $L_x = 0.8936\text{ m}$ ($x_{\mathrm{cl}} \in [0, 0.8936]\text{ m}$) and height $L_y = 1.0000\text{ m}$ ($y_{\mathrm{cl}} \in [0, 1.0000]\text{ m}$). The primary $\mathrm{sCO}_2$ fluid flows inside a circular pipe of inner diameter $D_i = 15.0\text{ mm}$ ($R_i = 7.50\text{ mm}$, $z \in [-7.50, +7.50]\text{ mm}$) surrounded by a $2.83\text{ mm}$ solid stainless-steel wall ($7.50 \le r \le 10.33\text{ mm}$, $\rho = 7969.4\text{ kg m}^{-3}$) and a $0.660\text{ m}$ concentric cooling-water jacket annulus ($10.33 \le r \le 12.96\text{ mm}$, outer jacket wall $12.96 \le r \le 15.65\text{ mm}$). In \texttt{HHHC}, the heater is on the bottom leg ($y_{\mathrm{cl}}=0$) and the cooler jacket spans $x \in [0.1168, 0.7768]\text{ m}$ on the top leg ($y_{\mathrm{cl}}=1.000\text{ m}$). In \texttt{HHVC} and small-loop \texttt{VHVC}, the cooler jacket is on the right vertical leg ($x_{\mathrm{cl}}=0.8936\text{ m}$), spanning $y \in [0.2232, 0.8832]\text{ m}$ and $y \in [0.2792, 0.9392]\text{ m}$, respectively, while the heater is on the bottom leg (\texttt{HHVC}) or lower left vertical leg (\texttt{VHVC}). Each small-loop case is exported on 97 cross-sectional planes (46 constant-$x$ and 51 constant-$y$ planes at $0.02\text{ m}$ nominal spacing), yielding $89,647\text{--}97,264$ primary $\mathrm{sCO}_2$ nodes per case ($\rho \le 950\text{ kg m}^{-3}$) after removing solid wall rows and $7,236\text{--}7,430$ outer cooling-water jacket nodes ($\rho \in [971.4, 1000.7]\text{ kg m}^{-3}$) and deduplicating intersecting plane coordinates.
\item \textbf{Large-scale loop (\texttt{VHVC}, 13 cases):} Centerline width $L_x = 3.0100\text{ m}$ ($x_{\mathrm{cl}} \in [0, 3.0100]\text{ m}$) and height $L_y = 4.0800\text{ m}$ ($y_{\mathrm{cl}} \in [0, 4.0800]\text{ m}$), with inner $\mathrm{sCO}_2$ pipe diameter $D_i = 13.84\text{ mm}$ ($R_i = 6.92\text{ mm}$, $z \in [-6.92, +6.92]\text{ mm}$), $3.58\text{ mm}$ solid stainless-steel wall ($6.92 \le r \le 10.50\text{ mm}$, $\rho = 7971.8\text{ kg m}^{-3}$), and a $1.130\text{ m}$ vertical cooler jacket on the right leg ($x_{\mathrm{cl}}=3.0100\text{ m}$, $y \in [2.220, 3.350]\text{ m}$, cooling-water annulus $10.50 \le r \le 38.93\text{ mm}$, outer jacket wall $38.93 \le r \le 45.80\text{ mm}$). Each large-loop case is sampled on 142 unique cross-sectional planes (61 constant-$x$ and 81 constant-$y$ planes at $0.05\text{ m}$ nominal spacing; raw exports contain 284 plane blocks due to a duplicated plane pass removed during coordinate deduplication), yielding $207,586$ primary $\mathrm{sCO}_2$ nodes per case (and $82,185$ outer cooling-water jacket nodes).
\end{itemize}

\begin{table}[htbp]
\centering
\scriptsize
\caption{Table S0a: Structural loop dimensions, radial conjugate layers, and cross-sectional sampling plane counts across the four configuration/scale groups.}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrr}
\toprule
Group & Cases & $L_x \times L_y$ (m) & $D_i$ (mm) & Wall $r$ (mm) & Jacket $r$ (mm) & Unique planes & Primary $\mathrm{sCO}_2$ nodes/case & Water jacket nodes/case \\
\midrule
\texttt{HHHC/small} & 20 & $0.8936 \times 1.0000$ & 15.00 & 7.50--10.33 & 10.33--12.96 & 97 (46$x$+51$y$) & 97,264 & 7,430 \\
\texttt{HHVC/small} & 5  & $0.8936 \times 1.0000$ & 15.00 & 7.50--10.33 & 10.33--12.96 & 97 (46$x$+51$y$) & 89,771 & 7,236 \\
\texttt{VHVC/small} & 4  & $0.8936 \times 1.0000$ & 15.00 & 7.50--10.33 & 10.33--12.96 & 97 (46$x$+51$y$) & 89,647 & 7,236 \\
\texttt{VHVC/large} & 13 & $3.0100 \times 4.0800$ & 13.84 & 6.92--10.50 & 10.50--38.93 & 142 (61$x$+81$y$) & 207,586 & 82,185 \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[htbp]
\centering
\scriptsize
\caption{Table S0b: Summary of plane-weighted mean $\mathrm{CO}_2$ density $\bar{\rho}_{\mathrm{plane},c}$, loop-mean temperature $\bar{T}_c$, case-mean gauge pressure $\bar{p}_c$, and CoolProp $(\partial p/\partial T)_\rho$ evaluated at $(\bar{\rho}_{\mathrm{plane},c}, \bar{T}_c)$ across operating subgroups.}
\resizebox{\textwidth}{!}{%
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
\end{tabular}%
}
\end{table}

\newpage
\begin{figure}[htbp]
\centering
\includegraphics[width=0.92\textwidth,height=0.90\textheight,keepaspectratio]{heldout_cfd_prediction_error_maps_median_worst.pdf}
\caption{Figure S1: Out-of-fold spatial reconstructions and absolute error maps for the median (rank 21/42) and worst (rank 42/42) held-out cases selected by case-nMAE rank (seed 2026) for density $\rho$ (\texttt{DeepONet}), temperature $T$ (\texttt{PINN continuity}), and vertical velocity $v$ (\texttt{Domain-informed FiLM-MLP}).}
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
