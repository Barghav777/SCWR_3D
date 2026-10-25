from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
TARGET_NAMES = {
    "density_kg_m3": "Density",
    "pressure_pa": "Pressure",
    "temperature_c": "Temperature",
    "turbulence_eddy_dissipation_m2_s3": "TED",
    "turbulence_kinetic_energy_m2_s2": "TKE",
    "velocity_u_m_s": "Velocity $u$",
    "velocity_v_m_s": "Velocity $v$",
    "velocity_w_m_s": "Velocity $w$",
}


def number(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}g}"


def write_group_holdout_table() -> None:
    source = ROOT / "geometry_holdout_absolute_metrics.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    data = pd.read_csv(source)
    required = {
        "split", "target", "model", "n_cases", "case_macro_mae",
        "case_macro_rmse", "median_case_r2", "q25_case_r2", "q75_case_r2", "nMAE",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{source.name} is missing columns: {sorted(missing)}")
    data = data.sort_values(["split", "target", "model"])
    lines = [
        r"\subsection{Legacy configuration/geometry-group holdout results}",
        r"\label{app:legacy-group-holdout}",
        r"These legacy all-exported-node results cover the seven models present in the preserved group-holdout run. Values are included for all four observed groups and all eight targets; the table is a historical sensitivity reference, not clean-domain evidence.",
        r"\begin{landscape}",
        r"\begin{longtable}{@{}p{2.0cm}p{2.2cm}p{2.7cm}r r r r p{2.4cm}@{}}",
        r"\caption{Legacy group-holdout metrics (all exported nodes, including secondary-fluid nodes).}\\",
        r"\toprule Group & Model & Target & Cases & Case-macro MAE & Case-macro RMSE & nMAE & Median case $R^2$ [IQR]\\",
        r"\midrule\endfirsthead",
        r"\toprule Group & Model & Target & Cases & Case-macro MAE & Case-macro RMSE & nMAE & Median case $R^2$ [IQR]\\",
        r"\midrule\endhead",
    ]
    for row in data.itertuples(index=False):
        group = str(row.split).replace("leave_out:", "").replace("|", "/")
        target = TARGET_NAMES.get(str(row.target), str(row.target))
        r2 = (
            f"{number(row.median_case_r2)} "
            f"[{number(row.q25_case_r2)}, {number(row.q75_case_r2)}]"
        )
        lines.append(
            f"{group} & {row.model} & {target} & {int(row.n_cases)} & "
            f"{number(row.case_macro_mae)} & {number(row.case_macro_rmse)} & "
            f"{number(row.nMAE)} & {r2}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{landscape}", ""])
    (ROOT / "appendix_legacy_group_holdout.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_rank_test_tables() -> None:
    rank_path = ROOT / "friedman_mean_model_ranks.csv"
    wilcoxon_path = ROOT / "wilcoxon_holm_vs_lowest_mean_rank.csv"
    for path in (rank_path, wilcoxon_path):
        if not path.exists():
            raise FileNotFoundError(path)
    ranks = pd.read_csv(rank_path).sort_values(["target", "mean_rank"])
    pairs = pd.read_csv(wilcoxon_path).sort_values(
        ["target", "holm_adjusted_p", "comparison_model"]
    )
    lines = [
        r"\subsection{Legacy paired model-ranking tests}",
        r"These preserved rank tests use legacy all-exported-node results and are separate from the pending clean-data tests.",
        r"\begin{landscape}",
        r"\begin{longtable}{@{}p{2.7cm}p{3.0cm}r r r r@{}}",
        r"\caption{Legacy Friedman ranks by target.}\\",
        r"\toprule Target & Model & Cases & Mean rank & Friedman statistic & Friedman $p$\\",
        r"\midrule\endfirsthead",
        r"\toprule Target & Model & Cases & Mean rank & Friedman statistic & Friedman $p$\\",
        r"\midrule\endhead",
    ]
    for row in ranks.itertuples(index=False):
        lines.append(
            f"{TARGET_NAMES.get(str(row.target), str(row.target))} & {row.model} & "
            f"{int(row.n_cases)} & {number(row.mean_rank)} & "
            f"{number(row.friedman_chi2)} & {number(row.friedman_p)}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    lines.extend(
        [
            r"\begin{longtable}{@{}p{2.7cm}p{3.1cm}p{3.1cm}r r@{}}",
            r"\caption{Legacy paired Wilcoxon tests with Holm adjustment.}\\",
            r"\toprule Target & Reference model & Comparator & Raw $p$ & Holm-adjusted $p$\\",
            r"\midrule\endfirsthead",
            r"\toprule Target & Reference model & Comparator & Raw $p$ & Holm-adjusted $p$\\",
            r"\midrule\endhead",
        ]
    )
    for row in pairs.itertuples(index=False):
        lines.append(
            f"{TARGET_NAMES.get(str(row.target), str(row.target))} & "
            f"{row.reference_lowest_mean_rank} & {row.comparison_model} & "
            f"{number(row.wilcoxon_p)} & {number(row.holm_adjusted_p)}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{landscape}", ""])
    (ROOT / "appendix_legacy_rank_tests.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    write_group_holdout_table()
    write_rank_test_tables()
