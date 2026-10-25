# Case-Disjoint Full-Field Surrogate Benchmarking for 3D Supercritical CO₂ Natural-Circulation Loops

**Authors:** D. Barghav and Goutam Dutta  
**Affiliation:** Department of Mechanical Engineering, Indian Institute of Technology Jammu, India

This repository contains the preprocessing pipeline, thermodynamic and geometric verification scripts, CPU and GPU surrogate benchmark implementations, statistical testing scripts, derived metric tables, publication figures, manuscript source (`results/scwr_full_field_surrogate_paper.tex` and `.pdf`), and supplementary tables (`supplementary/supplementary_tables_document.pdf`, Tables S1–S11) for benchmarking full-field surrogates across 42 three-dimensional steady-state supercritical $\mathrm{CO}_2$ ($\mathrm{sCO}_2$) natural-circulation loop CFD simulations.

## Key Physical & Benchmark Highlights

1. **CFD Solver & Conjugate Domain:**
   - 42 mesh-independent steady-state 3D RANS conjugate heat-transfer simulations solved in ANSYS Fluent using the standard $k$–$\varepsilon$ turbulence model, gravitational acceleration $\mathbf{g} = (0, -9.81, 0)\text{ m s}^{-2}$, and NIST `REFPROP` real-gas thermophysical properties (Span–Wagner equation of state). Each steady-state simulation requires ~1 day of wall-clock time (4–5 days for transient runs).
   - **Small-scale loops (`HHHC`, 20 cases; `HHVC`, 5 cases; `VHVC`, 4 cases):** Rectangular centerline $L_x \times L_y = 0.8936\text{ m} \times 1.0000\text{ m}$, inner $\mathrm{sCO}_2$ pipe diameter $D_i = 15.0\text{ mm}$ ($r \le 7.50\text{ mm}$), stainless-steel wall ($7.50 \le r \le 10.33\text{ mm}$), and concentric cooling-water jacket annulus ($10.33 \le r \le 12.96\text{ mm}$ along the $0.660\text{ m}$ cooler span), sampled on 97 cross-sectional planes.
   - **Large-scale loop (`VHVC`, 13 cases):** Rectangular centerline $L_x \times L_y = 3.0100\text{ m} \times 4.0800\text{ m}$, inner $\mathrm{sCO}_2$ pipe diameter $D_i = 13.84\text{ mm}$ ($r \le 6.92\text{ mm}$), stainless-steel wall ($6.92 \le r \le 10.50\text{ mm}$), and concentric cooling-water jacket annulus ($10.50 \le r \le 38.93\text{ mm}$ along the $1.130\text{ m}$ cooler span), sampled on 142 unique cross-sectional planes.
   - **Exact Primary $\mathrm{sCO}_2$ Isolation:** Filtering fluid nodes at $\rho \le 950\text{ kg m}^{-3}$ separates 100% of the inner $\mathrm{sCO}_2$ core nodes ($5,451,341$ nodes, $\rho \in [153.2, 892.5]\text{ kg m}^{-3}$, matching the Span–Wagner EOS to $0.0008\%$ median relative error) from 100% of the outer cooling-water jacket nodes ($1,279,595$ nodes, $\rho \in [971.4, 1000.7]\text{ kg m}^{-3}$) across a $2.83\text{--}3.58\text{ mm}$ solid stainless-steel pipe wall.
   - **Corrected Filename Parameter:** Case `HHHC_P2000_80_bar_0.05kgs_T_205` has a minimum fluid temperature of $32.72^\circ\mathrm{C} = 305.87\text{ K}$ and is set to its true coolant inlet temperature $T_{\mathrm{in}} = 305\text{ K}$ across all scripts, figures, and models.

2. **Node-Random Split Leakage vs. Case-Disjoint Validation:**
   - Across all six classical baselines (`Mean baseline`, `Ridge`, `Random Forest`, `Extra Trees`, `XGBoost`, `CatBoost`), 80/20 random node splitting understates five-fold case-disjoint error by a six-model median of **$7.09\times$ on density**, **$12.54\times$ on gauge pressure**, and **$6.42\times$ on temperature** in case-macro MAE ($3.43\times$ and $3.06\times$ in median case nMAE for density and temperature).

3. **Three-Seed (`2026`, `2027`, `2028`) Case-Disjoint Benchmark Summary:**

| Target (unit) | Best Pooled Model | Pooled Macro MAE | Pooled Macro RMSE | Pooled Median nMAE (±SD) | Best Within-Group Aligned Model |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Density $\rho$ ($\mathrm{kg\ m^{-3}}$) | `DeepONet` (`Concat-MLP` lowest MAE) | 35.1 (26.3) | 37.1 (29.1) | $0.837 \pm 0.160$ | `POD-RBF` ($0.966 \pm 0.281$) |
| Gauge pressure $p_{\mathrm{gauge}}$ ($\mathrm{kPa}$) | `Extra Trees` | 83.2 | 83.7 | n/a ($2.91$ res. nRMSE) | `Nearest case` ($0.0122$ res. nRMSE) |
| Temperature $T$ ($^\circ\mathrm{C}$) | `PINN continuity` | 5.48 | 7.32 | $1.350 \pm 0.044$ | `POD-RBF` ($0.974 \pm 0.041$) |
| TED $\varepsilon$ ($\mathrm{m^2\ s^{-3}}$) | `Domain-informed FiLM-MLP` | 0.0978 | 0.315 | $0.192 \pm 0.020$ | `POD-RBF` ($0.177 \pm 0.018$) |
| TKE $k$ ($10^{-4}\ \mathrm{m^2\ s^{-2}}$) | `Domain-informed FiLM-MLP` | 0.947 | 1.57 | $0.242 \pm 0.051$ | `POD-RBF` ($0.237 \pm 0.015$) |
| Velocity $u$ ($\mathrm{m\ s^{-1}}$) | `Domain-informed FiLM-MLP` | 0.0381 | 0.0569 | $0.162 \pm 0.015$ | `Nearest case` ($0.167 \pm 0.033$) |
| Velocity $v$ ($\mathrm{m\ s^{-1}}$) | `Domain-informed FiLM-MLP` | 0.0383 | 0.0591 | $0.175 \pm 0.004$ | `POD-RBF` ($0.186 \pm 0.052$) |
| Velocity $w$ ($10^{-3}\ \mathrm{m\ s^{-1}}$) | `Domain-informed FiLM-MLP` | 1.43 | 3.46 | $0.272 \pm 0.007$ | `Nearest case` ($0.185 \pm 0.009$) |

## Repository Structure

- `preprocess_cases.py`: Raw CFD-Post CSV parser, solid-wall filter ($\rho \le 2000\text{ kg m}^{-3}$), coordinate deduplicator, and Parquet builder.
- `scripts/`:
  - `local_revision_audit.py`: Thermodynamic equation-of-state (`CoolProp`) verification, temperature label audit, plane-duplication audit, and primary $\mathrm{sCO}_2$ dataset builder (`mega_dataset_co2_candidate.parquet`).
  - `cpu_clean_benchmarks.py`: Classical baseline (`Mean baseline`, `Ridge`, `Random Forest`, `Extra Trees`, `XGBoost`, `CatBoost`) and mesh-aligned (`POD-RBF`, `POD-GPR`, `Nearest case`) benchmark runner across `case_disjoint`, `random_node`, and within-group protocols.
  - `build_kaggle_clean_case_cv_notebook.py`: Generator for the self-contained multi-seed GPU benchmark notebook (`notebooks/kaggle_clean_case_cv_predictions.ipynb`).
  - `ingest_kaggle_outputs.py`: Seed output ingestion, bootstrap confidence interval calculation, and Friedman / Wilcoxon–Holm statistical test runner.
  - `finalize_clean_benchmark_outputs.py`: Three-seed summary aggregator and LaTeX fragment generator.
  - `build_supplementary_pdf.py`: Builder for `supplementary/supplementary_tables.tex` and `supplementary/supplementary_tables_document.pdf` (Tables S1–S11).
- `notebooks/`: Self-contained GPU and CPU benchmark notebooks/scripts.
- `figures/`: Publication-ready figures (`.pdf` and `.png`).
- `results/`: Manuscript LaTeX source (`scwr_full_field_surrogate_paper.tex` and `.pdf`) and all derived CSV benchmark tables.
- `supplementary/`: Complete supplementary tables (Tables S1–S11) in LaTeX and compiled PDF formats.

## Reproducing the Workflow

```bash
pip install -r requirements.txt
python preprocess_cases.py
python scripts/local_revision_audit.py
python scripts/cpu_clean_benchmarks.py --output-dir results/clean_full_benchmark_2026-10-21_final
python scripts/ingest_kaggle_outputs.py --input-dir results/clean_gpu_runs/seed_2026 --seed 2026
python scripts/ingest_kaggle_outputs.py --input-dir results/clean_gpu_runs/seed_2027 --seed 2027
python scripts/ingest_kaggle_outputs.py --input-dir results/clean_gpu_runs/seed_2028 --seed 2028
python scripts/finalize_clean_benchmark_outputs.py
python scripts/build_supplementary_pdf.py
```
