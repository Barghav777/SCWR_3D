# Complete Reconciliation Report & Point-by-Point Resolution of External Reviews

## 1. Executive Summary: Why the External Revision Received a Second "Major Revision"
You shared three documents from your external review process:
1. **`scwr_review.md` (Initial Review)**: Identified substantive technical and presentation issues (Problems A–I) in the 14-page manuscript, including an apparent arithmetic mismatch between Figure 5 (relative MAE heatmap) and Table 4 (`Mean-base MAE` and `Reduction (%)`), an inconsistent model count (`17`, `18`, `19`), an imprecise sentence about constant hydrostatic subtraction, a depth mismatch (`max_depth=6` vs. `8`) for `XGBoost` in the node-vs-case split comparison, missing detail on train-only preprocessing and the continuity penalty, and the need to show a fixed model shortlist in leave-one-group-out extrapolation and median/worst spatial maps in the main text.
2. **`scwr_latex_project` (External Assistant's Revised Draft)**: Because the external assistant **did not have access to your codebase, Kaggle notebooks, or CSV files**, it could not verify or fix the underlying numbers. Instead, it inserted explicit *"provisional pending reconciliation"* disclaimers into the Table 4 caption, left embedded instructions to the authors inside the manuscript text (*"If the authors cannot confirm... these results must be recomputed..."*), and left placeholders (`\repourl`).
3. **`scwr_revised_paper_review` (Second Review of `scwr_latex_project`)**: Rightly flagged that a journal submission cannot contain unresolved numerical contradictions, *"provisional pending reconciliation"* notes, or unverified *"must be audited"* statements, organizing all remaining items into **Major Concerns M1–M8** and **Questions for the Authors Q1–Q6**.

Because we have the complete codebase ([`notebooks/kaggle_clean_case_cv_predictions.ipynb`](file:///c:/Users/bargh/Downloads/BTP_data/notebooks/kaggle_clean_case_cv_predictions.ipynb), [`scripts/cpu_clean_benchmarks.py`](file:///c:/Users/bargh/Downloads/BTP_data/scripts/cpu_clean_benchmarks.py)) and all raw and aggregated CSVs in [`results/`](file:///c:/Users/bargh/Downloads/BTP_data/results), we have **directly resolved every single concern (M1–M8 and Q1–Q6)** using verified data and updated both the main manuscript ([`scwr_full_field_surrogate_paper.pdf`](file:///c:/Users/bargh/Downloads/BTP_data/results/scwr_full_field_surrogate_paper.pdf)) and supplementary material ([`supplementary_tables_document.pdf`](file:///c:/Users/bargh/Downloads/BTP_data/supplementary/supplementary_tables_document.pdf)).

---

## 2. Point-by-Point Resolution of Major Concerns (M1–M8)

### M1 & Q1: Numerical Reconciliation Between Figure 5 and Table 4, and Model-Count Taxonomy
* **Root Cause Identified**:
  In [`results/clean_gpu_three_seed_field_summary.csv`](file:///c:/Users/bargh/Downloads/BTP_data/results/clean_gpu_three_seed_field_summary.csv) and [`results/clean_gpu_three_seed_pressure_summary.csv`](file:///c:/Users/bargh/Downloads/BTP_data/results/clean_gpu_three_seed_pressure_summary.csv), the true 3-seed five-fold (`protocol == "kfold"`) `Mean baseline` case-macro MAEs are:
  $$\rho:\; 129.3\text{ kg m}^{-3},\quad p_{\mathrm{gauge}}:\; 235.9\text{ kPa},\quad T:\; 14.90^\circ\text{C},\quad \varepsilon:\; 0.3002\text{ m}^2\text{ s}^{-3},\quad k:\; 5.547\times 10^{-4}\text{ m}^2\text{ s}^{-2},\quad u:\; 0.1051\text{ m s}^{-1},\quad v:\; 0.1175\text{ m s}^{-1},\quad w:\; 1.577\times 10^{-3}\text{ m s}^{-1}.$$
  An earlier helper script had computed the `Mean-base MAE` column of Table 4 from [`results/all_case_metrics.csv`](file:///c:/Users/bargh/Downloads/BTP_data/results/all_case_metrics.csv) **without filtering `protocol == "kfold"`**, inadvertently averaging `kfold` and `geometry_holdout` rows into the `Mean-base MAE` column (`131.6, 254.0, 13.71, 0.1945, 3.635, 0.1763, 0.1972, 2.660`), while Figure 5 ([`final_three_seed_error_ratio_heatmap.pdf`](file:///c:/Users/bargh/Downloads/BTP_data/results/graphs/final_three_seed_error_ratio_heatmap.pdf)) and Supplementary Tables S1–S8 properly used `protocol == "kfold"`.
* **Exact Reconciliation (100% Match Between Table 4 and Figure 5)**:

| Target Field | Model | Scope | 3-Seed `kfold` Macro MAE | True 3-Seed `kfold` `Mean-base MAE` | Ratio ($\text{MAE}/\text{Mean-base}$) | Figure 5 Cell | Reduction (%) in Table 4 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| $\rho\text{ (kg m}^{-3}\text{)}$ | `DeepONet` | Pooled | $35.1$ | $129.3$ | $0.272$ | `0.27` | **$72.8\%$** |
| $\rho\text{ (kg m}^{-3}\text{)}$ | `Concat-MLP` | Pooled | $26.3$ | $129.3$ | $0.203$ | `0.20` | **$79.7\%$** |
| $p_{\mathrm{gauge}}\text{ (kPa)}$ | `Extra Trees` | Pooled | $83.2$ | $235.9$ | $0.353$ | `0.35` | **$64.7\%$** |
| $p_{\mathrm{gauge}}\text{ (kPa)}$ | `POD-RBF` | Aligned | $90.1$ | $235.9$ | $0.382$ | `0.38` | **$61.8\%$** |
| $p_{\mathrm{gauge}}\text{ (kPa)}$ | `Nearest case` | Aligned | $108.1$ | $235.9$ | $0.458$ | `0.46` | **$54.2\%$** |
| $T\text{ (}^\circ\text{C)}$ | `PINN continuity` | Pooled | $5.48$ | $14.90$ | $0.368$ | `0.37` | **$63.2\%$** |
| $T\text{ (}^\circ\text{C)}$ | `POD-RBF` | Aligned | $6.11$ | $14.90$ | $0.410$ | `0.41` | **$59.0\%$** |
| $\varepsilon\text{ (m}^2\text{ s}^{-3}\text{)}$ | `FiLM-MLP` | Pooled | $0.0978$ | $0.3002$ | $0.326$ | `0.33` | **$67.4\%$** |
| $\varepsilon\text{ (m}^2\text{ s}^{-3}\text{)}$ | `POD-RBF` | Aligned | $0.127$ | $0.3002$ | $0.422$ | `0.42` | **$57.8\%$** |
| $k\text{ (}10^{-4}\text{ m}^2\text{ s}^{-2}\text{)}$ | `FiLM-MLP` | Pooled | $0.947$ | $5.547$ | $0.171$ | `0.17` | **$82.9\%$** |
| $k\text{ (}10^{-4}\text{ m}^2\text{ s}^{-2}\text{)}$ | `POD-RBF` | Aligned | $2.00$ | $5.547$ | $0.360$ | `0.36` | **$64.0\%$** |
| $u\text{ (m s}^{-1}\text{)}$ | `FiLM-MLP` | Pooled | $0.0381$ | $0.1051$ | $0.363$ | `0.36` | **$63.7\%$** |
| $v\text{ (m s}^{-1}\text{)}$ | `FiLM-MLP` | Pooled | $0.0383$ | $0.1175$ | $0.326$ | `0.33` | **$67.4\%$** |
| $w\text{ (}10^{-3}\text{ m s}^{-1}\text{)}$ | `FiLM-MLP` | Pooled | $1.43$ | $1.577$ | $0.907$ | `0.91` | **$9.3\%$** |
| $w\text{ (}10^{-3}\text{ m s}^{-1}\text{)}$ | `POD-RBF` | Aligned | $1.01$ | $1.577$ | $0.641$ | `0.64` | **$35.9\%$** |
| $w\text{ (}10^{-3}\text{ m s}^{-1}\text{)}$ | `Nearest case` | Aligned | $1.08$ | $1.577$ | $0.686$ | `0.69` | **$31.4\%$** |

* **Unified Model-Count Taxonomy**:
  Across the Abstract, Introduction, Section 3.2, and Section 4.2, the benchmark model suite is now stated consistently as **19 evaluated surrogate model variants**:
  - **16 cross-configuration pooled models**: `Mean baseline`, `Ridge`, `Random Forest`, `Extra Trees`, `XGBoost`, `CatBoost`, `DeepONet`, `PINN data-only`, `PINN continuity`, `Domain-informed FiLM-MLP`, `Domain-informed Concat-MLP` (conditioning ablation), `Latent Neural Operator`, `Transolver-style`, `GAMS-GNN-style`, `F-FNO`, and `U-FNO3D`.
  - **3 within-group mesh-aligned models**: `POD-RBF`, `POD-GPR`, and `Nearest case`.
  - **1 non-competing discretization reference**: `Lattice floor (oracle)`, which quantifies the pure $64\times 64\times 16$ voxelization floor.
  - Figure 5 displays 18 models (omitting `POD-GPR` because its error profile closely tracks `POD-RBF`), while Supplementary Tables S1–S8 report all 20 rows (`19` competing models + `Lattice floor (oracle)`).

---

### M2 & Q2: Demonstrated Train-Only Preprocessing and Split Hygiene
* **Verified Code Audit from [`notebooks/kaggle_clean_case_cv_predictions.ipynb`](file:///c:/Users/bargh/Downloads/BTP_data/notebooks/kaggle_clean_case_cv_predictions.ipynb)**:
  1. **Pairwise Disjoint Case Partitioning (Cell 6, lines 302–335)**: Every five-fold and leave-one-group-out split enforces `assert not (set(s["train_ids"]) & set(s["test_ids"]))`, `assert not (set(s["val_ids"]) & set(s["test_ids"]))`, and `assert not (set(s["train_ids"]) & set(s["val_ids"]))`. Complete split manifests for all three seeds are archived in [`results/split_assignments.csv`](file:///c:/Users/bargh/Downloads/BTP_data/results/split_assignments.csv) and `results/clean_gpu_runs/seed_2026..2028/split_assignments.csv`.
  2. **Small-Stratum Protection During Validation Selection (`pick_val`, Cell 6, lines 281–300)**: Validation cases (4 per fold) are drawn **exclusively from strata with at least 8 non-test cases** (`eligible = [g for g, ids in by_group.items() if len(ids) >= 8]`, i.e., `HHHC/small` and `VHVC/large`). Consequently, small groups (`HHVC/small`, $n=5$; `VHVC/small`, $n=4$) are **never** depleted by validation holdout and retain 100% of their non-test cases ($4$ and $3\text{--}4$ training cases per fold, respectively) for model fitting.
  3. **Strictly Train-Only Feature & Target Scaling (`SplitData` & `TargetScaler`, Cell 8, lines 369–475)**: `cf_mean` and `cf_scale` (for coordinates and case descriptors) and `TargetScaler` ($\operatorname{arcsinh}(y/\operatorname{median}(|y_{\mathrm{train}}|))$ followed by z-score standardization on heavy-tailed $\varepsilon$ and $k$, and z-score standardization on the other 6 fields) are fit strictly on `split["train_ids"]`.
  4. **Strictly Train-Only Within-Group POD Basis and RBF Tuning (`run_group_method`, Cell 18, lines 1244–1294)**: For each group $g$, training cases are restricted to `tr = [c for c in split["train_ids"] if GROUP_OF[c] == g]`. Operating-parameter standardization (`pmean`, `pstd`), snapshot mean subtraction (`mean = fields_train.mean(axis=0)`), truncated SVD (`np.linalg.svd(cen, full_matrices=False)`, $\le 12$ modes, $99.99\%$ energy), and leave-one-out cross-validated RBF kernel selection (`tune_rbf` over $\ell \in \{0.25, 0.5, 1.0, 2.0, 4.0, 8.0\}$ and $\alpha \in \{10^{-6}, 10^{-4}, 10^{-2}, 10^{-1}, 1.0\}$) use only `tr`.

---

### M3 & Q3: CFD Provenance, EOS Verification vs. Flow Validation, and Hydrostatic Reference Field
* **Resolution in Manuscript (Sections 2.1, 2.2, and 4.4)**:
  1. **Legacy RANS Exports & EOS Consistency**: Section 2.1 explicitly states that the 42 cases are legacy steady-state RANS exports in which $k$ and $\varepsilon$ are modeled turbulence-closure variables (not DNS/LES or experimental measurements), and that the `CoolProp` check (median $0.0008\%$) verifies internal thermodynamic equation-of-state consistency between exported $(T, p_{\mathrm{abs}})$ and $\rho$ rather than mesh-independence or experimental flow validation.
  2. **Spatially Varying Hydrostatic Reference Field**: Section 2.2 corrects the explanation of small within-case gauge-pressure standard deviation ($\bar{\sigma}_c = 1.51\text{ kPa}$) to state that the solver subtracts a **spatially varying hydrostatic reference field** $\rho_0 \mathbf{g}\cdot\mathbf{x}$ (and notes explicitly that subtracting a spatially constant offset cannot change within-case variance).
  3. **Unrecorded Mass-Charge / Pressure-Control Descriptors**: Section 2.2 notes that because fluid mass-charge and solver pressure-initialization settings were not separately exported in the legacy CFD archive and $\bar{\rho}_{\mathrm{plane},c}$ varies across steady runs at fixed $(p_{\mathrm{op}}, T_{\mathrm{in}})$, the 42 cases are treated as independent steady-state operating points.

---

### M4: Matched vs. Unmatched Hyperparameters in the Node-Random vs. Case-Disjoint Experiment
* **Resolution in Manuscript (Abstract, Section 4.1, Table 3, and Figure 4)**:
  - Both Table 3 and Figure 4 ([`random_node_vs_case_disjoint_nmae.pdf`](file:///c:/Users/bargh/Downloads/BTP_data/results/graphs/random_node_vs_case_disjoint_nmae.pdf)) are now generated from the exact same single-seed (`seed=2026`) split-comparison dataset ([`results/co2_candidate_case_and_random_node_metrics.csv`](file:///c:/Users/bargh/Downloads/BTP_data/results/co2_candidate_case_and_random_node_metrics.csv)) and explicitly separate:
    1. **3 Strictly Matched-Configuration Tree Models (`Random Forest`, `Extra Trees`, `CatBoost`)**, which use identical hyperparameters across both splits: case-macro MAE inflation of **$6.78\times\text{--}8.25\times$** on $\rho$, **$11.29\times\text{--}30.08\times$** on $p_{\mathrm{gauge}}$, and **$5.62\times\text{--}10.24\times$** on $T$ (median case nMAE inflation of **$3.52\times\text{--}4.05\times$** on $\rho$ and **$2.95\times\text{--}4.39\times$** on $T$).
    2. **`XGBoost` (`max_depth=8` on `case_disjoint` vs. `max_depth=6` on `random_node`)**: reported in its own dedicated column in Table 3 ($10.37\times$ on $\rho$, $87.54\times$ on $p_{\mathrm{gauge}}$, $7.43\times$ on $T$) and dashed-dotted curve in Figure 4.

---

### M5: Complete Pressure Aggregation Definitions and Zero-Residual Baseline
* **Resolution in Manuscript (Section 3.1, Section 4.2, and Table 4)**:
  - **Population Standard Deviation (`ddof=0`)**: Section 3.1 defines $\sigma_c = \sqrt{N_c^{-1}\sum_{i=1}^{N_c}(y_{c,i}-\bar{y}_c)^2}$ with `ddof=0`, ensuring that the orthogonal decomposition $\mathrm{RMSE}_c^2 = (\hat{\bar{p}}_c - \bar{p}_c)^2 + \mathrm{RMSE}_{\mathrm{res},c}^2$ holds identically for every case $c$.
  - **Exact Case and Seed Aggregation Order**: For each seed $s \in \{2026, 2027, 2028\}$, $e_{\mathrm{level},c}$ and $\mathrm{RMSE}_{\mathrm{res},c}$ are averaged across the 42 cases while $\mathrm{nRMSE}_{\mathrm{res},c} = \mathrm{RMSE}_{\mathrm{res},c}/\max(\sigma_c, 1\text{ kPa})$ is summarized by its 42-case median, and the resulting seed-level statistics are averaged across the three seeds.
  - **Zero-Residual Baseline ($\tilde{\hat{p}}_{c,i} = 0$)**: Included directly in Table 4 (`Mean baseline`, `Zero-res` scope) and Section 4.2: predicting zero within-case spatial pressure variation already achieves $\mathrm{RMSE}_{\mathrm{res}} = 2.89\text{ kPa}$ ($2886\text{ Pa}$) and $\mathrm{nRMSE}_{\mathrm{res}} = 0.202$. Only `Nearest case` ($0.67\text{ kPa}$, $\mathrm{nRMSE}_{\mathrm{res}} = 0.0122$), `POD-GPR` ($0.46\text{ kPa}$, $0.0457$), `POD-RBF` ($0.47\text{ kPa}$, $0.0483$), and `XGBoost` ($1.01\text{ kPa}$, $0.151$) beat the zero-residual baseline.

---

### M6 & Q4: Optimization Protocol, Continuity Chain-Rule Scaling, Parameter Counts, Run Times, and `Lattice floor (oracle)`
* **Resolution in Manuscript (Section 3.2, Table 2, Section 4.2, and Table 4)**:
  - **Parameter Counts & Training Wall-Clock Times (Table 2)**:
    - `Domain-informed FiLM-MLP`: **$1,069,064$ parameters** ($384\text{ s/fold}$) vs. `Domain-informed Concat-MLP`: **$344,072$ parameters** ($111\text{ s/fold}$), explicitly noting that the $725\text{k}$ additional parameters in `FiLM-MLP` reside in the five per-block linear conditioning generators `nn.Linear(NF, 2*W)`.
    - `GAMS-GNN-style`: $869,096$ ($746\text{ s/fold}$); `Transolver-style`: $715,784$ ($430\text{ s/fold}$); `DeepONet`: $699,528$ ($184\text{ s/fold}$); `Latent Neural Operator`: $583,560$ ($666\text{ s/fold}$); `PINN data-only` / `PINN continuity`: $277,512$ ($86.8\text{ s}$ / $235\text{ s/fold}$); `U-FNO3D`: $212,408$ ($125\text{ s/fold}$); `F-FNO`: $155,888$ ($138\text{ s/fold}$).
  - **Physical Chain-Rule Scaling of the Continuity Penalty (Section 3.2)**:
    - Unstandardizes $\hat{\rho}$ and $\hat{\mathbf{u}}$ to physical units, differentiates $\hat{\rho}\hat{u}_a$ with respect to normalized coordinates $\tilde{x}_a \in [0,1]$, divides by each case's physical bounding-box span $\Delta x_{c,a}$ to form $\nabla_{\mathbf{x}}\cdot(\hat{\rho}\hat{\mathbf{u}}) = \sum_{a\in\{x,y,z\}} (\Delta x_{c,a})^{-1}\partial(\hat{\rho}\hat{u}_a)/\partial \tilde{x}_a$, normalizes by $S_{\mathrm{flux}} = (|\mu_\rho|+\sigma_\rho)\bar{\sigma}_u / \overline{\Delta x}$, and weights the mean squared dimensionless residual on $1,024$ collocation nodes/step by $\lambda_{\mathrm{cont}} = 0.05$.
  - **`Lattice floor (oracle)` Voxelization Reference (Table 4 & Section 4.2)**:
    - Reported directly in Table 4 for every field: projecting the true CFD fields onto the $64\times 64\times 16$ Cartesian grid and trilinearly interpolating back—with zero learning error—already incurs median case nMAE of **$0.336$ ($\rho$)**, **$0.334$ ($T$)**, **$0.339$ ($\varepsilon$)**, **$0.415$ ($k$)**, **$0.196$ ($u$)**, **$0.241$ ($v$)**, and **$0.323$ ($w$)**, exceeding `Domain-informed FiLM-MLP`'s out-of-fold error on all five velocity and turbulence fields.

---

### M7 & Q5: Leave-One-Group-Out Fixed Model Shortlist, Reframing as Joint Structural/Regime Shift, and Group-Wise / P90 Errors
* **Resolution in Manuscript (Section 3.1, Section 4.2, Section 4.3, Table 4, and Table 5)**:
  - **Reframed as Joint Structural and Operating-Regime Shift**: Section 3.1 and Section 4.3 explicitly frame leave-one-group-out as combined structural and operating-regime shift rather than pure geometric extrapolation.
  - **Fixed Preselected Shortlist in Table 5**: Table 5 now reports the full 3-seed median case nMAE tuple $(\rho / T / \varepsilon / k / v)$ across all four held-out groups for a **fixed preselected shortlist of seven models** (`Mean baseline`, `Extra Trees`, `XGBoost`, `DeepONet`, `PINN continuity`, `Concat-MLP`, `FiLM-MLP`) above the retrospective best-model summary.
  - **90th-Percentile (P90) and Group-Wise Errors**: Table 4 now includes a dedicated **P90 nMAE** column for every field, and Section 4.2 reports group-wise median nMAE across `VHVC/large`, `VHVC/small`, `HHHC/small`, and `HHVC/small`.

---

### M8 & Q5: Main-Text Median and Worst-Case Spatial Maps and Paired Case-Level Effect Sizes
* **Resolution in Manuscript (Section 4.2, Section 4.3, and Figure 7)**:
  - **Main-Text Median and Worst Spatial Reconstructions**: [`heldout_cfd_prediction_error_maps_median_worst.pdf`](file:///c:/Users/bargh/Downloads/BTP_data/results/graphs/heldout_cfd_prediction_error_maps_median_worst.pdf) is now included directly in the main text as **Figure 7** immediately following the showcase maps in Figure 6.
  - **Paired Case-Level Effect Sizes**: Section 4.2 now reports the median paired case-level nMAE difference $\Delta = \mathrm{Median}_c(\mathrm{nMAE}_{\mathrm{comp},c} - \mathrm{nMAE}_{\mathrm{FiLM},c})$ and its 95% bootstrap confidence interval across the 42 cases alongside the exploratory Friedman and Holm-adjusted Wilcoxon signed-rank tests.
