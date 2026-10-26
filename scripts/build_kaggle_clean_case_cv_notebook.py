from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / "benchmark.ipynb"
OUTPUT = RESULTS / "kaggle_clean_case_cv_predictions.ipynb"

INTRO = """# Clean CO₂-domain case-disjoint benchmark and OOF field predictions

This notebook is a self-contained, resumable Kaggle T4 ×2 run. It reads the filtered CO₂-candidate parquet when available; otherwise it loads the combined source table and applies the documented fallback `density <= 950 kg m⁻³`. The `T_205` case keeps its ID while its input feature is set to 305 K. A zone-only CFD-Post re-export is preferred and is required to confirm fluid identity.

The 17-model suite comprises the original pointwise, tree, neural-operator, graph, transformer, mesh-aligned, and regular-grid adaptations. The run also includes FiLM-versus-concatenation. A proposed loop-unrolled level/shape architecture prototype is provided separately, but remains pending registration and training until validated loop-path features and EOS inputs are available. Four configuration/geometry group holdouts are retained; mesh-aligned-only methods are explicitly marked not applicable to unseen groups. Per-case OOF predictions are saved for scheduled models. Seeds 2026, 2027, and 2028 write to separate subfolders under `/kaggle/working`. Fold/model checkpoints, progress logs, and a wall-clock guard support resumption; the status table distinguishes completed, pending, and not-applicable experiments.
"""

DATA_LOAD_CELL = r'''# Normalize Kaggle and local mega-dataset schemas before building case arrays.
t_load = time.monotonic()
import re

def _key(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

_full = (
    pq.read_table(PARQUET_PATH).to_pandas()
    if PARQUET_PATH.lower().endswith(".parquet")
    else pd.read_csv(PARQUET_PATH)
)
_lookup = {_key(column): column for column in _full.columns}
_aliases = {
    "case_id": ["case_id", "case id", "case", "case_name", "filename", "source_filename"],
    "configuration": ["configuration", "config", "orientation"],
    "geometry_class": ["geometry_class", "geometry", "loop_size"],
    "power_parameter_value": ["power_parameter_value", "power_w", "heater_power_w", "power", "q_w"],
    "pressure_bar": ["pressure_bar", "operating_pressure_bar", "p_op_bar", "pressure_input_bar"],
    "mass_flow_kg_s": ["mass_flow_kg_s", "massflow_kg_s", "mdot_kg_s", "coolant_inlet_mass_flow_kg_s"],
    "temperature_parameter_value": ["temperature_parameter_value", "temperature_input_k", "temperature_k", "t_in_k", "coolant_inlet_temperature_k"],
    "loop_x_extent_m": ["loop_x_extent_m", "x_extent_m", "loop_length_x_m"],
    "loop_y_extent_m": ["loop_y_extent_m", "y_extent_m", "loop_length_y_m"],
    "x_m": ["x_m", "x", "x_meters", "x_m_"],
    "y_m": ["y_m", "y", "y_meters", "y_m_"],
    "z_m": ["z_m", "z", "z_meters", "z_m_"],
    "density_kg_m3": ["density_kg_m3", "density", "density_kg_m3_"],
    "pressure_pa": ["pressure_pa", "pressure", "gauge_pressure_pa", "pressure_pa_"],
    "temperature_c": ["temperature_c", "temperature", "temperature_c_"],
    "turbulence_eddy_dissipation_m2_s3": ["turbulence_eddy_dissipation_m2_s3", "turbulence_eddy_dissipation", "eddy_dissipation", "turbulenceeddydissipationm2s3"],
    "turbulence_kinetic_energy_m2_s2": ["turbulence_kinetic_energy_m2_s2", "turbulence_kinetic_energy", "tke"],
    "velocity_u_m_s": ["velocity_u_m_s", "velocity_u", "velocity u m s^-1", "u_velocity", "u"],
    "velocity_v_m_s": ["velocity_v_m_s", "velocity_v", "velocity v m s^-1", "v_velocity", "v"],
    "velocity_w_m_s": ["velocity_w_m_s", "velocity_w", "velocity w m s^-1", "w_velocity", "w"],
}
_rename = {}
for _canonical, _candidates in _aliases.items():
    if _canonical not in _full.columns:
        _source = next((_lookup.get(_key(candidate)) for candidate in _candidates if _lookup.get(_key(candidate)) is not None), None)
        if _source is not None:
            _rename[_source] = _canonical
_full = _full.rename(columns=_rename)
if "case_id" not in _full.columns:
    raise ValueError(f"Case identifier/filename missing. Dataset columns: {list(_full.columns)}")
_full["case_id"] = _full["case_id"].astype(str).str.replace(r"\.csv$", "", regex=True)
_parsed = _full["case_id"].str.extract(
    r"^(?P<cfg>[A-Za-z]+)_P(?P<power>\d+(?:\.\d+)?)_(?P<pbar>\d+(?:\.\d+)?)_bar_"
    r"(?P<mdot>\d+(?:\.\d+)?)kgs_T_(?P<tin>-?\d+(?:\.\d+)?)$"
)
for _name, _parsed_name in [
    ("configuration", "cfg"),
    ("power_parameter_value", "power"),
    ("pressure_bar", "pbar"),
    ("mass_flow_kg_s", "mdot"),
    ("temperature_parameter_value", "tin"),
]:
    if _name not in _full.columns:
        _full[_name] = _parsed[_parsed_name]
    else:
        _full[_name] = _full[_name].fillna(_parsed[_parsed_name])
for _name in ["power_parameter_value", "pressure_bar", "mass_flow_kg_s", "temperature_parameter_value"]:
    _full[_name] = pd.to_numeric(_full[_name], errors="raise")
_required = ["case_id", "configuration", *COORDS, *TARGETS, *NUMERIC_CASE[:4]]
_missing = sorted(set(_required) - set(_full.columns))
if _missing:
    raise ValueError(f"Missing canonical fields {_missing}. Dataset columns: {list(_full.columns)}")
if _full["case_id"].isna().any():
    raise ValueError("Rows without a case identifier cannot be assigned to a case-level split.")
_full.loc[_full["case_id"].str.endswith("_T_205"), "temperature_parameter_value"] = 305.0
_case_metadata = ["configuration", *NUMERIC_CASE[:4]]
_metadata_variation = _full.groupby("case_id", observed=True)[_case_metadata].nunique(dropna=False)
if _metadata_variation.gt(1).any().any():
    raise ValueError("Operating metadata varies within at least one case.")
_candidate_nodes = _full.loc[
    _full["density_kg_m3"] <= 950.0, ["case_id", *COORDS, *TARGETS]
].copy()
_candidate_count = len(_candidate_nodes)
_nodes = _candidate_nodes.dropna(subset=["case_id", *COORDS, *TARGETS]).copy()
print(f"Removed {_candidate_count - len(_nodes):,} candidate rows with null case/coordinate/target values.")
if _nodes.empty:
    raise ValueError("The CO2-candidate filter removed all nodes.")
_extents = _nodes.groupby("case_id", sort=True).agg(
    x_min=("x_m", "min"), x_max=("x_m", "max"), y_min=("y_m", "min"), y_max=("y_m", "max")
)
_extents["loop_x_extent_m"] = _extents["x_max"] - _extents["x_min"]
_extents["loop_y_extent_m"] = _extents["y_max"] - _extents["y_min"]
_extents["geometry_class"] = np.where(
    np.maximum(_extents["loop_x_extent_m"], _extents["loop_y_extent_m"]) > 2.0, "large", "small"
)
_meta = _full.drop_duplicates("case_id").drop(
    columns=["geometry_class", "loop_x_extent_m", "loop_y_extent_m"], errors="ignore"
)
_meta = _meta.merge(
    _extents[["loop_x_extent_m", "loop_y_extent_m", "geometry_class"]],
    left_on="case_id", right_index=True, how="left", validate="one_to_one"
)
_meta["configuration"] = _meta["configuration"].astype(str)
case_table = _meta[META].sort_values("case_id").reset_index(drop=True)
if case_table["case_id"].duplicated().any():
    raise ValueError("A case ID has inconsistent operating metadata.")
case_table["mesh_group"] = case_table.configuration + "|" + case_table.geometry_class.astype(str)
CASE_IDS = case_table.case_id.tolist()
CASE_POS = {c: i for i, c in enumerate(CASE_IDS)}
GROUP_OF = dict(zip(case_table.case_id, case_table.mesh_group))
MESH_GROUPS = sorted(case_table.mesh_group.unique())
GROUP_ALIGNED = {g: True for g in MESH_GROUPS}
del _full, _meta, _parsed, _extents, _candidate_nodes
gc.collect()
print(f"CO2-candidate nodes after secondary-fluid and null filters: {len(_nodes):,}")
_nodes["case_id"] = pd.Categorical(_nodes["case_id"], categories=CASE_IDS, ordered=True)
_codes = _nodes["case_id"].cat.codes.to_numpy()
_cats = np.asarray(CASE_IDS)
_order = np.argsort(_codes, kind="stable")
_coords = _nodes[COORDS].to_numpy(np.float32)[_order]
_y = _nodes[TARGETS].to_numpy(np.float32)[_order]
_counts = np.bincount(_codes, minlength=len(_cats))
del _nodes, _codes, _order
gc.collect()
if not (np.isfinite(_coords).all() and np.isfinite(_y).all()):
    raise ValueError("Non-finite coordinates or targets remain after data normalization.")
RAW_C, RAW_Y, NORM, SPAN = {}, {}, {}, {}
_ends = np.cumsum(_counts); _starts = _ends - _counts
for i, cid in enumerate(_cats):
    c = _coords[_starts[i]:_ends[i]]
    RAW_C[cid] = c
    RAW_Y[cid] = _y[_starts[i]:_ends[i]]
    low, span = c.min(0), np.maximum(np.ptp(c, axis=0), 1e-6)
    NORM[cid] = ((c - low) / span).astype(np.float32)
    SPAN[cid] = span.astype(np.float32)
missing = set(CASE_IDS) - set(RAW_Y)
if missing:
    raise ValueError(f"Cases without retained CO2-candidate nodes: {sorted(missing)[:5]}")
case_table["n_nodes"] = [len(RAW_Y[c]) for c in CASE_IDS]
print(f"Loaded {len(CASE_IDS)} cases and {int(case_table.n_nodes.sum()):,} candidate nodes.")
display(case_table.groupby(["mesh_group"]).agg(cases=("case_id", "size"), nodes_per_case=("n_nodes", "mean")))
'''

BENCHMARK_MARKDOWN = """# Clean-data case-disjoint benchmark

## Data and scope

- The loader accepts a combined tabular Parquet or CSV table, normalizes local snake-case and common Fluent/Kaggle unit-bearing headers, and fails on missing or inconsistent case metadata.
- Operating inputs are read from columns when available and otherwise parsed from the case identifier. Loop extents are derived from retained CO₂-candidate nodes. The `T_205` feature is set to 305 K based on the raw nodal-temperature audit while the original case ID is preserved.
- Nodes with density above 950 kg m⁻³ are excluded as an EOS-informed secondary-fluid candidate filter. This is not a confirmed Fluent zone classification; check the zone IDs before interpreting the filtered table as definitively CO₂-only.
- The outputs are sampled on cross-sectional planes, not a complete volumetric mesh. No wall temperature or heat-transfer coefficient is available.

## Evaluation protocol

- Five-fold case-level cross-validation holds out each CFD case exactly once. No nodes from a test case are used for training or neural validation.
- A separate four-group leave-one-configuration/geometry-group-out protocol tests models that accept pooled coordinates and operating conditions. Nearest-case and POD transfer are mesh-aligned and are marked not applicable when their training group is absent.
- Neural checkpoint selection uses case-level validation data only. Exact case memberships, per-case metrics, run status, and training history are saved.
- F-FNO and U-FNO3D adaptations are retained and reported as regular-grid remapping models; the lattice-oracle output is a remapping diagnostic, not a predictive model.
- The FiLM-MLP and concatenation-conditioned MLP use the same coordinate features, hidden width/depth, loss, seed, and training budget. This is a conditioning-mechanism ablation, not a parameter-count-matched comparison.

## Reproducibility and outputs

- The default seed is 2026; set `SCWR_SEED` to 2027 or 2028 for independent runs. Each seed writes to its own output directory.
- Task checkpoints, progress logs, validation-based checkpoint selection, and an 11-hour wall-clock guard make the Kaggle run resumable.
- OOF predictions, case-level truth arrays, selection tables, maps, exact splits, and metrics are saved under the seed-specific output directory.
- Legacy all-exported-node scores remain separate from clean-domain results and are labeled as contamination-sensitivity references.
"""

CONCAT_NET_CLASS = r'''class ConcatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bands", (2.0 ** torch.arange(NB, dtype=torch.float32)) * math.pi)
        self.D, self.W = NN_DEPTH + 1, NN_WIDTH
        self.inp = nn.Linear(3 + 3 * 2 * NB + NF, self.W)
        self.lins = nn.ModuleList([nn.Linear(self.W, self.W) for _ in range(self.D)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.W) for _ in range(self.D)])
        self.out = nn.Linear(self.W, NT)

    def forward(self, cond, xyz):
        features = torch.cat([xyz, fourier(xyz, self.bands), cond], dim=1)
        h = F.gelu(self.inp(features))
        for i in range(self.D):
            h = h + F.gelu(self.norms[i](self.lins[i](h)))
        return self.out(h)
'''

POD_GPR_FUNCTIONS = r'''def pod_gpr_predict(x_train, fields_train, x_query):
    n, nodes, nt = fields_train.shape
    pred = np.empty((len(x_query), nodes, nt), np.float32)
    dtr, dq = sqdist(x_train, x_train), sqdist(x_query, x_train)
    for t in range(nt):
        mean = fields_train[:, :, t].mean(0, dtype=np.float64)
        centered = fields_train[:, :, t].astype(np.float64) - mean
        eigenvalues, eigenvectors = np.linalg.eigh(centered @ centered.T)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        total = float(eigenvalues.sum())
        if total <= 1e-20 or n < 2:
            pred[:, :, t] = mean
            continue
        rank = max(1, min(n - 1, POD_MAX_MODES, int(np.searchsorted(
            np.cumsum(eigenvalues) / total, POD_ENERGY
        ) + 1)))
        basis = (centered.T @ eigenvectors[:, :rank]) / np.sqrt(
            np.maximum(eigenvalues[:rank], 1e-20)
        )[None, :]
        scores = centered @ basis
        kernel_train = np.exp(-0.5 * dtr) + 1e-4 * np.eye(n)
        kernel_query = np.exp(-0.5 * dq)
        coefficients = np.linalg.solve(kernel_train, scores)
        pred[:, :, t] = (mean[None, :] + (kernel_query @ coefficients) @ basis.T).astype(
            np.float32
        )
    return pred

'''

PROPOSED_MODEL_MARKDOWN = """## Proposed Loop-Unrolled Offset-Shape Operator (pending geometry input)

The prototype below defines the gated architecture and differentiable EOS lookup. Training and score ingestion remain pending a PhD-scholar centerline/path export containing `s_star`, `r_star`, `theta`, leg type, heater/cooler indicators, and velocity rotation. Do not treat the prototype definition as an experiment or activate the proposed-method claim until the prespecified paired-bootstrap gate passes.

The four required ablations are no loop unrolling, no level/shape split, no EOS-derived density, and no periodic encoding. Their result cells remain pending because the model cannot be trained or scored until the geometry metadata is validated.
"""

PROPOSED_MODEL_CODE = r'''class DifferentiableCO2EOS(nn.Module):
    def __init__(self, temperature_k, pressure_pa, density_kg_m3):
        super().__init__()
        self.register_buffer("temperature_k", torch.as_tensor(temperature_k, dtype=torch.float32))
        self.register_buffer("pressure_pa", torch.as_tensor(pressure_pa, dtype=torch.float32))
        self.register_buffer("density_kg_m3", torch.as_tensor(density_kg_m3, dtype=torch.float32))

    def forward(self, temperature_k, pressure_pa):
        t = temperature_k.clamp(self.temperature_k[0], self.temperature_k[-1])
        p = pressure_pa.clamp(self.pressure_pa[0], self.pressure_pa[-1])
        ti = torch.bucketize(t.contiguous(), self.temperature_k).clamp(1, len(self.temperature_k) - 1)
        pi = torch.bucketize(p.contiguous(), self.pressure_pa).clamp(1, len(self.pressure_pa) - 1)
        t0, t1 = self.temperature_k[ti - 1], self.temperature_k[ti]
        p0, p1 = self.pressure_pa[pi - 1], self.pressure_pa[pi]
        wt = ((t - t0) / (t1 - t0).clamp_min(1e-8)).clamp(0, 1)
        wp = ((p - p0) / (p1 - p0).clamp_min(1e-8)).clamp(0, 1)
        d00 = self.density_kg_m3[ti - 1, pi - 1]
        d01 = self.density_kg_m3[ti - 1, pi]
        d10 = self.density_kg_m3[ti, pi - 1]
        d11 = self.density_kg_m3[ti, pi]
        return (1 - wt) * ((1 - wp) * d00 + wp * d01) + wt * ((1 - wp) * d10 + wp * d11)


class LoopUnrolledOffsetShapeOperator(nn.Module):
    def __init__(self, case_width, path_width, width=256, bands=6,
                 loop_unrolling=True, split_levels=True, eos_density=True, periodic=True):
        super().__init__()
        self.loop_unrolling = loop_unrolling
        self.split_levels = split_levels
        self.eos_density = eos_density
        self.periodic = periodic
        self.register_buffer("bands", 2.0 ** torch.arange(bands, dtype=torch.float32) * math.pi)
        spatial_width = path_width + 2 * bands if periodic else path_width
        self.level = nn.Sequential(nn.Linear(case_width, width), nn.GELU(),
                                   nn.Linear(width, width), nn.GELU(), nn.Linear(width, 10))
        self.film = nn.Sequential(nn.Linear(case_width, width), nn.GELU(),
                                  nn.Linear(width, 2 * width))
        self.spatial = nn.Sequential(nn.Linear(spatial_width, width), nn.GELU(),
                                     nn.Linear(width, width), nn.GELU())
        self.cartesian = nn.Sequential(nn.Linear(3, width), nn.GELU(),
                                       nn.Linear(width, width), nn.GELU())
        self.shape = nn.Sequential(nn.Linear(width + case_width, width), nn.GELU(),
                                   nn.Linear(width, width), nn.GELU(), nn.Linear(width, 7))
        self.absolute = nn.Linear(width + case_width, 8)

    def forward(self, case_embedding, path_features, rotation_angle,
                operating_pressure_pa, eos=None, cartesian_xyz=None):
        case_embedding = case_embedding.reshape(1, -1)
        levels = self.level(case_embedding).squeeze(0)
        expanded_case = case_embedding.expand(path_features.shape[0], -1)
        if self.loop_unrolling:
            if self.periodic:
                phase = path_features[:, :1] * self.bands
                encoded = torch.cat([path_features, torch.sin(phase), torch.cos(phase)], dim=1)
            else:
                encoded = path_features
            spatial = self.spatial(encoded)
        else:
            if cartesian_xyz is None:
                raise ValueError("The no-unrolling ablation requires Cartesian coordinates.")
            spatial = self.cartesian(cartesian_xyz)
        gamma, beta = self.film(expanded_case).chunk(2, dim=1)
        conditioned = spatial * (1 + gamma) + beta
        fused = torch.cat([conditioned, expanded_case], dim=1)
        if not self.split_levels:
            fields = self.absolute(fused)
            temperature_c, pressure_gauge = fields[:, 1], fields[:, 2]
            aligned_u, aligned_v = fields[:, 5], fields[:, 6]
            c, s = torch.cos(rotation_angle), torch.sin(rotation_angle)
            fields = torch.stack([fields[:, 0], temperature_c, pressure_gauge, fields[:, 3],
                                  fields[:, 4], aligned_u * c - aligned_v * s,
                                  aligned_u * s + aligned_v * c, fields[:, 7]], dim=1)
        else:
            shape = self.shape(fused)
            mean = torch.stack([levels[0], levels[1], *([levels.new_zeros(())] * 5)])
            scale = torch.exp(levels[3:10].clamp(-8, 8))
            field_shape = mean[None, :] + scale[None, :] * shape
            aligned_u, aligned_v = field_shape[:, 4], field_shape[:, 5]
            c, s = torch.cos(rotation_angle), torch.sin(rotation_angle)
            fields = torch.stack([levels.new_zeros(len(path_features)), field_shape[:, 0],
                                  field_shape[:, 1], field_shape[:, 2], field_shape[:, 3],
                                  aligned_u * c - aligned_v * s,
                                  aligned_u * s + aligned_v * c, field_shape[:, 6]], dim=1)
        t_k = fields[:, 1] + 273.15
        p_abs = fields[:, 2] + operating_pressure_pa
        if self.eos_density:
            if eos is None:
                raise ValueError("EOS-derived density requires a differentiable CO2 EOS table.")
            density = eos(t_k, p_abs)
            fields[:, 0] = density
        else:
            if self.split_levels:
                fields[:, 0] = self.absolute(fused)[:, 0]
        return fields, levels


def loop_unrolled_huber_loss(prediction, truth, case_level_prediction,
                             case_level_truth, energy_balance_residual,
                             energy_weight=1e-3):
    data_loss = F.huber_loss(prediction, truth, delta=1.0)
    level_loss = F.huber_loss(case_level_prediction, case_level_truth, delta=1.0)
    balance_loss = torch.mean(energy_balance_residual**2)
    return data_loss + level_loss + energy_weight * balance_loss


if not any(spec["name"] == "Loop-Unrolled Offset-Shape Operator" for spec in MODEL_SPECS):
    STATUS_ROWS = [
        row for row in STATUS_ROWS
        if not (
            row.get("model") == "Loop-Unrolled Offset-Shape Operator"
            and row.get("split") == "all_folds"
        )
    ]
    STATUS_ROWS.append({
        "protocol": "kfold",
        "split": "all_folds",
        "model": "Loop-Unrolled Offset-Shape Operator",
        "status": "pending_author_geometry",
        "note": "Training is gated on validated loop centerline, segment, heater/cooler, and rotation metadata.",
    })
    save_progress()
'''

PREDICTION_CELL = r'''# Select one fixed model per field by median case nMAE, then export maps for best/median/worst cases.
import matplotlib.pyplot as plt

MODEL_FOR_MAPS = {
    "Mean baseline", "Nearest case", "Random Forest", "Extra Trees",
    "XGBoost", "CatBoost", "DeepONet", "PINN data-only",
    "PINN continuity", "Domain-informed FiLM-MLP",
    "Domain-informed Concat-MLP", "Latent Neural Operator",
    "Transolver-style", "F-FNO", "U-FNO3D", "GAMS-GNN-style",
    "POD-RBF", "POD-GPR", "Loop-Unrolled Offset-Shape Operator",
}
metric_frame = pd.DataFrame(RESULT_ROWS)
oof_metrics = metric_frame.loc[
    (metric_frame["protocol"] == "kfold")
    & metric_frame["model"].isin(MODEL_FOR_MAPS)
].copy()
if oof_metrics.empty:
    raise RuntimeError("No out-of-fold metrics were produced.")

selections = []
for target in TARGETS:
    target_metrics = oof_metrics.loc[oof_metrics["target"] == target]
    model_medians = target_metrics.groupby("model")["mae_over_std"].median().sort_values()
    selected_model = model_medians.index[0]
    ranked = target_metrics.loc[target_metrics["model"] == selected_model].sort_values(
        ["mae_over_std", "case_id"]
    ).reset_index(drop=True)
    for role, row_index in [
        ("best", 0),
        ("median", len(ranked) // 2),
        ("worst", len(ranked) - 1),
    ]:
        row = ranked.iloc[row_index]
        selections.append({
            "target": target,
            "selected_model_by_median_case_nmae": selected_model,
            "case_role": role,
            "case_id": row["case_id"],
            "case_nmae": row["mae_over_std"],
            "case_r2": row["r2"],
        })
selection_frame = pd.DataFrame(selections)
selection_frame.to_csv(OUTPUT_DIR / "case_prediction_selection.csv", index=False)
print(selection_frame.to_string(index=False))

TARGET_GROUPS = {
    "main": ["density_kg_m3", "temperature_c", "pressure_pa"],
    "supplement": [t for t in TARGETS if t not in {"density_kg_m3", "temperature_c", "pressure_pa"}],
}
for group_name, targets_for_plot in TARGET_GROUPS.items():
    fig, axes = plt.subplots(3 * len(targets_for_plot), 3, figsize=(15, 4.0 * len(targets_for_plot)), squeeze=False)
    for row_index, selection in selection_frame.loc[
        selection_frame["target"].isin(targets_for_plot)
    ].reset_index(drop=True).iterrows():
        target = selection["target"]
        case_id = selection["case_id"]
        model_name = selection["selected_model_by_median_case_nmae"]
        target_index = TARGETS.index(target)
        model_file = PRED_DIR / f"{safe(model_name)}__{safe(case_id)}.npz"
        case_file = PRED_DATA_DIR / f"{safe(case_id)}.npz"
        if not model_file.exists() or not case_file.exists():
            raise FileNotFoundError(f"Missing OOF prediction or case data for {model_name}, {case_id}")
        with np.load(model_file) as pred_file, np.load(case_file) as case_file_data:
            prediction = pred_file["prediction"][:, target_index]
            truth = case_file_data["truth"][:, target_index]
            coords = case_file_data["coordinates"]
        rng = np.random.default_rng(SEED + target_index)
        keep = np.arange(len(truth))
        if len(keep) > 25_000:
            keep = np.sort(rng.choice(keep, 25_000, replace=False))
        truth_values, prediction_values = truth[keep], prediction[keep]
        error_values = np.abs(prediction_values - truth_values)
        lo, hi = np.nanpercentile(np.r_[truth_values, prediction_values], [1, 99])
        error_hi = max(float(np.nanpercentile(error_values, 99)), 1e-12)
        axes[row_index, 0].scatter(coords[keep, 0], coords[keep, 1], c=truth_values, s=2, cmap="inferno", vmin=lo, vmax=hi, rasterized=True)
        axes[row_index, 1].scatter(coords[keep, 0], coords[keep, 1], c=prediction_values, s=2, cmap="inferno", vmin=lo, vmax=hi, rasterized=True)
        axes[row_index, 2].scatter(coords[keep, 0], coords[keep, 1], c=error_values, s=2, cmap="magma", vmin=0, vmax=error_hi, rasterized=True)
        for col, title in enumerate(["CFD", model_name, "Absolute error"]):
            axes[row_index, col].set_title(title)
            axes[row_index, col].set_xlabel("X (m)")
            axes[row_index, col].set_ylabel("Y (m)")
            axes[row_index, col].set_aspect("equal", adjustable="datalim")
        axes[row_index, 0].text(
            -0.24, 0.5,
            f"{target}\n{selection['case_role']}\n{case_id}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8,
        )
    fig.suptitle(
        f"Out-of-fold reconstruction examples: {group_name} targets\n"
        "XY projection; at most 25,000 measured nodes shown per panel",
        y=1.002,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"prediction_maps_{group_name}_targets.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"prediction_maps_{group_name}_targets.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

print("OOF prediction files:", sum(1 for _ in PRED_DIR.glob("*.npz")))
print("Case data files:", sum(1 for _ in PRED_DATA_DIR.glob("*.npz")))
'''


def source_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)
    if text and not text.endswith("\n"):
        cell["source"][-1] += "\n"


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected one {description} replacement, found {count}.")
    return text.replace(old, new, 1)


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    sources = [source_text(cell) for cell in cells]
    load_index = next(i for i, source in enumerate(sources) if "# Load data" in source)
    config_index = next(i for i, source in enumerate(sources) if "PRED_DIR = OUTPUT_DIR" in source)
    runner_index = next(i for i, source in enumerate(sources) if "# Runner" in source)
    config_source = sources[config_index]
    metric_index = next(i for i, source in enumerate(sources) if "def metric_rows(ytrue, ypred, base):" in source)
    metric_source = replace_once(
        sources[metric_index],
        '        r = dict(base, target=t, n_nodes=len(truth), mae=float(np.abs(err).mean()), rmse=float(np.sqrt(sse / len(err))),\n'
        '                 median_ae=float(np.median(np.abs(err))), bias=float(err.mean()), std_y=sd,\n'
        '                 mae_over_std=float(np.abs(err).mean() / sd) if sd > 0 else np.nan,\n'
        '                 r2=1 - sse / sst if sst > 0 else np.nan, sum_y=float(truth.sum()), sum_y2=float(truth @ truth),\n'
        '                 sum_sq_err=sse)',
        '        mean_truth = float(truth.mean()); mean_prediction = float(ypred[:, j].mean())\n'
        '        residual_error = err - err.mean()\n'
        '        is_pressure = t == "pressure_pa"\n'
        '        r = dict(base, target=t, n_nodes=len(truth), mae=float(np.abs(err).mean()), rmse=float(np.sqrt(sse / len(err))),\n'
        '                 median_ae=float(np.median(np.abs(err))), bias=float(err.mean()), std_y=sd,\n'
        '                 mae_over_std=np.nan if is_pressure else (float(np.abs(err).mean() / sd) if sd > 0 else np.nan),\n'
        '                 r2=np.nan if is_pressure else (1 - sse / sst if sst > 0 else np.nan),\n'
        '                 sum_y=float(truth.sum()), sum_y2=float(truth @ truth), sum_sq_err=sse,\n'
        '                 case_mean_y=mean_truth, case_mean_prediction=mean_prediction,\n'
        '                 pressure_level_abs_error=abs(mean_prediction - mean_truth) if is_pressure else np.nan,\n'
        '                 pressure_level_sq_error=(mean_prediction - mean_truth) ** 2 if is_pressure else np.nan,\n'
        '                 pressure_residual_mae=float(np.abs(residual_error).mean()) if is_pressure else np.nan,\n'
        '                 pressure_residual_rmse=float(np.sqrt(np.mean(residual_error ** 2))) if is_pressure else np.nan,\n'
        '                 pressure_residual_nrmse=float(np.sqrt(np.mean(residual_error ** 2)) / max(sd, 1000.0)) if is_pressure else np.nan)',
        "pressure level and residual metrics",
    )
    set_source(cells[metric_index], metric_source)
    sources[metric_index] = metric_source

    config_source = replace_once(
        config_source,
        'PRED_DIR = OUTPUT_DIR / "showcase_predictions"; PRED_DIR.mkdir(exist_ok=True)',
        'PRED_DIR = OUTPUT_DIR / "case_predictions"; PRED_DIR.mkdir(exist_ok=True)\n'
        'PRED_DATA_DIR = OUTPUT_DIR / "case_data"; PRED_DATA_DIR.mkdir(exist_ok=True)\n'
        'PRED_MODELS = {"Mean baseline", "Nearest case", "Random Forest", "Extra Trees", '
        '"XGBoost", "CatBoost", "DeepONet", "PINN data-only", "PINN continuity", '
        '"Domain-informed FiLM-MLP", "Domain-informed Concat-MLP", '
        '"Latent Neural Operator", "Transolver-style", "F-FNO", "U-FNO3D", '
        '"GAMS-GNN-style", "POD-RBF", "POD-GPR", '
        '"Loop-Unrolled Offset-Shape Operator"}',
        "prediction directory setup",
    )
    set_source(cells[config_index], config_source)

    split_index = next(
        i for i, source in enumerate(sources) if 'SPLITS["geometry_holdout"] = SPLITS["geometry_holdout"][:1]' in source
    )
    split_source = replace_once(
        sources[split_index],
        'if SMOKE:\n    SPLITS["geometry_holdout"] = SPLITS["geometry_holdout"][:1]',
        'if SMOKE:\n    pass  # Retain all four observed configuration/geometry group holdouts.',
        "all geometry holdouts in smoke and full runs",
    )
    set_source(cells[split_index], split_source)
    config_source = source_text(cells[config_index])
    config_source = replace_once(
        config_source,
        "SEED = 2026",
        'SEED = int(os.environ.get("SCWR_SEED", "2026"))',
        "configurable run seed",
    )
    load_source = replace_once(
        sources[load_index],
        'MESH_GROUPS = sorted(case_table.mesh_group.unique())',
        'MESH_GROUPS = sorted(case_table.mesh_group.unique())\n'
        '# Mesh-aligned transfer is valid only within an observed configuration/geometry group.\n'
        'GROUP_ALIGNED = {g: True for g in MESH_GROUPS}',
        "aligned mesh-group flags",
    )
    set_source(cells[load_index], load_source)
    sources[load_index] = load_source
    config_source = replace_once(
        config_source,
        'OUTPUT_DIR = Path(os.environ.get("SCWR_OUT", "/kaggle/working" if Path("/kaggle/working").exists() else "mega_model_outputs"))\n'
        "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)",
        'OUTPUT_ROOT = Path(os.environ.get("SCWR_OUT", "/kaggle/working" if Path("/kaggle/working").exists() else "mega_model_outputs"))\n'
        'OUTPUT_DIR = OUTPUT_ROOT / f"seed_{SEED}"\n'
        "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)",
        "seed-specific output directory",
    )
    path_start = config_source.index("PARQUET_PATH =")
    path_end = config_source.index("OUTPUT_ROOT =")
    config_source = (
        config_source[:path_start]
        + 'PARQUET_PATH = os.environ.get("SCWR_PARQUET", "/kaggle/input/scwr-data/mega_dataset.parquet")\n'
        + 'if not Path(PARQUET_PATH).exists():\n'
        + '    candidates = sorted(Path("/kaggle/input").rglob("mega_dataset.parquet")) if Path("/kaggle/input").exists() else []\n'
        + '    if not candidates and Path("/kaggle/input").exists():\n'
        + '        candidates = sorted(Path("/kaggle/input").rglob("mega_dataset.csv"))\n'
        + '    if not candidates and (Path.cwd() / "mega_dataset.parquet").exists():\n'
        + '        candidates = [Path.cwd() / "mega_dataset.parquet"]\n'
        + '    if len(candidates) != 1:\n'
        + '        raise FileNotFoundError(f"Expected exactly one tabular mega_dataset.parquet or mega_dataset.csv; found {len(candidates)}. Set SCWR_PARQUET.")\n'
        + '    PARQUET_PATH = str(candidates[0])\n'
        + config_source[path_end:]
    )
    set_source(cells[config_index], config_source)

    set_source(cells[0], BENCHMARK_MARKDOWN)

    operator_index = next(
        i for i, source in enumerate(sources) if "## Transolver-style, GAMS-GNN-style, F-FNO" in source
    )
    operator_source = sources[operator_index]
    set_source(cells[operator_index], operator_source)

    set_source(cells[load_index], DATA_LOAD_CELL)

    transfer_index = next(
        i for i, source in enumerate(sources) if "def run_group_method(kind):" in source
    )
    transfer_source = sources[transfer_index]
    transfer_source = replace_once(
        transfer_source,
        "def run_group_method(kind):",
        POD_GPR_FUNCTIONS + "def run_group_method(kind):",
        "fixed-kernel POD-GPR implementation",
    )
    transfer_source = replace_once(
        transfer_source,
        '            if kind == "pod":\n'
        '                out = pod_rbf_predict(xtr.astype(np.float64), fields, xte.astype(np.float64))\n'
        '            else:\n'
        '                out = fields[sqdist(xte, xtr).argmin(1)]',
        '            if kind == "pod":\n'
        '                out = pod_rbf_predict(xtr.astype(np.float64), fields, xte.astype(np.float64))\n'
        '            elif kind == "pod_gpr":\n'
        '                out = pod_gpr_predict(xtr.astype(np.float64), fields, xte.astype(np.float64))\n'
        '            else:\n'
        '                out = fields[sqdist(xte, xtr).argmin(1)]',
        "POD-GPR transfer dispatch",
    )
    set_source(cells[transfer_index], transfer_source)

    film_index = next(i for i, source in enumerate(sources) if "class FiLMNet(nn.Module):" in source)
    film_source = replace_once(
        sources[film_index],
        "class FiLMNet(nn.Module):",
        CONCAT_NET_CLASS + "\n\nclass FiLMNet(nn.Module):",
        "FiLM concatenation ablation model",
    )
    set_source(cells[film_index], film_source)

    runner_source = sources[runner_index]
    runner_source = replace_once(
        runner_source,
        '                if protocol == "kfold" and cid == SHOWCASE.get(GROUP_OF[cid]):\n'
        '                    np.save(PRED_DIR / f"{safe(spec[\'name\'])}__{cid}.npy", pred.astype(np.float32))\n',
        '                if protocol == "kfold":\n'
        '                    case_data_path = PRED_DATA_DIR / f"{safe(cid)}.npz"\n'
        '                    if not case_data_path.exists():\n'
        '                        np.savez_compressed(case_data_path, coordinates=RAW_C[cid], truth=RAW_Y[cid])\n'
        '                    if spec["name"] in PRED_MODELS:\n'
        '                        np.savez_compressed(\n'
        '                            PRED_DIR / f"{safe(spec[\'name\'])}__{safe(cid)}.npz",\n'
        '                            prediction=pred.astype(np.float32),\n'
        '                        )\n'
        '                if protocol == "kfold" and cid == SHOWCASE.get(GROUP_OF[cid]):\n'
        '                    np.save(PRED_DIR / f"{safe(spec[\'name\'])}__{cid}.npy", pred.astype(np.float32))\n',
        "OOF prediction and case data export",
    )
    set_source(cells[runner_index], runner_source)

    registry_index = next(i for i, source in enumerate(sources) if "MODEL_SPECS = [" in source)
    registry_source = sources[registry_index]
    registry_source = replace_once(
        registry_source,
        '    dict(name="Domain-informed FiLM-MLP", fn=run_pointwise("Domain-informed FiLM-MLP", FiLMNet, kind="huber"), stage="core", geom=True),\n',
        '    dict(name="Domain-informed FiLM-MLP", fn=run_pointwise("Domain-informed FiLM-MLP", FiLMNet, kind="huber"), stage="core", geom=True),\n'
        '    dict(name="Domain-informed Concat-MLP", fn=run_pointwise("Domain-informed Concat-MLP", ConcatNet, kind="huber"), stage="core", geom=True),\n',
        "concat-conditioning ablation registry",
    )
    registry_source = replace_once(
        registry_source,
        '    dict(name="POD-RBF", fn=run_group_method("pod"), stage="core", geom=False, aligned=True),\n',
        '    dict(name="POD-RBF", fn=run_group_method("pod"), stage="core", geom=False, aligned=True),\n'
        '    dict(name="POD-GPR", fn=run_group_method("pod_gpr"), stage="core", geom=False, aligned=True),\n',
        "POD-GPR registry entry",
    )
    for line in [
        '    dict(name="Random Forest", fn=run_tabular("Random Forest"), stage="heavy", geom=False),\n',
        '    dict(name="Extra Trees", fn=run_tabular("Extra Trees"), stage="heavy", geom=False),\n',
        '    dict(name="CatBoost", fn=run_tabular("CatBoost"), stage="heavy", geom=False),\n',
        '    dict(name="Latent Neural Operator", fn=run_pointwise("Latent Neural Operator", LatentNeuralOperator), stage="heavy", geom=False),\n',
        '    dict(name="Transolver-style", fn=run_transolver, stage="heavy", geom=False),\n',
        '    dict(name="F-FNO", fn=run_fno("F-FNO", FFNO3D), stage="heavy", geom=False),\n',
        '    dict(name="U-FNO3D", fn=run_fno("U-FNO3D", UFNO3D), stage="heavy", geom=False),\n',
        '    dict(name="GAMS-GNN-style", fn=run_gams, stage="heavy", geom=False),\n',
    ]:
        registry_source = replace_once(
            registry_source,
            line,
            line.replace('geom=False', 'geom=True'),
            "group-holdout eligibility",
        )
    set_source(cells[registry_index], registry_source)

    intro = {
        "cell_type": "markdown",
        "metadata": {},
        "source": INTRO.splitlines(keepends=True),
        "id": "clean-benchmark-intro",
    }
    prediction_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": PREDICTION_CELL.splitlines(keepends=True),
        "id": "oof-case-map-export",
    }
    proposed_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": PROPOSED_MODEL_MARKDOWN.splitlines(keepends=True),
        "id": "proposed-loop-unrolled-model",
    }
    proposed_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": PROPOSED_MODEL_CODE.splitlines(keepends=True),
        "id": "proposed-loop-unrolled-implementation",
    }
    for cell in cells:
        cell["outputs"] = []
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
    notebook["cells"] = [intro, *cells, proposed_markdown, proposed_code, prediction_cell]
    notebook["metadata"].setdefault("kernelspec", {})
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
