from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULTS = ROOT / "results"
RUN_DIR = Path(
    os.environ.get(
        "SCWR_RUN_DIR", str(SOURCE_RESULTS / "clean_full_benchmark_2026-10-21_final")
    )
)
RUN_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path(
    os.environ.get("SCWR_DATA_PATH", str(ROOT / "mega_dataset_co2_candidate.parquet"))
)
SEED = 2026
N_FOLDS = 5
NODES_PER_CASE = 20_000
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
COORDS = ["x_m", "y_m", "z_m"]
NUMERIC = [
    "power_parameter_value",
    "pressure_bar",
    "mass_flow_kg_s",
    "temperature_parameter_value",
    "loop_x_extent_m",
    "loop_y_extent_m",
]
HEAVY = {TARGETS[3], TARGETS[4]}
MODELS = ["Mean baseline", "Ridge", "Random Forest", "Extra Trees", "XGBoost", "CatBoost"]


def target_transform_fit(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.ones(y.shape[1], dtype=np.float64)
    for j, target in enumerate(TARGETS):
        if target in HEAVY:
            scale[j] = max(float(np.median(np.abs(y[:, j]))), 1e-12)
    z = y.astype(np.float64).copy()
    for j, target in enumerate(TARGETS):
        if target in HEAVY:
            z[:, j] = np.arcsinh(z[:, j] / scale[j])
    mean = z.mean(axis=0)
    std = np.maximum(z.std(axis=0), 1e-12)
    return scale, mean, std


def transform_y(y: np.ndarray, scale: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = y.astype(np.float64).copy()
    for j, target in enumerate(TARGETS):
        if target in HEAVY:
            z[:, j] = np.arcsinh(z[:, j] / scale[j])
    return ((z - mean) / std).astype(np.float32)


def inverse_y(z: np.ndarray, scale: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    y = z.astype(np.float64) * std + mean
    for j, target in enumerate(TARGETS):
        if target in HEAVY:
            y[:, j] = scale[j] * np.sinh(np.clip(y[:, j], -30, 30))
    return y.astype(np.float32)


def estimator(name: str, protocol: str = "case_disjoint"):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=48,
            max_depth=20,
            max_leaf_nodes=4096,
            min_samples_leaf=5,
            max_features=0.8,
            bootstrap=True,
            max_samples=0.3,
            n_jobs=-1,
            random_state=SEED,
        )
    if name == "Extra Trees":
        return ExtraTreesRegressor(
            n_estimators=64,
            max_depth=20,
            max_leaf_nodes=4096,
            min_samples_leaf=5,
            max_features=0.8,
            bootstrap=True,
            max_samples=0.3,
            n_jobs=-1,
            random_state=SEED,
        )
    if name == "XGBoost":
        random_node = protocol == "random_node"
        return [
            XGBRegressor(
                n_estimators=100 if random_node else 300,
                max_depth=6 if random_node else 8,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                tree_method="hist",
                device="cpu",
                n_jobs=4 if random_node else 8,
                random_state=SEED,
            )
            for _ in TARGETS
        ]
    if name == "CatBoost":
        return CatBoostRegressor(
            iterations=350,
            depth=8,
            learning_rate=0.08,
            loss_function="MultiRMSE",
            verbose=False,
            random_seed=SEED,
            allow_writing_files=False,
            task_type="CPU",
            thread_count=8,
        )
    raise KeyError(name)


def fit_model(name: str, x: np.ndarray, y: np.ndarray, protocol: str = "case_disjoint"):
    if name == "Mean baseline":
        return y.mean(axis=0)
    model = estimator(name, protocol)
    if name == "XGBoost":
        for j, target_model in enumerate(model):
            target_model.fit(x, y[:, j])
        return model
    return model.fit(x, y)


def predict_model(name: str, model, x: np.ndarray) -> np.ndarray:
    if name == "Mean baseline":
        return np.repeat(model[None, :], len(x), axis=0).astype(np.float32)
    if name == "XGBoost":
        return np.column_stack([target_model.predict(x) for target_model in model]).astype(
            np.float32
        )
    return np.asarray(model.predict(x), dtype=np.float32)


def build_case_arrays(data: pd.DataFrame):
    case_table = data.groupby("case_id", sort=True).first().reset_index()
    case_table.loc[
        case_table["case_id"].str.endswith("_T_205"), "temperature_parameter_value"
    ] = 305.0
    config = pd.get_dummies(case_table["configuration"], dtype=np.float32)
    geometry = pd.get_dummies(case_table["geometry_class"], dtype=np.float32)
    flags = np.asarray(
        [
            [
                float(len(c) > 0 and c[0] == "V"),
                float(len(c) > 2 and c[2] == "V"),
            ]
            for c in case_table["configuration"]
        ],
        dtype=np.float32,
    )
    features = np.concatenate(
        [
            case_table[NUMERIC].to_numpy(np.float32),
            config.to_numpy(np.float32),
            geometry.to_numpy(np.float32),
            flags,
        ],
        axis=1,
    )
    arrays = {}
    for case_id, group in data.groupby("case_id", sort=True):
        coords = group[COORDS].to_numpy(np.float32)
        span = np.maximum(np.ptp(coords, axis=0), 1e-6)
        normalized = (coords - coords.min(axis=0)) / span
        arrays[case_id] = (
            normalized.astype(np.float32),
            group[TARGETS].to_numpy(np.float32),
            group["pressure_bar"].to_numpy(np.float32)[0],
        )
    case_table["feature_row"] = np.arange(len(case_table))
    return case_table, features, arrays


def case_features(
    case_table: pd.DataFrame,
    case_features_raw: np.ndarray,
    train_ids: list[str],
    ids: list[str],
) -> dict[str, np.ndarray]:
    train_rows = case_table.set_index("case_id").loc[train_ids, "feature_row"].to_numpy()
    mean = case_features_raw[train_rows].mean(axis=0)
    scale = case_features_raw[train_rows].std(axis=0)
    scale[scale < 1e-8] = 1.0
    table_index = case_table.set_index("case_id")["feature_row"]
    out = {}
    for case_id in ids:
        row = int(table_index.loc[case_id])
        condition = ((case_features_raw[row] - mean) / scale).astype(np.float32)
        xyz = ARRAYS[case_id][0]
        out[case_id] = np.column_stack(
            [np.repeat(condition[None, :], len(xyz), axis=0), xyz]
        ).astype(np.float32)
    return out


def metric_record(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    case_id: str,
    protocol: str,
    split: str,
    model_name: str,
):
    rows = []
    for j, target in enumerate(TARGETS):
        truth = y_true[:, j].astype(np.float64)
        pred = y_pred[:, j].astype(np.float64)
        error = pred - truth
        std = float(np.std(truth))
        sst = float(np.sum((truth - truth.mean()) ** 2))
        is_pressure = target == "pressure_pa"
        rows.append(
            {
                "protocol": protocol,
                "split": split,
                "model": model_name,
                "case_id": case_id,
                "target": target,
                "n_nodes": len(truth),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "nmae": np.nan if is_pressure else float(np.mean(np.abs(error)) / max(std, 1e-12)),
                "nrmse": np.nan if is_pressure else float(np.sqrt(np.mean(error**2)) / max(std, 1e-12)),
                "case_r2": np.nan if is_pressure else 1.0 - float(np.sum(error**2)) / max(sst, 1e-24),
                "sum_y": float(truth.sum()),
                "sum_y2": float(np.dot(truth, truth)),
                "sum_sq_err": float(np.dot(error, error)),
                "case_mean_y": float(truth.mean()),
                "case_mean_sq": float(sst),
                "pressure_bar": float(ARRAYS[case_id][2]),
            }
        )
    return rows


def pressure_metric_record(
    truth: np.ndarray,
    prediction: np.ndarray,
    case_id: str,
    protocol: str,
    split: str,
    model_name: str,
    floor_pa: float = 1000.0,
) -> dict:
    pressure_index = TARGETS.index("pressure_pa")
    true_pressure = truth[:, pressure_index].astype(np.float64)
    predicted_pressure = prediction[:, pressure_index].astype(np.float64)
    true_level = float(true_pressure.mean())
    predicted_level = float(predicted_pressure.mean())
    true_residual = true_pressure - true_level
    predicted_residual = predicted_pressure - predicted_pressure.mean()
    residual_error = predicted_residual - true_residual
    residual_rmse = float(np.sqrt(np.mean(residual_error**2)))
    residual_std = float(np.std(true_residual))
    return {
        "protocol": protocol,
        "split": split,
        "model": model_name,
        "case_id": case_id,
        "n_nodes": len(true_pressure),
        "pressure_level_true_pa": true_level,
        "pressure_level_pred_pa": predicted_level,
        "pressure_level_abs_error_pa": abs(predicted_level - true_level),
        "pressure_residual_mae_pa": float(np.mean(np.abs(residual_error))),
        "pressure_residual_rmse_pa": residual_rmse,
        "pressure_residual_std_pa": residual_std,
        "pressure_residual_nrmse_floor_1kpa": residual_rmse
        / max(residual_std, floor_pa),
        "pressure_residual_floor_pa": floor_pa,
    }


def fit_fold(
    protocol: str,
    split_id: str,
    train_ids: list[str],
    test_data: dict[str, np.ndarray],
    case_table: pd.DataFrame,
    features: np.ndarray,
    arrays: dict,
    metrics: list,
    pressure_metrics: list,
    status_rows: list,
    train_index_by_case: dict[str, np.ndarray] | None = None,
):
    global ARRAYS
    ARRAYS = arrays
    rng = np.random.default_rng(SEED + sum(map(ord, split_id)) + (1000 if protocol == "random_node" else 0))
    train_x, train_y = [], []
    for case_id in train_ids:
        xyz, y, _ = arrays[case_id]
        idx = (
            train_index_by_case[case_id]
            if train_index_by_case is not None
            else np.arange(len(y), dtype=np.int64)
        )
        if len(idx) == 0:
            continue
        max_nodes = min(NODES_PER_CASE, len(idx))
        if protocol == "random_node":
            max_nodes = min(5_000, max_nodes)
        take = rng.choice(idx, max_nodes, replace=False)
        row = int(case_table.set_index("case_id").loc[case_id, "feature_row"])
        train_x.append((take, row))
        train_y.append(y[take])
    ids_to_encode = train_ids + list(test_data)
    cond_x = case_features(case_table, features, train_ids, ids_to_encode)
    x_train_parts = []
    y_train = np.concatenate(train_y)
    for (idx, row), case_id in zip(train_x, [c for c in train_ids if len(
        train_index_by_case[c] if train_index_by_case is not None else arrays[c][1]
    )]):
        x_train_parts.append(cond_x[case_id][idx])
    x_train = np.concatenate(x_train_parts).astype(np.float32)
    y_train = y_train.astype(np.float32)
    target_scale, target_mean, target_std = target_transform_fit(y_train)
    y_scaled = transform_y(y_train, target_scale, target_mean, target_std)

    model_names = MODELS
    for name in model_names:
        start = time.monotonic()
        model_x = x_train
        model_y = y_scaled if name != "Mean baseline" else y_train
        if name == "XGBoost" and protocol == "random_node":
            rng_xgb = np.random.default_rng(SEED + 4099)
            take = rng_xgb.choice(len(model_x), min(len(model_x), 250_000), replace=False)
            model_x = model_x[take]
            model_y = model_y[take]
        model = fit_model(
            name,
            model_x,
            model_y,
            "random_node" if protocol == "random_node" else "case_disjoint",
        )
        elapsed = time.monotonic() - start
        if name == "Mean baseline":
            mean_y = y_train.mean(axis=0)
        for case_id, (test_idx, _) in test_data.items():
            if not len(test_idx):
                continue
            pred_parts = []
            for start_idx in range(0, len(test_idx), 100_000):
                idx = test_idx[start_idx : start_idx + 100_000]
                if name == "Mean baseline":
                    pred = np.repeat(mean_y[None, :], len(idx), axis=0)
                else:
                    pred_scaled = predict_model(name, model, cond_x[case_id][idx])
                    pred = inverse_y(pred_scaled, target_scale, target_mean, target_std)
                pred_parts.append(pred)
            pred_all = np.concatenate(pred_parts)
            truth = arrays[case_id][1][test_idx]
            metrics.extend(
                metric_record(truth, pred_all, case_id, protocol, split_id, name)
            )
            pressure_metrics.append(
                pressure_metric_record(
                    truth, pred_all, case_id, protocol, split_id, name
                )
            )
        status_rows.append(
            {
                "protocol": protocol,
                "split": split_id,
                "model": name,
                "seconds": round(elapsed, 2),
                "train_nodes": len(y_train),
                "model_train_nodes": len(model_x),
                "test_cases": sum(bool(len(idx)) for idx, _ in test_data.values()),
                "settings_note": (
                    "100 trees, depth 6, 4 threads; 250,000-node cap"
                    if name == "XGBoost" and protocol == "random_node"
                    else "350 CatBoost iterations"
                    if name == "CatBoost"
                    else ""
                ),
            }
        )
        print(
            f"{protocol} {split_id} {name}: {elapsed:.1f}s, "
            f"{len(y_train):,} training nodes",
            flush=True,
        )
        del model
        gc.collect()


def add_case_transfer_metrics(
    fold_assignments: pd.DataFrame,
    case_table: pd.DataFrame,
    features: np.ndarray,
    arrays: dict,
    metrics: list,
    pressure_metrics: list,
):
    global ARRAYS
    ARRAYS = arrays
    condition_names = [
        "power_parameter_value",
        "pressure_bar",
        "mass_flow_kg_s",
        "temperature_parameter_value",
    ]
    case_features_df = case_table.set_index("case_id")
    for split_id in sorted(fold_assignments["split"].unique()):
        fold = fold_assignments.loc[fold_assignments["split"] == split_id]
        train_ids = sorted(fold.loc[fold["role"] == "train", "case_id"])
        test_ids = sorted(fold.loc[fold["role"] == "test", "case_id"])
        groups = sorted(
            set(case_table["configuration"].astype(str) + "|" + case_table["geometry_class"].astype(str))
        )
        for group in groups:
            group_train = [
                cid
                for cid in train_ids
                if f"{case_features_df.loc[cid, 'configuration']}|{case_features_df.loc[cid, 'geometry_class']}"
                == group
            ]
            group_test = [
                cid
                for cid in test_ids
                if f"{case_features_df.loc[cid, 'configuration']}|{case_features_df.loc[cid, 'geometry_class']}"
                == group
            ]
            if not group_test or not group_train:
                continue
            raw_inputs = case_features_df[condition_names].copy()
            for cid in group_train:
                if cid.endswith("_T_205"):
                    raw_inputs.loc[cid, "temperature_parameter_value"] = 305.0
            tr = raw_inputs.loc[group_train].to_numpy(np.float64)
            mu, sd = tr.mean(axis=0), tr.std(axis=0)
            sd[sd < 1e-12] = 1.0
            xtr = (tr - mu) / sd
            xte = (raw_inputs.loc[group_test].to_numpy(np.float64) - mu) / sd
            fields = np.stack([arrays[cid][1] for cid in group_train]).astype(np.float32)
            pod_models = []
            aligned_fields = fields.astype(np.float64)
            for target_index in range(len(TARGETS)):
                target_fields = aligned_fields[:, :, target_index]
                center = target_fields.mean(axis=0)
                value_scale = max(float(target_fields.std()), 1e-12)
                normalized = (target_fields - center[None, :]) / value_scale
                u, singular, vh = np.linalg.svd(normalized, full_matrices=False)
                rank = min(12, max(1, len(group_train) - 1), len(singular))
                basis = vh[:rank].T
                scores = u[:, :rank] * singular[None, :rank]
                pod_models.append(
                    (
                        center,
                        value_scale,
                        basis,
                        RBFInterpolator(
                            xtr,
                            scores,
                            kernel="gaussian",
                            epsilon=1.0,
                            smoothing=1e-3,
                        ),
                    )
                )
            pod_gpr_models = []
            for target_index in range(len(TARGETS)):
                target_fields = aligned_fields[:, :, target_index]
                center = target_fields.mean(axis=0)
                value_scale = max(float(target_fields.std()), 1e-12)
                normalized = (target_fields - center[None, :]) / value_scale
                u, singular, vh = np.linalg.svd(normalized, full_matrices=False)
                rank = min(12, max(1, len(group_train) - 1), len(singular))
                basis = vh[:rank].T
                scores = u[:, :rank] * singular[None, :rank]
                kernel = (
                    ConstantKernel(1.0, constant_value_bounds="fixed")
                    * RBF(1.0, length_scale_bounds="fixed")
                    + WhiteKernel(1e-4, noise_level_bounds="fixed")
                )
                gpr = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    optimizer=None,
                    normalize_y=True,
                ).fit(xtr, scores)
                pod_gpr_models.append((center, value_scale, basis, gpr))
            for query_id, x in zip(group_test, xte):
                distances = np.linalg.norm(xtr - x[None, :], axis=1)
                nearest = int(np.argmin(distances))
                ordered = np.argsort(distances)[: min(3, len(distances))]
                if distances[ordered[0]] < 1e-12:
                    weights = np.zeros(len(ordered))
                    weights[0] = 1.0
                else:
                    inverse = 1.0 / np.maximum(distances[ordered], 1e-12)
                    weights = inverse / inverse.sum()
                predictions = {
                    "Nearest case": fields[nearest],
                    "Parameter 3-NN inverse-distance": np.tensordot(
                        weights, fields[ordered], axes=(0, 0)
                    ),
                }
                pod_prediction = np.empty_like(arrays[query_id][1], dtype=np.float32)
                pod_gpr_prediction = np.empty_like(pod_prediction)
                for target_index, (center, value_scale, basis, rbf) in enumerate(pod_models):
                    score = np.asarray(rbf(x[None, :])).reshape(-1)
                    pod_prediction[:, target_index] = (
                        center + value_scale * (basis @ score)
                    ).astype(np.float32)
                for target_index, (center, value_scale, basis, gpr) in enumerate(
                    pod_gpr_models
                ):
                    score = np.asarray(gpr.predict(x[None, :])).reshape(-1)
                    pod_gpr_prediction[:, target_index] = (
                        center + value_scale * (basis @ score)
                    ).astype(np.float32)
                predictions["POD-RBF (fixed Gaussian)"] = pod_prediction
                predictions["POD-GPR (fixed kernel)"] = pod_gpr_prediction
                for model_name, prediction in predictions.items():
                    metrics.extend(
                        metric_record(
                            arrays[query_id][1],
                            prediction,
                            query_id,
                            "kfold",
                            split_id,
                            model_name,
                        )
                    )
                    pressure_metrics.append(
                        pressure_metric_record(
                            arrays[query_id][1],
                            prediction,
                            query_id,
                            "kfold",
                            split_id,
                            model_name,
                        )
                    )


def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    output = []
    for (protocol, model, target), group in metrics_df.groupby(
        ["protocol", "model", "target"], sort=True
    ):
        if target == "pressure_pa":
            continue
        case_mae = group["mae"].to_numpy()
        case_nmae = group["nmae"].to_numpy()
        case_r2 = group["case_r2"].to_numpy()
        case_nrmse = group["nrmse"].to_numpy()
        boot_median_nmae = np.empty(2000)
        boot_mean_mae = np.empty(2000)
        for b in range(2000):
            idx = rng.integers(0, len(group), len(group))
            boot_median_nmae[b] = np.median(case_nmae[idx])
            boot_mean_mae[b] = case_mae[idx].mean()
        n = int(group["n_nodes"].sum())
        sum_y = float(group["sum_y"].sum())
        sum_y2 = float(group["sum_y2"].sum())
        sse = float(group["sum_sq_err"].sum())
        gauge_sst = sum_y2 - sum_y**2 / n
        pooled_gauge = 1.0 - sse / max(gauge_sst, 1e-24)
        centered_sst = float(group["case_mean_sq"].sum())
        pooled_centered = 1.0 - sse / max(centered_sst, 1e-24)
        pooled_absolute = np.nan
        if target == "pressure_pa":
            # Add the per-case operating pressure to both truth and prediction.
            means = group["case_mean_y"].to_numpy()
            offsets = group["pressure_bar"].to_numpy() * 1e5
            nodes = group["n_nodes"].to_numpy()
            s_abs = sum_y2 + float(np.sum(2 * means * offsets * nodes + offsets**2 * nodes))
            sum_abs = sum_y + float(np.sum(offsets * nodes))
            pooled_absolute = 1.0 - sse / max(s_abs - sum_abs**2 / n, 1e-24)
        output.append(
            {
                "protocol": protocol,
                "model": model,
                "target": target,
                "n_cases": len(group),
                "case_macro_mae": case_mae.mean(),
                "bootstrap_mean_mae_lo": np.quantile(boot_mean_mae, 0.025),
                "bootstrap_mean_mae_hi": np.quantile(boot_mean_mae, 0.975),
                "median_nmae": np.median(case_nmae),
                "q25_nmae": np.quantile(case_nmae, 0.25),
                "q75_nmae": np.quantile(case_nmae, 0.75),
                "bootstrap_median_nmae_lo": np.quantile(boot_median_nmae, 0.025),
                "bootstrap_median_nmae_hi": np.quantile(boot_median_nmae, 0.975),
                "median_case_r2": np.median(case_r2),
                "q25_case_r2": np.quantile(case_r2, 0.25),
                "q75_case_r2": np.quantile(case_r2, 0.75),
                "median_nrmse": np.median(case_nrmse),
                "q25_nrmse": np.quantile(case_nrmse, 0.25),
                "q75_nrmse": np.quantile(case_nrmse, 0.75),
                "pooled_r2_gauge": pooled_gauge,
                "pooled_r2_case_centered": pooled_centered,
                "pooled_r2_absolute_pressure": pooled_absolute,
            }
        )
    return pd.DataFrame(output)


def summarize_pressure_metrics(pressure_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    output = []
    for (protocol, model), group in pressure_df.groupby(
        ["protocol", "model"], sort=True
    ):
        level_mae = group["pressure_level_abs_error_pa"].to_numpy(float)
        level_error = (
            group["pressure_level_pred_pa"].to_numpy(float)
            - group["pressure_level_true_pa"].to_numpy(float)
        )
        residual_mae = group["pressure_residual_mae_pa"].to_numpy(float)
        residual_rmse = group["pressure_residual_rmse_pa"].to_numpy(float)
        residual_nrmse = group["pressure_residual_nrmse_floor_1kpa"].to_numpy(float)
        boots = np.empty((2000, 5), dtype=np.float64)
        for b in range(len(boots)):
            idx = rng.integers(0, len(group), len(group))
            boots[b] = [
                level_mae[idx].mean(),
                np.sqrt(np.mean(level_error[idx] ** 2)),
                residual_mae[idx].mean(),
                np.sqrt(np.mean(residual_rmse[idx] ** 2)),
                np.median(residual_nrmse[idx]),
            ]
        row = {
            "protocol": protocol,
            "model": model,
            "n_cases": group["case_id"].nunique(),
            "case_level_mean_pressure_mae_pa": level_mae.mean(),
            "case_level_mean_pressure_rmse_pa": np.sqrt(np.mean(level_error**2)),
            "centered_residual_mae_pa": residual_mae.mean(),
            "centered_residual_rmse_pa": np.sqrt(np.mean(residual_rmse**2)),
            "centered_residual_nrmse_median": np.median(residual_nrmse),
        }
        names = [
            "case_level_mean_pressure_mae_pa",
            "case_level_mean_pressure_rmse_pa",
            "centered_residual_mae_pa",
            "centered_residual_rmse_pa",
            "centered_residual_nrmse_median",
        ]
        for col, name in enumerate(names):
            row[f"{name}_lo"] = np.quantile(boots[:, col], 0.025)
            row[f"{name}_hi"] = np.quantile(boots[:, col], 0.975)
        output.append(row)
    return pd.DataFrame(output)


def paired_case_tests(
    metrics_df: pd.DataFrame, pressure_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    field_targets = [target for target in TARGETS if target != "pressure_pa"]
    test_rows, pair_rows = [], []
    for target in [*field_targets, "pressure_pa"]:
        if target == "pressure_pa":
            frame = pressure_df.loc[
                pressure_df["protocol"].isin(["case_disjoint", "kfold"])
            ]
            value = "pressure_residual_nrmse_floor_1kpa"
        else:
            frame = metrics_df.loc[
                (metrics_df["protocol"].isin(["case_disjoint", "kfold"]))
                & (metrics_df["target"] == target)
            ]
            value = "nmae"
        matrix = frame.pivot_table(
            index="case_id", columns="model", values=value, aggfunc="first"
        ).dropna(axis=0, how="any").dropna(axis=1, how="any")
        if matrix.shape[0] < 3 or matrix.shape[1] < 2:
            continue
        values = matrix.to_numpy(float)
        ranks = np.apply_along_axis(rankdata, 1, values)
        mean_ranks = ranks.mean(axis=0)
        friedman = friedmanchisquare(*[values[:, i] for i in range(values.shape[1])])
        statistic = float(friedman.statistic)
        p_value = float(friedman.pvalue)
        reference_index = int(np.argmin(mean_ranks))
        reference = str(matrix.columns[reference_index])
        for model, mean_rank in zip(matrix.columns, mean_ranks):
            test_rows.append(
                {
                    "target": target,
                    "n_cases": len(matrix),
                    "model": model,
                    "mean_rank": mean_rank,
                    "friedman_statistic": statistic,
                    "friedman_p_value": p_value,
                    "metric": value,
                }
            )
        comparisons = []
        for index, model in enumerate(matrix.columns):
            if index == reference_index:
                continue
            left, right = values[:, reference_index], values[:, index]
            if np.allclose(left, right):
                p_pair = 1.0
            else:
                p_pair = float(wilcoxon(left, right, alternative="two-sided").pvalue)
            comparisons.append((str(model), p_pair))
        order = np.argsort([item[1] for item in comparisons])
        adjusted = np.empty(len(comparisons), dtype=float)
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, min(1.0, (len(order) - rank) * comparisons[index][1]))
            adjusted[index] = running
        for (model, p_pair), p_holm in zip(comparisons, adjusted):
            pair_rows.append(
                {
                    "target": target,
                    "n_cases": len(matrix),
                    "reference_lowest_mean_rank": reference,
                    "comparator": model,
                    "wilcoxon_p": p_pair,
                    "holm_adjusted_p": p_holm,
                    "metric": value,
                }
            )
    return pd.DataFrame(test_rows), pd.DataFrame(pair_rows)


def main():
    global ARRAYS
    data = pd.read_parquet(DATA_PATH)
    case_table, features, arrays = build_case_arrays(data)
    ARRAYS = arrays
    assignments = pd.read_csv(SOURCE_RESULTS / "split_assignments.csv")
    assignments = assignments.loc[assignments["protocol"] == "kfold"].copy()
    metrics, pressure_metrics, statuses = [], [], []

    def save_progress():
        pd.DataFrame(metrics).to_csv(
            RUN_DIR / "case_and_random_node_metrics_progress.csv", index=False
        )
        pd.DataFrame(statuses).to_csv(
            RUN_DIR / "run_status_progress.csv", index=False
        )
        pd.DataFrame(pressure_metrics).to_csv(
            RUN_DIR / "pressure_metrics_progress.csv", index=False
        )

    for split_id in sorted(assignments["split"].unique()):
        fold = assignments.loc[assignments["split"] == split_id]
        train_ids = sorted(fold.loc[fold["role"] == "train", "case_id"])
        test_ids = sorted(fold.loc[fold["role"] == "test", "case_id"])
        test_data = {
            cid: (np.arange(len(arrays[cid][1]), dtype=np.int64), None) for cid in test_ids
        }
        fit_fold(
            "case_disjoint",
            split_id,
            train_ids,
            test_data,
            case_table,
            features,
            arrays,
            metrics,
            pressure_metrics,
            statuses,
        )
        save_progress()

    rng = np.random.default_rng(SEED)
    train_indices, test_indices = {}, {}
    for case_id, (_, y, _) in arrays.items():
        indices = rng.permutation(len(y))
        n_test = max(1, int(round(0.2 * len(indices))))
        test_indices[case_id] = np.sort(indices[:n_test])
        train_indices[case_id] = np.sort(indices[n_test:])
    random_data = {
        cid: (test_indices[cid], None) for cid in sorted(arrays)
    }
    fit_fold(
        "random_node",
        "80_20",
        sorted(arrays),
        random_data,
        case_table,
        features,
        arrays,
        metrics,
        pressure_metrics,
        statuses,
        train_index_by_case=train_indices,
    )
    save_progress()

    add_case_transfer_metrics(
        assignments, case_table, features, arrays, metrics, pressure_metrics
    )
    case_metrics = pd.DataFrame(metrics)
    case_metrics.to_csv(RUN_DIR / "case_and_random_node_metrics.csv", index=False)
    summary = summarize(case_metrics)
    summary.to_csv(RUN_DIR / "cpu_benchmark_summary.csv", index=False)
    pressure_frame = pd.DataFrame(pressure_metrics)
    pressure_frame.to_csv(
        RUN_DIR / "pressure_level_and_residual_metrics.csv", index=False
    )
    summarize_pressure_metrics(pressure_frame).to_csv(
        RUN_DIR / "pressure_summary.csv", index=False
    )
    friedman, wilcoxon_holm = paired_case_tests(case_metrics, pressure_frame)
    friedman.to_csv(RUN_DIR / "friedman_tests.csv", index=False)
    wilcoxon_holm.to_csv(RUN_DIR / "wilcoxon_holm_tests.csv", index=False)
    pd.DataFrame(statuses).to_csv(RUN_DIR / "run_status.csv", index=False)
    print("Saved case metrics and summaries.")
    print(
        summary.loc[
            summary["protocol"].isin(["case_disjoint", "random_node"]),
            [
                "protocol",
                "model",
                "target",
                "case_macro_mae",
                "median_nmae",
                "median_case_r2",
                "pooled_r2_gauge",
            ],
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
