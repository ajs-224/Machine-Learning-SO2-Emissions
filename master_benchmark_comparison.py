# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python (sheat_mip ML)
#     language: python
#     name: sheat_mip_ml
# ---

# %% [markdown]
# # Master Cross-Benchmark Evaluation & Comparative Diagnostics Pipeline
# ### Evaluates and compares point-source $\text{SO}_2$ inversion performance across:
# ### - 5 Sensor Tiers:
# ###   1. `ideal`: Pure physical forward dispersion benchmark ($\sigma = 0.00\text{ DU}$)
# ###   2. `next_gen`: Next-Gen geostationary sensor ($\sigma = 0.05\text{ DU}$, high SNR)
# ###   3. `moderate`: Moderate Sentinel-4 / TROPOMI class sensor ($\sigma = 0.15\text{ DU}$)
# ###   4. `gems_realistic`: Operational GEMS synthetic OSSE baseline ($\sigma = 0.35\text{ DU}$, 68% QA dropout)
# ###   5. `log_scaled`: Real operational GEMS Level-2 satellite observations
# ### - 2 Architectural Methodologies:
# ###   A. `mass_sum`: 3-Input Physics-Anchored Regressor (Spatial 7ch + Met 25 + Integrated Mass Anchor)
# ###   B. `endtoend`: 2-Input Direct Multi-Modal Regressor (Spatial 7ch + Met 25)
# ### - Comprehensive Diagnostic Suite:
# ###   * Master $2 \times 5$ sequence-level density parity matrix (Linear & Log-Log toggle).
# ###   * Standalone $2 \times 2$ intermediate emitter matrix (excluding G01 super-cluster).
# ###   * Sensor noise degradation curves with real operational data reference anchor.
# ###   * Emitter scale error stratification (Faint $<20\text{ t}$, Medium $20\text{--}80\text{ t}$, Super-Emitters $>80\text{ t}$).
# ###   * Master $2 \times 5$ multi-year cluster-level mean emissions parity matrix.
# ###   * Peninsula-wide cumulative mass conservation & systematic regional budget bias.
# ###   * Spatial disaggregation error heatmap ($13\text{ Clusters} \times 10\text{ Models}$).
# ###   * Executive summary tabular performance report.

# %%
import os
import sys
import glob
import re
import json
import time
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Configure structured logging and base styling
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("MasterBenchmark")

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#222222"
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["grid.color"] = "#e0e0e0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"

# =============================================================================
# 1. DIRECTORY PATHS, CONSTANTS & BENCHMARK REGISTRIES
# =============================================================================
WORKFLOW_DIR: str = "/home/users/ajs224/sheat_mip_gws/code/WindRotatedMLWorkflow"
MODEL_BASE_DIR: str = os.path.join(WORKFLOW_DIR, "Output_Model")
FIGURES_DIR: str = os.path.join(WORKFLOW_DIR, "Figures_Comparison")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Dataset directory mappings across all mission tiers
DATASET_DIRECTORIES: Dict[str, str] = {
    "ideal": os.path.join(WORKFLOW_DIR, "Output_Tensor_Synthetic_Ideal/"),
    "next_gen": os.path.join(WORKFLOW_DIR, "Output_Tensor_Synthetic_NextGen/"),
    "moderate": os.path.join(WORKFLOW_DIR, "Output_Tensor_Synthetic_Moderate/"),
    "gems_realistic": os.path.join(WORKFLOW_DIR, "Output_Tensor_Synthetic_GEMSRealistic/"),
    "log_scaled": os.path.join(WORKFLOW_DIR, "Output_Tensor_DualBranch_LogScaled/")
}

# Physical Conversion Constants
DU_CONVERSION_FACTOR: float = 2.6867e16  # molecules/cm^2 per Dobson Unit (DU)
KG_PER_DU_PER_M2: float = 2.857e-5      # 1 DU = 2.857e-5 kg SO2 / m^2

# Target Benchmark Configurations (5 Tiers x 2 Architectures = 10 Models)
BENCHMARK_TIERS: List[Dict[str, Any]] = [
    {"key": "ideal",          "name": "Ideal",            "sigma": 0.00, "is_real": False},
    {"key": "next_gen",       "name": "Next-Gen",          "sigma": 0.05, "is_real": False},
    {"key": "moderate",       "name": "Moderate",          "sigma": 0.15, "is_real": False},
    {"key": "gems_realistic", "name": "GEMS Realistic",    "sigma": 0.35, "is_real": False},
    {"key": "log_scaled",     "name": "Real GEMS Level-2", "sigma": np.nan, "is_real": True},
]

ARCHITECTURES: List[Dict[str, str]] = [
    {"key": "mass_sum", "label": "Physics-Anchored", "color": "#1f77b4"},
    {"key": "endtoend", "label": "Direct End-to-End ML",        "color": "#ff7f0e"}
]

# Continuous Density Colormap matching publication style (Cyan -> Yellow -> Red -> Maroon)
DENSITY_COLORMAP = LinearSegmentedColormap.from_list(
    "density_ch4_style",
    [
        (0.00, "#9ee5ff"),  # Light cyan (low density background)
        (0.20, "#5fd3f3"),  # Sky blue
        (0.40, "#4cd964"),  # Greenish-yellow transition
        (0.55, "#ffe600"),  # Vibrant yellow
        (0.70, "#ff8800"),  # Orange
        (0.85, "#e62200"),  # Bright red
        (1.00, "#730000"),  # Dark maroon (peak density)
    ],
    N=256
)

# Industrial Sector Color Mapping
SECTOR_COLOR_MAP: Dict[str, str] = {
    "Steel, Refining & Petrochem": "#e74c3c",
    "Refining & Smelting": "#e67e22",
    "Steel & Heavy Alloys": "#c0392b",
    "Coal-Fired Power": "#2c3e50",
    "Coastal Power": "#34495e",
    "Urban Steel & Heat": "#8e44ad",
    "Inland CHP": "#16a085",
    "Mountain Smelter": "#d35400",
    "Glass & Ceramics": "#27ae60",
    "Power Generation": "#2980b9",
    "Dyeing & Energy": "#7f8c8d",
    "General Industry": "#95a5a6"
}

# Master 13 Active Regional Clusters Registry
ACTIVE_CLUSTERS: List[str] = [
    "G01", "G02", "G06", "G07", "G08", "G09", "G10", "G12", "G13", "G14", "G15", "G16", "G17"
]

# Default 3-Fold Registry (Synchronized dynamically if JSON config is detected)
DEFAULT_SPATIAL_FOLDS: Dict[int, List[str]] = {
    1: ["G01", "G09", "G13", "G14", "G16"],
    2: ["G02", "G07", "G10", "G17"],
    3: ["G06", "G08", "G12", "G15"]
}

# =============================================================================
# 2. MULTI-MODAL DATA SEQUENCE & CHECKPOINT DISCOVERY
# =============================================================================
def parse_tensor_metadata(file_path: str) -> Tuple[str, pd.Timestamp, pd.Timestamp, int]:
    """Extracts Cluster ID, Start Date, End Date, and Window Index from NetCDF filename."""
    filename = os.path.basename(file_path)
    match = re.search(r"tensor_([A-Z0-9]+)_(\d{8})_(\d{8})_win(\d+)", filename)
    if match:
        gid = match.group(1)
        start_d = pd.to_datetime(match.group(2), format="%Y%m%d")
        end_d = pd.to_datetime(match.group(3), format="%Y%m%d")
        win_idx = int(match.group(4))
        return gid, start_d, end_d, win_idx
    return "UNKNOWN", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-04"), 0


class DualBranchEvaluationSequence(tf.keras.utils.Sequence):
    """
    Standardized In-Memory Multi-Input Sequence Generator for Cross-Model Benchmarking.
    Yields strictly 7 physical spatial channels (Shape: [25, 25, 7]) + 25 met features + 1 scalar log mass anchor.
    """

    def __init__(
        self,
        file_list: List[str],
        batch_size: int = 32,
    ) -> None:
        self.file_paths = file_list
        self.batch_size = batch_size

        self.cached_spatial: List[np.ndarray] = []
        self.cached_met: List[np.ndarray] = []
        self.cached_log_m: List[np.ndarray] = []
        self.cached_target: List[float] = []
        self.cached_meta: List[Dict[str, Any]] = []

        for fpath in self.file_paths:
            try:
                with xr.open_dataset(fpath) as ds:
                    raw_spatial = ds["spatial_plume_tensor"].values.astype(np.float32)

                    # Extract raw 4 daily snapshots (Channels 0..3)
                    if raw_spatial.shape[0] >= 4 and raw_spatial.shape[0] <= 7:
                        raw_4d = np.transpose(raw_spatial[:4, :, :], (1, 2, 0))
                    elif raw_spatial.shape[-1] >= 4:
                        raw_4d = raw_spatial[:, :, :4]
                    else:
                        raise ValueError(f"Unexpected spatial shape: {raw_spatial.shape}")

                    met_vec = ds["met_vector"].values.astype(np.float32)  # Shape: (25,)

                    # Extract CleanSYS physical ground truth target
                    if "target_physical_kg" in ds:
                        true_kg = float(ds["target_physical_kg"].values)
                        true_tonnes = true_kg / 1000.0
                        target_y = float(np.log10(true_tonnes + 0.1))
                    elif "target_physical_tonnes" in ds:
                        true_tonnes = float(ds["target_physical_tonnes"].values)
                        true_kg = true_tonnes * 1000.0
                        target_y = float(np.log10(true_tonnes + 0.1))
                    elif "target_log_emissions" in ds:
                        target_y = float(ds["target_log_emissions"].values)
                        true_tonnes = max(0.0, float(10.0 ** target_y - 0.1))
                        true_kg = true_tonnes * 1000.0
                    else:
                        true_tonnes = 10.0
                        true_kg = 10000.0
                        target_y = float(np.log10(true_tonnes + 0.1))

                    # Deterministic 15% baseline floor subtraction & delta computation
                    net_snapshots_list: List[np.ndarray] = []
                    for d in range(4):
                        snap_d = raw_4d[:, :, d]
                        floor_15_d = float(np.percentile(snap_d, 15))
                        net_snap_d = np.maximum(0.0, snap_d - floor_15_d)
                        net_snapshots_list.append(net_snap_d)

                    delta_1to2 = net_snapshots_list[1] - net_snapshots_list[0]
                    delta_2to3 = net_snapshots_list[2] - net_snapshots_list[1]
                    delta_3to4 = net_snapshots_list[3] - net_snapshots_list[2]

                    spatial_7ch = np.stack(
                        net_snapshots_list + [delta_1to2, delta_2to3, delta_3to4],
                        axis=-1
                    ).astype(np.float32)  # Shape: (25, 25, 7)

                    # Compute grid cell area in m^2
                    lats = ds["latitude"].values
                    lons = ds["longitude"].values
                    c_lat = float(np.mean(lats))
                    dx_km = (lons[1] - lons[0]) * 111.32 * np.cos(np.radians(c_lat))
                    dy_km = (lats[1] - lats[0]) * 110.57
                    cell_m2 = float(abs(dx_km * 1000.0 * dy_km * 1000.0))

                    # Physical Mass Integration
                    mean_net_vcd = np.mean(np.stack(net_snapshots_list, axis=0), axis=0)
                    m_int_kg = float(np.sum(mean_net_vcd) * cell_m2 * KG_PER_DU_PER_M2)
                    m_int_tonnes = max(0.001, m_int_kg / 1000.0)
                    log_m_arr = np.array([np.log10(m_int_tonnes + 0.1)], dtype=np.float32)

                    self.cached_spatial.append(spatial_7ch)
                    self.cached_met.append(met_vec)
                    self.cached_log_m.append(log_m_arr)
                    self.cached_target.append(target_y)
                    self.cached_meta.append({
                        "filepath": fpath,
                        "filename": os.path.basename(fpath),
                        "m_int_tonnes": m_int_tonnes,
                        "target_physical_kg": true_kg,
                        "target_physical_tonnes": true_tonnes,
                        "cluster_id": ds.attrs.get("cluster_id", parse_tensor_metadata(fpath)[0]),
                        "cluster_name": ds.attrs.get("cluster_name", "Industrial Cluster"),
                        "primary_sector": ds.attrs.get("primary_sector", "General Industry"),
                        "sequence_start_date": ds.attrs.get("sequence_start_date", "N/A"),
                        "sequence_end_date": ds.attrs.get("sequence_end_date", "N/A"),
                    })
            except Exception as e_load:
                logger.error("Failed loading tensor %s: %s", fpath, e_load)
                raise

        self.indices = np.arange(len(self.file_paths))

    def __len__(self) -> int:
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        batch_idx_slice = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        batch_x_spatial = []
        batch_x_met = []
        batch_x_log_m = []
        batch_y = []

        for b_idx in batch_idx_slice:
            batch_x_spatial.append(self.cached_spatial[b_idx])
            batch_x_met.append(self.cached_met[b_idx])
            batch_x_log_m.append(self.cached_log_m[b_idx])
            batch_y.append(self.cached_target[b_idx])

        return (
            (
                np.array(batch_x_spatial, dtype=np.float32),
                np.array(batch_x_met, dtype=np.float32),
                np.array(batch_x_log_m, dtype=np.float32),
            ),
            np.array(batch_y, dtype=np.float32)
        )


def discover_model_checkpoint(
    base_dir: str,
    tier_key: str,
    arch_key: str,
    fold_num: int,
    total_folds: int = 3
) -> Optional[str]:
    """
    Locates trained model checkpoints supporting nested subfolders and flat directories.
    Matches: Output_Model/{tier}_{arch}/...fold{N}.keras, Output_Model/{tier}_{arch}.keras, etc.
    """
    search_patterns = [
        # Subdirectory patterns: e.g., Output_Model/gems_realistic_mass_sum/best_plume_regressor_*fold1.keras
        os.path.join(base_dir, f"{tier_key}_{arch_key}", f"*fold{fold_num}*.keras"),
        os.path.join(base_dir, f"{tier_key}_{arch_key}", f"*.keras"),
        # Direct parent directory patterns:
        os.path.join(base_dir, f"best_plume_regressor_dualbranch_{tier_key}_{arch_key}_{total_folds}folds_fold{fold_num}.keras"),
        os.path.join(base_dir, f"best_plume_regressor_dualbranch_{tier_key}_{total_folds}folds_fold{fold_num}.keras"),
        os.path.join(base_dir, f"best_plume_regressor_*{tier_key}*{arch_key}*fold{fold_num}*.keras"),
        os.path.join(base_dir, f"*{tier_key}*{arch_key}*fold{fold_num}*.keras"),
        os.path.join(base_dir, f"*{tier_key}*fold{fold_num}*.keras"),
    ]

    for pat in search_patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    return None

# =============================================================================
# 3. MULTI-TIER AUTOMATED EVALUATION & CACHING ENGINE
# =============================================================================
CACHE_CSV_PATH: str = os.path.join(MODEL_BASE_DIR, "unified_cross_benchmark_eval_10models.csv")

if os.path.exists(CACHE_CSV_PATH):
    logger.info("Loading cached master benchmark predictions from: %s", CACHE_CSV_PATH)
    df_master_all = pd.read_csv(CACHE_CSV_PATH)
else:
    logger.info("Cached predictions not found. Executing full multi-tier evaluation pipeline...")
    master_records: List[Dict[str, Any]] = []

    for tier_info in BENCHMARK_TIERS:
        tier_key = tier_info["key"]
        tier_name = tier_info["name"]
        tier_sigma = tier_info["sigma"]
        is_real = tier_info["is_real"]
        tensor_directory = DATASET_DIRECTORIES.get(tier_key)

        if not tensor_directory or not os.path.exists(tensor_directory):
            logger.warning("Tensor directory for tier '%s' not found (%s). Skipping.", tier_key, tensor_directory)
            continue

        tier_files = sorted(glob.glob(os.path.join(tensor_directory, "tensor_G*.nc")))
        active_tier_files = [f for f in tier_files if parse_tensor_metadata(f)[0] in ACTIVE_CLUSTERS]

        if not active_tier_files:
            logger.warning("No active NetCDF tensors discovered in %s. Skipping tier.", tensor_directory)
            continue

        # Establish 80% chronological historical cutoff date
        all_dates = [parse_tensor_metadata(f)[1] for f in active_tier_files]
        cutoff_date = min(all_dates) + 0.80 * (max(all_dates) - min(all_dates))

        for arch_info in ARCHITECTURES:
            arch_key = arch_info["key"]
            arch_label = arch_info["label"]
            logger.info("==========================================================================")
            logger.info("EVALUATING: Tier='%s' (%s) | Architecture='%s'", tier_name, tier_key, arch_label)
            logger.info("==========================================================================")

            for fold_num, val_cluster_ids in DEFAULT_SPATIAL_FOLDS.items():
                ckpt_path = discover_model_checkpoint(MODEL_BASE_DIR, tier_key, arch_key, fold_num)

                if ckpt_path is None or not os.path.exists(ckpt_path):
                    logger.warning("Checkpoint for [%s | %s | Fold %d] not found. Skipping fold.", tier_key, arch_key, fold_num)
                    continue

                logger.info(" -> Loading Checkpoint: %s", os.path.basename(ckpt_path))
                try:
                    fold_model = tf.keras.models.load_model(ckpt_path, compile=False)
                except Exception as e_load_m:
                    logger.error("Failed loading model %s: %s", ckpt_path, e_load_m)
                    continue

                # Isolate withheld out-of-sample validation tensors
                fold_val_files = [
                    f for f in active_tier_files
                    if (parse_tensor_metadata(f)[0] in val_cluster_ids and parse_tensor_metadata(f)[1] <= cutoff_date)
                ]

                if not fold_val_files:
                    continue

                val_seq = DualBranchEvaluationSequence(file_list=fold_val_files, batch_size=32)

                for b_idx in range(len(val_seq)):
                    (bx_spatial, bx_met, bx_log_m), by_true = val_seq[b_idx]

                    # Dynamic Multi-Modal Forward Pass: 3 Inputs vs 2 Inputs
                    if len(fold_model.inputs) == 3:
                        by_pred = fold_model.predict_on_batch([bx_spatial, bx_met, bx_log_m]).flatten()
                    else:
                        by_pred = fold_model.predict_on_batch([bx_spatial, bx_met]).flatten()

                    for s_idx in range(len(by_true)):
                        meta = val_seq.cached_meta[b_idx * 32 + s_idx]
                        y_t = float(by_true[s_idx])
                        y_p = float(by_pred[s_idx])
                        true_tonnes = float(meta["target_physical_tonnes"])
                        pred_tonnes = max(0.0, float(10.0 ** y_p - 0.1))

                        master_records.append({
                            "Tier_Key": tier_key,
                            "Tier_Name": tier_name,
                            "Noise_Sigma": tier_sigma,
                            "Is_Real": is_real,
                            "Architecture_Key": arch_key,
                            "Architecture_Label": arch_label,
                            "Model_Config": f"{tier_key}_{arch_key}",
                            "Fold": fold_num,
                            "Cluster_ID": meta["cluster_id"],
                            "Cluster_Name": meta["cluster_name"],
                            "Sector": meta["primary_sector"],
                            "Filename": meta["filename"],
                            "Target_Native": y_t,
                            "Pred_Native": y_p,
                            "Target_Physical_Tonnes": true_tonnes,
                            "Pred_Physical_Tonnes": pred_tonnes,
                            "Residual_Tonnes": pred_tonnes - true_tonnes,
                            "Relative_Error_Pct": ((pred_tonnes - true_tonnes) / max(0.1, true_tonnes)) * 100.0,
                            "Abs_Error_Tonnes": abs(pred_tonnes - true_tonnes)
                        })

    df_master_all = pd.DataFrame(master_records)
    df_master_all.to_csv(CACHE_CSV_PATH, index=False)
    logger.info("Evaluation complete. Cached %d prediction records to %s", len(df_master_all), CACHE_CSV_PATH)

# =============================================================================
# 4. STATISTICAL METRIC & POINT DENSITY HELPERS
# =============================================================================
def compute_normalized_point_density(
    x_data: np.ndarray,
    y_data: np.ndarray,
    subsample_limit: int = 4000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates normalized 2D Gaussian Kernel Density for each (x, y) point
    and sorts arrays so highest-density points are rendered on top.
    """
    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
    x = np.asarray(x_data[valid_mask], dtype=np.float64)
    y = np.asarray(y_data[valid_mask], dtype=np.float64)

    n_points = len(x)
    if n_points < 3:
        return x, y, np.ones_like(x, dtype=np.float64)

    # Subsample if dataset is large to maintain fast interactive plotting
    if n_points > subsample_limit:
        idx_sub = np.random.choice(n_points, size=subsample_limit, replace=False)
        xy_fit = np.vstack([x[idx_sub], y[idx_sub]])
    else:
        xy_fit = np.vstack([x, y])

    try:
        kde = gaussian_kde(xy_fit)
        densities = kde(np.vstack([x, y]))
    except Exception as e_kde:
        logger.warning("Gaussian KDE calculation fallback triggered: %s", e_kde)
        counts, x_edges, y_edges = np.histogram2d(x, y, bins=40)
        x_bins = np.digitize(x, x_edges[:-1]) - 1
        y_bins = np.digitize(y, y_edges[:-1]) - 1
        x_bins = np.clip(x_bins, 0, counts.shape[0] - 1)
        y_bins = np.clip(y_bins, 0, counts.shape[1] - 1)
        densities = counts[x_bins, y_bins].astype(np.float64)

    # Normalize density to [0.0, 1.0]
    d_min, d_max = float(np.min(densities)), float(np.max(densities))
    if d_max > d_min:
        z_norm = (densities - d_min) / (d_max - d_min)
    else:
        z_norm = np.ones_like(densities)

    # Sort so high-density points are plotted on top
    sort_indices = np.argsort(z_norm)
    return x[sort_indices], y[sort_indices], z_norm[sort_indices]


def compute_publication_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float, float, float, float]:
    """
    Computes R², NRMSE (%), Systematic Bias (%), Log-Space Standard Deviation (dex),
    and Multiplicative Geometric Spread Factor (GSD).
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    # R² Score
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))

    # NRMSE (%) = (RMSE / Mean(y_true)) * 100
    rmse = np.sqrt(np.mean((y_p - y_t) ** 2))
    mean_true = np.mean(y_t)
    nrmse_pct = float((rmse / max(1e-4, mean_true)) * 100.0)

    # Systematic Bias (%) = [(Sum(y_pred) - Sum(y_true)) / Sum(y_true)] * 100
    sum_true = np.sum(y_t)
    sum_pred = np.sum(y_p)
    bias_pct = float(((sum_pred - sum_true) / max(1e-4, sum_true)) * 100.0)

    # Log-Normal Multiplicative Dispersion (Solution A)
    log_t = np.log10(np.maximum(0.0, y_t) + 0.1)
    log_p = np.log10(np.maximum(0.0, y_p) + 0.1)
    log_residuals = log_p - log_t
    sigma_log = float(np.std(log_residuals, ddof=1))
    
    # Geometric Standard Deviation (Multiplicative Factor)
    gsd = float(10.0 ** sigma_log)
    sigma_pct = float((gsd - 1.0) * 100.0)

    return r2, nrmse_pct, bias_pct, gsd, sigma_pct

# =============================================================================
# 5. FIGURE 1: MASTER 2x5 DENSITY PARITY MATRIX (WITH LOG/LINEAR TOGGLE)
# =============================================================================
def plot_density_parity_matrix(
    df: pd.DataFrame,
    use_log_scale: bool = False,
    output_filepath: Optional[str] = None,
    max_axis_val: Optional[float] = None,
    dpi: int = 150
) -> Tuple[Figure, np.ndarray]:
    """
    Renders the complete 2x5 Multi-Panel Parity Matrix using 2D Data Density Colormapping
    and Log-Normal Multiplicative 1-Sigma Uncertainty Fan Envelopes. Supports toggling
    between Linear and Log-Log scale.
    """
    try:
        if output_filepath is None:
            scale_tag = "log_scale" if use_log_scale else "linear_scale"
            output_filepath = os.path.join(FIGURES_DIR, f"Figure1_Density_Parity_Matrix_{scale_tag}.png")

        n_rows = len(ARCHITECTURES)
        n_cols = len(BENCHMARK_TIERS)

        fig, axes_grid = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(19.5, 8.2),
            dpi=dpi,
            sharex=True,
            sharey=True
        )
        fig.patch.set_facecolor("white")

        axes: np.ndarray = np.atleast_2d(axes_grid)
        panel_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        scatter_handle = None

        # Determine Axis Boundaries & Evaluation Lines
        if use_log_scale:
            min_axis: float = 0.8
            max_axis: float = 350.0 if max_axis_val is None else max_axis_val
            x_fan_eval = np.geomspace(min_axis, max_axis, 200)
        else:
            min_axis = 0.0
            max_axis = 300.0 if max_axis_val is None else max_axis_val
            x_fan_eval = np.linspace(min_axis, max_axis, 200)

        for r_idx, arch_info in enumerate(ARCHITECTURES):
            arch_key = arch_info["key"]
            arch_label = arch_info["label"]

            for c_idx, tier_info in enumerate(BENCHMARK_TIERS):
                tier_key = tier_info["key"]
                tier_name = tier_info["name"]
                panel_idx = r_idx * n_cols + c_idx
                letter_tag = panel_letters[panel_idx]

                ax: Axes = axes[r_idx, c_idx]
                ax.set_facecolor("white")
                ax.set_aspect("equal", adjustable="box")

                sub_df = df[(df["Tier_Key"] == tier_key) & (df["Architecture_Key"] == arch_key)]

                if sub_df.empty:
                    ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes, color="gray")
                    continue

                raw_x = sub_df["Target_Physical_Tonnes"].values
                raw_y = sub_df["Pred_Physical_Tonnes"].values

                # 1. Compute 2D Point Density & Sort
                x_sorted, y_sorted, z_density = compute_normalized_point_density(raw_x, raw_y)

                # 2. Compute Publication Standard Metrics + Log-Normal 1-Sigma
                r2, nrmse, bias, gsd, sigma_pct = compute_publication_metrics(raw_x, raw_y)

                # 3. Apply Scaling Mode
                if use_log_scale:
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    x_plot = np.maximum(min_axis, x_sorted)
                    y_plot = np.maximum(min_axis, y_sorted)
                else:
                    x_plot = x_sorted
                    y_plot = y_sorted

                # 4. Render 2D Density Scatter
                scatter_handle = ax.scatter(
                    x_plot,
                    y_plot,
                    c=z_density,
                    cmap=DENSITY_COLORMAP,
                    vmin=0.0,
                    vmax=1.0,
                    s=14,
                    alpha=0.85,
                    edgecolors="none",
                    zorder=3
                )

                # 5. Reference Lines & Log-Normal 1-Sigma Fan
                ax.plot(
                    [min_axis, max_axis],
                    [min_axis, max_axis],
                    color="#4da6ff",
                    linestyle="-",
                    linewidth=1.3,
                    zorder=2
                )

                y_fan_upper = gsd * (x_fan_eval + 0.1) - 0.1
                y_fan_lower = np.maximum(min_axis, (1.0 / gsd) * (x_fan_eval + 0.1) - 0.1)

                ax.plot(x_fan_eval, y_fan_upper, color="#555555", linestyle="--", linewidth=0.9, alpha=0.85, zorder=2)
                ax.plot(x_fan_eval, y_fan_lower, color="#555555", linestyle="--", linewidth=0.9, alpha=0.85, zorder=2)
                ax.fill_between(x_fan_eval, y_fan_lower, y_fan_upper, color="#777777", alpha=0.06, zorder=1)

                # 6. Panel Title & Statistical Metrics Header
                ax.text(
                    0.04, 0.94,
                    f"$\\mathbf{{{letter_tag}}}$   {tier_name}",
                    transform=ax.transAxes,
                    fontsize=9.0,
                    verticalalignment="top"
                )

                bias_sign = "+" if bias >= 0 else ""
                metrics_text = (
                    f"$R^2={r2:.2f}$\n"
                    f"$\\mathrm{{NRMSE}}={nrmse:.1f}\\%$\n"
                    f"$\\mathrm{{Bias}}={bias_sign}{bias:.1f}\\%$\n"
                    f"$1\\sigma=\\pm{sigma_pct:.1f}\\%$"
                )
                ax.text(
                    0.04, 0.85,
                    metrics_text,
                    transform=ax.transAxes,
                    fontsize=7.6,
                    verticalalignment="top",
                    linespacing=1.20
                )

                # 7. Axis Limits & Ticks Formatting
                ax.set_xlim(min_axis, max_axis)
                ax.set_ylim(min_axis, max_axis)

                if use_log_scale:
                    major_ticks = [1, 10, 100]
                    ax.set_xticks(major_ticks)
                    ax.set_yticks(major_ticks)
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

                    minor_ticks = [2, 5, 20, 50, 200]
                    ax.set_xticks(minor_ticks, minor=True)
                    ax.set_yticks(minor_ticks, minor=True)
                    ax.xaxis.set_minor_formatter(ticker.FormatStrFormatter("%d"))
                    ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%d"))
                else:
                    ax.set_xticks([0, 75, 150, 225, 300])
                    ax.set_yticks([0, 75, 150, 225, 300])

                ax.tick_params(which="both", labelsize=8.0, length=3.5, width=0.8)

                # Axis Labels
                if c_idx == 0:
                    ax.set_ylabel("Estimated flux (t/seq)", fontsize=9.0, fontweight="bold")
                if r_idx == n_rows - 1:
                    ax.set_xlabel("True flux (t/seq)", fontsize=9.0, fontweight="bold")

                # Right Row Header Label
                if c_idx == n_cols - 1:
                    ax.text(
                        1.08, 0.5,
                        arch_label,
                        transform=ax.transAxes,
                        rotation=-90,
                        va="center",
                        ha="left",
                        fontsize=9.5,
                        fontweight="bold"
                    )

        # 8. Shared Colorbar Docking
        if scatter_handle is not None:
            cbar_ax = fig.add_axes([0.935, 0.18, 0.012, 0.64])
            cb = fig.colorbar(scatter_handle, cax=cbar_ax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            cb.set_label("Data density", fontsize=9.5, fontweight="bold", labelpad=8)
            cb.ax.tick_params(labelsize=8.0)

        fig.subplots_adjust(left=0.06, right=0.91, bottom=0.10, top=0.95, wspace=0.12, hspace=0.14)

        scale_title = "Log-Log Scale" if use_log_scale else "Linear Scale"
        fig.suptitle(
            f"All Model Predictions Parity Matrix ({scale_title})",
            fontsize=12.0,
            fontweight="bold",
            y=0.99
        )
        
        out_dir = os.path.dirname(output_filepath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fig.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
        logger.info("Successfully rendered and saved density parity matrix to: %s", output_filepath)
        plt.show()

        return fig, axes

    except Exception as e_plot:
        logger.error("Failed rendering density parity matrix: %s", e_plot, exc_info=True)
        raise

# Render Figure 1 (Linear Scale by default)
fig_density, axes_density = plot_density_parity_matrix(
    df=df_master_all,
    use_log_scale=False,
    dpi=150
)

# =============================================================================
# 6. STANDALONE 2x2 FILTERED BENCHMARK (EXCLUDING G01 SUPER-CLUSTER)
# =============================================================================
EXCLUDED_CLUSTER_IDS: List[str] = ["G01"]

TARGET_TIERS_2X2: List[Dict[str, Any]] = [
    {"key": "next_gen", "name": "Next-Gen", "sigma": 0.05},
    {"key": "moderate", "name": "Moderate",  "sigma": 0.15}
]

TARGET_ARCHITECTURES_2X2: List[Dict[str, str]] = [
    {"key": "mass_sum", "label": "Physics-Anchored"},
    {"key": "endtoend", "label": "Direct End-to-End ML"}
]


def plot_filtered_2x2_parity_matrix(
    df: pd.DataFrame,
    excluded_clusters: List[str] = EXCLUDED_CLUSTER_IDS,
    use_log_scale: bool = False,
    output_filepath: Optional[str] = None,
    max_axis_val: Optional[float] = None,
    dpi: int = 150
) -> Tuple[Figure, np.ndarray]:
    """
    Renders a 2x2 density scatter matrix (Next-Gen vs. Moderate x Mass-Sum vs. End-to-End)
    with the largest emitter cluster filtered out. Supports toggling between Linear and Log-Log scale.
    """
    try:
        if output_filepath is None:
            scale_tag = "log_scale" if use_log_scale else "linear_scale"
            output_filepath = os.path.join(FIGURES_DIR, f"Figure_NextGen_Moderate_NoG01_Parity_{scale_tag}.png")

        df_filtered = df[~df["Cluster_ID"].isin(excluded_clusters)].copy()
        logger.info("Filtered out clusters %s: Retained %d / %d samples.", excluded_clusters, len(df_filtered), len(df))

        n_rows = len(TARGET_ARCHITECTURES_2X2)
        n_cols = len(TARGET_TIERS_2X2)

        fig, axes_grid = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(10.2, 8.5),
            dpi=dpi,
            sharex=True,
            sharey=True
        )
        fig.patch.set_facecolor("white")

        axes: np.ndarray = np.atleast_2d(axes_grid)
        panel_letters = ["A", "B", "C", "D"]
        scatter_handle = None

        if use_log_scale:
            min_axis: float = 0.8
            max_axis: float = 100.0 if max_axis_val is None else max_axis_val
            x_fan_eval = np.geomspace(min_axis, max_axis, 200)
        else:
            min_axis = 0.0
            max_axis = 80.0 if max_axis_val is None else max_axis_val
            x_fan_eval = np.linspace(min_axis, max_axis, 200)

        for r_idx, arch_info in enumerate(TARGET_ARCHITECTURES_2X2):
            arch_key = arch_info["key"]
            arch_label = arch_info["label"]

            for c_idx, tier_info in enumerate(TARGET_TIERS_2X2):
                tier_key = tier_info["key"]
                tier_name = tier_info["name"]
                panel_idx = r_idx * n_cols + c_idx
                letter_tag = panel_letters[panel_idx]

                ax: Axes = axes[r_idx, c_idx]
                ax.set_facecolor("white")
                ax.set_aspect("equal", adjustable="box")

                sub_df = df_filtered[
                    (df_filtered["Tier_Key"] == tier_key) & 
                    (df_filtered["Architecture_Key"] == arch_key)
                ]

                if sub_df.empty:
                    ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes, color="gray")
                    continue

                raw_x = sub_df["Target_Physical_Tonnes"].values
                raw_y = sub_df["Pred_Physical_Tonnes"].values

                x_sorted, y_sorted, z_density = compute_normalized_point_density(raw_x, raw_y)
                r2, nrmse, bias, gsd, sigma_pct = compute_publication_metrics(raw_x, raw_y)

                if use_log_scale:
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    x_plot = np.maximum(min_axis, x_sorted)
                    y_plot = np.maximum(min_axis, y_sorted)
                else:
                    x_plot = x_sorted
                    y_plot = y_sorted

                scatter_handle = ax.scatter(
                    x_plot,
                    y_plot,
                    c=z_density,
                    cmap=DENSITY_COLORMAP,
                    vmin=0.0,
                    vmax=1.0,
                    s=16,
                    alpha=0.85,
                    edgecolors="none",
                    zorder=3
                )

                ax.plot(
                    [min_axis, max_axis],
                    [min_axis, max_axis],
                    color="#4da6ff",
                    linestyle="-",
                    linewidth=1.3,
                    zorder=2
                )

                y_fan_upper = gsd * (x_fan_eval + 0.1) - 0.1
                y_fan_lower = np.maximum(min_axis, (1.0 / gsd) * (x_fan_eval + 0.1) - 0.1)

                ax.plot(x_fan_eval, y_fan_upper, color="#555555", linestyle="--", linewidth=0.9, alpha=0.85, zorder=2)
                ax.plot(x_fan_eval, y_fan_lower, color="#555555", linestyle="--", linewidth=0.9, alpha=0.85, zorder=2)
                ax.fill_between(x_fan_eval, y_fan_lower, y_fan_upper, color="#777777", alpha=0.06, zorder=1)

                ax.text(
                    0.04, 0.94,
                    f"$\\mathbf{{{letter_tag}}}$   {tier_name}",
                    transform=ax.transAxes,
                    fontsize=9.2,
                    verticalalignment="top"
                )

                bias_sign = "+" if bias >= 0 else ""
                metrics_text = (
                    f"$R^2={r2:.2f}$\n"
                    f"$\\mathrm{{NRMSE}}={nrmse:.1f}\\%$\n"
                    f"$\\mathrm{{Bias}}={bias_sign}{bias:.1f}\\%$\n"
                    f"$1\\sigma=\\pm{sigma_pct:.1f}\\%$"
                )
                ax.text(
                    0.04, 0.85,
                    metrics_text,
                    transform=ax.transAxes,
                    fontsize=7.8,
                    verticalalignment="top",
                    linespacing=1.20
                )

                ax.set_xlim(min_axis, max_axis)
                ax.set_ylim(min_axis, max_axis)

                if use_log_scale:
                    major_ticks = [1, 10, 100]
                    ax.set_xticks(major_ticks)
                    ax.set_yticks(major_ticks)
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

                    minor_ticks = [2, 5, 20, 50]
                    ax.set_xticks(minor_ticks, minor=True)
                    ax.set_yticks(minor_ticks, minor=True)
                    ax.xaxis.set_minor_formatter(ticker.FormatStrFormatter("%d"))
                    ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%d"))
                else:
                    ax.set_xticks([0, 20, 40, 60, 80])
                    ax.set_yticks([0, 20, 40, 60, 80])

                ax.tick_params(which="both", labelsize=8.0, length=3.5, width=0.8)

                if c_idx == 0:
                    ax.set_ylabel("Estimated flux (t/seq)", fontsize=9.5, fontweight="bold")
                if r_idx == n_rows - 1:
                    ax.set_xlabel("True flux (t/seq)", fontsize=9.5, fontweight="bold")

                if c_idx == n_cols - 1:
                    ax.text(
                        1.08, 0.5,
                        arch_label,
                        transform=ax.transAxes,
                        rotation=-90,
                        va="center",
                        ha="left",
                        fontsize=9.5,
                        fontweight="bold"
                    )

        if scatter_handle is not None:
            cbar_ax = fig.add_axes([0.930, 0.18, 0.015, 0.64])
            cb = fig.colorbar(scatter_handle, cax=cbar_ax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            cb.set_label("Data density", fontsize=9.5, fontweight="bold", labelpad=8)
            cb.ax.tick_params(labelsize=8.0)

        scale_title = "Log-Log Scale" if use_log_scale else "Linear Scale"
        fig.suptitle(
            f"Next-Gen and Moderate Simulation Model Predictions (G01 cluster removed) ({scale_title})",
            fontsize=11.5,
            fontweight="bold",
            y=0.99
        )
        
        fig.subplots_adjust(left=0.08, right=0.89, bottom=0.10, top=0.95, wspace=0.12, hspace=0.14)

        out_dir = os.path.dirname(output_filepath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fig.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
        logger.info("Successfully exported filtered 2x2 parity matrix to: %s", output_filepath)
        plt.show()

        return fig, axes

    except Exception as e_plot:
        logger.error("Failed rendering filtered 2x2 matrix: %s", e_plot, exc_info=True)
        raise

# Render 2x2 Filtered Comparison (Linear Scale)
fig_filtered, axes_filtered = plot_filtered_2x2_parity_matrix(
    df=df_master_all,
    excluded_clusters=EXCLUDED_CLUSTER_IDS,
    use_log_scale=False,
    dpi=150
)

# =============================================================================
# 7. FIGURE 2: SENSOR NOISE DEGRADATION CURVES & REAL-DATA ANCHOR
# =============================================================================
def plot_figure2_degradation_curves(df: pd.DataFrame, output_path: str) -> None:
    """
    Renders 3 side-by-side metric curves (R², MAE, Slope) vs Instrument Noise Floor σ,
    with Real Operational GEMS Level-2 results overlaid as diamond anchor points.
    """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5.5), dpi=150)
    fig.patch.set_facecolor("white")

    metrics_list = [
        {"key": "R2",    "title": "A. Goodness of Fit ($R^2$ Score)",        "ylabel": "$R^2$ Score",                  "ylim": [-0.1, 1.05]},
        {"key": "MAE",   "title": "B. Bulk Prediction Error (MAE)",          "ylabel": "MAE [metric tonnes / seq]",     "ylim": [0, 35]},
        {"key": "Slope", "title": "C. Calibration Sensitivity (Slope $m$)",  "ylabel": "Fitted Regression Slope ($m$)", "ylim": [0.0, 1.25]}
    ]

    for m_idx, m_info in enumerate(metrics_list):
        ax = axes[m_idx]
        ax.set_facecolor("#fbfbfb")
        ax.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=1)

        for arch_info in ARCHITECTURES:
            arch_key = arch_info["key"]
            arch_label = arch_info["label"]
            arch_color = arch_info["color"]

            sigmas, metric_vals = [], []
            real_val = None

            for tier_info in BENCHMARK_TIERS:
                sub_df = df[(df["Tier_Key"] == tier_info["key"]) & (df["Architecture_Key"] == arch_key)]
                if sub_df.empty:
                    continue

                y_true = sub_df["Target_Physical_Tonnes"].values
                y_pred = sub_df["Pred_Physical_Tonnes"].values

                if m_info["key"] == "R2":
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    val = float(1.0 - (ss_res / (ss_tot + 1e-8)))
                elif m_info["key"] == "MAE":
                    val = float(np.mean(np.abs(y_pred - y_true)))
                elif m_info["key"] == "Slope":
                    val = float(np.polyfit(y_true, y_pred, 1)[0]) if len(y_true) > 1 else 1.0

                if not tier_info["is_real"]:
                    sigmas.append(tier_info["sigma"])
                    metric_vals.append(val)
                else:
                    real_val = val

            ax.plot(
                sigmas, metric_vals, marker="o", markersize=7, linewidth=2.2,
                color=arch_color, label=f"{arch_label} (Synthetic)", zorder=3
            )

            if real_val is not None:
                ax.scatter(
                    0.35, real_val, marker="D", s=110, color=arch_color, edgecolors="black", linewidths=1.2,
                    label=f"{arch_label} (Real GEMS L2)", zorder=4
                )

        if m_info["key"] == "R2":
            ax.axhline(1.0, color="green", linestyle=":", alpha=0.6, label="Ideal Limit ($R^2=1.0$)")
        elif m_info["key"] == "Slope":
            ax.axhline(1.0, color="black", linestyle="--", alpha=0.7, label="Ideal Slope ($m=1.0$)")

        ax.set_title(m_info["title"], fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Instrument Noise Floor $\\sigma_{\\mathrm{inst}}$ [Dobson Units (DU)]", fontsize=9.5, fontweight="bold")
        ax.set_ylabel(m_info["ylabel"], fontsize=9.5, fontweight="bold")
        ax.set_ylim(m_info["ylim"])
        ax.tick_params(labelsize=8.5)
        if m_idx == 0:
            ax.legend(loc="lower left", fontsize=7.8, framealpha=0.90)

    plt.suptitle("Figure 2: Instrument Noise Degradation Curves & Operational Real-Data Benchmark Overlay\nSynthetic OSSE Performance vs. Sensor Noise Floor with Real Operational GEMS Level-2 Results", fontsize=12.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1.0, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved Figure 2 to: %s", output_path)
    plt.show()

plot_figure2_degradation_curves(df_master_all, os.path.join(FIGURES_DIR, "Figure2_Degradation_Curves.png"))

# =============================================================================
# 8. FIGURE 3: DYNAMIC RANGE & EMITTER SCALE STRATIFICATION
# =============================================================================
def plot_figure3_dynamic_range_stratification(df: pd.DataFrame, output_path: str) -> None:
    """
    Evaluates relative percentage error across 3 emission regimes:
    Small (<20 t/seq), Medium (20-80 t/seq), and Super-Emitters (>80 t/seq).
    """
    df_copy = df.copy()

    def assign_bin(q: float) -> str:
        if q < 20.0:
            return "1. Small (<20 t)"
        elif q <= 80.0:
            return "2. Medium (20-80 t)"
        else:
            return "3. Super-Emitter (>80 t)"

    df_copy["Emission_Scale"] = df_copy["Target_Physical_Tonnes"].apply(assign_bin)
    scale_bins = ["1. Small (<20 t)", "2. Medium (20-80 t)", "3. Super-Emitter (>80 t)"]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6.0), dpi=150, sharey=True)
    fig.patch.set_facecolor("white")

    for b_idx, s_bin in enumerate(scale_bins):
        ax = axes[b_idx]
        ax.set_facecolor("#fbfbfb")
        ax.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=1)

        sub_bin_df = df_copy[df_copy["Emission_Scale"] == s_bin]

        plot_data = []
        positions = []
        colors = []
        labels = []

        pos_cursor = 1.0
        for tier_info in BENCHMARK_TIERS:
            tier_key = tier_info["key"]
            t_name = tier_info["name"]

            for arch_info in ARCHITECTURES:
                arch_key = arch_info["key"]
                vals = sub_bin_df[(sub_bin_df["Tier_Key"] == tier_key) & (sub_bin_df["Architecture_Key"] == arch_key)]["Relative_Error_Pct"].values

                plot_data.append(vals if len(vals) > 0 else np.array([0.0]))
                positions.append(pos_cursor)
                colors.append(arch_info["color"])
                pos_cursor += 1.0

            labels.append(t_name)
            pos_cursor += 0.8

        bp = ax.boxplot(
            plot_data, positions=positions, patch_artist=True, widths=0.7,
            showmeans=True, meanline=True, showfliers=False,
            boxprops=dict(linewidth=0.8), medianprops=dict(color="black", linewidth=1.2),
            meanprops=dict(color="yellow", linewidth=1.2, linestyle="-")
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        tick_locs = [1.5 + i * 2.8 for i in range(len(BENCHMARK_TIERS))]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.0, fontweight="bold")

        ax.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
        ax.axhline(25, color="red", linestyle=":", alpha=0.5)
        ax.axhline(-25, color="red", linestyle=":", alpha=0.5)

        ax.set_title(f"Bin {s_bin}\n(N={len(sub_bin_df)} samples)", fontsize=10.5, fontweight="bold", pad=8)
        if b_idx == 0:
            ax.set_ylabel("Relative Prediction Error (%)\n$[(\\widehat{Q} - Q_{\\mathrm{true}})/Q_{\\mathrm{true}}] \\times 100\\%$", fontsize=9.5, fontweight="bold")
        ax.set_ylim(-120, 160)
        ax.tick_params(labelsize=8)

    legend_patches = [
        mpatches.Patch(facecolor="#1f77b4", edgecolor="black", label="Physics-Anchored (Mass Skip)"),
        mpatches.Patch(facecolor="#ff7f0e", edgecolor="black", label="Direct End-to-End ML")
    ]
    axes[0].legend(handles=legend_patches, loc="upper left", fontsize=8.0, framealpha=0.90)

    plt.suptitle("Figure 3: Relative Inversion Error Distribution Stratified by Emitter Scale\nRobustness Evaluation: Faint Emitters vs. Intermediate Hubs vs. Super-Emitters", fontsize=12.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1.0, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved Figure 3 to: %s", output_path)
    plt.show()

plot_figure3_dynamic_range_stratification(df_master_all, os.path.join(FIGURES_DIR, "Figure3_Scale_Stratification.png"))

# =============================================================================
# 9. FIGURE 4: MASTER 2x5 CLUSTER-LEVEL MULTI-YEAR MEAN PARITY MATRIX
# =============================================================================
def aggregate_cluster_multiyear_metrics(df_subset: pd.DataFrame) -> pd.DataFrame:
    """Computes multi-year temporal mean, standard deviation, and sample counts per cluster."""
    cluster_records: List[Dict[str, Any]] = []

    for gid, group in df_subset.groupby("Cluster_ID"):
        c_name = group["Cluster_Name"].iloc[0] if "Cluster_Name" in group.columns else gid
        sector = group["Sector"].iloc[0] if "Sector" in group.columns else "Industrial"

        true_vals = group["Target_Physical_Tonnes"].values
        pred_vals = group["Pred_Physical_Tonnes"].values

        m_true = float(np.mean(true_vals))
        m_pred = float(np.mean(pred_vals))
        std_pred = float(np.std(pred_vals, ddof=1)) if len(pred_vals) > 1 else 0.0

        cluster_records.append({
            "Cluster_ID": gid,
            "Cluster_Name": c_name,
            "Sector": sector,
            "Sample_Count": len(group),
            "Mean_True_Tonnes": m_true,
            "Mean_Pred_Tonnes": m_pred,
            "Std_Pred_Tonnes": std_pred,
            "Abs_Error_Tonnes": abs(m_pred - m_true)
        })

    return pd.DataFrame(cluster_records).sort_values(by="Mean_True_Tonnes", ascending=False)


def compute_cluster_panel_statistics(
    df_clusters: pd.DataFrame
) -> Tuple[float, float, float, float, float]:
    """Calculates cluster-level Pearson r, MAE, slope, and log-normal geometric dispersion."""
    y_t = df_clusters["Mean_True_Tonnes"].values
    y_p = df_clusters["Mean_Pred_Tonnes"].values

    if len(y_t) < 2:
        return 0.0, 0.0, 1.0, 1.0, 0.0

    r_val = float(np.corrcoef(y_t, y_p)[0, 1])
    mae_val = float(np.mean(np.abs(y_p - y_t)))
    slope_val = float(np.polyfit(y_t, y_p, 1)[0])

    log_t = np.log10(np.maximum(0.1, y_t) + 0.1)
    log_p = np.log10(np.maximum(0.1, y_p) + 0.1)
    log_residuals = log_p - log_t
    sigma_log = float(np.std(log_residuals, ddof=1))

    gsd = float(10.0 ** sigma_log)
    sigma_pct = float((gsd - 1.0) * 100.0)

    return r_val, mae_val, slope_val, gsd, sigma_pct


def plot_figure4_master_cluster_parity(
    df_master: pd.DataFrame,
    use_log_scale: bool = True,
    output_filepath: Optional[str] = None,
    dpi: int = 150
) -> Tuple[Figure, np.ndarray]:
    """
    Renders the complete 2x5 Multi-Year Cluster Mean Emissions Parity Matrix across all 10 configurations.
    """
    try:
        if output_filepath is None:
            scale_tag = "log_scale" if use_log_scale else "linear_scale"
            output_filepath = os.path.join(FIGURES_DIR, f"Figure4_Master_Cluster_Parity_2x5_{scale_tag}.png")

        n_rows = len(ARCHITECTURES)
        n_cols = len(BENCHMARK_TIERS)

        fig, axes_grid = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(20.5, 8.8),
            dpi=dpi,
            sharex=True,
            sharey=True
        )
        fig.patch.set_facecolor("white")
        axes: np.ndarray = np.atleast_2d(axes_grid)

        panel_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

        if use_log_scale:
            min_axis: float = 0.8
            max_axis: float = 320.0
            x_fan_eval = np.geomspace(min_axis, max_axis, 200)
        else:
            min_axis = 0.0
            max_axis = 240.0
            x_fan_eval = np.linspace(min_axis, max_axis, 200)

        for r_idx, arch_info in enumerate(ARCHITECTURES):
            arch_key = arch_info["key"]
            arch_label = arch_info["label"]

            for c_idx, tier_info in enumerate(BENCHMARK_TIERS):
                tier_key = tier_info["key"]
                tier_name = tier_info["name"]
                panel_idx = r_idx * n_cols + c_idx
                letter_tag = panel_letters[panel_idx]

                ax: Axes = axes[r_idx, c_idx]
                ax.set_facecolor("white")
                ax.set_aspect("equal", adjustable="box")

                sub_df = df_master[
                    (df_master["Tier_Key"] == tier_key) &
                    (df_master["Architecture_Key"] == arch_key)
                ]

                if sub_df.empty:
                    ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes, color="gray")
                    continue

                df_clusters = aggregate_cluster_multiyear_metrics(sub_df)
                r_val, mae_val, slope_val, gsd, sigma_pct = compute_cluster_panel_statistics(df_clusters)

                if use_log_scale:
                    ax.set_xscale("log")
                    ax.set_yscale("log")

                ax.plot(
                    [min_axis, max_axis],
                    [min_axis, max_axis],
                    color="#4da6ff",
                    linestyle="-",
                    linewidth=1.3,
                    zorder=2
                )

                y_fan_upper = gsd * (x_fan_eval + 0.1) - 0.1
                y_fan_lower = np.maximum(min_axis, (1.0 / gsd) * (x_fan_eval + 0.1) - 0.1)

                ax.plot(x_fan_eval, y_fan_upper, color="#666666", linestyle="--", linewidth=0.9, alpha=0.85, zorder=2)
                ax.plot(x_fan_eval, y_fan_lower, color="#666666", linestyle="--", linewidth=0.9, alpha=0.85, zorder=2)
                ax.fill_between(x_fan_eval, y_fan_lower, y_fan_upper, color="#777777", alpha=0.06, zorder=1)

                for _, row in df_clusters.iterrows():
                    x_val = row["Mean_True_Tonnes"]
                    y_val = row["Mean_Pred_Tonnes"]
                    y_err = row["Std_Pred_Tonnes"]

                    if use_log_scale:
                        y_err_lower = min(y_err, max(0.0, y_val - min_axis))
                        y_err_plot = [[y_err_lower], [y_err]]
                    else:
                        y_err_plot = [[y_err], [y_err]]

                    ax.errorbar(
                        x_val,
                        y_val,
                        yerr=y_err_plot,
                        fmt="none",
                        ecolor="#1f4e79",
                        elinewidth=1.0,
                        capsize=2.5,
                        capthick=0.9,
                        alpha=0.60,
                        zorder=3
                    )

                    ax.scatter(
                        x_val,
                        y_val,
                        color="#1f4e79",
                        edgecolors="black",
                        linewidths=0.9,
                        s=55,
                        alpha=0.90,
                        zorder=4
                    )

                ax.text(
                    0.04, 0.94,
                    f"$\\mathbf{{{letter_tag}}}$   {tier_name}",
                    transform=ax.transAxes,
                    fontsize=9.0,
                    verticalalignment="top"
                )

                metrics_str = (
                    f"Cluster $r = {r_val:.2f}$\n"
                    f"$\\mathrm{{MAE}} = {mae_val:.1f}\\text{{ t}}$\n"
                    f"Slope $m = {slope_val:.2f}$\n"
                    f"$1\\sigma = \\pm{sigma_pct:.1f}\\%$"
                )
                ax.text(
                    0.04, 0.85,
                    metrics_str,
                    transform=ax.transAxes,
                    fontsize=7.6,
                    verticalalignment="top",
                    linespacing=1.20
                )

                ax.set_xlim(min_axis, max_axis)
                ax.set_ylim(min_axis, max_axis)

                if use_log_scale:
                    major_ticks = [1, 10, 100]
                    ax.set_xticks(major_ticks)
                    ax.set_yticks(major_ticks)
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

                    minor_ticks = [2, 5, 20, 50, 200]
                    ax.set_xticks(minor_ticks, minor=True)
                    ax.set_yticks(minor_ticks, minor=True)
                    ax.xaxis.set_minor_formatter(ticker.FormatStrFormatter("%d"))
                    ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%d"))
                else:
                    ax.set_xticks([0, 50, 100, 150, 200, 250])
                    ax.set_yticks([0, 50, 100, 150, 200, 250])

                ax.tick_params(which="both", labelsize=8.0, length=3.5, width=0.8)

                if c_idx == 0:
                    ax.set_ylabel("Inverted Model Mean (t/seq)", fontsize=9.0, fontweight="bold")
                if r_idx == n_rows - 1:
                    ax.set_xlabel("True CleanSYS Mean (t/seq)", fontsize=9.0, fontweight="bold")

                if c_idx == n_cols - 1:
                    ax.text(
                        1.08, 0.5,
                        arch_label,
                        transform=ax.transAxes,
                        rotation=-90,
                        va="center",
                        ha="left",
                        fontsize=9.5,
                        fontweight="bold"
                    )

        scale_title = "Log-Log Scale [1, 300 t/seq]" if use_log_scale else "Linear Scale [0, 240 t/seq]"
        fig.suptitle(
            f"Cluster-Level Multi-Year Mean Emissions Parity Matrix ({scale_title})",
            fontsize=12.0,
            fontweight="bold",
            y=0.99
        )

        fig.subplots_adjust(left=0.06, right=0.92, bottom=0.09, top=0.94, wspace=0.10, hspace=0.12)

        out_dir = os.path.dirname(output_filepath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fig.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
        logger.info("Successfully exported Figure 4 to: %s", output_filepath)
        plt.show()

        return fig, axes

    except Exception as e_fig4:
        logger.error("Failed generating Figure 4: %s", e_fig4, exc_info=True)
        raise

# Render Figure 4 (Log-Log Scale by default)
fig_f4, axes_f4 = plot_figure4_master_cluster_parity(
    df_master=df_master_all,
    use_log_scale=True,
    dpi=150
)

# =============================================================================
# 10. FIGURE 5: PENINSULA-WIDE BUDGET CLOSURE & SYSTEMATIC BIAS WATERFALL
# =============================================================================
def plot_figure5_peninsula_budget_closure(df: pd.DataFrame, output_path: str) -> None:
    """
    Renders a 2-panel chart:
    - Top Panel: Peninsula-wide cumulative SO2 emitted (CleanSYS Truth vs 10 Models).
    - Bottom Panel: Systematic Regional Budget Bias Percentage.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 9.0), dpi=150, sharex=True)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("#fbfbfb")
    ax2.set_facecolor("#fbfbfb")
    ax1.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=1)
    ax2.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=1)

    labels = []
    tot_true_list = []
    tot_pred_list = []
    bias_pct_list = []
    bar_colors = []

    for tier_info in BENCHMARK_TIERS:
        t_key = tier_info["key"]
        t_name = tier_info["name"]

        for arch_info in ARCHITECTURES:
            a_key = arch_info["key"]
            a_label = "PhysSkip" if a_key == "mass_sum" else "EndToEnd"

            sub_df = df[(df["Tier_Key"] == t_key) & (df["Architecture_Key"] == a_key)]
            if sub_df.empty:
                tot_true = 0.0
                tot_pred = 0.0
                bias = 0.0
            else:
                tot_true = float(sub_df["Target_Physical_Tonnes"].sum())
                tot_pred = float(sub_df["Pred_Physical_Tonnes"].sum())
                bias = ((tot_pred - tot_true) / max(1.0, tot_true)) * 100.0

            labels.append(f"{t_name}\n({a_label})")
            tot_true_list.append(tot_true)
            tot_pred_list.append(tot_pred)
            bias_pct_list.append(bias)
            bar_colors.append(arch_info["color"])

    x = np.arange(len(labels))
    width = 0.65

    # Top Panel: Total Mass Emitted
    ax1.bar(x, tot_pred_list, width=width, color=bar_colors, edgecolor="black", linewidth=0.8, alpha=0.85, zorder=3)
    if any(t > 0 for t in tot_true_list):
        ref_truth = max(tot_true_list)
        ax1.axhline(ref_truth, color="red", linestyle="--", linewidth=1.5, label=f"CleanSYS Ground Truth ({ref_truth:,.0f} tonnes)", zorder=4)

    ax1.set_ylabel("Total Reconstructed Mass [tonnes]", fontsize=9.5, fontweight="bold")
    ax1.set_title("A. Peninsula-Wide Multi-Year Cumulative SO2 Mass Conservation", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9.0, framealpha=0.90)
    ax1.tick_params(labelsize=8.5)

    # Bottom Panel: Systematic Bias Percentage
    bias_colors = ["#27ae60" if abs(b) <= 5.0 else ("#e67e22" if abs(b) <= 15.0 else "#c0392b") for b in bias_pct_list]
    ax2.bar(x, bias_pct_list, width=width, color=bias_colors, edgecolor="black", linewidth=0.8, alpha=0.85, zorder=3)
    ax2.axhline(0, color="black", linestyle="-", linewidth=1.0)
    ax2.axhline(10, color="red", linestyle=":", alpha=0.5)
    ax2.axhline(-10, color="red", linestyle=":", alpha=0.5)

    for i, b in enumerate(bias_pct_list):
        ax2.text(i, b + (1.2 if b >= 0 else -3.0), f"{b:+.1f}%", ha="center", fontsize=8.0, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5, fontweight="bold")
    ax2.set_ylabel("Systematic Regional Bias (%)\n$[(\\sum \\widehat{Q} - \\sum Q)/\\sum Q] \\times 100\\%$", fontsize=9.5, fontweight="bold")
    ax2.set_title("B. Systematic Regional Budget Drift Relative to CleanSYS Ground Truth", fontsize=11, fontweight="bold")
    ax2.set_ylim(min(-25, min(bias_pct_list) - 8), max(25, max(bias_pct_list) + 8))
    ax2.tick_params(labelsize=8.5)

    plt.suptitle("Figure 5: Peninsula-Wide Multi-Year Budget Closure & Regional Bias Benchmark\nEvaluating Top-Down Mass Conservation for 10 Inversion Configurations", fontsize=12.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1.0, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved Figure 5 to: %s", output_path)
    plt.show()

plot_figure5_peninsula_budget_closure(df_master_all, os.path.join(FIGURES_DIR, "Figure5_Peninsula_Budget_Closure.png"))

# =============================================================================
# 11. FIGURE 6: CLUSTER x MODEL INVERSION ERROR HEATMAP (MAPE %)
# =============================================================================
def plot_figure6_cluster_error_heatmap(df: pd.DataFrame, output_path: str) -> None:
    """
    Renders a 2D Heatmap Matrix of Mean Absolute Percentage Error (MAPE):
    - Rows: 13 Industrial Clusters (sorted by emission scale G01 -> G17)
    - Columns: 10 Benchmark Configurations (5 Tiers x 2 Architectures)
    """
    clusters_sorted = df.groupby("Cluster_ID")["Target_Physical_Tonnes"].mean().sort_values(ascending=False).index.tolist()

    col_keys = []
    col_labels = []
    for tier_info in BENCHMARK_TIERS:
        for arch_info in ARCHITECTURES:
            cfg_key = f"{tier_info['key']}_{arch_info['key']}"
            col_keys.append(cfg_key)
            col_labels.append(f"{tier_info['name']}\n({arch_info['label'][:8]})")

    heatmap_matrix = np.full((len(clusters_sorted), len(col_keys)), np.nan)

    for r_idx, gid in enumerate(clusters_sorted):
        for c_idx, cfg_k in enumerate(col_keys):
            t_k, a_k = cfg_k.split("_", 1)
            sub_df = df[(df["Cluster_ID"] == gid) & (df["Tier_Key"] == t_k) & (df["Architecture_Key"] == a_k)]
            if not sub_df.empty:
                mape = float(np.mean(np.abs(sub_df["Pred_Physical_Tonnes"] - sub_df["Target_Physical_Tonnes"]) / np.maximum(0.1, sub_df["Target_Physical_Tonnes"]))) * 100.0
                heatmap_matrix[r_idx, c_idx] = mape

    fig, ax = plt.subplots(figsize=(16, 9.0), dpi=150)
    fig.patch.set_facecolor("white")

    im = ax.imshow(heatmap_matrix, cmap="YlOrRd", aspect="auto", vmin=5.0, vmax=60.0)

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cb.set_label("Mean Absolute Percentage Error (MAPE %)", fontsize=9.5, fontweight="bold")
    cb.ax.tick_params(labelsize=8.5)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8.5, fontweight="bold", rotation=30, ha="right")

    y_labels = [f"{gid} ({df[df['Cluster_ID']==gid]['Cluster_Name'].iloc[0][:15]})" for gid in clusters_sorted]
    ax.set_yticks(np.arange(len(clusters_sorted)))
    ax.set_yticklabels(y_labels, fontsize=8.5, fontweight="bold")

    for r in range(heatmap_matrix.shape[0]):
        for c in range(heatmap_matrix.shape[1]):
            val = heatmap_matrix[r, c]
            if not np.isnan(val):
                txt_color = "white" if val > 35.0 else "black"
                ax.text(c, r, f"{val:.1f}%", ha="center", va="center", color=txt_color, fontsize=8.0, fontweight="bold")

    ax.set_title("Figure 6: Cluster-by-Cluster Annual Inversion Error Matrix (MAPE %)\nSpatial Disaggregation of Model Performance Across 10 Benchmark Configurations", fontsize=12.0, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved Figure 6 to: %s", output_path)
    plt.show()

plot_figure6_cluster_error_heatmap(df_master_all, os.path.join(FIGURES_DIR, "Figure6_Cluster_Error_Heatmap.png"))

# =============================================================================
# 12. EXECUTIVE SUMMARY BENCHMARK REPORT
# =============================================================================
logger.info("Generating Peninsula-Wide Executive Summary Diagnostic Report...")

summary_records = []
for tier_info in BENCHMARK_TIERS:
    t_k = tier_info["key"]
    for arch_info in ARCHITECTURES:
        a_k = arch_info["key"]
        sub = df_master_all[(df_master_all["Tier_Key"] == t_k) & (df_master_all["Architecture_Key"] == a_k)]
        if sub.empty:
            continue

        y_t = sub["Target_Physical_Tonnes"].values
        y_p = sub["Pred_Physical_Tonnes"].values

        mae = float(np.mean(np.abs(y_p - y_t)))
        rmse = float(np.sqrt(np.mean((y_p - y_t) ** 2)))
        r = float(np.corrcoef(y_t, y_p)[0, 1]) if len(y_t) > 1 else 0.0
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
        slope = float(np.polyfit(y_t, y_p, 1)[0]) if len(y_t) > 1 else 1.0
        bias_pct = ((np.sum(y_p) - np.sum(y_t)) / max(1.0, np.sum(y_t))) * 100.0

        summary_records.append({
            "Tier": tier_info["name"],
            "Noise_Sigma": f"{tier_info['sigma']:.2f} DU" if not tier_info["is_real"] else "Real L2",
            "Architecture": arch_info["label"],
            "Samples": len(sub),
            "R2_Score": r2,
            "Pearson_r": r,
            "Slope_m": slope,
            "MAE_Tonnes": mae,
            "RMSE_Tonnes": rmse,
            "Regional_Bias_%": bias_pct
        })

df_exec_summary = pd.DataFrame(summary_records)

print("=" * 135)
print("EXECUTIVE BENCHMARK SUMMARY REPORT: 5 SENSOR TIERS x 2 ARCHITECTURES (10 INVERSION MODELS)")
print("=" * 135)
print(df_exec_summary.to_string(index=False))
print("=" * 135)
print(f"All comparative figures successfully rendered and exported to: {FIGURES_DIR}")