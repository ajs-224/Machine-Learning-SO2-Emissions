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
# # Stage 2: Physics-Anchored Dual-Branch Model Testing, Inversion & Diagnostics
# ### Comprehensive evaluation pipeline for the 3-Input Physics-Anchored Regressor:
# ### - Evaluates on geographically withheld validation folds (100% spatial isolation).
# ### - Dynamically feeds 3 Inputs: Spatial Tensor [25, 25, 7] + Met Vector [25] + Log Mass Anchor [1].
# ### - Applies physical inversion: $\widehat{Q}_{\text{tonnes}} = \max\left(0.0, \, 10^{\hat{y}} - 0.1\right)$.
# ### - Evaluates K-model ensemble on chronologically withheld future weather sequences (Double-blind test).
# ### - Supports Real GEMS Level-2 data and all Synthetic OSSE Mission Tiers (`ideal`, `next_gen`, `moderate`, `gems_realistic`, `log_scaled`).

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
import matplotlib.pyplot as plt

# Configure structured logging and global plotting aesthetics
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("DualBranchEvaluator")

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

# =============================================================================
# 1. DIRECTORY PATHS & DATASET CONFIGURATION
# =============================================================================
WORKFLOW_DIR: str = "/home/users/ajs224/sheat_mip_gws/code/WindRotatedMLWorkflow"
MODEL_OUT_DIR: str = os.path.join(WORKFLOW_DIR, "Output_Model")
TOP50_PATH: str = "/home/users/ajs224/sheat_mip_gws/data/ClimateTRACE/top50emitters.csv"

# Dataset directories mapping across all mission tiers
DATASET_DIRECTORIES: Dict[str, str] = {
    "next_gen": os.path.join(WORKFLOW_DIR, "Output_Tensor_Synthetic_NextGen/"),
    "ideal": os.path.join(WORKFLOW_DIR, "Output_Tensor_Synthetic_Ideal/"),
    "moderate": os.path.join(WORKFLOW_DIR, "Output_Tensor_Synthetic_Moderate/"),
    "gems_realistic": os.path.join(WORKFLOW_DIR, "Output_Tensor_Synthetic_GEMSRealistic/"),
    "log_scaled": os.path.join(WORKFLOW_DIR, "Output_Tensor_DualBranch_LogScaled/"),
    "integrated_mass": os.path.join(WORKFLOW_DIR, "Output_Tensor_DualBranch_IntegratedMass/")
}

# Physical Conversion Constants
DU_CONVERSION_FACTOR: float = 2.6867e16  # molecules/cm^2 per Dobson Unit (DU)
KG_PER_DU_PER_M2: float = 2.857e-5      # 1 DU = 2.857e-5 kg SO2 / m^2

# Fallback evaluation tier if training configuration JSON is not found
DEFAULT_EVAL_MODE: str = "moderate"

# %% [markdown]
# ## Section 1: Dynamic Configuration & Automatic Fold Synchronization

# %%
logger.info("Synchronizing spatial fold configuration from Output_Model/...")

config_candidates = sorted(glob.glob(os.path.join(MODEL_OUT_DIR, "adaptive_spatial_folds_*_config.json")))

if config_candidates:
    active_config_path = config_candidates[-1]
    logger.info("Loading training fold configuration from: %s", os.path.basename(active_config_path))
    with open(active_config_path, "r") as f_cfg:
        fold_config = json.load(f_cfg)
        
    EVAL_MODE: str = fold_config.get("mode", DEFAULT_EVAL_MODE).lower().strip()
    actual_k_folds: int = int(fold_config["actual_k_folds"])
    active_clusters: List[str] = fold_config["active_clusters"]
    excluded_clusters: List[str] = fold_config.get("excluded_clusters", [])
    SPATIAL_FOLDS: Dict[int, List[str]] = {int(k): v for k, v in fold_config["spatial_folds"].items()}
else:
    logger.warning("No training configuration JSON found. Using default 13-cluster 3-fold registry.")
    EVAL_MODE = DEFAULT_EVAL_MODE
    actual_k_folds = 3
    active_clusters = ["G01", "G02", "G06", "G07", "G08", "G09", "G10", "G12", "G13", "G14", "G15", "G16", "G17"]
    excluded_clusters = ["G03", "G04", "G05", "G11"]
    SPATIAL_FOLDS = {
        1: ["G01", "G09", "G13", "G14", "G16"],
        2: ["G02", "G07", "G10", "G17"],
        3: ["G06", "G08", "G12", "G15"]
    }

TENSOR_DIR = DATASET_DIRECTORIES.get(EVAL_MODE, DATASET_DIRECTORIES["next_gen"])

logger.info("==========================================================================================")
logger.info("EVALUATION ENGINE INITIALIZED:")
logger.info(" ├─ Inversion Mode:          '%s'", EVAL_MODE.upper())
logger.info(" ├─ Target Tensor Directory: %s", TENSOR_DIR)
logger.info(" ├─ Active Clusters (%d total): %s", len(active_clusters), active_clusters)
logger.info(" └─ Spatial Folds (K=%d):        %s", actual_k_folds, SPATIAL_FOLDS)
logger.info("==========================================================================================")


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


class DualBranchSequence(tf.keras.utils.Sequence):
    """
    Standardized In-Memory Multi-Input Keras Sequence Generator for Physics-Anchored Evaluation.
    Yields strictly 7 physical spatial channels (Shape: [25, 25, 7]) + 25 met features + 1 scalar log mass input.
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
                    
                    # Extract 4 daytime snapshots
                    if raw_spatial.shape[0] >= 4 and raw_spatial.shape[0] <= 7:
                        raw_4d = np.transpose(raw_spatial[:4, :, :], (1, 2, 0))
                    elif raw_spatial.shape[-1] >= 4:
                        raw_4d = raw_spatial[:, :, :4]
                    else:
                        raise ValueError(f"Unexpected spatial shape: {raw_spatial.shape}")
                        
                    met_vec = ds["met_vector"].values.astype(np.float32)  # Shape: (25,)

                    # Extract target emissions in log space
                    if "target_log_emissions" in ds:
                        target_y = float(ds["target_log_emissions"].values)
                    else:
                        raw_kg = float(ds["target_physical_kg"].values)
                        target_y = float(np.log10(raw_kg / 1000.0 + 0.1))

                    # Perform deterministic 15% baseline floor subtraction for evaluation
                    net_snapshots_list = []
                    for d in range(4):
                        snap_d = raw_4d[:, :, d]
                        floor_15_d = float(np.percentile(snap_d, 15))
                        net_snap_d = np.maximum(0.0, snap_d - floor_15_d)
                        net_snapshots_list.append(net_snap_d)

                    # Compute 3 inter-day differential deltas in DU
                    delta_1to2 = net_snapshots_list[1] - net_snapshots_list[0]
                    delta_2to3 = net_snapshots_list[2] - net_snapshots_list[1]
                    delta_3to4 = net_snapshots_list[3] - net_snapshots_list[2]

                    spatial_7ch = np.stack(
                        net_snapshots_list + [delta_1to2, delta_2to3, delta_3to4],
                        axis=-1
                    ).astype(np.float32)

                    # Compute physical grid cell surface area in m^2
                    lats = ds["latitude"].values
                    lons = ds["longitude"].values
                    c_lat = float(np.mean(lats))
                    dx_km = (lons[1] - lons[0]) * 111.32 * np.cos(np.radians(c_lat))
                    dy_km = (lats[1] - lats[0]) * 110.57
                    cell_m2 = float(abs(dx_km * 1000.0 * dy_km * 1000.0))

                    # Physical mass integration & log-scalar anchor
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
                        "target_physical_kg": (
                            float(ds["target_physical_kg"].values)
                            if "target_physical_kg" in ds
                            else 0.0
                        ),
                        "target_physical_tonnes": (
                            (float(ds["target_physical_kg"].values) / 1000.0)
                            if "target_physical_kg" in ds
                            else 0.0
                        ),
                        "cluster_id": ds.attrs.get("cluster_id", "UNKNOWN"),
                        "cluster_name": ds.attrs.get("cluster_name", "Unknown Cluster"),
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

# %% [markdown]
# ## Section 2: Out-of-Sample Validation Loop & Physical Inversion Engine
# ### Evaluates each fold model on withheld clusters using the 3-input physics-anchored architecture:
# ### - Inputs: Spatial Plume Tensor [25, 25, 7] + Meteorological Vector [25] + Log Mass Anchor [1]
# ### - Physical Inversion: $\widehat{Q}_{\text{tonnes}} = \max\left(0.0, \, 10^{\hat{y}} - 0.1\right)$

# %%
logger.info("Scanning compiled NetCDF tensors in %s...", TENSOR_DIR)
all_tensor_files = sorted(glob.glob(os.path.join(TENSOR_DIR, "tensor_G*.nc")))

if not all_tensor_files:
    raise FileNotFoundError(f"No NetCDF tensor files found in: {TENSOR_DIR}. Run compilation script first.")

active_tensor_files = [f for f in all_tensor_files if parse_tensor_metadata(f)[0] in active_clusters]
logger.info("Discovered %d / %d tensors matching active clusters.", len(active_tensor_files), len(all_tensor_files))

# Establish 80% chronological historical cutoff date
all_dates = [parse_tensor_metadata(f)[1] for f in active_tensor_files]
cutoff_date = min(all_dates) + 0.80 * (max(all_dates) - min(all_dates))

val_prediction_records: List[Dict[str, Any]] = []
trained_fold_models: Dict[int, tf.keras.Model] = {}

t_start_val = time.time()

for fold_num in range(1, actual_k_folds + 1):
    val_cluster_ids = SPATIAL_FOLDS[fold_num]
    
    # Auto-locate model checkpoint matching active mode and fold index
    model_candidates = [
        os.path.join(MODEL_OUT_DIR, f"best_plume_regressor_dualbranch_{EVAL_MODE}_{actual_k_folds}folds_fold{fold_num}.keras"),
        os.path.join(MODEL_OUT_DIR, f"best_plume_regressor_dualbranch_*_{actual_k_folds}folds_fold{fold_num}.keras"),
        os.path.join(MODEL_OUT_DIR, f"best_plume_regressor_dualbranch_*_fold{fold_num}.keras"),
        os.path.join(MODEL_OUT_DIR, f"*dualbranch*_fold{fold_num}.keras")
    ]
    
    selected_model_path = None
    for pattern in model_candidates:
        matches = sorted(glob.glob(pattern))
        if matches:
            preferred = [m for m in matches if EVAL_MODE in os.path.basename(m).lower()]
            selected_model_path = preferred[0] if preferred else matches[0]
            break

    if selected_model_path is None or not os.path.exists(selected_model_path):
        logger.error("Missing model checkpoint for Fold %d. Skipping fold.", fold_num)
        continue

    logger.info("--- EVALUATING FOLD %d/%d (Loaded: %s) ---", fold_num, actual_k_folds, os.path.basename(selected_model_path))
    
    # Load model natively (100% native Keras layers)
    fold_model = tf.keras.models.load_model(selected_model_path, compile=False)
    trained_fold_models[fold_num] = fold_model

    # Isolate out-of-sample validation tensors (Withheld clusters, Historical weather <= cutoff)
    fold_val_files = [
        f for f in active_tensor_files 
        if (parse_tensor_metadata(f)[0] in val_cluster_ids and parse_tensor_metadata(f)[1] <= cutoff_date)
    ]
    
    logger.info("Fold %d Out-of-Sample Validation Files: %d samples across Clusters %s", fold_num, len(fold_val_files), val_cluster_ids)
    if not fold_val_files:
        continue

    val_seq = DualBranchSequence(file_list=fold_val_files, batch_size=32)

    for b_idx in range(len(val_seq)):
        # Unpack 3 Inputs: (Spatial_7ch, Met_25, Log_Mass_1) and Target Label
        (bx_spatial, bx_met, bx_log_m), by_true = val_seq[b_idx]

        # Multi-modal forward pass: dynamically supports 3-input skip and legacy 2-input models
        if len(fold_model.inputs) == 3:
            by_pred = fold_model.predict_on_batch([bx_spatial, bx_met, bx_log_m]).flatten()
        else:
            by_pred = fold_model.predict_on_batch([bx_spatial, bx_met]).flatten()

        for s_in_batch in range(len(by_true)):
            global_sample_idx = b_idx * 32 + s_in_batch
            meta = val_seq.cached_meta[global_sample_idx]
            
            y_t = float(by_true[s_in_batch])
            y_p = float(by_pred[s_in_batch])
            true_phys_tonnes = float(meta["target_physical_tonnes"])
            true_phys_kg = float(meta["target_physical_kg"])

            # Physical Inversion: Q_pred = 10^y_p - 0.1 metric tonnes
            pred_phys_tonnes = max(0.0, float(10.0 ** y_p - 0.1))
            pred_phys_kg = pred_phys_tonnes * 1000.0

            val_prediction_records.append({
                "Fold": fold_num,
                "Cluster_ID": meta["cluster_id"],
                "Cluster_Name": meta["cluster_name"],
                "Sector": meta["primary_sector"],
                "Start_Date": meta["sequence_start_date"],
                "End_Date": meta["sequence_end_date"],
                "Filename": meta["filename"],
                "File_Path": meta["filepath"],
                "Target_Native": y_t,
                "Pred_Native": y_p,
                "Target_Physical_Tonnes": true_phys_tonnes,
                "Pred_Physical_Tonnes": pred_phys_tonnes,
                "Target_Physical_kg": true_phys_kg,
                "Pred_Physical_kg": pred_phys_kg,
                "Residual_Tonnes": pred_phys_tonnes - true_phys_tonnes
            })

df_val_all = pd.DataFrame(val_prediction_records)
t_val_total = time.time() - t_start_val
logger.info("Out-of-sample evaluation complete: %d total validation samples processed in %.2fs.", len(df_val_all), t_val_total)

# %% [markdown]
# ## Section 3: Single-Sample Physical Diagnostic & Inversion Verification
# ### Displays 7 physical channels (4 Snapshots + 3 Deltas), 1D transects, and physical reconstruction.

# %%
if not df_val_all.empty:
    sample_row = df_val_all.sample(n=1, random_state=int(time.time() % 10000)).iloc[0]
    sample_nc_path = sample_row["File_Path"]

    with xr.open_dataset(sample_nc_path) as ds_sample:
        raw_sp = ds_sample["spatial_plume_tensor"].values.astype(np.float32)
        
        # Extract raw 4 daily snapshots (Channels 0..3)
        if raw_sp.shape[0] >= 4 and raw_sp.shape[0] <= 7:
            raw_4d_s = np.transpose(raw_sp[:4, :, :], (1, 2, 0))
        elif raw_sp.shape[-1] >= 4:
            raw_4d_s = raw_sp[:, :, :4]
        else:
            raw_4d_s = raw_sp
            
        met_v = ds_sample["met_vector"].values.astype(np.float32)  # Shape: (25,)
        lats_s = ds_sample["latitude"].values
        lons_s = ds_sample["longitude"].values
        c_lat_s = float(np.mean(lats_s))
        c_lon_s = float(np.mean(lons_s))
        dx_s_km = (lons_s - c_lon_s) * 111.32 * np.cos(np.radians(c_lat_s))
        dy_s_km = (lats_s - c_lat_s) * 110.57
        cell_m2_s = float(abs((dx_s_km[1] - dx_s_km[0]) * 1000.0 * (dy_s_km[1] - dy_s_km[0]) * 1000.0))

        # Dynamic 15% baseline floor subtraction for single-sample evaluation
        net_snapshots_s = []
        for d in range(4):
            snap_d = raw_4d_s[:, :, d]
            floor_15_d = float(np.percentile(snap_d, 15))
            net_snap_d = np.maximum(0.0, snap_d - floor_15_d)
            net_snapshots_s.append(net_snap_d)

        delta_1to2_s = net_snapshots_s[1] - net_snapshots_s[0]
        delta_2to3_s = net_snapshots_s[2] - net_snapshots_s[1]
        delta_3to4_s = net_snapshots_s[3] - net_snapshots_s[2]

        sp_tensor = np.stack(
            net_snapshots_s + [delta_1to2_s, delta_2to3_s, delta_3to4_s], axis=-1
        ).astype(np.float32)  # Shape: (25, 25, 7)

        mean_net_vcd_s = np.mean(np.stack(net_snapshots_s, axis=0), axis=0)
        m_int_kg_s = float(np.sum(mean_net_vcd_s) * cell_m2_s * 2.857e-5)
        m_int_val = max(0.001, m_int_kg_s / 1000.0)

    # 1. Extract 3rd Input (Log Plume Mass Anchor)
    log_m_input_val = np.array([[np.log10(m_int_val + 0.1)]], dtype=np.float32)  # Shape: (1, 1)

    # 2. Multi-Input Model Forward Pass (Handles 3-Input Physics Skip & 2-Input Models Dynamically)
    sample_fold_model = trained_fold_models[int(sample_row["Fold"])]

    if len(sample_fold_model.inputs) == 3:
        pred_native_val = float(
            sample_fold_model.predict_on_batch([
                sp_tensor[np.newaxis, ...],
                met_v[np.newaxis, ...],
                log_m_input_val,
            ])[0, 0]
        )
    else:
        pred_native_val = float(
            sample_fold_model.predict_on_batch([
                sp_tensor[np.newaxis, ...],
                met_v[np.newaxis, ...],
            ])[0, 0]
        )

    recon_tonnes_val = max(0.0, float(10.0 ** pred_native_val - 0.1))
    true_tonnes_val = sample_row["Target_Physical_Tonnes"]
    err_t = abs(recon_tonnes_val - true_tonnes_val)

    # 3. Render 3-Row Diagnostic Dashboard
    fig_diag = plt.figure(figsize=(22, 13), dpi=150)
    fig_diag.patch.set_facecolor("white")
    gs_d = fig_diag.add_gridspec(3, 4, height_ratios=[1.2, 0.9, 1.1], hspace=0.38, wspace=0.25)

    # --- ROW 1: 4 DAILY SNAPSHOTS ---
    vmax_sp = float(np.percentile(sp_tensor[:, :, :4], 99.5)) if np.any(sp_tensor[:, :, :4] > 0) else 0.50
    vmax_sp = max(0.40, vmax_sp)
    
    for i in range(4):
        ax = fig_diag.add_subplot(gs_d[0, i])
        ax.set_facecolor("#1a1a1a")
        im = ax.imshow(
            sp_tensor[:, :, i], origin="lower", cmap="inferno",
            extent=[dx_s_km.min(), dx_s_km.max(), dy_s_km.min(), dy_s_km.max()],
            vmin=0.0, vmax=vmax_sp,
        )
        cb = fig_diag.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Plume VCD [DU]", fontsize=7.5, fontweight="bold")
        ax.plot(0, 0, marker="*", color="cyan", markersize=12, markeredgecolor="white", label="Origin" if i == 0 else "")
        ax.annotate("Downwind ->", xy=(dx_s_km.max() * 0.65, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="cyan", lw=1.2), fontsize=6.5, color="cyan", ha="center")
        ax.set_title(f"Channel {i}: Snapshot $\\Omega_{{{i+1}}}$ (Day {i+1})", fontsize=9.0, fontweight="bold")
        ax.set_xlabel("Along-Wind [km]", fontsize=7.5)
        ax.set_ylabel("Cross-Wind [km]" if i == 0 else "", fontsize=7.5)
        ax.tick_params(labelsize=7)
        if i == 0: ax.legend(loc="upper right", fontsize=6.5)

    # --- ROW 2: 3 DELTA GRIDS & 1D TRANSECT ---
    vmax_d = float(np.percentile(np.abs(sp_tensor[:, :, 4:7]), 99.0)) if np.any(sp_tensor[:, :, 4:7] != 0) else 0.08
    vmax_d = max(0.08, vmax_d)
    
    for i in range(3):
        ax_dg = fig_diag.add_subplot(gs_d[1, i])
        ax_dg.set_facecolor("#222222")
        im_dg = ax_dg.imshow(
            sp_tensor[:, :, 4 + i], origin="lower", cmap="bwr",
            extent=[dx_s_km.min(), dx_s_km.max(), dy_s_km.min(), dy_s_km.max()],
            vmin=-vmax_d, vmax=vmax_d,
        )
        cb_dg = fig_diag.colorbar(im_dg, ax=ax_dg, fraction=0.046, pad=0.04)
        cb_dg.set_label("$\\Delta\\Omega$ Change [DU]", fontsize=7.5, fontweight="bold")
        ax_dg.plot(0, 0, marker="*", color="black", markersize=11, markeredgecolor="white")

        u_night_r = met_v[16 + i * 2] * 20.0
        v_night_r = met_v[17 + i * 2] * 20.0
        sp_night = np.sqrt(u_night_r**2 + v_night_r**2)
        ax_dg.quiver(0, 0, u_night_r / (sp_night + 1e-4), v_night_r / (sp_night + 1e-4), color="lime", scale=4.5, width=0.012, headwidth=4.5, zorder=5)
        ax_dg.set_title(f"Channel {4+i}: Delta $\\Delta\\Omega_{{{i+1}\\to {i+2}}}$\n(Overnight: {sp_night:.1f} m/s)", fontsize=9.0, fontweight="bold")
        ax_dg.set_xlabel("Along-Wind [km]", fontsize=7.5)
        ax_dg.set_ylabel("Cross-Wind [km]" if i == 0 else "", fontsize=7.5)
        ax_dg.tick_params(labelsize=7)

    # 1D Transect in 4th column of Row 2
    ax_tr = fig_diag.add_subplot(gs_d[1, 3])
    ax_tr.set_facecolor("#f9f9f9")
    ax_tr.grid(True, linestyle="--", alpha=0.5, color="gray")
    cy_s = len(lats_s) // 2
    colors_s = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i in range(4):
        transect = np.nanmean(sp_tensor[cy_s - 1 : cy_s + 2, :, i], axis=0)
        ax_tr.plot(dx_s_km, transect, color=colors_s[i], lw=1.8, label=f"Day {i+1}")
    ax_tr.axvline(0, color="black", linestyle=":", lw=1.2)
    ax_tr.set_title("1D Along-Plume Transects ($y=0$)", fontsize=9.0, fontweight="bold")
    ax_tr.set_xlabel("Along-Wind [km]", fontsize=7.5)
    ax_tr.legend(loc="upper right", fontsize=6.5)
    ax_tr.tick_params(labelsize=7)

    # --- ROW 3: 25-METEOROLOGY VECTOR & PHYSICAL RECONSTRUCTION BOX ---
    ax_met = fig_diag.add_subplot(gs_d[2, :])
    ax_met.set_facecolor("#fcfcfc")
    ax_met.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=1)

    feature_labels = (
        [f"w_day{i+1}" for i in range(4)]
        + [f"T2m_day{i+1}" for i in range(4)]
        + [f"Rad_day{i+1}" for i in range(4)]
        + [f"PBLH_day{i+1}" for i in range(4)]
        + [f"u_night_{i+1}->{i+2}" for i in range(3)]
        + [f"v_night_{i+1}->{i+2}" for i in range(3)]
        + [f"Δθ_{i+1}->{i+2}" for i in range(3)]
    )

    feat_x = np.arange(len(met_v))
    ax_met.bar(feat_x, met_v, color="#2980b9", edgecolor="black", linewidth=0.7, zorder=3)
    ax_met.set_xticks(feat_x)
    ax_met.set_xticklabels(feature_labels, rotation=45, ha="right", fontsize=8.0, fontweight="bold")
    ax_met.set_ylabel("Normalized Magnitude", fontsize=8.5, fontweight="bold")

    ax_met.text(
        0.015, 0.72,
        f"PHYSICS-ANCHORED MULTI-MODAL INVERSION CHECK:\n"
        f" ├─ Model Output log10(Q):    y_true = {sample_row['Target_Native']:.4f} | y_pred = {pred_native_val:.4f}\n"
        f" ├─ Integrated Mass Anchor:   M_plume = {m_int_val:.2f} tonnes (log10(M) = {np.log10(m_int_val+0.1):.3f})\n"
        f" ├─ Actual CleanSYS SO2:      Q_true  = {true_tonnes_val:.2f} metric tonnes\n"
        f" └─ Inverted Model SO2:       Q_pred  = {recon_tonnes_val:.2f} metric tonnes (Error: {err_t:+.2f} t)",
        transform=ax_met.transAxes, fontsize=8.0, fontweight="bold", color="#145a32",
        bbox=dict(boxstyle="round,pad=0.35", fc="#eafaf1", ec="#27ae60", lw=1.2, alpha=0.95),
    )
    ax_met.set_title(
        f"1D Meteorological Vector (Branch B: {len(met_v)} Features) | Sample: {sample_row['Filename']}",
        fontsize=10.0, fontweight="bold", pad=8,
    )

    plt.suptitle(
        f"Single-Sample Multi-Modal Inversion Diagnostic: {sample_row['Cluster_ID']} ({sample_row['Cluster_Name']})\n"
        f"Sequence: {sample_row['Start_Date']} to {sample_row['End_Date']} | Evaluated on Out-of-Sample Fold {sample_row['Fold']}",
        fontsize=12.0, fontweight="bold", y=0.995,
    )
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Section 4: Peninsula-Wide Bulk Statistics & 1:1 Conservation Dashboards
# ### Compares out-of-sample predictions across all clusters in Physical Emissions Space and Model-Native Space.

# %%
logger.info("Computing Peninsula-wide out-of-sample validation metrics (%s)...", EVAL_MODE.upper())

true_tonnes = df_val_all["Target_Physical_Tonnes"].values
pred_tonnes = df_val_all["Pred_Physical_Tonnes"].values

true_native = df_val_all["Target_Native"].values
pred_native = df_val_all["Pred_Native"].values

# Physical Space Metrics
total_true_t = float(np.sum(true_tonnes))
total_pred_t = float(np.sum(pred_tonnes))
bias_phys_pct = ((total_pred_t - total_true_t) / (total_true_t + 1e-8)) * 100.0
mae_phys_t = float(np.mean(np.abs(true_tonnes - pred_tonnes)))
rmse_phys_t = float(np.sqrt(np.mean((true_tonnes - pred_tonnes) ** 2)))
r_phys = float(np.corrcoef(true_tonnes, pred_tonnes)[0, 1]) if len(true_tonnes) > 1 else 0.0
ss_res_p = np.sum((true_tonnes - pred_tonnes) ** 2)
ss_tot_p = np.sum((true_tonnes - np.mean(true_tonnes)) ** 2)
r2_phys = float(1.0 - (ss_res_p / (ss_tot_p + 1e-8)))

# Model-Native Space Metrics
mae_nat = float(np.mean(np.abs(true_native - pred_native)))
r_nat = float(np.corrcoef(true_native, pred_native)[0, 1]) if len(true_native) > 1 else 0.0
ss_res_n = np.sum((true_native - pred_native) ** 2)
ss_tot_n = np.sum((true_native - np.mean(true_native)) ** 2)
r2_nat = float(1.0 - (ss_res_n / (ss_tot_n + 1e-8)))

# Render Dual-Panel Dashboard
fig_bulk, axes = plt.subplots(1, 2, figsize=(16, 7.5), dpi=150)
fig_bulk.patch.set_facecolor("white")

# 1. Physical Emissions Space
ax1 = axes[0]
ax1.set_facecolor("#f9f9f9")
ax1.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=1)

ax1.scatter(true_tonnes, pred_tonnes, color="navy", alpha=0.55, edgecolors="white", linewidths=0.4, s=35, label="Out-of-Sample Validation Windows", zorder=3)
max_t_val = max(true_tonnes.max(), pred_tonnes.max(), 1.0)
ax1.plot([0, max_t_val], [0, max_t_val], color="red", linestyle="--", linewidth=1.5, label="1:1 Perfect Conservation Line", zorder=2)

if len(true_tonnes) > 1:
    sl_p, int_p = np.polyfit(true_tonnes, pred_tonnes, 1)
    x_tr_p = np.array([0.0, max_t_val])
    ax1.plot(x_tr_p, sl_p * x_tr_p + int_p, color="purple", linestyle="-", linewidth=1.5, label=f"Fitted Trend (Slope: {sl_p:.2f}, Intercept: {int_p:.1f} t)", zorder=2)

ax1.set_title("A. Physical Emissions Space (Metric Tonnes / Sequence)\nReconstructed via Physical Inversion", fontsize=11, fontweight="bold")
ax1.set_xlabel("CleanSYS Ground Truth SO2 [metric tonnes / Sequence]", fontsize=9.5)
ax1.set_ylabel("Inverted PlumeRegressor SO2 [metric tonnes / Sequence]", fontsize=9.5)
ax1.set_xlim(-0.03 * max_t_val, max_t_val * 1.05)
ax1.set_ylim(-0.03 * max_t_val, max_t_val * 1.05)
ax1.tick_params(labelsize=8.5)
ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.85)

# 2. Model-Native Space
ax2 = axes[1]
ax2.set_facecolor("#f9f9f9")
ax2.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=1)

ax2.scatter(true_native, pred_native, color="darkorange", alpha=0.55, edgecolors="white", linewidths=0.4, s=35, label="Out-of-Sample Validation Windows", zorder=3)
max_n_val = max(true_native.max(), pred_native.max(), 1.0)
ax2.plot([0, max_n_val], [0, max_n_val], color="red", linestyle="--", linewidth=1.5, label="1:1 Reference Line", zorder=2)

if len(true_native) > 1:
    sl_n, int_n = np.polyfit(true_native, pred_native, 1)
    x_tr_n = np.array([0.0, max_n_val])
    ax2.plot(x_tr_n, sl_n * x_tr_n + int_n, color="purple", linestyle="-", linewidth=1.5, label=f"Fitted Trend (Slope: {sl_n:.2f})", zorder=2)

ax2.set_title("B. Model-Native Space ($\\log_{10}(Q_{\\mathrm{tonnes}} + 0.1)$)\nOptimization Objective Distribution", fontsize=11, fontweight="bold")
ax2.set_xlabel("True Target ($\\log_{10}(Q_{\\mathrm{tonnes}} + 0.1)$)", fontsize=9.5)
ax2.set_ylabel("Predicted Output ($\\log_{10}(Q_{\\mathrm{tonnes}} + 0.1)$)", fontsize=9.5)
ax2.set_xlim(-0.03 * max_n_val, max_n_val * 1.05)
ax2.set_ylim(-0.03 * max_n_val, max_n_val * 1.05)
ax2.tick_params(labelsize=8.5)
ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.85)

plt.suptitle(
    f"Dual-Branch Spatial Cross-Validation Performance Summary ({EVAL_MODE.upper()})\n"
    f"Evaluated on {len(df_val_all):,} Out-of-Sample Sequences across All {len(active_clusters)} Active Clusters",
    fontsize=12.0, fontweight="bold", y=0.99
)
plt.tight_layout()
plt.show()

print("=" * 95)
print(f"PENINSULA-WIDE BULK EMISSIONS INVERSION REPORT (MODE: {EVAL_MODE.upper()}):")
print("=" * 95)
print(f" Total Out-of-Sample Validation Windows:          {len(df_val_all):,}")
print(f" Total Actual Bulk Mass Emitted:                 {total_true_t:,.1f} metric tonnes")
print(f" Total Inverted Predicted Mass:                  {total_pred_t:,.1f} metric tonnes")
print(f" Systematic Regional Budget Bias:                {bias_phys_pct:+.2f}%")
print(f" Bulk Physical Mean Absolute Error (MAE):         {mae_phys_t:.2f} metric tonnes / sequence")
print(f" Bulk Physical Root Mean Squared Error (RMSE):    {rmse_phys_t:.2f} metric tonnes / sequence")
print(f" Pearson Correlation Coefficient (r) - Physical:  {r_phys:.4f}")
print(f" Coefficient of Determination (R2) - Physical:    {r2_phys:.4f}")
print(f" Model-Native Space MAE:                         {mae_nat:.4f}")
print(f" Model-Native Space R2 Score:                    {r2_nat:.4f}")
print("=" * 95)

# %% [markdown]
# ## Section 5: Cluster-by-Cluster Multi-Year Mean Emissions Audit

# %%
logger.info("Computing cluster-by-cluster multi-year mean emission rate comparisons...")

cluster_records = []
for gid, group in df_val_all.groupby("Cluster_ID"):
    c_name = group["Cluster_Name"].iloc[0]
    c_sec = group["Sector"].iloc[0]
    fold_n = group["Fold"].iloc[0]

    m_true = float(group["Target_Physical_Tonnes"].mean())
    m_pred = float(group["Pred_Physical_Tonnes"].mean())
    c_mae = float(np.mean(np.abs(group["Target_Physical_Tonnes"] - group["Pred_Physical_Tonnes"])))
    c_r = float(np.corrcoef(group["Target_Physical_Tonnes"], group["Pred_Physical_Tonnes"])[0, 1]) if len(group) > 1 else 0.0

    cluster_records.append({
        "Cluster_ID": gid,
        "Cluster_Name": c_name,
        "Sector": c_sec,
        "Fold": fold_n,
        "Sample_Count": len(group),
        "Mean_True_Tonnes": m_true,
        "Mean_Pred_Tonnes": m_pred,
        "MAE_Tonnes": c_mae,
        "Pearson_r": c_r
    })

df_clusters_summary = pd.DataFrame(cluster_records).sort_values(by="Mean_True_Tonnes", ascending=False)

fig_c, ax_c = plt.subplots(figsize=(10, 8.5), dpi=150)
ax_c.set_facecolor("#f9f9f9")
ax_c.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=1)

max_c_val = max(df_clusters_summary["Mean_True_Tonnes"].max(), df_clusters_summary["Mean_Pred_Tonnes"].max(), 1.0)
ax_c.plot([0, max_c_val], [0, max_c_val], color="red", linestyle="--", linewidth=1.5, label="1:1 Perfect Agreement", zorder=2)

scatter_c = ax_c.scatter(
    df_clusters_summary["Mean_True_Tonnes"], df_clusters_summary["Mean_Pred_Tonnes"],
    c=df_clusters_summary["Fold"], cmap="tab10", edgecolors="black", linewidths=0.8, s=90, zorder=4
)

for idx, row in df_clusters_summary.iterrows():
    ax_c.annotate(
        f"{row['Cluster_ID']}: {row['Cluster_Name'][:14]}",
        (row["Mean_True_Tonnes"], row["Mean_Pred_Tonnes"]),
        fontsize=7.5, fontweight="bold", xytext=(4, 4), textcoords="offset points", zorder=5
    )

ax_c.set_title(f"Cluster-by-Cluster Multi-Year Mean Emissions Audit ({EVAL_MODE.upper()})\nOut-of-Sample Predictions (Color-Coded by Validation Fold)", fontsize=11, fontweight="bold", pad=10)
ax_c.set_xlabel("CleanSYS Multi-Year Mean [metric tonnes / Sequence]", fontsize=9.5)
ax_c.set_ylabel("Inverted PlumeRegressor Mean [metric tonnes / Sequence]", fontsize=9.5)
ax_c.set_xlim(0, 150)
ax_c.set_ylim(0, 150)
ax_c.tick_params(labelsize=8.5)
ax_c.legend(loc="upper left", fontsize=8.5, framealpha=0.85)

plt.tight_layout()
plt.show()

print("=" * 110)
print(f"CLUSTER-LEVEL MULTI-YEAR PERFORMANCE SUMMARY ({EVAL_MODE.upper()}):")
print("=" * 110)
print(df_clusters_summary[["Cluster_ID", "Cluster_Name", "Fold", "Sample_Count", "Mean_True_Tonnes", "Mean_Pred_Tonnes", "MAE_Tonnes", "Pearson_r"]].to_string(index=False))
print("=" * 110)

# %% [markdown]
# ## Section 6: Sector-Specific Performance Breakdown

# %%
logger.info("Computing sector-specific validation metrics...")

sector_records = []
for sector_name, group in df_val_all.groupby("Sector"):
    t_s = group["Target_Physical_Tonnes"].values
    p_s = group["Pred_Physical_Tonnes"].values
    if len(t_s) < 3: continue

    s_true_tot = float(np.sum(t_s))
    s_pred_tot = float(np.sum(p_s))
    s_bias = ((s_pred_tot - s_true_tot) / (s_true_tot + 1e-8)) * 100.0
    s_mae = float(np.mean(np.abs(t_s - p_s)))
    s_r = float(np.corrcoef(t_s, p_s)[0, 1]) if len(t_s) > 1 else 0.0

    ss_r = np.sum((t_s - p_s) ** 2)
    ss_t = np.sum((t_s - np.mean(t_s)) ** 2)
    s_r2 = float(1.0 - (ss_r / (ss_t + 1e-8)))

    sector_records.append({
        "Sector": sector_name,
        "Cluster_Count": len(group["Cluster_ID"].unique()),
        "Sample_Count": len(group),
        "Total_True_Tonnes": s_true_tot,
        "Total_Pred_Tonnes": s_pred_tot,
        "Systematic_Bias_Pct": s_bias,
        "MAE_Tonnes": s_mae,
        "Pearson_r": s_r,
        "R2_Score": s_r2
    })

df_sectors_summary = pd.DataFrame(sector_records).sort_values(by="Total_True_Tonnes", ascending=False)

print("=" * 110)
print(f"SECTOR-SPECIFIC PHYSICAL INVERSION BREAKDOWN ({EVAL_MODE.upper()}):")
print("=" * 110)
print(df_sectors_summary[["Sector", "Cluster_Count", "Sample_Count", "Total_True_Tonnes", "Systematic_Bias_Pct", "MAE_Tonnes", "Pearson_r", "R2_Score"]].to_string(index=False))
print("=" * 110)

# %% [markdown]
# ## Section 7: K-Model Ensemble Future-Weather Test Evaluation (Double-Blind Test)

# %%
logger.info("Evaluating K-Model Ensemble across Future Meteorological Test Set...")

test_files = [f for f in active_tensor_files if parse_tensor_metadata(f)[1] > cutoff_date]
logger.info("Discovered %d future weather test samples (Windows > %s).", len(test_files), cutoff_date.strftime('%Y-%m-%d'))

test_records: List[Dict[str, Any]] = []

if test_files:
    test_seq = DualBranchSequence(file_list=test_files, batch_size=32)

    for b_idx in range(len(test_seq)):
        # Unpack 3 Inputs: (Spatial_7ch, Met_25, Log_Mass_1) and Target Label
        (bx_spatial, bx_met, bx_log_m), by_true = test_seq[b_idx]

        k_preds = []
        for fold_num in range(1, actual_k_folds + 1):
            if fold_num in trained_fold_models:
                m = trained_fold_models[fold_num]
                if len(m.inputs) == 3:
                    m_p = m.predict_on_batch([bx_spatial, bx_met, bx_log_m]).flatten()
                else:
                    m_p = m.predict_on_batch([bx_spatial, bx_met]).flatten()
                k_preds.append(m_p)

        ens_pred_native = np.mean(np.array(k_preds), axis=0)

        for s_in_batch in range(len(by_true)):
            global_idx = b_idx * 32 + s_in_batch
            meta = test_seq.cached_meta[global_idx]

            y_pred_n = float(ens_pred_native[s_in_batch])
            y_true_n = float(by_true[s_in_batch])
            true_phys_tonnes = float(meta["target_physical_tonnes"])
            pred_phys_tonnes = max(0.0, float(10.0 ** y_pred_n - 0.1))

            test_records.append({
                "Cluster_ID": meta["cluster_id"],
                "Cluster_Name": meta["cluster_name"],
                "Sector": meta["primary_sector"],
                "Start_Date": meta["sequence_start_date"],
                "End_Date": meta["sequence_end_date"],
                "Target_Native": y_true_n,
                "Pred_Native": y_pred_n,
                "Target_Physical_Tonnes": true_phys_tonnes,
                "Pred_Physical_Tonnes": pred_phys_tonnes,
            })

    df_test_all = pd.DataFrame(test_records)

    t_true_t = df_test_all["Target_Physical_Tonnes"].values
    t_pred_t = df_test_all["Pred_Physical_Tonnes"].values

    t_tot_true = float(np.sum(t_true_t))
    t_tot_pred = float(np.sum(t_pred_t))
    t_bias_pct = ((t_tot_pred - t_tot_true) / (t_tot_true + 1e-8)) * 100.0
    t_mae_t = float(np.mean(np.abs(t_true_t - t_pred_t)))
    t_r = float(np.corrcoef(t_true_t, t_pred_t)[0, 1]) if len(t_true_t) > 1 else 0.0

    ss_r_test = np.sum((t_true_t - t_pred_t) ** 2)
    ss_t_test = np.sum((t_true_t - np.mean(t_true_t)) ** 2)
    t_r2 = float(1.0 - (ss_r_test / (ss_t_test + 1e-8)))

    # Render Future Test Scatter Plot
    fig_test, ax_test = plt.subplots(figsize=(8, 7), dpi=150)
    ax_test.set_facecolor("#f9f9f9")
    ax_test.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=1)

    max_t_test = max(t_true_t.max(), t_pred_t.max(), 1.0)
    ax_test.scatter(
        t_true_t, t_pred_t, color="darkorange", alpha=0.55, edgecolors="white", linewidths=0.4, s=40,
        label="Future Weather Test Sequences", zorder=3,
    )
    ax_test.plot([0, max_t_test], [0, max_t_test], color="red", linestyle="--", linewidth=1.5, label="1:1 Perfect Conservation Line", zorder=2)

    if len(t_true_t) > 1:
        sl_test, int_test = np.polyfit(t_true_t, t_pred_t, 1)
        x_tr_test = np.array([0.0, max_t_test])
        ax_test.plot(x_tr_test, sl_test * x_tr_test + int_test, color="purple", linewidth=1.5, label=f"Trend (Slope: {sl_test:.2f}, Intercept: {int_test:.1f} t)", zorder=2)

    ax_test.set_title(
        f"K-Model Ensemble Future Weather Test Evaluation ({EVAL_MODE.upper()})\n(Chronologically Withheld Sequences > {cutoff_date.strftime('%Y-%m-%d')})",
        fontsize=11, fontweight="bold"
    )
    ax_test.set_xlabel("CleanSYS Ground Truth SO2 [metric tonnes / Sequence]", fontsize=9.5)
    ax_test.set_ylabel("Inverted PlumeRegressor SO2 [metric tonnes / Sequence]", fontsize=9.5)
    ax_test.set_xlim(-0.03 * max_t_test, max_t_test * 1.05)
    ax_test.set_ylim(-0.03 * max_t_test, max_t_test * 1.05)
    ax_test.legend(loc="upper left", fontsize=8.5, framealpha=0.85)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Section 8: Multi-Fold Training & Validation Convergence History Dashboard

# %%
logger.info("Discovering and ingesting multi-fold training history logs from Output_Model/...")

FOLD_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

fold_histories: Dict[int, pd.DataFrame] = {}
fold_best_metrics: List[Dict[str, Any]] = []

for fold_num in range(1, actual_k_folds + 1):
    hist_candidates = [
        os.path.join(MODEL_OUT_DIR, f"training_history_dualbranch_{EVAL_MODE}_{actual_k_folds}folds_fold{fold_num}.csv"),
        os.path.join(MODEL_OUT_DIR, f"training_history_dualbranch_*_{actual_k_folds}folds_fold{fold_num}.csv"),
        os.path.join(MODEL_OUT_DIR, f"training_history_dualbranch_*_fold{fold_num}.csv"),
        os.path.join(MODEL_OUT_DIR, f"*dualbranch*_fold{fold_num}.csv")
    ]
    
    selected_csv_path = None
    for pattern in hist_candidates:
        matches = sorted(glob.glob(pattern))
        if matches:
            preferred = [m for m in matches if EVAL_MODE in os.path.basename(m).lower()]
            selected_csv_path = preferred[0] if preferred else matches[0]
            break

    if selected_csv_path is None or not os.path.exists(selected_csv_path):
        logger.warning("History CSV not found for Fold %d. Skipping fold.", fold_num)
        continue

    try:
        df_h = pd.read_csv(selected_csv_path)
        fold_histories[fold_num] = df_h
        
        if "val_loss" in df_h.columns and not df_h["val_loss"].isnull().all():
            best_idx = int(df_h["val_loss"].idxmin())
            best_val_loss = float(df_h["val_loss"].iloc[best_idx])
            best_train_loss = float(df_h["loss"].iloc[best_idx])
            best_val_mae = float(df_h["val_mae"].iloc[best_idx]) if "val_mae" in df_h.columns else np.nan
        else:
            best_idx = int(df_h["loss"].idxmin())
            best_val_loss = np.nan
            best_train_loss = float(df_h["loss"].iloc[best_idx])
            best_val_mae = np.nan
            
        best_epoch_num = best_idx + 1
        gen_ratio = (best_val_loss / (best_train_loss + 1e-8)) if not np.isnan(best_val_loss) else np.nan
        val_clusters_str = ", ".join(SPATIAL_FOLDS.get(fold_num, []))

        fold_best_metrics.append({
            "Fold": fold_num,
            "Withheld_Clusters": val_clusters_str,
            "Total_Epochs": len(df_h),
            "Best_Epoch": best_epoch_num,
            "Best_Train_Loss": best_train_loss,
            "Best_Val_Loss": best_val_loss,
            "Best_Val_MAE": best_val_mae,
            "Generalization_Ratio": gen_ratio
        })
        
    except Exception as e_csv:
        logger.error("Failed to read CSV log %s: %s", selected_csv_path, e_csv)

if fold_histories:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), dpi=150)
    fig.patch.set_facecolor("white")

    ax1 = axes[0]
    ax1.set_facecolor("#fcfcfc")
    ax1.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=1)

    max_epoch_seen = 1

    for fold_num, df_h in fold_histories.items():
        color_f = FOLD_PALETTE[(fold_num - 1) % len(FOLD_PALETTE)]
        epochs_arr = np.arange(1, len(df_h) + 1)
        max_epoch_seen = max(max_epoch_seen, len(df_h))

        ax1.plot(epochs_arr, df_h["loss"], color=color_f, linewidth=1.2, alpha=0.45, linestyle="-", label=f"Fold {fold_num} Train" if fold_num <= 5 else "")

        if "val_loss" in df_h.columns:
            ax1.plot(epochs_arr, df_h["val_loss"], color=color_f, linewidth=2.0, alpha=0.90, linestyle="--", label=f"Fold {fold_num} Val" if fold_num <= 5 else "")
            best_idx = int(df_h["val_loss"].idxmin())
            ax1.scatter(best_idx + 1, df_h["val_loss"].iloc[best_idx], color=color_f, s=110, marker="*", edgecolors="black", linewidths=0.8, zorder=5)

    ax1.set_title(f"A. Multi-Fold Loss Convergence ({EVAL_MODE.upper()}, K={actual_k_folds} Folds)\n[Solid = Train | Dashed = Val (Unseen Withheld Clusters) | Stars = Best Epoch]", fontsize=10.5, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch Number", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Loss Value (MSE Metric)", fontsize=9.5, fontweight="bold")
    ax1.set_xlim(1, max_epoch_seen)
    ax1.tick_params(labelsize=8.5)
    ax1.legend(loc="upper right", fontsize=8.0, ncol=2, framealpha=0.90)

    ax2 = axes[1]
    ax2.set_facecolor("#fcfcfc")
    ax2.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=1)

    for fold_num, df_h in fold_histories.items():
        color_f = FOLD_PALETTE[(fold_num - 1) % len(FOLD_PALETTE)]
        epochs_arr = np.arange(1, len(df_h) + 1)

        if "val_mae" in df_h.columns:
            ax2.plot(epochs_arr, df_h["val_mae"], color=color_f, linewidth=1.8, alpha=0.85, linestyle="-", label=f"Fold {fold_num} Val MAE")
            best_idx = int(df_h["val_loss"].idxmin()) if "val_loss" in df_h.columns else int(df_h["loss"].idxmin())
            ax2.scatter(best_idx + 1, df_h["val_mae"].iloc[best_idx], color=color_f, s=90, marker="*", edgecolors="black", linewidths=0.8, zorder=5)

    ax2.set_title(f"B. Multi-Fold Validation Mean Absolute Error (MAE)\nTracking Linear Target Error Across Epochs ({EVAL_MODE.upper()})", fontsize=10.5, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch Number", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Validation MAE", fontsize=9.5, fontweight="bold")
    ax2.set_xlim(1, max_epoch_seen)
    ax2.tick_params(labelsize=8.5)
    ax2.legend(loc="upper right", fontsize=8.0, framealpha=0.90)

    plt.suptitle(
        f"Dual-Branch Multi-Modal Adaptive {actual_k_folds}-Fold Training History Summary ({EVAL_MODE.upper()})\n"
        f"Evaluated Across {len(fold_histories)} Trained Fold Models",
        fontsize=12.0, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Section 8.1: Convergence Diagnostics Summary Table

# %%
if fold_best_metrics:
    df_convergence_summary = pd.DataFrame(fold_best_metrics)
    
    print("=" * 115)
    print(f"MULTI-FOLD TRAINING CONVERGENCE & MODEL CHECKPOINT DIAGNOSTIC REPORT ({EVAL_MODE.upper()}, K={actual_k_folds}):")
    print("=" * 115)
    
    df_display = df_convergence_summary.copy()
    df_display["Best_Train_Loss"] = df_display["Best_Train_Loss"].map(lambda x: f"{x:.6f}")
    df_display["Best_Val_Loss"] = df_display["Best_Val_Loss"].map(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
    df_display["Best_Val_MAE"] = df_display["Best_Val_MAE"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    df_display["Generalization_Ratio"] = df_display["Generalization_Ratio"].map(lambda x: f"{x:.2f}x" if pd.notna(x) else "N/A")
    
    display_cols = ["Fold", "Withheld_Clusters", "Total_Epochs", "Best_Epoch", "Best_Train_Loss", "Best_Val_Loss", "Best_Val_MAE", "Generalization_Ratio"]
    print(df_display[display_cols].to_string(index=False))
    print("=" * 115)