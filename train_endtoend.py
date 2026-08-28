#!/usr/bin/env python
# coding: utf-8

"""
Module: train_endtoend.py
Description: Parallel K-Fold Trainer for the Direct End-to-End Multi-Modal Plume Regressor.
             Directly maps spatial plume dispersion grids and meteorological transport features
             to point-source SO2 emissions without an explicit physical mass skip connection.

Architecture (Direct 2-Input Multi-Modal Regressor):
  - Branch A (Spatial Plume CNN, Shape: [25, 25, 7]):
      * 3-Stage Conv2D hierarchy (16 -> 32 -> 64 filters) + MaxPooling2D + GlobalAveragePooling2D.
      * Processes 4 daytime VCD snapshots and 3 inter-day differential deltas in Dobson Units (DU).
      * Output: 32-dimensional spatial plume embedding (h_plume).
  - Branch B (Meteorological MLP, Shape: [25]):
      * 2-Layer Dense MLP (32 -> 16 units, ReLU).
      * Encodes scale-free dynamic advection, radiation, PBLH, and overnight wind rotation features.
      * Output: 16-dimensional meteorological embedding (h_met).
  - Multimodal Regression Head:
      * Concatenates (h_plume, h_met) -> 48-dim representation + Dropout(0.10) + Dense(32, L2).
      * Direct linear prediction of target label: log10(CleanSYS_Tonnes + 0.1).

Data Augmentation & Preprocessing Pipeline:
  1. Multiplicative Scale Jittering: Scales raw VCD grids and CleanSYS targets by lambda in [0.5, 2.0].
  2. Symmetric Cross-Wind Reflection: 50% random flipud (y -> -y) with conformal v_rot -> -v_rot.
  3. Dynamic 15% Baseline Floor Subtraction: Subtracts 15th-percentile floor on-the-fly from scaled grids.
  4. Delta Computation & Target Normalization: Derives 3 inter-day differential deltas and sets target log10(Q + 0.1).

Validation & Spatial Isolation:
  - Adaptive K-fold spatial cross-validation with zero geographic leakage across regional clusters.
  - Chronological splitting: 80% historical training/validation vs. 20% future weather testing.
"""

import os
import sys
import glob
import re
import json
import time
import shutil
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("DualBranchTrainer")

# =============================================================================
# 1. DIRECTORY PATHS & OUTPUT CONFIGURATION
# =============================================================================
WORKFLOW_DIR: str = "/home/users/ajs224/sheat_mip_gws/code/WindRotatedMLWorkflow/"
MODEL_OUT_DIR: str = os.path.join(WORKFLOW_DIR, "Output_Model/")
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

# Dataset directory mappings for real and synthetic OSSE mission tiers
DATASET_DIRECTORIES: Dict[str, str] = {
    "next_gen": os.path.join(WORKFLOW_DIR, "Output_Tensor_Synthetic_NextGen/"),
    "ideal": os.path.join(WORKFLOW_DIR, "Output_Tensor_Synthetic_Ideal/"),
    "moderate": os.path.join(WORKFLOW_DIR, "Output_Tensor_Synthetic_Moderate/"),
    "gems_realistic": os.path.join(WORKFLOW_DIR, "Output_Tensor_Synthetic_GEMSRealistic/"),
    "log_scaled": os.path.join(WORKFLOW_DIR, "Output_Tensor_DualBranch_LogScaled/")
}

# =============================================================================
# 2. USER HYPERPARAMETERS & PHYSICAL CONSTANTS
# =============================================================================
# Active dataset tier: "next_gen", "ideal", "moderate", "gems_realistic", or "log_scaled"
DEFAULT_TENSOR_MODE: str = "next_gen"

# Exclude West Coast clusters with spatial overlap (G03, G04, G05, G11)
EXCLUDED_CLUSTER_IDS: List[str] = ["G03", "G04", "G05", "G11"]
INCLUDED_CLUSTER_IDS: Optional[List[str]] = None

# Optimization & Cross-Validation Hyperparameters
TARGET_N_FOLDS: int = 3
RANDOM_FOLD_SEED: int = 42
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_EPOCHS: int = 100
DEFAULT_LR: float = 3e-4
DENSE_L2_REG: float = 1e-5

# Physical Conversion Constants
DU_CONVERSION_FACTOR: float = 2.6867e16  # molecules/cm^2 per Dobson Unit (DU)
KG_PER_DU_PER_M2: float = 2.857e-5      # 1 DU = 2.857e-5 kg SO2 / m^2

# =============================================================================
# 3. MASTER REGIONAL CLUSTER REGISTRY
# =============================================================================
MASTER_CLUSTER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "G01": {"name": "Gwangyang-Yeosu Bay Super-Cluster", "ranks": [1, 5, 17, 24, 26, 29, 30, 31, 33, 40, 45, 47], "sector": "Steel, Refining & Petrochem"},
    "G02": {"name": "Ulsan-Onsan Super-Cluster",         "ranks": [16, 20, 21, 27, 34, 44, 48],                        "sector": "Refining & Smelting"},
    "G03": {"name": "Dangjin Coastal Complex",           "ranks": [3, 6, 8],                                           "sector": "Steel & Coal Power"},
    "G04": {"name": "Daesan Petrochemical Hub",          "ranks": [9, 25, 32],                                         "sector": "Refining & Petrochem"},
    "G05": {"name": "Taean Power Station",               "ranks": [4],                                                 "sector": "Power Generation"},
    "G06": {"name": "Pohang Steel Port Hub",             "ranks": [2, 22, 35, 36, 50],                                 "sector": "Steel & Heavy Alloys"},
    "G07": {"name": "Boryeong Power Hub",                "ranks": [10, 15],                                            "sector": "Coal-Fired Power"},
    "G08": {"name": "Gunsan Industrial Estuary",         "ranks": [18, 38, 42],                                        "sector": "Glass & Energy"},
    "G09": {"name": "Donghae-Samcheok Hub",              "ranks": [12, 19, 23],                                        "sector": "Coastal Power"},
    "G10": {"name": "Samcheonpo-Goseong Hub",            "ranks": [11, 13],                                            "sector": "Coastal Power"},
    "G11": {"name": "Yeongheung Power Island",           "ranks": [7],                                                 "sector": "Island Power"},
    "G12": {"name": "Incheon-Banwol Coastal Hub",        "ranks": [39, 46],                                            "sector": "Urban Steel & Heat"},
    "G13": {"name": "Gumi-Gimcheon Inland Hub",          "ranks": [37, 41],                                            "sector": "Inland CHP"},
    "G14": {"name": "Seokpo Smelter (Isolated)",         "ranks": [43],                                                "sector": "Mountain Smelter"},
    "G15": {"name": "KCC Yeoju (Isolated)",              "ranks": [14],                                                "sector": "Glass & Ceramics"},
    "G16": {"name": "GS Pocheon (Isolated)",             "ranks": [28],                                                "sector": "Power Generation"},
    "G17": {"name": "Busan Fashion Color",               "ranks": [49],                                                "sector": "Dyeing & Energy"}
}

# =============================================================================
# 4. PRE-TRAINING ARTIFACT ARCHIVING
# =============================================================================
def archive_previous_artifacts(model_out_dir: str) -> None:
    """
    Moves existing model checkpoints and training logs into a timestamped archive subfolder
    prior to launching Fold 1 training.
    """
    patterns = ["*.keras", "*.h5", "*.csv", "*.json"]
    existing_files: List[str] = []
    
    for p in patterns:
        for fpath in glob.glob(os.path.join(model_out_dir, p)):
            if os.path.dirname(fpath) == model_out_dir.rstrip("/"):
                existing_files.append(fpath)

    if existing_files:
        timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(model_out_dir, f"archive_{timestamp_str}")
        os.makedirs(archive_dir, exist_ok=True)

        archived_count = 0
        for fpath in existing_files:
            try:
                shutil.move(fpath, os.path.join(archive_dir, os.path.basename(fpath)))
                archived_count += 1
            except Exception as e_move:
                logger.warning("Could not archive %s: %s", fpath, e_move)

        logger.info("==========================================================================================")
        logger.info("PRE-TRAINING ARCHIVE: Successfully moved %d previous artifact(s) to:", archived_count)
        logger.info("  -> %s", archive_dir)
        logger.info("==========================================================================================")

# =============================================================================
# 5. SPATIAL CROSS-VALIDATION FOLD BUILDER
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


def build_adaptive_spatial_folds(
    all_tensor_files: List[str],
    target_k_folds: int = 3,
    excluded_clusters: Optional[List[str]] = None,
    included_clusters: Optional[List[str]] = None,
    seed: int = 42
) -> Tuple[Dict[int, List[str]], List[str], List[str]]:
    """
    Partitions active clusters evenly across K validation folds ensuring zero spatial leakage.

    Parameters:
      all_tensor_files (List[str]): List of all discovered NetCDF tensor paths.
      target_k_folds (int): Target number of validation folds (K).
      excluded_clusters (List[str], optional): Cluster IDs to exclude.
      included_clusters (List[str], optional): Explicit cluster IDs to retain.
      seed (int): Random seed for reproducible fold generation.

    Returns:
      Tuple containing:
        - spatial_folds_dict (Dict[int, List[str]]): Mapping of fold indices to withheld cluster IDs.
        - active_clusters (List[str]): List of all retained cluster IDs.
        - active_files (List[str]): NetCDF file paths matching active clusters.
    """
    excluded_clusters = excluded_clusters or []
    all_found = sorted(list(set(parse_tensor_metadata(f)[0] for f in all_tensor_files if parse_tensor_metadata(f)[0] != "UNKNOWN")))
    
    if included_clusters is not None:
        active_clusters = [c for c in all_found if c in included_clusters and c not in excluded_clusters]
    else:
        active_clusters = [c for c in all_found if c not in excluded_clusters]

    if not active_clusters:
        raise ValueError("No active clusters remain after applying inclusion/exclusion filters!")

    actual_k = min(target_k_folds, len(active_clusters))
    np.random.seed(seed)
    shuffled = list(np.random.permutation(active_clusters))
    split_groups = np.array_split(shuffled, actual_k)
    
    spatial_folds_dict: Dict[int, List[str]] = {
        fold_num: sorted(list(group)) for fold_num, group in enumerate(split_groups, start=1)
    }

    active_files = [f for f in all_tensor_files if parse_tensor_metadata(f)[0] in active_clusters]
    return spatial_folds_dict, active_clusters, active_files

# =============================================================================
# 6. IN-MEMORY MULTI-MODAL DATA GENERATOR (DUALBRANCHSEQUENCE)
# =============================================================================
class DualBranchSequence(tf.keras.utils.Sequence):
    """
    Standardized In-Memory Multi-Input Keras Sequence Generator for Direct 2-Input Regression:
    Yields: ((Spatial_7ch [25, 25, 7], Met_Vector [25]), Target_Log_Q [Scalar]).
    """

    def __init__(
        self,
        file_list: List[str],
        batch_size: int = 32,
        training: bool = True,
    ) -> None:
        self.file_paths = file_list
        self.batch_size = batch_size
        self.training = training

        if not self.file_paths:
            raise FileNotFoundError("Empty file list passed to DualBranchSequence generator!")

        self.cached_raw_snapshots: List[np.ndarray] = []
        self.cached_met: List[np.ndarray] = []
        self.cached_meta: List[Dict[str, Any]] = []

        logger.info("Caching %d raw tensors into RAM (Direct 2-Input, training=%s)...", len(self.file_paths), self.training)

        for idx, fpath in enumerate(self.file_paths, start=1):
            try:
                with xr.open_dataset(fpath) as ds:
                    raw_spatial = ds["spatial_plume_tensor"].values.astype(np.float32)
                    
                    # Extract the raw 4 daily snapshots (Channels 0..3) in shape (25, 25, 4)
                    if raw_spatial.shape[0] >= 4 and raw_spatial.shape[0] <= 7:
                        raw_snapshots_4d = np.transpose(raw_spatial[:4, :, :], (1, 2, 0))
                    elif raw_spatial.shape[-1] >= 4:
                        raw_snapshots_4d = raw_spatial[:, :, :4]
                    else:
                        raise ValueError(f"Unexpected spatial tensor shape: {raw_spatial.shape}")

                    met_vec = ds["met_vector"].values.astype(np.float32)  # Shape: (25,)

                    # Extract CleanSYS physical emissions in metric tonnes
                    if "target_physical_kg" in ds:
                        q_true_tonnes = float(ds["target_physical_kg"].values) / 1000.0
                    elif "target_physical_tonnes" in ds:
                        q_true_tonnes = float(ds["target_physical_tonnes"].values)
                    elif "target_log_emissions" in ds:
                        q_true_tonnes = max(0.0, float(10.0 ** float(ds["target_log_emissions"].values) - 0.1))
                    else:
                        q_true_tonnes = 10.0

                    self.cached_raw_snapshots.append(raw_snapshots_4d)
                    self.cached_met.append(met_vec)
                    self.cached_meta.append({
                        "filename": os.path.basename(fpath),
                        "q_true_tonnes": q_true_tonnes,
                        "cluster_id": ds.attrs.get("cluster_id", "UNKNOWN"),
                        "start_date": ds.attrs.get("sequence_start_date", "N/A"),
                    })

            except Exception as e_load:
                logger.error("Failed loading tensor %s: %s", fpath, e_load)
                raise

        logger.info("RAM Caching Complete: Cached %d raw sample tensors.", len(self.file_paths))
        self.indices = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.training:
            np.random.shuffle(self.indices)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        batch_idx_slice = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        batch_x_spatial = []
        batch_x_met = []
        batch_y = []

        for b_idx in batch_idx_slice:
            raw_4d = self.cached_raw_snapshots[b_idx].copy()  # Shape: (25, 25, 4)
            met_arr = self.cached_met[b_idx].copy()           # Shape: (25,)
            meta_info = self.cached_meta[b_idx]
            q_true = meta_info["q_true_tonnes"]

            # =========================================================================
            # STEP 1: APPLY MULTIPLICATIVE SCALE JITTERING & SYMMETRIC FLIP
            # =========================================================================
            if self.training:
                # Scale raw VCD grids and target emissions by lambda in [0.5, 2.0]
                lam = float(np.random.uniform(0.5, 2.0))
                do_flip = bool(np.random.rand() > 0.5)
            else:
                lam = 1.0
                do_flip = False

            scaled_raw_4d = raw_4d * lam
            q_scaled = q_true * lam

            if do_flip:
                scaled_raw_4d = np.flipud(scaled_raw_4d)
                # Invert cross-wind overnight vector components conformally (v_rot -> -v_rot)
                for pair_offset in [16, 18, 20]:
                    met_arr[pair_offset + 1] = -met_arr[pair_offset + 1]

            # =========================================================================
            # STEP 2: DYNAMIC 15% BASELINE FLOOR SUBTRACTION & DELTA COMPUTATION
            # =========================================================================
            net_snapshots_list = []
            for d in range(4):
                snap_d = scaled_raw_4d[:, :, d]
                floor_15_d = float(np.percentile(snap_d, 15))
                net_snap_d = np.maximum(0.0, snap_d - floor_15_d)
                net_snapshots_list.append(net_snap_d)

            # Compute 3 inter-day differential deltas (Channels 4, 5, 6) in DU
            delta_1to2 = net_snapshots_list[1] - net_snapshots_list[0]
            delta_2to3 = net_snapshots_list[2] - net_snapshots_list[1]
            delta_3to4 = net_snapshots_list[3] - net_snapshots_list[2]

            # Stack into 7-channel spatial tensor (Branch A, shape: [25, 25, 7])
            spatial_7ch = np.stack(
                net_snapshots_list + [delta_1to2, delta_2to3, delta_3to4],
                axis=-1
            ).astype(np.float32)

            # =========================================================================
            # STEP 3: TARGET EMISSION LOG NORMALIZATION
            # =========================================================================
            target_val = float(np.log10(q_scaled + 0.1))

            batch_x_spatial.append(spatial_7ch)
            batch_x_met.append(met_arr)
            batch_y.append(target_val)

        return (
            (
                np.array(batch_x_spatial, dtype=np.float32),
                np.array(batch_x_met, dtype=np.float32),
            ),
            np.array(batch_y, dtype=np.float32)
        )

# =============================================================================
# 7. MODEL ARCHITECTURE BUILDER
# =============================================================================
def build_direct_dual_branch_regressor(
    spatial_shape: Tuple[int, int, int] = (25, 25, 7),
    met_features_count: int = 25,
    dense_l2: float = DENSE_L2_REG
) -> tf.keras.Model:
    """
    Constructs the Direct 2-Input Multi-Modal Regressor (Branch A: Spatial CNN, Branch B: Meteorological MLP).
    Predicts log10(CleanSYS_Tonnes + 0.1) directly from fused multi-modal representations.
    """
    spatial_input = tf.keras.layers.Input(shape=spatial_shape, name="spatial_plume_input")
    met_input = tf.keras.layers.Input(shape=(met_features_count,), name="met_vector_input")

    # --- Branch A: 2D Spatial Plume CNN (7 Physical Channels) ---
    x1 = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(spatial_input)
    x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)

    x2 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x1)
    x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)

    x3 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x2)
    gap = tf.keras.layers.GlobalAveragePooling2D(name="spatial_gap")(x3)
    h_plume = tf.keras.layers.Dense(32, activation="relu", name="plume_embedding")(gap)

    # --- Branch B: 1D Meteorological MLP (25 Dynamic Features) ---
    m1 = tf.keras.layers.Dense(32, activation="relu")(met_input)
    h_met = tf.keras.layers.Dense(16, activation="relu", name="met_embedding")(m1)

    # --- Multimodal Fusion & Direct Emission Output ---
    fused = tf.keras.layers.Concatenate(name="multimodal_fusion")([h_plume, h_met])
    fused = tf.keras.layers.Dropout(0.10, name="fusion_dropout")(fused)
    d1 = tf.keras.layers.Dense(
        32, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(dense_l2)
    )(fused)

    # Direct prediction of log10(CleanSYS_Tonnes + 0.1)
    log_q_pred = tf.keras.layers.Dense(1, activation="linear", name="log_emissions_prediction")(d1)

    model = tf.keras.Model(
        inputs=[spatial_input, met_input],
        outputs=log_q_pred,
        name="DirectDualBranchPlumeRegressor"
    )
    return model

# =============================================================================
# 8. TRAINING ORCHESTRATION & MAIN DRIVER
# =============================================================================
def main() -> None:
    """Executes single-fold training, callbacks setup, and performance logging."""
    parser = argparse.ArgumentParser(description="Simplified Direct 2-Input Dual-Branch Trainer")
    parser.add_argument("--mode", type=str, default=DEFAULT_TENSOR_MODE, 
                        choices=["next_gen", "ideal", "moderate", "gems_realistic", "log_scaled"], 
                        help="Active dataset mode")
    parser.add_argument("--fold", type=int, required=True, help="1-based fold index (1 to K)")
    parser.add_argument("--n_folds", type=int, default=TARGET_N_FOLDS, help="Total spatial validation folds (K)")
    parser.add_argument("--tensor_dir", type=str, default=None, help="Optional custom tensor directory override")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Initial learning rate")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "mae"], help="Loss function")
    args = parser.parse_args()

    active_mode = args.mode.lower().strip()
    tensor_directory = args.tensor_dir or DATASET_DIRECTORIES.get(active_mode, DATASET_DIRECTORIES["next_gen"])
    fold_idx = args.fold

    # 1. Pre-Training Archiving on Fold 1
    if fold_idx == 1:
        archive_previous_artifacts(MODEL_OUT_DIR)

    # 2. Discover NetCDF tensors
    all_nc_files = sorted(glob.glob(os.path.join(tensor_directory, "tensor_G*.nc")))
    if not all_nc_files:
        raise FileNotFoundError(f"No NetCDF tensors found in: {tensor_directory}")

    # 3. Build Adaptive Spatial Folds
    spatial_folds, active_clusters, active_files = build_adaptive_spatial_folds(
        all_tensor_files=all_nc_files,
        target_k_folds=args.n_folds,
        excluded_clusters=EXCLUDED_CLUSTER_IDS,
        included_clusters=INCLUDED_CLUSTER_IDS,
        seed=RANDOM_FOLD_SEED
    )

    actual_k = len(spatial_folds)
    if fold_idx < 1 or fold_idx > actual_k:
        raise ValueError(f"Requested Fold {fold_idx} is out of bounds for active K={actual_k} folds!")

    val_clusters = spatial_folds[fold_idx]
    train_clusters = [c for c in active_clusters if c not in val_clusters]

    # Save active fold configuration for evaluation synchronization
    config_save_path = os.path.join(MODEL_OUT_DIR, f"adaptive_spatial_folds_{actual_k}k_config.json")
    with open(config_save_path, "w") as f_cfg:
        json.dump({
            "mode": active_mode,
            "actual_k_folds": actual_k,
            "active_clusters": active_clusters,
            "excluded_clusters": EXCLUDED_CLUSTER_IDS,
            "included_clusters": INCLUDED_CLUSTER_IDS,
            "spatial_folds": spatial_folds
        }, f_cfg, indent=4)

    logger.info("==========================================================================================")
    logger.info("DIRECT DUAL-BRANCH TRAINING (MODE='%s'): FOLD %d/%d", active_mode.upper(), fold_idx, actual_k)
    logger.info(" |- Target Dataset:          %s", tensor_directory)
    logger.info(" |- Active Clusters (%d total): %s", len(active_clusters), active_clusters)
    logger.info(" |- Withheld Validation:     %s (Zero Spatial Leakage)", val_clusters)
    logger.info(" |- Training Clusters:       %s", train_clusters)
    logger.info(" |- Spatial Input Channels:  7 Physical DU Channels (4 Snapshots + 3 Deltas)")
    logger.info(" |- Loss Function:           Standard %s", args.loss.upper())
    logger.info(" |- Hyperparameters:         Batch=%d, LR=%s, Epochs=%d", args.batch_size, args.lr, args.epochs)
    logger.info("==========================================================================================")

    # 4. Temporal Cutoff Splicing (80% Historical vs 20% Future Weather)
    all_dates = [parse_tensor_metadata(f)[1] for f in active_files]
    cutoff_date = min(all_dates) + 0.80 * (max(all_dates) - min(all_dates))

    train_files, val_files, test_files = [], [], []
    for f in active_files:
        gid, start_d, _, _ = parse_tensor_metadata(f)
        if gid in val_clusters:
            if start_d <= cutoff_date: val_files.append(f)
            else: test_files.append(f)
        else:
            if start_d <= cutoff_date: train_files.append(f)
            else: test_files.append(f)

    logger.info("Partitions: Train=%d samples, Val=%d samples, Test=%d samples", len(train_files), len(val_files), len(test_files))

    # 5. Configure GPU Acceleration
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU active: %d device(s).", len(gpus))
        except RuntimeError as e_gpu:
            logger.warning("GPU memory error: %s", e_gpu)

    # 6. Instantiate Data Generators
    train_generator = DualBranchSequence(file_list=train_files, batch_size=args.batch_size, training=True)
    val_generator = DualBranchSequence(file_list=val_files, batch_size=args.batch_size, training=False)

    # 7. Build and Compile Model
    model = build_direct_dual_branch_regressor(
        spatial_shape=(25, 25, 7), 
        met_features_count=25,
        dense_l2=DENSE_L2_REG
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    model.compile(
        optimizer=optimizer,
        loss=args.loss.lower(),
        metrics=["mae", "mse"],
    )

    model_tag = f"dualbranch_{active_mode}_{actual_k}folds"
    model_checkpoint_path = os.path.join(MODEL_OUT_DIR, f"best_plume_regressor_{model_tag}_fold{fold_idx}.keras")
    history_path = os.path.join(MODEL_OUT_DIR, f"training_history_{model_tag}_fold{fold_idx}.csv")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            min_delta=1e-4,
            start_from_epoch=10,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(filename=history_path, separator=",", append=False)
    ]

    logger.info("Starting Training Loop for Fold %d/%d (%s)...", fold_idx, actual_k, active_mode)
    try:
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1
        )
    except Exception as e_train:
        logger.error("Training failed on Fold %d: %s", fold_idx, e_train, exc_info=True)
        raise

    logger.info("==========================================================================================")
    logger.info("FOLD %d/%d TRAINING COMPLETE (%s).", fold_idx, actual_k, active_mode)
    logger.info(" - Best Model Checkpoint: %s", model_checkpoint_path)
    logger.info(" - Metrics History Log:  %s", history_path)
    logger.info("==========================================================================================")


if __name__ == "__main__":
    main()