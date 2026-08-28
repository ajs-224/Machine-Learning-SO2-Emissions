#!/usr/bin/env python
# coding: utf-8

"""
Module: compile_synthetic_osse_tensors.py
Description: High-throughput parallel compiler for synthetic OSSE multi-day inversion tensors.
             Generates paired spatial plume grids and meteorological feature vectors across
             17 South Korean industrial clusters for the 2022-2024 study period.

Tensor Architecture:
  - Branch A (Spatial Tensor, Shape: [7, 25, 25]):
      * Channels 0..3: 4 daytime wind-aligned column density snapshots (in DU).
      * Channels 4..6: 3 inter-day differential deltas (Delta Omega = Omega_{t+1} - Omega_t in DU).
  - Branch B (Meteorology Vector, Shape: [25]):
      * Scale-free dynamic transport, photochemical, and boundary layer features.
  - Labels & Metadata:
      * Target log emissions: log10(CleanSYS_Tonnes + 0.1)
      * Physical emissions totals (kg and tonnes) and integrated domain mass (kg).

Simulated Mission Tiers (Simultaneous Single-Pass Dual Export):
  1. Ideal OSSE: Pure forward dispersion (sigma = 0.00 DU, no background).
  2. Next-Gen Geo: High-SNR geostationary sensor (sigma = 0.05 DU, 15% QA dropout).
  3. Moderate: TROPOMI / Sentinel-4 class sensor (sigma = 0.15 DU, 35% QA dropout).
  4. GEMS Realistic: Operational GEMS Level-2 baseline (sigma = 0.35 DU, 68% QA dropout).

Execution:
  - Supports standalone sequential execution or distributed SLURM job array tasks (1 to 17).
"""

import os
import sys
import glob
import re
import time
import atexit
import argparse
import logging
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
from typing import Dict, List, Tuple, Any, Optional
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("BulkOSSECompiler")

# =============================================================================
# 1. DIRECTORY PATHS & OUTPUT CONFIGURATION
# =============================================================================
WORKFLOW_DIR: str = "/home/users/ajs224/sheat_mip_gws/code/WindRotatedMLWorkflow/"
DAILY_EMISSIONS_CSV: str = "/home/users/ajs224/sheat_mip_gws/code/GroundData/Output_GroundData/daily_emissions_point_sources_2022-24.csv"
ERA5_ML_DIR: str = "/badc/ecmwf-era5/data/oper/an_ml/"
ERA5_SFC_AN_DIR: str = "/badc/ecmwf-era5/data/oper/an_sfc/"
ERA5_SFC_FC_DIR: str = "/badc/ecmwf-era5/data/oper/fc_sfc/"
BLH_DIR: str = "/home/users/ajs224/sheat_mip_gws/data/BLH/"

# Destination directories for synthetic mission tiers
OUTPUT_DIRS: Dict[str, str] = {
    "ideal": os.path.join(WORKFLOW_DIR, "Output_Tensor_Synthetic_Ideal/"),
    "next_gen": os.path.join(WORKFLOW_DIR, "Output_Tensor_Synthetic_NextGen/"),
    "moderate": os.path.join(WORKFLOW_DIR, "Output_Tensor_Synthetic_Moderate/"),
    "gems_realistic": os.path.join(WORKFLOW_DIR, "Output_Tensor_Synthetic_GEMSRealistic/")
}

for d_path in OUTPUT_DIRS.values():
    os.makedirs(d_path, exist_ok=True)

# =============================================================================
# 2. COMPILER HYPERPARAMETERS & PHYSICAL CONSTANTS
# =============================================================================
# --- Temporal Sequence Settings ---
SEQUENCE_LENGTH_DAYS: int = 4                            # Consecutive days per sequence (T = 4)
BUFFER_GAP_DAYS: int = 1                                 # Temporal gap between consecutive sequences
STRIDE_DAYS: int = SEQUENCE_LENGTH_DAYS + BUFFER_GAP_DAYS  # 5-day stride for strict temporal isolation
START_DATE_STR: str = "2022-01-01"
END_DATE_STR: str = "2024-12-31"

# --- Domain & Grid Geometry ---
DOMAIN_HALF_WIDTH_DEG: float = 0.25                      # Half-width in degrees (~55 x 55 km domain)
GRID_RESOLUTION_PIXELS: int = 25                         # Pixel resolution (25 x 25 grid, 0.02 deg spacing)
MIN_WIND_SPEED_MS: float = 0.5                           # Minimum advection wind threshold (m/s)

# --- Plume Dispersion Parameters ---
PLUME_CORE_SIGMA_0_KM: float = 1.00                      # Initial source dispersion radius (km)
PLUME_DISPERSION_ALPHA: float = 0.26                     # Downwind spread scaling exponent
PLUME_UPWIND_SMOOTH_M: float = 800.0                     # Upwind smoothing length scale (m)
WIND_MEANDER_STD_DEG: float = 22.0                       # Sub-hourly wind meander jitter (degrees)

# --- Satellite Observation & Noise Parameters ---
GEMS_NATIVE_DX_KM: float = 3.5                           # GEMS native along-track footprint (km)
GEMS_NATIVE_DY_KM: float = 8.0                           # GEMS native cross-track footprint (km)
GEMS_QA_DROPOUT_RATE: float = 0.68                       # Baseline cloud/quality dropout probability
GEMS_SIGMA_SHOT_DU: float = 0.04                         # Uncorrelated detector shot noise (DU)
GEMS_SIGMA_MATERN_DU: float = 0.26                       # Spatially correlated retrieval noise (DU)
GEMS_MATERN_CORR_LENGTH_KM: float = 30.0                 # Retrieval noise spatial correlation length (km)
GEMS_BG_FLOOR_DU: float = 0.22                           # Ambient regional background baseline (DU)

# --- Physical Unit Conversion Constants ---
DU_CONVERSION_FACTOR: float = 2.6867e16                  # molecules/cm^2 per Dobson Unit (DU)
KG_PER_DU_PER_M2: float = 2.857e-5                      # 1 DU = 2.857e-5 kg SO2 / m^2

# Internal I/O caching dictionaries
_DIR_CACHE_ML: Dict[str, List[str]] = {}
_DIR_CACHE_AN: Dict[str, List[str]] = {}
_DIR_CACHE_FC: Dict[str, List[str]] = {}
_BLH_CACHE: Dict[int, Optional[Dict[str, Any]]] = {}

# Mission tier parameter mapping
NOISE_TIERS: Dict[str, Dict[str, Any]] = {
    "ideal": {
        "dir": OUTPUT_DIRS["ideal"],
        "sigma_shot": 0.00,
        "sigma_matern": 0.00,
        "corr_length_km": 0.0,
        "qa_dropout": 0.00,
        "bg_floor": 0.00,
        "include_bg": False,
        "scheme": "Synthetic OSSE Scenario 1: Ideal Physics (sigma = 0.00 DU, Zero Noise)"
    },
    "next_gen": {
        "dir": OUTPUT_DIRS["next_gen"],
        "sigma_shot": 0.01,
        "sigma_matern": 0.04,
        "corr_length_km": 15.0,
        "qa_dropout": 0.15,
        "bg_floor": GEMS_BG_FLOOR_DU,
        "include_bg": True,
        "scheme": "Synthetic OSSE Scenario 2: Next-Gen Geostationary (sigma = 0.05 DU, High SNR)"
    },
    "moderate": {
        "dir": OUTPUT_DIRS["moderate"],
        "sigma_shot": 0.02,
        "sigma_matern": 0.12,
        "corr_length_km": 20.0,
        "qa_dropout": 0.35,
        "bg_floor": GEMS_BG_FLOOR_DU,
        "include_bg": True,
        "scheme": "Synthetic OSSE Scenario 3: Moderate Sensor (sigma = 0.15 DU, TROPOMI-Class)"
    },
    "gems_realistic": {
        "dir": OUTPUT_DIRS["gems_realistic"],
        "sigma_shot": GEMS_SIGMA_SHOT_DU,
        "sigma_matern": GEMS_SIGMA_MATERN_DU,
        "corr_length_km": GEMS_MATERN_CORR_LENGTH_KM,
        "qa_dropout": GEMS_QA_DROPOUT_RATE,
        "bg_floor": GEMS_BG_FLOOR_DU,
        "include_bg": True,
        "scheme": "Synthetic OSSE Scenario 4: Current GEMS Baseline (sigma = 0.35 DU, 68% Dropouts)"
    }
}

# =============================================================================
# 3. FACILITY & CLUSTER REGISTRIES
# =============================================================================
# Master registry of surveyed South Korean point sources (coordinates and baseline annual emissions)
MASTER_50_EMITTERS: Dict[int, Dict[str, Any]] = {
    1:  {"name": "POSCO Gwangyang",        "lat": 34.920086, "lon": 127.748650, "em_t": 18200.0, "sector": "Steel & Iron"},
    2:  {"name": "POSCO Pohang",           "lat": 36.009310, "lon": 129.394508, "em_t": 17500.0, "sector": "Steel & Iron"},
    3:  {"name": "Hyundai Steel Dangjin",  "lat": 36.986282, "lon": 126.697350, "em_t": 10800.0, "sector": "Steel & Iron"},
    4:  {"name": "Taean Power",            "lat": 36.903319, "lon": 126.234263, "em_t": 14800.0, "sector": "Power Generation"},
    5:  {"name": "Hadong Bitdream",        "lat": 34.952354, "lon": 127.819851, "em_t": 16400.0, "sector": "Power Generation"},
    6:  {"name": "Dangjin Power",          "lat": 37.055100, "lon": 126.511300, "em_t": 15600.0, "sector": "Power Generation"},
    7:  {"name": "Yeongheung Power",       "lat": 37.236940, "lon": 126.436093, "em_t": 14100.0, "sector": "Power Generation"},
    8:  {"name": "Hyundai Green Power",    "lat": 36.986282, "lon": 126.697350, "em_t": 3900.0,  "sector": "Power Generation"},
    9:  {"name": "HD Hyundai Oilbank",     "lat": 37.009528, "lon": 126.402701, "em_t": 9800.0,  "sector": "Oil Refining"},
    10: {"name": "Boryeong Power",         "lat": 36.400853, "lon": 126.490831, "em_t": 8900.0,  "sector": "Power Generation"},
    11: {"name": "Samcheonpo Power",       "lat": 34.910715, "lon": 128.108397, "em_t": 15200.0, "sector": "Power Generation"},
    12: {"name": "Donghae Power",          "lat": 37.486296, "lon": 129.147287, "em_t": 1200.0,  "sector": "Power Generation"},
    13: {"name": "Goseong Green Power",    "lat": 34.904909, "lon": 128.125406, "em_t": 2800.0,  "sector": "Power Generation"},
    14: {"name": "KCC Yeoju",              "lat": 37.243000, "lon": 127.593100, "em_t": 2100.0,  "sector": "Glass & Ceramics"},
    15: {"name": "Shin-Boryeong Power",    "lat": 36.384613, "lon": 126.486169, "em_t": 3100.0,  "sector": "Power Generation"},
    16: {"name": "Korea Zinc Onsan",       "lat": 35.429158, "lon": 129.346715, "em_t": 18200.0, "sector": "Non-Ferrous Metals"},
    17: {"name": "Hanwha Yeosu",           "lat": 34.823570, "lon": 127.665190, "em_t": 1850.0,  "sector": "Power Generation"},
    18: {"name": "LX Glass Gunsan",        "lat": 35.969100, "lon": 126.611000, "em_t": 1750.0,  "sector": "Glass & Ceramics"},
    19: {"name": "Samcheok Blue Power",    "lat": 37.406870, "lon": 129.177800, "em_t": 1600.0,  "sector": "Power Generation"},
    20: {"name": "LS MnM Onsan",           "lat": 35.437000, "lon": 129.349000, "em_t": 1450.0,  "sector": "Non-Ferrous Metals"},
    21: {"name": "Hanju Corporation",      "lat": 35.492200, "lon": 129.329500, "em_t": 1250.0,  "sector": "Power Generation"},
    22: {"name": "OCI Pohang",             "lat": 35.998000, "lon": 129.412100, "em_t": 1300.0,  "sector": "Chemicals"},
    23: {"name": "GS Bukpyung Power",      "lat": 37.480546, "lon": 129.143343, "em_t": 1100.0,  "sector": "Power Generation"},
    24: {"name": "SNNC Gwangyang",         "lat": 34.912000, "lon": 127.729000, "em_t": 1050.0,  "sector": "Non-Ferrous Metals"},
    25: {"name": "SEETEC Daesan",          "lat": 37.004700, "lon": 126.380800, "em_t": 1020.0,  "sector": "Petrochemicals"},
    26: {"name": "Kumho 2nd Yeosu",        "lat": 34.814800, "lon": 127.645300, "em_t": 920.0,   "sector": "Power Generation"},
    27: {"name": "S-Oil Onsan",            "lat": 35.448700, "lon": 129.330800, "em_t": 8900.0,  "sector": "Oil Refining"},
    28: {"name": "GS Pocheon Green Energy","lat": 37.958400, "lon": 127.231500, "em_t": 850.0,   "sector": "Power Generation"},
    29: {"name": "Namhae Chem Yeosu",      "lat": 34.849600, "lon": 127.746100, "em_t": 780.0,   "sector": "Chemicals"},
    30: {"name": "Boim CHP Yeosu",         "lat": 34.833000, "lon": 127.688000, "em_t": 710.0,   "sector": "Power Generation"},
    31: {"name": "GS Caltex Yeosu",        "lat": 34.851378, "lon": 127.690818, "em_t": 14200.0, "sector": "Oil Refining"},
    32: {"name": "HD Hyundai OCI Daesan",  "lat": 37.000200, "lon": 126.409800, "em_t": 540.0,   "sector": "Petrochemicals"},
    33: {"name": "Orion Yeosu",            "lat": 34.847300, "lon": 127.733100, "em_t": 510.0,   "sector": "Chemicals"},
    34: {"name": "SK Energy Ulsan",        "lat": 35.504000, "lon": 129.355000, "em_t": 7200.0,  "sector": "Oil Refining"},
    35: {"name": "SIMPAC FeSi Pohang",     "lat": 35.944400, "lon": 129.419600, "em_t": 480.0,   "sector": "Metals"},
    36: {"name": "Dongkuk Steel 1 Pohang", "lat": 36.001500, "lon": 129.397600, "em_t": 420.0,   "sector": "Steel & Iron"},
    37: {"name": "GS Gumi CHP",            "lat": 36.097200, "lon": 128.372200, "em_t": 450.0,   "sector": "Power Generation"},
    38: {"name": "SGC Energy Gunsan",      "lat": 35.977700, "lon": 126.638000, "em_t": 410.0,   "sector": "Power Generation"},
    39: {"name": "Hyundai Steel Incheon",  "lat": 37.489700, "lon": 126.637900, "em_t": 380.0,   "sector": "Steel & Iron"},
    40: {"name": "OCI Gwangyang",          "lat": 34.936000, "lon": 127.758200, "em_t": 320.0,   "sector": "Chemicals"},
    41: {"name": "Gimcheon Energy",        "lat": 36.156100, "lon": 128.126900, "em_t": 310.0,   "sector": "Power Generation"},
    42: {"name": "Dongwon Systems Gunsan", "lat": 35.974400, "lon": 126.646700, "em_t": 180.0,   "sector": "Packaging & Metals"},
    43: {"name": "Young Poong Seokpo",     "lat": 37.043100, "lon": 129.057900, "em_t": 290.0,   "sector": "Non-Ferrous Metals"},
    44: {"name": "SK Multi Utility Ulsan", "lat": 35.469900, "lon": 129.357000, "em_t": 240.0,   "sector": "Power Generation"},
    45: {"name": "Kumho 1st Yeosu",        "lat": 34.809200, "lon": 127.660000, "em_t": 210.0,   "sector": "Power Generation"},
    46: {"name": "GS Banwol Ansan",        "lat": 37.297700, "lon": 126.800600, "em_t": 200.0,   "sector": "Power Generation"},
    47: {"name": "BASF Korea Yeosu",       "lat": 34.819600, "lon": 127.650300, "em_t": 160.0,   "sector": "Chemicals"},
    48: {"name": "FarmHannong Ulsan",       "lat": 35.519000, "lon": 129.355000, "em_t": 150.0,   "sector": "Chemicals"},
    49: {"name": "Busan Fashion Color",     "lat": 35.092800, "lon": 128.955900, "em_t": 120.0,   "sector": "Dyeing & Energy"},
    50: {"name": "Dongkuk Steel 2 Pohang", "lat": 36.001400, "lon": 129.397600, "em_t": 120.0,   "sector": "Steel & Iron"}
}

# Unified 17 regional cluster definitions
CLUSTER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "G01": {"name": "Gwangyang-Yeosu Bay Super-Cluster", "ranks": [1, 5, 17, 24, 26, 29, 30, 31, 33, 40, 45, 47], "centroid": (34.903478, 127.748507), "sector": "Steel, Refining & Petrochem"},
    "G02": {"name": "Ulsan-Onsan Super-Cluster",         "ranks": [16, 20, 21, 27, 34, 44, 48],                        "centroid": (35.451255, 129.344134), "sector": "Refining & Smelting"},
    "G03": {"name": "Dangjin Coastal Complex",           "ranks": [3, 6, 8],                                           "centroid": (37.021713, 126.601562), "sector": "Steel & Coal Power"},
    "G04": {"name": "Daesan Petrochemical Hub",          "ranks": [9, 25, 32],                                         "centroid": (37.008651, 126.401072), "sector": "Refining & Petrochem"},
    "G05": {"name": "Taean Power Station",               "ranks": [4],                                                 "centroid": (36.903319, 126.234263), "sector": "Power Generation"},
    "G06": {"name": "Pohang Steel Port Hub",             "ranks": [2, 22, 35, 36, 50],                                 "centroid": (36.006783, 129.396354), "sector": "Steel & Heavy Alloys"},
    "G07": {"name": "Boryeong Power Hub",                "ranks": [10, 15],                                            "centroid": (36.396658, 126.489627), "sector": "Coal-Fired Power"},
    "G08": {"name": "Gunsan Industrial Estuary",         "ranks": [18, 38, 42],                                        "centroid": (35.971015, 126.618477), "sector": "Glass & Energy"},
    "G09": {"name": "Donghae-Samcheok Hub",              "ranks": [12, 19, 23],                                        "centroid": (37.452089, 129.158693), "sector": "Coastal Power"},
    "G10": {"name": "Samcheonpo-Goseong Hub",            "ranks": [11, 13],                                            "centroid": (34.909812, 128.111043), "sector": "Coastal Power"},
    "G11": {"name": "Yeongheung Power Island",           "ranks": [7],                                                 "centroid": (37.236940, 126.436093), "sector": "Island Power"},
    "G12": {"name": "Incheon-Banwol Coastal Hub",        "ranks": [39, 46],                                            "centroid": (37.423493, 126.694003), "sector": "Urban Steel & Heat"},
    "G13": {"name": "Gumi-Gimcheon Inland Hub",          "ranks": [37, 41],                                            "centroid": (36.121225, 128.272143), "sector": "Inland CHP"},
    "G14": {"name": "Seokpo Smelter (Isolated)",         "ranks": [43],                                                "centroid": (37.043100, 129.057900), "sector": "Mountain Smelter"},
    "G15": {"name": "KCC Yeoju (Isolated)",              "ranks": [14],                                                "centroid": (37.243000, 127.593100), "sector": "Glass & Ceramics"},
    "G16": {"name": "GS Pocheon (Isolated)",             "ranks": [28],                                                "centroid": (37.958400, 127.231500), "sector": "Power Generation"},
    "G17": {"name": "Busan Fashion Color",               "ranks": [49],                                                "centroid": (35.092800, 128.955900), "sector": "Dyeing & Energy"}
}

CLUSTER_KEYS_ORDERED: List[str] = sorted(list(CLUSTER_REGISTRY.keys()))

# =============================================================================
# 4. UTILITY & FILE MANAGEMENT HELPERS
# =============================================================================
def close_blh_datasets() -> None:
    """Closes all cached BLH NetCDF dataset handles upon program termination."""
    for year, cache in _BLH_CACHE.items():
        if cache is not None and "dataset" in cache:
            try:
                cache["dataset"].close()
            except Exception:
                pass


atexit.register(close_blh_datasets)


def clean_synthetic_output_directories(directories: List[str]) -> None:
    """Deletes existing NetCDF tensor files from target directories before compilation."""
    for d in directories:
        if os.path.exists(d):
            existing_nc = glob.glob(os.path.join(d, "*.nc"))
            deleted_count = 0
            for f in existing_nc:
                try:
                    os.remove(f)
                    deleted_count += 1
                except Exception as e_del:
                    logger.warning("Failed to delete %s: %s", f, e_del)
            logger.info("Cleaned %d existing NetCDF tensor(s) from: %s", deleted_count, d)


def generate_disjoint_5day_sequences(
    start_str: str,
    end_str: str,
    seq_len: int = 4,
    gap: int = 1
) -> List[List[pd.Timestamp]]:
    """
    Generates strictly disjoint temporal sequences separated by a buffer gap.

    Parameters:
      start_str (str): Start date string (YYYY-MM-DD).
      end_str (str): End date string (YYYY-MM-DD).
      seq_len (int): Active sequence duration in days (default: 4).
      gap (int): Separation buffer in days between sequences (default: 1).

    Returns:
      List[List[pd.Timestamp]]: Chronological list of date arrays for each sequence.
    """
    start_d = pd.Timestamp(start_str)
    end_d = pd.Timestamp(end_str)
    date_range = pd.date_range(start=start_d, end=end_d, freq="D")
    
    sequences: List[List[pd.Timestamp]] = []
    stride = seq_len + gap
    i = 0
    while i + seq_len <= len(date_range):
        w_dates = list(date_range[i : i + seq_len])
        sequences.append(w_dates)
        i += stride
    return sequences

# =============================================================================
# 5. METEOROLOGICAL RETRIEVAL (ERA5 REANALYSIS & BOUNDARY LAYER DYNAMICS)
# =============================================================================
def get_era5_pbl_wind_fast(
    time_val: pd.Timestamp,
    center_lat: float,
    center_lon: float,
    pblh_m: Optional[float] = None
) -> Optional[Tuple[float, float]]:
    """
    Extracts depth-averaged (u, v) wind components from the surface up to the instantaneous PBLH.

    Level Integration Scheme:
      n_levels = int(np.clip(pblh_m / 42.0, 6, 24))
      - Lower bound (6 levels, ~200 m): Avoids surface friction drag during shallow winter inversions.
      - Upper bound (24 levels, ~1000 m): Captures deep convective mixing without free-tropospheric shear.

    Parameters:
      time_val (pd.Timestamp): Target timestamp (UTC).
      center_lat (float): Latitude coordinate in degrees.
      center_lon (float): Longitude coordinate in degrees.
      pblh_m (float, optional): Instantaneous planetary boundary layer height in metres.

    Returns:
      Optional[Tuple[float, float]]: Depth-averaged (u, v) wind vector in m/s, or None if unavailable.
    """
    y, m, d, h = time_val.strftime("%Y"), time_val.strftime("%m"), time_val.strftime("%d"), time_val.strftime("%H")
    day_dir = os.path.join(ERA5_ML_DIR, y, m, d)

    if day_dir not in _DIR_CACHE_ML:
        if os.path.exists(day_dir):
            _DIR_CACHE_ML[day_dir] = os.listdir(day_dir)
        else:
            return None

    files_list = _DIR_CACHE_ML[day_dir]
    pattern = f"{y}{m}{d}{h}"
    u_file = next((os.path.join(day_dir, f) for f in files_list if f.endswith(".u.nc") and pattern in f), None)
    v_file = next((os.path.join(day_dir, f) for f in files_list if f.endswith(".v.nc") and pattern in f), None)
    if not (u_file and v_file):
        return None

    try:
        if pblh_m is not None and not np.isnan(pblh_m):
            n_levels = int(np.clip(pblh_m / 42.0, 6, 24))
        else:
            n_levels = 10  # Fallback: ~350 m transport layer

        with nc.Dataset(u_file, "r") as ds_u, nc.Dataset(v_file, "r") as ds_v:
            lat_var = next((c for c in ds_u.variables if c.lower() in ["latitude", "lat"]), "latitude")
            lon_var = next((c for c in ds_u.variables if c.lower() in ["longitude", "lon"]), "longitude")
            lats_raw = ds_u.variables[lat_var][:]
            lons_raw = ds_u.variables[lon_var][:]
            
            norm_lon = center_lon if (np.max(lons_raw) <= 180.0 or center_lon >= 0) else center_lon + 360.0
            lat_i = int(np.argmin(np.abs(lats_raw - center_lat)))
            lon_i = int(np.argmin(np.abs(lons_raw - norm_lon)))

            var_u = next((v for v in ds_u.variables if v.lower() in ["u", "var131", "u100"]), "u")
            var_v = next((v for v in ds_v.variables if v.lower() in ["v", "var132", "v100"]), "v")
            u_raw = ds_u.variables[var_u]
            v_raw = ds_v.variables[var_v]

            if len(u_raw.shape) == 4:
                u_pbl = float(np.mean(u_raw[0, -n_levels:, lat_i, lon_i]))
                v_pbl = float(np.mean(v_raw[0, -n_levels:, lat_i, lon_i]))
            else:
                u_pbl = float(np.mean(u_raw[-n_levels:, lat_i, lon_i]))
                v_pbl = float(np.mean(v_raw[-n_levels:, lat_i, lon_i]))
        return u_pbl, v_pbl
    except Exception as e_wind:
        logger.debug("ERA5 PBL wind extraction failed at %s: %s", time_val, e_wind)
        return None


def get_era5_t2m_fast(time_val: pd.Timestamp, center_lat: float, center_lon: float) -> Optional[float]:
    """Extracts 2-metre temperature in Celsius from ERA5 surface analysis (*2t.nc)."""
    y, m, d, h = time_val.strftime("%Y"), time_val.strftime("%m"), time_val.strftime("%d"), time_val.strftime("%H")
    day_dir = os.path.join(ERA5_SFC_AN_DIR, y, m, d)

    if day_dir not in _DIR_CACHE_AN:
        if os.path.exists(day_dir): _DIR_CACHE_AN[day_dir] = os.listdir(day_dir)
        else: return None

    files_list = _DIR_CACHE_AN[day_dir]
    hour_pattern = f"{y}{m}{d}{h}"
    t2m_file = next((f for f in files_list if (f.endswith(".2t.nc") or ".2t." in f) and hour_pattern in f), None)
    if not t2m_file: return None

    try:
        with nc.Dataset(os.path.join(day_dir, t2m_file), "r") as ds_t:
            lat_var = next((c for c in ds_t.variables if c.lower() in ["latitude", "lat"]), "latitude")
            lon_var = next((c for c in ds_t.variables if c.lower() in ["longitude", "lon"]), "longitude")
            lats_raw = ds_t.variables[lat_var][:]
            lons_raw = ds_t.variables[lon_var][:]
            norm_lon = center_lon if (np.max(lons_raw) <= 180.0 or center_lon >= 0) else center_lon + 360.0
            lat_i = int(np.argmin(np.abs(lats_raw - center_lat)))
            lon_i = int(np.argmin(np.abs(lons_raw - norm_lon)))
            var_t = next((v for v in ds_t.variables if v.lower() in ["2t", "t2m", "var167"]), "2t")
            t_k = float(ds_t.variables[var_t][..., lat_i, lon_i].flatten()[0])
            return t_k - 273.15
    except Exception:
        return None


def get_era5_solar_radiation_fast(time_val: pd.Timestamp, center_lat: float, center_lon: float) -> Optional[float]:
    """Extracts mean surface net short-wave radiation flux (msnswrf in W/m^2) from ERA5 forecasts."""
    y, m, d, h = time_val.strftime("%Y"), time_val.strftime("%m"), time_val.strftime("%d"), time_val.strftime("%H")
    day_dir = os.path.join(ERA5_SFC_FC_DIR, y, m, d)

    if day_dir not in _DIR_CACHE_FC:
        if os.path.exists(day_dir): _DIR_CACHE_FC[day_dir] = os.listdir(day_dir)
        else:
            day_dir_an = os.path.join(ERA5_SFC_AN_DIR, y, m, d)
            if os.path.exists(day_dir_an):
                _DIR_CACHE_FC[day_dir] = os.listdir(day_dir_an)
                day_dir = day_dir_an
            else: return None

    files_list = _DIR_CACHE_FC.get(day_dir, [])
    hour_pattern = f"{y}{m}{d}{h}"
    rad_file = next((f for f in files_list if (f.endswith(".msnswrf.nc") or ".msnswrf." in f or f.endswith(".ssrd.nc")) and hour_pattern in f), None)
    if not rad_file: return None

    try:
        with nc.Dataset(os.path.join(day_dir, rad_file), "r") as ds_r:
            lat_var = next((c for c in ds_r.variables if c.lower() in ["latitude", "lat"]), "latitude")
            lon_var = next((c for c in ds_r.variables if c.lower() in ["longitude", "lon"]), "longitude")
            lats_raw = ds_r.variables[lat_var][:]
            lons_raw = ds_r.variables[lon_var][:]
            norm_lon = center_lon if (np.max(lons_raw) <= 180.0 or center_lon >= 0) else center_lon + 360.0
            lat_i = int(np.argmin(np.abs(lats_raw - center_lat)))
            lon_i = int(np.argmin(np.abs(lons_raw - norm_lon)))
            var_r = next((v for v in ds_r.variables if v.lower() in ["msnswrf", "ssrd", "var176", "var169"]), "msnswrf")
            raw_val = float(ds_r.variables[var_r][..., lat_i, lon_i].flatten()[0])
            flux_w_m2 = raw_val / 3600.0 if ("ssrd" in var_r.lower() and raw_val > 5000.0) else raw_val
            return max(0.0, flux_w_m2)
    except Exception:
        return None


def get_era5_blh_fast(time_val: pd.Timestamp, center_lat: float, center_lon: float) -> Optional[float]:
    """Queries boundary layer height (BLH in metres) with persistent NetCDF dataset caching."""
    year = time_val.year
    if year not in _BLH_CACHE:
        blh_file = os.path.join(BLH_DIR, f"era5_south_korea_blh_{year}.nc")
        if not os.path.exists(blh_file): return None
        ds = None
        try:
            ds = nc.Dataset(blh_file, "r")
            lat_var = next((c for c in ds.variables if c.lower() in ["latitude", "lat"]), "latitude")
            lon_var = next((c for c in ds.variables if c.lower() in ["longitude", "lon"]), "longitude")
            time_var = next((c for c in ds.variables if c.lower() in ["time", "valid_time"]), "time")
            blh_var = next((c for c in ds.variables if c.lower() in ["blh", "boundary_layer_height", "var159"]), "blh")

            lats_raw = ds.variables[lat_var][:]
            lons_raw = ds.variables[lon_var][:]
            raw_times = ds.variables[time_var][:]
            time_units = getattr(ds.variables[time_var], "units", "hours since 1900-01-01 00:00:00.0")
            cal = getattr(ds.variables[time_var], "calendar", "standard")

            raw_dates = nc.num2date(raw_times, units=time_units, calendar=cal)
            parsed_dates = [
                pd.Timestamp(d.year, d.month, d.day, d.hour, d.minute, d.second) if hasattr(d, "year") else pd.Timestamp(str(d))
                for d in raw_dates
            ]
            _BLH_CACHE[year] = {
                "dataset": ds, "lats": lats_raw, "lons": lons_raw, "times": pd.DatetimeIndex(parsed_dates), "blh_var": ds.variables[blh_var]
            }
        except Exception:
            if ds is not None:
                try: ds.close()
                except Exception: pass
            _BLH_CACHE[year] = None
            return None

    cache = _BLH_CACHE.get(year)
    if cache is None: return None

    try:
        t_idx = int(np.argmin(np.abs(cache["times"] - time_val.round("h"))))
        lats_raw = cache["lats"]
        lons_raw = cache["lons"]
        norm_lon = center_lon if (np.max(lons_raw) <= 180.0 or center_lon >= 0) else center_lon + 360.0
        lat_i = int(np.argmin(np.abs(lats_raw - center_lat)))
        lon_i = int(np.argmin(np.abs(lons_raw - norm_lon)))
        return max(50.0, float(cache["blh_var"][t_idx, lat_i, lon_i]))
    except Exception:
        return None


def get_interday_transition_wind(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    center_lat: float,
    center_lon: float,
    weighting_power: float = 1.0
) -> Tuple[float, float, float]:
    """Computes speed-weighted mean wind vector across the 24-hour overnight gap using PBLH-adaptive depth averaging."""
    sample_hours = pd.date_range(start=start_time, end=end_time, freq="2H")
    u_list, v_list = [], []
    for t_step in sample_hours:
        blh_t = get_era5_blh_fast(t_step, center_lat, center_lon)
        w = get_era5_pbl_wind_fast(t_step, center_lat, center_lon, pblh_m=blh_t)
        if w is not None:
            u_list.append(w[0])
            v_list.append(w[1])

    if not u_list: return 3.0, 0.0, 3.0
    u_arr, v_arr = np.array(u_list, dtype=np.float32), np.array(v_list, dtype=np.float32)
    speeds = np.sqrt(u_arr**2 + v_arr**2)
    weights = speeds ** weighting_power
    sum_w = np.sum(weights)
    if sum_w > 0:
        norm_w = weights / sum_w
        return float(np.sum(u_arr * norm_w)), float(np.sum(v_arr * norm_w)), float(np.sum(speeds * norm_w))
    return float(np.mean(u_arr)), float(np.mean(v_arr)), float(np.mean(speeds))

# =============================================================================
# 6. PLUME DISPERSION & SYNTHETIC NOISE GENERATION ENGINES
# =============================================================================
def ddeq_single_stack_plume(
    q_tonnes_day: float,
    u_wind_ms: float,
    rad_flux_wm2: float,
    x_grid_km: np.ndarray,
    y_grid_km: np.ndarray,
    stack_x_along: float = 0.0,
    stack_y_cross: float = 0.0,
    meander_angle_deg: float = 0.0,
    sigma_0_km: float = PLUME_CORE_SIGMA_0_KM,
    alpha_disp: float = PLUME_DISPERSION_ALPHA,
    upwind_smooth_m: float = PLUME_UPWIND_SMOOTH_M
) -> np.ndarray:
    """
    Computes Gaussian plume column density (DU) for an individual surveyed point source.
    Accounts for cross-wind diffusion sigma_y(x) and photochemical OH oxidation decay.
    """
    q_kg_s = (q_tonnes_day * 1000.0) / 86400.0
    u_eff = max(0.8, u_wind_ms)
    
    dx_stk = x_grid_km - stack_x_along
    dy_stk = y_grid_km - stack_y_cross
    
    th_m = np.radians(meander_angle_deg)
    xm = dx_stk * np.cos(th_m) + dy_stk * np.sin(th_m)
    ym = -dx_stk * np.sin(th_m) + dy_stk * np.cos(th_m)
    
    # Photochemical OH oxidation lifetime (tau_OH ~ 5h summer midday, ~36h winter)
    tau_oh_hours = max(3.5, 36.0 / (1.0 + 6.0 * (rad_flux_wm2 / 650.0)))
    tau_oh_s = tau_oh_hours * 3600.0
    
    dist_downwind = np.maximum(0.0, xm)
    sigma_y_km = sigma_0_km + alpha_disp * (dist_downwind ** 0.80)
    sigma_y_m = sigma_y_km * 1000.0
    
    along_decay = np.exp(- (dist_downwind * 1000.0) / (u_eff * tau_oh_s))
    along_decay = np.where(xm < -3.0, 0.0, along_decay)
    
    cross_dist_m = ym * 1000.0
    gauss_cross = np.exp(- (cross_dist_m**2) / (2.0 * sigma_y_m**2))
    
    upwind_factor = np.where(
        xm >= 0,
        1.0,
        np.exp(- (xm * 1000.0)**2 / (2.0 * (upwind_smooth_m)**2))
    )
    
    column_density_kg_m2 = (q_kg_s / (np.sqrt(2.0 * np.pi) * u_eff * sigma_y_m)) * gauss_cross * along_decay * upwind_factor
    vcd_du = column_density_kg_m2 / KG_PER_DU_PER_M2
    return vcd_du.astype(np.float32)


def ddeq_multi_stack_superposition(
    stack_emissions_dict: Dict[int, float],
    stack_offsets_dict: Dict[int, Tuple[float, float]],
    target_ranks: List[int],
    u_wind_ms: float,
    v_wind_ms: float,
    rad_flux_wm2: float,
    x_along_grid: np.ndarray,
    y_cross_grid: np.ndarray,
    meander_angle_deg: float = 0.0
) -> np.ndarray:
    """Superimposes individual Gaussian plumes across constituent facility stacks in along-wind coordinates."""
    speed = np.sqrt(u_wind_ms**2 + v_wind_ms**2)
    theta_w = 0.0 if speed < 0.5 else np.arctan2(v_wind_ms, u_wind_ms)
    cos_th = np.cos(theta_w)
    sin_th = np.sin(theta_w)
    
    total_composite_vcd = np.zeros_like(x_along_grid, dtype=np.float32)
    
    for r in target_ranks:
        q_r = stack_emissions_dict.get(r, 0.0)
        if q_r <= 0.0: continue
        
        dx_r, dy_r = stack_offsets_dict.get(r, (0.0, 0.0))
        x_stk_along = dx_r * cos_th + dy_r * sin_th
        y_stk_cross = -dx_r * sin_th + dy_r * cos_th
        
        plume_r = ddeq_single_stack_plume(
            q_r, speed, rad_flux_wm2, x_along_grid, y_cross_grid,
            stack_x_along=x_stk_along, stack_y_cross=y_stk_cross,
            meander_angle_deg=meander_angle_deg
        )
        total_composite_vcd += plume_r
        
    return total_composite_vcd


def generate_ambient_regional_background(
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    base_du: float = GEMS_BG_FLOOR_DU
) -> np.ndarray:
    """Generates synthetic ambient mesoscale background SO2 column density field (in DU)."""
    gradient_x = -0.04 * (x_mesh / (np.max(x_mesh) + 1e-4))
    wave_pattern = 0.05 * np.cos(2.0 * np.pi * (x_mesh + 15.0) / 90.0) * np.cos(np.pi * y_mesh / 60.0)
    return (base_du + gradient_x + wave_pattern).astype(np.float32)


def compute_matern_spatial_noise_field(
    shape: Tuple[int, int],
    cell_size_km: float = 2.2,
    corr_length_km: float = 30.0,
    variance: float = 0.0676
) -> np.ndarray:
    """Synthesizes 2D isotropic spatially-correlated Gaussian noise using a Matérn spectral power density."""
    if corr_length_km <= 0.0 or variance <= 0.0:
        return np.zeros(shape, dtype=np.float32)
        
    ny, nx = shape
    kx = np.fft.fftfreq(nx, d=cell_size_km)
    ky = np.fft.fftfreq(ny, d=cell_size_km)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_radial = np.sqrt(kx_grid**2 + ky_grid**2)
    k0 = 1.0 / max(1.0, corr_length_km)

    psd = 1.0 / (1.0 + (k_radial / k0)**2)**2.0
    psd[0, 0] = 0.0

    white_noise = np.random.normal(0, 1, size=shape) + 1j * np.random.normal(0, 1, size=shape)
    grf_complex = np.fft.ifft2(np.fft.fft2(white_noise) * np.sqrt(psd))
    grf_real = np.real(grf_complex)
    
    grf_norm = (grf_real - np.mean(grf_real)) / (np.std(grf_real) + 1e-6)
    return (grf_norm * np.sqrt(variance)).astype(np.float32)


def emulate_gems_l2_swath_gridding(
    pure_vcd_grid: np.ndarray,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    sigma_shot: float = GEMS_SIGMA_SHOT_DU,
    sigma_matern: float = GEMS_SIGMA_MATERN_DU,
    corr_length_km: float = GEMS_MATERN_CORR_LENGTH_KM,
    bg_floor: float = GEMS_BG_FLOOR_DU,
    qa_dropout_rate: float = GEMS_QA_DROPOUT_RATE,
    native_dx_km: float = GEMS_NATIVE_DX_KM,
    native_dy_km: float = GEMS_NATIVE_DY_KM
) -> np.ndarray:
    """
    Emulates GEMS Level-2 swath measurement, noise injection, and Delaunay triangulation gridding.
    Injects uncorrelated shot noise, spatially correlated Matérn noise, and random QA dropouts.
    """
    if sigma_shot == 0.0 and sigma_matern == 0.0 and bg_floor == 0.0:
        return pure_vcd_grid.astype(np.float32)

    x_min, x_max = float(x_mesh.min()), float(x_mesh.max())
    y_min, y_max = float(y_mesh.min()), float(y_mesh.max())

    x_nodes = np.arange(x_min - 8.0, x_max + 8.0, native_dx_km)
    y_nodes = np.arange(y_min - 8.0, y_max + 8.0, native_dy_km)
    xx_nodes, yy_nodes = np.meshgrid(x_nodes, y_nodes)

    x_jitter = np.random.uniform(-0.8, 0.8, size=xx_nodes.shape)
    y_jitter = np.random.uniform(-1.2, 1.2, size=yy_nodes.shape)
    x_pts = (xx_nodes + x_jitter).flatten()
    y_pts = (yy_nodes + y_jitter).flatten()

    grid_flat_x = x_mesh.flatten()
    grid_flat_y = y_mesh.flatten()
    grid_flat_vcd = pure_vcd_grid.flatten()

    pure_native = griddata(
        (grid_flat_x, grid_flat_y), grid_flat_vcd, (x_pts, y_pts),
        method="linear", fill_value=0.0
    )

    bg_native = bg_floor + 0.04 * np.cos(2.0 * np.pi * (x_pts + 15.0) / 90.0) - 0.03 * (x_pts / (x_max + 1e-4))

    if sigma_matern > 0 and corr_length_km > 0:
        matern_native = np.random.normal(0, sigma_matern, size=x_pts.shape)
        matern_2d = matern_native.reshape(xx_nodes.shape)
        matern_smooth = gaussian_filter(matern_2d, sigma=1.2).flatten()
        matern_smooth = (matern_smooth / (np.std(matern_smooth) + 1e-6)) * sigma_matern
    else:
        matern_smooth = 0.0

    shot_native = np.random.normal(0, sigma_shot, size=x_pts.shape) if sigma_shot > 0 else 0.0
    noisy_native_vcd = np.maximum(0.0, pure_native + bg_native + matern_smooth + shot_native)

    n_pts = len(x_pts)
    qa_mask = np.random.uniform(0, 1, size=n_pts) > qa_dropout_rate
    near_stack = (x_pts**2 + y_pts**2) < 25.0
    qa_mask = qa_mask | near_stack

    surviving_pts = np.column_stack((x_pts[qa_mask], y_pts[qa_mask]))
    surviving_vcd = noisy_native_vcd[qa_mask]

    gridded_linear = griddata(surviving_pts, surviving_vcd, (x_mesh, y_mesh), method="linear")

    if np.any(np.isnan(gridded_linear)):
        nan_mask = np.isnan(gridded_linear)
        gridded_nn = griddata(surviving_pts, surviving_vcd, (x_mesh, y_mesh), method="nearest")
        gridded_linear[nan_mask] = gridded_nn[nan_mask]

    return gridded_linear.astype(np.float32)

# =============================================================================
# 7. PARALLEL SYNTHETIC TENSOR COMPILER
# =============================================================================
def compile_cluster_synthetic_tensors(target_gid: str) -> int:
    """
    Compiles all synthetic OSSE tensors for a regional cluster across the 2022-2024 period.
    Simultaneously exports NetCDF files across all 4 mission tiers in a single pass.

    Parameters:
      target_gid (str): Regional cluster identifier (e.g., 'G01').

    Returns:
      int: Total number of 4-day sequences compiled.
    """
    cdata = CLUSTER_REGISTRY[target_gid]
    c_name = cdata["name"]
    c_lat, c_lon = cdata["centroid"]
    c_ranks = cdata["ranks"]
    c_sector = cdata["sector"]

    logger.info("=" * 110)
    logger.info("SYNTHETIC OSSE COMPILATION: %s - %s (%s)", target_gid, c_name, c_sector)
    logger.info("Centroid: (%.4f°N, %.4f°E) | Constituent Stacks: %s", c_lat, c_lon, c_ranks)
    logger.info("Stride: %d Days (4-Day Sequence + 1-Day Gap) | Span: %s to %s", STRIDE_DAYS, START_DATE_STR, END_DATE_STR)
    logger.info("=" * 110)

    # 1. Setup Cartesian Spatial Mesh (25x25 pixels, 0.5° x 0.5° domain ~55 x 55 km)
    lons_vec = np.linspace(c_lon - DOMAIN_HALF_WIDTH_DEG, c_lon + DOMAIN_HALF_WIDTH_DEG, GRID_RESOLUTION_PIXELS)
    lats_vec = np.linspace(c_lat - DOMAIN_HALF_WIDTH_DEG, c_lat + DOMAIN_HALF_WIDTH_DEG, GRID_RESOLUTION_PIXELS)
    lon_mesh, lat_mesh = np.meshgrid(lons_vec, lats_vec)

    dx_km = (lons_vec - c_lon) * 111.32 * np.cos(np.radians(c_lat))
    dy_km = (lats_vec - c_lat) * 110.57

    cell_area_m2 = float(abs((dx_km[1] - dx_km[0]) * 1000.0 * (dy_km[1] - dy_km[0]) * 1000.0))
    total_domain_area_m2 = cell_area_m2 * (GRID_RESOLUTION_PIXELS ** 2)

    # Setup along-wind / cross-wind coordinate mesh
    x_along_vec = np.linspace(dx_km.min(), dx_km.max(), GRID_RESOLUTION_PIXELS)
    y_cross_vec = np.linspace(dy_km.min(), dy_km.max(), GRID_RESOLUTION_PIXELS)
    x_along_mesh, y_cross_mesh = np.meshgrid(x_along_vec, y_cross_vec)

    # 2. Resolve surveyed stack coordinates from master registry
    stack_offsets_dict: Dict[int, Tuple[float, float]] = {}
    for r in c_ranks:
        if r in MASTER_50_EMITTERS:
            stk_lat = MASTER_50_EMITTERS[r]["lat"]
            stk_lon = MASTER_50_EMITTERS[r]["lon"]
        else:
            stk_lat, stk_lon = c_lat, c_lon

        dx_stk_km = float((stk_lon - c_lon) * 111.32 * np.cos(np.radians(c_lat)))
        dy_stk_km = float((stk_lat - c_lat) * 110.57)
        stack_offsets_dict[r] = (dx_stk_km, dy_stk_km)

    # 3. Load disaggregated CleanSYS daily emissions
    df_daily_em = pd.read_csv(DAILY_EMISSIONS_CSV)
    df_daily_em["Date"] = pd.to_datetime(df_daily_em["Date"])

    # 4. Partition disjoint 4-day sequences with 1-day isolation gap
    valid_sequences = generate_disjoint_5day_sequences(
        START_DATE_STR, END_DATE_STR, seq_len=SEQUENCE_LENGTH_DAYS, gap=BUFFER_GAP_DAYS
    )
    total_seqs = len(valid_sequences)
    logger.info("Generated %d non-overlapping 5-day stride sequences for %s.", total_seqs, target_gid)

    compiled_sequence_count = 0
    t_start_cluster = time.time()

    # 5. Compile sequences across all mission tiers
    for seq_idx, seq_days in enumerate(valid_sequences, start=1):
        daily_wind_speeds: List[float] = []
        daily_t2m_norm: List[float] = []
        daily_solar_rad_norm: List[float] = []
        daily_pblh_norm: List[float] = []
        daily_first_morning_angles: List[float] = []
        daily_last_evening_angles: List[float] = []
        daily_resultant_angles: List[float] = []
        daily_mean_timestamps: List[pd.Timestamp] = []

        day_pure_plumes: List[List[np.ndarray]] = []

        for day_val in seq_days:
            daytime_hours = pd.date_range(start=day_val + pd.Timedelta(hours=9), periods=8, freq="1H")
            
            u_scans, v_scans, theta_scans = [], [], []
            t2m_scans, rad_scans, blh_scans = [], [], []
            scans_pure_vcd = []

            # Ingest facility emissions for this day
            df_day_em = df_daily_em[(df_daily_em["Rank"].isin(c_ranks)) & (df_daily_em["Date"] == day_val)]
            day_stack_emissions: Dict[int, float] = {}
            for r in c_ranks:
                match_r = df_day_em[df_day_em["Rank"] == r]
                if not match_r.empty:
                    day_stack_emissions[r] = float(match_r["SO2_kg_day"].iloc[0]) / 1000.0
                else:
                    day_stack_emissions[r] = float(MASTER_50_EMITTERS.get(r, {}).get("em_t", 1000.0)) / 365.0

            for t_scan in daytime_hours:
                # Query instantaneous surface meteorology and boundary layer height
                t2m_val = get_era5_t2m_fast(t_scan, c_lat, c_lon)
                rad_val = get_era5_solar_radiation_fast(t_scan, c_lat, c_lon)
                blh_val = get_era5_blh_fast(t_scan, c_lat, c_lon)

                # Extract PBLH-adaptive depth-averaged wind
                wind = get_era5_pbl_wind_fast(t_scan, c_lat, c_lon, pblh_m=blh_val)
                u_pbl, v_pbl = wind if wind is not None else (3.5, 1.0)
                
                speed = float(np.sqrt(u_pbl**2 + v_pbl**2))
                theta_w = float(np.degrees(np.arctan2(v_pbl, u_pbl))) if speed >= MIN_WIND_SPEED_MS else 0.0

                t2m_scans.append(t2m_val if t2m_val is not None else 15.0)
                rad_scans.append(rad_val if rad_val is not None else 450.0)
                blh_scans.append(blh_val if blh_val is not None else 850.0)

                u_scans.append(u_pbl)
                v_scans.append(v_pbl)
                theta_scans.append(theta_w)

                # Sub-hourly wind meander perturbation
                meander_deg = float(np.random.normal(0, WIND_MEANDER_STD_DEG))

                # Pure multi-stack plume superposition in along-wind coordinates
                pure_vcd = ddeq_multi_stack_superposition(
                    day_stack_emissions, stack_offsets_dict, c_ranks,
                    u_pbl, v_pbl, rad_val if rad_val is not None else 450.0,
                    x_along_mesh, y_cross_mesh, meander_angle_deg=meander_deg
                )
                scans_pure_vcd.append(pure_vcd)

            day_pure_plumes.append(scans_pure_vcd)
            daily_mean_timestamps.append(pd.Timestamp(np.mean([t.timestamp() for t in daytime_hours]), unit="s"))

            daily_first_morning_angles.append(theta_scans[0])
            daily_last_evening_angles.append(theta_scans[-1])
            mean_u, mean_v = float(np.mean(u_scans)), float(np.mean(v_scans))
            daily_resultant_angles.append(float(np.degrees(np.arctan2(mean_v, mean_u))))

            w_speed_day = float(np.sqrt(mean_u**2 + mean_v**2))
            daily_wind_speeds.append(w_speed_day / 20.0)
            daily_t2m_norm.append(float(np.mean(t2m_scans)) / 40.0)
            daily_solar_rad_norm.append(float(np.mean(rad_scans)) / 1000.0)
            daily_pblh_norm.append(float(np.mean(blh_scans)) / 2000.0)

        # Compute overnight inter-day transport vectors (3 deltas)
        overnight_u_rot: List[float] = []
        overnight_v_rot: List[float] = []
        delta_thetas_norm: List[float] = []

        for i in range(SEQUENCE_LENGTH_DAYS - 1):
            u_geo, v_geo, sp_tr = get_interday_transition_wind(
                daily_mean_timestamps[i], daily_mean_timestamps[i+1], c_lat, c_lon, weighting_power=1.0
            )

            th_ev = np.radians(daily_last_evening_angles[i])
            th_mo = np.radians(daily_first_morning_angles[i+1])
            th_mid_rad = np.arctan2(np.sin(th_ev) + np.sin(th_mo), np.cos(th_ev) + np.cos(th_mo))

            u_r = float(u_geo * np.cos(-th_mid_rad) - v_geo * np.sin(-th_mid_rad))
            v_r = float(u_geo * np.sin(-th_mid_rad) + v_geo * np.cos(-th_mid_rad))
            overnight_u_rot.append(u_r / 20.0)
            overnight_v_rot.append(v_r / 20.0)

            raw_dth = daily_resultant_angles[i+1] - daily_resultant_angles[i]
            wrapped_dth = float((raw_dth + 180.0) % 360.0 - 180.0)
            delta_thetas_norm.append(wrapped_dth / 180.0)

        # Assemble shared 25-feature meteorological vector (Branch B)
        met_vector = np.array(
            daily_wind_speeds + daily_t2m_norm + daily_solar_rad_norm + daily_pblh_norm +
            overnight_u_rot + overnight_v_rot + delta_thetas_norm,
            dtype=np.float32
        )

        # Total 4-day CleanSYS ground truth emissions
        df_seq_em = df_daily_em[
            (df_daily_em["Rank"].isin(c_ranks)) &
            (df_daily_em["Date"] >= seq_days[0]) &
            (df_daily_em["Date"] <= seq_days[-1])
        ]
        q_total_kg = float(df_seq_em["SO2_kg_day"].sum())
        q_total_tonnes = q_total_kg / 1000.0
        y_target_log10 = float(np.log10(q_total_tonnes + 0.1))

        start_str = seq_days[0].strftime("%Y%m%d")
        end_str = seq_days[-1].strftime("%Y%m%d")
        out_filename = f"tensor_{target_gid}_{start_str}_{end_str}_win{seq_idx:04d}.nc"

        # Export simultaneously across all 4 synthetic mission directories
        for tier_key, tier_cfg in NOISE_TIERS.items():
            tier_dir = tier_cfg["dir"]
            sig_shot = tier_cfg["sigma_shot"]
            sig_mat = tier_cfg["sigma_matern"]
            corr_len = tier_cfg["corr_length_km"]
            qa_drop = tier_cfg["qa_dropout"]
            bg_floor = tier_cfg["bg_floor"]
            include_bg = tier_cfg["include_bg"]
            scheme_desc = tier_cfg["scheme"]

            daily_vcd_composites_tier: List[np.ndarray] = []

            for d_idx in range(SEQUENCE_LENGTH_DAYS):
                scans_pure_list = day_pure_plumes[d_idx]
                tier_scans_obs = []

                for pure_vcd_scan in scans_pure_list:
                    if include_bg or sig_shot > 0:
                        obs_vcd = emulate_gems_l2_swath_gridding(
                            pure_vcd_scan, x_along_mesh, y_cross_mesh,
                            sigma_shot=sig_shot, sigma_matern=sig_mat,
                            corr_length_km=corr_len, bg_floor=bg_floor,
                            qa_dropout_rate=qa_drop,
                            native_dx_km=GEMS_NATIVE_DX_KM, native_dy_km=GEMS_NATIVE_DY_KM
                        )
                    else:
                        obs_vcd = pure_vcd_scan

                    tier_scans_obs.append(obs_vcd)

                # Daytime mean composite
                scans_stack = np.stack(tier_scans_obs, axis=0)
                daily_mean_comp = np.mean(scans_stack, axis=0)
                daily_vcd_composites_tier.append(daily_mean_comp)

            # Compute 3 differential deltas (Channels 4, 5, 6) in DU
            tier_deltas = [
                daily_vcd_composites_tier[i+1] - daily_vcd_composites_tier[i]
                for i in range(SEQUENCE_LENGTH_DAYS - 1)
            ]

            # Stack into 7-channel spatial tensor (Branch A)
            tensor_spatial_7ch = np.stack(
                [s.astype(np.float32) for s in daily_vcd_composites_tier] +
                [dg.astype(np.float32) for dg in tier_deltas],
                axis=0
            ).astype(np.float32)

            # Integrate spatial plume mass
            mean_seq_vcd = np.nanmean(np.stack(daily_vcd_composites_tier, axis=0), axis=0)
            m_int_kg = float(np.nansum(mean_seq_vcd) * cell_area_m2 * KG_PER_DU_PER_M2)

            # Construct NetCDF dataset
            ds_out = xr.Dataset(
                data_vars={
                    "spatial_plume_tensor": (["channel", "latitude", "longitude"], tensor_spatial_7ch, {
                        "units": "Dobson Units (DU)",
                        "long_name": "7 Spatial Channels: 4 Daily Snapshots (Ch 0..3) + 3 Differential Deltas (Ch 4..6) in DU"
                    }),
                    "met_vector": (["met_feature"], met_vector, {
                        "units": "Scale-Free Normalized",
                        "long_name": "25-Feature Scale-Free Meteorological & Advection Vector"
                    }),
                    "target_log_emissions": ([], y_target_log10, {
                        "units": "log10(Tonnes + 0.1)",
                        "long_name": "Monotonic Target Label: log10(CleanSYS_Tonnes + 0.1)"
                    }),
                    "target_physical_kg": ([], q_total_kg, {"units": "kg SO2", "long_name": "True CleanSYS Emissions Total in kg"}),
                    "target_physical_tonnes": ([], q_total_tonnes, {"units": "Tonnes SO2", "long_name": "True CleanSYS Emissions Total in Tonnes"}),
                    "m_int_kg": ([], m_int_kg, {"units": "kg SO2", "long_name": "Integrated Spatial Plume Mass in kg"})
                },
                coords={
                    "channel": ["snap_1", "snap_2", "snap_3", "snap_4", "delta_1to2", "delta_2to3", "delta_3to4"],
                    "latitude": lats_vec,
                    "longitude": lons_vec,
                    "met_feature": np.arange(len(met_vector))
                },
                attrs={
                    "cluster_id": target_gid,
                    "cluster_name": c_name,
                    "primary_sector": c_sector,
                    "constituent_ranks": str(c_ranks),
                    "sequence_start_date": seq_days[0].strftime("%Y-%m-%d"),
                    "sequence_end_date": seq_days[-1].strftime("%Y-%m-%d"),
                    "simulation_tier": tier_key,
                    "normalization_scheme": scheme_desc,
                    "spatial_channel_count": 7,
                    "temporal_isolation": "Strictly Disjoint (5-Day Stride: 4-Day Sequence + 1-Day Gap)"
                }
            )
            ds_out.to_netcdf(os.path.join(tier_dir, out_filename))

        compiled_sequence_count += 1

    t_cluster = time.time() - t_start_cluster
    logger.info("CLUSTER %s COMPLETE: Compiled %d sequences (x 4 tiers = %d tensors) in %.1fs.", 
                target_gid, compiled_sequence_count, compiled_sequence_count * 4, t_cluster)
    return compiled_sequence_count

# =============================================================================
# 8. COMMAND LINE INTERFACE & MAIN DISPATCHER
# =============================================================================
def main() -> None:
    """Dispatches sequential compilation or SLURM task array jobs across cluster IDs."""
    parser = argparse.ArgumentParser(description="Parallel Bulk Synthetic OSSE 7-Channel Tensor Compiler")
    parser.add_argument("--cluster_idx", type=int, default=None, help="1-based cluster index (1-17)")
    parser.add_argument("--cluster_id", type=str, default=None, help="Target cluster ID (e.g. G01)")
    parser.add_argument("--clean", action="store_true", help="Force pre-compilation cleaning of output directories")
    args = parser.parse_args()

    task_idx = args.cluster_idx
    if task_idx is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        task_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

    # Clean directories on Task 1 or when explicitly requested
    if args.clean or task_idx == 1:
        logger.info("Pre-compilation directory clean initiated across all 4 synthetic output directories...")
        clean_synthetic_output_directories(list(OUTPUT_DIRS.values()))

    if task_idx is not None:
        if task_idx < 1 or task_idx > len(CLUSTER_KEYS_ORDERED):
            raise ValueError(f"Cluster task index {task_idx} is out of bounds (1 to {len(CLUSTER_KEYS_ORDERED)})")
        target_gid = CLUSTER_KEYS_ORDERED[task_idx - 1]
        compile_cluster_synthetic_tensors(target_gid)
    elif args.cluster_id is not None:
        if args.cluster_id not in CLUSTER_REGISTRY:
            raise KeyError(f"Cluster ID '{args.cluster_id}' not found in master registry.")
        compile_cluster_synthetic_tensors(args.cluster_id)
    else:
        logger.info("Compiling all 17 clusters sequentially across all 4 synthetic output directories...")
        clean_synthetic_output_directories(list(OUTPUT_DIRS.values()))
        total_compiled = sum(compile_cluster_synthetic_tensors(gid) for gid in CLUSTER_KEYS_ORDERED)
        logger.info("==========================================================================================")
        logger.info("BULK SYNTHETIC OSSE COMPILATION COMPLETE: %d sequences compiled across all 17 clusters.", total_compiled)
        for t_k, t_p in OUTPUT_DIRS.items():
            logger.info(" |- Tier '%s': %s", t_k, t_p)
        logger.info("==========================================================================================")


if __name__ == "__main__":
    main()