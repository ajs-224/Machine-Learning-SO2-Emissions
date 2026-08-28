#!/usr/bin/env python
# coding: utf-8

"""
Module: compile_dual_normalization_tensors.py
Description: Parallel batch compiler for multi-day operational GEMS satellite inversion tensors.
             Extracts real GEMS Level-2 column densities and dynamic ERA5 boundary layer meteorology
             across 17 regional clusters (2022-2024), exporting two normalization schemes in a single pass.

Tensor Architecture:
  - Branch A (Spatial Tensor, Shape: [7, 25, 25]):
      * Channels 0..3: 4 daytime wind-aligned column density snapshots (in DU).
      * Channels 4..6: 3 inter-day differential deltas (Delta Omega = Omega_{t+1} - Omega_t in DU).
  - Branch B (Meteorology Vector, Shape: [25]):
      * Scale-free dynamic transport, photochemical, and boundary layer features.

Dual Export Schemes (Simultaneous Single-Pass Output):
  1. Output_Tensor_DualBranch_IntegratedMass/ (Method 1: L1-Spatial PDF Inversion):
      - Spatial channels normalized to domain sum: Omega / sum(Omega).
      - Dimensionless target: Q_CleanSYS / M_int_reg (atmospheric turnover rate).
  2. Output_Tensor_DualBranch_LogScaled/ (Method 2: Physical VCD + Log Inversion):
      - Spatial channels stored in physical column densities (DU).
      - Monotonic target: log10(CleanSYS_Tonnes + 0.1).

Execution:
  - Supports standalone sequential execution or distributed SLURM job array tasks (1 to 17).
"""

import os
import sys
import glob
import re
import time
import json
import atexit
import argparse
import logging
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
from typing import Dict, List, Tuple, Any, Optional
from scipy.interpolate import griddata
from scipy.ndimage import rotate

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("DualExportBatchCompiler")

# =============================================================================
# 1. DIRECTORY PATHS & OUTPUT CONFIGURATION
# =============================================================================
WORKFLOW_DIR: str = "/home/users/ajs224/sheat_mip_gws/code/WindRotatedMLWorkflow/"
GEMS_DIR: str = "/home/users/ajs224/sheat_mip_gws/data/GEMS/"
BLH_DIR: str = "/home/users/ajs224/sheat_mip_gws/data/BLH/"
ERA5_ML_DIR: str = "/badc/ecmwf-era5/data/oper/an_ml/"
ERA5_SFC_AN_DIR: str = "/badc/ecmwf-era5/data/oper/an_sfc/"
ERA5_SFC_FC_DIR: str = "/badc/ecmwf-era5/data/oper/fc_sfc/"
PREPROCESSING_DIR: str = "/home/users/ajs224/sheat_mip_gws/code/MLWorkflow/Output_Preprocessing/"
DAILY_EMISSIONS_CSV: str = "/home/users/ajs224/sheat_mip_gws/code/GroundData/Output_GroundData/daily_emissions_point_sources_2022-24.csv"

# Destination directories for dual-export schemes
OUTPUT_DIR_INT_MASS: str = os.path.join(WORKFLOW_DIR, "Output_Tensor_DualBranch_IntegratedMass/")
OUTPUT_DIR_LOG_SCALED: str = os.path.join(WORKFLOW_DIR, "Output_Tensor_DualBranch_LogScaled/")
os.makedirs(OUTPUT_DIR_INT_MASS, exist_ok=True)
os.makedirs(OUTPUT_DIR_LOG_SCALED, exist_ok=True)

# =============================================================================
# 2. COMPILER HYPERPARAMETERS & PHYSICAL CONSTANTS
# =============================================================================
# --- Temporal Sequence Settings ---
SEQUENCE_LENGTH_DAYS: int = 4                            # Consecutive days per sequence (T = 4)
MIN_SCANS_PER_DAY: int = 5                               # Minimum clear-sky GEMS overpasses required per day
MAX_SPAN_DAYS: int = 4                                   # Maximum calendar span for consecutive sequence
STRIDE_DAYS: int = SEQUENCE_LENGTH_DAYS                  # 4-day non-overlapping stride

# --- Domain & Advection Settings ---
APPLY_BACKGROUND_SUBTRACTION: bool = False               # False = preserve raw VCD (subtracted on-the-fly in training)
DOMAIN_HALF_WIDTH_DEG: float = 0.25                      # Half-width in degrees (~55 x 55 km domain)
GRID_RESOLUTION_PIXELS: int = 25                         # Pixel resolution (25 x 25 grid, 0.02 deg spacing)
MIN_WIND_SPEED_MS: float = 0.5                           # Minimum advection wind threshold (m/s)

# --- Physical Unit Conversion Constants ---
DU_CONVERSION_FACTOR: float = 2.6867e16                  # molecules/cm^2 per Dobson Unit (DU)
KG_PER_DU_PER_M2: float = 2.857e-5                      # 1 DU = 2.857e-5 kg SO2 / m^2

# Internal I/O caching dictionaries
_DIR_CACHE_ML: Dict[str, List[str]] = {}
_DIR_CACHE_AN: Dict[str, List[str]] = {}
_DIR_CACHE_FC: Dict[str, List[str]] = {}
_BLH_CACHE: Dict[int, Optional[Dict[str, Any]]] = {}

# =============================================================================
# 3. FACILITY & CLUSTER REGISTRIES
# =============================================================================
# Master 17 unified regional cluster registry
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


def clean_tensor_directories(directories: List[str]) -> None:
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


def get_nc_variable_fast(root: nc.Dataset, name: str) -> np.ndarray:
    """Recursively queries a NetCDF dataset for a target variable with quality flag fallbacks."""
    target = name.lower()
    for v in root.variables:
        if v.lower() == target:
            return root.variables[v][:]
    for g_name in root.groups:
        grp = root.groups[g_name]
        for v in grp.variables:
            if v.lower() == target:
                return grp.variables[v][:]
    if any(k in target for k in ["quality", "flag", "qa"]):
        for v in root.variables:
            if any(k in v.lower() for k in ["quality", "flag", "qa"]):
                return root.variables[v][:]
        for g_name in root.groups:
            grp = root.groups[g_name]
            for v in grp.variables:
                if any(k in v.lower() for k in ["quality", "flag", "qa"]):
                    return grp.variables[v][:]
    raise KeyError(f"Variable '{name}' not found in NetCDF dataset.")


def extract_gems_time(filename: str) -> pd.Timestamp:
    """Extracts UTC observation timestamp from GEMS Level-2 NetCDF filename."""
    match = re.search(r"L2_(\d{8})_(\d{4})_SO2", filename)
    if match:
        return pd.to_datetime(f"{match.group(1)}T{match.group(2)}", format="%Y%m%dT%H%M")
    return pd.Timestamp("2022-01-01T00:00:00")


def load_gems_vcd_fast(
    nc_path: str,
    center_lat: float,
    center_lon: float,
    lon_mesh: np.ndarray,
    lat_mesh: np.ndarray
) -> Optional[np.ndarray]:
    """Loads raw GEMS SO2 column densities, filters quality flags, and interpolates onto Cartesian grid (in DU)."""
    try:
        with nc.Dataset(nc_path, "r") as root:
            lats = np.ma.filled(get_nc_variable_fast(root, "Latitude"), np.nan)
            lons = np.ma.filled(get_nc_variable_fast(root, "Longitude"), np.nan)
            vcd = np.ma.filled(get_nc_variable_fast(root, "ColumnAmountSO2"), np.nan)
            qa = np.ma.filled(get_nc_variable_fast(root, "VcdQualityFlags"), -9999)

        lats = np.where((lats < -900.0) | (lats > 900.0), np.nan, lats)
        lons = np.where((lons < -900.0) | (lons > 900.0), np.nan, lons)
        vcd = np.where(vcd < -900.0, np.nan, vcd)

        buf = 0.45
        crop_m = (
            (lats >= center_lat - buf) & (lats <= center_lat + buf) &
            (lons >= center_lon - buf) & (lons <= center_lon + buf)
        )
        if not np.any(crop_m):
            return None

        vcd_c, lats_c, lons_c, qa_c = vcd[crop_m], lats[crop_m], lons[crop_m], qa[crop_m]
        valid_m = (qa_c <= 0) & (~np.isnan(vcd_c)) & (~np.isnan(lats_c)) & (~np.isnan(lons_c))
        if np.count_nonzero(valid_m) < 10:
            return None

        pts = np.column_stack((lons_c[valid_m], lats_c[valid_m]))
        vcd_du = vcd_c[valid_m] / DU_CONVERSION_FACTOR

        gridded = griddata(pts, vcd_du, (lon_mesh, lat_mesh), method="linear")
        if np.any(np.isnan(gridded)):
            nan_m = np.isnan(gridded)
            if not np.all(nan_m):
                gridded_nn = griddata(pts, vcd_du, (lon_mesh, lat_mesh), method="nearest")
                gridded[nan_m] = gridded_nn[nan_m]
        return gridded
    except Exception as e_gems:
        logger.debug("Failed loading GEMS scan %s: %s", nc_path, e_gems)
        return None

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


def compute_trimmed_daily_composite(scans_stack: np.ndarray, min_scans: int = 4) -> np.ndarray:
    """Computes pixel-wise outlier-trimmed mean across daytime overpasses (rejects single-hour optical spikes)."""
    n_scans = scans_stack.shape[0]
    if n_scans >= min_scans:
        sorted_scans = np.sort(scans_stack, axis=0)
        return np.nanmean(sorted_scans[1:-1, :, :], axis=0)
    return np.nanmean(scans_stack, axis=0)

# =============================================================================
# 6. DUAL-EXPORT BATCH COMPILER PER CLUSTER
# =============================================================================
def compile_cluster_tensors(target_gid: str) -> int:
    """
    Compiles all non-overlapping GEMS Level-2 observation tensors for a regional cluster.
    Simultaneously exports NetCDF files across both IntegratedMass and LogScaled formats.

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
    logger.info("PROCESSING CLUSTER: %s - %s (%s)", target_gid, c_name, c_sector)
    logger.info("Centroid: (%.4f°N, %.4f°E) | Constituent Stacks: %s", c_lat, c_lon, c_ranks)
    logger.info("Spatial Channels: 7 (4 Snapshots + 3 Deltas) | Stride: %d Days (Strict Isolation)", STRIDE_DAYS)
    logger.info("=" * 110)

    # 1. Setup Cartesian spatial template (25x25 grid, 0.02 deg resolution)
    lons_vec = np.linspace(c_lon - DOMAIN_HALF_WIDTH_DEG, c_lon + DOMAIN_HALF_WIDTH_DEG, GRID_RESOLUTION_PIXELS)
    lats_vec = np.linspace(c_lat - DOMAIN_HALF_WIDTH_DEG, c_lat + DOMAIN_HALF_WIDTH_DEG, GRID_RESOLUTION_PIXELS)
    lon_mesh, lat_mesh = np.meshgrid(lons_vec, lats_vec)

    dx_km = (lons_vec - c_lon) * 111.32 * np.cos(np.radians(c_lat))
    dy_km = (lats_vec - c_lat) * 110.57

    cell_area_m2 = float((np.mean(np.diff(dx_km)) * 1000.0) * (np.mean(np.diff(dy_km)) * 1000.0))
    total_domain_area_m2 = cell_area_m2 * (GRID_RESOLUTION_PIXELS ** 2)

    # 2. Ingest GEMS overpasses matching constituent facility ranks
    merged_cache: Dict[int, List[str]] = {}
    cache_files = sorted(glob.glob(os.path.join(PREPROCESSING_DIR, "gems_valid_dates_cache_batch_*.json")))
    for cf in cache_files:
        try:
            with open(cf, "r") as f_in:
                b_dict = json.load(f_in)
                for r_str, f_list in b_dict.items():
                    merged_cache[int(r_str)] = f_list
        except Exception as e_cache:
            logger.warning("Could not read cache file %s: %s", cf, e_cache)

    cluster_files: List[str] = []
    for r in c_ranks:
        cluster_files.extend(merged_cache.get(r, []))

    cluster_files = sorted(list(set(cluster_files)), key=extract_gems_time)
    df_cluster = pd.DataFrame([
        {"filename": fn, "timestamp": extract_gems_time(fn), "date": extract_gems_time(fn).floor("D")}
        for fn in cluster_files
    ])

    if df_cluster.empty:
        logger.warning("No GEMS overpasses found for %s. Skipping cluster.", target_gid)
        return 0

    # 3. Discover strictly non-overlapping 4-day qualified sequences (>= 5 scans/day)
    daily_counts = df_cluster.groupby("date")["filename"].count()
    qualified_dates = sorted(daily_counts[daily_counts >= MIN_SCANS_PER_DAY].index)

    valid_sequences: List[List[pd.Timestamp]] = []
    if len(qualified_dates) >= SEQUENCE_LENGTH_DAYS:
        i = 0
        last_accepted_end_date = pd.Timestamp("2000-01-01")
        
        while i <= len(qualified_dates) - SEQUENCE_LENGTH_DAYS:
            w_dates = qualified_dates[i : i + SEQUENCE_LENGTH_DAYS]
            start_d, end_d = w_dates[0], w_dates[-1]
            span = (end_d - start_d).days + 1
            
            if span <= MAX_SPAN_DAYS and start_d > last_accepted_end_date:
                valid_sequences.append(w_dates)
                last_accepted_end_date = end_d
                i += SEQUENCE_LENGTH_DAYS
            else:
                i += 1

    total_seqs = len(valid_sequences)
    logger.info("Discovered %d strictly NON-OVERLAPPING 4-day sequences across 2022-2024 for %s.", total_seqs, target_gid)

    if total_seqs == 0:
        return 0

    # 4. Load CleanSYS ground truth daily emissions
    df_daily_em = pd.read_csv(DAILY_EMISSIONS_CSV)
    df_daily_em["Date"] = pd.to_datetime(df_daily_em["Date"])

    compiled_tensor_count = 0
    t_start_cluster = time.time()

    # 5. Process each qualified sequence
    for seq_idx, seq_days in enumerate(valid_sequences, start=1):
        daily_vcd_composites_net: List[np.ndarray] = []
        daily_bg_floors: List[float] = []
        daily_mean_timestamps: List[pd.Timestamp] = []
        daily_wind_speeds: List[float] = []
        daily_t2m_norm: List[float] = []
        daily_solar_rad_norm: List[float] = []
        daily_pblh_norm: List[float] = []
        daily_first_morning_angles: List[float] = []
        daily_last_evening_angles: List[float] = []
        daily_resultant_angles: List[float] = []

        skip_sequence = False
        for day_val in seq_days:
            day_scans = df_cluster[df_cluster["date"] == day_val]["filename"].tolist()
            day_scans = sorted(day_scans, key=extract_gems_time)

            rot_scans, scan_times, u_scans, v_scans, theta_scans = [], [], [], [], []
            t2m_scans, rad_scans, blh_scans = [], [], []

            for fname in day_scans:
                t_val = extract_gems_time(fname)
                g_path = os.path.join(GEMS_DIR, fname)
                if not os.path.exists(g_path):
                    continue

                vcd_2d = load_gems_vcd_fast(g_path, c_lat, c_lon, lon_mesh, lat_mesh)
                if vcd_2d is None or np.all(np.isnan(vcd_2d)):
                    continue

                # Query instantaneous surface meteorology and boundary layer height
                t2m_val = get_era5_t2m_fast(t_val, c_lat, c_lon)
                rad_val = get_era5_solar_radiation_fast(t_val, c_lat, c_lon)
                blh_val = get_era5_blh_fast(t_val, c_lat, c_lon)

                # Extract PBLH-adaptive depth-averaged wind
                wind = get_era5_pbl_wind_fast(t_val, c_lat, c_lon, pblh_m=blh_val)
                if wind is None:
                    continue

                u_pbl, v_pbl = wind
                speed = np.sqrt(u_pbl**2 + v_pbl**2)

                if speed >= MIN_WIND_SPEED_MS:
                    theta_w = float(np.degrees(np.arctan2(v_pbl, u_pbl)))
                    vcd_fill = np.where(np.isnan(vcd_2d), np.nanmedian(vcd_2d), vcd_2d)
                    # Rotate Cartesian grid: axes=(1, 0) locks downwind transport along +x (Eastward)
                    vcd_rot = rotate(vcd_fill, angle=-theta_w, axes=(1, 0), reshape=False, order=1, mode="nearest")
                else:
                    theta_w = 0.0
                    vcd_rot = vcd_2d

                t2m_scans.append(t2m_val if t2m_val is not None else 15.0)
                rad_scans.append(rad_val if rad_val is not None else 450.0)
                blh_scans.append(blh_val if blh_val is not None else 850.0)

                rot_scans.append(vcd_rot)
                scan_times.append(t_val)
                u_scans.append(u_pbl)
                v_scans.append(v_pbl)
                theta_scans.append(theta_w)

            if len(rot_scans) < 3:
                skip_sequence = True
                break

            # Daytime composite via outlier-trimmed mean
            scans_stack = np.stack(rot_scans, axis=0)
            comp_raw = compute_trimmed_daily_composite(scans_stack, min_scans=4)

            # 15th-percentile baseline subtraction (if enabled)
            if APPLY_BACKGROUND_SUBTRACTION:
                bg_floor = float(np.nanpercentile(comp_raw, 15))
                comp_net = np.maximum(0.0, comp_raw - bg_floor)
            else:
                bg_floor = 0.0
                comp_net = comp_raw

            daily_vcd_composites_net.append(comp_net)
            daily_bg_floors.append(bg_floor)
            daily_mean_timestamps.append(pd.Timestamp(np.mean([t.timestamp() for t in scan_times]), unit="s"))

            daily_first_morning_angles.append(theta_scans[0])
            daily_last_evening_angles.append(theta_scans[-1])
            mean_u, mean_v = float(np.mean(u_scans)), float(np.mean(v_scans))
            daily_resultant_angles.append(float(np.degrees(np.arctan2(mean_v, mean_u))))

            w_speed_day = float(np.sqrt(mean_u**2 + mean_v**2))
            mean_t2m_day = float(np.mean(t2m_scans))
            mean_rad_day = float(np.mean(rad_scans))
            mean_blh_day = float(np.mean(blh_scans))

            daily_wind_speeds.append(w_speed_day / 20.0)
            daily_t2m_norm.append(mean_t2m_day / 40.0)
            daily_solar_rad_norm.append(mean_rad_day / 1000.0)
            daily_pblh_norm.append(mean_blh_day / 2000.0)

        if skip_sequence or len(daily_vcd_composites_net) != SEQUENCE_LENGTH_DAYS:
            continue

        # Compute physical differential deltas (Channels 4, 5, 6 in DU)
        raw_net_deltas: List[np.ndarray] = []
        overnight_u_rot: List[float] = []
        overnight_v_rot: List[float] = []
        delta_thetas_norm: List[float] = []

        for i in range(SEQUENCE_LENGTH_DAYS - 1):
            raw_net_deltas.append(daily_vcd_composites_net[i+1] - daily_vcd_composites_net[i])

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

        # Spatial mass integration
        mean_seq_net_vcd = np.nanmean(np.stack(daily_vcd_composites_net, axis=0), axis=0)
        sum_net_vcd_du = float(np.nansum(mean_seq_net_vcd))
        m_int_kg = float(sum_net_vcd_du * cell_area_m2 * KG_PER_DU_PER_M2)
        m_floor_kg = float(0.02 * total_domain_area_m2 * KG_PER_DU_PER_M2)
        m_int_reg_kg = float(max(m_int_kg, m_floor_kg))

        start_str = seq_days[0].strftime("%Y%m%d")
        end_str = seq_days[-1].strftime("%Y%m%d")
        out_filename = f"tensor_{target_gid}_{start_str}_{end_str}_win{seq_idx:04d}.nc"

        # =========================================================================
        # SCHEME 1: Output_Tensor_DualBranch_IntegratedMass/ (L1-PDF Inversion)
        # =========================================================================
        norm_denom_du = float(max(sum_net_vcd_du, 1e-4))
        norm_base_pdf_list = [(s / norm_denom_du).astype(np.float32) for s in daily_vcd_composites_net]
        norm_deltas_pdf_list = [(dg / norm_denom_du).astype(np.float32) for dg in raw_net_deltas]
        tensor_spatial_l1 = np.stack(norm_base_pdf_list + norm_deltas_pdf_list, axis=0).astype(np.float32)

        y_target_flushing_rate = float(q_total_kg / m_int_reg_kg)

        ds_out_l1 = xr.Dataset(
            data_vars={
                "spatial_plume_tensor": (["channel", "latitude", "longitude"], tensor_spatial_l1, {
                    "units": "L1-Normalized Probability Density",
                    "long_name": "7 Spatial Channels: 4 Daily Snapshots (Ch 0..3) + 3 Differential Deltas (Ch 4..6)"
                }),
                "met_vector": (["met_feature"], met_vector, {
                    "units": "Scale-Free Normalized",
                    "long_name": "25-Feature Scale-Free Meteorological & Advection Vector"
                }),
                "target_dimensionless": ([], y_target_flushing_rate, {
                    "units": "sequence^-1",
                    "long_name": "Flushing Rate Target (Q_CleanSYS / M_int_reg)"
                }),
                "target_physical_kg": ([], q_total_kg, {"units": "kg SO2", "long_name": "True CleanSYS Emissions Total"}),
                "m_int_reg_kg": ([], m_int_reg_kg, {"units": "kg SO2", "long_name": "Integrated Column Mass for Physical Inversion"}),
                "vcd_sum_net_du": ([], sum_net_vcd_du, {"units": "DU", "long_name": "Domain Integrated Net Column Density Sum"})
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
                "normalization_scheme": "Method 1: L1-Spatial PDF & Dimensionless Flushing Rate",
                "spatial_channel_count": 7,
                "temporal_isolation": "Strictly Non-Overlapping (Stride = 4 Days)"
            }
        )
        ds_out_l1.to_netcdf(os.path.join(OUTPUT_DIR_INT_MASS, out_filename))

        # =========================================================================
        # SCHEME 2: Output_Tensor_DualBranch_LogScaled/ (Physical VCD + Log Inversion)
        # =========================================================================
        tensor_spatial_phys = np.stack(
            [s.astype(np.float32) for s in daily_vcd_composites_net] + 
            [dg.astype(np.float32) for dg in raw_net_deltas],
            axis=0
        ).astype(np.float32)

        y_target_log10 = float(np.log10(q_total_tonnes + 0.1))

        ds_out_log = xr.Dataset(
            data_vars={
                "spatial_plume_tensor": (["channel", "latitude", "longitude"], tensor_spatial_phys, {
                    "units": "Dobson Units (DU)",
                    "long_name": "7 Physical Spatial Channels: 4 Daily Snapshots (Ch 0..3) + 3 Differential Deltas (Ch 4..6) in DU"
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
                "normalization_scheme": "Method 2: Physical DU Columns & Monotonic Log10 Target",
                "spatial_channel_count": 7,
                "temporal_isolation": "Strictly Non-Overlapping (Stride = 4 Days)"
            }
        )
        ds_out_log.to_netcdf(os.path.join(OUTPUT_DIR_LOG_SCALED, out_filename))

        compiled_tensor_count += 1

    t_cluster = time.time() - t_start_cluster
    logger.info("CLUSTER %s COMPLETE: Compiled %d dual-export tensors in %.1fs.", target_gid, compiled_tensor_count, t_cluster)
    return compiled_tensor_count

# =============================================================================
# 7. COMMAND LINE INTERFACE & MAIN DISPATCHER
# =============================================================================
def main() -> None:
    """Dispatches sequential compilation or SLURM task array jobs across cluster IDs."""
    parser = argparse.ArgumentParser(description="Parallel Dual-Export 7-Channel Tensor Batch Compiler")
    parser.add_argument("--cluster_idx", type=int, default=None, help="1-based cluster index (1-17)")
    parser.add_argument("--cluster_id", type=str, default=None, help="Target cluster ID (e.g. G01)")
    parser.add_argument("--clean", action="store_true", help="Force pre-compilation cleaning of output directories")
    args = parser.parse_args()

    task_idx = args.cluster_idx
    if task_idx is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        task_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

    # Clean previous tensors on Task 1 or when explicitly requested
    if args.clean or task_idx == 1:
        logger.info("Pre-compilation directory clean initiated...")
        clean_tensor_directories([OUTPUT_DIR_INT_MASS, OUTPUT_DIR_LOG_SCALED])

    if task_idx is not None:
        if task_idx < 1 or task_idx > len(CLUSTER_KEYS_ORDERED):
            raise ValueError(f"Cluster task index {task_idx} is out of bounds (1 to {len(CLUSTER_KEYS_ORDERED)})")
        target_gid = CLUSTER_KEYS_ORDERED[task_idx - 1]
        compile_cluster_tensors(target_gid)
    elif args.cluster_id is not None:
        if args.cluster_id not in CLUSTER_REGISTRY:
            raise KeyError(f"Cluster ID '{args.cluster_id}' not found in master registry.")
        compile_cluster_tensors(args.cluster_id)
    else:
        logger.info("Compiling all 17 clusters across both IntegratedMass and LogScaled formats...")
        clean_tensor_directories([OUTPUT_DIR_INT_MASS, OUTPUT_DIR_LOG_SCALED])
        total_compiled = sum(compile_cluster_tensors(gid) for gid in CLUSTER_KEYS_ORDERED)
        logger.info("==========================================================================================")
        logger.info("DUAL-EXPORT COMPILATION COMPLETE: %d tensors written to:", total_compiled)
        logger.info(" 1. %s", OUTPUT_DIR_INT_MASS)
        logger.info(" 2. %s", OUTPUT_DIR_LOG_SCALED)
        logger.info("==========================================================================================")


if __name__ == "__main__":
    main()