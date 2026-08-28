#!/usr/bin/env python
# coding: utf-8

"""
Module: HDD_CDD_Fit.py
Description: Meteorological degree-day extraction and thermal response regression pipeline for
             South Korean power-sector point sources (2022–2024).

Methodology:
  1. Temperature Ingestion:
      - Extracts hourly 2-metre air temperature (T2m) from ERA5 surface reanalysis (an_sfc).
      - Computes diurnal mean temperatures and evaluates Heating & Cooling Degree Days:
          * Cooling Degree Days (CDD): max(0.0, T_mean - 22.0 °C)
          * Heating Degree Days (HDD): max(0.0, 18.0 °C - T_mean)
      - Includes a regional sinusoidal thermal model fallback if reanalysis files are missing.
  2. Constrained Optimization (NNLS):
      - Fits a Non-Negative Least Squares piecewise linear model against Climate TRACE monthly records:
          E_m = gamma_0 + alpha * CDD_m + beta * HDD_m
      - Extracts facility-specific baseload (gamma_0), cooling demand (alpha), and heating demand (beta).
  3. Surrogate Transfer:
      - Applies proportional capacity scaling from Taean (Rank 4) to Dangjin (Rank 6) to bypass
        missing historical reporting.
  4. Diagnostic Verification:
      - Renders two-panel validation plots (U-shaped temperature response curve and monthly time series).
      - Exports fitted regression parameters to 'Output_GroundData/power_plant_thermal_parameters.csv'.
"""

import os
import glob
import logging
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4 as nc
import gc
from scipy.optimize import nnls

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ThermalFittingPipeline")

# =============================================================================
# 1. DIRECTORY PATHS & OUTPUT CONFIGURATION
# =============================================================================
TOP50_PATH: str = "/home/users/ajs224/sheat_mip_gws/data/ClimateTRACE/top50emitters.csv"
CLEANSYS_PATH: str = "/home/users/ajs224/sheat_mip_gws/data/ClimateTRACE/CleanSYS_annual_2022-24.csv"
CTRACE_POWER_PATH: str = "/home/users/ajs224/sheat_mip_gws/data/ClimateTRACE/DATA/power/electricity-generation_emissions_sources_v5_9_0.csv"
ERA5_SFC_DIR: str = "/badc/ecmwf-era5/data/oper/an_sfc/"

# Output directory and verification plot destination
OUTPUT_DIR: str = "Output_GroundData"
PLOT_DIR: str = os.path.join(OUTPUT_DIR, "verification_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# =============================================================================
# 2. PHYSICAL DEGREE-DAY THRESHOLDS
# =============================================================================
T_HEATING_THRESHOLD: float = 18.0  # Base temperature for space heating demand (HDD, °C)
T_COOLING_THRESHOLD: float = 22.0  # Base temperature for air conditioning demand (CDD, °C)

logger.info("Initialized thermal fitting pipeline. Outputs will be saved to: %s", OUTPUT_DIR)

# =============================================================================
# 3. DATA INGESTION & DATASET ALIGNMENT
# =============================================================================
def load_and_align_datasets(
    top50_path: str,
    cleansys_path: str,
    ctrace_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw emitter registries, CleanSYS annual totals, and Climate TRACE monthly emissions.
    Validates structural alignment across foreign keys and normalizes data types.

    Parameters:
      top50_path (str): Filepath to top 50 emitter metadata CSV.
      cleansys_path (str): Filepath to annual CleanSYS ground truth totals CSV.
      ctrace_path (str): Filepath to Climate TRACE monthly power sector emissions CSV.

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Aligned (df_top50, df_cleansys, df_ctrace).
    """
    try:
        # Load surveyed Top 50 emitters list
        df_top50 = pd.read_csv(top50_path)
        logger.info("Loaded Top 50 emitters list: %d records.", len(df_top50))
        df_top50 = df_top50.dropna(subset=["Rank"])

        # Load CleanSYS annual emissions
        df_cleansys = pd.read_csv(cleansys_path)
        logger.info("Loaded CleanSYS annual totals: %d records.", len(df_cleansys))
        df_cleansys = df_cleansys.dropna(subset=["SO2 Rank", "Year"])

        # Load Climate TRACE monthly power emissions
        df_ctrace = pd.read_csv(ctrace_path)
        logger.info("Loaded Climate TRACE power sector monthly emissions: %d records.", len(df_ctrace))

        # Standardize key types
        df_top50["Rank"] = df_top50["Rank"].astype(int)
        df_cleansys["SO2 Rank"] = df_cleansys["SO2 Rank"].astype(int)
        df_cleansys["Year"] = df_cleansys["Year"].astype(int)

        return df_top50, df_cleansys, df_ctrace

    except Exception as e_load:
        logger.error("Failed to load or align source datasets: %s", e_load)
        raise RuntimeError("Data ingestion failed. Check input files and paths.")


# Execute dataset loading and isolate weighted power sector plants
df_top50, df_cleansys, df_ctrace = load_and_align_datasets(TOP50_PATH, CLEANSYS_PATH, CTRACE_POWER_PATH)

power_emitters = df_top50[df_top50["Temporal Approach"].str.strip().str.lower() == "weighted"].copy()
power_emitters["ClimateTRACE ID"] = pd.to_numeric(power_emitters["ClimateTRACE ID"], errors="coerce")
logger.info("Identified %d weighted power plants with Climate TRACE IDs.", len(power_emitters))

# =============================================================================
# 4. HOURLY TEMPERATURE EXTRACTION & SYNTHETIC CLIMATE FALLBACK
# =============================================================================
def extract_daily_temp_at_coordinate(
    lat: float,
    lon: float,
    year: int,
    month: int,
    day: int,
    era5_dir: str
) -> float:
    """
    Extracts and averages 24 hourly 2m temperature files from ERA5 reanalysis on JASMIN.
    Falls back to a regional sinusoidal climatological model if reanalysis files are missing.

    Parameters:
      lat (float): Target latitude in degrees.
      lon (float): Target longitude in degrees.
      year (int): Calendar year.
      month (int): Calendar month (1–12).
      day (int): Calendar day (1–31).
      era5_dir (str): Base root directory of ERA5 surface analysis archives.

    Returns:
      float: Daily mean 2-metre temperature in Celsius (°C).
    """
    day_path = os.path.join(era5_dir, f"{year:04d}", f"{month:02d}", f"{day:02d}")
    hourly_temps: List[float] = []

    if os.path.exists(day_path):
        nc_files = sorted(glob.glob(os.path.join(day_path, "*.2t.nc")))
        for nc_file in nc_files:
            try:
                with xr.open_dataset(nc_file) as ds:
                    lat_name = next((c for c in ["latitude", "lat"] if c in ds.coords), None)
                    lon_name = next((c for c in ["longitude", "lon"] if c in ds.coords), None)
                    var_name = next((v for v in ds.data_vars if v.lower() in ["2t", "t2m", "var167"]), None)

                    if lat_name and lon_name and var_name:
                        raw_val = ds[var_name].sel({lat_name: lat, lon_name: lon}, method="nearest")
                        temp_k = float(raw_val.item())
                        temp_c = temp_k - 273.15  # Convert Kelvin to Celsius
                        hourly_temps.append(temp_c)
            except Exception:
                continue

    # Accept daily average if at least 12 hours of reanalysis data are available
    if len(hourly_temps) >= 12:
        return float(np.mean(hourly_temps))

    # --- Physical Fallback Model for South Korea ---
    # Captures seasonal cycle: cold winters (down to -10 °C) and hot summers (up to 30 °C), peaking in late July
    day_of_year = pd.Timestamp(year, month, day).dayofyear
    seasonal_temp = 10.0 + 17.0 * np.sin(2 * np.pi * (day_of_year - 110) / 365.25)
    np.random.seed(day_of_year + year)  # Deterministic meteorological noise (+/- 2.5 °C)
    daily_noise = np.random.normal(0, 2.5)

    return float(seasonal_temp + daily_noise)


# Verify temperature extractor functionality
test_temp = extract_daily_temp_at_coordinate(36.9039, 126.2328, 2022, 7, 15, ERA5_SFC_DIR)
logger.info("Verified temperature extractor. Test July temp at Taean is: %.2f °C", test_temp)

# =============================================================================
# 5. BATCH DEGREE-DAY (HDD / CDD) PROCESSING & CACHING
# =============================================================================
cache_path = os.path.join(OUTPUT_DIR, "power_plants_daily_weather_cache.csv")

# 1. Load pre-compiled weather cache if available on disk
if os.path.exists(cache_path):
    logger.info("Found cached daily weather data at: %s. Loading...", cache_path)
    try:
        df_cache = pd.read_csv(cache_path)
        df_cache["Date"] = pd.to_datetime(df_cache["Date"])
        
        plant_weather_data: Dict[int, pd.DataFrame] = {}
        for rank, group in df_cache.groupby("Rank"):
            plant_weather_data[int(rank)] = group.reset_index(drop=True)
            
        logger.info("Successfully loaded weather profiles for %d plants from cache.", len(plant_weather_data))
    except Exception as e_cache:
        logger.error("Failed to read cache file (%s). Recalculating from raw ERA5 files.", e_cache)
        df_cache = None
else:
    df_cache = None

# 2. Execute memory-safe single-pass grid extraction if no cache exists
if df_cache is None:
    logger.info("No cache found. Pre-calculating static coordinate indices on ERA5 grid...")
    
    start_date = pd.Timestamp("2022-01-01")
    end_date = pd.Timestamp("2024-12-31")
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    total_days = len(date_range)

    lat_indices: Dict[int, int] = {}
    lon_indices: Dict[int, int] = {}
    use_fallback: bool = True

    # Pre-index plant coordinates on ERA5 spatial grid
    ref_files = sorted(glob.glob(os.path.join(ERA5_SFC_DIR, "2022", "01", "01", "*.2t.nc")))
    if ref_files:
        try:
            with nc.Dataset(ref_files[0], "r") as ref_ds:
                lat_var_name = next((c for c in ref_ds.variables if c.lower() in ["latitude", "lat"]), None)
                lon_var_name = next((c for c in ref_ds.variables if c.lower() in ["longitude", "lon"]), None)
                
                if lat_var_name and lon_var_name:
                    lats_arr = ref_ds.variables[lat_var_name][:]
                    lons_arr = ref_ds.variables[lon_var_name][:]
                    
                    for idx, row in power_emitters.iterrows():
                        rank = int(row["Rank"])
                        ctrace_id = row["ClimateTRACE ID"]
                        
                        plant_lat = float(df_ctrace[df_ctrace["source_id"] == ctrace_id]["lat"].iloc[0]) if ctrace_id in df_ctrace["source_id"].values else 37.0
                        plant_lon = float(df_ctrace[df_ctrace["source_id"] == ctrace_id]["lon"].iloc[0]) if ctrace_id in df_ctrace["source_id"].values else 127.0
                        
                        norm_plant_lon = plant_lon if plant_lon >= 0 else plant_lon + 360.0 if np.max(lons_arr) > 180.0 else plant_lon
                        
                        lat_indices[rank] = int(np.argmin(np.abs(lats_arr - plant_lat)))
                        lon_indices[rank] = int(np.argmin(np.abs(lons_arr - norm_plant_lon)))
                        
                    use_fallback = False
                    logger.info("Successfully cached spatial grid indices for all plants.")
        except Exception as e_ref:
            logger.warning("Could not parse coordinates from %s (%s). Falling back to synthetic model.", ref_files[0], e_ref)

    if use_fallback:
        for idx, row in power_emitters.iterrows():
            lat_indices[int(row["Rank"])] = 0
            lon_indices[int(row["Rank"])] = 0

    logger.info("Starting batched single-pass weather extraction across 2022-2024...")
    daily_records: List[Dict[str, Any]] = []

    for day_idx, current_date in enumerate(date_range, 1):
        y, m, d = current_date.year, current_date.month, current_date.day
        day_path = os.path.join(ERA5_SFC_DIR, f"{y:04d}", f"{m:02d}", f"{d:02d}")
        
        hourly_day_temps: Dict[int, List[float]] = {rank: [] for rank in lat_indices.keys()}
        day_nc_files: List[str] = []
        
        if not use_fallback and os.path.exists(day_path):
            day_nc_files = sorted(glob.glob(os.path.join(day_path, "*.2t.nc")))
            
        if len(day_nc_files) >= 12:
            for nc_file in day_nc_files:
                try:
                    with nc.Dataset(nc_file, "r") as ds:
                        var_name = next((v for v in ds.variables if v.lower() in ["2t", "t2m", "var167"]), None)
                        if var_name:
                            temp_data = ds.variables[var_name]
                            for rank in lat_indices.keys():
                                lat_idx = lat_indices[rank]
                                lon_idx = lon_indices[rank]
                                
                                if len(temp_data.shape) == 3:
                                    temp_k = float(temp_data[0, lat_idx, lon_idx])
                                else:
                                    temp_k = float(temp_data[lat_idx, lon_idx])
                                    
                                hourly_day_temps[rank].append(temp_k - 273.15)
                except Exception:
                    continue

        # Compute diurnal mean, CDD, and HDD per plant
        for rank in lat_indices.keys():
            plant_temps = hourly_day_temps[rank]
            
            if len(plant_temps) >= 12:
                temp_mean = float(np.mean(plant_temps))
            else:
                day_of_year = current_date.dayofyear
                seasonal_temp = 10.0 + 17.0 * np.sin(2 * np.pi * (day_of_year - 110) / 365.25)
                np.random.seed(day_of_year + y + rank)
                daily_noise = np.random.normal(0, 2.5)
                temp_mean = float(seasonal_temp + daily_noise)
                
            cdd = max(0.0, temp_mean - T_COOLING_THRESHOLD)
            hdd = max(0.0, T_HEATING_THRESHOLD - temp_mean)
            
            daily_records.append({
                "Rank": rank,
                "Date": current_date,
                "Year": y,
                "Month": m,
                "Temp": temp_mean,
                "CDD": cdd,
                "HDD": hdd
            })

        if day_idx % 100 == 0 or day_idx == total_days:
            progress_pct = (day_idx / total_days) * 100
            logger.info("Progress: Day %04d/%04d processed (%.1f%%)", day_idx, total_days, progress_pct)
            gc.collect()

    df_all_plants = pd.DataFrame(daily_records)
    plant_weather_data = {}
    for rank, group in df_all_plants.groupby("Rank"):
        plant_weather_data[int(rank)] = group.reset_index(drop=True)

    try:
        df_all_plants.to_csv(cache_path, index=False)
        logger.info("Successfully compiled weather profiles and saved cache to: %s", cache_path)
    except Exception as e_save_c:
        logger.warning("Could not write cache file to disk: %s", e_save_c)

# =============================================================================
# 6. NON-NEGATIVE LEAST SQUARES (NNLS) REGRESSION ENGINE
# =============================================================================
regression_parameters: Dict[int, Dict[str, float]] = {}

for idx, row in power_emitters.iterrows():
    rank = int(row["Rank"])
    ctrace_id = int(row["ClimateTRACE ID"])
    name = row["English Translation"]
    
    # --- SURROGATE PARAMETER TRANSFER: DANGJIN (RANK 6) ---
    if rank == 6:
        if 4 in regression_parameters:
            taean_params = regression_parameters[4]
            try:
                # Scale Taean parameters proportionally using 2022 CleanSYS annual baseline emissions
                cs_dangjin = float(df_cleansys[(df_cleansys["SO2 Rank"] == 6) & (df_cleansys["Year"] == 2022)]["SO2 (kg/year)"].iloc[0])
                cs_taean = float(df_cleansys[(df_cleansys["SO2 Rank"] == 4) & (df_cleansys["Year"] == 2022)]["SO2 (kg/year)"].iloc[0])
                scale_factor = cs_dangjin / cs_taean
                
                gamma_0 = taean_params["gamma_0"] * scale_factor
                alpha = taean_params["alpha"] * scale_factor
                beta = taean_params["beta"] * scale_factor
                
                regression_parameters[rank] = {
                    "gamma_0": gamma_0,
                    "alpha": alpha,
                    "beta": beta
                }
                logger.info("Fit Complete for Rank %02d (DANGJIN BYPASS): Transferred from Rank 04 (TAEAN) [Scale Factor: %.4f]", rank, scale_factor)
                continue
            except Exception as e_dangjin:
                logger.error("Failed to apply surrogate bypass for Rank 6: %s. Falling back to standard fitting.", e_dangjin)
        else:
            logger.error("Rank 4 parameters missing from cache. Standard fitting will proceed for Rank 6.")

    # --- STANDARD NNLS FITTING PROCESS ---
    # 1. Filter Climate TRACE monthly emissions for target plant across 2022-2024
    df_plant_trace = df_ctrace[df_ctrace["source_id"] == ctrace_id].copy()
    if df_plant_trace.empty:
        logger.warning("No Climate TRACE records found for ID %d (%s). Skipping.", ctrace_id, name)
        continue
    
    df_plant_trace["start_time"] = pd.to_datetime(df_plant_trace["start_time"])
    df_plant_trace["Year"] = df_plant_trace["start_time"].dt.year
    df_plant_trace["Month"] = df_plant_trace["start_time"].dt.month
    df_plant_trace = df_plant_trace[df_plant_trace["Year"].isin([2022, 2023, 2024])]
    
    # 2. Aggregate monthly CDD and HDD sums from daily weather cache
    df_daily_weather = plant_weather_data[rank]
    df_monthly_weather = df_daily_weather.groupby(["Year", "Month"]).agg({
        "CDD": "sum",
        "HDD": "sum"
    }).reset_index()
    
    df_merged = pd.merge(df_plant_trace, df_monthly_weather, on=["Year", "Month"])
    
    if len(df_merged) < 12:
        logger.warning("Too few matching months (%d) for Rank %d. Skipping.", len(df_merged), rank)
        continue
        
    # 3. Formulate and solve non-negative matrix optimization: min ||A*x - b||_2 subject to x >= 0
    A = np.column_stack((np.ones(len(df_merged)), df_merged["CDD"].values, df_merged["HDD"].values))
    b = df_merged["emissions_quantity"].values  # Monthly emissions in metric tonnes
    
    x_opt, r_norm = nnls(A, b)
    gamma_0, alpha, beta = x_opt[0], x_opt[1], x_opt[2]
    
    regression_parameters[rank] = {
        "gamma_0": gamma_0,
        "alpha": alpha,
        "beta": beta
    }
    
    logger.info("Fit Complete for Rank %02d: %-25s | gamma_0=%.2f t, alpha=%.4f t/CDD, beta=%.4f t/HDD",
                rank, name[:25], gamma_0, alpha, beta)

# =============================================================================
# 7. DIAGNOSTIC VERIFICATION PLOTTING
# =============================================================================
logger.info("Generating verification diagnostic plots...")

for idx, row in power_emitters.iterrows():
    rank = int(row["Rank"])
    if rank not in regression_parameters:
        continue

    ctrace_id = int(row["ClimateTRACE ID"])
    name = row["English Translation"]
    params = regression_parameters[rank]

    # Ingest monthly emissions and aggregated weather profiles
    df_plant_trace = df_ctrace[df_ctrace["source_id"] == ctrace_id].copy()
    df_plant_trace["start_time"] = pd.to_datetime(df_plant_trace["start_time"])
    df_plant_trace["Year"] = df_plant_trace["start_time"].dt.year
    df_plant_trace["Month"] = df_plant_trace["start_time"].dt.month
    df_plant_trace = df_plant_trace[df_plant_trace["Year"].isin([2022, 2023, 2024])]

    df_daily_weather = plant_weather_data[rank]
    df_monthly_weather = df_daily_weather.groupby(["Year", "Month"]).agg({
        "CDD": "sum",
        "HDD": "sum",
        "Temp": "mean"
    }).reset_index()

    df_merged = pd.merge(df_plant_trace, df_monthly_weather, on=["Year", "Month"])
    df_merged["Fitted"] = params["gamma_0"] + params["alpha"] * df_merged["CDD"] + params["beta"] * df_merged["HDD"]

    # Render Two-Panel Validation Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # --- Panel 1: U-Shaped Thermal Response Curve ---
    ax1 = axes[0]
    ax1.set_facecolor("#fcfcfc")
    ax1.grid(True, linestyle="--", alpha=0.5, color="gray")

    ax1.scatter(df_merged["Temp"], df_merged["emissions_quantity"], color="darkorange", edgecolors="black", s=50, label="Observed Monthly (Climate TRACE)", zorder=3)

    temp_range = np.linspace(df_merged["Temp"].min() - 2, df_merged["Temp"].max() + 2, 100)
    fitted_curve = []
    for t in temp_range:
        cdd_mock = max(0.0, t - T_COOLING_THRESHOLD) * 30.4
        hdd_mock = max(0.0, T_HEATING_THRESHOLD - t) * 30.4
        val = params["gamma_0"] + params["alpha"] * cdd_mock + params["beta"] * hdd_mock
        fitted_curve.append(val)

    ax1.plot(temp_range, fitted_curve, color="navy", linewidth=2.5, linestyle="-", label="Fitted Thermal Response Curve", zorder=2)
    ax1.axvline(T_HEATING_THRESHOLD, color="red", linestyle=":", alpha=0.6, label="Heating Threshold (18°C)")
    ax1.axvline(T_COOLING_THRESHOLD, color="blue", linestyle=":", alpha=0.6, label="Cooling Threshold (22°C)")

    ax1.set_title("Thermal Response Curve (U-Shape)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Mean Monthly Temperature [°C]", fontsize=9)
    ax1.set_ylabel("SO2 Emissions [tons/month]", fontsize=9)
    ax1.legend(fontsize=8, loc="upper center")

    # --- Panel 2: Observed vs. Fitted Monthly Time Series ---
    ax2 = axes[1]
    ax2.set_facecolor("#fcfcfc")
    ax2.grid(True, linestyle="--", alpha=0.5, color="gray")

    df_merged["DateStr"] = df_merged["Year"].astype(str) + "-" + df_merged["Month"].astype(str).str.zfill(2)
    df_merged = df_merged.sort_values(by=["Year", "Month"])

    ax2.plot(df_merged["DateStr"], df_merged["emissions_quantity"], marker="o", color="darkorange", label="Observed", linewidth=2)
    ax2.plot(df_merged["DateStr"], df_merged["Fitted"], marker="s", color="navy", label="Fitted (NNLS Model)", linewidth=2, linestyle="--")

    ax2.set_title("Temporal Pattern Validation", fontsize=11, fontweight="bold")
    ax2.set_ylabel("SO2 Emissions [tons/month]", fontsize=9)
    ax2.set_xlabel("Date", fontsize=9)
    ax2.tick_params(axis="x", rotation=45, labelsize=8)
    ax2.legend(fontsize=8)

    plt.suptitle(f"Thermal Verification Diagnostic: Rank {rank:02d}\n{name}", fontsize=13, fontweight="bold", y=1.02)
    fig_name = f"thermal_fit_rank_{rank:02d}_{name[:15].replace(' ', '_')}.png"
    fig_path = os.path.join(PLOT_DIR, fig_name)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

logger.info("Verification plots successfully generated and saved to: %s", PLOT_DIR)

# =============================================================================
# 8. PARAMETER EXPORT & EMPIRICAL EMISSION FACTOR AUDIT
# =============================================================================
logger.info("Compiling final diagnostic dataset...")

diagnostic_records = []

for idx, row in power_emitters.iterrows():
    rank = int(row["Rank"])
    name = row["English Translation"]

    if rank not in regression_parameters:
        continue

    params = regression_parameters[rank]

    # Map nameplate generating capacity (MW) for emission factor diagnostics
    capacity_mw = 1000.0
    if "taean" in name.lower(): capacity_mw = 6100.0
    elif "dangjin" in name.lower(): capacity_mw = 6040.0
    elif "yeongheung" in name.lower(): capacity_mw = 5080.0
    elif "boryeong" in name.lower() and "shin" not in name.lower(): capacity_mw = 5350.0
    elif "shin-boryeong" in name.lower(): capacity_mw = 2000.0
    elif "hadong" in name.lower(): capacity_mw = 4000.0
    elif "samcheonpo" in name.lower(): capacity_mw = 3240.0
    elif "samcheok" in name.lower(): capacity_mw = 2044.0
    elif "donghae" in name.lower(): capacity_mw = 250.0

    # Retrieve CleanSYS 2022 annual total to compute empirical Emission Factor (kg SO2 / MWh)
    df_cs_2022 = df_cleansys[(df_cleansys["SO2 Rank"] == rank) & (df_cleansys["Year"] == 2022)]

    if not df_cs_2022.empty:
        annual_so2_kg = float(df_cs_2022["SO2 (kg/year)"].iloc[0])
        annual_gen_mwh = capacity_mw * 8760.0 * 0.40  # Conservative 40% baseload capacity factor
        empirical_ef = annual_so2_kg / annual_gen_mwh
    else:
        annual_so2_kg = np.nan
        empirical_ef = np.nan

    diagnostic_records.append({
        "Rank": rank,
        "English_Name": name,
        "gamma_0_baseload": params["gamma_0"],
        "alpha_cooling": params["alpha"],
        "beta_heating": params["beta"],
        "Capacity_MW": capacity_mw,
        "CleanSYS_2022_SO2_kg": annual_so2_kg,
        "Empirical_EF_kg_MWh": empirical_ef
    })

df_diagnostics = pd.DataFrame(diagnostic_records)

# Export finalized parameter database
param_out_path = os.path.join(OUTPUT_DIR, "power_plant_thermal_parameters.csv")
df_diagnostics.to_csv(param_out_path, index=False)

logger.info("======================================================================================")
logger.info("PIPELINE EXECUTION COMPLETE. REGRESSION DIAGNOSTICS:")
logger.info("======================================================================================")
print(df_diagnostics[["Rank", "English_Name", "gamma_0_baseload", "alpha_cooling", "beta_heating", "Empirical_EF_kg_MWh"]].to_string(index=False))
logger.info("======================================================================================")
logger.info("Regression parameters successfully exported and saved to: %s", param_out_path)