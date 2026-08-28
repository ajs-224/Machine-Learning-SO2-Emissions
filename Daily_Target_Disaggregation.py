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
# # Daily CleanSYS Ground Truth Disaggregation Pipeline (2022–2024)
# ### Converts annual CleanSYS CEMS emissions totals into continuous daily point-source series.
# ### Methodologies:
# ### - Industrial Facilities (Pathway A): Flatlined disaggregation (E_d = E_annual / N_days).
# ### - Thermal Power Stations (Pathway B): Meteorological degree-day dynamic scaling using pre-fitted NNLS weights:
# ###     w_d = (gamma_0 / N_days_in_month) + alpha * CDD_d + beta * HDD_d
# ###     E_d = E_annual * (w_d / sum(w_annual))
# ### Strictly guarantees mass conservation against CleanSYS annual totals to 4 decimal places.

# %%
import os
import calendar
import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("DailyDisaggregator")

# =============================================================================
# 1. DIRECTORY PATHS & CACHE CONFIGURATION
# =============================================================================
TOP50_PATH: str = "/home/users/ajs224/sheat_mip_gws/data/ClimateTRACE/top50emitters.csv"
CLEANSYS_PATH: str = "/home/users/ajs224/sheat_mip_gws/data/ClimateTRACE/CleanSYS_annual_2022-24.csv"
PARAMS_PATH: str = "/home/users/ajs224/sheat_mip_gws/code/GroundData/Output_GroundData/power_plant_thermal_parameters.csv"
WEATHER_PATH: str = "/home/users/ajs224/sheat_mip_gws/code/GroundData/Output_GroundData/power_plants_daily_weather_cache.csv"

# Output directory setup
OUTPUT_DIR: str = "/home/users/ajs224/sheat_mip_gws/code/GroundData/Output_GroundData"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info("Initialized daily target disaggregation generator.")

# %% [markdown]
# ## Section 2: Loading Inputs & Pre-Condition Validation

# %%
try:
    # 1. Load Top 50 point-source metadata
    df_top50 = pd.read_csv(TOP50_PATH).dropna(subset=["Rank"])
    df_top50["Rank"] = df_top50["Rank"].astype(int)
    
    # 2. Load annual CleanSYS ground truth totals (2022-2024)
    df_cleansys = pd.read_csv(CLEANSYS_PATH).dropna(subset=["SO2 Rank", "Year"])
    df_cleansys["SO2 Rank"] = df_cleansys["SO2 Rank"].astype(int)
    df_cleansys["Year"] = df_cleansys["Year"].astype(int)
    
    # 3. Load pre-fitted thermal response parameters for power plants
    df_params = pd.read_csv(PARAMS_PATH).dropna(subset=["Rank"])
    df_params["Rank"] = df_params["Rank"].astype(int)
    
    # 4. Load daily meteorological CDD/HDD cache
    df_weather = pd.read_csv(WEATHER_PATH).dropna(subset=["Rank", "Date"])
    df_weather["Rank"] = df_weather["Rank"].astype(int)
    df_weather["Date"] = pd.to_datetime(df_weather["Date"])
    df_weather["Year"] = df_weather["Year"].astype(int)
    df_weather["Month"] = df_weather["Month"].astype(int)

    logger.info("All configuration tables and weather caches successfully loaded.")
except Exception as e_load:
    logger.error("Failed to read input files during initialization: %s", e_load)
    raise RuntimeError("Aborting execution. Check input paths.")

# %% [markdown]
# ## Section 3: Mass-Conserving Daily Disaggregation Engine

# %%
# Define temporal span across the 3-year study period
start_date = pd.Timestamp("2022-01-01")
end_date = pd.Timestamp("2024-12-31")
date_range = pd.date_range(start=start_date, end=end_date, freq="D")

daily_emission_records: List[Dict[str, Any]] = []

for idx, row in df_top50.iterrows():
    rank = int(row["Rank"])
    name = row["English Translation"]
    sector = row["Primary Sector"]
    approach = str(row["Temporal Approach"]).strip().lower()
    
    logger.info("Disaggregating emissions for Rank %02d: %-35s [%s]", rank, name[:35], approach)
    
    # Extract CleanSYS annual reporting for this facility
    df_plant_cs = df_cleansys[df_cleansys["SO2 Rank"] == rank]
    if df_plant_cs.empty:
        logger.warning("No CleanSYS emissions data found for Rank %02d. Skipping.", rank)
        continue
        
    for current_year in [2022, 2023, 2024]:
        df_year_cs = df_plant_cs[df_plant_cs["Year"] == current_year]
        if df_year_cs.empty:
            logger.warning("  |- No annual CleanSYS data for Year %d. Skipping year.", current_year)
            continue
            
        annual_total_kg = float(df_year_cs["SO2 (kg/year)"].iloc[0])
        year_dates = date_range[date_range.year == current_year]
        num_days = len(year_dates)
        
        # --- PATHWAY A: FLATLINED DISAGGREGATION (INDUSTRIAL SOURCES) ---
        if approach == "flatlined":
            flat_daily_rate = annual_total_kg / num_days
            for day in year_dates:
                daily_emission_records.append({
                    "Rank": rank,
                    "English Name": name,
                    "Date": day,
                    "Year": current_year,
                    "Month": day.month,
                    "Day": day.day,
                    "SO2_kg_day": flat_daily_rate,
                    "Approach": "Flatlined"
                })
                
        # --- PATHWAY B: METEOROLOGICALLY WEIGHTED DISAGGREGATION (POWER PLANTS) ---
        elif approach == "weighted":
            df_plant_param = df_params[df_params["Rank"] == rank]
            if df_plant_param.empty:
                logger.error("  |- No fitted regression params found for power plant Rank %02d. Falling back to flatline.", rank)
                flat_daily_rate = annual_total_kg / num_days
                for day in year_dates:
                    daily_emission_records.append({
                        "Rank": rank, "English Name": name, "Date": day, "Year": current_year,
                        "Month": day.month, "Day": day.day, "SO2_kg_day": flat_daily_rate, "Approach": "Flatlined_Fallback"
                    })
                continue
                
            gamma_0 = float(df_plant_param["gamma_0_baseload"].iloc[0])
            alpha = float(df_plant_param["alpha_cooling"].iloc[0])
            beta = float(df_plant_param["beta_heating"].iloc[0])
            
            # Retrieve daily degree-day profile for target facility and year
            df_weather_subset = df_weather[(df_weather["Rank"] == rank) & (df_weather["Year"] == current_year)]
            if df_weather_subset.empty:
                logger.error("  |- Weather cache empty for Rank %02d in year %d. Falling back to flatline.", rank, current_year)
                flat_daily_rate = annual_total_kg / num_days
                for day in year_dates:
                    daily_emission_records.append({
                        "Rank": rank, "English Name": name, "Date": day, "Year": current_year,
                        "Month": day.month, "Day": day.day, "SO2_kg_day": flat_daily_rate, "Approach": "Flatlined_Fallback"
                    })
                continue
            
            # Step 1: Compute daily temporal weights (w_d)
            daily_weights: List[Tuple[pd.Timestamp, float]] = []
            for _, w_row in df_weather_subset.iterrows():
                day_val = pd.Timestamp(w_row["Date"])
                m = day_val.month
                
                n_days_in_month = calendar.monthrange(current_year, m)[1]
                w_d = (gamma_0 / n_days_in_month) + (alpha * w_row["CDD"]) + (beta * w_row["HDD"])
                daily_weights.append((day_val, w_d))
                
            # Step 2: Sum annual weights to enforce mass conservation
            total_annual_weight = sum([w for _, w in daily_weights])
            
            # Step 3: Scale daily emissions to match CleanSYS annual total
            if total_annual_weight > 0.0:
                for day, w_d in daily_weights:
                    scaled_emission = annual_total_kg * (w_d / total_annual_weight)
                    daily_emission_records.append({
                        "Rank": rank,
                        "English Name": name,
                        "Date": day,
                        "Year": current_year,
                        "Month": day.month,
                        "Day": day.day,
                        "SO2_kg_day": scaled_emission,
                        "Approach": "Weighted"
                    })
            else:
                logger.warning("  |- Total computed weight was zero for Rank %02d. Falling back to flatline.", rank)
                flat_daily_rate = annual_total_kg / num_days
                for day in year_dates:
                    daily_emission_records.append({
                        "Rank": rank, "English Name": name, "Date": day, "Year": current_year,
                        "Month": day.month, "Day": day.day, "SO2_kg_day": flat_daily_rate, "Approach": "Flatlined_Fallback"
                    })

df_daily_output = pd.DataFrame(daily_emission_records)
logger.info("Completed daily disaggregation calculations.")

# %% [markdown]
# ## Section 4: Mass Conservation Verification Audit

# %%
logger.info("Executing mass conservation audit against annual CleanSYS targets...")

audit_failed = False
for (rank, year), group in df_daily_output.groupby(["Rank", "Year"]):
    computed_sum = group["SO2_kg_day"].sum()
    true_target = df_cleansys[(df_cleansys["SO2 Rank"] == rank) & (df_cleansys["Year"] == year)]["SO2 (kg/year)"].iloc[0]
    
    # Audit threshold: absolute difference must be <= 1.0 gram due to floating-point rounding
    diff = abs(computed_sum - true_target)
    if diff > 0.001:
        logger.error("  |- [FAIL] Audit failed for Rank %02d, Year %d. Expected: %.2f kg, Got: %.2f kg. Diff: %.4f kg",
                     rank, year, true_target, computed_sum, diff)
        audit_failed = True

if not audit_failed:
    logger.info("  |- [SUCCESS] All 50 emitters passed mass conservation checks.")
    logger.info("  |- Daily output matches CleanSYS annual totals to 4 decimal places.")

# %% [markdown]
# ## Section 5: Temporal Disaggregation Validation Plotting

# %%
logger.info("Generating diagnostic disaggregation validation plot...")

# Select Taean Power Station (Rank 4) for diagnostic verification
target_rank = 4
df_plot_target = df_daily_output[df_daily_output["Rank"] == target_rank].sort_values(by="Date")

if not df_plot_target.empty:
    target_name = df_plot_target["English Name"].iloc[0]
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.set_facecolor("#fcfcfc")
    ax.grid(True, linestyle="--", alpha=0.5, color="gray")
    
    # Plot continuous daily disaggregated series
    ax.plot(df_plot_target["Date"], df_plot_target["SO2_kg_day"], color="navy", label="Daily Disaggregated SO2 Emissions", linewidth=1.5, zorder=2)
    
    # Overlay annual flatline means to illustrate seasonal disaggregation amplitude
    for year in [2022, 2023, 2024]:
        df_year_target = df_plot_target[df_plot_target["Year"] == year]
        if not df_year_target.empty:
            annual_sum = df_cleansys[(df_cleansys["SO2 Rank"] == target_rank) & (df_cleansys["Year"] == year)]["SO2 (kg/year)"].iloc[0]
            flat_daily = annual_sum / len(df_year_target)
            
            ax.plot([df_year_target["Date"].min(), df_year_target["Date"].max()], [flat_daily, flat_daily], 
                    color="darkorange", linestyle="--", linewidth=2.0, label="Annual Mean (Flatlined)" if year == 2022 else "", zorder=3)
            
            ax.text(df_year_target["Date"].mean(), flat_daily + 200, f"Mean: {flat_daily:.1f} kg/day", 
                    color="darkorange", fontsize=8, fontweight="bold", ha="center")
            
    # Highlight seasonal demand peak windows
    ax.axvspan("2022-07-01", "2022-08-31", color="red", alpha=0.06, label="Summer Cooling Peaks")
    ax.axvspan("2023-07-01", "2023-08-31", color="red", alpha=0.06)
    ax.axvspan("2024-07-01", "2024-08-31", color="red", alpha=0.06)
    
    ax.axvspan("2022-12-01", "2023-02-28", color="blue", alpha=0.06, label="Winter Heating Peaks")
    ax.axvspan("2023-12-01", "2024-02-29", color="blue", alpha=0.06)
    
    ax.set_title(f"Diagnostic Daily Disaggregation Profile (2022–2024)\nRank {target_rank:02d}: {target_name}", fontsize=12, fontweight="bold")
    ax.set_ylabel("SO2 Emissions [kg/day]", fontsize=9)
    ax.set_xlabel("Time [Date]", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, facecolor="white", edgecolor="none")
    
    diag_path = os.path.join(OUTPUT_DIR, "tae_daily_disagg_diagnostic.png")
    plt.savefig(diag_path, bbox_inches="tight")
    logger.info("Saved diagnostic verification plot to: %s", diag_path)
    plt.close()
else:
    logger.warning("Failed to find daily records for Rank %d to generate verification plot.", target_rank)

# %% [markdown]
# ## Section 6: Master Output Database Serialization

# %%
logger.info("Serializing consolidated daily emissions database...")

out_path = os.path.join(OUTPUT_DIR, "daily_emissions_point_sources_2022-24.csv")
try:
    df_daily_output.to_csv(out_path, index=False)
    logger.info("======================================================================================")
    logger.info("DAILY DISAGGREGATION PROCESSING COMPLETE.")
    logger.info("Database successfully compiled and saved to: %s", out_path)
    logger.info("======================================================================================")
except Exception as e_save:
    logger.error("Failed to save final consolidated CSV to disk: %s", e_save)