#!/usr/bin/env python3
import xarray as xr
import numpy as np
import os
import sys

# --- 1. SETUP AND PATHS ---
WDPA_PATH = "/capacity/occr_davin/mguzman/chari_P2_review/data/PA_masks/WDPA_percentage_map.nc"
HIST_LUH_PATH = "/capacity/occr_davin/chari/P1/LUH2/remapped_luh2_historical.nc"
OUTPUT_DIR = "/capacity/occr_davin/mguzman/chari_P2_review/data"
HIST_OUT_PATH = os.path.join(OUTPUT_DIR, "control_wdpa17_2015_hist.nc")

natural_vars = ['primf', 'primn', 'secdf', 'secdn', 'range']
human_vars = ['c3ann', 'c3per', 'c4ann', 'c4per', 'c3nfx', 'pastr', 'urban']
all_vars = natural_vars + human_vars

# --- 3. LOGIC FUNCTIONS ---


def normalize_coastal_safe(ds, var_list):
    """
    Only normalizes pixels where land area is intended to be 100%.
    This prevents 'inflating' coastal pixels or dividing by zero in oceans.
    """
    total_sum = sum(ds[v] for v in var_list)

    # Threshold 0.99 captures 'full land' pixels with minor delta drift,
    # but ignores coastal pixels and ocean.
    mask = total_sum > 0.95

    for v in var_list:
        # Only modify pixels that meet the threshold
        normalized = ds[v] / (total_sum + 1e-12)
        ds[v] = xr.where(mask, normalized, ds[v]).fillna(0).clip(0, 1.0)
    return ds


def hist_safety_net(ds_luh, pa_mask):
    """Ensures PAs have natural cover while maintaining land-fraction integrity."""
    new_ds = ds_luh.copy()
    total_nat = sum(ds_luh[v] for v in natural_vars)
    total_hum = sum(ds_luh[v] for v in human_vars)

    # A 'hole' is a pixel with PA > 0 but no nature AND it must be land (total_hum > 0)
    is_hole = (total_nat < 1e-6) & (pa_mask > 0.001) & (total_hum > 0.1)

    # Calculate current land fraction to avoid inflating coastal cells
    land_fraction = (total_nat + total_hum)

    for v in natural_vars:
        fallback = land_fraction if v == 'secdn' else 0
        new_ds[v] = xr.where(is_hole, fallback, ds_luh[v])

    for v in human_vars:
        new_ds[v] = xr.where(is_hole, 0, ds_luh[v])

    return normalize_coastal_safe(new_ds, all_vars)


def reallocate_luh_with_pa(ds_future, pa_mask, ds_template_2015):
    """Task B: Restoration logic that respects coastal land fractions."""
    new_ds = ds_future.copy()
    total_hum = sum(ds_future[v] for v in human_vars)
    total_nat = sum(ds_future[v] for v in natural_vars)
    land_fraction = (total_hum + total_nat)

    # PAs can only occupy the land portion of the cell
    max_pa_allowed = xr.where(pa_mask > land_fraction, land_fraction, pa_mask)
    available_for_human = (land_fraction - max_pa_allowed).clip(0, 1.0)

    hum_scaling = xr.where(total_hum > available_for_human,
                           available_for_human / (total_hum + 1e-12), 1.0)
    for v in human_vars:
        new_ds[v] = ds_future[v] * hum_scaling

    total_hum_new = sum(new_ds[v] for v in human_vars)
    new_nat_total = (land_fraction - total_hum_new).clip(0, 1.0)

    future_nat_sum = sum(ds_future[v] for v in natural_vars)
    template_nat_sum = sum(ds_template_2015[v] for v in natural_vars)

    for v in natural_vars:
        ratio_f = ds_future[v] / (future_nat_sum + 1e-12)
        ratio_15 = ds_template_2015[v] / (template_nat_sum + 1e-12)
        chosen_ratio = xr.where(future_nat_sum > 1e-6, ratio_f, ratio_15)
        new_ds[v] = xr.where((future_nat_sum < 1e-6) & (template_nat_sum < 1e-6),
                             (new_nat_total if v == 'secdn' else 0),
                             chosen_ratio * new_nat_total)

    return normalize_coastal_safe(new_ds, all_vars)


# --- 4. EXECUTION ---
print(">> Start Process - Logging to restoration_sanity_check.txt")

# Load historical and PA mask
hist = xr.open_dataset(HIST_LUH_PATH, decode_times=False)
wdpa = xr.open_dataset(WDPA_PATH, decode_times=False)['WDPA_percentage_map.tif']

# Fix Longitude if WDPA is 0-360 and Hist is -180-180
if (wdpa.lon.max() > 180) and (hist.lon.min() < 0):
    wdpa = wdpa.assign_coords(lon=(((wdpa.lon + 180) % 360) - 180)).sortby('lon')

# Align PA mask to LUH2 grid
pa_frac = (wdpa / 100.0).interp(lat=hist.lat, lon=hist.lon, 
                                method="nearest").fillna(0).clip(0, 1.0)

# Extract 2015 Baseline
luh_2015_slice = hist.isel(time=1165).drop_vars(["time_bnds", "time"], errors="ignore")
baseline_2015_raw = hist_safety_net(luh_2015_slice, pa_frac)

# Save Baseline
baseline_2015_raw.to_netcdf(HIST_OUT_PATH)
print(f">> Baseline 2015 saved to: {HIST_OUT_PATH}")

# Loop through scenarios
scenarios = ["ssp126", "ssp460", "ssp245"]

for scenario in scenarios:
    print(f"\n>> Processing {scenario} 2080...")
    ssp_path = f"/capacity/occr_davin/chari/P1/LUH2/remapped_luh2_{scenario}.nc"
    
    if not os.path.exists(ssp_path):
        print(f"⚠️ Skipping {scenario}: File not found at {ssp_path}")
        continue
    
    # Load Future slice (index 65 = 2080)
    future_ds = xr.open_dataset(ssp_path, decode_times=False)
    future_raw = future_ds.isel(time=65).drop_vars(["time_bnds", "time"], errors="ignore")

    # --- Task A: Strict Conservation (Freeze PAs) ---
    # Formula: Baseline + Delta * (1 - PA_Fraction)
    fut_strict = baseline_2015_raw.copy()
    for v in all_vars:
        delta = future_raw[v] - luh_2015_slice[v]
        fut_strict[v] = (baseline_2015_raw[v] + (delta * (1.0 - pa_frac))).clip(0, 1.0)

    # Normalize to maintain land budget
    fut_strict = normalize_coastal_safe(fut_strict, all_vars)

    # --- Task B: Restoration ---
    # Forces human land use out of protected fractions
    fut_restored = reallocate_luh_with_pa(future_raw, pa_frac, baseline_2015_raw)

    # --- Integrity Analysis ---
    print(f"--- ANALYSIS: {scenario} ---")
    for name, ds in [("Strict", fut_strict), ("Restored", fut_restored)]:
        sum_val = sum(ds[v] for v in all_vars)
        land_mask = sum_val > 0.98
        # Check if land pixels sum to ~1.0
        max_err = abs(float(sum_val.where(land_mask).max()) - 1.0)
        integrity = "PASS" if max_err < 1e-5 else f"WARN (Dev: {max_err:.2e})"
        print(f"  [CHECK] {name:10} | Land Sum Integrity: {integrity}")

    # Save outputs
    fut_strict.to_netcdf(os.path.join(OUTPUT_DIR, f"control_wdpa17_2080_{scenario}.nc"))
    fut_restored.to_netcdf(os.path.join(OUTPUT_DIR, f"restoration_wdpa17_2080_{scenario}.nc"))

print("\n>> All scenarios processed successfully.")