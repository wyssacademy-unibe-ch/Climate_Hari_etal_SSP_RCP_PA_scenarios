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
print(">> Start Process - Management Effectiveness Version")

# Load historical and PA mask
hist = xr.open_dataset(HIST_LUH_PATH, decode_times=False)
wdpa = xr.open_dataset(WDPA_PATH, decode_times=False)[
    'WDPA_percentage_map.tif']

# Align PA mask to LUH2 grid
pa_frac = (wdpa / 100.0).interp(lat=hist.lat, lon=hist.lon,
                                method="nearest").fillna(0).clip(0, 1.0)

# Extract 2015 Baseline
luh_2015_slice = hist.isel(time=1165).drop_vars(
    ["time_bnds", "time"], errors="ignore")
baseline_2015_raw = hist_safety_net(luh_2015_slice, pa_frac)

# Effectiveness coefficient from the paper (23.6%)
eff_coeff = 0.2359

scenarios = ["ssp126", "ssp460", "ssp245"]

for scenario in scenarios:
    print(f"\n>> Processing {scenario} with 23.6% effectiveness...")
    ssp_path = f"/capacity/occr_davin/chari/P1/LUH2/remapped_luh2_{scenario}.nc"

    if not os.path.exists(ssp_path):
        continue

    future_ds = xr.open_dataset(ssp_path, decode_times=False)
    future_raw = future_ds.isel(time=65).drop_vars(
        ["time_bnds", "time"], errors="ignore")

    # --- Task A: Partial Conservation (Limited Effectiveness) ---
    # Formula based on paper: (0.2359 * Baseline) + (0.764 * Future)
    # This applies ONLY to the portion of the cell that is protected (pa_frac)

    fut_effective = baseline_2015_raw.copy()

    for v in all_vars:
        # Standard change (Delta)
        delta = future_raw[v] - luh_2015_slice[v]

        # Applying the weighting:
        # In protected areas (pa_frac), we only allow 76.4% of the change.
        # This is equivalent to: Baseline + (Delta * 0.764)
        protection_constraint = (1.0 - eff_coeff)  # This is 0.764

        # We only apply this constraint to the fraction of the cell covered by PAs
        # Non-protected fraction (1 - pa_frac) gets 100% of the delta.
        # Protected fraction (pa_frac) gets 76.4% of the delta.

        effective_delta_multiplier = (
            1.0 - pa_frac) + (pa_frac * protection_constraint)

        fut_effective[v] = (baseline_2015_raw[v] +
                            (delta * effective_delta_multiplier)).clip(0, 1.0)

    # Normalize to maintain land budget
    fut_effective = normalize_coastal_safe(fut_effective, all_vars)

    # Save output
    out_path = os.path.join(
        OUTPUT_DIR, f"control_lim_eff_wdpa17_2080_{scenario}.nc")
    fut_effective.to_netcdf(out_path)
    print(f"Saved: {out_path}")

print("\n>> Process complete.")
