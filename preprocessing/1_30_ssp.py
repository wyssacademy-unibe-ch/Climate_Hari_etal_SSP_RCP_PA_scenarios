import xarray as xr
import numpy as np
import os

# --- PATHS ---
OUTPUT_DIR = "/capacity/occr_davin/mguzman/chari_P2_review/data"
HIST_2015_BASE = os.path.join(OUTPUT_DIR, "control_wdpa17_2015_hist.nc")

PA_FILES = {
    "bio_biome": "/capacity/occr_davin/mguzman/chari_P2_review/data/PA_masks/bio_biome_withPA.nc",
    "bcw_biome": "/capacity/occr_davin/mguzman/chari_P2_review/data/PA_masks/bcw_biome_withPA.nc"
}

# Adding another SSP/RCP scenario: /capacity/occr_davin/chari/P1/LUH2/remapped_luh2_ssp245.nc
SCENARIOS = ["ssp126", "ssp460", "ssp245"]
ALL_VARS = ['primf', 'primn', 'secdf', 'secdn', 'range', 'c3ann',
            'c3per', 'c4ann', 'c4per', 'c3nfx', 'pastr', 'urban']


def generate_30pct_scenarios():
    # 1. Load the 2015 baseline (this is the V_hist)
    print(f">> Loading 2015 Baseline: {HIST_2015_BASE}")
    ds_2015 = xr.open_dataset(HIST_2015_BASE, decode_times=False)

    for pa_name, pa_path in PA_FILES.items():
        print(f"\n--- Processing Expansion Scenario: {pa_name} ---")

        # 2. Load the 30% mask and extract 'pa_fraction'
        pa_ds = xr.open_dataset(pa_path)

        # Ensure coordinates match LUH2 names (lat/lon)
        if 'latitude' in pa_ds.coords:
            pa_ds = pa_ds.rename({'latitude': 'lat'})
        if 'longitude' in pa_ds.coords:
            pa_ds = pa_ds.rename({'longitude': 'lon'})

        pa_mask = pa_ds['fraction'].fillna(0).clip(0, 1)

        # Ensure mask is aligned with the baseline spatial grid
        pa_mask = pa_mask.interp(
            lat=ds_2015.lat, lon=ds_2015.lon, method="nearest")

        for ssp in SCENARIOS:
            print(f"  > Creating Strict Conservation for {ssp}...")

            # 3. Load the raw SSP future (Year 2080 is index 65)
            ssp_path = f"/capacity/occr_davin/chari/P1/LUH2/remapped_luh2_{ssp}.nc"
            future_raw = xr.open_dataset(ssp_path, decode_times=False).isel(
                time=65).drop_vars(["time_bnds", "time"], errors="ignore")

            # Initialize output as a copy of 2015
            fut_30pct = ds_2015.copy()

            for v in ALL_VARS:
                # Calculate the delta projected by the SSP
                delta_v = future_raw[v] - ds_2015[v]

                # Apply the formula: New = 2015 + (Change * Unprotected Fraction)
                # This freezes land use where pa_mask is 1.0
                fut_30pct[v] = ds_2015[v] + (delta_v * (1.0 - pa_mask))

                # Clip to prevent small floating point errors below 0 or above 1
                fut_30pct[v] = fut_30pct[v].clip(0, 1.0)

            # 4. Sanity Check: All fractions in a cell must sum to 1.0
            sum_total = sum(fut_30pct[v] for v in ALL_VARS)

            # Only normalize pixels where land is supposed to be full (> 0.9)
            # # This prevents division by zero in oceans and inflation in coasts
            mask = sum_total > 0.95
            for v in ALL_VARS:
                normalized = fut_30pct[v] / (sum_total + 1e-12)
                fut_30pct[v] = xr.where(
                    mask, normalized, fut_30pct[v]).fillna(0).clip(0, 1)

            # 5. Save the final file
            out_name = f"full_pa30_{pa_name}_{ssp}_2080.nc"
            save_path = os.path.join(OUTPUT_DIR, out_name)
            fut_30pct.to_netcdf(save_path)
            print(f"    ✅ File Saved: {out_name}")


if __name__ == "__main__":
    generate_30pct_scenarios()
    print("\n>> All 6 Expansion Scenarios Generated.")
