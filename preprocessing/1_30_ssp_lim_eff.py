import xarray as xr
import numpy as np
import os

# --- PATHS ---
OUTPUT_DIR = "/capacity/occr_davin/mguzman/chari_P2_review/data"
HIST_2015_BASE = os.path.join(OUTPUT_DIR, "control_wdpa17_2015_hist.nc")

PA_FILES = {
    "bio": "/capacity/occr_davin/mguzman/chari_P2_review/data/PA_masks/bio-only_withPA.nc",
    "bcw": "/capacity/occr_davin/mguzman/chari_P2_review/data/PA_masks/bcw_withPA.nc"
}

SCENARIOS = ["ssp126", "ssp460", "ssp245"]
ALL_VARS = ['primf', 'primn', 'secdf', 'secdn', 'range', 'c3ann',
            'c3per', 'c4ann', 'c4per', 'c3nfx', 'pastr', 'urban']

# Effectiveness parameter from paper
EFF_COEFF = 0.2359


def generate_30pct_scenarios():
    # 1. Load the 2015 baseline
    print(f">> Loading 2015 Baseline: {HIST_2015_BASE}")
    ds_2015 = xr.open_dataset(HIST_2015_BASE, decode_times=False)

    for pa_name, pa_path in PA_FILES.items():
        print(
            f"\n--- Processing Expansion Scenario: {pa_name} (Limited Effectiveness) ---")

        # 2. Load the 30% mask
        pa_ds = xr.open_dataset(pa_path)
        if 'latitude' in pa_ds.coords:
            pa_ds = pa_ds.rename({'latitude': 'lat'})
        if 'longitude' in pa_ds.coords:
            pa_ds = pa_ds.rename({'longitude': 'lon'})

        pa_mask = pa_ds['fraction'].fillna(0).clip(0, 1)
        pa_mask = pa_mask.interp(
            lat=ds_2015.lat, lon=ds_2015.lon, method="nearest")

        for ssp in SCENARIOS:
            print(f"  > Creating Limited Effectiveness for {ssp}...")

            # 3. Load the raw SSP future
            ssp_path = f"/capacity/occr_davin/chari/P1/LUH2/remapped_luh2_{ssp}.nc"
            future_raw = xr.open_dataset(ssp_path, decode_times=False).isel(
                time=65).drop_vars(["time_bnds", "time"], errors="ignore")

            fut_30pct = ds_2015.copy()

            for v in ALL_VARS:
                # Delta projected by SSP
                delta_v = future_raw[v] - ds_2015[v]

                # --- UPDATED LOGIC ---
                # Paper Logic: New = Baseline + Delta * (1 - (PA_Fraction * 0.2359))
                # If pa_mask = 1.0, 76.41% of change occurs.
                # If pa_mask = 0.0, 100% of change occurs.
                multiplier = (1.0 - (pa_mask * EFF_COEFF))
                fut_30pct[v] = ds_2015[v] + (delta_v * multiplier)

                fut_30pct[v] = fut_30pct[v].clip(0, 1.0)

            # 4. Normalization to maintain land-use budget (1.0)
            sum_total = sum(fut_30pct[v] for v in ALL_VARS)
            mask = sum_total > 0.95
            for v in ALL_VARS:
                normalized = fut_30pct[v] / (sum_total + 1e-12)
                fut_30pct[v] = xr.where(
                    mask, normalized, fut_30pct[v]).fillna(0).clip(0, 1)

            # 5. Save the final file
            # Note the updated filename to reflect "lim_eff"
            out_name = f"lim_eff_pa30_{pa_name}_{ssp}_2080.nc"
            save_path = os.path.join(OUTPUT_DIR, out_name)
            fut_30pct.to_netcdf(save_path)
            print(f"    ✅ File Saved: {out_name}")


if __name__ == "__main__":
    generate_30pct_scenarios()
    print("\n>> All 6 Limited Effectiveness Scenarios Generated.")
