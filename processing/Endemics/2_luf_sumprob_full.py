#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. Setup & Arguments ---
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', type=str, nargs="+", required=True)
ap.add_argument('-a', '--taxa', type=str, nargs="+", required=True)
ap.add_argument('-g', '--gcm', type=str, nargs="+", required=True)
ap.add_argument('-s', '--scenario', type=str, required=True)
ap.add_argument('-y', '--year', type=str, required=True)
ap.add_argument('--type', type=str,
                choices=['baseline',
                         'climate_only', 'pa30_bio', 'pa30_bcw'],
                required=True)
args = ap.parse_args()

DATA_DIR = "/capacity/occr_davin/mguzman/chari_P2_review/data"
CONV_TABLE = "/capacity/occr_davin/chari/P1/IUCN_LUH_converion_table_Carlson.csv"
# Path for endemic species filtering
MATCH_FILE_PATH = "/capacity/occr_davin/mguzman/chari_P2_review/Endemics/data/modeled_endemics_matches.csv"

# --- 2. DYNAMIC PATHING ---
# Note: Added "/Endemics/" to the path to keep these results separate
if args.year == "2015":
    OUTPUT_DIR = "/capacity/occr_davin/mguzman/chari_P2_review/Endemics/PA_scenarios/Sumprob/WDPA/Historical"
    SUFFIX = "hist"
    LU_FILE = os.path.join(DATA_DIR, "control_wdpa17_2015_hist.nc")
else:
    ssp = "ssp126" if args.scenario == "rcp26" else "ssp460"

    if args.type == 'baseline':
        OUTPUT_DIR = "/capacity/occr_davin/mguzman/chari_P2_review/Endemics/PA_scenarios/Sumprob/WDPA/Baseline"
        LU_FILE = os.path.join(DATA_DIR, f"control_wdpa17_2080_{ssp}.nc")
        SUFFIX = "wdpa17"
    elif args.type == 'climate_only':
        OUTPUT_DIR = "/capacity/occr_davin/mguzman/chari_P2_review/Endemics/PA_scenarios/Sumprob/WDPA/Climate_Only"
        LU_FILE = os.path.join(DATA_DIR, "control_wdpa17_2015_hist.nc")
        SUFFIX = "clim_only"
    elif args.type == 'pa30_bio':
        OUTPUT_DIR = "/capacity/occr_davin/mguzman/chari_P2_review/Endemics/PA_scenarios/Sumprob/PA30_Bio_new"
        LU_FILE = os.path.join(DATA_DIR, f"full_pa30_bio_{ssp}_2080.nc")
        SUFFIX = "pa30_bio"
    elif args.type == 'pa30_bcw':
        OUTPUT_DIR = "/capacity/occr_davin/mguzman/chari_P2_review/Endemics/PA_scenarios/Sumprob/PA30_BCW_new"
        LU_FILE = os.path.join(DATA_DIR, f"full_pa30_bcw_{ssp}_2080.nc")
        SUFFIX = "pa30_bcw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pre-load Endemic Metadata
df_matches = pd.read_csv(MATCH_FILE_PATH)

# Pre-load Land Use Template
with xr.open_dataset(LU_FILE, decode_times=False) as ds_lu:
    da_lu = ds_lu.load()
    available_luh_vars = [v.lower() for v in da_lu.data_vars]

# Pre-load Conversion Table
convcodes = pd.read_csv(CONV_TABLE)
convcodes['IUCN_hab'] = convcodes['IUCN_hab'].astype(str).str.strip()

# --- 3. Processing Loops ---
for sdm in args.model:
    for gcm in args.gcm:
        for taxa in args.taxa:

            # Whitelist endemic species for THIS specific taxa
            endemic_keys = {name.replace(" ", "_") for name in
                            df_matches[df_matches['taxon'] == taxa]['modeled_name'].unique()}

            gcm_map = {"GFDL-ESM2M": "GFDL.ESM2M", "IPSL-CM5A-LR": "IPSL.CM5A-LR",
                       "MIROC5": "MIROC5", "HadGEM2-ES": "HadGEM2.ES"}

            if args.year == "2015":
                current_col = "EWEMBI"
            else:
                clean_gcm_for_col = gcm_map.get(gcm, gcm)
                current_col = f"{clean_gcm_for_col}_{args.scenario}_{args.year}"

            if args.year == "2015":
                base_name = f"summed_prob_{taxa}_{sdm}_{gcm}_{args.year}_{SUFFIX}"
            else:
                base_name = f"summed_prob_{taxa}_{sdm}_{gcm}_{args.scenario}_{args.year}_{SUFFIX}"

            print(
                f"\nProcessing ENDEMICS: {taxa} | {sdm} | {gcm} | {args.year}")

            dir_hab = f"/capacity/occr_davin/chari/P1/Habitat_Classifications/{taxa}/"
            dir_sp = f"/capacity/occr_davin/chari/P1/BioScen15/individual_projections/{taxa}_{sdm}_results_climate/"

            available_files = [f for f in os.listdir(
                dir_sp) if f"{sdm}_dispersal.csv.xz" in f]

            sum_raw = xr.DataArray(np.zeros((360, 720), dtype=np.float32),
                                   coords=[da_lu.lat, da_lu.lon], dims=['lat', 'lon'])
            sum_luf = sum_raw.copy()

            processed_count = 0
            for fname in available_files:
                sp_key = fname.split(f"_{sdm}")[0]

                # --- FILTER: Only process if in endemic whitelist ---
                if sp_key not in endemic_keys:
                    continue

                try:
                    df = pd.read_csv(os.path.join(dir_sp, fname),
                                     compression='xz',
                                     usecols=[
                                         'x', 'y', 'dispersal1', current_col],
                                     dtype={current_col: np.float32, 'dispersal1': np.float32})  # dispersal1: d/4

                    if processed_count == 0 and current_col not in df.columns:
                        print(
                            f"FATAL ERROR: Column {current_col} not found in {fname}!")
                        sys.exit(1)

                    if da_lu.lon.min() < 0 and df['x'].max() > 180:
                        df['x'] = np.where(
                            df['x'] > 180, df['x'] - 360, df['x'])

                    df['p'] = (df[current_col] * df["dispersal1"])
                    ds_sp = df.rename(columns={'x': 'lon', 'y': 'lat'})[
                        ['lat', 'lon', 'p']].set_index(['lat', 'lon']).to_xarray()
                    p_raw = ds_sp['p'].interp(
                        lat=da_lu.lat, lon=da_lu.lon, method='nearest').fillna(0).clip(0, 1)

                    # Land Use Filter
                    p_luf = p_raw * 0
                    h_path = os.path.join(dir_hab, f"{sp_key}.csv")
                    if os.path.exists(h_path):
                        IUCN = pd.read_csv(h_path)
                        suitable = IUCN[IUCN['result.suitability'] == 'Suitable']['result.code'].astype(
                            str).str.strip().unique()
                        matched_rows = convcodes[convcodes['IUCN_hab'].isin(
                            suitable)]

                        luh_vars = []
                        for val in matched_rows['LUH'].dropna():
                            luh_vars.extend([v.strip().lower() for v in str(val).split(
                                '.') if v.strip().lower() in available_luh_vars])

                        if luh_vars:
                            frac = sum(da_lu[v] for v in list(
                                set(luh_vars))).clip(0, 1).fillna(0)
                            p_luf = (p_raw * frac).clip(0, 1)

                    sum_raw += p_raw.astype(np.float32)
                    sum_luf += p_luf.astype(np.float32)
                    processed_count += 1

                    if processed_count % 100 == 0:
                        print(
                            f"   ... [{taxa}] {processed_count} endemic species done")
                except Exception:
                    continue

            # Save NetCDF
            ds_final = xr.Dataset(
                {'prob_clim_disp': sum_raw, 'prob_clim_disp_luf': sum_luf})
            ds_final.to_netcdf(os.path.join(OUTPUT_DIR, f"{base_name}.nc"))
            print(
                f"FINISHED: {processed_count} endemics saved to {base_name}.nc")
