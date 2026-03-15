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

                choices=['baseline', 'restoration',

                         'climate_only', 'pa30_bio', 'pa30_bcw'],

                required=True)

args = ap.parse_args()


DATA_DIR = "/capacity/occr_davin/mguzman/chari_P2_review/data"

CONV_TABLE = "/capacity/occr_davin/chari/P1/IUCN_LUH_converion_table_Carlson.csv"


# --- 2. DYNAMIC PATHING & SSP MAPPING ---

ssp_map = {

    "rcp26": "ssp126",
    "rcp45": "ssp245",
    "rcp60": "ssp460"
}

if args.year == "2015":
    OUTPUT_DIR = "/capacity/occr_davin/mguzman/chari_P2_review/Global/PA_scenarios/Sumprob/WDPA/Historical"
    SUFFIX = "hist"
    LU_FILE = os.path.join(DATA_DIR, "control_wdpa17_2015_hist.nc")
    ssp = "historical"

else:

    # 1. Check if the scenario exists in our map

    if args.scenario not in ssp_map:
        print(
            f"ERROR: Scenario '{args.scenario}' is not recognized in the ssp_map.")
        print(f"Valid options are: {list(ssp_map.keys())}")
        sys.exit(1)  # This stops the process immediately

    # 2. If it exists, proceed safely
    ssp = ssp_map[args.scenario]
    if args.type == 'baseline':
        OUTPUT_DIR = "/capacity/occr_davin/mguzman/chari_P2_review/Global/PA_scenarios/Sumprob/WDPA/Baseline"
        LU_FILE = os.path.join(DATA_DIR, f"control_wdpa17_2080_{ssp}.nc")
        SUFFIX = "wdpa17"

    elif args.type == 'climate_only':
        OUTPUT_DIR = "/capacity/occr_davin/mguzman/chari_P2_review/Global/PA_scenarios/Sumprob/WDPA/Climate_Only"
        LU_FILE = os.path.join(DATA_DIR, "control_wdpa17_2015_hist.nc")
        SUFFIX = "clim_only"

    elif args.type == 'pa30_bio':
        OUTPUT_DIR = "/capacity/occr_davin/mguzman/chari_P2_review/Global/PA_scenarios/Sumprob/PA30_Bio_new"
        LU_FILE = os.path.join(DATA_DIR, f"full_pa30_bio_{ssp}_2080.nc")
        SUFFIX = "pa30_bio"

    elif args.type == 'pa30_bcw':
        OUTPUT_DIR = "/capacity/occr_davin/mguzman/chari_P2_review/Global/PA_scenarios/Sumprob/PA30_BCW_new"
        LU_FILE = os.path.join(DATA_DIR, f"full_pa30_bcw_{ssp}_2080.nc")
        SUFFIX = "pa30_bcw"


if not os.path.exists(LU_FILE):
    print(f"ERROR: Land use file not found: {LU_FILE}")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)


with xr.open_dataset(LU_FILE, decode_times=False) as ds_lu:
    da_lu = ds_lu.load()
    available_luh_vars = [v.lower() for v in da_lu.data_vars]

convcodes = pd.read_csv(CONV_TABLE)
convcodes['IUCN_hab'] = convcodes['IUCN_hab'].astype(str).str.strip()


# --- 3. Processing Loops ---

for sdm in args.model:
    for gcm in args.gcm:
        for taxa in args.taxa:
            gcm_map = {
                "GFDL-ESM2M": "GFDL.ESM2M",
                "IPSL-CM5A-LR": "IPSL.CM5A-LR",
                "MIROC5": "MIROC5",
                "HadGEM2-ES": "HadGEM2.ES"
            }

            if args.year == "2015":
                current_col = "EWEMBI"

            else:
                clean_gcm_for_col = gcm_map.get(gcm, gcm)
                current_col = f"{clean_gcm_for_col}_{args.scenario}_{args.year}"
            base_name = f"summed_prob_{taxa}_{sdm}_{gcm}_{args.year if args.year=='2015' else args.scenario+'_'+args.year}_{SUFFIX}"

            print(
                f"\n Processing: {taxa} | {sdm} | {gcm} | {args.scenario} | {args.year}")
            dir_hab = f"/capacity/occr_davin/chari/P1/Habitat_Classifications/{taxa}/"
            dir_sp = f"/capacity/occr_davin/chari/P1/BioScen15/individual_projections/{taxa}_{sdm}_results_climate/"

            if not os.path.exists(dir_sp):
                print(f" Directory not found: {dir_sp}. Skipping.")
                continue

            available_files = [f for f in os.listdir(
                dir_sp) if f"{sdm}_dispersal.csv.xz" in f]

            sum_raw = xr.DataArray(np.zeros((360, 720), dtype=np.float32),
                                   coords=[da_lu.lat, da_lu.lon], dims=['lat', 'lon'])

            sum_luf = sum_raw.copy()

            processed_count = 0
            processed_count = 0
            found_col_issue = False  # Flag to print columns only once

            for fname in available_files:
                sp_key = fname.split(f"_{sdm}")[0]
                f_path = os.path.join(dir_sp, fname)

                try:

                    # Check header first

                    header = pd.read_csv(f_path, compression='xz', nrows=0)
                    if current_col not in header.columns:
                        if not found_col_issue:
                            print(
                                f" Column '{current_col}' NOT FOUND in {fname}")
                            print(
                                f" Available columns in file are: {list(header.columns)}")
                            found_col_issue = True  # Don't spam the log for every species

                        continue

                    # If it exists, proceed with loading

                    df = pd.read_csv(f_path, compression='xz',
                                     usecols=[
                                         'x', 'y', 'dispersal1', current_col],
                                     dtype={current_col: np.float32, 'dispersal1': np.float32})
                    if da_lu.lon.min() < 0 and df['x'].max() > 180:
                        df['x'] = np.where(
                            df['x'] > 180, df['x'] - 360, df['x'])
                    df['p'] = (df[current_col] * df["dispersal1"])
                    ds_sp = df.rename(columns={'x': 'lon', 'y': 'lat'})[
                        ['lat', 'lon', 'p']].set_index(['lat', 'lon']).to_xarray()

                    p_raw = ds_sp['p'].interp(
                        lat=da_lu.lat, lon=da_lu.lon, method='nearest').fillna(0).clip(0, 1)

                    # Land Use Filter

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

                        else:
                            p_luf = p_raw * 0

                    else:
                        p_luf = p_raw * 0

                    sum_raw += p_raw.astype(np.float32)
                    sum_luf += p_luf.astype(np.float32)
                    processed_count += 1

                    if processed_count % 1000 == 0:

                        print(
                            f"   ... [{taxa}] {processed_count} species summed")

                except Exception as e:

                    print(f" Error processing {fname}: {e}")
                    continue

            # Save NetCDF

            ds_final = xr.Dataset(
                {'prob_clim_disp': sum_raw, 'prob_clim_disp_luf': sum_luf})
            ds_final.to_netcdf(os.path.join(OUTPUT_DIR, f"{base_name}.nc"))
            print(f" FINISHED: {base_name}.nc saved")
            print(f" TOTAL SPECIES PROCESSED FOR {taxa}: {processed_count}")

print("\n>> All tasks completed.")
