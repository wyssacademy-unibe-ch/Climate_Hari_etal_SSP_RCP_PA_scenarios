import xarray as xr
import os

# --- Configuration ---
ROOT_DIR = "/capacity/occr_davin/mguzman/chari_P2_review/Endemics/PA_scenarios/Sumprob/WDPA"
HIST_DIR = os.path.join(ROOT_DIR, "Historical")
TAXAS = ["Amphibians", "Bird", "Mammals"]
SDMS = ["GAM", "GBM"]
CLIMATE_DATA = "EWEMBI"  # Historical uses EWEMBI instead of GCMs
YEAR_HIST = "1995"
SUFFIX_HIST = "hist"
VARS = ['prob_clim_disp', 'prob_clim_disp_luf']


def run_historical_ensembles():
    out_dir = os.path.join(HIST_DIR, "Ensembles")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n--- Processing HISTORICAL (1995) using {CLIMATE_DATA} ---")

    taxa_ensembles = {}

    for taxa in TAXAS:
        print(f"  > Creating Historical Ensemble for: {taxa}")
        sdm_results = []

        for sdm in SDMS:
            # Handle Bird/GBM Proxy (use GAM)
            current_sdm = sdm
            if taxa == "Bird" and sdm == "GBM":
                current_sdm = "GAM"
                print(f"    ! Using GAM as proxy for Bird GBM")

            fname = f"summed_prob_{taxa}_{current_sdm}_{CLIMATE_DATA}_{YEAR_HIST}_{SUFFIX_HIST}.nc"
            fpath = os.path.join(HIST_DIR, fname)

            if os.path.exists(fpath):
                # Load the single EWEMBI file
                ds = xr.open_dataset(fpath)[VARS].load()
                sdm_results.append(ds)
            else:
                print(f"    ⚠️ Missing: {fname}")

        if sdm_results:
            # Mean across SDMs (GAM/GBM)
            # Since there is only 1 climate source (EWEMBI), we only average the models
            taxa_mean = xr.concat(sdm_results, dim='sdm').mean(dim='sdm')
            taxa_ensembles[taxa] = taxa_mean

            # Save individual Taxa Ensemble
            t_out_name = f"{taxa}_historical_{YEAR_HIST}.nc"
            taxa_mean.to_netcdf(os.path.join(out_dir, t_out_name))
            print(f"    ✅ Created: {t_out_name}")

    # Create Total Richness Historical (Sum of Mammals + Birds + Amphibians)
    if taxa_ensembles:
        print(f"  > Creating Total Richness Historical Ensemble...")
        total_richness = sum(taxa_ensembles.values())

        out_name = f"total_richness_historical_{YEAR_HIST}.nc"
        total_richness.to_netcdf(os.path.join(out_dir, out_name))
        print(f"✅ Final Historical Richness Created: {out_name}")


if __name__ == "__main__":
    run_historical_ensembles()
