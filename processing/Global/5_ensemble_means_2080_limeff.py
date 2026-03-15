import xarray as xr
import os

# --- Configuration ---

BASE_PATH = "/capacity/occr_davin/mguzman/chari_P2_review/Global/PA_scenarios/Sumprob"

TAXAS = ["Amphibians", "Bird", "Mammals"]
SDMS = ["GAM", "GBM"]
GCMS = ["GFDL-ESM2M", "IPSL-CM5A-LR", "HadGEM2-ES", "MIROC5"]
SCENARIOS = ["rcp26", "rcp60"]
YEAR = "2080"
VARS = ['prob_clim_disp', 'prob_clim_disp_luf']

# Mapping the Run Name to the (Folder_Path, Filename_Suffix)
RUN_TYPES = {
    "Baseline": ("WDPA/Baseline", "wdpa17"),
    "Climate_Only": ("WDPA/Climate_Only", "clim_only"),
    "PA30_Bio": ("PA30_Bio_new", "pa30_bio"),
    "PA30_BCW": ("PA30_BCW_new", "pa30_bcw")
}


def run_future_ensembles():
    for run_name, (sub_dir, suffix) in RUN_TYPES.items():
        # Build paths dynamically based on the folder structure
        base_dir = os.path.join(BASE_PATH, sub_dir)
        out_dir = os.path.join(base_dir, "Ensembles")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n--- Processing {run_name.upper()} ---")
        print(f"Source Directory: {base_dir}")

        for sce in SCENARIOS:
            taxa_ensembles = {}

            for taxa in TAXAS:
                print(f"  > Creating Ensemble for: {taxa} ({sce})")
                sdm_results = []

                for sdm in SDMS:
                    gcm_results = []

                    # Bird GBM proxy logic
                    current_sdm = sdm
                    if taxa == "Bird" and sdm == "GBM":
                        current_sdm = "GAM"
                        print(f"    ! Using GAM as proxy for Bird GBM")

                    for gcm in GCMS:
                        # Construct filename: summed_prob_Bird_GAM_MIROC5_rcp26_2080_pa30_bio.nc
                        fname = f"lim_eff_summed_prob_{taxa}_{current_sdm}_{gcm}_{sce}_{YEAR}_{suffix}.nc"
                        fpath = os.path.join(base_dir, fname)

                        if os.path.exists(fpath):
                            # Load both variables
                            ds = xr.open_dataset(fpath)[VARS].load()
                            gcm_results.append(ds)
                        else:
                            print(f" Missing: {fpath}")

                    if gcm_results:
                        # 1. Mean across GCMs
                        gcm_mean = xr.concat(
                            gcm_results, dim='gcm').mean(dim='gcm')
                        sdm_results.append(gcm_mean)

                if sdm_results:
                    # 2. Mean across SDMs (GAM/GBM)
                    taxa_mean = xr.concat(
                        sdm_results, dim='sdm').mean(dim='sdm')
                    taxa_ensembles[taxa] = taxa_mean

                    # Save individual Taxa Ensemble (e.g., Mammals_pa30_bio_rcp26_2080.nc)
                    t_out_name = f"lim_eff_{taxa}_{run_name.lower()}_{sce}_{YEAR}.nc"
                    taxa_mean.to_netcdf(os.path.join(out_dir, t_out_name))

            # 3. Create Total Richness (Sum of all taxa ensembles)
            if len(taxa_ensembles) == len(TAXAS):
                print(
                    f"  > Creating Total Richness Ensemble for {run_name}...")
                total_richness = sum(taxa_ensembles.values())

                out_name = f"lim_eff_total_richness_{run_name.lower()}_{sce}_{YEAR}.nc"
                total_richness.to_netcdf(os.path.join(out_dir, out_name))
                print(f"Created: {out_name}")
            else:
                print(
                    f" Skipped Total Richness: Not all taxa ensembles were created for {sce}")


if __name__ == "__main__":
    run_future_ensembles()
