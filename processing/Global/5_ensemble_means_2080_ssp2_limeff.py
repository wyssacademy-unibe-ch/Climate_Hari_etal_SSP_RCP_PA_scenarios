import xarray as xr
import os

# --- Configuration ---
BASE_PATH = "/capacity/occr_davin/mguzman/chari_P2_review/Global/PA_scenarios/Sumprob"

TAXAS = ["Amphibians", "Bird", "Mammals"]
SDMS = ["GAM", "GBM"]
GCMS = ["GFDL-ESM2M", "IPSL-CM5A-LR", "HadGEM2-ES", "MIROC5"]
SCENARIOS = ["rcp26", "rcp60"]
YEAR = "2080"
LU_SSP = "ssp245"  # fixed land-use scenario
VARS = ['prob_clim_disp', 'prob_clim_disp_luf']

# Mapping the Run Name to the (Folder_Path, Base_Suffix)
RUN_TYPES = {
    "Baseline": ("WDPA/Baseline", f"wdpa17"),
    "Climate_Only": ("WDPA/Climate_Only", "clim_only"),
    "PA30_Bio": ("PA30_Bio_new", f"pa30_bio"),
    "PA30_BCW": ("PA30_BCW_new", f"pa30_bcw")
}


def run_mixed_ensembles():
    for run_name, (sub_dir, suffix) in RUN_TYPES.items():
        base_dir = os.path.join(BASE_PATH, sub_dir)
        out_dir = os.path.join(base_dir, f"Ensembles")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n--- Processing {run_name.upper()} (Fixed LU: {LU_SSP}) ---")

        for sce in SCENARIOS:
            taxa_ensembles = {}

            for taxa in TAXAS:
                print(f"  > Creating Ensemble for: {taxa} ({sce} climate)")
                sdm_results = []

                for sdm in SDMS:
                    gcm_results = []
                    current_sdm = sdm
                    if taxa == "Bird" and sdm == "GBM":
                        current_sdm = "GAM"

                    for gcm in GCMS:
                        # NEW FILENAME FORMAT:
                        # lim_eff_summed_prob_Mammals_GBM_MIROC5_rcp26_2080_LU_ssp245_pa30_bcw.nc
                        fname = f"lim_eff_summed_prob_{taxa}_{current_sdm}_{gcm}_{sce}_{YEAR}_LU_{LU_SSP}_{suffix}.nc"
                        fpath = os.path.join(base_dir, fname)

                        if os.path.exists(fpath):
                            ds = xr.open_dataset(fpath)[VARS].load()
                            gcm_results.append(ds)
                        else:
                            print(f"    Missing: {fname}")

                    if gcm_results:
                        # Mean across GCMs
                        gcm_mean = xr.concat(
                            gcm_results, dim='gcm').mean(dim='gcm')
                        sdm_results.append(gcm_mean)

                if sdm_results:
                    # Mean across SDMs
                    taxa_mean = xr.concat(
                        sdm_results, dim='sdm').mean(dim='sdm')
                    taxa_ensembles[taxa] = taxa_mean

                    # Save individual Taxa Ensemble
                    t_out_name = f"lim_eff_{taxa}_{run_name.lower()}_clim_{sce}_LU_{LU_SSP}_{YEAR}.nc"
                    taxa_mean.to_netcdf(os.path.join(out_dir, t_out_name))

            # Create Total Richness (Sum of all taxa ensembles)
            if len(taxa_ensembles) == len(TAXAS):
                print(
                    f"  > Creating Total Richness Ensemble (Climate: {sce}, LU: {LU_SSP})...")
                total_richness = sum(taxa_ensembles.values())

                out_name = f"lim_eff_total_richness_{run_name.lower()}_clim_{sce}_LU_{LU_SSP}_{YEAR}.nc"
                total_richness.to_netcdf(os.path.join(out_dir, out_name))
                print(f"Created: {out_name}")
            else:
                print(f" Skipped Total Richness for {sce}: Missing Taxa")


if __name__ == "__main__":
    run_mixed_ensembles()
