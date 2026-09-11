#!/usr/bin/env python3

"""
Calculate equally weighted ensemble means from revised Land-Use Filter outputs.

For each taxonomic group and experiment, the script calculates a simple
arithmetic mean across every included SDM-GCM combination. It then sums the
three taxon-specific ensemble surfaces to obtain the combined vertebrate
richness estimate.

Historical outputs use EWEMBI and are averaged across the included SDM files.
Future outputs are averaged across the included SDM-GCM files.

The script processes:

    1. Historical 2015 baseline
    2. Climate-only projections for 2080
    3. Existing 17%-PA configuration
    4. 30% biodiversity-only configuration
    5. 30% multi-objective configuration

PA configurations are processed under complete and funding-constrained
land-use retention.
"""

import os
import sys
import warnings

import numpy as np
import xarray as xr


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)


# =====================================================================
# 1. PATHS
# =====================================================================

BASE_PATH = (
    "/capacity/occr_davin/mguzman/chari_P2_review/Global/PA_scenarios/Sumprob_revised"
)

ENSEMBLE_ROOT = os.path.join(
    BASE_PATH,
    "Ensembles",
)


# =====================================================================
# 2. ENSEMBLE SETTINGS
# =====================================================================

TAXA = [
    "Amphibians",
    "Bird",
    "Mammals",
]

# Model combinations included in the analysis.
MODELS_BY_TAXON = {
    "Amphibians": [
        "GAM",
        "GBM",
    ],
    "Bird": [
        "GAM",
    ],
    "Mammals": [
        "GAM",
        "GBM",
    ],
}

GCMS = [
    "GFDL-ESM2M",
    "IPSL-CM5A-LR",
    "HadGEM2-ES",
    "MIROC5",
]

SCENARIOS = [
    "rcp26",
    "rcp45",
    "rcp60",
]

RETENTIONS = [
    "complete",
    "funding_constrained",
]

VARIABLES = [
    "prob_clim_disp",
    "prob_clim_disp_luf",
]

YEAR_HIST = "2015"
YEAR_FUTURE = "2080"
HIST_GCM = "EWEMBI"

# Existing non-empty ensemble outputs are skipped when False.
OVERWRITE_EXISTING = False


# =====================================================================
# 3. PA AND SCENARIO DEFINITIONS
# =====================================================================

PA_RUNS = {
    "pa17": {
        "folder": "PA17",
        "label": "Existing 17%-PA network",
    },
    "pa30_bio": {
        "folder": "PA30_Bio",
        "label": "30% biodiversity-only expansion",
    },
    "pa30_bcw": {
        "folder": "PA30_BCW",
        "label": "30% multi-objective expansion",
    },
}

LAND_USE_LABELS = {
    "rcp26": "SSP1-2.6",
    "rcp45": "SSP2-4.5",
    "rcp60": "SSP4-6.0",
}

CLIMATE_LABELS = {
    "rcp26": "RCP2.6",
    "rcp45": "RCP2.6",
    "rcp60": "RCP6.0",
}


# =====================================================================
# 4. GENERAL FUNCTIONS
# =====================================================================

def ensure_dir(path):
    """Create a directory if it does not already exist."""

    os.makedirs(
        path,
        exist_ok=True,
    )


def output_is_complete(path):
    """Return True when an output exists and is non-empty."""

    return (
        os.path.isfile(path)
        and os.path.getsize(path) > 0
    )


def standardize_coordinates(dataset):
    """Standardize spatial coordinate names to lat and lon."""

    rename_dict = {}

    if (
        "latitude" in dataset.coords
        or "latitude" in dataset.dims
    ):
        rename_dict["latitude"] = "lat"

    if (
        "longitude" in dataset.coords
        or "longitude" in dataset.dims
    ):
        rename_dict["longitude"] = "lon"

    if rename_dict:
        dataset = dataset.rename(
            rename_dict
        )

    if (
        "lat" not in dataset.coords
        or "lon" not in dataset.coords
    ):
        raise ValueError(
            "Dataset does not contain lat and lon coordinates."
        )

    return dataset


def grids_match(first, second):
    """Check whether two datasets have identical spatial grids."""

    return (
        first.sizes.get("lat")
        == second.sizes.get("lat")
        and first.sizes.get("lon")
        == second.sizes.get("lon")
        and np.allclose(
            first["lat"].values,
            second["lat"].values,
            equal_nan=True,
        )
        and np.allclose(
            first["lon"].values,
            second["lon"].values,
            equal_nan=True,
        )
    )


def load_dataset(file_path):
    """Load and validate one summed-probability file."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Required input file not found: {file_path}"
        )

    with xr.open_dataset(
        file_path,
        decode_times=False,
    ) as source:

        missing_variables = [
            variable
            for variable in VARIABLES
            if variable not in source.data_vars
        ]

        if missing_variables:
            raise KeyError(
                f"Input file is missing variables "
                f"{missing_variables}: {file_path}"
            )

        dataset = source[
            VARIABLES
        ].load()

    dataset = standardize_coordinates(
        dataset
    )

    for variable in VARIABLES:
        dataset[variable] = (
            dataset[variable]
            .astype(np.float32)
        )

    return dataset


def mean_datasets(
    datasets,
    member_labels,
):
    """
    Calculate a simple equally weighted arithmetic mean.

    All supplied members receive the same weight. Missing values are not
    skipped, preventing an ensemble cell from being calculated from fewer
    members without warning.
    """

    if not datasets:
        raise ValueError(
            "No datasets were supplied for ensemble averaging."
        )

    reference = datasets[0]

    for label, dataset in zip(
        member_labels[1:],
        datasets[1:],
    ):
        if not grids_match(
            reference,
            dataset,
        ):
            raise ValueError(
                f"Spatial-grid mismatch for ensemble member: {label}"
            )

    member_coordinate = xr.IndexVariable(
        "ensemble_member",
        member_labels,
    )

    combined = xr.concat(
        datasets,
        dim=member_coordinate,
        join="exact",
        compat="equals",
        coords="minimal",
    )

    ensemble_mean = combined.mean(
        dim="ensemble_member",
        skipna=False,
    )

    return ensemble_mean.astype(
        np.float32
    )


def add_variable_metadata(dataset):
    """Add descriptions to the ensemble variables."""

    dataset["prob_clim_disp"].attrs.update(
        {
            "long_name": (
                "Ensemble mean of summed climate-driven and "
                "limited-dispersal occurrence probabilities"
            ),
        }
    )

    dataset["prob_clim_disp_luf"].attrs.update(
        {
            "long_name": (
                "Ensemble mean of summed land-use-filtered "
                "occurrence probabilities"
            ),
        }
    )

    return dataset


def save_dataset(
    dataset,
    output_path,
):
    """
    Save an ensemble output.

    Existing non-empty files are skipped unless OVERWRITE_EXISTING is True.
    Zero-byte files are removed and regenerated.
    """

    if (
        output_is_complete(output_path)
        and not OVERWRITE_EXISTING
    ):
        print(
            f"Skipping existing ensemble: {output_path}"
        )

        return False

    if (
        os.path.exists(output_path)
        and not output_is_complete(output_path)
    ):
        print(
            f"Removing incomplete output: {output_path}"
        )

        os.remove(
            output_path
        )

    ensure_dir(
        os.path.dirname(output_path)
    )

    dataset.encoding.pop(
        "unlimited_dims",
        None,
    )

    encoding = {
        variable: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
        }
        for variable in VARIABLES
    }

    dataset.to_netcdf(
        output_path,
        encoding=encoding,
        unlimited_dims=[],
    )

    print(
        f"Saved: {output_path}"
    )

    return True


# =====================================================================
# 5. INPUT PATH FUNCTIONS
# =====================================================================

def historical_input(
    taxon,
    model,
):
    """Return the path to one historical LUF output."""

    filename = (
        f"summed_prob_{taxon}_{model}_{HIST_GCM}_"
        f"{YEAR_HIST}_baseline_2015.nc"
    )

    return os.path.join(
        BASE_PATH,
        "WDPA",
        "Historical",
        filename,
    )


def climate_only_input(
    taxon,
    model,
    gcm,
    scenario,
):
    """Return the path to one climate-only LUF output."""

    filename = (
        f"summed_prob_{taxon}_{model}_{gcm}_"
        f"{scenario}_{YEAR_FUTURE}_climate_only.nc"
    )

    return os.path.join(
        BASE_PATH,
        "WDPA",
        "Climate_Only",
        filename,
    )


def pa_input(
    run_type,
    retention,
    taxon,
    model,
    gcm,
    scenario,
):
    """Return the path to one PA-scenario LUF output."""

    filename = (
        f"summed_prob_{taxon}_{model}_{gcm}_"
        f"{scenario}_{YEAR_FUTURE}_"
        f"{run_type}_{retention}.nc"
    )

    return os.path.join(
        BASE_PATH,
        PA_RUNS[run_type]["folder"],
        retention,
        filename,
    )


# =====================================================================
# 6. TAXON-SPECIFIC ENSEMBLES
# =====================================================================

def build_historical_taxon(taxon):
    """
    Calculate one taxon-specific historical ensemble.

    The historical ensemble is the equally weighted mean across the
    included historical SDM outputs.
    """

    datasets = []
    member_labels = []
    missing_files = []

    for model in MODELS_BY_TAXON[taxon]:
        file_path = historical_input(
            taxon,
            model,
        )

        if not os.path.exists(file_path):
            missing_files.append(
                file_path
            )

            continue

        datasets.append(
            load_dataset(file_path)
        )

        member_labels.append(
            model
        )

    if missing_files:
        print(
            f"Missing historical inputs for {taxon}:"
        )

        for file_path in missing_files:
            print(
                f"  {file_path}"
            )

        for dataset in datasets:
            dataset.close()

        return None

    expected_members = len(
        MODELS_BY_TAXON[taxon]
    )

    if len(datasets) != expected_members:
        print(
            f"Incomplete historical ensemble for {taxon}. "
            f"Expected {expected_members} members but found "
            f"{len(datasets)}."
        )

        for dataset in datasets:
            dataset.close()

        return None

    result = mean_datasets(
        datasets=datasets,
        member_labels=member_labels,
    )

    result = add_variable_metadata(
        result
    )

    result.attrs.update(
        {
            "analysis": "Historical baseline",
            "year": YEAR_HIST,
            "taxonomic_group": taxon,
            "ensemble_method": (
                "Equally weighted arithmetic mean"
            ),
            "ensemble_member_count": len(
                member_labels
            ),
        }
    )

    for dataset in datasets:
        dataset.close()

    return result


def build_future_taxon(
    taxon,
    scenario,
    path_builder,
):
    """
    Calculate one taxon-specific future ensemble.

    Every included SDM-GCM combination receives the same weight.
    """

    datasets = []
    member_labels = []
    missing_files = []

    for model in MODELS_BY_TAXON[taxon]:

        for gcm in GCMS:

            file_path = path_builder(
                taxon,
                model,
                gcm,
                scenario,
            )

            if not os.path.exists(file_path):
                missing_files.append(
                    file_path
                )

                continue

            datasets.append(
                load_dataset(file_path)
            )

            member_labels.append(
                f"{model}_{gcm}"
            )

    if missing_files:
        print(
            f"Missing future inputs for {taxon}, {scenario}:"
        )

        for file_path in missing_files:
            print(
                f"  {file_path}"
            )

        for dataset in datasets:
            dataset.close()

        return None

    expected_members = (
        len(MODELS_BY_TAXON[taxon])
        * len(GCMS)
    )

    if len(datasets) != expected_members:
        print(
            f"Incomplete ensemble for {taxon}, {scenario}. "
            f"Expected {expected_members} members but found "
            f"{len(datasets)}."
        )

        for dataset in datasets:
            dataset.close()

        return None

    result = mean_datasets(
        datasets=datasets,
        member_labels=member_labels,
    )

    result = add_variable_metadata(
        result
    )

    result.attrs.update(
        {
            "taxonomic_group": taxon,
            "scenario_label": scenario,
            "ensemble_method": (
                "Equally weighted arithmetic mean across "
                "included SDM-GCM combinations"
            ),
            "ensemble_member_count": len(
                member_labels
            ),
            "gcm_count": len(
                GCMS
            ),
        }
    )

    for dataset in datasets:
        dataset.close()

    return result


# =====================================================================
# 7. COMBINED TAXONOMIC TOTAL
# =====================================================================

def sum_taxa(taxa_results):
    """
    Sum the taxon-specific ensemble means.

    The output is the sum of the continuous richness estimates for
    amphibians, birds, and mammals.
    """

    missing_taxa = [
        taxon
        for taxon in TAXA
        if taxon not in taxa_results
    ]

    if missing_taxa:
        raise ValueError(
            f"Missing taxon ensembles: {missing_taxa}"
        )

    ordered_datasets = [
        taxa_results[taxon]
        for taxon in TAXA
    ]

    reference = ordered_datasets[0]

    for taxon, dataset in zip(
        TAXA[1:],
        ordered_datasets[1:],
    ):
        if not grids_match(
            reference,
            dataset,
        ):
            raise ValueError(
                f"Spatial-grid mismatch for taxon: {taxon}"
            )

    combined_total = xr.Dataset()

    for variable in VARIABLES:

        combined_total[variable] = sum(
            dataset[variable]
            for dataset in ordered_datasets
        ).astype(np.float32)

    combined_total = add_variable_metadata(
        combined_total
    )

    combined_total.attrs.update(
        {
            "taxonomic_scope": (
                "Amphibians, Bird, and Mammals"
            ),
            "combination_method": (
                "Sum of taxon-specific ensemble means"
            ),
        }
    )

    return combined_total


# =====================================================================
# 8. HISTORICAL ENSEMBLES
# =====================================================================

def run_historical():
    """Generate historical taxon-specific and combined ensembles."""

    print("")
    print("====================================================")
    print("HISTORICAL ENSEMBLES")
    print("====================================================")

    output_directory = os.path.join(
        ENSEMBLE_ROOT,
        "Historical",
    )

    taxa_results = {}

    for taxon in TAXA:

        print(
            f"Building historical ensemble: {taxon}"
        )

        result = build_historical_taxon(
            taxon
        )

        if result is None:
            return False

        taxa_results[taxon] = result

        output_path = os.path.join(
            output_directory,
            f"{taxon}_historical_"
            f"{YEAR_HIST}_ensemble.nc",
        )

        save_dataset(
            result,
            output_path,
        )

    combined_total = sum_taxa(
        taxa_results
    )

    combined_total.attrs.update(
        {
            "analysis": "Historical baseline",
            "year": YEAR_HIST,
        }
    )

    combined_path = os.path.join(
        output_directory,
        f"total_richness_historical_"
        f"{YEAR_HIST}_ensemble.nc",
    )

    save_dataset(
        combined_total,
        combined_path,
    )

    for dataset in taxa_results.values():
        dataset.close()

    combined_total.close()

    return True


# =====================================================================
# 9. CLIMATE-ONLY ENSEMBLES
# =====================================================================

def run_climate_only():
    """Generate climate-only taxon-specific and combined ensembles."""

    print("")
    print("====================================================")
    print("CLIMATE-ONLY ENSEMBLES")
    print("====================================================")

    output_directory = os.path.join(
        ENSEMBLE_ROOT,
        "Climate_Only",
    )

    for scenario in SCENARIOS:

        print(
            f"Climate-only scenario: {scenario}"
        )

        taxa_results = {}

        for taxon in TAXA:

            result = build_future_taxon(
                taxon=taxon,
                scenario=scenario,
                path_builder=climate_only_input,
            )

            if result is None:
                return False

            result.attrs.update(
                {
                    "analysis": "Climate-only",
                    "year": YEAR_FUTURE,
                    "land_use_condition": (
                        "Harmonized 2015 baseline held constant"
                    ),
                    "climate_forcing": (
                        CLIMATE_LABELS[scenario]
                    ),
                }
            )

            taxa_results[taxon] = result

            output_path = os.path.join(
                output_directory,
                f"{taxon}_climate_only_"
                f"{scenario}_{YEAR_FUTURE}_ensemble.nc",
            )

            save_dataset(
                result,
                output_path,
            )

        combined_total = sum_taxa(
            taxa_results
        )

        combined_total.attrs.update(
            {
                "analysis": "Climate-only",
                "year": YEAR_FUTURE,
                "scenario_label": scenario,
                "climate_forcing": (
                    CLIMATE_LABELS[scenario]
                ),
            }
        )

        combined_path = os.path.join(
            output_directory,
            f"total_richness_climate_only_"
            f"{scenario}_{YEAR_FUTURE}_ensemble.nc",
        )

        save_dataset(
            combined_total,
            combined_path,
        )

        for dataset in taxa_results.values():
            dataset.close()

        combined_total.close()

    return True


# =====================================================================
# 10. PA-SCENARIO ENSEMBLES
# =====================================================================

def run_pa_scenarios():
    """Generate ensembles for every PA and retention combination."""

    print("")
    print("====================================================")
    print("PA-SCENARIO ENSEMBLES")
    print("====================================================")

    for run_type, run_information in PA_RUNS.items():

        for retention in RETENTIONS:

            output_directory = os.path.join(
                ENSEMBLE_ROOT,
                run_information["folder"],
                retention,
            )

            for scenario in SCENARIOS:

                print(
                    f"{run_type} | {retention} | {scenario}"
                )

                taxa_results = {}

                def path_builder(
                    taxon,
                    model,
                    gcm,
                    scenario_name,
                ):
                    return pa_input(
                        run_type=run_type,
                        retention=retention,
                        taxon=taxon,
                        model=model,
                        gcm=gcm,
                        scenario=scenario_name,
                    )

                for taxon in TAXA:

                    result = build_future_taxon(
                        taxon=taxon,
                        scenario=scenario,
                        path_builder=path_builder,
                    )

                    if result is None:
                        return False

                    result.attrs.update(
                        {
                            "analysis": (
                                "PA-mediated land-use change"
                            ),
                            "year": YEAR_FUTURE,
                            "pa_configuration": (
                                run_information["label"]
                            ),
                            "land_use_retention_assumption": (
                                retention
                            ),
                            "land_use_scenario": (
                                LAND_USE_LABELS[scenario]
                            ),
                            "climate_forcing": (
                                CLIMATE_LABELS[scenario]
                            ),
                        }
                    )

                    taxa_results[taxon] = result

                    output_path = os.path.join(
                        output_directory,
                        (
                            f"{taxon}_{run_type}_{retention}_"
                            f"{scenario}_{YEAR_FUTURE}_ensemble.nc"
                        ),
                    )

                    save_dataset(
                        result,
                        output_path,
                    )

                combined_total = sum_taxa(
                    taxa_results
                )

                combined_total.attrs.update(
                    {
                        "analysis": (
                            "PA-mediated land-use change"
                        ),
                        "year": YEAR_FUTURE,
                        "scenario_label": scenario,
                        "pa_configuration": (
                            run_information["label"]
                        ),
                        "land_use_retention_assumption": (
                            retention
                        ),
                        "land_use_scenario": (
                            LAND_USE_LABELS[scenario]
                        ),
                        "climate_forcing": (
                            CLIMATE_LABELS[scenario]
                        ),
                    }
                )

                combined_path = os.path.join(
                    output_directory,
                    (
                        f"total_richness_{run_type}_{retention}_"
                        f"{scenario}_{YEAR_FUTURE}_ensemble.nc"
                    ),
                )

                save_dataset(
                    combined_total,
                    combined_path,
                )

                for dataset in taxa_results.values():
                    dataset.close()

                combined_total.close()

    return True


# =====================================================================
# 11. MAIN WORKFLOW
# =====================================================================

def main():
    """Run all consolidated ensemble calculations."""

    ensure_dir(
        ENSEMBLE_ROOT
    )

    print(
        f"Input root: {BASE_PATH}"
    )

    print(
        f"Output root: {ENSEMBLE_ROOT}"
    )

    print(
        f"Overwrite existing outputs: {OVERWRITE_EXISTING}"
    )

    historical_success = run_historical()

    if not historical_success:
        raise RuntimeError(
            "Historical ensembles could not be completed."
        )

    climate_success = run_climate_only()

    if not climate_success:
        raise RuntimeError(
            "Climate-only ensembles could not be completed."
        )

    pa_success = run_pa_scenarios()

    if not pa_success:
        raise RuntimeError(
            "PA-scenario ensembles could not be completed."
        )

    print("")
    print(
        "All consolidated ensembles were generated successfully."
    )


# =====================================================================
# 12. RUN
# =====================================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print(
            f"\nERROR: {error}",
            file=sys.stderr,
        )

        sys.exit(1)
