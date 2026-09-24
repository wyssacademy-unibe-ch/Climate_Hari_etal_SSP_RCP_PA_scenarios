#!/usr/bin/env python3

"""
Generate PA-adjusted 2080 land-use projections for the two 30% PA
expansion configurations.

PA CONFIGURATIONS
-----------------
1. Biodiversity-only 30% expansion
2. Multi-objective 30% expansion

LAND-USE RETENTION ASSUMPTIONS
------------------------------
1. Complete land-use retention - Full-effectiveness

   The protected fraction of each grid cell retains its harmonized
   2015 land-use composition. The retention coefficient is 1.

2. Funding-constrained land-use retention - Limited-effectiveness

   The proportion of projected change prevented within the protected
   fraction is adjusted using:

       current global PA expenditure / lower-bound funding requirement
       for a 30% PA system

       24.3 / 103 = 0.2359

LAND-USE CALCULATION
--------------------
For each LUH2 land-use class, the raw projected change is calculated as:

    raw_delta = raw_LUH2_2080 - raw_LUH2_2015

This raw LUH2 delta is applied to the common harmonised 2015 baseline:

    adjusted_2080 =
        harmonized_2015
        + raw_delta
        * (1 - PA_fraction * retention_coefficient)

Interpretation:

    PA_fraction = 0:
        The complete raw LUH2 change proceeds.

    PA_fraction = 1 and retention_coefficient = 1:
        The projected change is completely prevented.

    PA_fraction = 1 and retention_coefficient = 0.2359:
        23.59% of the projected change is prevented and 76.41% proceeds.

    PA_fraction = 0.4 and retention_coefficient = 0.2359:
        0.4 * 0.2359 = 0.09436

        Therefore, 9.436% of the grid-cell change is prevented and
        90.564% proceeds.

IMPORTANT
---------
The harmonised 2015 baseline is used as the common starting condition
for all PA configurations.

The projected delta is calculated using the original raw LUH2 2015 and
2080 fractions. This preserves the trajectory projected by LUH2 while
allowing all PA scenarios to start from the same harmonised baseline.

The funding coefficient is a scenario parameter. It is not an empirical
estimate of protected-area effectiveness.
"""

import os
import sys
import warnings

import numpy as np
import xarray as xr


warnings.filterwarnings("ignore", category=RuntimeWarning)


# =====================================================================
# 1. PATHS
# =====================================================================

OUTPUT_DIR = (
    "/capacity/occr_davin/mguzman/chari_P2_review/data/new_LU"
)

# New harmonized baseline generated with the corrected fractional
# adjustment for the existing protected-area network.
HARMONIZED_2015_PATH = os.path.join(
    OUTPUT_DIR,
    "baseline_wdpa17_2015_hist.nc",
)

# Original historical LUH2 data, used to calculate the raw change
# projected by LUH2 between 2015 and 2080.
RAW_HISTORICAL_PATH = (
    "/capacity/occr_davin/chari/P1/LUH2/"
    "remapped_luh2_historical.nc"
)

# Directory containing the future LUH2 projections.
FUTURE_LUH2_DIR = (
    "/capacity/occr_davin/chari/P1/LUH2"
)

# Proportional masks for the two 30% PA configurations.
PA_FILES = {
    "bio": (
        "/capacity/occr_davin/mguzman/chari_P2_review/"
        "data/PA_masks/bio-only_withPA.nc"
    ),
    "bcw": (
        "/capacity/occr_davin/mguzman/chari_P2_review/"
        "data/PA_masks/bcw_withPA.nc"
    ),
}


# =====================================================================
# 2. PA CONFIGURATION METADATA
# =====================================================================

PA_DESCRIPTIONS = {
    "bio": "30% biodiversity-only PA expansion",
    "bcw": "30% multi-objective PA expansion",
}

# Name of the proportional PA variable in the 30% mask files.
PA_VARIABLE = "fraction"


# =====================================================================
# 3. LAND-USE RETENTION ASSUMPTIONS
# =====================================================================

RETENTION_ASSUMPTIONS = {
    "complete": {
        "coefficient": 1.0,
        "description": "Complete land-use retention",
        "calculation": "1.0",
    },
    "funding_constrained": {
        "coefficient": 0.2359,
        "description": "Funding-constrained land-use retention",
        "calculation": "24.3 / 103",
    },
}


# =====================================================================
# 4. LUH2 SETTINGS
# =====================================================================

SCENARIOS = [
    "ssp126",
    "ssp245",
    "ssp460",
]

NATURAL_VARS = [
    "primf",
    "primn",
    "secdf",
    "secdn",
    "range",
]

HUMAN_VARS = [
    "c3ann",
    "c3per",
    "c4ann",
    "c4per",
    "c3nfx",
    "pastr",
    "urban",
]

ALL_VARS = NATURAL_VARS + HUMAN_VARS

# Indices retained from the original analysis.
#
# Historical index 1165 corresponds to the original 2015 LUH2 slice in
# the workflow supplied by the authors.
#
# Future index 65 corresponds to the 2080 slice in the SSP files.
HISTORICAL_2015_INDEX = 1165
FUTURE_2080_INDEX = 65


# =====================================================================
# 5. GENERAL HELPER FUNCTIONS
# =====================================================================

def standardize_coordinates(dataset):
    """
    Standardize geographical coordinate names to 'lat' and 'lon'.
    """

    rename_dict = {}

    if "latitude" in dataset.coords or "latitude" in dataset.dims:
        rename_dict["latitude"] = "lat"

    if "longitude" in dataset.coords or "longitude" in dataset.dims:
        rename_dict["longitude"] = "lon"

    if rename_dict:
        dataset = dataset.rename(rename_dict)

    if "lat" not in dataset.coords or "lon" not in dataset.coords:
        raise ValueError(
            "Dataset does not contain recognizable latitude and "
            "longitude coordinates."
        )

    return dataset


def validate_land_use_variables(dataset, dataset_name):
    """
    Verify that all required LUH2 land-use variables are available.
    """

    missing_variables = [
        variable
        for variable in ALL_VARS
        if variable not in dataset.data_vars
    ]

    if missing_variables:
        raise KeyError(
            f"{dataset_name} is missing the following LUH2 variables: "
            f"{missing_variables}"
        )


def sort_spatial_coordinates(data):
    """
    Sort latitude and longitude coordinates when their order is
    descending.

    Sorting avoids unexpected behavior during interpolation.
    """

    if (
        data.sizes.get("lat", 0) > 1
        and data["lat"].values[0] > data["lat"].values[-1]
    ):
        data = data.sortby("lat")

    if (
        data.sizes.get("lon", 0) > 1
        and data["lon"].values[0] > data["lon"].values[-1]
    ):
        data = data.sortby("lon")

    return data


def coordinates_match(source, target):
    """
    Test whether two objects use the same latitude-longitude grid.
    """

    if source.sizes.get("lat") != target.sizes.get("lat"):
        return False

    if source.sizes.get("lon") != target.sizes.get("lon"):
        return False

    latitude_match = np.allclose(
        source["lat"].values,
        target["lat"].values,
        equal_nan=True,
    )

    longitude_match = np.allclose(
        source["lon"].values,
        target["lon"].values,
        equal_nan=True,
    )

    return latitude_match and longitude_match


# =====================================================================
# 6. LOAD COMMON HARMONIZED 2015 BASELINE
# =====================================================================

def load_harmonized_2015():
    """
    Load the corrected harmonized 2015 land-use baseline.

    This baseline was generated by introducing a natural-vegetation
    component equal to PA coverage only in PA-containing cells
    represented entirely by anthropogenic LUH2 classes.
    """

    if not os.path.exists(HARMONIZED_2015_PATH):
        raise FileNotFoundError(
            "Corrected harmonized 2015 baseline not found: "
            f"{HARMONIZED_2015_PATH}"
        )

    print(
        f">> Loading harmonized 2015 baseline: "
        f"{HARMONIZED_2015_PATH}"
    )

    with xr.open_dataset(
        HARMONIZED_2015_PATH,
        decode_times=False,
    ) as source:

        source = standardize_coordinates(source)

        harmonized_2015 = source.load()

    validate_land_use_variables(
        harmonized_2015,
        "Harmonized 2015 baseline",
    )

    harmonized_2015 = sort_spatial_coordinates(
        harmonized_2015
    )

    return harmonized_2015


# =====================================================================
# 7. LOAD ORIGINAL LUH2 2015 DATA
# =====================================================================

def load_raw_luh2_2015(target_grid):
    """
    Load the original, unmodified LUH2 2015 land-use fractions.

    These values are used only to calculate the raw projected LUH2
    delta between 2015 and 2080.
    """

    if not os.path.exists(RAW_HISTORICAL_PATH):
        raise FileNotFoundError(
            "Historical LUH2 file not found: "
            f"{RAW_HISTORICAL_PATH}"
        )

    print(
        f">> Loading original LUH2 2015 data: "
        f"{RAW_HISTORICAL_PATH}"
    )

    with xr.open_dataset(
        RAW_HISTORICAL_PATH,
        decode_times=False,
    ) as source:

        source = standardize_coordinates(source)

        validate_land_use_variables(
            source,
            "Historical LUH2 dataset",
        )

        if "time" not in source.dims:
            raise ValueError(
                "Historical LUH2 dataset has no time dimension."
            )

        if HISTORICAL_2015_INDEX >= source.sizes["time"]:
            raise IndexError(
                f"Historical index {HISTORICAL_2015_INDEX} is outside "
                f"the available time dimension, which contains "
                f"{source.sizes['time']} time steps."
            )

        raw_2015 = (
            source
            .isel(time=HISTORICAL_2015_INDEX)
            .drop_vars(
                ["time_bnds", "time"],
                errors="ignore",
            )
            .load()
        )

    raw_2015 = sort_spatial_coordinates(
        raw_2015
    )

    if not coordinates_match(raw_2015, target_grid):
        print(
            ">> Aligning original LUH2 2015 data with the "
            "harmonized baseline grid."
        )

        raw_2015 = raw_2015.interp(
            lat=target_grid["lat"],
            lon=target_grid["lon"],
            method="nearest",
        )

    return raw_2015


# =====================================================================
# 8. LOAD FUTURE LUH2 PROJECTION
# =====================================================================

def load_raw_luh2_2080(scenario, target_grid):
    """
    Load the original LUH2 2080 land-use fractions for one SSP scenario.

    The output is aligned with the common harmonized 2015 grid.
    """

    future_path = os.path.join(
        FUTURE_LUH2_DIR,
        f"remapped_luh2_{scenario}.nc",
    )

    if not os.path.exists(future_path):
        raise FileNotFoundError(
            f"Future LUH2 file not found: {future_path}"
        )

    print(
        f">> Loading original LUH2 2080 projection for {scenario}"
    )

    with xr.open_dataset(
        future_path,
        decode_times=False,
    ) as source:

        source = standardize_coordinates(source)

        validate_land_use_variables(
            source,
            f"Future LUH2 dataset for {scenario}",
        )

        if "time" not in source.dims:
            raise ValueError(
                f"Future LUH2 dataset for {scenario} has no "
                "time dimension."
            )

        if FUTURE_2080_INDEX >= source.sizes["time"]:
            raise IndexError(
                f"Future index {FUTURE_2080_INDEX} is outside the "
                f"available time dimension for {scenario}, which "
                f"contains {source.sizes['time']} time steps."
            )

        raw_2080 = (
            source
            .isel(time=FUTURE_2080_INDEX)
            .drop_vars(
                ["time_bnds", "time"],
                errors="ignore",
            )
            .load()
        )

    raw_2080 = sort_spatial_coordinates(
        raw_2080
    )

    if not coordinates_match(raw_2080, target_grid):
        print(
            f">> Aligning {scenario} LUH2 projection with the "
            "harmonized baseline grid."
        )

        raw_2080 = raw_2080.interp(
            lat=target_grid["lat"],
            lon=target_grid["lon"],
            method="nearest",
        )

    return raw_2080


# =====================================================================
# 9. LOAD 30% PA MASK
# =====================================================================

def load_pa_mask(pa_path, target_grid):
    """
    Load and align a proportional 30% PA mask.

    PA-mask interpretation:

        0 = none of the grid cell belongs to the PA configuration
        1 = the entire grid cell belongs to the PA configuration

    Intermediate values represent partial PA coverage.
    """

    if not os.path.exists(pa_path):
        raise FileNotFoundError(
            f"PA mask not found: {pa_path}"
        )

    with xr.open_dataset(
        pa_path,
        decode_times=False,
    ) as source:

        source = standardize_coordinates(source)

        if PA_VARIABLE not in source.data_vars:
            raise KeyError(
                f"Variable '{PA_VARIABLE}' was not found in "
                f"{pa_path}. Available variables: "
                f"{list(source.data_vars)}"
            )

        pa_mask = (
            source[PA_VARIABLE]
            .load()
            .squeeze(drop=True)
        )

    extra_dimensions = [
        dimension
        for dimension in pa_mask.dims
        if dimension not in ["lat", "lon"]
    ]

    if extra_dimensions:
        raise ValueError(
            f"PA mask {pa_path} contains unsupported dimensions: "
            f"{extra_dimensions}"
        )

    pa_mask = sort_spatial_coordinates(
        pa_mask
    )

    if not coordinates_match(pa_mask, target_grid):
        print(
            "    Aligning PA mask with the harmonized "
            "2015 baseline grid."
        )

        pa_mask = pa_mask.interp(
            lat=target_grid["lat"],
            lon=target_grid["lon"],
            method="nearest",
        )

    pa_mask = (
        pa_mask
        .fillna(0)
        .clip(min=0, max=1)
        .astype(np.float32)
    )

    return pa_mask


# =====================================================================
# 10. COASTAL-SAFE NORMALIZATION
# =====================================================================

def normalize_coastal_safe(dataset):
    """
    Normalize full-land grid cells while preserving coastal land area.

    LUH2 fractions may sum to less than one in coastal cells because only
    part of a 0.5-degree grid cell is terrestrial. Normalizing every
    non-zero cell to one would inflate land cover in coastal cells.

    Following the original workflow, normalization is applied only where
    the combined land-use fraction exceeds 0.95.
    """

    total_land_fraction = sum(
        dataset[variable]
        for variable in ALL_VARS
    )

    full_land_mask = (
        total_land_fraction > 0.95
    )

    for variable in ALL_VARS:

        normalized_fraction = (
            dataset[variable]
            / (total_land_fraction + 1e-12)
        )

        dataset[variable] = (
            xr.where(
                full_land_mask,
                normalized_fraction,
                dataset[variable],
            )
            .fillna(0)
            .clip(min=0, max=1)
            .astype(np.float32)
        )

    return dataset


# =====================================================================
# 11. GENERATE ONE ADJUSTED PROJECTION
# =====================================================================

def generate_adjusted_projection(
    harmonized_2015,
    raw_2015,
    raw_2080,
    pa_mask,
    pa_name,
    scenario,
    assumption_name,
    retention_coefficient,
):
    """
    Generate one PA-adjusted land-use projection.

    The original LUH2 delta is calculated as:

        raw_2080 - raw_2015

    The delta is then applied to the common harmonized baseline:

        adjusted_2080 =
            harmonized_2015
            + raw_delta
            * (1 - PA_fraction * retention_coefficient)
    """

    # Include only the LUH2 land-use variables in the output.
    adjusted_2080 = harmonized_2015[
        ALL_VARS
    ].copy(deep=True)

    # This is the proportion of projected grid-cell change prevented.
    prevented_change_fraction = (
        pa_mask * retention_coefficient
    ).clip(
        min=0,
        max=1,
    ).astype(np.float32)

    # This is the proportion of projected change allowed to proceed.
    allowed_change_fraction = (
        1.0 - prevented_change_fraction
    ).astype(np.float32)

    for variable in ALL_VARS:

        # Signed change projected by the original LUH2 trajectory.
        raw_projected_delta = (
            raw_2080[variable]
            - raw_2015[variable]
        )

        adjusted_2080[variable] = (
            harmonized_2015[variable]
            + raw_projected_delta * allowed_change_fraction
        ).clip(
            min=0,
            max=1,
        ).astype(np.float32)

    adjusted_2080 = normalize_coastal_safe(
        adjusted_2080
    )

    # Store the PA mask and prevented-change fraction in the output to
    # make the calculation traceable.
    adjusted_2080["pa_fraction"] = (
        pa_mask
        .fillna(0)
        .clip(min=0, max=1)
        .astype(np.float32)
    )

    adjusted_2080["prevented_change_fraction"] = (
        prevented_change_fraction
        .fillna(0)
        .clip(min=0, max=1)
        .astype(np.float32)
    )

    adjusted_2080["pa_fraction"].attrs.update(
        {
            "long_name": (
                "Proportion of the grid cell included in the "
                "30% protected-area configuration"
            ),
            "units": "1",
            "valid_min": 0.0,
            "valid_max": 1.0,
        }
    )

    adjusted_2080["prevented_change_fraction"].attrs.update(
        {
            "long_name": (
                "Proportion of projected grid-cell land-use change "
                "prevented under the PA scenario"
            ),
            "units": "1",
            "calculation": (
                "pa_fraction multiplied by retention_coefficient"
            ),
        }
    )

    # -------------------------------------------------------------
    # Global metadata
    # -------------------------------------------------------------

    adjusted_2080.attrs.update(
        {
            "title": (
                "PA-adjusted LUH2 land-use projection for 2080"
            ),
            "pa_configuration": PA_DESCRIPTIONS[pa_name],
            "pa_coverage": "30%",
            "land_use_retention_assumption": (
                RETENTION_ASSUMPTIONS[assumption_name]["description"]
            ),
            "retention_coefficient": float(
                retention_coefficient
            ),
            "retention_coefficient_calculation": (
                RETENTION_ASSUMPTIONS[assumption_name]["calculation"]
            ),
            "common_baseline": HARMONIZED_2015_PATH,
            "raw_historical_source": RAW_HISTORICAL_PATH,
            "baseline_year": 2015,
            "projection_year": 2080,
            "ssp_scenario": scenario,
            "adjustment_equation": (
                "adjusted_2080 = harmonized_2015 + "
                "(raw_LUH2_2080 - raw_LUH2_2015) * "
                "(1 - pa_fraction * retention_coefficient)"
            ),
        }
    )

    if assumption_name == "complete":

        adjusted_2080.attrs["methodological_note"] = (
            "The complete-retention scenario assumes that protection "
            "prevents all projected land-use change within the "
            "protected fraction of each grid cell."
        )

    elif assumption_name == "funding_constrained":

        adjusted_2080.attrs.update(
            {
                "current_expenditure_usd_billion_per_year": 24.3,
                "minimum_funding_requirement_usd_billion_per_year": (
                    103.0
                ),
                "funding_ratio": float(
                    retention_coefficient
                ),
                "funding_ratio_interpretation": (
                    "Proportion of the lower-bound estimated funding "
                    "requirement for a 30% PA network covered by "
                    "current expenditure"
                ),
                "methodological_note": (
                    "The funding ratio is used as a scenario parameter "
                    "scaling the proportion of projected land-use "
                    "change prevented within the protected fraction. "
                    "It is not an empirical estimate of PA "
                    "effectiveness or observed avoided conversion."
                ),
            }
        )

    return adjusted_2080


# =====================================================================
# 12. SANITY CHECKS
# =====================================================================

def run_sanity_checks(
    adjusted_dataset,
    pa_mask,
    pa_name,
    scenario,
    assumption_name,
    retention_coefficient,
):
    """
    Verify that the generated projection behaves as expected.

    Checks include:

        1. PA values remain between 0 and 1.
        2. Prevented-change values remain between 0 and 1.
        3. Full-land cells sum to one after normalization.
        4. The maximum prevented change does not exceed the
           retention coefficient.
    """

    prevented_change_fraction = (
        adjusted_dataset["prevented_change_fraction"]
    )

    pa_min = float(
        pa_mask.min(skipna=True).values
    )

    pa_max = float(
        pa_mask.max(skipna=True).values
    )

    prevented_min = float(
        prevented_change_fraction.min(skipna=True).values
    )

    prevented_max = float(
        prevented_change_fraction.max(skipna=True).values
    )

    total_land_fraction = sum(
        adjusted_dataset[variable]
        for variable in ALL_VARS
    )

    full_land_values = total_land_fraction.where(
        total_land_fraction > 0.95
    )

    full_land_min = float(
        full_land_values.min(skipna=True).values
    )

    full_land_max = float(
        full_land_values.max(skipna=True).values
    )

    print("")
    print(
        f"      Sanity checks: {pa_name}, {scenario}, "
        f"{assumption_name}"
    )

    print(
        f"        PA fraction: "
        f"{pa_min:.6f} to {pa_max:.6f}"
    )

    print(
        f"        Prevented change: "
        f"{prevented_min:.6f} to {prevented_max:.6f}"
    )

    print(
        f"        Full-land total after normalization: "
        f"{full_land_min:.6f} to {full_land_max:.6f}"
    )

    if pa_min < -1e-6 or pa_max > 1.0 + 1e-6:
        raise ValueError(
            f"PA mask for {pa_name} contains values outside 0-1."
        )

    if (
        prevented_min < -1e-6
        or prevented_max > 1.0 + 1e-6
    ):
        raise ValueError(
            f"Prevented-change fraction for {pa_name}, "
            f"{assumption_name} contains values outside 0-1."
        )

    if prevented_max > retention_coefficient + 1e-5:
        raise ValueError(
            f"Maximum prevented-change fraction "
            f"({prevented_max:.6f}) exceeds the retention coefficient "
            f"({retention_coefficient:.6f})."
        )

    if not np.isclose(
        full_land_min,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            f"Minimum full-land total is {full_land_min:.8f}."
        )

    if not np.isclose(
        full_land_max,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            f"Maximum full-land total is {full_land_max:.8f}."
        )


# =====================================================================
# 13. SAVE OUTPUT
# =====================================================================

def save_adjusted_projection(
    adjusted_dataset,
    pa_name,
    scenario,
    assumption_name,
):
    """
    Save one adjusted projection using a transparent filename.
    """

    if assumption_name == "complete":

        output_name = (
            f"complete_retention_pa30_"
            f"{pa_name}_{scenario}_2080.nc"
        )

    elif assumption_name == "funding_constrained":

        output_name = (
            f"funding_constrained_pa30_"
            f"{pa_name}_{scenario}_2080.nc"
        )

    else:
        raise ValueError(
            f"Unknown assumption: {assumption_name}"
        )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name,
    )

    variables_to_encode = (
        ALL_VARS
        + [
            "pa_fraction",
            "prevented_change_fraction",
        ]
    )

    encoding = {
        variable: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
        }
        for variable in variables_to_encode
        if variable in adjusted_dataset.data_vars
    }

    # Remove inherited unlimited-dimension information because the
    # time dimension was removed when the 2080 slice was selected.
    adjusted_dataset.encoding.pop(
        "unlimited_dims",
        None,
    )

    adjusted_dataset.to_netcdf(
        output_path,
        encoding=encoding,
        unlimited_dims=[],
    )

    print(
        f"      Saved: {output_path}"
    )

    return output_path


# =====================================================================
# 14. MAIN EXECUTION
# =====================================================================

def main():
    """
    Generate all 30% PA-adjusted 2080 land-use projections.

    Number of outputs:

        2 PA configurations
        x 3 SSP scenarios
        x 2 retention assumptions
        = 12 files
    """

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print(
        ">> Starting 30% PA land-use projection workflow"
    )

    print(
        ">> Outputs will be written to: "
        f"{OUTPUT_DIR}"
    )

    # -----------------------------------------------------------------
    # Step 1. Load the common harmonized baseline
    # -----------------------------------------------------------------

    harmonized_2015 = load_harmonized_2015()

    # -----------------------------------------------------------------
    # Step 2. Load the original LUH2 2015 state
    # -----------------------------------------------------------------

    raw_2015 = load_raw_luh2_2015(
        target_grid=harmonized_2015,
    )

    # -----------------------------------------------------------------
    # Step 3. Load all raw future projections once
    # -----------------------------------------------------------------

    raw_future_scenarios = {}

    for scenario in SCENARIOS:

        raw_future_scenarios[scenario] = load_raw_luh2_2080(
            scenario=scenario,
            target_grid=harmonized_2015,
        )

    # -----------------------------------------------------------------
    # Step 4. Process each PA configuration
    # -----------------------------------------------------------------

    generated_files = []

    for pa_name, pa_path in PA_FILES.items():

        print("")
        print("====================================================")
        print(
            f">> Processing {PA_DESCRIPTIONS[pa_name]}"
        )
        print(
            f">> PA mask: {pa_path}"
        )
        print("====================================================")

        pa_mask = load_pa_mask(
            pa_path=pa_path,
            target_grid=harmonized_2015,
        )

        print(
            ">> Aligned PA-fraction range: "
            f"{float(pa_mask.min(skipna=True)):.6f} to "
            f"{float(pa_mask.max(skipna=True)):.6f}"
        )

        # -------------------------------------------------------------
        # Step 5. Process every SSP scenario
        # -------------------------------------------------------------

        for scenario in SCENARIOS:

            raw_2080 = raw_future_scenarios[scenario]

            # ---------------------------------------------------------
            # Step 6. Process both retention assumptions
            # ---------------------------------------------------------

            for (
                assumption_name,
                assumption_information,
            ) in RETENTION_ASSUMPTIONS.items():

                retention_coefficient = (
                    assumption_information["coefficient"]
                )

                print("")
                print(
                    f">> Generating {pa_name}, {scenario}, "
                    f"{assumption_name}"
                )

                print(
                    f"   Retention coefficient: "
                    f"{retention_coefficient:.4f}"
                )

                adjusted_2080 = generate_adjusted_projection(
                    harmonized_2015=harmonized_2015,
                    raw_2015=raw_2015,
                    raw_2080=raw_2080,
                    pa_mask=pa_mask,
                    pa_name=pa_name,
                    scenario=scenario,
                    assumption_name=assumption_name,
                    retention_coefficient=retention_coefficient,
                )

                output_path = save_adjusted_projection(
                    adjusted_dataset=adjusted_2080,
                    pa_name=pa_name,
                    scenario=scenario,
                    assumption_name=assumption_name,
                )

                generated_files.append(
                    output_path
                )

                run_sanity_checks(
                    adjusted_dataset=adjusted_2080,
                    pa_mask=pa_mask,
                    pa_name=pa_name,
                    scenario=scenario,
                    assumption_name=assumption_name,
                    retention_coefficient=retention_coefficient,
                )

                # Release each output before generating the next one.
                adjusted_2080.close()
                del adjusted_2080

    # -----------------------------------------------------------------
    # Step 7. Verify output count
    # -----------------------------------------------------------------

    expected_files = (
        len(PA_FILES)
        * len(SCENARIOS)
        * len(RETENTION_ASSUMPTIONS)
    )

    print("")
    print("====================================================")
    print(
        f">> Generated {len(generated_files)} of "
        f"{expected_files} expected files."
    )
    print("====================================================")

    if len(generated_files) != expected_files:
        raise RuntimeError(
            "The number of generated files does not match the "
            "expected number of PA, SSP, and retention combinations."
        )

    missing_outputs = [
        path
        for path in generated_files
        if not os.path.exists(path)
    ]

    if missing_outputs:
        raise RuntimeError(
            "The following expected output files were not found after "
            f"saving: {missing_outputs}"
        )

    print(
        ">> All 30% PA-adjusted land-use projections were "
        "generated successfully."
    )

    print("")
    print("Generated files:")

    for path in generated_files:
        print(
            f"  - {path}"
        )


# =====================================================================
# 15. RUN SCRIPT
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