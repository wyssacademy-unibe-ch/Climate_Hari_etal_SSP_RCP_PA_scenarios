#!/usr/bin/env python3

"""
Construct the harmonized 2015 land-use baseline for the existing
17%-PA network and generate 2080 projections under two land-use
retention assumptions:

    1. Complete land-use retention: Full-effectiveness
    2. Funding-constrained land-use retention: Limited-effectiveness

WORKFLOW
========

STAGE 1: HARMONIZED 2015 BASELINE
---------------------------------

The original 2015 LUH2 land-use fractions are combined with the
proportional WDPA mask.

The original LUH2 composition is retained for most grid cells.

An adjustment is applied only to cells that:

    1. have PA coverage greater than zero;
    2. contain valid terrestrial land;
    3. are represented entirely by anthropogenic LUH2 classes.

In these cells, an amount equal to PA coverage is reassigned from
anthropogenic land-use classes to secondary non-forested land.

For example:

    PA fraction = 0.20
    natural land = 0.00
    anthropogenic land = 1.00

After harmonisation:

    secondary non-forested land = 0.20
    anthropogenic land = 0.80

The relative proportions of the anthropogenic classes are preserved
when their total fraction is reduced.

This operation defines an assumed 2015 baseline condition. It is not
a future restoration simulation.


STAGE 2: RAW LUH2 PROJECTED CHANGE
----------------------------------

For each land-use class and SSP scenario, the original LUH2 change is
calculated as:

    raw_delta = raw_LUH2_2080 - raw_LUH2_2015

This preserves the land-use trajectory projected by LUH2 independently
of the baseline harmonisation.


STAGE 3: PA-MEDIATED LAND-USE CHANGE
------------------------------------

The raw LUH2 delta is applied to the harmonised 2015 baseline:

    adjusted_2080 =
        harmonized_2015
        + raw_delta
        * (1 - PA_fraction * retention_coefficient)

The term:

    PA_fraction * retention_coefficient

represents the assumed proportion of projected grid-cell change
prevented by protection.


COMPLETE LAND-USE RETENTION
---------------------------

For complete retention:

    retention_coefficient = 1.0

Examples:

    PA fraction = 0:
        0% of projected change is prevented.
        100% proceeds.

    PA fraction = 0.4:
        40% of projected change is prevented.
        60% proceeds.

    PA fraction = 1:
        100% of projected change is prevented.
        The harmonised 2015 composition is retained.


FUNDING-CONSTRAINED LAND-USE RETENTION
--------------------------------------

For the existing 17%-PA network:

    retention_coefficient = 24.3 / 67.6 = 0.3595

The ratio represents current estimated global PA expenditure divided
by the estimated minimum adequate annual budget for the existing PA
system.

Examples:

    PA fraction = 0:
        0% of projected change is prevented.
        100% proceeds.

    PA fraction = 0.4:
        0.4 * 0.3595 = 0.1438
        14.38% of projected change is prevented.
        85.62% proceeds.

    PA fraction = 1:
        35.95% of projected change is prevented.
        64.05% proceeds.

The funding ratio is a scenario parameter. It is not an empirical
measurement of protected-area effectiveness or observed avoided
land-use conversion.


OUTPUTS
=======

One harmonised baseline:

    baseline_wdpa17_2015_hist.nc

Three complete-retention projections:

    complete_retention_pa17_2080_ssp126.nc
    complete_retention_pa17_2080_ssp245.nc
    complete_retention_pa17_2080_ssp460.nc

Three funding-constrained projections:

    funding_constrained_pa17_2080_ssp126.nc
    funding_constrained_pa17_2080_ssp245.nc
    funding_constrained_pa17_2080_ssp460.nc
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

WDPA_PATH = (
    "/capacity/occr_davin/mguzman/chari_P2_review/"
    "data/PA_masks/WDPA_percentage_map.nc"
)

HIST_LUH_PATH = (
    "/capacity/occr_davin/chari/P1/LUH2/"
    "remapped_luh2_historical.nc"
)

FUTURE_LUH_DIR = (
    "/capacity/occr_davin/chari/P1/LUH2"
)

OUTPUT_DIR = (
    "/capacity/occr_davin/mguzman/chari_P2_review/"
    "data/new_LU"
)

HIST_OUT_PATH = os.path.join(
    OUTPUT_DIR,
    "baseline_wdpa17_2015_hist.nc",
)


# =====================================================================
# 2. VARIABLES
# =====================================================================

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

WDPA_VARIABLE = "WDPA_percentage_map.tif"


# =====================================================================
# 3. SCENARIOS AND RETENTION PARAMETERS
# =====================================================================

SCENARIOS = [
    "ssp126",
    "ssp245",
    "ssp460",
]

RETENTION_ASSUMPTIONS = {
    "complete": {
        "coefficient": 1.0,
        "description": "Complete land-use retention",
        "output_prefix": "complete_retention",
        "calculation": "1.0",
    },
    "funding_constrained": {
        "coefficient": 0.3595,
        "description": "Funding-constrained land-use retention",
        "output_prefix": "funding_constrained",
        "calculation": "24.3 / 67.6",
    },
}


# =====================================================================
# 4. TIME INDICES
# =====================================================================

# These indices are retained from the established workflow.
#
# Historical index 1165 represents 2015 in the original LUH2 dataset.
# Future index 65 represent 2080 in the original LUH2 dataset.
HISTORICAL_2015_INDEX = 1165
FUTURE_2080_INDEX = 65


# =====================================================================
# 5. NUMERICAL THRESHOLDS
# =====================================================================

NATURAL_COVER_TOLERANCE = 1e-6
PA_COVERAGE_TOLERANCE = 0.001
LAND_COVER_TOLERANCE = 0.1

# Cells with a combined land-use fraction above this threshold are
# treated as full-land cells during normalization.
FULL_LAND_THRESHOLD = 0.95

FLOAT_TOLERANCE = 1e-5


# =====================================================================
# 6. COORDINATE AND VALIDATION FUNCTIONS
# =====================================================================

def standardize_coordinates(dataset):
    """
    Rename spatial dimensions to 'lat' and 'lon' when necessary.
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


def sort_spatial_coordinates(data):
    """
    Sort latitude and longitude in ascending order if necessary.
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
    Test whether two datasets or arrays use the same spatial grid.
    """

    if source.sizes.get("lat") != target.sizes.get("lat"):
        return False

    if source.sizes.get("lon") != target.sizes.get("lon"):
        return False

    lat_match = np.allclose(
        source["lat"].values,
        target["lat"].values,
        equal_nan=True,
    )

    lon_match = np.allclose(
        source["lon"].values,
        target["lon"].values,
        equal_nan=True,
    )

    return lat_match and lon_match


def validate_land_use_variables(dataset, dataset_name):
    """
    Confirm that all required LUH2 variables are present.
    """

    missing_variables = [
        variable
        for variable in ALL_VARS
        if variable not in dataset.data_vars
    ]

    if missing_variables:
        raise KeyError(
            f"{dataset_name} is missing the following variables: "
            f"{missing_variables}"
        )


def validate_time_index(dataset, index, dataset_name):
    """
    Confirm that the requested time index exists.
    """

    if "time" not in dataset.dims:
        raise ValueError(
            f"{dataset_name} does not contain a time dimension."
        )

    if index >= dataset.sizes["time"]:
        raise IndexError(
            f"Time index {index} is outside the available range for "
            f"{dataset_name}. The dataset contains "
            f"{dataset.sizes['time']} time steps."
        )


# =====================================================================
# 7. COASTAL-SAFE NORMALIZATION
# =====================================================================

def normalize_coastal_safe(dataset, variable_list):
    """
    Normalize full-land cells while preserving coastal land fractions.

    LUH2 fractions may sum to less than one in coastal cells because
    only part of a 0.5-degree grid cell is terrestrial. Normalizing
    every non-zero cell to one would inflate terrestrial land in
    coastal cells.

    Normalization is therefore applied only where the combined
    land-use fraction exceeds FULL_LAND_THRESHOLD.
    """

    total_sum = sum(
        dataset[variable]
        for variable in variable_list
    )

    full_land_mask = total_sum > FULL_LAND_THRESHOLD

    for variable in variable_list:
        normalized = (
            dataset[variable]
            / (total_sum + 1e-12)
        )

        dataset[variable] = (
            xr.where(
                full_land_mask,
                normalized,
                dataset[variable],
            )
            .fillna(0)
            .clip(min=0, max=1)
            .astype(np.float32)
        )

    return dataset


# =====================================================================
# 8. LOAD ORIGINAL 2015 LUH2 DATA
# =====================================================================

def load_raw_luh2_2015():
    """
    Load the original LUH2 2015 land-use fractions.
    """

    if not os.path.exists(HIST_LUH_PATH):
        raise FileNotFoundError(
            f"Historical LUH2 file not found: {HIST_LUH_PATH}"
        )

    print(
        f">> Loading original historical LUH2 data: "
        f"{HIST_LUH_PATH}"
    )

    with xr.open_dataset(
        HIST_LUH_PATH,
        decode_times=False,
    ) as source:

        source = standardize_coordinates(source)
        source = sort_spatial_coordinates(source)

        validate_land_use_variables(
            source,
            "Historical LUH2 dataset",
        )

        validate_time_index(
            source,
            HISTORICAL_2015_INDEX,
            "Historical LUH2 dataset",
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

    raw_2015 = sort_spatial_coordinates(raw_2015)

    return raw_2015


# =====================================================================
# 9. LOAD AND ALIGN WDPA MASK
# =====================================================================

def load_wdpa_fraction(target_grid):
    """
    Load the WDPA percentage layer and convert it to a 0-to-1 fraction.
    """

    if not os.path.exists(WDPA_PATH):
        raise FileNotFoundError(
            f"WDPA file not found: {WDPA_PATH}"
        )

    print(
        f">> Loading WDPA percentage mask: {WDPA_PATH}"
    )

    with xr.open_dataset(
        WDPA_PATH,
        decode_times=False,
    ) as source:

        source = standardize_coordinates(source)

        if WDPA_VARIABLE not in source.data_vars:
            raise KeyError(
                f"Variable '{WDPA_VARIABLE}' was not found in "
                f"{WDPA_PATH}. Available variables are: "
                f"{list(source.data_vars)}"
            )

        wdpa_percentage = (
            source[WDPA_VARIABLE]
            .load()
            .squeeze(drop=True)
        )

    wdpa_percentage = sort_spatial_coordinates(
        wdpa_percentage
    )

    pa_fraction = (
        wdpa_percentage / 100.0
    )

    if not coordinates_match(
        pa_fraction,
        target_grid,
    ):
        print(
            ">> Aligning WDPA mask with the LUH2 grid."
        )

        pa_fraction = pa_fraction.interp(
            lat=target_grid["lat"],
            lon=target_grid["lon"],
            method="nearest",
        )

    pa_fraction = (
        pa_fraction
        .fillna(0)
        .clip(min=0, max=1)
        .astype(np.float32)
    )

    print(
        ">> Aligned PA-fraction range: "
        f"{float(pa_fraction.min(skipna=True)):.6f} to "
        f"{float(pa_fraction.max(skipna=True)):.6f}"
    )

    return pa_fraction


# =====================================================================
# 10. CONSTRUCT HARMONIZED 2015 BASELINE
# =====================================================================

def harmonize_2015_baseline(
    raw_2015,
    pa_fraction,
):
    """
    Construct the common harmonized 2015 land-use baseline.

    The adjustment is limited to PA-containing grid cells represented
    entirely by anthropogenic LUH2 classes.

    Within those cells, an amount equal to PA coverage is transferred
    from anthropogenic land-use classes to secondary non-forested land.

    Anthropogenic classes are reduced proportionally to preserve their
    relative composition.
    """

    harmonized = raw_2015[
        ALL_VARS
    ].copy(deep=True)

    total_natural = sum(
        raw_2015[variable]
        for variable in NATURAL_VARS
    )

    total_human = sum(
        raw_2015[variable]
        for variable in HUMAN_VARS
    )

    land_fraction = (
        total_natural + total_human
    ).clip(min=0, max=1)

    affected_cells = (
        (total_natural < NATURAL_COVER_TOLERANCE)
        & (pa_fraction > PA_COVERAGE_TOLERANCE)
        & (total_human > LAND_COVER_TOLERANCE)
    )

    # Protection cannot exceed the terrestrial portion of a coastal cell.
    protected_land_fraction = xr.where(
        pa_fraction > land_fraction,
        land_fraction,
        pa_fraction,
    ).clip(
        min=0,
        max=1,
    )

    # Amount of anthropogenic land retained after assigning the
    # protected fraction to secondary non-forested land.
    retained_human_total = (
        land_fraction - protected_land_fraction
    ).clip(
        min=0,
        max=1,
    )

    # Reduce anthropogenic classes proportionally.
    human_scaling = xr.where(
        affected_cells & (total_human > 1e-12),
        retained_human_total / (total_human + 1e-12),
        1.0,
    )

    for variable in HUMAN_VARS:
        harmonized[variable] = xr.where(
            affected_cells,
            raw_2015[variable] * human_scaling,
            raw_2015[variable],
        ).astype(np.float32)

    # Preserve the original natural classes and add the transferred
    # fraction to secondary non-forested land.
    for variable in NATURAL_VARS:
        if variable == "secdn":
            harmonized[variable] = xr.where(
                affected_cells,
                raw_2015[variable] + protected_land_fraction,
                raw_2015[variable],
            ).astype(np.float32)

        else:
            harmonized[variable] = (
                raw_2015[variable]
                .astype(np.float32)
            )

    harmonized = normalize_coastal_safe(
        harmonized,
        ALL_VARS,
    )

    harmonized["pa_fraction"] = (
        pa_fraction
        .fillna(0)
        .clip(min=0, max=1)
        .astype(np.float32)
    )

    harmonized["pa_fraction"].attrs.update(
        {
            "long_name": (
                "Proportion of grid cell covered by the existing "
                "protected-area network"
            ),
            "units": "1",
            "source": WDPA_PATH,
        }
    )

    harmonized.attrs.update(
        {
            "title": (
                "Harmonized 2015 LUH2 and WDPA baseline"
            ),
            "baseline_year": 2015,
            "baseline_adjustment": (
                "In PA-containing cells represented entirely by "
                "anthropogenic LUH2 classes, an amount equal to PA "
                "coverage was reassigned to secondary non-forested land."
            ),
            "baseline_adjustment_interpretation": (
                "One-time baseline assumption, not a projection of "
                "future restoration."
            ),
            "adjusted_cells_count": int(
                affected_cells.sum().values
            ),
        }
    )

    print(
        f">> Baseline cells adjusted: "
        f"{int(affected_cells.sum().values)}"
    )

    return harmonized


# =====================================================================
# 11. SAVE HARMONIZED BASELINE
# =====================================================================

def save_harmonized_baseline(harmonized_2015):
    """
    Save the harmonized 2015 baseline.
    """

    variables_to_encode = ALL_VARS + ["pa_fraction"]

    encoding = {
        variable: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
        }
        for variable in variables_to_encode
        if variable in harmonized_2015.data_vars
    }

    # Remove inherited encoding associated with the historical
    # time dimension, which is absent from the 2015 slice.
    harmonized_2015.encoding.pop(
        "unlimited_dims",
        None,
    )

    harmonized_2015.to_netcdf(
        HIST_OUT_PATH,
        encoding=encoding,
        unlimited_dims=[],
    )

    print(
        f">> Harmonized 2015 baseline saved: "
        f"{HIST_OUT_PATH}"
    )


# =====================================================================
# 12. LOAD RAW 2080 LUH2 PROJECTION
# =====================================================================

def load_raw_luh2_2080(
    scenario,
    target_grid,
):
    """
    Load the original LUH2 2080 projection for one SSP scenario.
    """

    future_path = os.path.join(
        FUTURE_LUH_DIR,
        f"remapped_luh2_{scenario}.nc",
    )

    if not os.path.exists(future_path):
        raise FileNotFoundError(
            f"Future LUH2 file not found: {future_path}"
        )

    print(
        f">> Loading raw LUH2 2080 projection: {scenario}"
    )

    with xr.open_dataset(
        future_path,
        decode_times=False,
    ) as source:

        source = standardize_coordinates(source)
        source = sort_spatial_coordinates(source)

        validate_land_use_variables(
            source,
            f"LUH2 projection for {scenario}",
        )

        validate_time_index(
            source,
            FUTURE_2080_INDEX,
            f"LUH2 projection for {scenario}",
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

    if not coordinates_match(
        raw_2080,
        target_grid,
    ):
        print(
            f">> Aligning {scenario} with the 2015 grid."
        )

        raw_2080 = raw_2080.interp(
            lat=target_grid["lat"],
            lon=target_grid["lon"],
            method="nearest",
        )

    return raw_2080


# =====================================================================
# 13. GENERATE ONE PA-ADJUSTED 2080 PROJECTION
# =====================================================================

def generate_pa_adjusted_projection(
    harmonized_2015,
    raw_2015,
    raw_2080,
    pa_fraction,
    scenario,
    assumption_name,
    assumption_info,
):
    """
    Generate one 17%-PA-adjusted 2080 land-use projection.
    """

    retention_coefficient = assumption_info["coefficient"]

    adjusted_2080 = harmonized_2015[
        ALL_VARS
    ].copy(deep=True)

    # Proportion of projected grid-cell change prevented.
    prevented_change_fraction = (
        pa_fraction * retention_coefficient
    ).clip(
        min=0,
        max=1,
    ).astype(np.float32)

    # Proportion of projected grid-cell change allowed to proceed.
    allowed_change_fraction = (
        1.0 - prevented_change_fraction
    ).astype(np.float32)

    for variable in ALL_VARS:
        # Original LUH2 change between 2015 and 2080.
        raw_projected_delta = (
            raw_2080[variable]
            - raw_2015[variable]
        )

        # Apply the original LUH2 delta to the harmonized baseline.
        adjusted_2080[variable] = (
            harmonized_2015[variable]
            + raw_projected_delta * allowed_change_fraction
        ).clip(
            min=0,
            max=1,
        ).astype(np.float32)

    adjusted_2080 = normalize_coastal_safe(
        adjusted_2080,
        ALL_VARS,
    )

    # Store supporting layers for reproducibility.
    adjusted_2080["pa_fraction"] = (
        pa_fraction
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
                "Proportion of grid cell covered by the existing "
                "protected-area network"
            ),
            "units": "1",
        }
    )

    adjusted_2080[
        "prevented_change_fraction"
    ].attrs.update(
        {
            "long_name": (
                "Assumed proportion of projected grid-cell land-use "
                "change prevented"
            ),
            "units": "1",
            "calculation": (
                "pa_fraction multiplied by retention_coefficient"
            ),
        }
    )

    adjusted_2080.attrs.update(
        {
            "title": (
                "17%-PA-adjusted LUH2 land-use projection"
            ),
            "pa_configuration": (
                "Existing 17%-PA network"
            ),
            "land_use_retention_assumption": (
                assumption_info["description"]
            ),
            "retention_coefficient": float(
                retention_coefficient
            ),
            "retention_coefficient_calculation": (
                assumption_info["calculation"]
            ),
            "baseline_year": 2015,
            "projection_year": 2080,
            "ssp_scenario": scenario,
            "adjustment_equation": (
                "adjusted_2080 = harmonized_2015 + "
                "(raw_LUH2_2080 - raw_LUH2_2015) * "
                "(1 - pa_fraction * retention_coefficient)"
            ),
            "common_baseline": HIST_OUT_PATH,
            "raw_historical_source": HIST_LUH_PATH,
        }
    )

    if assumption_name == "complete":
        adjusted_2080.attrs["methodological_note"] = (
            "Complete land-use retention assumes that protection "
            "prevents all projected land-use change within the "
            "protected fraction of each grid cell."
        )

    elif assumption_name == "funding_constrained":
        adjusted_2080.attrs.update(
            {
                "current_expenditure_usd_billion_per_year": 24.3,
                "minimum_funding_requirement_usd_billion_per_year": (
                    67.6
                ),
                "funding_ratio": float(
                    retention_coefficient
                ),
                "funding_ratio_interpretation": (
                    "Proportion of the estimated minimum funding "
                    "requirement for the existing PA system covered "
                    "by current expenditure"
                ),
                "methodological_note": (
                    "The funding ratio is used as a scenario parameter "
                    "scaling the share of projected land-use change "
                    "prevented. It is not an empirical estimate of PA "
                    "effectiveness or avoided land-use conversion."
                ),
            }
        )

    return adjusted_2080


# =====================================================================
# 14. SANITY CHECKS
# =====================================================================

def run_sanity_checks(
    dataset,
    scenario,
    assumption_name,
    retention_coefficient,
):
    """
    Check PA values, prevented-change values, and land-use totals.
    """

    pa_fraction = dataset["pa_fraction"]

    prevented_fraction = dataset[
        "prevented_change_fraction"
    ]

    pa_min = float(
        pa_fraction.min(skipna=True).values
    )

    pa_max = float(
        pa_fraction.max(skipna=True).values
    )

    prevented_min = float(
        prevented_fraction.min(skipna=True).values
    )

    prevented_max = float(
        prevented_fraction.max(skipna=True).values
    )

    land_total = sum(
        dataset[variable]
        for variable in ALL_VARS
    )

    full_land_values = land_total.where(
        land_total > FULL_LAND_THRESHOLD
    )

    full_land_min = float(
        full_land_values.min(skipna=True).values
    )

    full_land_max = float(
        full_land_values.max(skipna=True).values
    )

    print(
        f">> Sanity checks for {scenario}, "
        f"{assumption_name}"
    )

    print(
        f"   PA fraction: "
        f"{pa_min:.6f} to {pa_max:.6f}"
    )

    print(
        f"   Prevented change: "
        f"{prevented_min:.6f} to "
        f"{prevented_max:.6f}"
    )

    print(
        f"   Full-land total after normalization: "
        f"{full_land_min:.6f} to "
        f"{full_land_max:.6f}"
    )

    if pa_min < -FLOAT_TOLERANCE:
        raise ValueError(
            "PA fraction contains values below zero."
        )

    if pa_max > 1.0 + FLOAT_TOLERANCE:
        raise ValueError(
            "PA fraction contains values above one."
        )

    if prevented_min < -FLOAT_TOLERANCE:
        raise ValueError(
            "Prevented-change fraction contains values below zero."
        )

    if prevented_max > retention_coefficient + FLOAT_TOLERANCE:
        raise ValueError(
            "Maximum prevented-change fraction exceeds the "
            "retention coefficient."
        )

    if not np.isclose(
        full_land_min,
        1.0,
        atol=FLOAT_TOLERANCE,
    ):
        raise ValueError(
            f"Minimum full-land total is "
            f"{full_land_min:.8f}, not 1."
        )

    if not np.isclose(
        full_land_max,
        1.0,
        atol=FLOAT_TOLERANCE,
    ):
        raise ValueError(
            f"Maximum full-land total is "
            f"{full_land_max:.8f}, not 1."
        )


# =====================================================================
# 15. SAVE ONE PROJECTION
# =====================================================================

def save_projection(
    dataset,
    scenario,
    assumption_name,
    assumption_info,
):
    """
    Save one complete or funding-constrained 17%-PA projection.
    """

    output_name = (
        f"{assumption_info['output_prefix']}_"
        f"pa17_2080_{scenario}.nc"
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
        if variable in dataset.data_vars
    }

    # Remove inherited information about the deleted time dimension.
    dataset.encoding.pop(
        "unlimited_dims",
        None,
    )

    dataset.to_netcdf(
        output_path,
        encoding=encoding,
        unlimited_dims=[],
    )

    print(
        f">> Saved: {output_path}"
    )

    return output_path


# =====================================================================
# 16. MAIN WORKFLOW
# =====================================================================

def main():
    """
    Construct the harmonized baseline and generate six 2080 projections.
    """

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print(
        ">> Starting 17%-PA baseline and projection workflow"
    )

    print(
        f">> Output directory: {OUTPUT_DIR}"
    )

    # -----------------------------------------------------------------
    # Step 1. Load original 2015 LUH2 data.
    # -----------------------------------------------------------------

    raw_2015 = load_raw_luh2_2015()

    # -----------------------------------------------------------------
    # Step 2. Load and align WDPA coverage.
    # -----------------------------------------------------------------

    pa_fraction = load_wdpa_fraction(
        target_grid=raw_2015,
    )

    # -----------------------------------------------------------------
    # Step 3. Construct and save harmonized 2015 baseline.
    # -----------------------------------------------------------------

    harmonized_2015 = harmonize_2015_baseline(
        raw_2015=raw_2015,
        pa_fraction=pa_fraction,
    )

    save_harmonized_baseline(
        harmonized_2015
    )

    # -----------------------------------------------------------------
    # Step 4. Generate complete and funding-constrained projections.
    # -----------------------------------------------------------------

    generated_files = []

    for scenario in SCENARIOS:
        print("")
        print("====================================================")
        print(
            f">> Processing 17%-PA scenario: {scenario}"
        )
        print("====================================================")

        raw_2080 = load_raw_luh2_2080(
            scenario=scenario,
            target_grid=raw_2015,
        )

        for (
            assumption_name,
            assumption_info,
        ) in RETENTION_ASSUMPTIONS.items():

            retention_coefficient = (
                assumption_info["coefficient"]
            )

            print("")
            print(
                f">> Applying "
                f"{assumption_info['description']}"
            )

            print(
                f">> Retention coefficient: "
                f"{retention_coefficient:.4f}"
            )

            adjusted_2080 = generate_pa_adjusted_projection(
                harmonized_2015=harmonized_2015,
                raw_2015=raw_2015,
                raw_2080=raw_2080,
                pa_fraction=pa_fraction,
                scenario=scenario,
                assumption_name=assumption_name,
                assumption_info=assumption_info,
            )

            output_path = save_projection(
                dataset=adjusted_2080,
                scenario=scenario,
                assumption_name=assumption_name,
                assumption_info=assumption_info,
            )

            generated_files.append(
                output_path
            )

            run_sanity_checks(
                dataset=adjusted_2080,
                scenario=scenario,
                assumption_name=assumption_name,
                retention_coefficient=retention_coefficient,
            )

            adjusted_2080.close()
            del adjusted_2080

        raw_2080.close()
        del raw_2080

    # -----------------------------------------------------------------
    # Step 5. Verify all expected projections.
    # -----------------------------------------------------------------

    expected_projections = (
        len(SCENARIOS)
        * len(RETENTION_ASSUMPTIONS)
    )

    print("")
    print("====================================================")
    print(
        f">> Generated {len(generated_files)} of "
        f"{expected_projections} expected 2080 projections."
    )
    print("====================================================")

    if len(generated_files) != expected_projections:
        raise RuntimeError(
            "The number of generated projections does not match "
            "the expected number."
        )

    missing_outputs = [
        path
        for path in generated_files
        if not os.path.exists(path)
    ]

    if missing_outputs:
        raise RuntimeError(
            "The following outputs were not found after writing: "
            f"{missing_outputs}"
        )

    print(
        ">> 17%-PA workflow completed successfully."
    )

    print("")
    print("Generated files:")

    print(
        f"  - {HIST_OUT_PATH}"
    )

    for output_path in generated_files:
        print(
            f"  - {output_path}"
        )


# =====================================================================
# 17. RUN
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
