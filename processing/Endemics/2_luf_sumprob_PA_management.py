#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply a species-specific land-use filter to climate-driven,
limited-dispersal occurrence probabilities for endemic species only.

Supported analyses:
    1. Harmonized 2015 baseline
    2. Climate-only projections for 2080
    3. Existing 17%-PA configuration for 2080
    4. 30% biodiversity-only PA configuration for 2080
    5. 30% multi-objective PA configuration for 2080

The future PA configurations support:
    - complete land-use retention
    - funding-constrained land-use retention

For each endemic species:

    P_clim_disp = P_clim * dispersal_mask

The fraction of suitable land use is calculated by summing all
distinct LUH2 land-use categories associated with IUCN habitats
classified as Suitable:

    H_suit = sum of suitable LUH2 land-use fractions

The land-use-filtered occurrence probability is:

    P_LUF = P_clim_disp * H_suit

If no habitat file exists, or no suitable LUH2 category can be
matched, P_LUF is set to zero.

Probabilities are summed across endemic species within each taxonomic
group. Outputs therefore represent taxon-specific summed occurrence
probabilities used as continuous estimates of endemic species richness.

Historical clarification:
    The historical analysis uses the harmonized 2015 land-use baseline,
    but reads the EWEMBI_1995 probability column retained in the
    original species projection files.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)


# =====================================================================
# 1. ARGUMENTS
# =====================================================================

parser = argparse.ArgumentParser(
    description=(
        "Apply the species-specific land-use filter to climate-driven "
        "and limited-dispersal occurrence probabilities for endemic "
        "species only."
    )
)

parser.add_argument(
    "-m",
    "--model",
    type=str,
    nargs="+",
    required=True,
    help="Species distribution model or models.",
)

parser.add_argument(
    "-a",
    "--taxa",
    type=str,
    nargs="+",
    required=True,
    help="Taxonomic group or groups.",
)

parser.add_argument(
    "-g",
    "--gcm",
    type=str,
    nargs="+",
    required=True,
    help=(
        "General circulation model or models. "
        "Use EWEMBI for the historical 2015 run."
    ),
)

parser.add_argument(
    "-s",
    "--scenario",
    type=str,
    choices=[
        "rcp26",
        "rcp45",
        "rcp60",
    ],
    required=True,
    help=(
        "Scenario label. For rcp45, SSP2-4.5 land use is paired "
        "with RCP2.6 climate because RCP4.5 SDM projections are "
        "unavailable."
    ),
)

parser.add_argument(
    "-y",
    "--year",
    type=str,
    choices=[
        "2015",
        "2080",
    ],
    required=True,
    help="Analysis year.",
)

parser.add_argument(
    "--type",
    type=str,
    choices=[
        "baseline",
        "climate_only",
        "pa17",
        "pa30_bio",
        "pa30_bcw",
    ],
    required=True,
    help="Land-use and protected-area configuration.",
)

parser.add_argument(
    "--retention",
    type=str,
    choices=[
        "complete",
        "funding_constrained",
    ],
    default=None,
    help=(
        "Land-use-retention assumption for 2080 PA scenarios. "
        "Not used for the historical baseline or climate-only run."
    ),
)

args = parser.parse_args()


# =====================================================================
# 2. PATHS
# =====================================================================

LAND_USE_DIR = (
    "/capacity/occr_davin/mguzman/chari_P2_review/"
    "data/new_LU"
)

CONVERSION_TABLE = (
    "/capacity/occr_davin/chari/P1/"
    "IUCN_LUH_converion_table_Carlson.csv"
)

HABITAT_ROOT = (
    "/capacity/occr_davin/chari/P1/"
    "Habitat_Classifications"
)

SPECIES_PROJECTION_ROOT = (
    "/capacity/occr_davin/chari/P1/"
    "BioScen15/individual_projections"
)

MATCH_FILE_PATH = (
    "/capacity/occr_davin/mguzman/chari_P2_review/"
    "Endemics/data/modeled_endemics_matches.csv"
)

OUTPUT_ROOT = (
    "/capacity/occr_davin/mguzman/chari_P2_review/"
    "Endemics/PA_scenarios/Sumprob_revised"
)


# =====================================================================
# 3. SCENARIO AND GCM MAPPINGS
# =====================================================================

SSP_MAP = {
    "rcp26": "ssp126",
    "rcp45": "ssp245",
    "rcp60": "ssp460",
}

GCM_MAP = {
    "GFDL-ESM2M": "GFDL.ESM2M",
    "IPSL-CM5A-LR": "IPSL.CM5A-LR",
    "MIROC5": "MIROC5",
    "HadGEM2-ES": "HadGEM2.ES",
}

PA_TYPES = {
    "pa17",
    "pa30_bio",
    "pa30_bcw",
}


# =====================================================================
# 4. VALIDATE ARGUMENT COMBINATIONS
# =====================================================================

if args.year == "2015" and args.type != "baseline":
    parser.error(
        "Year 2015 must be used with --type baseline."
    )

if args.year == "2080" and args.type == "baseline":
    parser.error(
        "The baseline type represents 2015. Use --type pa17 "
        "for the existing 17%-PA configuration in 2080."
    )

if args.type in PA_TYPES and args.year != "2080":
    parser.error(
        "pa17, pa30_bio, and pa30_bcw are 2080 projections."
    )

if args.type in PA_TYPES and args.retention is None:
    parser.error(
        "--retention is required for PA scenarios."
    )

if args.type not in PA_TYPES and args.retention is not None:
    parser.error(
        "--retention applies only to PA scenarios."
    )

if args.type == "climate_only" and args.year != "2080":
    parser.error(
        "climate_only must be evaluated for 2080."
    )

if args.year == "2015" and args.gcm != ["EWEMBI"]:
    parser.error(
        "Historical 2015 processing must use -g EWEMBI."
    )


# =====================================================================
# 5. SELECT LAND-USE INPUT AND OUTPUT DIRECTORY
# =====================================================================

ssp = SSP_MAP[args.scenario]

if args.year == "2015":
    LU_FILE = os.path.join(
        LAND_USE_DIR,
        "baseline_wdpa17_2015_hist.nc",
    )
    OUTPUT_DIR = os.path.join(
        OUTPUT_ROOT,
        "WDPA",
        "Historical",
    )
    SUFFIX = "baseline_2015"

elif args.type == "climate_only":
    LU_FILE = os.path.join(
        LAND_USE_DIR,
        "baseline_wdpa17_2015_hist.nc",
    )
    OUTPUT_DIR = os.path.join(
        OUTPUT_ROOT,
        "WDPA",
        "Climate_Only",
    )
    SUFFIX = "climate_only"

elif args.type == "pa17":
    prefix = (
        "complete_retention"
        if args.retention == "complete"
        else "funding_constrained"
    )
    LU_FILE = os.path.join(
        LAND_USE_DIR,
        f"{prefix}_pa17_2080_{ssp}.nc",
    )
    OUTPUT_DIR = os.path.join(
        OUTPUT_ROOT,
        "PA17",
        args.retention,
    )
    SUFFIX = f"pa17_{args.retention}"

elif args.type == "pa30_bio":
    prefix = (
        "complete_retention"
        if args.retention == "complete"
        else "funding_constrained"
    )
    LU_FILE = os.path.join(
        LAND_USE_DIR,
        f"{prefix}_pa30_bio_{ssp}_2080.nc",
    )
    OUTPUT_DIR = os.path.join(
        OUTPUT_ROOT,
        "PA30_Bio",
        args.retention,
    )
    SUFFIX = f"pa30_bio_{args.retention}"

elif args.type == "pa30_bcw":
    prefix = (
        "complete_retention"
        if args.retention == "complete"
        else "funding_constrained"
    )
    LU_FILE = os.path.join(
        LAND_USE_DIR,
        f"{prefix}_pa30_bcw_{ssp}_2080.nc",
    )
    OUTPUT_DIR = os.path.join(
        OUTPUT_ROOT,
        "PA30_BCW",
        args.retention,
    )
    SUFFIX = f"pa30_bcw_{args.retention}"

else:
    raise RuntimeError(
        f"Unsupported analysis type: {args.type}"
    )

for required_path, label in [
    (LU_FILE, "Land-use file"),
    (CONVERSION_TABLE, "Conversion table"),
    (MATCH_FILE_PATH, "Endemic-species match file"),
]:
    if not os.path.exists(required_path):
        raise FileNotFoundError(
            f"{label} not found: {required_path}"
        )

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# =====================================================================
# 6. LOAD AND VALIDATE ENDEMIC-SPECIES METADATA
# =====================================================================

endemic_matches = pd.read_csv(
    MATCH_FILE_PATH
)

required_match_columns = {
    "taxon",
    "modeled_name",
}

missing_match_columns = (
    required_match_columns
    - set(endemic_matches.columns)
)

if missing_match_columns:
    raise ValueError(
        "Endemic-species match file is missing columns: "
        f"{sorted(missing_match_columns)}"
    )

endemic_matches = endemic_matches.dropna(
    subset=[
        "taxon",
        "modeled_name",
    ]
).copy()

endemic_matches["taxon"] = (
    endemic_matches["taxon"]
    .astype(str)
    .str.strip()
)

endemic_matches["modeled_name"] = (
    endemic_matches["modeled_name"]
    .astype(str)
    .str.strip()
)

if endemic_matches.empty:
    raise ValueError(
        "Endemic-species match file contains no usable records."
    )


# =====================================================================
# 7. LOAD LAND-USE DATA
# =====================================================================

print(f">> Land-use input: {LU_FILE}")
print(f">> Endemic-species match file: {MATCH_FILE_PATH}")
print(f">> Output directory: {OUTPUT_DIR}")

with xr.open_dataset(
    LU_FILE,
    decode_times=False,
) as source:
    land_use_data = source.load()

rename_dict = {}

if (
    "latitude" in land_use_data.coords
    or "latitude" in land_use_data.dims
):
    rename_dict["latitude"] = "lat"

if (
    "longitude" in land_use_data.coords
    or "longitude" in land_use_data.dims
):
    rename_dict["longitude"] = "lon"

if rename_dict:
    land_use_data = land_use_data.rename(
        rename_dict
    )

if (
    "lat" not in land_use_data.coords
    or "lon" not in land_use_data.coords
):
    raise ValueError(
        "Land-use file does not contain lat and lon coordinates."
    )

land_use_data = (
    land_use_data
    .sortby("lat")
    .sortby("lon")
)

# Only these variables are valid LUH2 land-use categories.
# Diagnostic variables such as pa_fraction and
# prevented_change_fraction are excluded.
LUH2_VARS = {
    "primf",
    "primn",
    "secdf",
    "secdn",
    "range",
    "c3ann",
    "c3per",
    "c4ann",
    "c4per",
    "c3nfx",
    "pastr",
    "urban",
}

# Map lowercase names to the exact variable names in the file. This
# makes lookup robust if a file contains an unexpected capitalisation.
luh_variable_lookup = {
    variable.lower(): variable
    for variable in land_use_data.data_vars
    if variable.lower() in LUH2_VARS
}

available_luh_vars = set(
    luh_variable_lookup
)

missing_luh_vars = sorted(
    LUH2_VARS - available_luh_vars
)

if missing_luh_vars:
    raise ValueError(
        "Land-use input is missing LUH2 variables: "
        f"{missing_luh_vars}"
    )


# =====================================================================
# 8. LOAD IUCN-LUH2 CROSSWALK
# =====================================================================

conversion_codes = pd.read_csv(
    CONVERSION_TABLE
)

required_crosswalk_columns = {
    "IUCN_hab",
    "LUH",
}

missing_crosswalk_columns = (
    required_crosswalk_columns
    - set(conversion_codes.columns)
)

if missing_crosswalk_columns:
    raise ValueError(
        "Conversion table is missing columns: "
        f"{sorted(missing_crosswalk_columns)}"
    )

conversion_codes["IUCN_hab"] = (
    conversion_codes["IUCN_hab"]
    .astype(str)
    .str.strip()
)


# =====================================================================
# 9. HELPER FUNCTIONS
# =====================================================================


def probability_column_for(
    year,
    scenario,
    gcm,
):
    """Return the source probability column for one run."""

    if year == "2015":
        # The revised analysis uses 2015 land use. The original
        # historical probability column is EWEMBI_1995.
        return "EWEMBI_1995"

    climate_scenario = (
        "rcp26"
        if scenario == "rcp45"
        else scenario
    )

    clean_gcm = GCM_MAP.get(
        gcm,
        gcm,
    )

    return (
        f"{clean_gcm}_"
        f"{climate_scenario}_"
        f"{year}"
    )


def positive_count(data_array):
    """Count finite values greater than zero."""

    values = data_array.values
    return int(
        (
            np.isfinite(values)
            & (values > 0)
        ).sum()
    )


def finite_count(data_array):
    """Count finite values."""

    return int(
        np.isfinite(
            data_array.values
        ).sum()
    )


def validate_output(
    sum_raw,
    sum_luf,
    processed_count,
    endemic_whitelist_count,
    endemic_candidate_file_count,
    missing_column_count,
    failed_species_count,
    label,
):
    """Prevent incomplete or zero-filled endemic outputs from being saved."""

    raw_finite = finite_count(sum_raw)
    luf_finite = finite_count(sum_luf)
    raw_positive = positive_count(sum_raw)
    luf_positive = positive_count(sum_luf)

    raw_maximum = float(
        sum_raw.max(skipna=True).item()
    )
    luf_maximum = float(
        sum_luf.max(skipna=True).item()
    )
    raw_sum = float(
        sum_raw.sum(skipna=True).item()
    )
    luf_sum = float(
        sum_luf.sum(skipna=True).item()
    )

    print("")
    print("Scientific output validation")
    print(
        f"  Endemic species in whitelist: "
        f"{endemic_whitelist_count:,}"
    )
    print(
        f"  Endemic projection files found: "
        f"{endemic_candidate_file_count:,}"
    )
    print(
        f"  Endemic species processed: "
        f"{processed_count:,}"
    )
    print(
        f"  Missing probability column: "
        f"{missing_column_count:,}"
    )
    print(
        f"  Processing failures: "
        f"{failed_species_count:,}"
    )
    print(f"  Raw finite cells: {raw_finite:,}")
    print(f"  Raw positive cells: {raw_positive:,}")
    print(f"  Raw maximum: {raw_maximum:.6f}")
    print(f"  Raw global sum: {raw_sum:.6f}")
    print(f"  LUF finite cells: {luf_finite:,}")
    print(f"  LUF positive cells: {luf_positive:,}")
    print(f"  LUF maximum: {luf_maximum:.6f}")
    print(f"  LUF global sum: {luf_sum:.6f}")

    if endemic_whitelist_count == 0:
        raise RuntimeError(
            f"{label}: no endemic species were listed for this taxon."
        )

    if endemic_candidate_file_count == 0:
        raise RuntimeError(
            f"{label}: no projection files matched the endemic whitelist."
        )

    if processed_count == 0:
        raise RuntimeError(
            f"{label}: no endemic species were processed. "
            "Output will not be written."
        )

    if missing_column_count == endemic_candidate_file_count:
        raise RuntimeError(
            f"{label}: the required source probability column was "
            "absent from every endemic projection file. Output will "
            "not be written."
        )

    if raw_finite == 0:
        raise RuntimeError(
            f"{label}: the climate-driven endemic sum contains no "
            "finite values. Output will not be written."
        )

    if raw_positive == 0 or raw_maximum <= 0 or raw_sum <= 0:
        raise RuntimeError(
            f"{label}: the climate-driven endemic sum contains no "
            "positive values. Check the source probability column and "
            "species projection files. Output will not be written."
        )

    if luf_finite == 0:
        raise RuntimeError(
            f"{label}: the land-use-filtered endemic sum contains no "
            "finite values. Output will not be written."
        )

    if luf_positive == 0 or luf_maximum <= 0 or luf_sum <= 0:
        raise RuntimeError(
            f"{label}: the land-use-filtered endemic sum contains no "
            "positive values. Check land use, habitat files and the "
            "IUCN-LUH2 crosswalk. Output will not be written."
        )


# =====================================================================
# 10. PROCESS MODELS, GCMS, AND TAXONOMIC GROUPS
# =====================================================================

for sdm in args.model:
    for gcm in args.gcm:
        for taxon in args.taxa:

            species_probability_column = probability_column_for(
                year=args.year,
                scenario=args.scenario,
                gcm=gcm,
            )

            time_label = (
                args.year
                if args.year == "2015"
                else f"{args.scenario}_{args.year}"
            )

            output_basename = (
                f"summed_prob_endemics_{taxon}_{sdm}_{gcm}_"
                f"{time_label}_{SUFFIX}"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                f"{output_basename}.nc",
            )

            print("")
            print("=" * 72)
            print(f">> ENDEMIC taxon: {taxon}")
            print(f">> SDM: {sdm}")
            print(f">> GCM label: {gcm}")
            print(
                f">> Probability column: "
                f"{species_probability_column}"
            )
            print(f">> Analysis year: {args.year}")
            print(f">> Analysis type: {args.type}")
            print(f">> Retention: {args.retention}")
            print(f">> Output: {output_path}")
            print("=" * 72)

            habitat_directory = os.path.join(
                HABITAT_ROOT,
                taxon,
            )

            species_projection_directory = os.path.join(
                SPECIES_PROJECTION_ROOT,
                f"{taxon}_{sdm}_results_climate",
            )

            if not os.path.isdir(species_projection_directory):
                raise FileNotFoundError(
                    "Species-projection directory not found: "
                    f"{species_projection_directory}"
                )

            if not os.path.isdir(habitat_directory):
                print(
                    "WARNING: Habitat directory not found. Species "
                    "without habitat information will receive P_LUF = 0: "
                    f"{habitat_directory}"
                )

            # Use the same taxon labels as stored in the match table.
            taxon_rows = endemic_matches.loc[
                endemic_matches["taxon"] == taxon,
                "modeled_name",
            ]

            endemic_keys = {
                name.replace(" ", "_")
                for name in taxon_rows.unique()
            }

            endemic_whitelist_count = len(endemic_keys)

            if endemic_whitelist_count == 0:
                available_taxa = sorted(
                    endemic_matches["taxon"].unique()
                )
                raise RuntimeError(
                    f"No endemic species found for taxon {taxon!r} in "
                    f"{MATCH_FILE_PATH}. Available taxon labels: "
                    f"{available_taxa}"
                )

            all_projection_files = sorted(
                filename
                for filename in os.listdir(
                    species_projection_directory
                )
                if filename.endswith(
                    f"_{sdm}_dispersal.csv.xz"
                )
            )

            candidate_file_count_all_species = len(
                all_projection_files
            )

            if candidate_file_count_all_species == 0:
                raise RuntimeError(
                    f"No projection files found for {taxon} | {sdm} "
                    f"in {species_projection_directory}"
                )

            # Filter to endemic species before reading file headers or data.
            endemic_species_files = []
            projection_keys_found = set()

            for filename in all_projection_files:
                species_key = filename.split(
                    f"_{sdm}"
                )[0]

                if species_key in endemic_keys:
                    endemic_species_files.append(filename)
                    projection_keys_found.add(species_key)

            endemic_candidate_file_count = len(
                endemic_species_files
            )

            endemic_keys_without_projection = sorted(
                endemic_keys - projection_keys_found
            )

            print(
                f">> All candidate projection files: "
                f"{candidate_file_count_all_species:,}"
            )
            print(
                f">> Endemic species in whitelist: "
                f"{endemic_whitelist_count:,}"
            )
            print(
                f">> Endemic projection files found: "
                f"{endemic_candidate_file_count:,}"
            )
            print(
                f">> Whitelisted endemics without projection file: "
                f"{len(endemic_keys_without_projection):,}"
            )

            if endemic_keys_without_projection:
                preview = endemic_keys_without_projection[:10]
                print(
                    ">> First unmatched endemic names: "
                    f"{preview}"
                )

            if endemic_candidate_file_count == 0:
                raise RuntimeError(
                    f"No projection files matched the endemic whitelist "
                    f"for {taxon} | {sdm}. Check taxon labels and "
                    "modeled_name values in the match file."
                )

            sum_climate_dispersal = xr.DataArray(
                np.zeros(
                    (
                        land_use_data.sizes["lat"],
                        land_use_data.sizes["lon"],
                    ),
                    dtype=np.float32,
                ),
                coords={
                    "lat": land_use_data["lat"],
                    "lon": land_use_data["lon"],
                },
                dims=[
                    "lat",
                    "lon",
                ],
            )

            sum_luf = sum_climate_dispersal.copy(
                deep=True
            )

            processed_count = 0
            missing_column_count = 0
            missing_habitat_count = 0
            no_suitable_match_count = 0
            failed_species_count = 0

            first_missing_column_file = None
            first_header_columns = None

            for filename in endemic_species_files:
                species_key = filename.split(
                    f"_{sdm}"
                )[0]

                species_path = os.path.join(
                    species_projection_directory,
                    filename,
                )

                try:
                    header = pd.read_csv(
                        species_path,
                        compression="xz",
                        nrows=0,
                    )

                    if species_probability_column not in header.columns:
                        missing_column_count += 1

                        if first_missing_column_file is None:
                            first_missing_column_file = filename
                            first_header_columns = list(
                                header.columns
                            )

                        continue

                    required_columns = [
                        "x",
                        "y",
                        "dispersal1",
                        species_probability_column,
                    ]

                    species_table = pd.read_csv(
                        species_path,
                        compression="xz",
                        usecols=required_columns,
                        dtype={
                            species_probability_column: np.float32,
                            "dispersal1": np.float32,
                            "x": np.float32,
                            "y": np.float32,
                        },
                    )

                    if species_table.empty:
                        raise ValueError(
                            "Species projection table is empty."
                        )

                    if (
                        float(land_use_data["lon"].min()) < 0
                        and float(species_table["x"].max()) > 180
                    ):
                        species_table["x"] = np.where(
                            species_table["x"] > 180,
                            species_table["x"] - 360,
                            species_table["x"],
                        )

                    species_table["p_clim_disp"] = (
                        species_table[species_probability_column]
                        * species_table["dispersal1"]
                    )

                    species_dataset = (
                        species_table
                        .rename(
                            columns={
                                "x": "lon",
                                "y": "lat",
                            }
                        )[
                            [
                                "lat",
                                "lon",
                                "p_clim_disp",
                            ]
                        ]
                        .set_index(
                            [
                                "lat",
                                "lon",
                            ]
                        )
                        .to_xarray()
                    )

                    p_clim_disp = (
                        species_dataset["p_clim_disp"]
                        .interp(
                            lat=land_use_data["lat"],
                            lon=land_use_data["lon"],
                            method="nearest",
                        )
                        .fillna(0)
                        .clip(min=0, max=1)
                        .astype(np.float32)
                    )

                    habitat_path = os.path.join(
                        habitat_directory,
                        f"{species_key}.csv",
                    )

                    if os.path.exists(habitat_path):
                        habitat_table = pd.read_csv(
                            habitat_path
                        )

                        required_habitat_columns = {
                            "result.suitability",
                            "result.code",
                        }

                        if not required_habitat_columns.issubset(
                            habitat_table.columns
                        ):
                            raise KeyError(
                                "Habitat file lacks required columns "
                                "result.suitability and result.code: "
                                f"{habitat_path}"
                            )

                        suitable_codes = (
                            habitat_table
                            .loc[
                                habitat_table["result.suitability"]
                                == "Suitable",
                                "result.code",
                            ]
                            .astype(str)
                            .str.strip()
                            .unique()
                        )

                        matched_rows = conversion_codes[
                            conversion_codes["IUCN_hab"].isin(
                                suitable_codes
                            )
                        ]

                        suitable_luh_variables = []

                        for crosswalk_value in matched_rows[
                            "LUH"
                        ].dropna():
                            for variable in str(
                                crosswalk_value
                            ).split("."):
                                variable = variable.strip().lower()

                                if variable in available_luh_vars:
                                    suitable_luh_variables.append(
                                        variable
                                    )

                        suitable_luh_variables = sorted(
                            set(suitable_luh_variables)
                        )

                        if suitable_luh_variables:
                            suitable_fraction = sum(
                                land_use_data[
                                    luh_variable_lookup[variable]
                                ]
                                for variable in suitable_luh_variables
                            )

                            suitable_fraction = (
                                suitable_fraction
                                .clip(min=0, max=1)
                                .fillna(0)
                                .astype(np.float32)
                            )

                            p_luf = (
                                p_clim_disp
                                * suitable_fraction
                            ).clip(
                                min=0,
                                max=1,
                            ).astype(np.float32)

                        else:
                            p_luf = xr.zeros_like(
                                p_clim_disp,
                                dtype=np.float32,
                            )
                            no_suitable_match_count += 1

                    else:
                        p_luf = xr.zeros_like(
                            p_clim_disp,
                            dtype=np.float32,
                        )
                        missing_habitat_count += 1

                    sum_climate_dispersal = (
                        sum_climate_dispersal
                        + p_clim_disp
                    ).astype(np.float32)

                    sum_luf = (
                        sum_luf
                        + p_luf
                    ).astype(np.float32)

                    processed_count += 1

                    if processed_count % 100 == 0:
                        print(
                            f"   ... [{taxon}] "
                            f"{processed_count:,} endemic species "
                            "processed"
                        )

                except Exception as error:
                    failed_species_count += 1
                    print(
                        f"WARNING: Error processing {filename}: "
                        f"{type(error).__name__}: {error}"
                    )

            print("")
            print("Processing counters")
            print(
                f"  All candidate projection files: "
                f"{candidate_file_count_all_species:,}"
            )
            print(
                f"  Endemic species in whitelist: "
                f"{endemic_whitelist_count:,}"
            )
            print(
                f"  Endemic projection files found: "
                f"{endemic_candidate_file_count:,}"
            )
            print(
                f"  Processed endemic species: "
                f"{processed_count:,}"
            )
            print(
                f"  Missing probability column: "
                f"{missing_column_count:,}"
            )
            print(
                f"  Missing habitat files: "
                f"{missing_habitat_count:,}"
            )
            print(
                f"  No suitable LUH2 match: "
                f"{no_suitable_match_count:,}"
            )
            print(
                f"  Processing failures: "
                f"{failed_species_count:,}"
            )
            print(
                f"  Whitelisted endemics without projection file: "
                f"{len(endemic_keys_without_projection):,}"
            )

            if first_missing_column_file is not None:
                print(
                    f"  First file missing "
                    f"{species_probability_column!r}: "
                    f"{first_missing_column_file}"
                )
                print(
                    "  Available columns in that file: "
                    f"{first_header_columns}"
                )

            validate_output(
                sum_raw=sum_climate_dispersal,
                sum_luf=sum_luf,
                processed_count=processed_count,
                endemic_whitelist_count=endemic_whitelist_count,
                endemic_candidate_file_count=(
                    endemic_candidate_file_count
                ),
                missing_column_count=missing_column_count,
                failed_species_count=failed_species_count,
                label=(
                    f"endemic {taxon} | {sdm} | "
                    f"{gcm} | {args.year}"
                ),
            )

            final_dataset = xr.Dataset(
                {
                    "prob_clim_disp": sum_climate_dispersal,
                    "prob_clim_disp_luf": sum_luf,
                }
            )

            final_dataset["prob_clim_disp"].attrs.update(
                {
                    "long_name": (
                        "Summed climate-driven and limited-dispersal "
                        "occurrence probability for endemic species"
                    ),
                    "interpretation": (
                        "Sum of endemic species-level occurrence "
                        "probabilities before land-use filtering"
                    ),
                }
            )

            final_dataset["prob_clim_disp_luf"].attrs.update(
                {
                    "long_name": (
                        "Summed land-use-filtered occurrence "
                        "probability for endemic species"
                    ),
                    "interpretation": (
                        "Sum of endemic species-level climate-driven "
                        "and limited-dispersal occurrence probabilities "
                        "weighted by suitable land-use fraction"
                    ),
                }
            )

            final_dataset.attrs.update(
                {
                    "species_subset": "endemic_species",
                    "endemic_match_file": MATCH_FILE_PATH,
                    "taxonomic_group": taxon,
                    "species_distribution_model": sdm,
                    "general_circulation_model": gcm,
                    "climate_scenario": args.scenario,
                    "land_use_scenario": ssp,
                    "analysis_year": args.year,
                    "analysis_type": args.type,
                    "source_probability_column": (
                        species_probability_column
                    ),
                    "historical_sdm_reference_year": (
                        "1995"
                        if args.year == "2015"
                        else "not_applicable"
                    ),
                    "land_use_baseline_year": (
                        "2015"
                        if args.year == "2015"
                        else "not_applicable"
                    ),
                    "land_use_retention_assumption": (
                        args.retention
                        if args.retention is not None
                        else "not_applicable"
                    ),
                    "land_use_input": LU_FILE,
                    "all_candidate_species_files": (
                        candidate_file_count_all_species
                    ),
                    "endemic_species_in_whitelist": (
                        endemic_whitelist_count
                    ),
                    "endemic_projection_files_found": (
                        endemic_candidate_file_count
                    ),
                    "endemic_species_processed": processed_count,
                    "endemic_species_without_projection_file": (
                        len(endemic_keys_without_projection)
                    ),
                    "species_missing_probability_column": (
                        missing_column_count
                    ),
                    "species_missing_habitat_file": (
                        missing_habitat_count
                    ),
                    "species_without_suitable_luh_match": (
                        no_suitable_match_count
                    ),
                    "species_processing_failures": (
                        failed_species_count
                    ),
                    "luf_equation": (
                        "P_LUF = "
                        "(P_clim * limited_dispersal_mask) * H_suit"
                    ),
                    "habitat_rule": (
                        "Only IUCN habitats classified as Suitable "
                        "were included. Marginal and unsuitable "
                        "habitats were excluded."
                    ),
                    "missing_habitat_rule": (
                        "P_LUF was set to zero when habitat "
                        "information or suitable LUH2 matches "
                        "were absent."
                    ),
                }
            )

            encoding = {
                variable: {
                    "dtype": "float32",
                    "zlib": True,
                    "complevel": 4,
                }
                for variable in [
                    "prob_clim_disp",
                    "prob_clim_disp_luf",
                ]
            }

            final_dataset.to_netcdf(
                output_path,
                encoding=encoding,
            )

            with xr.open_dataset(
                output_path,
                decode_times=False,
            ) as saved_dataset:
                saved_raw_positive = positive_count(
                    saved_dataset["prob_clim_disp"]
                )
                saved_luf_positive = positive_count(
                    saved_dataset["prob_clim_disp_luf"]
                )
                saved_raw_maximum = float(
                    saved_dataset["prob_clim_disp"]
                    .max(skipna=True)
                    .item()
                )
                saved_luf_maximum = float(
                    saved_dataset["prob_clim_disp_luf"]
                    .max(skipna=True)
                    .item()
                )

            if (
                saved_raw_positive == 0
                or saved_luf_positive == 0
                or saved_raw_maximum <= 0
                or saved_luf_maximum <= 0
            ):
                os.remove(output_path)
                raise RuntimeError(
                    "Saved-file validation failed. The invalid output "
                    f"was removed: {output_path}"
                )

            print("")
            print(f">> Saved and validated: {output_path}")
            print(
                f">> Endemic species processed: "
                f"{processed_count:,}"
            )
            print(
                f">> Saved raw maximum: "
                f"{saved_raw_maximum:.6f}"
            )
            print(
                f">> Saved LUF maximum: "
                f"{saved_luf_maximum:.6f}"
            )


print("")
print(
    ">> All requested endemic-species Land-Use Filter "
    "calculations completed."
)