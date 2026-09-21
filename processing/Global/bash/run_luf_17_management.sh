#!/usr/bin/env bash

set -u
set -o pipefail


# =====================================================================
# 1. CONFIGURATION
# =====================================================================

TAXA=(
    "Amphibians"
    "Mammals"
    "Bird"
)

GCMS=(
    "GFDL-ESM2M"
    "IPSL-CM5A-LR"
    "MIROC5"
    "HadGEM2-ES"
)

# Scenario interpretation:
#
# rcp26:
#   SSP1-2.6 land use + RCP2.6 climate
#
# rcp45:
#   SSP2-4.5 land use + RCP2.6 climate
#
# rcp60:
#   SSP4-6.0 land use + RCP6.0 climate

SCENARIOS=(
    "rcp26"
    "rcp45"
    "rcp60"
)

PA_TYPES=(
    "pa17"
)

RETENTION_ASSUMPTIONS=(
    "complete"
    "funding_constrained"
)

PYTHON_SCRIPT="/capacity/occr_davin/mguzman/chari_P2_review/scripts_git/processing/Global/2_luf_sumprob_PA_funding.py"

OUTPUT_ROOT="/capacity/occr_davin/mguzman/chari_P2_review/Global/PA_scenarios/Sumprob_revised"

LOG_DIR="/capacity/occr_davin/mguzman/chari_P2_review/scripts/logs"

RUN_DATE="$(date +%F_%H-%M-%S)"

LOG_FILE="${LOG_DIR}/run_luf_revised_${RUN_DATE}.log"

PYTHON_EXECUTABLE="python3"

HISTORICAL_GCM="EWEMBI"
HISTORICAL_SCENARIO="rcp26"
HISTORICAL_YEAR="2015"

FUTURE_YEAR="2080"


# =====================================================================
# 2. RUN OPTIONS
# =====================================================================

# When false:
#   scientifically valid existing outputs are skipped.
#
# When true:
#   all requested outputs are regenerated.

OVERWRITE_EXISTING=false


# =====================================================================
# 3. MODEL AVAILABILITY BY TAXON
# =====================================================================

models_for_taxon() {
    local taxon="$1"

    case "$taxon" in

        "Amphibians"|"Mammals")
            printf "%s\n" \
                "GAM" \
                "GBM"
            ;;

        "Bird")
            printf "%s\n" \
                "GAM"
            ;;

        *)
            printf "ERROR: Unknown taxon: %s\n" \
                "$taxon" >&2

            return 1
            ;;

    esac
}


# =====================================================================
# 4. INITIAL VALIDATION
# =====================================================================

mkdir -p \
    "$LOG_DIR"

touch \
    "$LOG_FILE"


if [[ ! -f "$PYTHON_SCRIPT" ]]; then

    echo "ERROR: Python script not found:" >&2
    echo "$PYTHON_SCRIPT" >&2

    exit 1

fi


if ! command -v \
    "$PYTHON_EXECUTABLE" \
    >/dev/null 2>&1
then

    echo "ERROR: Python executable not found:" >&2
    echo "$PYTHON_EXECUTABLE" >&2

    exit 1

fi


if ! "$PYTHON_EXECUTABLE" -m py_compile \
    "$PYTHON_SCRIPT"
then

    echo "ERROR: Python script failed syntax validation:" >&2
    echo "$PYTHON_SCRIPT" >&2

    exit 1

fi


# =====================================================================
# 5. LOGGING
# =====================================================================

log_msg() {
    printf "%s\n" "$1" \
        | tee -a "$LOG_FILE"
}


log_command() {
    local description="$1"

    shift

    log_msg \
        "Launching: ${description}"

    {
        printf "Command:"
        printf " %q" "$@"
        printf "\n"
    } >> "$LOG_FILE"
}


# =====================================================================
# 6. COUNTERS AND JOB TRACKING
# =====================================================================

LAUNCHED_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0
COMPLETED_COUNT=0
REMOVED_INVALID_COUNT=0

CURRENT_PIDS=()
CURRENT_DESCRIPTIONS=()
CURRENT_OUTPUTS=()


reset_job_batch() {
    CURRENT_PIDS=()
    CURRENT_DESCRIPTIONS=()
    CURRENT_OUTPUTS=()
}


# =====================================================================
# 7. EXPECTED OUTPUT PATH
# =====================================================================

get_expected_output_path() {
    local model="$1"
    local taxon="$2"
    local gcm="$3"
    local scenario="$4"
    local year="$5"
    local analysis_type="$6"
    local retention="${7:-}"

    local output_dir
    local suffix
    local time_label
    local filename

    if [[ "$year" == "2015" ]]; then

        output_dir="${OUTPUT_ROOT}/WDPA/Historical"
        suffix="baseline_2015"
        time_label="2015"

    elif [[ "$analysis_type" == "climate_only" ]]; then

        output_dir="${OUTPUT_ROOT}/WDPA/Climate_Only"
        suffix="climate_only"
        time_label="${scenario}_${year}"

    elif [[ "$analysis_type" == "pa17" ]]; then

        output_dir="${OUTPUT_ROOT}/PA17/${retention}"
        suffix="pa17_${retention}"
        time_label="${scenario}_${year}"

    elif [[ "$analysis_type" == "pa30_bio" ]]; then

        output_dir="${OUTPUT_ROOT}/PA30_Bio/${retention}"
        suffix="pa30_bio_${retention}"
        time_label="${scenario}_${year}"

    elif [[ "$analysis_type" == "pa30_bcw" ]]; then

        output_dir="${OUTPUT_ROOT}/PA30_BCW/${retention}"
        suffix="pa30_bcw_${retention}"
        time_label="${scenario}_${year}"

    else

        printf "ERROR: Unsupported analysis type: %s\n" \
            "$analysis_type" >&2

        return 1

    fi

    filename="summed_prob_${taxon}_${model}_${gcm}_${time_label}_${suffix}.nc"

    printf "%s\n" "${output_dir}/${filename}"
}


# =====================================================================
# 8. SCIENTIFIC OUTPUT VALIDATION
# =====================================================================

output_is_scientifically_valid() {
    local output_path="$1"


    if [[ ! -s "$output_path" ]]; then
        return 1
    fi


    "$PYTHON_EXECUTABLE" \
        - "$output_path" <<'PY'

import sys

import numpy as np
import xarray as xr


path = sys.argv[1]

required_variables = [
    "prob_clim_disp",
    "prob_clim_disp_luf",
]


try:

    with xr.open_dataset(
        path,
        decode_times=False,
    ) as dataset:

        for variable in required_variables:

            if variable not in dataset.data_vars:
                raise ValueError(
                    f"missing variable: {variable}"
                )

            values = dataset[variable].values

            finite_mask = np.isfinite(
                values
            )

            if not finite_mask.any():
                raise ValueError(
                    f"no finite values: {variable}"
                )

            positive_mask = (
                finite_mask
                & (values > 0)
            )

            if not positive_mask.any():
                raise ValueError(
                    f"no positive values: {variable}"
                )


except Exception as error:

    print(
        f"INVALID: {path}: {error}",
        file=sys.stderr,
    )

    sys.exit(1)


sys.exit(0)

PY
}


# =====================================================================
# 9. CHECK WHETHER A JOB IS NEEDED
# =====================================================================

output_is_complete() {
    local output_path="$1"


    if [[ "$OVERWRITE_EXISTING" == "true" ]]; then
        return 1
    fi


    output_is_scientifically_valid \
        "$output_path"
}


# =====================================================================
# 10. LAUNCH ONE LUF JOB
# =====================================================================

run_luf_job() {
    local model="$1"
    local taxon="$2"
    local gcm="$3"
    local scenario="$4"
    local year="$5"
    local analysis_type="$6"
    local retention="${7:-}"

    local output_path
    local description
    local pid
    local -a command_arguments

    output_path="$(
        get_expected_output_path \
            "$model" \
            "$taxon" \
            "$gcm" \
            "$scenario" \
            "$year" \
            "$analysis_type" \
            "$retention"
    )" || return 1

    description="${taxon} | ${model} | ${gcm} | ${scenario} | ${year} | ${analysis_type}"

    if [[ -n "$retention" ]]; then
        description="${description} | ${retention}"
    fi

    if output_is_complete \
        "$output_path"
    then

        log_msg \
            "Skipping validated existing output: ${description}"

        log_msg \
            "  Output: ${output_path}"

        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))

        return 0

    fi


    # If an output exists but does not pass validation, remove it before
    # launching the replacement calculation.

    if [[ -e "$output_path" ]]; then

        log_msg \
            "Removing incomplete or scientifically invalid output:"

        log_msg \
            "  ${output_path}"

        rm -f \
            "$output_path"

        REMOVED_INVALID_COUNT=$((REMOVED_INVALID_COUNT + 1))

    fi


    mkdir -p \
        "$(dirname "$output_path")"


    command_arguments=(
        "$PYTHON_EXECUTABLE"
        "$PYTHON_SCRIPT"
        -m "$model"
        -a "$taxon"
        -g "$gcm"
        -s "$scenario"
        -y "$year"
        --type "$analysis_type"
    )


    if [[ -n "$retention" ]]; then

        command_arguments+=(
            --retention
            "$retention"
        )

    fi


    log_command \
        "$description" \
        "${command_arguments[@]}"


    "${command_arguments[@]}" \
        >> "$LOG_FILE" \
        2>&1 &


    pid=$!


    CURRENT_PIDS+=(
        "$pid"
    )

    CURRENT_DESCRIPTIONS+=(
        "$description"
    )

    CURRENT_OUTPUTS+=(
        "$output_path"
    )


    LAUNCHED_COUNT=$((LAUNCHED_COUNT + 1))
}


# =====================================================================
# 11. WAIT FOR CURRENT JOB BATCH
# =====================================================================

wait_for_current_batch() {
    local index
    local pid
    local description
    local output_path

    local batch_failed=0


    for index in "${!CURRENT_PIDS[@]}"; do

        pid="${CURRENT_PIDS[$index]}"

        description="${CURRENT_DESCRIPTIONS[$index]}"

        output_path="${CURRENT_OUTPUTS[$index]}"


        if wait "$pid"; then

            if output_is_scientifically_valid \
                "$output_path"
            then

                log_msg \
                    "Completed and validated: ${description}"

                log_msg \
                    "  Output: ${output_path}"

                COMPLETED_COUNT=$((COMPLETED_COUNT + 1))


            else

                log_msg \
                    "ERROR: Job exited successfully, but output failed validation:"

                log_msg \
                    "  Job: ${description}"

                log_msg \
                    "  Output: ${output_path}"


                rm -f \
                    "$output_path"


                FAILED_COUNT=$((FAILED_COUNT + 1))

                batch_failed=$((batch_failed + 1))

            fi


        else

            log_msg \
                "ERROR: Job failed:"

            log_msg \
                "  Job: ${description}"

            log_msg \
                "  Output: ${output_path}"


            rm -f \
                "$output_path"


            FAILED_COUNT=$((FAILED_COUNT + 1))

            batch_failed=$((batch_failed + 1))

        fi

    done


    reset_job_batch


    if (( batch_failed > 0 )); then
        return 1
    fi


    return 0
}


# =====================================================================
# 12. LAUNCH ALL VALID TAXON-MODEL COMBINATIONS
# =====================================================================

launch_taxon_model_batch() {
    local gcm="$1"
    local scenario="$2"
    local year="$3"
    local analysis_type="$4"
    local retention="${5:-}"

    local taxon
    local model


    reset_job_batch


    for taxon in "${TAXA[@]}"; do

        while IFS= read -r model; do

            run_luf_job \
                "$model" \
                "$taxon" \
                "$gcm" \
                "$scenario" \
                "$year" \
                "$analysis_type" \
                "$retention"

        done < <(
            models_for_taxon \
                "$taxon"
        )

    done
}


# =====================================================================
# 13. STARTUP SUMMARY
# =====================================================================

log_msg \
    "======================================================="

log_msg \
    "STARTING REVISED LAND-USE FILTER WORKFLOW"

log_msg \
    "Start: $(date)"

log_msg \
    "Python script: ${PYTHON_SCRIPT}"

log_msg \
    "Output root: ${OUTPUT_ROOT}"

log_msg \
    "Log file: ${LOG_FILE}"

log_msg \
    "Overwrite existing files: ${OVERWRITE_EXISTING}"

log_msg \
    "======================================================="


log_msg ""

log_msg \
    "Model availability:"

log_msg \
    "  Amphibians: GAM and GBM"

log_msg \
    "  Mammals: GAM and GBM"

log_msg \
    "  Bird: GAM only"


log_msg ""

log_msg \
    "Historical definition:"

log_msg \
    "  Analysis year: 2015"

log_msg \
    "  Land-use baseline: harmonized 2015"

log_msg \
    "  GCM output label: EWEMBI"

log_msg \
    "  Source SDM probability column: EWEMBI_1995"


# =====================================================================
# 14. PHASE 1: HARMONIZED 2015 BASELINE
# =====================================================================

log_msg ""

log_msg \
    "======================================================="

log_msg \
    "PHASE 1: HARMONIZED 2015 BASELINE"

log_msg \
    "======================================================="


launch_taxon_model_batch \
    "$HISTORICAL_GCM" \
    "$HISTORICAL_SCENARIO" \
    "$HISTORICAL_YEAR" \
    "baseline"


if ! wait_for_current_batch; then

    log_msg \
        "ERROR: One or more historical baseline jobs failed."

    log_msg \
        "Inspect the detailed log: ${LOG_FILE}"

    exit 1

fi


log_msg \
    "PHASE 1 COMPLETED"


# =====================================================================
# 15. PHASE 2: CLIMATE-ONLY PROJECTIONS
# =====================================================================

log_msg ""

log_msg \
    "======================================================="

log_msg \
    "PHASE 2: CLIMATE-ONLY PROJECTIONS FOR 2080"

log_msg \
    "======================================================="


for scenario in "${SCENARIOS[@]}"; do

    for gcm in "${GCMS[@]}"; do

        log_msg ""

        log_msg \
            "Climate-only: ${scenario} | ${gcm}"


        launch_taxon_model_batch \
            "$gcm" \
            "$scenario" \
            "$FUTURE_YEAR" \
            "climate_only"


        if ! wait_for_current_batch; then

            log_msg \
                "ERROR: Climate-only batch failed: ${scenario} | ${gcm}"

            log_msg \
                "Inspect the detailed log: ${LOG_FILE}"

            exit 1

        fi

    done

done


log_msg \
    "PHASE 2 COMPLETED"


# =====================================================================
# 16. PHASE 3: PA-MEDIATED LAND-USE CHANGE
# =====================================================================

log_msg ""

log_msg \
    "======================================================="

log_msg \
    "PHASE 3: PA-MEDIATED LAND-USE CHANGE FOR 2080"

log_msg \
    "======================================================="


for pa_type in "${PA_TYPES[@]}"; do

    for retention in "${RETENTION_ASSUMPTIONS[@]}"; do

        for scenario in "${SCENARIOS[@]}"; do

            for gcm in "${GCMS[@]}"; do

                log_msg ""

                log_msg \
                    "PA batch: ${pa_type} | ${retention} | ${scenario} | ${gcm}"


                launch_taxon_model_batch \
                    "$gcm" \
                    "$scenario" \
                    "$FUTURE_YEAR" \
                    "$pa_type" \
                    "$retention"


                if ! wait_for_current_batch; then

                    log_msg \
                        "ERROR: PA batch failed: ${pa_type} | ${retention} | ${scenario} | ${gcm}"

                    log_msg \
                        "Inspect the detailed log: ${LOG_FILE}"

                    exit 1

                fi

            done

        done

    done

done


log_msg \
    "PHASE 3 COMPLETED"


# =====================================================================
# 17. FINAL STATUS
# =====================================================================

log_msg ""

log_msg \
    "======================================================="

log_msg \
    "WORKFLOW SUMMARY"

log_msg \
    "Jobs launched: ${LAUNCHED_COUNT}"

log_msg \
    "Jobs completed and validated: ${COMPLETED_COUNT}"

log_msg \
    "Validated outputs skipped: ${SKIPPED_COUNT}"

log_msg \
    "Invalid outputs removed: ${REMOVED_INVALID_COUNT}"

log_msg \
    "Failed jobs: ${FAILED_COUNT}"

log_msg \
    "End: $(date)"

log_msg \
    "Log file: ${LOG_FILE}"

log_msg \
    "======================================================="


if (( FAILED_COUNT > 0 )); then
    exit 1
fi


log_msg \
    "ALL REQUESTED LUF OUTPUTS ARE PRESENT AND VALID."