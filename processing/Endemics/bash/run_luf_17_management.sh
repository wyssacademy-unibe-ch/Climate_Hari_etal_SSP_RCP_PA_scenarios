#!/usr/bin/env bash

# Runs endemic-species analyses for:
#   1. Harmonized 2015 historical baseline
#   2. Climate-only projections for 2080
#   3. Existing 17%-PA projections for 2080
#
# The PA17 runs use RETENTION below. Historical and climate-only runs do not
# receive --retention because the Python script rejects it for those analyses.
#
# Run with:
# nohup bash \
# /capacity/occr_davin/mguzman/chari_P2_review/scripts_git/processing/Endemics/bash/run_luf_historical_climate_pa17.sh \
# > /capacity/occr_davin/mguzman/chari_P2_review/scripts/logs/nohup_luf_endemics_all.log \
# 2>&1 &

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
# rcp26 = SSP1-2.6 land use + RCP2.6 climate
# rcp45 = SSP2-4.5 land use + RCP2.6 climate
# rcp60 = SSP4-6.0 land use + RCP6.0 climate
SCENARIOS=(
    "rcp26"
    "rcp45"
    "rcp60"
)

HISTORICAL_GCM="EWEMBI"
HISTORICAL_SCENARIO="rcp26"
HISTORICAL_YEAR="2015"
FUTURE_YEAR="2080"

PA_TYPE="pa17"

# Use either:
#   complete
#   funding_constrained
RETENTION="complete"

# Maximum number of Python processes running simultaneously.
MAX_PARALLEL_JOBS=14

# Phase controls.
RUN_HISTORICAL=true
RUN_CLIMATE_ONLY=true
RUN_PA17=true

PYTHON_SCRIPT="/capacity/occr_davin/mguzman/chari_P2_review/scripts_git/processing/Endemics/2_luf_sumprob_PA_management.py"
OUTPUT_ROOT="/capacity/occr_davin/mguzman/chari_P2_review/Endemics/PA_scenarios/Sumprob_revised"
MATCH_FILE="/capacity/occr_davin/mguzman/chari_P2_review/Endemics/data/modeled_endemics_matches.csv"

LOG_DIR="/capacity/occr_davin/mguzman/chari_P2_review/scripts/logs"
RUN_DATE="$(date +%F_%H-%M-%S)"
LOG_FILE="${LOG_DIR}/run_luf_endemics_historical_climate_pa17_${RUN_DATE}.log"

PYTHON_EXECUTABLE="python3"

# false: skip an existing output only when it passes validation.
# true: regenerate every requested output.
OVERWRITE_EXISTING=false

# =====================================================================
# 2. MODEL AVAILABILITY BY TAXON
# =====================================================================

models_for_taxon() {
    local taxon="$1"

    case "$taxon" in
        "Amphibians"|"Mammals")
            printf "%s\n" "GAM" "GBM"
            ;;
        "Bird")
            printf "%s\n" "GAM"
            ;;
        *)
            printf "ERROR: Unknown taxon: %s\n" "$taxon" >&2
            return 1
            ;;
    esac
}

# =====================================================================
# 3. INITIAL VALIDATION
# =====================================================================

mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "ERROR: Python script not found: $PYTHON_SCRIPT" >&2
    exit 1
fi

if ! command -v "$PYTHON_EXECUTABLE" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: $PYTHON_EXECUTABLE" >&2
    exit 1
fi

if ! "$PYTHON_EXECUTABLE" -m py_compile "$PYTHON_SCRIPT"; then
    echo "ERROR: Python script failed syntax validation: $PYTHON_SCRIPT" >&2
    exit 1
fi

if ! [[ "$MAX_PARALLEL_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_PARALLEL_JOBS must be a positive integer." >&2
    exit 1
fi

if [[ "$RETENTION" != "complete" && "$RETENTION" != "funding_constrained" ]]; then
    echo "ERROR: RETENTION must be complete or funding_constrained." >&2
    exit 1
fi

for phase_value in "$RUN_HISTORICAL" "$RUN_CLIMATE_ONLY" "$RUN_PA17"; do
    if [[ "$phase_value" != "true" && "$phase_value" != "false" ]]; then
        echo "ERROR: Phase controls must be true or false." >&2
        exit 1
    fi
done

if [[ ! -f "$MATCH_FILE" ]]; then
    echo "ERROR: Endemic-species match file not found: $MATCH_FILE" >&2
    exit 1
fi

if ! "$PYTHON_EXECUTABLE" - "$MATCH_FILE" "${TAXA[@]}" <<'PY_CHECK'
import sys
import pandas as pd

path = sys.argv[1]
requested = sys.argv[2:]
data = pd.read_csv(path)

required = {"taxon", "modeled_name"}
missing = required.difference(data.columns)
if missing:
    raise SystemExit(
        f"ERROR: Match file is missing columns: {sorted(missing)}"
    )

available = set(
    data["taxon"]
    .dropna()
    .astype(str)
    .str.strip()
)

missing_taxa = [taxon for taxon in requested if taxon not in available]
if missing_taxa:
    raise SystemExit(
        "ERROR: Taxon labels absent from match file: "
        f"{missing_taxa}. Available labels: {sorted(available)}"
    )
PY_CHECK
then
    exit 1
fi

# =====================================================================
# 4. LOGGING
# =====================================================================

log_msg() {
    printf "%s\n" "$1" | tee -a "$LOG_FILE"
}

log_command() {
    local description="$1"
    shift

    log_msg "Launching: ${description}"

    {
        printf "Command:"
        printf " %q" "$@"
        printf "\n"
    } >> "$LOG_FILE"
}

# =====================================================================
# 5. COUNTERS AND ACTIVE BATCH
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
# 6. EXPECTED OUTPUT PATH
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

    if [[ "$year" == "2015" && "$analysis_type" == "baseline" ]]; then
        output_dir="${OUTPUT_ROOT}/WDPA/Historical"
        suffix="baseline_2015"
        time_label="2015"

    elif [[ "$year" == "2080" && "$analysis_type" == "climate_only" ]]; then
        output_dir="${OUTPUT_ROOT}/WDPA/Climate_Only"
        suffix="climate_only"
        time_label="${scenario}_${year}"

    elif [[ "$year" == "2080" && "$analysis_type" == "pa17" ]]; then
        if [[ -z "$retention" ]]; then
            printf "ERROR: Retention is required for pa17.\n" >&2
            return 1
        fi
        output_dir="${OUTPUT_ROOT}/PA17/${retention}"
        suffix="pa17_${retention}"
        time_label="${scenario}_${year}"

    else
        printf "ERROR: Unsupported combination: year=%s type=%s\n" \
            "$year" "$analysis_type" >&2
        return 1
    fi

    filename="summed_prob_endemics_${taxon}_${model}_${gcm}_${time_label}_${suffix}.nc"
    printf "%s\n" "${output_dir}/${filename}"
}

# =====================================================================
# 7. SCIENTIFIC OUTPUT VALIDATION
# =====================================================================

output_is_scientifically_valid() {
    local output_path="$1"

    if [[ ! -s "$output_path" ]]; then
        return 1
    fi

    "$PYTHON_EXECUTABLE" - "$output_path" <<'PY_VALIDATE'
import sys
import numpy as np
import xarray as xr

path = sys.argv[1]
required_variables = ["prob_clim_disp", "prob_clim_disp_luf"]

try:
    with xr.open_dataset(path, decode_times=False) as dataset:
        for variable in required_variables:
            if variable not in dataset.data_vars:
                raise ValueError(f"missing variable: {variable}")

            values = dataset[variable].values
            finite = np.isfinite(values)

            if not finite.any():
                raise ValueError(f"no finite values: {variable}")

            if not (finite & (values > 0)).any():
                raise ValueError(f"no positive values: {variable}")

        subset = dataset.attrs.get("species_subset")
        if subset != "endemic_species":
            raise ValueError(
                f"unexpected species_subset attribute: {subset!r}"
            )

except Exception as error:
    print(f"INVALID: {path}: {error}", file=sys.stderr)
    sys.exit(1)

sys.exit(0)
PY_VALIDATE
}

output_is_complete() {
    local output_path="$1"

    if [[ "$OVERWRITE_EXISTING" == "true" ]]; then
        return 1
    fi

    output_is_scientifically_valid "$output_path"
}

# =====================================================================
# 8. WAIT FOR CURRENT BATCH
# =====================================================================

wait_for_current_batch() {
    local index
    local pid
    local description
    local output_path
    local batch_failed=0

    if (( ${#CURRENT_PIDS[@]} == 0 )); then
        return 0
    fi

    log_msg "Waiting for batch of ${#CURRENT_PIDS[@]} job(s)..."

    for index in "${!CURRENT_PIDS[@]}"; do
        pid="${CURRENT_PIDS[$index]}"
        description="${CURRENT_DESCRIPTIONS[$index]}"
        output_path="${CURRENT_OUTPUTS[$index]}"

        if wait "$pid"; then
            if output_is_scientifically_valid "$output_path"; then
                log_msg "Completed and validated: ${description}"
                log_msg "  Output: ${output_path}"
                COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
            else
                log_msg "ERROR: Job exited successfully but output failed validation:"
                log_msg "  Job: ${description}"
                log_msg "  Output: ${output_path}"
                rm -f "$output_path"
                FAILED_COUNT=$((FAILED_COUNT + 1))
                batch_failed=$((batch_failed + 1))
            fi
        else
            log_msg "ERROR: Job failed:"
            log_msg "  Job: ${description}"
            log_msg "  Output: ${output_path}"
            rm -f "$output_path"
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
# 9. LAUNCH ONE ENDEMIC LUF JOB
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

    if output_is_complete "$output_path"; then
        log_msg "Skipping validated existing output: ${description}"
        log_msg "  Output: ${output_path}"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        return 0
    fi

    if [[ -e "$output_path" ]]; then
        log_msg "Removing incomplete or invalid output:"
        log_msg "  ${output_path}"
        rm -f "$output_path"
        REMOVED_INVALID_COUNT=$((REMOVED_INVALID_COUNT + 1))
    fi

    mkdir -p "$(dirname "$output_path")"

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

    # Retention must be passed only to PA scenarios.
    if [[ -n "$retention" ]]; then
        command_arguments+=(
            --retention "$retention"
        )
    fi

    log_command "$description" "${command_arguments[@]}"

    "${command_arguments[@]}" >> "$LOG_FILE" 2>&1 &
    pid=$!

    CURRENT_PIDS+=("$pid")
    CURRENT_DESCRIPTIONS+=("$description")
    CURRENT_OUTPUTS+=("$output_path")
    LAUNCHED_COUNT=$((LAUNCHED_COUNT + 1))

    if (( ${#CURRENT_PIDS[@]} >= MAX_PARALLEL_JOBS )); then
        if ! wait_for_current_batch; then
            log_msg "ERROR: One or more jobs in the parallel batch failed."
            log_msg "Inspect the detailed log: ${LOG_FILE}"
            exit 1
        fi
    fi
}

# =====================================================================
# 10. LAUNCH ALL VALID TAXON-MODEL COMBINATIONS FOR ONE ANALYSIS
# =====================================================================

queue_analysis_jobs() {
    local gcm="$1"
    local scenario="$2"
    local year="$3"
    local analysis_type="$4"
    local retention="${5:-}"
    local taxon
    local model

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
        done < <(models_for_taxon "$taxon")
    done
}

# =====================================================================
# 11. STARTUP SUMMARY
# =====================================================================

log_msg "======================================================="
log_msg "STARTING ENDEMIC HISTORICAL, CLIMATE-ONLY AND PA17 WORKFLOW"
log_msg "Start: $(date)"
log_msg "Python script: ${PYTHON_SCRIPT}"
log_msg "Output root: ${OUTPUT_ROOT}"
log_msg "Log file: ${LOG_FILE}"
log_msg "PA17 retention assumption: ${RETENTION}"
log_msg "Maximum parallel jobs: ${MAX_PARALLEL_JOBS}"
log_msg "Overwrite existing files: ${OVERWRITE_EXISTING}"
log_msg "Run historical: ${RUN_HISTORICAL}"
log_msg "Run climate only: ${RUN_CLIMATE_ONLY}"
log_msg "Run PA17: ${RUN_PA17}"
log_msg "======================================================="

reset_job_batch

# =====================================================================
# 12. PHASE 1: HARMONIZED 2015 BASELINE
# =====================================================================

if [[ "$RUN_HISTORICAL" == "true" ]]; then
    log_msg ""
    log_msg "QUEUEING HISTORICAL 2015 BASELINE"

    queue_analysis_jobs \
        "$HISTORICAL_GCM" \
        "$HISTORICAL_SCENARIO" \
        "$HISTORICAL_YEAR" \
        "baseline"
else
    log_msg ""
    log_msg "HISTORICAL 2015 BASELINE SKIPPED"
fi

# =====================================================================
# 13. PHASE 2: CLIMATE-ONLY PROJECTIONS
# =====================================================================

if [[ "$RUN_CLIMATE_ONLY" == "true" ]]; then
    log_msg ""
    log_msg "QUEUEING CLIMATE-ONLY PROJECTIONS FOR 2080"

    for scenario in "${SCENARIOS[@]}"; do
        for gcm in "${GCMS[@]}"; do
            queue_analysis_jobs \
                "$gcm" \
                "$scenario" \
                "$FUTURE_YEAR" \
                "climate_only"
        done
    done
else
    log_msg ""
    log_msg "CLIMATE-ONLY PROJECTIONS SKIPPED"
fi

# =====================================================================
# 14. PHASE 3: EXISTING 17%-PA PROJECTIONS
# =====================================================================

if [[ "$RUN_PA17" == "true" ]]; then
    log_msg ""
    log_msg "QUEUEING PA17 PROJECTIONS FOR 2080"

    for scenario in "${SCENARIOS[@]}"; do
        for gcm in "${GCMS[@]}"; do
            queue_analysis_jobs \
                "$gcm" \
                "$scenario" \
                "$FUTURE_YEAR" \
                "$PA_TYPE" \
                "$RETENTION"
        done
    done
else
    log_msg ""
    log_msg "PA17 PROJECTIONS SKIPPED"
fi

# Wait for the final partial batch.
if ! wait_for_current_batch; then
    log_msg "ERROR: One or more jobs in the final batch failed."
    log_msg "Inspect the detailed log: ${LOG_FILE}"
    exit 1
fi

# =====================================================================
# 15. FINAL STATUS
# =====================================================================

log_msg ""
log_msg "======================================================="
log_msg "WORKFLOW SUMMARY"
log_msg "Jobs launched: ${LAUNCHED_COUNT}"
log_msg "Jobs completed and validated: ${COMPLETED_COUNT}"
log_msg "Validated outputs skipped: ${SKIPPED_COUNT}"
log_msg "Invalid outputs removed: ${REMOVED_INVALID_COUNT}"
log_msg "Failed jobs: ${FAILED_COUNT}"
log_msg "End: $(date)"
log_msg "Log file: ${LOG_FILE}"
log_msg "======================================================="

if (( FAILED_COUNT > 0 )); then
    exit 1
fi

log_msg "ALL REQUESTED ENDEMIC OUTPUTS ARE PRESENT AND VALID."


