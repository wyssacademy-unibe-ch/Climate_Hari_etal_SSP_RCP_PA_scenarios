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
    "pa30_bio"
    "pa30_bcw"
)

RETENTION_ASSUMPTIONS=(
    "complete"
    "funding_constrained"
)

PYTHON_SCRIPT="/capacity/occr_davin/mguzman/chari_P2_review/scripts_git/processing/Global/2_luf_sumprob_PA_management.py"

PYTHON_EXECUTABLE="python3"

FUTURE_YEAR="2080"

OUTPUT_ROOT="/capacity/occr_davin/mguzman/chari_P2_review/Global/PA_scenarios/Sumprob_revised"

LOG_DIR="/capacity/occr_davin/mguzman/chari_P2_review/scripts/logs"

RUN_DATE="$(date +%F_%H-%M-%S)"

LOG_FILE="${LOG_DIR}/run_luf_pa_only_${RUN_DATE}.log"


# =====================================================================
# 2. RUN OPTIONS
# =====================================================================

# Run no more than six Python processes simultaneously.
MAX_PARALLEL=6

# When false, existing non-empty outputs are skipped.
OVERWRITE_EXISTING=false

# A zero-byte file is considered incomplete and will be regenerated.
REQUIRE_NONEMPTY_OUTPUT=true


# =====================================================================
# 3. LOGGING
# =====================================================================

mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

log_msg() {
    printf "%b\n" "$1" | tee -a "$LOG_FILE"
}


# =====================================================================
# 4. VALIDATION
# =====================================================================

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    log_msg "ERROR: Python script not found:"
    log_msg "  $PYTHON_SCRIPT"
    exit 1
fi

if ! command -v "$PYTHON_EXECUTABLE" >/dev/null 2>&1; then
    log_msg "ERROR: Python executable not found:"
    log_msg "  $PYTHON_EXECUTABLE"
    exit 1
fi

# wait -n requires Bash 4.3 or newer.
if (( BASH_VERSINFO[0] < 4 )) || \
   (( BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3 )); then

    log_msg "ERROR: Bash 4.3 or newer is required."
    log_msg "Current version: ${BASH_VERSION}"
    exit 1
fi


# =====================================================================
# 5. MODEL AVAILABILITY BY TAXON
# =====================================================================

models_for_taxon() {
    local taxon="$1"

    case "$taxon" in
        "Amphibians")
            printf "%s\n" "GAM" "GBM"
            ;;

        "Mammals")
            printf "%s\n" "GAM" "GBM"
            ;;

        "Bird")
            # Bird GBM is excluded because the corresponding SDM files
            # are not accessible.
            printf "%s\n" "GAM"
            ;;

        *)
            log_msg "ERROR: Unknown taxonomic group: $taxon"
            return 1
            ;;
    esac
}


# =====================================================================
# 6. EXPECTED OUTPUT PATH
# =====================================================================

get_expected_output_path() {
    local model="$1"
    local taxon="$2"
    local gcm="$3"
    local scenario="$4"
    local pa_type="$5"
    local retention="$6"

    local output_dir
    local suffix
    local filename

    case "$pa_type" in

        "pa30_bio")
            output_dir="${OUTPUT_ROOT}/PA30_Bio/${retention}"
            suffix="pa30_bio_${retention}"
            ;;

        "pa30_bcw")
            output_dir="${OUTPUT_ROOT}/PA30_BCW/${retention}"
            suffix="pa30_bcw_${retention}"
            ;;

        *)
            log_msg "ERROR: Unsupported PA type: $pa_type"
            return 1
            ;;
    esac

    filename="summed_prob_${taxon}_${model}_${gcm}_${scenario}_${FUTURE_YEAR}_${suffix}.nc"

    printf "%s\n" "${output_dir}/${filename}"
}


# =====================================================================
# 7. CHECK WHETHER AN OUTPUT IS COMPLETE
# =====================================================================

output_is_complete() {
    local output_path="$1"

    if [[ "$OVERWRITE_EXISTING" == "true" ]]; then
        return 1
    fi

    if [[ "$REQUIRE_NONEMPTY_OUTPUT" == "true" ]]; then
        # -s is true only when the file exists and has a size above zero.
        [[ -s "$output_path" ]]
    else
        [[ -e "$output_path" ]]
    fi
}


# =====================================================================
# 8. RUN ONE JOB
# =====================================================================

execute_luf_job() {
    local model="$1"
    local taxon="$2"
    local gcm="$3"
    local scenario="$4"
    local pa_type="$5"
    local retention="$6"
    local output_path="$7"

    local description

    description="${taxon} | ${model} | ${gcm} | ${scenario} | ${pa_type} | ${retention}"

    {
        printf "START: %s | %s\n" "$(date '+%F %T')" "$description"
        printf "EXPECTED OUTPUT: %s\n" "$output_path"
        printf "COMMAND:"

        printf " %q" \
            "$PYTHON_EXECUTABLE" \
            "$PYTHON_SCRIPT" \
            -m "$model" \
            -a "$taxon" \
            -g "$gcm" \
            -s "$scenario" \
            -y "$FUTURE_YEAR" \
            --type "$pa_type" \
            --retention "$retention"

        printf "\n"
    } >> "$LOG_FILE"

    if "$PYTHON_EXECUTABLE" \
        "$PYTHON_SCRIPT" \
        -m "$model" \
        -a "$taxon" \
        -g "$gcm" \
        -s "$scenario" \
        -y "$FUTURE_YEAR" \
        --type "$pa_type" \
        --retention "$retention" \
        >> "$LOG_FILE" 2>&1; then

        if [[ -s "$output_path" ]]; then
            log_msg "COMPLETED: $description"
            return 0
        fi

        log_msg "ERROR: Job finished but output is missing or empty:"
        log_msg "  Job: $description"
        log_msg "  Expected output: $output_path"

        return 1
    fi

    log_msg "ERROR: Python job failed:"
    log_msg "  Job: $description"
    log_msg "  Expected output: $output_path"

    return 1
}


# =====================================================================
# 9. SIX-JOB WORKER POOL
# =====================================================================

RUNNING_JOBS=0
LAUNCHED_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0


wait_for_one_job() {
    if wait -n; then
        :
    else
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi

    RUNNING_JOBS=$((RUNNING_JOBS - 1))
}


wait_for_all_jobs() {
    while (( RUNNING_JOBS > 0 )); do
        wait_for_one_job
    done
}


launch_job() {
    local model="$1"
    local taxon="$2"
    local gcm="$3"
    local scenario="$4"
    local pa_type="$5"
    local retention="$6"

    local output_path
    local description

    output_path="$(
        get_expected_output_path \
            "$model" \
            "$taxon" \
            "$gcm" \
            "$scenario" \
            "$pa_type" \
            "$retention"
    )"

    description="${taxon} | ${model} | ${gcm} | ${scenario} | ${pa_type} | ${retention}"

    # Skip outputs that have already been created.
    if output_is_complete "$output_path"; then
        log_msg "SKIPPED: $description"
        log_msg "  Existing output: $output_path"

        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        return 0
    fi

    # Remove an incomplete zero-byte file before relaunching its job.
    if [[ -e "$output_path" && ! -s "$output_path" ]]; then
        log_msg "Removing zero-byte output:"
        log_msg "  $output_path"

        rm -f "$output_path"
    fi

    mkdir -p "$(dirname "$output_path")"

    # Do not launch another process until fewer than six are running.
    while (( RUNNING_JOBS >= MAX_PARALLEL )); do
        wait_for_one_job
    done

    execute_luf_job \
        "$model" \
        "$taxon" \
        "$gcm" \
        "$scenario" \
        "$pa_type" \
        "$retention" \
        "$output_path" &

    RUNNING_JOBS=$((RUNNING_JOBS + 1))
    LAUNCHED_COUNT=$((LAUNCHED_COUNT + 1))

    log_msg "LAUNCHED ${LAUNCHED_COUNT}: $description"
    log_msg "  Running jobs: $RUNNING_JOBS / $MAX_PARALLEL"
}


# =====================================================================
# 10. RUN PA SCENARIOS
# =====================================================================

log_msg "======================================================="
log_msg "STARTING PA-ONLY LAND-USE FILTER WORKFLOW"
log_msg "Start: $(date)"
log_msg "Python script: $PYTHON_SCRIPT"
log_msg "Output root: $OUTPUT_ROOT"
log_msg "Maximum concurrent jobs: $MAX_PARALLEL"
log_msg "Overwrite existing outputs: $OVERWRITE_EXISTING"
log_msg "Log file: $LOG_FILE"
log_msg "======================================================="

log_msg ""
log_msg "Model availability:"
log_msg "  Amphibians: GAM and GBM"
log_msg "  Mammals: GAM and GBM"
log_msg "  Bird: GAM only"
log_msg "  Bird GBM jobs will not be launched."

log_msg ""
log_msg "Scenario definitions:"
log_msg "  rcp26 = SSP1-2.6 land use + RCP2.6 climate"
log_msg "  rcp45 = SSP2-4.5 land use + RCP2.6 climate"
log_msg "  rcp60 = SSP4-6.0 land use + RCP6.0 climate"

log_msg ""
log_msg "PA configurations:"
log_msg "  pa30_bio"
log_msg "  pa30_bcw"

log_msg ""
log_msg "Retention assumptions:"
log_msg "  complete"
log_msg "  funding_constrained"


for pa_type in "${PA_TYPES[@]}"; do

    for retention in "${RETENTION_ASSUMPTIONS[@]}"; do

        for scenario in "${SCENARIOS[@]}"; do

            for gcm in "${GCMS[@]}"; do

                for taxon in "${TAXA[@]}"; do

                    while IFS= read -r model; do

                        launch_job \
                            "$model" \
                            "$taxon" \
                            "$gcm" \
                            "$scenario" \
                            "$pa_type" \
                            "$retention"

                    done < <(
                        models_for_taxon "$taxon"
                    )

                done

            done

        done

    done

done


# Wait for the remaining jobs after all combinations have been queued.
wait_for_all_jobs


# =====================================================================
# 11. FINAL STATUS
# =====================================================================

log_msg ""
log_msg "======================================================="
log_msg "PA-ONLY WORKFLOW SUMMARY"
log_msg "======================================================="
log_msg "Jobs launched: $LAUNCHED_COUNT"
log_msg "Existing outputs skipped: $SKIPPED_COUNT"
log_msg "Failed jobs: $FAILED_COUNT"
log_msg "End: $(date)"
log_msg "Log file: $LOG_FILE"
log_msg "======================================================="

if (( FAILED_COUNT > 0 )); then
    log_msg "WORKFLOW FINISHED WITH FAILURES."
    log_msg "Run the same script again to retry missing outputs."
    exit 1
fi

log_msg "ALL REQUESTED PA LUF OUTPUTS ARE PRESENT."