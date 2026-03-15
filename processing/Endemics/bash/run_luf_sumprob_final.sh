#!/bin/bash

# --- 1. Configuration ---
TAXAS=("Amphibians" "Mammals" "Bird")
MODELS=("GAM" "GBM")

PYTHON_SCRIPT="/capacity/occr_davin/mguzman/chari_P2_review/scripts/Endemics/Processing/2_luf_sumprob_full.py"

# --- 2. Logging Setup ---
LOG_DIR="/capacity/occr_davin/mguzman/chari_P2_review/scripts/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_endemics_30_scenarios_$(date +%F).log"
touch "$LOG_FILE"

log_msg() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log_msg "======================================================="
log_msg "STARTING ENDEMIC SPECIES ANALYSIS: $(date)"
log_msg "======================================================="

GCMS=("GFDL-ESM2M" "IPSL-CM5A-LR" "MIROC5" "HadGEM2-ES")
# Future scenarios to run
SCENARIOS=("rcp26" "rcp60")
# All types including baseline and restoration
FUTURE_TYPES=("pa30_bcw" "pa30_bio" "baseline" "clim_only")

# --- 3. Execution Function ---
# Helper function to avoid repeating the loop logic
run_analysis() {
    local year=$1
    local scenario=$2
    local type_arg=$3

    log_msg "\n>>> RUNNING: Year $year | Scenario $scenario | Type $type_arg"
    
    for g in "${GCMS[@]}"; do
        for m in "${MODELS[@]}"; do
            for t in "${TAXAS[@]}"; do
                
                log_msg "Launching: $t | $m | $g"
                
                # Run in background
                python3 $PYTHON_SCRIPT -m "$m" -a "$t" -g "$g" -s "$scenario" -y "$year" --type "$type_arg" >> "$LOG_FILE" 2>&1 &
            done
            # Wait for all taxa for a specific SDM/GCM to finish to prevent CPU overload
            wait 
        done
    done
}

# --- 4. Main Execution ---

# PHASE: Future Scenarios
for sc in "${SCENARIOS[@]}"; do
    for ty in "${FUTURE_TYPES[@]}"; do
        run_analysis "2080" "$sc" "$ty"
    done
done

log_msg "\n======================================================="
log_msg "ALL ENDEMIC SCENARIOS COMPLETED: $(date)"
log_msg "======================================================="