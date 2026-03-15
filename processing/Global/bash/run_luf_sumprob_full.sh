#!/bin/bash

# --- 1. Configuration ---
TAXAS=("Amphibians" "Mammals" "Bird")
MODELS=("GAM" "GBM")
PYTHON_SCRIPT="/capacity/occr_davin/mguzman/chari_P2_review/scripts/global/2_luf_sumprob_PA_full.py"

# --- 2. Logging Setup ---
LOG_DIR="/capacity/occr_davin/mguzman/chari_P2_review/scripts/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_master_wdpa_$(date +%F).log"
touch "$LOG_FILE"

log_msg() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log_msg "======================================================="
log_msg "STARTING MASTER RUN (WDPA: HIST + BASE + CLIM)"
log_msg "Target Scenarios: RCP 2.6, 4.5, 6.0"
log_msg "======================================================="

# --- 3. PHASE 1: HISTORICAL (2015) ---
log_msg "\n>>> PHASE 1: HISTORICAL (2015)"
for m in "${MODELS[@]}"; do
    for t in "${TAXAS[@]}"; do
        
        log_msg "Launching: $t | $m | 2015"
        # -s "hist" is ignored by the py script for 2015 but satisfies argparse
        python $PYTHON_SCRIPT -m "$m" -a "$t" -g "EWEMBI" -s "hist" -y "1995" --type baseline >> "$LOG_FILE" 2>&1 &
        sleep 1
    done
    wait # Wait for all taxa in this model to finish
done

# --- 4. PHASE 2: FUTURE SCENARIOS (2080) ---
GCMS=("GFDL-ESM2M" "IPSL-CM5A-LR" "MIROC5" "HadGEM2-ES")
SCENARIOS=("rcp26" "rcp60")
# Includes 'climate_only' to see the effect of climate without LU change
TYPES=("baseline" "climate_only") 
YEAR="2080"

for type_arg in "${TYPES[@]}"; do
    log_msg "\n>>> PHASE: FUTURE ${type_arg^^}"
    for sc in "${SCENARIOS[@]}"; do
        for g in "${GCMS[@]}"; do
            log_msg "------------------------------------------------"
            log_msg "Scenario: $sc | GCM: $g | Type: $type_arg"
            log_msg "------------------------------------------------"
            for m in "${MODELS[@]}"; do
                for t in "${TAXAS[@]}"; do
                    
                    python $PYTHON_SCRIPT -m "$m" -a "$t" -g "$g" -s "$sc" -y "$YEAR" --type "$type_arg" >> "$LOG_FILE" 2>&1 &
                done
            done
            wait # Wait for all Taxa/Models for THIS GCM to finish before next GCM
        done
    done
done

log_msg "\n======================================================="
log_msg "ALL WDPA SCENARIOS COMPLETED: $(date)"
log_msg "======================================================="