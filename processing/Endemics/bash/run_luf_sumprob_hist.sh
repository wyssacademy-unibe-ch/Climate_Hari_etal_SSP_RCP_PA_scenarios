#!/bin/bash

# --- 1. Configuration ---
TAXAS=("Amphibians" "Mammals" "Bird")
MODELS=("GAM" "GBM")
PYTHON_SCRIPT="/capacity/occr_davin/mguzman/chari_P2_review/scripts/Endemics/Processing/2_luf_sumprob_full.py"

# --- 2. Logging Setup ---
LOG_DIR="/capacity/occr_davin/mguzman/chari_P2_review/scripts/logs"
mkdir -p "$LOG_DIR"
# New log name to distinguish from your ongoing future runs
LOG_FILE="$LOG_DIR/run_endemics_HIST_ONLY_$(date +%F).log"

log_msg() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log_msg "======================================================="
log_msg "STARTING HISTORICAL ENDEMIC ANALYSIS (EWEMBI): $(date)"
log_msg "======================================================="

# --- 3. Execution Function ---
run_historical() {
    log_msg "\n>>> PHASE: HISTORICAL (1995)"
    # Setting g to EWEMBI ensures the filename: summed_prob_TAXA_SDM_EWEMBI_1995_hist.nc
    local g="EWEMBI" 
    
    for m in "${MODELS[@]}"; do
        for t in "${TAXAS[@]}"; do
            # Skip Birds for GBM
            if [[ "$t" == "Bird" && "$m" == "GBM" ]]; then 
                continue 
            fi
            
            log_msg "Launching: $t | $m | $g (1995)"
            
            # Note: -s rcp26 is just a placeholder to satisfy the parser
            python3 $PYTHON_SCRIPT -m "$m" -a "$t" -g "$g" -s "rcp26" -y "1995" --type "baseline" >> "$LOG_FILE" 2>&1 &
        done
        # Wait for this model's taxa to finish before starting the next model
        wait 
    done
}

# --- 4. Execution ---

# Clean up the previous 20 redundant files first if they exist
rm /capacity/occr_davin/mguzman/chari_P2_review/Endemics/PA_scenarios/Sumprob/WDPA/Historical/*.nc

run_historical

log_msg "\n======================================================="
log_msg "HISTORICAL ENDEMIC RUN COMPLETED: $(date)"
log_msg "======================================================="