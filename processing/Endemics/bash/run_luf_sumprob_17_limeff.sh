#!/bin/bash

# --- 1. Configuration ---
TAXAS=("Amphibians" "Mammals" "Bird")
MODELS=("GBM" "GAM")
PYTHON_SCRIPT="/capacity/occr_davin/mguzman/chari_P2_review/scripts_git/processing/Endemics/Processing/3_luf_sumprob_PA_limeff.py"

# --- 2. Logging Setup ---
LOG_DIR="/capacity/occr_davin/mguzman/chari_P2_review/scripts/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_17endemics_limeff_$(date +%F).log"
touch "$LOG_FILE"

log_msg() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log_msg "======================================================="
log_msg "STARTING 30x30 EXPANSION RUN (BIO + CWB): $(date)"
log_msg "Log file saved to: $LOG_FILE"
log_msg "======================================================="

# --- FUTURE SCENARIOS CONFIG ---
GCMS=("GFDL-ESM2M" "IPSL-CM5A-LR" "MIROC5" "HadGEM2-ES")
SCENARIOS=("rcp60" "rcp26")
TYPES=("baseline")
YEAR="2080"

# --- 3. Execution Loops ---
for type_arg in "${TYPES[@]}"; do
    log_msg "\n>>> PHASE: FUTURE ${type_arg^^}"
    
    for sc in "${SCENARIOS[@]}"; do
        for g in "${GCMS[@]}"; do
            log_msg "-------------------------------------------------------"
            log_msg "Scenario: $sc | GCM: $g | Mode: $type_arg"
            log_msg "-------------------------------------------------------"
            
            for m in "${MODELS[@]}"; do
                for t in "${TAXAS[@]}"; do
                    
                    # Skipping Birds for GBM
                    if [[ "$t" == "Bird" && "$m" == "GBM" ]]; then 
                        log_msg "Skipping $t for $m"
                        continue 
                    fi
                    
                    log_msg "Launching: $t | $m"
                    
                    # Using python3 explicitly is safer
                    python3 "$PYTHON_SCRIPT" -m "$m" -a "$t" -g "$g" -s "$sc" -y "$YEAR" --type "$type_arg" >> "$LOG_FILE" 2>&1 &
                    sleep 1
                done
                # Wait for taxa to finish for this model before moving to next model/GCM
                wait 
            done
        done
    done
done

# Final wait to be absolutely sure everything is written to disk
wait

log_msg "\n======================================================="
log_msg "30x30 EXPANSION SCENARIOS COMPLETED: $(date)"
log_msg "======================================================="