#!/bin/bash

# --- 1. Configuration ---
TAXAS=("Amphibians" "Mammals" "Bird")
MODELS=("GAM" "GBM")
PYTHON_SCRIPT="/capacity/occr_davin/mguzman/chari_P2_review/scripts_git/processing/Endemics/Processing/2_luf_sumprob_ssp2_full.py"

# --- 2. Logging Setup ---
LOG_DIR="/capacity/occr_davin/mguzman/chari_P2_review/scripts/logs"
mkdir -p "$LOG_DIR"
# Named specifically for this scenario to avoid confusion
LOG_FILE="$LOG_DIR/run_endemics_bcw_17_SSP245_$(date +%F).log"
touch "$LOG_FILE"

log_msg() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log_msg "======================================================="
log_msg "STARTING TARGETED BIO RUN (LU: SSP245)"
log_msg "Time: $(date)"
log_msg "======================================================="

# --- SCENARIOS CONFIG ---
GCMS=("HadGEM2-ES" "GFDL-ESM2M" "IPSL-CM5A-LR" "MIROC5")
CLIM_SCENARIOS=("rcp26")
LU_SCENARIO="ssp245"
# Only one entry here - ensure no stray commas or extra parentheses
TYPES=("baseline" "pa30_bcw")
YEAR="2080"

# --- 3. Execution Loops ---
for type_arg in "${TYPES[@]}"; do
    log_msg "\n>>> PHASE: FUTURE ${type_arg^^}"
    
    for sc in "${CLIM_SCENARIOS[@]}"; do
        for g in "${GCMS[@]}"; do
            log_msg "-------------------------------------------------------"
            log_msg "Climate: $sc | LandUse: $LU_SCENARIO | GCM: $g | Mode: $type_arg"
            log_msg "-------------------------------------------------------"
            
            for m in "${MODELS[@]}"; do
                for t in "${TAXAS[@]}"; do
                    
                    # Skipping Birds for GBM
                    if [[ "$t" == "Bird" && "$m" == "GBM" ]]; then 
                        log_msg "Skipping $t for $m"
                        continue 
                    fi
                    
                    log_msg "Launching: $t | $m"
                    
                    # Note the backslashes (\) for clean multi-line command
                    python3 "$PYTHON_SCRIPT" \
                        -m "$m" \
                        -a "$t" \
                        -g "$g" \
                        -sc "$sc" \
                        -sl "$LU_SCENARIO" \
                        -y "$YEAR" \
                        --type "$type_arg" >> "$LOG_FILE" 2>&1 &
                    
                    sleep 1
                done
                # Wait for current Taxas to finish before starting next model
                wait 
            done
        done
    done
done

wait
log_msg "\n======================================================="
log_msg "TARGETED BCW & 17 RUN COMPLETED: $(date)"
log_msg "======================================================="