#!/bin/bash
set -x

PAGE_SIZES=(1 16)
BS_VALUES=(512 128 16) # 1024 OOM
ILEN_VALUES=(256)
# Define the list of OLEN values to loop through (Output Length / Max Generated Tokens)
# The values correspond to: 3, (512-256=256), (1024-256=768), (2048-256=1792), (4096-256=3840)
OLEN_VALUES=(3 256 768 1792 3840)

MODEL_PATH="/data/huggingface/hub/amd/grok-1-W4A8KV8"
TOKENIZER_PATH="/data/huggingface/hub/Xenova/grok-1-tokenizer"

# Function to run the benchmark for a specific version and page size
# Arguments: $1 = QKV_VERSION (GOLDEN or EXPERIMENTAL), $2 = PageSize
run_benchmark() {
    local version=$1      # QKV_VERSION (e.g., GOLDEN or EXPERIMENTAL)
    local page_size=$2    # Current PageSize value

    # BS, ILEN, OLEN are now controlled by the outer loops and exported globally
    
    # Construct SGLANG_ARGS based on the current page_size, BS, ILEN, and OLEN
    SGLANG_ARGS="--batch-size ${BS} --input ${ILEN} --output ${OLEN} --tp 8 --page-size ${page_size} \
                 --quantization fp8 --trust-remote-code \
                 --model ${MODEL_PATH} \
                 --tokenizer-path ${TOKENIZER_PATH} \
                 --attention-backend aiter --enable-profile-decode-rpd"
                   
    echo "======================================================"
    echo "Starting Benchmark for Version: ${version} | BS: ${BS} | ILEN: ${ILEN} | PageSize: ${page_size} | OLEN: ${OLEN}"
    echo "======================================================"

    export QKV_VERSION="${version}"
    export PageSize="${page_size}" # Export PageSize for file naming
    
    # Create the output filename (now includes BS, ILEN, OLEN, and PageSize for uniqueness)
    export OUT="E2E_bs${BS}_ilen${ILEN}_olen${OLEN}_PageSize${PageSize}_${QKV_VERSION}"
    
    # Execute the benchmark run
    RCCL_MSCCL_ENABLE=0 SGLANG_USE_AITER=1 SGLANG_INT4_WEIGHT=1 python -m \
        sglang.bench_one_batch ${SGLANG_ARGS} 2>&1 | tee "${OUT}.log"
    
    # --- RPD Trace File Processing ---
    # 1. Rename the RPD file
    mv trace.rpd ${OUT}.rpd
    # 2. Convert trace.rpd into a CSV file (top-level view)
    sqlite3 ${OUT}.rpd ".mode csv" ".header on" ".output ${OUT}.csv" "select * from top;" ".output stdout"
    # 3. Convert trace.rpd to JSON tracing format
    python3 /app/rocmProfileData/tools/rpd2tracing.py ${OUT}.rpd ${OUT}.json 2>&1 | tee -a "${OUT}.log"
    # 4. Helper script to process the JSON trace
    python ~/PR/aiter/prof/rpd_trace_helper.py -i ${OUT}.json 2>&1 | tee -a "${OUT}.log"
}

# --- Main Execution Loop ---
# Loop through all defined Page Sizes (PS) - Outermost loop
for PS in "${PAGE_SIZES[@]}"; do
    
    # Loop through all defined Batch Sizes (BS)
    for CURR_BS in "${BS_VALUES[@]}"; do
        export BS=${CURR_BS} # Set the batch size for the current set of runs

        # Loop through all defined Input Lengths (ILEN)
        for CURR_ILEN in "${ILEN_VALUES[@]}"; do
            export ILEN=${CURR_ILEN} # Set the input length for the current set of runs

            # Loop through all defined Output Lengths (OLEN)
            for CURR_OLEN in "${OLEN_VALUES[@]}"; do
                export OLEN=${CURR_OLEN} # Set the output length for the current set of runs

                # Run both versions for the current configuration (PageSize is fixed in this outer loop)
                run_benchmark "EXPERIMENTAL" "${PS}"
                run_benchmark "GOLDEN" "${PS}"
            done
        done
    done
done