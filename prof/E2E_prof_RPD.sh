#!/bin/bash
# Enable command tracing/printing
set -x

# --- MANDATORY VARIABLE DEFINITION ---
# The original script used 'if MODEL == "GROK1":' but 'MODEL' was never defined.
# Set the MODEL variable here. Defaulting to GROK1 for a runnable script.
# You can change this or pass it as an argument: ./script.sh GROK2
MODEL=${1:-"GROK1"}

# Copy the benchmark file
cp ~/PR/aiter/prof/SGLangv0.5.5.post3_bench_one_batch_rpd.py \
    /sgl-workspace/sglang/python/sglang/bench_one_batch.py

# --- Configuration Arrays ---
PAGE_SIZES=(16 1)
BS_VALUES=(1024 512 128 16)
ILEN_VALUES=(256 512 1024 2048 4096)
OLEN_VALUES=(3 2000)

# --- Model Specific Configuration (Bash Conditional Syntax Fixed) ---
if [[ "$MODEL" == "GROK1" ]]; then
    MODEL_PATH="/data/huggingface/hub/amd/grok-1-W4A8KV8"
    TOKENIZER_PATH="/data/huggingface/hub/Xenova/grok-1-tokenizer"
    ENV="RCCL_MSCCL_ENABLE=0 SGLANG_USE_AITER=1 SGLANG_INT4_WEIGHT=1"
elif [[ "$MODEL" == "GROK2" ]]; then
    MODEL_PATH="/data/huggingface/hub/xai-org/grok-2"
    TOKENIZER_PATH="/data/huggingface/hub/xai-org/grok-2/tokenizer.tok.json"
    ENV="RCCL_MSCCL_ENABLE=0 SGLANG_USE_AITER=1 SGLANG_INT4_WEIGHT=0 SGLANG_ROCM_DISABLE_LINEARQUANT=0"
else
    echo "ERROR: MODEL environment variable must be set to GROK1 or GROK2." >&2
    exit 1
fi

# --- Main Execution Loop ---
for PS in "${PAGE_SIZES[@]}"; do
    for BS in "${BS_VALUES[@]}"; do
        for ILEN in "${ILEN_VALUES[@]}"; do
            
            # Skip conditions for memory management (Arithmetic comparison is correct Bash syntax)
            if (( BS >= 1024 && ILEN > 256 )); then
                echo "Skipping (BS=${BS}, ILEN=${ILEN}): Memory constraint 1."
                continue
            fi
            if (( BS >= 512 && ILEN >= 1024 )); then
                echo "Skipping (BS=${BS}, ILEN=${ILEN}): Memory constraint 2."
                continue
            fi
            if (( ILEN >= 4096 )); then
                echo "Skipping (ILEN=${ILEN}): Too long input."
                continue
            fi

            for OLEN in "${OLEN_VALUES[@]}"; do
                for QKV_VERSION_MODE in "GOLDEN" "EXPERIMENTAL"; do
                    # Calculate input length
                    # Note: The original logic `ilen_minus3=$(($ILEN - 3))` is correct Bash arithmetic expansion
                    ilen_minus3=$((ILEN - 3))

                    # Construct SGLANG_ARGS (Indentation/newlines preserved using quotes and backslashes)
                    SGLANG_ARGS="--batch-size ${BS} --input ${ilen_minus3} --output ${OLEN} \
                                 --tp 8 --page-size ${PS} --quantization fp8 --trust-remote-code \
                                 --model ${MODEL_PATH} \
                                 --tokenizer-path ${TOKENIZER_PATH} \
                                 --attention-backend aiter --enable-profile-decode-rpd \
                                 --mem-fraction-static 0.8"
                        
                    echo "======================================================"
                    echo "Starting Benchmark for Version: ${QKV_VERSION_MODE} | MODEL: ${MODEL} | BS: ${BS} | ILEN: ${ILEN} | PageSize: ${PS} | OLEN: ${OLEN}"
                    echo "======================================================"

                    # Export runtime environment variables
                    export QKV_VERSION="${QKV_VERSION_MODE}"
                    export PageSize="${PS}"
                        
                    # Create the output filename prefix
                    OUT="E2E_bs${BS}_ilen${ILEN}_olen${OLEN}_PageSize${PS}_${QKV_VERSION_MODE}"
                    
                    if [[ -f "${OUT}.rpd" ]]; then
                        echo "Skipping: ${OUT}.rpd already exists."
                        continue
                    fi
                        
                    # Execute the benchmark run
                    echo "CMD: $ENV python -m sglang.bench_one_batch ${SGLANG_ARGS}" | tee "${OUT}.log"
                    
                    # Execute command, redirecting stdout and stderr to the log file
                    eval "$ENV python -m sglang.bench_one_batch ${SGLANG_ARGS} 2>&1 | tee -a \"${OUT}.log\""
                        
                    # --- RPD Trace Processing ---
                    # Move and rename trace file
                    mv trace.rpd "${OUT}.rpd"
                    
                    # Convert RPD to CSV (top-level view)
                    sqlite3 "${OUT}.rpd" ".mode csv" ".header on" ".output ${OUT}.csv" "select * from top;" ".output stdout"
                    
                    # Parse csv and save decode kernel into json
                    python ~/PR/aiter/prof/E2E_prof_RPD_helper.py --csv "${OUT}.csv" 2>&1 | tee -a "${OUT}.log"

                done
            done
        done
    done
done