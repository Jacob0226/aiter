#!/bin/bash
set -x

cp ~/PR/aiter/prof/SGLangv0.5.5.post3_bench_one_batch_rpd.py \
    /sgl-workspace/sglang/python/sglang/bench_one_batch.py

PAGE_SIZES=(1 16)
BS_VALUES=(1024 512 128 16)
ILEN_VALUES=(256 512 1024 2048 4096)
OLEN_VALUES=(3)



MODEL_PATH="/data/huggingface/hub/amd/grok-1-W4A8KV8"
TOKENIZER_PATH="/data/huggingface/hub/Xenova/grok-1-tokenizer"

# --- Main Execution Loop ---
for PS in "${PAGE_SIZES[@]}"; do
    for BS in "${BS_VALUES[@]}"; do
        for ILEN in "${ILEN_VALUES[@]}"; do
            # No idea why SGLang reports Memory access fault 
            if (( BS >= 1024 && ILEN > 256 )); then
                continue
            fi
            if (( BS >= 512 && ILEN >= 1024 )); then
                continue
            fi
            if (( ILEN >= 4096 )); then
                continue
            fi


            for OLEN in "${OLEN_VALUES[@]}"; do
                for QKV_VERSION_MODE in "GOLDEN" "EXPERIMENTAL"; do
                    # Construct SGLANG_ARGS
                    ilen_minus3=$(($ILEN - 3))
                    SGLANG_ARGS="--batch-size ${BS} --input ${ilen_minus3} --output ${OLEN}  \
                                 --tp 8 --page-size ${PS} --quantization fp8 --trust-remote-code \
                                 --model ${MODEL_PATH} \
                                 --tokenizer-path ${TOKENIZER_PATH} \
                                 --attention-backend aiter --enable-profile-decode-rpd \
                                 --mem-fraction-static 0.8"
                        
                    echo "======================================================"
                    echo "Starting Benchmark for Version: ${QKV_VERSION_MODE} | BS: ${BS} | ILEN: ${ILEN} | PageSize: ${PS} | OLEN: ${OLEN}"
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
                    echo "CMD:" 2>&1 | tee ${OUT}.log
                    echo "RCCL_MSCCL_ENABLE=0 SGLANG_USE_AITER=1 SGLANG_INT4_WEIGHT=1 "  2>&1 | tee ${OUT}.log
                    echo "python -m sglang.bench_one_batch ${SGLANG_ARGS}" 2>&1 | tee ${OUT}.log

                    RCCL_MSCCL_ENABLE=0 SGLANG_USE_AITER=1 SGLANG_INT4_WEIGHT=1 python -m \
                        sglang.bench_one_batch ${SGLANG_ARGS} 2>&1 | tee "${OUT}.log"
                      
                    # --- RPD Trace Processing ---
                    mv trace.rpd "${OUT}.rpd"
                    # Convert RPD to CSV (top-level view)
                    sqlite3 "${OUT}.rpd" ".mode csv" ".header on" ".output ${OUT}.csv" "select * from top;" ".output stdout"
                    # # Convert RPD to Chrome tracing JSON format
                    # python3 /app/rocmProfileData/tools/rpd2tracing.py "${OUT}.rpd" "${OUT}.json" 2>&1 | tee -a "${OUT}.log"
                    # Parse csv and save decode kernel into json
                    python ~/PR/aiter/prof/E2E_prof_RPD_helper.py --csv "${OUT}.csv" 2>&1 | tee -a "${OUT}.log"

                done
            done
        done
    done
done