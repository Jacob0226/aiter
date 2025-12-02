#!/bin/bash
set -x

PAGE_SIZES=(1 16)
BS_VALUES=(1024 512 128 16) # 1024 OOM
BS_VALUES=(1024) # 1024 OOM
ILENS=(256 512 1024 2048 4096)
VERSIONS=("GOLDEN" "EXPERIMENTAL")
WARNUP=10
ITERS=5
UT_SCRIPT=$HOME/PR/aiter/op_tests/test_pa_ragged_experimental.py
UT_HELPER_PY=$HOME/PR/aiter/prof/UT_prof_RPD_helper.py

run_and_trace() {
    local BS=$1
    local ILEN=$2
    local VERSION=$3
    local NUM_ITERS=$4
    local SUFFIX=$5
    local PS=$6
    local PS_PARAM=$7
    export RPDT_FILENAME=trace.rpd
    python3 -m rocpd.schema --create trace.rpd

    echo "Running PS=${PS} BS=${BS} ILEN=${ILEN} VERSION=${VERSION} SUFFIX=${SUFFIX} with ITERS=${NUM_ITERS}"

    LD_PRELOAD=libroctx64.so:librpd_tracer.so \
    python $UT_SCRIPT \
        -n "$BS" -c "$ILEN" --warmup "$WARNUP" --num-iters "$NUM_ITERS" \
        --page-size "$PS_PARAM" --profile "$VERSION"

    # Convert trace.rpd into csv
    sqlite3 trace.rpd ".mode csv" ".header on" ".output trace.csv" "select * from top;" ".output stdout"

    # Rename files
    mv trace.rpd "trace_PS${PS}_BS${BS}_ILEN${ILEN}_${VERSION}_${SUFFIX}.rpd"
    mv trace.csv "trace_PS${PS}_BS${BS}_ILEN${ILEN}_${VERSION}_${SUFFIX}.csv"
}

# --- Main Execution Loop ---
# Loop through all defined Page Sizes (PS) - Outermost loop
for PS in "${PAGE_SIZES[@]}"; do
    # Loop through all defined Batch Sizes (BS)
    for BS in "${BS_VALUES[@]}"; do
        # Loop through all defined Input Lengths (ILEN)
        for ILEN in "${ILENS[@]}"; do
            for VERSION in "${VERSIONS[@]}"; do
                
                # *** 1. Warmup Only ***
                # NUM_ITERS=0
                run_and_trace "$BS" "$ILEN" "$VERSION" 0 "WARMUP" "$PS" "$PAGE_SIZES"

                # *** 2. Warmup + Coldrun (AllRuns) ***
                # NUM_ITERS=$ITERS
                run_and_trace "$BS" "$ILEN" "$VERSION" "$ITERS" "AllRuns" "$PS" "$PAGE_SIZES"
                
                # Coldrun = (Warmup + Coldrun) - warmup 
                python $UT_HELPER_PY \
                    --allruns-csv  "trace_PS${PS}_BS${BS}_ILEN${ILEN}_${VERSION}_AllRuns.csv" \
                    --warmup-csv   "trace_PS${PS}_BS${BS}_ILEN${ILEN}_${VERSION}_WARMUP.csv"
                echo "-------------------------------------------------------------------------------------"
            done
        done
    done
done