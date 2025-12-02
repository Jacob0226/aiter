#!/bin/bash
set -x

PAGE_SIZES=(1 16)
BS_VALUES=(1024 512 128 16)
ILENS=(256 512 1024 2048 4096)
VERSIONS=("GOLDEN" "EXPERIMENTAL")
WARNUP=30
ITERS=30
UT_SCRIPT=$HOME/PR/aiter/op_tests/test_pa_ragged_experimental.py
UT_HELPER_PY=$HOME/PR/aiter/prof/UT_prof_RPD_helper.py

# --- Main Execution Loop ---
# Loop through all defined Page Sizes (PS) - Outermost loop
for PS in "${PAGE_SIZES[@]}"; do
    # Loop through all defined Batch Sizes (BS)
    for BS in "${BS_VALUES[@]}"; do
        # Loop through all defined Input Lengths (ILEN)
        for ILEN in "${ILENS[@]}"; do
            python $UT_SCRIPT \
                -n "$BS" -c "$ILEN" --warmup "$WARNUP" --num-iters "$ITERS" \
                --page-size "$PS" --enable-profile

            RPD_FILENAME="trace_PS${PS}_BS${BS}_ILEN${ILEN}"
            sqlite3 trace_UT.rpd ".mode csv" ".header on" ".output trace_UT.csv" "select * from top;" ".output stdout"
            mv trace_UT.rpd "${RPD_FILENAME}.rpd"
            mv trace_UT.csv "${RPD_FILENAME}.csv"
            
            python $UT_HELPER_PY --csv  "${RPD_FILENAME}.csv"
            echo "-------------------------------------------------------------------------------------"
        done
    done
done