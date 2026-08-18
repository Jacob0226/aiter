#!/usr/bin/env bash
# A/B the wide and legacy geometries of indexer_qk_rope_quant_and_cache.
#
# aiter's JIT does not reliably notice edits to an already built module, so the
# module_cache object is dropped first; every run here starts from a fresh
# compile of csrc/kernels/cache_kernels.cu.
set -euo pipefail

AITER_DIR=${AITER_DIR:-/home/jacchang/PR/aiter}
MODE=${1:-all}

rm -rf "$AITER_DIR/aiter/jit/build/module_cache" "$AITER_DIR/aiter/jit/module_cache.so"

cd "$AITER_DIR"
export PYTHONPATH="$AITER_DIR"
TEST=op_tests/test_indexer_qk_rope_quant_and_cache.py

if [[ "$MODE" == "all" || "$MODE" == "check" ]]; then
    echo "########## reference check (wide) ##########"
    python3 "$TEST" --check
    echo "########## golden: legacy -> wide ##########"
    AITER_INDEXER_QK_DISABLE_WIDE=1 python3 "$TEST" --save-golden /tmp/indexer_qk_legacy.pt
    python3 "$TEST" --cmp-golden /tmp/indexer_qk_legacy.pt
fi

if [[ "$MODE" == "all" || "$MODE" == "bench" ]]; then
    echo "########## perf: wide ##########"
    python3 "$TEST" --bench
    echo "########## perf: legacy ##########"
    AITER_INDEXER_QK_DISABLE_WIDE=1 python3 "$TEST" --bench
fi
