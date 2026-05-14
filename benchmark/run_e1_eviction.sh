#!/bin/bash
# ============================================================
# E1: 缓存替换策略消融 (LRU vs SCLRU vs ARC)
# ============================================================
# Usage:
#   bash benchmark/run_e1_eviction.sh
#
# Configurable via env vars:
#   GPU_TEMP_LIMIT=70     bash benchmark/run_e1_eviction.sh   # lower temp threshold (default 80)
#   LONG_ONLY=1           bash benchmark/run_e1_eviction.sh   # LongBench only
#   RULER_ONLY=1          bash benchmark/run_e1_eviction.sh   # RULER only
# ============================================================

set -euo pipefail
source "$(dirname "$0")/common_exp.sh"
setup_logging "E1_eviction"

POLICIES=("lru" "sclru" "arc")
HIGH_SENS_LB=("musique" "gov_report" "passage_count")
LOW_SENS_LB=("passage_retrieval_en" "trec" "triviaqa")
HIGH_SENS_RULER=("niah_multikey_1")
LOW_SENS_RULER=("niah_single_1")

for policy in "${POLICIES[@]}"; do
    log_msg "========== E1: policy=${policy} =========="

    # LongBench
    if [ "${RULER_ONLY:-0}" != "1" ]; then
        for task in "${HIGH_SENS_LB[@]}"; do
            run_longbench "${task}" "top-p" "True" "${policy}" "RetroInfer" "${TOP_P}" "${REUSE_THRESHOLD}"
        done
        for task in "${LOW_SENS_LB[@]}"; do
            run_longbench "${task}" "top-p" "True" "${policy}" "RetroInfer" "${TOP_P}" "${REUSE_THRESHOLD}"
        done
        eval_longbench "top-p" "True" "${policy}" "RetroInfer" "${TOP_P}" "${REUSE_THRESHOLD}"
    fi

    # RULER
    if [ "${LONG_ONLY:-0}" != "1" ]; then
        for task in "${HIGH_SENS_RULER[@]}"; do
            run_ruler "${task}" "top-p" "True" "${policy}" "RetroInfer" "${TOP_P}" "${REUSE_THRESHOLD}"
        done
        for task in "${LOW_SENS_RULER[@]}"; do
            run_ruler "${task}" "top-p" "True" "${policy}" "RetroInfer" "${TOP_P}" "${REUSE_THRESHOLD}"
        done
    fi
done

print_summary
