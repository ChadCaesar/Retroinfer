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

# ============================================================
# LongBench tasks
# ============================================================
if [ "${RULER_ONLY:-0}" != "1" ]; then
    log_msg "========== E1 LongBench =========="

    # High-sensitivity tasks (信息分散，缓存压力大)
    HIGH_SENS_TASKS=("musique" "gov_report" "passage_count")

    # Low-sensitivity tasks (对照组，注意力集中)
    LOW_SENS_TASKS=("passage_retrieval_en" "trec" "triviaqa")

    for policy in "${POLICIES[@]}"; do
        log_msg "--- E1 LongBench: policy=${policy} ---"

        for task in "${HIGH_SENS_TASKS[@]}"; do
            run_longbench "${task}" "top-p" "True" "${policy}" "RetroInfer" "${TOP_P}"
        done

        for task in "${LOW_SENS_TASKS[@]}"; do
            run_longbench "${task}" "top-p" "True" "${policy}" "RetroInfer" "${TOP_P}"
        done
    done

    # Evaluation
    log_msg "--- E1 LongBench Evaluation ---"
    for policy in "${POLICIES[@]}"; do
        eval_longbench "top-p" "True" "${policy}" "RetroInfer" "${TOP_P}"
    done
fi

# ============================================================
# RULER tasks
# ============================================================
if [ "${LONG_ONLY:-0}" != "1" ]; then
    log_msg "========== E1 RULER =========="

    # High-sensitivity (多针/多查询，缓存压力大)
    HIGH_SENS_TASKS=("niah_multikey_1" "niah_multiquery" "vt")

    # Low-sensitivity (单针对照组)
    LOW_SENS_TASKS=("niah_single_1")

    for policy in "${POLICIES[@]}"; do
        log_msg "--- E1 RULER: policy=${policy} ---"

        for task in "${HIGH_SENS_TASKS[@]}"; do
            run_ruler "${task}" "top-p" "True" "${policy}" "RetroInfer" "${TOP_P}"
        done
    done

    # Low-sensitivity for all policies
    for policy in "${POLICIES[@]}"; do
        for task in "${LOW_SENS_TASKS[@]}"; do
            run_ruler "${task}" "top-p" "True" "${policy}" "RetroInfer" "${TOP_P}"
        done
    done
fi

print_summary
