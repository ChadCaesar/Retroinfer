#!/bin/bash
# ============================================================
# E2: 聚类选择方式消融 (Top-k vs Top-p)
# ============================================================
# Usage:
#   bash benchmark/run_e2_selection.sh
#
# Configurable via env vars:
#   SKIP_THROUGHPUT=1 bash benchmark/run_e2_selection.sh  # skip throughput tests
# ============================================================

set -euo pipefail
source "$(dirname "$0")/common_exp.sh"
setup_logging "E2_selection"

SELECT_MODES=("top-k" "top-p")
FIXED_POLICY="sclru"
DIFFUSE_LB=("gov_report" "musique" "passage_count")
FOCUSED_LB=("passage_retrieval_en" "narrativeqa" "qasper")
SINGLE_RULER=("niah_single_1")
MULTI_RULER=("niah_multikey_1")

for mode in "${SELECT_MODES[@]}"; do
    log_msg "========== E2: mode=${mode} =========="

    # LongBench
    for task in "${DIFFUSE_LB[@]}"; do
        run_longbench "${task}" "${mode}" "True" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}"
    done
    for task in "${FOCUSED_LB[@]}"; do
        run_longbench "${task}" "${mode}" "True" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}"
    done
    eval_longbench "${mode}" "True" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}"

    # RULER
    for task in "${SINGLE_RULER[@]}"; do
        run_ruler "${task}" "${mode}" "True" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}"
    done
    for task in "${MULTI_RULER[@]}"; do
        run_ruler "${task}" "${mode}" "True" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}"
    done

    # Throughput
    if [ "${SKIP_THROUGHPUT:-0}" != "1" ]; then
        run_throughput "${mode}" "True" "${FIXED_POLICY}" "${TOP_P}" "${mode}"
    fi
done

print_summary
