#!/bin/bash
# ============================================================
# E4: Top-p 阈值敏感性 (top_p ∈ {0.3, 0.4, 0.5, 0.6})
# ============================================================
# Usage:
#   bash benchmark/run_e4_topp.sh
# ============================================================

set -euo pipefail
source "$(dirname "$0")/common_exp.sh"
setup_logging "E4_topp"

TOP_P_VALUES=("0.3" "0.4" "0.5" "0.6")
FIXED_SELECT="top-p"
FIXED_REUSE="True"
FIXED_POLICY="sclru"

for tp in "${TOP_P_VALUES[@]}"; do
    log_msg "========== E4: top_p=${tp} =========="

    # LongBench
    run_longbench "gov_report"          "${FIXED_SELECT}" "${FIXED_REUSE}" "${FIXED_POLICY}" "RetroInfer" "${tp}" "${REUSE_THRESHOLD}"
    run_longbench "passage_retrieval_en" "${FIXED_SELECT}" "${FIXED_REUSE}" "${FIXED_POLICY}" "RetroInfer" "${tp}" "${REUSE_THRESHOLD}"
    eval_longbench "${FIXED_SELECT}" "${FIXED_REUSE}" "${FIXED_POLICY}" "RetroInfer" "${tp}" "${REUSE_THRESHOLD}"

    # RULER
    run_ruler "niah_single_1"   "${FIXED_SELECT}" "${FIXED_REUSE}" "${FIXED_POLICY}" "RetroInfer" "${tp}" "${REUSE_THRESHOLD}"
    run_ruler "niah_multikey_1" "${FIXED_SELECT}" "${FIXED_REUSE}" "${FIXED_POLICY}" "RetroInfer" "${tp}" "${REUSE_THRESHOLD}"

    # Throughput
    run_throughput "${FIXED_SELECT}" "${FIXED_REUSE}" "${FIXED_POLICY}" "${tp}" "top_p_${tp}"
done

print_summary
