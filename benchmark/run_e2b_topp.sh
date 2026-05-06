#!/bin/bash
# ============================================================
# E2b: Top-p 阈值敏感性 (top_p ∈ {0.3, 0.4, 0.5, 0.6})
# ============================================================
# Usage:
#   bash benchmark/run_e2b_topp.sh
# ============================================================

set -euo pipefail
source "$(dirname "$0")/common_exp.sh"
setup_logging "E2b_topp"

TOP_P_VALUES=("0.3" "0.4" "0.5" "0.6")
FIXED_SELECT="top-p"
FIXED_REUSE="True"
FIXED_POLICY="sclru"

log_msg "========== E2b: Top-p sensitivity =========="

for tp in "${TOP_P_VALUES[@]}"; do
    log_msg "--- E2b: top_p=${tp} ---"

    # LongBench
    run_longbench "gov_report"          "${FIXED_SELECT}" "${FIXED_REUSE}" "${FIXED_POLICY}" "RetroInfer" "${tp}"
    run_longbench "passage_retrieval_en" "${FIXED_SELECT}" "${FIXED_REUSE}" "${FIXED_POLICY}" "RetroInfer" "${tp}"

    # RULER
    run_ruler "niah_single_1"   "${FIXED_SELECT}" "${FIXED_REUSE}" "${FIXED_POLICY}" "RetroInfer" "${tp}"
    run_ruler "niah_multikey_1" "${FIXED_SELECT}" "${FIXED_REUSE}" "${FIXED_POLICY}" "RetroInfer" "${tp}"
done

# Evaluation
for tp in "${TOP_P_VALUES[@]}"; do
    eval_longbench "${FIXED_SELECT}" "${FIXED_REUSE}" "${FIXED_POLICY}" "RetroInfer" "${tp}"
done

print_summary
