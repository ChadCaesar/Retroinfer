#!/bin/bash
# ============================================================
# E5: 聚类复用阈值敏感性 (threshold ∈ {0.85, 0.9, 0.95, 0.99})
# ============================================================
# Usage:
#   bash benchmark/run_e5_reuse_threshold.sh
# ============================================================

set -euo pipefail
source "$(dirname "$0")/common_exp.sh"
setup_logging "E5_reuse_threshold"

THRESHOLDS=("0.85" "0.9" "0.95" "0.99")
FIXED_SELECT="top-p"
FIXED_REUSE="True"
FIXED_POLICY="sclru"

for rt in "${THRESHOLDS[@]}"; do
    log_msg "========== E5: reuse_threshold=${rt} =========="

    # LongBench
    run_longbench "gov_report" "${FIXED_SELECT}" "${FIXED_REUSE}" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}" "${rt}"
    run_longbench "musique"    "${FIXED_SELECT}" "${FIXED_REUSE}" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}" "${rt}"
    eval_longbench "${FIXED_SELECT}" "${FIXED_REUSE}" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}"

    # RULER
    run_ruler "niah_multikey_1" "${FIXED_SELECT}" "${FIXED_REUSE}" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}" "${rt}"
    run_ruler "niah_single_1"   "${FIXED_SELECT}" "${FIXED_REUSE}" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}" "${rt}"

    # Throughput
    run_throughput "${FIXED_SELECT}" "${FIXED_REUSE}" "${FIXED_POLICY}" "${TOP_P}" "rt_${rt}" "${rt}"
done

print_summary
