#!/bin/bash
# ============================================================
# E3: 聚类复用消融 (Reuse ON vs OFF)
# ============================================================
# Usage:
#   bash benchmark/run_e3_reuse.sh
#
# Configurable via env vars:
#   SKIP_THROUGHPUT=1  bash benchmark/run_e3_reuse.sh
#   ACCURACY_ONLY=1    bash benchmark/run_e3_reuse.sh
# ============================================================

set -euo pipefail
source "$(dirname "$0")/common_exp.sh"
setup_logging "E3_reuse"

REUSE_VALUES=("False" "True")
FIXED_POLICY="sclru"
FIXED_SELECT="top-p"

for reuse in "${REUSE_VALUES[@]}"; do
    log_msg "========== E3: reuse=${reuse} =========="

    # LongBench
    run_longbench "gov_report" "${FIXED_SELECT}" "${reuse}" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}"
    run_longbench "musique"    "${FIXED_SELECT}" "${reuse}" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}"
    run_longbench "qasper"     "${FIXED_SELECT}" "${reuse}" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}"
    eval_longbench "${FIXED_SELECT}" "${reuse}" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}"

    # RULER
    run_ruler "niah_multikey_1" "${FIXED_SELECT}" "${reuse}" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}"

    # Throughput
    if [ "${ACCURACY_ONLY:-0}" != "1" ] && [ "${SKIP_THROUGHPUT:-0}" != "1" ]; then
        run_throughput "${FIXED_SELECT}" "${reuse}" "${FIXED_POLICY}" "${TOP_P}" "reuse_${reuse}"
    fi
done

print_summary
