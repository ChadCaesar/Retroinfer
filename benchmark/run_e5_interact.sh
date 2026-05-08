#!/bin/bash
# ============================================================
# E5: 参数交互效应验证 (eviction × cluster_select × cluster_reuse)
# ============================================================
# 2×2×2 = 8 combinations sampled on 4 tasks
# Usage:
#   bash benchmark/run_e5_interact.sh
# ============================================================

set -euo pipefail
source "$(dirname "$0")/common_exp.sh"
setup_logging "E5_interact"

EVICTIONS=("lru" "arc")
SELECTS=("top-k" "top-p")
REUSES=("True" "False")

log_msg "========== E5: Interaction Effects =========="

for ev in "${EVICTIONS[@]}"; do
    for sel in "${SELECTS[@]}"; do
        for re in "${REUSES[@]}"; do
            log_msg "--- E5: ev=${ev} sel=${sel} reuse=${re} ---"

            # LongBench
            run_longbench "musique"    "${sel}" "${re}" "${ev}" "RetroInfer" "${TOP_P}"
            run_longbench "gov_report" "${sel}" "${re}" "${ev}" "RetroInfer" "${TOP_P}"
            eval_longbench "${sel}" "${re}" "${ev}" "RetroInfer" "${TOP_P}"

            # RULER
            run_ruler "niah_multikey_1" "${sel}" "${re}" "${ev}" "RetroInfer" "${TOP_P}"
            run_ruler "niah_single_1"   "${sel}" "${re}" "${ev}" "RetroInfer" "${TOP_P}"

            # Throughput
            run_throughput "${sel}" "${re}" "${ev}" "${TOP_P}" "${ev}_${sel}_${re}"
        done
    done
done

print_summary
