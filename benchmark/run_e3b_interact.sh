#!/bin/bash
# ============================================================
# E3b: 参数交互效应验证 (eviction × cluster_select × cluster_reuse)
# ============================================================
# 2×2×2 = 8 combinations sampled on 4 tasks
# Usage:
#   bash benchmark/run_e3b_interact.sh
# ============================================================

set -euo pipefail
source "$(dirname "$0")/common_exp.sh"
setup_logging "E3b_interact"

# Key design: test LRU vs ARC (extremes), top-k vs top-p, reuse ON vs OFF
EVICTIONS=("lru" "arc")
SELECTS=("top-k" "top-p")
REUSES=("True" "False")

log_msg "========== E3b: Interaction Effects =========="

for ev in "${EVICTIONS[@]}"; do
    for sel in "${SELECTS[@]}"; do
        for re in "${REUSES[@]}"; do
            tp="${TOP_P}"
            log_msg "--- E3b: ev=${ev} sel=${sel} reuse=${re} ---"

            # LongBench
            run_longbench "musique"    "${sel}" "${re}" "${ev}" "RetroInfer" "${tp}"
            run_longbench "gov_report" "${sel}" "${re}" "${ev}" "RetroInfer" "${tp}"

            # RULER
            run_ruler "niah_multikey_1" "${sel}" "${re}" "${ev}" "RetroInfer" "${tp}"
            run_ruler "niah_single_1"   "${sel}" "${re}" "${ev}" "RetroInfer" "${tp}"

            # Eval each combination individually
            eval_longbench "${sel}" "${re}" "${ev}" "RetroInfer" "${tp}"
        done
    done
done

print_summary
