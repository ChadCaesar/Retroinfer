#!/bin/bash
# ============================================================
# E2: 聚类选择方式消融 (Top-k vs Top-p)
# ============================================================
# Usage:
#   bash benchmark/run_e2_selection.sh
#
# Configurable via env vars:
#   SKIP_THROUGHPUT=1 bash benchmark/run_e2_selection.sh  # skip throughput tests
#   ACCURACY_ONLY=1  bash benchmark/run_e2_selection.sh  # accuracy only, no throughput
# ============================================================

set -euo pipefail
source "$(dirname "$0")/common_exp.sh"
setup_logging "E2_selection"

SELECT_MODES=("top-k" "top-p")
FIXED_POLICY="sclru"

# ============================================================
# LongBench accuracy
# ============================================================
log_msg "========== E2 LongBench =========="

# Attention-diffuse tasks (注意力分散，预期 top-p ≈ top-k)
DIFFUSE_TASKS=("gov_report" "musique" "passage_count")

# Attention-focused tasks (注意力集中，预期 top-p 吞吐 > top-k)
FOCUSED_TASKS=("passage_retrieval_en" "narrativeqa" "qasper")

for mode in "${SELECT_MODES[@]}"; do
    log_msg "--- E2 LongBench: mode=${mode} ---"

    for task in "${DIFFUSE_TASKS[@]}"; do
        run_longbench "${task}" "${mode}" "True" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}"
    done

    for task in "${FOCUSED_TASKS[@]}"; do
        run_longbench "${task}" "${mode}" "True" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}"
    done
done

# Evaluation
for mode in "${SELECT_MODES[@]}"; do
    eval_longbench "${mode}" "True" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}"
done

# ============================================================
# RULER accuracy
# ============================================================
log_msg "========== E2 RULER =========="

SINGLE_TASKS=("niah_single_1" "niah_single_2" "niah_single_3")
MULTI_TASKS=("niah_multikey_1" "niah_multiquery" "niah_multivalue")

for mode in "${SELECT_MODES[@]}"; do
    log_msg "--- E2 RULER: mode=${mode} ---"
    for task in "${SINGLE_TASKS[@]}"; do
        run_ruler "${task}" "${mode}" "True" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}"
    done
    for task in "${MULTI_TASKS[@]}"; do
        run_ruler "${task}" "${mode}" "True" "${FIXED_POLICY}" "RetroInfer" "${TOP_P}"
    done
done

# ============================================================
# Throughput comparison
# ============================================================
if [ "${SKIP_THROUGHPUT:-0}" != "1" ]; then
    log_msg "========== E2 Throughput =========="
    for mode in "${SELECT_MODES[@]}"; do
        run_throughput "${mode}" "True" "${FIXED_POLICY}" "${TOP_P}" "${mode}"
    done
fi

print_summary
