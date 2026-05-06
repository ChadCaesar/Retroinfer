#!/bin/bash
# ============================================================
# E4: 整体对比 (Full_Flash_Attn vs RetroInfer 最佳配置)
# ============================================================
# Uses ARC + top-p + Reuse ON as the best config (from E1-E3).
# Compares against Full_Flash_Attn baseline on full task suites.
# Usage:
#   bash benchmark/run_e4_overall.sh
# ============================================================

set -euo pipefail
source "$(dirname "$0")/common_exp.sh"
setup_logging "E4_overall"

BEST_POLICY="arc"
BEST_SELECT="top-p"
BEST_REUSE="True"

FULL_ATTN="Full_Flash_Attn"
RETRO="RetroInfer"

# ============================================================
# LongBench full suite
# ============================================================
log_msg "========== E4 LongBench =========="

LONGBENCH_TASKS=("qasper" "repobench-p" "lcc" "gov_report" "triviaqa")

for attn in "${FULL_ATTN}" "${RETRO}"; do
    log_msg "--- E4 LongBench: attn=${attn} ---"
    for task in "${LONGBENCH_TASKS[@]}"; do
        run_longbench "${task}" "${BEST_SELECT}" "${BEST_REUSE}" "${BEST_POLICY}" "${attn}" "${TOP_P}"
    done
done

# Evaluation
eval_longbench "${BEST_SELECT}" "${BEST_REUSE}" "${BEST_POLICY}" "Full_Flash_Attn" "${TOP_P}"
eval_longbench "${BEST_SELECT}" "${BEST_REUSE}" "${BEST_POLICY}" "RetroInfer" "${TOP_P}"

# ============================================================
# RULER full suite (all 13 tasks)
# ============================================================
log_msg "========== E4 RULER =========="

RULER_ALL=(
    "niah_single_1" "niah_single_2" "niah_single_3"
    "niah_multikey_1" "niah_multikey_2" "niah_multikey_3"
    "niah_multivalue" "niah_multiquery"
    "vt" "cwe" "fwe" "qa_1" "qa_2"
)

for attn in "${FULL_ATTN}" "${RETRO}"; do
    log_msg "--- E4 RULER: attn=${attn} ---"
    for task in "${RULER_ALL[@]}"; do
        run_ruler "${task}" "${BEST_SELECT}" "${BEST_REUSE}" "${BEST_POLICY}" "${attn}" "${TOP_P}"
    done
done

# ============================================================
# Throughput comparison
# ============================================================
if [ "${SKIP_THROUGHPUT:-0}" != "1" ]; then
    log_msg "========== E4 Throughput =========="
    run_throughput "${BEST_SELECT}" "${BEST_REUSE}" "${BEST_POLICY}" "RetroInfer_best"
    log_msg "Full Flash Attention throughput is already covered by run_different_lengths.sh"
fi

print_summary
