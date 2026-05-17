#!/bin/bash
# ============================================================
# E0: 基线 vs 全优化对比
# 对比 Baseline (top-k + 不复用 + LRU) 和
#       Optimized (top-p + 复用 + SCLRU) 在 10 个 LongBench 任务上的表现
# 作为所有消融实验的起点和全局参照
# ============================================================
# Usage:
#   bash benchmark/run_e0_baseline.sh
#
# Configurable via env vars:
#   GPU_TEMP_LIMIT=70  bash benchmark/run_e0_baseline.sh
# ============================================================

set -euo pipefail
source "$(dirname "$0")/common_exp.sh"
setup_logging "E0_baseline"

# 两种配置：基线（无优化） vs 全优化
CONFIGS=(
    "top-k False lru"      # baseline
    "top-p True sclru"     # optimized
)

# 10 个代表性 LongBench 任务，覆盖多种推理类型
LB_TASKS=(
    "musique"               # 多跳推理
    "gov_report"            # 长文档摘要（注意力分散）
    "passage_count"         # 跨段落计数（缓存压力大）
    "passage_retrieval_en"  # 段落检索（注意力集中）
    "trec"                  # 文本分类
    "triviaqa"              # 短问答
    "narrativeqa"           # 叙事性长问答
    "qasper"                # 论文级长问答
    "lcc"                   # 代码补全
    "qmsum"                 # 会议摘要
)

log_msg "========== E0: Baseline vs Optimized =========="

for config_str in "${CONFIGS[@]}"; do
    read -r select reuse policy <<< "${config_str}"
    log_msg "--- Config: ${select} / reuse=${reuse} / ${policy} ---"

    for task in "${LB_TASKS[@]}"; do
        run_longbench "${task}" "${select}" "${reuse}" "${policy}" "RetroInfer" "${TOP_P}" "${REUSE_THRESHOLD}"
    done

    eval_longbench "${select}" "${reuse}" "${policy}" "RetroInfer" "${TOP_P}" "${REUSE_THRESHOLD}"
done

print_summary
