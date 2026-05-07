#!/bin/bash
# Common utilities for RetroInfer ablation experiment scripts.
# Source this file in experiment scripts: source "$(dirname "$0")/common_exp.sh"

# ============================================================
# Configurable defaults (override via environment variables)
# ============================================================
MODEL_SHORT="${MODEL_SHORT:-llama-3-8b-1048k}"
MODEL_PATH="${MODEL_PATH:-gradientai/Llama-3-8B-Instruct-Gradient-1048k}"
DTYPE="${DTYPE:-fp16}"
BUDGET_RATIO="${BUDGET_RATIO:-0.018}"
ESTIMATE_RATIO="${ESTIMATE_RATIO:-0.25}"
TOP_P="${TOP_P:-0.4}"
RULER_CONTEXT="${RULER_CONTEXT:-131072}"
GPU_TEMP_LIMIT="${GPU_TEMP_LIMIT:-80}"              # wait until temp drops below this
export GPU_TEMP_LIMIT                              # pass to pred.sh / ruler_run.sh

# Workspace roots
BENCHMARK_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "${BENCHMARK_DIR}")"
LOG_BASE_DIR="${LOG_BASE_DIR:-${BENCHMARK_DIR}/exp_logs}"

# ============================================================
# Logging
# ============================================================
setup_logging() {
    local exp_name="$1"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="${LOG_BASE_DIR}/${exp_name}_${TIMESTAMP}"
    mkdir -p "${LOG_DIR}"
    echo "Experiment logs: ${LOG_DIR}"
}

log_msg() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG_DIR}/_run.log"
}

# ============================================================
# Run a command with logging
# ============================================================
run_cmd() {
    local desc="$1"
    shift
    log_msg ""
    log_msg "============================================"
    log_msg "RUN: ${desc}"
    log_msg "CMD: $*"
    log_msg "============================================"

    local log_name
    log_name=$(echo "${desc}" | tr ' ' '_' | tr -d '()' | tr '/' '-')
    local log_file="${LOG_DIR}/${log_name}.log"

    "$@" > "${log_file}" 2>&1
    local ret=$?

    if [ $ret -ne 0 ]; then
        log_msg "FAILED (exit=${ret}): ${desc}"
        log_msg "Log: ${log_file}"
        # Show last 10 lines on failure
        tail -10 "${log_file}" | while read -r line; do
            log_msg "  | ${line}"
        done
    else
        log_msg "OK: ${desc}"
    fi
    return $ret
}

# ============================================================
# LongBench runner — calls pred.sh with consistent params
# ============================================================
run_longbench() {
    local task="$1"
    local cluster_select="$2"
    local cluster_reuse="$3"
    local eviction_policy="$4"
    local attn_type="${5:-RetroInfer}"
    local top_p="${6:-${TOP_P}}"

    local desc="LB_${task}_${attn_type}_${cluster_select}_${cluster_reuse}_${eviction_policy}_${top_p}"
    run_cmd "${desc}" \
        bash "${BENCHMARK_DIR}/LongBench/pred.sh" \
            "${MODEL_SHORT}" "${task}" "${attn_type}" "${DTYPE}" \
            "${BUDGET_RATIO}" "${ESTIMATE_RATIO}" \
            "${cluster_select}" "${cluster_reuse}" "${eviction_policy}" "${top_p}"
}

# ============================================================
# RULER runner — calls ruler_run.sh with consistent params
# ============================================================
run_ruler() {
    local task="$1"
    local cluster_select="$2"
    local cluster_reuse="$3"
    local eviction_policy="$4"
    local attn_type="${5:-RetroInfer}"
    local top_p="${6:-${TOP_P}}"

    local desc="RULER_${task}_${attn_type}_${cluster_select}_${cluster_reuse}_${eviction_policy}_${top_p}"
    run_cmd "${desc}" \
        bash "${BENCHMARK_DIR}/ruler/ruler_run.sh" \
            "${MODEL_SHORT}" "synthetic" "${attn_type}" "${RULER_CONTEXT}" "${task}" \
            "${DTYPE}" "${BUDGET_RATIO}" "${ESTIMATE_RATIO}" \
            "${cluster_select}" "${cluster_reuse}" "${eviction_policy}" "${top_p}"
}

# ============================================================
# LongBench evaluation helper
# ============================================================
eval_longbench() {
    local cluster_select="$1"
    local cluster_reuse="$2"
    local eviction_policy="$3"
    local attn_type="${4:-RetroInfer}"
    local top_p="${5:-${TOP_P}}"

    local desc="eval_LB_${attn_type}_${cluster_select}_${cluster_reuse}_${eviction_policy}_${top_p}"
    run_cmd "${desc}" \
        python -u "${BENCHMARK_DIR}/LongBench/eval.py" \
            --model "${MODEL_SHORT}" --attn_type "${attn_type}" \
            --cluster_select "${cluster_select}" --cluster_reuse "${cluster_reuse}" \
            --eviction_policy "${eviction_policy}" --top_p "${top_p}"
}

# ============================================================
# Throughput test helper — patches run_different_lengths.sh and runs it
# ============================================================
run_throughput() {
    local cluster_select="$1"
    local cluster_reuse="$2"
    local eviction_policy="$3"
    local top_p="${4:-${TOP_P}}"
    local desc="${5:-throughput}"

    local tp_script="${PROJECT_DIR}/throughput_eval/run_different_lengths.sh"
    if [ ! -f "${tp_script}" ]; then
        log_msg "Throughput script not found: ${tp_script} (skipping)"
        return 1
    fi

    local tmp_script="${LOG_DIR}/run_tp_${desc}.sh"
    sed -e "s/^CLUSTER_SELECT=.*/CLUSTER_SELECT=\"${cluster_select}\"/" \
        -e "s/^CLUSTER_REUSE=.*/CLUSTER_REUSE=\"${cluster_reuse}\"/" \
        -e "s/^EVICTION_POLICY=.*/EVICTION_POLICY=\"${eviction_policy}\"/" \
        -e "s/^TOP_P=.*/TOP_P=\"${top_p}\"/" \
        "${tp_script}" > "${tmp_script}"
    chmod +x "${tmp_script}"
    run_cmd "throughput_${desc}" bash "${tmp_script}"
}

# ============================================================
# Print experiment summary at the end
# ============================================================
print_summary() {
    log_msg ""
    log_msg "============================================"
    log_msg "EXPERIMENT COMPLETE"
    log_msg "============================================"
    log_msg "Total runs: $(grep -c '^RUN:' "${LOG_DIR}/_run.log" 2>/dev/null || echo 0)"
    log_msg "Failures:   $(grep -c 'FAILED' "${LOG_DIR}/_run.log" 2>/dev/null || echo 0)"
    log_msg "Log directory: ${LOG_DIR}"
    log_msg ""
    log_msg "To evaluate LongBench results:"
    log_msg "  cd ${BENCHMARK_DIR}/LongBench"
    log_msg "  python eval.py --model ${MODEL_SHORT} --attn_type RetroInfer --cluster_select <...> --cluster_reuse <...> --eviction_policy <...> --top_p <...>"
    log_msg ""
    log_msg "To evaluate RULER results:"
    log_msg "  cd ${BENCHMARK_DIR}/ruler"
    log_msg "  python eval/evaluate.py --data_dir <pred_dir> --benchmark synthetic"
    log_msg ""
    log_msg "To aggregate all results:"
    log_msg "  python ${BENCHMARK_DIR}/aggregate_results.py --output ${LOG_DIR}/summary.csv"
    log_msg ""
}
