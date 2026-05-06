# !/bin/bash

if [ $# -ne 8 ]; then
    echo "Usage: $0 <model> $1 <attn_type> $2 <budget_ratio> $3 <estimate_ratio> $4 <dtype> $5 <cluster_select> $6 <cluster_reuse> $7 <eviction_policy>"
    exit 1
fi

MODEL=${1}
ATTN_TYPE=${2}
BUDGET_RATIO=${3}
ESTIMATE_RATIO=${4}
DTYPE=${5}
CLUSTER_SELECT=${6}
CLUSTER_REUSE=${7}
EVICTION_POLICY=${8}

RESULT_DIR="./results/pred/${MODEL}/${ATTN_TYPE}"

tasks=(qasper repobench-p lcc gov_report triviaqa)

for task in "${tasks[@]}"; do
    echo "Parameters: ${MODEL} ${task} ${ATTN_TYPE} ${DTYPE} ${BUDGET_RATIO} ${ESTIMATE_RATIO} ${CLUSTER_SELECT} ${CLUSTER_REUSE} ${EVICTION_POLICY}"
    bash pred.sh ${MODEL} ${task} ${ATTN_TYPE} ${DTYPE} ${BUDGET_RATIO} ${ESTIMATE_RATIO} ${CLUSTER_SELECT} ${CLUSTER_REUSE} ${EVICTION_POLICY}
done

echo "Start to evaluate..."
python -u eval.py \
    --attn_type ${ATTN_TYPE} \
    --model ${MODEL} \

echo "Results:"
cat "${RESULT_DIR}/result.json"
