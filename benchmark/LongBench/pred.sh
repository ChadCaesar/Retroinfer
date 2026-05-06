# !/bin/bash

if [ $# -ne 9 ]; then
    echo "Usage: $0 <model_name> $1 <task_name> $2 <attn_type> $3 <dtype> $4 <budget_ratio> $5 <estimate_ratio> $6 <cluster_select> $7 <cluster_reuse> $8 <eviction_policy>"
    exit 1
fi

NUM_EXAMPLES=-1
MODEL=${1}
TASK=${2}
ATTN_TYPE=${3}
DTYPE=${4}
BUDGET_RATIO=${5}
ESTIMATE_RATIO=${6}
CLUSTER_SELECT=${7}
CLUSTER_REUSE=${8}
EVICTION_POLICY=${9}

RESULT_DIR="./results/pred/${MODEL}/${ATTN_TYPE}"
RESULT_DIR_E="./results/pred_e/${MODEL}/${ATTN_TYPE}"

echo "remove previous result file..."
rm -f "${RESULT_DIR}/${TASK}.jsonl"
rm -f "${RESULT_DIR_E}/${TASK}.jsonl"

echo "Start to predict..."
python -u pred.py \
    --task ${TASK} \
    --attn_type ${ATTN_TYPE} \
    --model ${MODEL} \
    --dtype ${DTYPE} \
    --device auto \
    --budget_ratio ${BUDGET_RATIO} \
    --estimate_ratio ${ESTIMATE_RATIO} \
    --cluster_select ${CLUSTER_SELECT} \
    --cluster_reuse ${CLUSTER_REUSE} \
    --eviction_policy ${EVICTION_POLICY} \
    --num_examples ${NUM_EXAMPLES}
