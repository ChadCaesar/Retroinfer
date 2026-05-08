export CUDA_VISIBLE_DEVICES=0

mkdir -p different_tasks_logs

# Optimization parameters (configurable)
CLUSTER_SELECT="top-p"
CLUSTER_REUSE="True"
EVICTION_POLICY="sclru"
TOP_P="0.4"
RETROINFER_ARGS="--cluster_select ${CLUSTER_SELECT} --cluster_reuse ${CLUSTER_REUSE} --eviction_policy ${EVICTION_POLICY} --top_p ${TOP_P}"

################################ Full Attention ################################
for bsz in 1 4
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --context_len 120000 \
            --task_name fwe \
            --batch_size $bsz > different_tasks_logs/full_attn_fwe_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 1 4
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --context_len 120000 \
            --task_name vt \
            --batch_size $bsz > different_tasks_logs/full_attn_vt_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 1 4
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --context_len 120000 \
            --task_name qa1 \
            --batch_size $bsz > different_tasks_logs/full_attn_qa1_bsz${bsz}_${round}.log 2>&1
    done
done


################################ RetroInfer ################################
# fwe
for bsz in 1
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 120000 \
            --task_name fwe \
            ${RETROINFER_ARGS} --batch_size $bsz > different_tasks_logs/retroinfer_${CLUSTER_SELECT}_${CLUSTER_REUSE}_${EVICTION_POLICY}_${TOP_P}_fwe_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 32
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 120000 \
            --task_name fwe \
            ${RETROINFER_ARGS} --batch_size $bsz > different_tasks_logs/retroinfer_${CLUSTER_SELECT}_${CLUSTER_REUSE}_${EVICTION_POLICY}_${TOP_P}_fwe_bsz${bsz}_${round}.log 2>&1
    done
done

# vt
for bsz in 1
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 120000 \
            --task_name vt \
            ${RETROINFER_ARGS} --batch_size $bsz > different_tasks_logs/retroinfer_${CLUSTER_SELECT}_${CLUSTER_REUSE}_${EVICTION_POLICY}_${TOP_P}_vt_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 32
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 120000 \
            --task_name vt \
            ${RETROINFER_ARGS} --batch_size $bsz > different_tasks_logs/retroinfer_${CLUSTER_SELECT}_${CLUSTER_REUSE}_${EVICTION_POLICY}_${TOP_P}_vt_bsz${bsz}_${round}.log 2>&1
    done
done

# qa1
for bsz in 1
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 120000 \
            --task_name qa1 \
            ${RETROINFER_ARGS} --batch_size $bsz > different_tasks_logs/retroinfer_${CLUSTER_SELECT}_${CLUSTER_REUSE}_${EVICTION_POLICY}_${TOP_P}_qa1_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 32
do
    for round in 1
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 120000 \
            --task_name qa1 \
            ${RETROINFER_ARGS} --batch_size $bsz > different_tasks_logs/retroinfer_${CLUSTER_SELECT}_${CLUSTER_REUSE}_${EVICTION_POLICY}_${TOP_P}_qa1_bsz${bsz}_${round}.log 2>&1
    done
done

unset CUDA_VISIBLE_DEVICES
