export CUDA_VISIBLE_DEVICES=0

mkdir -p different_lengths_logs

# Optimization parameters (configurable)
CLUSTER_SELECT="top-p"
CLUSTER_REUSE="True"
EVICTION_POLICY="sclru"
TOP_P="0.4"
REUSE_THRESHOLD="0.95"
RETROINFER_ARGS="--cluster_select ${CLUSTER_SELECT} --cluster_reuse ${CLUSTER_REUSE} --eviction_policy ${EVICTION_POLICY} --top_p ${TOP_P} --reuse_threshold ${REUSE_THRESHOLD}"

################################ Full Attention ################################
for bsz in 1 2 4 8
do
    for round in 1 2 3
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --context_len 60000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/full_attn_60k_bsz${bsz}_${round}.log 2>&1
    done
done


for bsz in 1 2 4
do
    for round in 1 2 3
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --context_len 120000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/full_attn_120k_bsz${bsz}_${round}.log 2>&1
    done
done


for bsz in 1 2
do
    for round in 1 2 3
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --context_len 240000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/full_attn_240k_bsz${bsz}_${round}.log 2>&1
    done
done


for bsz in 1
do
    for round in 1 2 3
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type Full_Flash_Attn \
            --context_len 480000 \
            --task_name NIAH \
            --batch_size $bsz > different_lengths_logs/full_attn_480k_bsz${bsz}_${round}.log 2>&1
    done
done


################################ RetroInfer ################################
# 60K
for bsz in 1 2 4 8 16 32
do
    for round in 1 2 3
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 60000 \
            --task_name NIAH \
            ${RETROINFER_ARGS} --batch_size $bsz > different_lengths_logs/retroinfer_${CLUSTER_SELECT}_${CLUSTER_REUSE}_${EVICTION_POLICY}_${TOP_P}_60k_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 64
do
    for round in 1 2 3
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 60000 \
            --task_name NIAH \
            ${RETROINFER_ARGS} --batch_size $bsz > different_lengths_logs/retroinfer_${CLUSTER_SELECT}_${CLUSTER_REUSE}_${EVICTION_POLICY}_${TOP_P}_60k_bsz${bsz}_${round}.log 2>&1
    done
done

# 120K
for bsz in 1 2 4 8 16
do
    for round in 1 2 3
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 120000 \
            --task_name NIAH \
            ${RETROINFER_ARGS} --batch_size $bsz > different_lengths_logs/retroinfer_${CLUSTER_SELECT}_${CLUSTER_REUSE}_${EVICTION_POLICY}_${TOP_P}_120k_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 32
do
    for round in 1 2 3
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 120000 \
            --task_name NIAH \
            ${RETROINFER_ARGS} --batch_size $bsz > different_lengths_logs/retroinfer_${CLUSTER_SELECT}_${CLUSTER_REUSE}_${EVICTION_POLICY}_${TOP_P}_120k_bsz${bsz}_${round}.log 2>&1
    done
done

# 240K
for bsz in 1 2 4 8
do
    for round in 1 2 3
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 240000 \
            --task_name NIAH \
            ${RETROINFER_ARGS} --batch_size $bsz > different_lengths_logs/retroinfer_${CLUSTER_SELECT}_${CLUSTER_REUSE}_${EVICTION_POLICY}_${TOP_P}_240k_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 16
do
    for round in 1 2 3
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 240000 \
            --task_name NIAH \
            ${RETROINFER_ARGS} --batch_size $bsz > different_lengths_logs/retroinfer_${CLUSTER_SELECT}_${CLUSTER_REUSE}_${EVICTION_POLICY}_${TOP_P}_240k_bsz${bsz}_${round}.log 2>&1
    done
done

# 480K
for bsz in 1 2 4
do
    for round in 1 2 3
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 480000 \
            --task_name NIAH \
            ${RETROINFER_ARGS} --batch_size $bsz > different_lengths_logs/retroinfer_${CLUSTER_SELECT}_${CLUSTER_REUSE}_${EVICTION_POLICY}_${TOP_P}_480k_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 8
do
    for round in 1 2 3
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 480000 \
            --task_name NIAH \
            ${RETROINFER_ARGS} --batch_size $bsz > different_lengths_logs/retroinfer_${CLUSTER_SELECT}_${CLUSTER_REUSE}_${EVICTION_POLICY}_${TOP_P}_480k_bsz${bsz}_${round}.log 2>&1
    done
done

# 1024K
for bsz in 1 2
do
    for round in 1 2 3
    do
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 1024000 \
            --task_name NIAH \
            ${RETROINFER_ARGS} --batch_size $bsz > different_lengths_logs/retroinfer_${CLUSTER_SELECT}_${CLUSTER_REUSE}_${EVICTION_POLICY}_${TOP_P}_1024k_bsz${bsz}_${round}.log 2>&1
    done
done

for bsz in 4
do
    for round in 1 2 3
    do
        numactl --cpunodebind=0 --membind=0,1 python -u test.py \
            --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
            --attn_type RetroInfer \
            --context_len 1024000 \
            --task_name NIAH \
            ${RETROINFER_ARGS} --batch_size $bsz > different_lengths_logs/retroinfer_${CLUSTER_SELECT}_${CLUSTER_REUSE}_${EVICTION_POLICY}_${TOP_P}_1024k_bsz${bsz}_${round}.log 2>&1
    done
done

unset CUDA_VISIBLE_DEVICES
