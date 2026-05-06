# RetroInfer 消融实验操作指南

实验设计详见 [ablation_experiments.md](ablation_experiments.md)。

---

## 1. 环境准备

### 依赖

```bash
cd Retroinfer
pip install -r requirements.txt
cd library/retroinfer && pip install -e . && cd ../..
```

### 模型放置

代码从 `../models/<model_name最后一段>/` 加载模型（见 [llama.py](model_hub/llama.py#L14)）。

目录结构：
```
项目代码/
├── models/
│   ├── Llama-3-8B-Instruct-Gradient-1048k/
│   └── Qwen2.5-7B-Instruct/
└── Retroinfer/
```

下载模型：
```bash
huggingface-cli download gradientai/Llama-3-8B-Instruct-Gradient-1048k \
    --local-dir ../models/Llama-3-8B-Instruct-Gradient-1048k
```

`MODEL_PATH` 参数只是用来提取 `split("/")[-1]` 作为目录名，前缀可任意。

### 验证

```bash
python simple_test.py \
    --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
    --attn_type RetroInfer --batch_size 1 --gen_len 200 \
    --cluster_select top-p --cluster_reuse True --eviction_policy sclru --top_p 0.4
```

预期日志包含 `Cache hit rate` 和 `Reuse hit rate`。

---

## 2. 实验脚本

所有实验在 `benchmark/` 下一键运行。每次 prefill+decode 之间默认冷却 30 秒，超过 80°C 自动等待降温。

| 脚本 | 实验 | 耗时 |
|------|------|------|
| `run_e1_eviction.sh` | LRU vs SCLRU vs ARC | ~3h |
| `run_e2_selection.sh` | top-k vs top-p | ~2.5h |
| `run_e3_reuse.sh` | Reuse ON vs OFF | ~2.5h |
| `run_e2b_topp.sh` | top_p ∈ {0.3,0.4,0.5,0.6} | ~1h |
| `run_e3b_interact.sh` | 2×2×2 交互效应 | ~2h |
| `run_e4_overall.sh` | Full_Flash_Attn vs RetroInfer | ~3h |

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SAMPLE_COOLDOWN` | `30` | 样本间冷却秒数 |
| `GPU_TEMP_LIMIT` | `80` | GPU 温度上限 (°C) |
| `MODEL_SHORT` | `llama-3-8b-1048k` | LongBench 短名称 |
| `MODEL_PATH` | `gradientai/Llama-3-8B-Instruct-Gradient-1048k` | RULER 路径名 |
| `BUDGET_RATIO` | `0.018` | 检索预算 |
| `ESTIMATE_RATIO` | `0.25` | 估计区比例 |
| `TOP_P` | `0.4` | top-p 阈值 |
| `LONG_ONLY` | `0` | 只跑 LongBench |
| `RULER_ONLY` | `0` | 只跑 RULER |
| `SKIP_THROUGHPUT` | `0` | 跳过吞吐量 |

### 自定义运行

```bash
# 调长冷却
SAMPLE_COOLDOWN=60 bash benchmark/run_e1_eviction.sh

# 只跑 LongBench
LONG_ONLY=1 bash benchmark/run_e1_eviction.sh

# 换模型
MODEL_SHORT=qwen2.5-7b MODEL_PATH=Qwen/Qwen2.5-7B-Instruct bash benchmark/run_e2_selection.sh
```

---

## 3. 运行顺序

E1 → E2 → E3 → E3b → 确定最佳配置 → E4。E2b 可并行。

```bash
bash benchmark/run_e1_eviction.sh
bash benchmark/run_e2_selection.sh
bash benchmark/run_e3_reuse.sh
bash benchmark/run_e3b_interact.sh
# 确定最佳配置后：
bash benchmark/run_e4_overall.sh
```

---

## 4. 评测与汇总

### 自动评测

实验脚本已自动调用 eval.py。日志位于 `benchmark/exp_logs/<实验名>_<时间戳>/`。

### 汇总 CSV

```bash
python benchmark/aggregate_results.py
```

生成三个文件：
- `benchmark/throughput_summary.csv` — 吞吐量（含 mean/std）
- `benchmark/longbench_summary.csv` — LongBench 各任务得分
- `benchmark/ruler_summary.csv` — RULER 各任务得分

### 手动评测

```bash
cd benchmark/LongBench/
python eval.py --model llama-3-8b-1048k --attn_type RetroInfer \
    --cluster_select top-p --cluster_reuse True --eviction_policy lru --top_p 0.4

cd benchmark/ruler/
python eval/evaluate.py \
    --data_dir ./ruler_eval_result/gradientai/Llama-3-8B-Instruct-Gradient-1048k/synthetic/131072/RetroInfer_top-p_True_lru_0.4/pred \
    --benchmark synthetic
```

### 关键指标

| 指标 | 日志输出 |
|------|---------|
| 准确率 | `result.json` / `summary.csv` |
| 吞吐量 | `Throughput: XX tokens/s` |
| 延迟 | `Decoding latency: XX ms/step` |
| 缓存命中率 | `Cache hit rate: X.XXXX` |
| 复用命中率 | `Reuse hit rate: X.XXXX` |

---

## 5. 结果路径

```
LongBench:  results/pred/{MODEL}/{ATTN_TYPE}_{select}_{reuse}_{policy}_{top_p}/{task}.jsonl
RULER:      ruler_eval_result/{MODEL_PATH}/synthetic/131072/{ATTN_TYPE}_{select}_{reuse}_{policy}_{top_p}/pred/
Throughput: throughput_eval/different_lengths_logs/{attn}_{len}_bsz{N}_{round}.log
Exp log:    benchmark/exp_logs/{EXP}_YYYYMMDD_HHMMSS/
```

---

## 6. 可用模型

| MODEL_PATH | 本地目录 |
|-----------|---------|
| `gradientai/Llama-3-8B-Instruct-Gradient-1048k` | `../models/Llama-3-8B-Instruct-Gradient-1048k/` |
| `meta-llama/Llama-3.1-8B-Instruct` | `../models/Llama-3.1-8B-Instruct/` |
| `Qwen/Qwen2.5-7B-Instruct` | `../models/Qwen2.5-7B-Instruct/` |
| `Qwen/Qwen2.5-72B-Instruct` | `../models/Qwen2.5-72B-Instruct/` |
