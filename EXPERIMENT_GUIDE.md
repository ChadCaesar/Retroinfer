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

所有实验在 `benchmark/` 下一键运行。

| 脚本 | 实验 | 耗时 |
|------|------|------|
| `run_e1_eviction.sh` | LRU vs SCLRU vs ARC | ~3h |
| `run_e2_selection.sh` | top-k vs top-p | ~2.5h |
| `run_e3_reuse.sh` | Reuse ON vs OFF | ~2.5h |
| `run_e4_topp.sh` | top_p ∈ {0.3,0.4,0.5,0.6} | ~1h |
| `run_e5_reuse_threshold.sh` | threshold ∈ {0.85,0.9,0.95,0.99} | ~2h |

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GPU_TEMP_LIMIT` | `80` | GPU 温度上限 (°C)，超过则等待降温 |
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
# 降低温度阈值
GPU_TEMP_LIMIT=70 bash benchmark/run_e1_eviction.sh

# 只跑 LongBench
LONG_ONLY=1 bash benchmark/run_e1_eviction.sh

# 换模型
MODEL_SHORT=qwen2.5-7b MODEL_PATH=Qwen/Qwen2.5-7B-Instruct bash benchmark/run_e2_selection.sh
```

---

## 3. 运行顺序

E1 → E2 → E3 → E4 → E5。E4 可并行。

```bash
bash benchmark/run_e1_eviction.sh
bash benchmark/run_e2_selection.sh
bash benchmark/run_e3_reuse.sh
bash benchmark/run_e4_topp.sh
bash benchmark/run_e5_reuse_threshold.sh
```

---

## 4. 评测与汇总

### 自动评测

实验脚本已自动调用 eval.py。日志位于 `benchmark/exp_logs/<实验名>_<时间戳>/`。

### 汇总 CSV

```bash
python benchmark/aggregate_results.py
```

生成实验级汇总表（对应 `ablation_experiments.md` 的结果模板）：
- `benchmark/e1_eviction.csv` — E1 策略 × 准确率 × 命中率
- `benchmark/e2_selection.csv` — E2 模式 × 准确率 × 吞吐量 × 延迟
- `benchmark/e3_reuse.csv` — E3 复用 × 分长度吞吐量 × 准确率 × 复用命中率
- `benchmark/e4_topp.csv` — E4 top_p × 准确率 × 吞吐量
- `benchmark/e5_reuse_threshold.csv` — E5 阈值 × musique/gov_report/单针/多针 acc × 复用命中率 × 吞吐量

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

### LongBench

| 内容 | 路径 | 示例 |
|------|------|------|
| 预测结果 | `results/pred/{MODEL}/{ATTN}_{select}_{reuse}_{policy}_{top_p}/{task}.jsonl` | `results/pred/llama-3-8b-1048k/RetroInfer_top-p_True_lru_0.4/musique.jsonl` |
| 评测分数 | 同上目录下 `result.json` | `{"musique": 32.5, "gov_report": 18.2, ...}` |

### RULER

| 内容 | 路径 | 示例 |
|------|------|------|
| 生成数据 | `ruler_eval_result/{MODEL}/synthetic/131072/{ATTN}_{config}/data/{task}.jsonl` | `.../RetroInfer_top-p_True_lru_0.4/data/niah_multikey_1.jsonl` |
| 预测结果 | 同上 `pred/{task}.jsonl` | `.../pred/niah_multikey_1.jsonl` |
| 评测分数 | 同上 `pred/summary.csv` | CSV 表格 |

### Throughput

| 内容 | 路径 |
|------|------|
| 单次日志 | `throughput_eval/different_lengths_logs/{attn}_{len}_bsz{N}_{round}.log` |
| 日志中包含 | `Throughput: XX tokens/s`、`Cache hit rate: X.XXXX`、`Reuse hit rate: X.XXXX` |

### 实验运行日志

| 内容 | 路径 |
|------|------|
| 全部运行记录 | `benchmark/exp_logs/{EXP}_YYYYMMDD_HHMMSS/_run.log` |
| 单次命令输出 | `benchmark/exp_logs/{EXP}_YYYYMMDD_HHMMSS/{desc}.log` |

### 聚合汇总

```bash
python benchmark/aggregate_results.py
```

生成文件（对应 `ablation_experiments.md` 的结果汇总模板）：
| 文件 | 内容 |
|------|------|
| `benchmark/e1_eviction.csv` | E1 策略 × 高/低敏感 LB × 多/单针 RULER × 命中率 |
| `benchmark/e2_selection.csv` | E2 模式 × 分散/集中 LB × 单/多针 RULER × 吞吐量 × 延迟 |
| `benchmark/e3_reuse.csv` | E3 复用 × 分长度吞吐量 × 延迟 × LB/RULER acc × 复用命中率 |
| `benchmark/e4_topp.csv` | E4 top_p × gov_report/passage_retrieval/单针/多针 acc × 吞吐量 |
| `benchmark/e5_reuse_threshold.csv` | E5 阈值 × musique/gov_report/单针/多针 acc × 复用命中率 × 吞吐量 |

---

## 6. 异常中断清理

如果测试中途中断，残留的不完整文件可能导致下次运行跳过该任务。

### 清理单任务（推荐）

删掉中断任务对应的 `.jsonl` 文件。RULER 的测试数据文件无需重生成。

**LongBench 示例**：
```bash
rm benchmark/LongBench/results/pred/llama-3-8b-1048k/RetroInfer_top-p_True_lru_0.4/musique.jsonl
```

**RULER 示例**：
```bash
rm benchmark/ruler/ruler_eval_result/gradientai/Llama-3-8B-Instruct-Gradient-1048k/synthetic/131072/RetroInfer_top-p_True_lru_0.4/pred/niah_multikey_1.jsonl
```

### 清理整次配置

删掉整个配置目录（下次运行会重建）：
```bash
# LongBench
rm -rf benchmark/LongBench/results/pred/llama-3-8b-1048k/RetroInfer_top-p_True_lru_0.4/

# RULER（数据 + 预测 + 评测）
rm -rf benchmark/ruler/ruler_eval_result/gradientai/Llama-3-8B-Instruct-Gradient-1048k/synthetic/131072/RetroInfer_top-p_True_lru_0.4/

# Throughput
rm throughput_eval/different_lengths_logs/retroinfer_*.log
```

### 清理评测缓存

如果预测正确但评测分数异常，只需删掉评测结果重新评估：
```bash
# LongBench
rm benchmark/LongBench/results/pred/llama-3-8b-1048k/RetroInfer_top-p_True_lru_0.4/result.json

# RULER
rm benchmark/ruler/ruler_eval_result/gradientai/Llama-3-8B-Instruct-Gradient-1048k/synthetic/131072/RetroInfer_top-p_True_lru_0.4/pred/summary.csv
```

然后重新运行评测命令（参见第 4 节）。

---

## 7. 可用模型

| MODEL_PATH | 本地目录 |
|-----------|---------|
| `gradientai/Llama-3-8B-Instruct-Gradient-1048k` | `../models/Llama-3-8B-Instruct-Gradient-1048k/` |
| `meta-llama/Llama-3.1-8B-Instruct` | `../models/Llama-3.1-8B-Instruct/` |
| `Qwen/Qwen2.5-7B-Instruct` | `../models/Qwen2.5-7B-Instruct/` |
| `Qwen/Qwen2.5-72B-Instruct` | `../models/Qwen2.5-72B-Instruct/` |
