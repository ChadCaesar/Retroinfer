# RetroInfer 消融实验方案

## 实验总览

| 实验编号 | 消融对象 | 固定配置 | 变化量 | 主要指标 | 次要指标 |
|---------|---------|---------|--------|---------|---------|
| E1 | 缓存替换策略 | top-p + 复用 | LRU / SCLRU / ARC | 命中率 + 准确率 | 吞吐量 |
| E2 | 聚类选择方式 | SCLRU + 复用 | top-k / top-p | 吞吐量 + 准确率 | 命中率 |
| E3 | 聚类复用 | SCLRU + top-p | True / False | 吞吐量 / 延迟 | 准确率 |
| E4 | 整体对比 | 最佳配置 | Full_Flash_Attn / RetroInfer | 准确率 + 吞吐量 | 命中率 |

**公共参数（所有实验固定）**: `budget_ratio=0.018`, `estimate_ratio=0.25`, `dtype=fp16`, `model=llama-3-8b-1048k`

---

## 关于 NIAH（大海捞针）

RULER 提供了系统的 NIAH（Needle In A Haystack）测试，所有 NIAH 任务均在 **131072 tokens** 的长上下文中检索极少量的关键信息。这与 RetroInfer 的核心机制（通过 k-means 聚类 + 质心检索来定位关键 token）**高度相关**——检索到了正确的聚类，就找到了"针"。

| RULER 任务 | 针的数量 | 特点 | 对应测试意图 |
|-----------|---------|------|-------------|
| `niah_single_1` | 1 个 key, 1 个 value | repeat haystack | 基础大海捞针（最简单） |
| `niah_single_2` | 1 个 key, 1 个 value | essay haystack | 自然文本大海捞针 |
| `niah_single_3` | 1 个 key, 1 个 value | essay + UUID | 最困难（无语义线索） |
| `niah_multikey_1` | **4 个 key** | essay haystack | 多针检索，需同时定位 4 处 |
| `niah_multikey_2` | 1 个 key | **needle 噪声** haystack | 干扰针（其他 needle 作为干草堆） |
| `niah_multikey_3` | 1 个 key | **needle 噪声 + UUID** | 干扰针且无语义线索 |
| `niah_multivalue` | 1 个 key, **4 个 value** | essay haystack | 关联多值检索 |
| `niah_multiquery` | 1 个 key, **4 个 query** | essay haystack | 多次查询同一上下文 |

此外还有非 NIAH 的 RULER 任务：`vt`（变量追踪，4 跳链式追踪）、`cwe`（常见词提取）、`fwe`（高频词提取）、`qa_1`（SQuAD 问答）、`qa_2`（HotpotQA 问答）。

---

## E1: 缓存替换策略消融 (LRU vs SCLRU vs ARC)

### 目的

量化不同缓存替换策略对命中率和准确率的提升。ARC 维护 4 个链表自适应平衡 recency/frequency，理论上 ARC ≥ SCLRU ≥ LRU。

策略差异在**需要同时维系多个分散聚类**的场景最显著：当 GPU 缓存放不下所有热点聚类时，更好的替换策略能保留真正需要的聚类，减少 CPU→GPU 的重新加载。

### LongBench 测试任务

**高敏感度**（信息分散，需频繁跨段访问，缓存压力大）：

| 任务 | 类型 | 为何敏感 |
|------|------|---------|
| `hotpotqa`, `2wikimqa`, `musique` | 多跳推理 | 需在多个证据段落间反复跳转，访问模式分散 |
| `gov_report`, `qmsum`, `multi_news` | 长文本摘要 | 信息分布全文，需整合多处理解 |
| `passage_count` | 段落去重计数 | 跨段落反复比较 |

**低敏感度对照组**（注意力集中，缓存压力小）：

| 任务 | 类型 | 为何不敏感 |
|------|------|-----------|
| `passage_retrieval_en` | 段落定位 | 找到目标段落即可，检索集中 |
| `trec` | 文本分类 | 依赖整体语义，非特定信息定位 |
| `triviaqa` | 短问答 | 单点提取 |

### RULER 测试任务

**高敏感度**（多针/多查询，需同时保持多个检索目标）：

| 任务 | 为何敏感 |
|------|---------|
| `niah_multikey_1` | 4 个不同的 key 分散在 128K 上下文中，需同时定位 4 个聚类区域 |
| `niah_multiquery` | 4 次查询访问同一上下文的不同位置，类似"重访"模式 |
| `niah_multivalue` | 1 个 key 对应 4 个 value，需关联查找多个位置 |
| `vt` | 变量追踪需在长链中连续 hop，缓存中保持中间变量位置很重要 |

**低敏感度对照组**（单针，检索集中）：

| 任务 | 为何不敏感 |
|------|-----------|
| `niah_single_1` | 仅 1 个针，检索高度集中，缓存压力极小 |

### 运行命令

#### Step 1 — LongBench 高敏感度任务，LRU

```bash
cd benchmark/LongBench/

bash pred.sh llama-3-8b-1048k hotpotqa      RetroInfer fp16 0.018 0.25 top-p True lru
bash pred.sh llama-3-8b-1048k 2wikimqa      RetroInfer fp16 0.018 0.25 top-p True lru
bash pred.sh llama-3-8b-1048k musique       RetroInfer fp16 0.018 0.25 top-p True lru
bash pred.sh llama-3-8b-1048k gov_report    RetroInfer fp16 0.018 0.25 top-p True lru
bash pred.sh llama-3-8b-1048k qmsum         RetroInfer fp16 0.018 0.25 top-p True lru
bash pred.sh llama-3-8b-1048k multi_news    RetroInfer fp16 0.018 0.25 top-p True lru
bash pred.sh llama-3-8b-1048k passage_count RetroInfer fp16 0.018 0.25 top-p True lru
```

#### Step 2 — LongBench 高敏感度任务，SCLRU

```bash
cd benchmark/LongBench/

bash pred.sh llama-3-8b-1048k hotpotqa      RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k 2wikimqa      RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k musique       RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k gov_report    RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k qmsum         RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k multi_news    RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k passage_count RetroInfer fp16 0.018 0.25 top-p True sclru
```

#### Step 3 — LongBench 高敏感度任务，ARC

```bash
cd benchmark/LongBench/

bash pred.sh llama-3-8b-1048k hotpotqa      RetroInfer fp16 0.018 0.25 top-p True arc
bash pred.sh llama-3-8b-1048k 2wikimqa      RetroInfer fp16 0.018 0.25 top-p True arc
bash pred.sh llama-3-8b-1048k musique       RetroInfer fp16 0.018 0.25 top-p True arc
bash pred.sh llama-3-8b-1048k gov_report    RetroInfer fp16 0.018 0.25 top-p True arc
bash pred.sh llama-3-8b-1048k qmsum         RetroInfer fp16 0.018 0.25 top-p True arc
bash pred.sh llama-3-8b-1048k multi_news    RetroInfer fp16 0.018 0.25 top-p True arc
bash pred.sh llama-3-8b-1048k passage_count RetroInfer fp16 0.018 0.25 top-p True arc
```

#### Step 4 — LongBench 低敏感度对照组，LRU

```bash
cd benchmark/LongBench/

bash pred.sh llama-3-8b-1048k passage_retrieval_en RetroInfer fp16 0.018 0.25 top-p True lru
bash pred.sh llama-3-8b-1048k trec                 RetroInfer fp16 0.018 0.25 top-p True lru
bash pred.sh llama-3-8b-1048k triviaqa             RetroInfer fp16 0.018 0.25 top-p True lru
```

#### Step 5 — LongBench 低敏感度对照组，SCLRU

```bash
cd benchmark/LongBench/

bash pred.sh llama-3-8b-1048k passage_retrieval_en RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k trec                 RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k triviaqa             RetroInfer fp16 0.018 0.25 top-p True sclru
```

#### Step 6 — LongBench 低敏感度对照组，ARC

```bash
cd benchmark/LongBench/

bash pred.sh llama-3-8b-1048k passage_retrieval_en RetroInfer fp16 0.018 0.25 top-p True arc
bash pred.sh llama-3-8b-1048k trec                 RetroInfer fp16 0.018 0.25 top-p True arc
bash pred.sh llama-3-8b-1048k triviaqa             RetroInfer fp16 0.018 0.25 top-p True arc
```

#### Step 7 — RULER 高敏感度，LRU

```bash
cd benchmark/ruler/

bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multikey_1 fp16 0.018 0.25 top-p True lru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multiquery fp16 0.018 0.25 top-p True lru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multivalue fp16 0.018 0.25 top-p True lru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 vt              fp16 0.018 0.25 top-p True lru
```

#### Step 8 — RULER 高敏感度，SCLRU

```bash
cd benchmark/ruler/

bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multikey_1 fp16 0.018 0.25 top-p True sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multiquery fp16 0.018 0.25 top-p True sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multivalue fp16 0.018 0.25 top-p True sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 vt              fp16 0.018 0.25 top-p True sclru
```

#### Step 9 — RULER 高敏感度，ARC

```bash
cd benchmark/ruler/

bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multikey_1 fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multiquery fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multivalue fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 vt              fp16 0.018 0.25 top-p True arc
```

#### Step 10 — RULER 低敏感度对照组

```bash
cd benchmark/ruler/

bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_single_1 fp16 0.018 0.25 top-p True lru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_single_1 fp16 0.018 0.25 top-p True sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_single_1 fp16 0.018 0.25 top-p True arc
```

#### 数据收集

每个 run 的日志中自动输出缓存命中率：
```
Cache hit rate: 0.7234 (hits=15234, misses=5821)
```

LongBench 评测：
```bash
cd benchmark/LongBench/
python eval.py --model llama-3-8b-1048k --attn_type RetroInfer
```

RULER 评测（各任务分别生成 `summary.csv`）：
```bash
cd benchmark/ruler/
# 评测脚本会自动读取 pred 目录下各任务的预测结果
python eval/evaluate.py --data_dir ./ruler_eval_result/gradientai/Llama-3-8B-Instruct-Gradient-1048k/synthetic/131072/RetroInfer/pred --benchmark synthetic
```

### 预期结果

| 指标 | LRU | SCLRU | ARC |
|------|-----|-------|-----|
| 高敏感 LongBench 准确率 | 基线 | ↑ | ↑↑ |
| 低敏感 LongBench 准确率 | — | ≈LRU | ≈LRU |
| 多针 RULER 准确率 | 基线 | ↑ | ↑↑ |
| 单针 RULER 准确率 | — | ≈LRU | ≈LRU |
| 缓存命中率 | 基线 | ↑ | ↑↑ |

关键验证点：ARC 在高敏感任务上的命中率和准确率应显著优于 LRU；低敏感/单针任务三策略无差异→策略无副作用。

---

## E2: 聚类选择方式消融 (Top-k vs Top-p)

### 目的

Top-k 固定检索 `nprobe` 个聚类，Top-p 根据 query-centroid 注意力分布的累积概率动态决定检索数量。核心假设：Top-p 在**注意力集中时**自动减少检索（提升吞吐），在**注意力分散时**自动增加检索（保持准确率）。

### LongBench 测试任务

**注意力分散**（预期 Top-p 检索量 ≈ Top-k，准确率不降）：

| 任务 | 原因 |
|------|------|
| `gov_report`, `qmsum`, `multi_news` | 摘要需覆盖全文，注意力分散到大量聚类 |
| `hotpotqa`, `2wikimqa`, `musique` | 多跳推理需关联多个不连续段落 |
| `passage_count` | 跨段落逐段比较 |

**注意力集中**（预期 Top-p 检索量 < Top-k，吞吐提升）：

| 任务 | 原因 |
|------|------|
| `passage_retrieval_en` | 定位单个段落 |
| `narrativeqa`, `qasper` | 长文中找特定答案，信息局部化 |
| `trec`, `triviaqa` | 短文本任务 |

### RULER 测试任务

**注意力集中**（预期 Top-p 检索量 < Top-k，吞吐提升）：

| 任务 | 原因 |
|------|------|
| `niah_single_1` | 单针检索，query 聚焦于极少几个聚类 |
| `niah_single_2` | 同上，自然文本背景 |
| `niah_single_3` | UUID 版本，最难的语义检索但仍是单针 |

**注意力分散**（预期 Top-p 检索量 ≈ Top-k，准确率不降）：

| 任务 | 原因 |
|------|------|
| `niah_multikey_1` | 4 个针分散在全文，注意力需要覆盖多个位置 |
| `niah_multiquery` | 4 个不同查询，注意力分布更均匀 |
| `niah_multivalue` | 1 key 关联 4 个分散的 value |

### 运行命令

#### Step 1 — LongBench 注意力分散任务，Top-k

```bash
cd benchmark/LongBench/

bash pred.sh llama-3-8b-1048k hotpotqa      RetroInfer fp16 0.018 0.25 top-k True sclru
bash pred.sh llama-3-8b-1048k 2wikimqa      RetroInfer fp16 0.018 0.25 top-k True sclru
bash pred.sh llama-3-8b-1048k musique       RetroInfer fp16 0.018 0.25 top-k True sclru
bash pred.sh llama-3-8b-1048k gov_report    RetroInfer fp16 0.018 0.25 top-k True sclru
bash pred.sh llama-3-8b-1048k qmsum         RetroInfer fp16 0.018 0.25 top-k True sclru
bash pred.sh llama-3-8b-1048k multi_news    RetroInfer fp16 0.018 0.25 top-k True sclru
bash pred.sh llama-3-8b-1048k passage_count RetroInfer fp16 0.018 0.25 top-k True sclru
```

#### Step 2 — LongBench 注意力分散任务，Top-p

```bash
cd benchmark/LongBench/

bash pred.sh llama-3-8b-1048k hotpotqa      RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k 2wikimqa      RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k musique       RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k gov_report    RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k qmsum         RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k multi_news    RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k passage_count RetroInfer fp16 0.018 0.25 top-p True sclru
```

#### Step 3 — LongBench 注意力集中任务，Top-k

```bash
cd benchmark/LongBench/

bash pred.sh llama-3-8b-1048k passage_retrieval_en RetroInfer fp16 0.018 0.25 top-k True sclru
bash pred.sh llama-3-8b-1048k narrativeqa          RetroInfer fp16 0.018 0.25 top-k True sclru
bash pred.sh llama-3-8b-1048k qasper               RetroInfer fp16 0.018 0.25 top-k True sclru
bash pred.sh llama-3-8b-1048k trec                 RetroInfer fp16 0.018 0.25 top-k True sclru
bash pred.sh llama-3-8b-1048k triviaqa             RetroInfer fp16 0.018 0.25 top-k True sclru
```

#### Step 4 — LongBench 注意力集中任务，Top-p

```bash
cd benchmark/LongBench/

bash pred.sh llama-3-8b-1048k passage_retrieval_en RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k narrativeqa          RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k qasper               RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k trec                 RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k triviaqa             RetroInfer fp16 0.018 0.25 top-p True sclru
```

#### Step 5 — RULER 注意力集中（单针），Top-k

```bash
cd benchmark/ruler/

bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_single_1 fp16 0.018 0.25 top-k True sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_single_2 fp16 0.018 0.25 top-k True sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_single_3 fp16 0.018 0.25 top-k True sclru
```

#### Step 6 — RULER 注意力集中（单针），Top-p

```bash
cd benchmark/ruler/

bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_single_1 fp16 0.018 0.25 top-p True sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_single_2 fp16 0.018 0.25 top-p True sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_single_3 fp16 0.018 0.25 top-p True sclru
```

#### Step 7 — RULER 注意力分散（多针），Top-k

```bash
cd benchmark/ruler/

bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multikey_1 fp16 0.018 0.25 top-k True sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multiquery fp16 0.018 0.25 top-k True sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multivalue fp16 0.018 0.25 top-k True sclru
```

#### Step 8 — RULER 注意力分散（多针），Top-p

```bash
cd benchmark/ruler/

bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multikey_1 fp16 0.018 0.25 top-p True sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multiquery fp16 0.018 0.25 top-p True sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multivalue fp16 0.018 0.25 top-p True sclru
```

#### Step 9 — 吞吐量对比

```bash
cd throughput_eval/
# 先修改 run_different_lengths.sh 顶部 CLUSTER_SELECT="top-k" ，然后：
bash run_different_lengths.sh
# 再修改 run_different_lengths.sh 顶部 CLUSTER_SELECT="top-p" ，然后：
bash run_different_lengths.sh
```

#### 评测

```bash
cd benchmark/LongBench/
python eval.py --model llama-3-8b-1048k --attn_type RetroInfer

cd benchmark/ruler/
python eval/evaluate.py --data_dir ./ruler_eval_result/gradientai/Llama-3-8B-Instruct-Gradient-1048k/synthetic/131072/RetroInfer/pred --benchmark synthetic
```

### 预期结果

| 指标 | Top-k | Top-p |
|------|-------|-------|
| 注意力分散 LongBench 准确率 | 基线 | ≈Top-k |
| 注意力集中 LongBench 准确率 | 基线 | ≈Top-k |
| 单针 RULER 准确率 | 基线 | ≈Top-k |
| 多针 RULER 准确率 | 基线 | ≈Top-k |
| 吞吐量 (tokens/s) — 注意力集中 | 基线 | ↑ |
| 吞吐量 (tokens/s) — 注意力分散 | 基线 | ≈Top-k |

---

## E3: 聚类复用消融 (Reuse ON vs OFF)

### 目的

聚类复用（`cluster_reuse`）在连续 decode 步中，若当前 query 与上一步 query 的余弦相似度 > 0.95，则直接复用上一步的聚类选择结果，跳过当步的 `batch_gemm_softmax` 计算。纯粹的**性能优化**，核心指标是吞吐量和延迟。

### 运行命令

#### 吞吐量测试（主要，覆盖 5 个上下文长度 × 多种 batch size）

```bash
cd throughput_eval/
# 先修改 run_different_lengths.sh 顶部 CLUSTER_REUSE="False" ，然后：
bash run_different_lengths.sh
# 再修改 run_different_lengths.sh 顶部 CLUSTER_REUSE="True" ，然后：
bash run_different_lengths.sh
```

对比 60K/120K/240K/480K/1024K 上下文下的 `tokens/s` 和 `ms/step`。

#### 准确率抽样验证 — LongBench

```bash
cd benchmark/LongBench/

# Reuse OFF
bash pred.sh llama-3-8b-1048k gov_report RetroInfer fp16 0.018 0.25 top-p False sclru
bash pred.sh llama-3-8b-1048k musique   RetroInfer fp16 0.018 0.25 top-p False sclru
bash pred.sh llama-3-8b-1048k qasper    RetroInfer fp16 0.018 0.25 top-p False sclru

# Reuse ON
bash pred.sh llama-3-8b-1048k gov_report RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k musique   RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k qasper    RetroInfer fp16 0.018 0.25 top-p True sclru
```

#### 准确率抽样验证 — RULER

```bash
cd benchmark/ruler/

# Reuse OFF
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_single_1   fp16 0.018 0.25 top-p False sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multikey_1 fp16 0.018 0.25 top-p False sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 qa_1            fp16 0.018 0.25 top-p False sclru

# Reuse ON
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_single_1   fp16 0.018 0.25 top-p True sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multikey_1 fp16 0.018 0.25 top-p True sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 qa_1            fp16 0.018 0.25 top-p True sclru
```

### 预期结果

| 指标 | Reuse OFF | Reuse ON |
|------|-----------|----------|
| Decode 延迟 (ms/step) | 基线 | ↓（跳过 GEMM+softmax 步骤） |
| 吞吐量 (tokens/s) | 基线 | ↑ |
| LongBench 准确率（抽样） | 基线 | ≈OFF |
| RULER 准确率（抽样） | 基线 | ≈OFF |

---

## E4: 整体对比 (Full_Flash_Attn vs RetroInfer 最佳配置)

### 目的

基于 E1-E3 的结果确定最佳配置，与全注意力基线全面对比。验证 RetroInfer 在保持可接受准确率的前提下，在吞吐量和长上下文支持上的收益。

### 最佳配置预期

基于理论分析：`ARC + top-p + Reuse ON`（由 E1-E3 结果最终确认）

### LongBench 全量

```bash
cd benchmark/LongBench/

# Full_Flash_Attn 基线
bash pred.sh llama-3-8b-1048k qasper          Full_Flash_Attn fp16 0.018 0.25 top-p True arc
bash pred.sh llama-3-8b-1048k repobench-p     Full_Flash_Attn fp16 0.018 0.25 top-p True arc
bash pred.sh llama-3-8b-1048k lcc             Full_Flash_Attn fp16 0.018 0.25 top-p True arc
bash pred.sh llama-3-8b-1048k gov_report      Full_Flash_Attn fp16 0.018 0.25 top-p True arc
bash pred.sh llama-3-8b-1048k triviaqa        Full_Flash_Attn fp16 0.018 0.25 top-p True arc

# RetroInfer 最佳配置
bash pred.sh llama-3-8b-1048k qasper          RetroInfer fp16 0.018 0.25 top-p True arc
bash pred.sh llama-3-8b-1048k repobench-p     RetroInfer fp16 0.018 0.25 top-p True arc
bash pred.sh llama-3-8b-1048k lcc             RetroInfer fp16 0.018 0.25 top-p True arc
bash pred.sh llama-3-8b-1048k gov_report      RetroInfer fp16 0.018 0.25 top-p True arc
bash pred.sh llama-3-8b-1048k triviaqa        RetroInfer fp16 0.018 0.25 top-p True arc

# 评测
python eval.py --model llama-3-8b-1048k --attn_type Full_Flash_Attn
python eval.py --model llama-3-8b-1048k --attn_type RetroInfer
```

### RULER 全量（全 13 个任务）

```bash
cd benchmark/ruler/

# ===== Full_Flash_Attn 基线 =====
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic Full_Flash_Attn 131072 niah_single_1    fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic Full_Flash_Attn 131072 niah_single_2    fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic Full_Flash_Attn 131072 niah_single_3    fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic Full_Flash_Attn 131072 niah_multikey_1  fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic Full_Flash_Attn 131072 niah_multikey_2  fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic Full_Flash_Attn 131072 niah_multikey_3  fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic Full_Flash_Attn 131072 niah_multivalue  fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic Full_Flash_Attn 131072 niah_multiquery  fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic Full_Flash_Attn 131072 vt               fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic Full_Flash_Attn 131072 cwe              fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic Full_Flash_Attn 131072 fwe              fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic Full_Flash_Attn 131072 qa_1             fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic Full_Flash_Attn 131072 qa_2             fp16 0.018 0.25 top-p True arc

# ===== RetroInfer 最佳配置 =====
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_single_1    fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_single_2    fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_single_3    fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multikey_1  fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multikey_2  fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multikey_3  fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multivalue  fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multiquery  fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 vt               fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 cwe              fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 fwe              fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 qa_1             fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 qa_2             fp16 0.018 0.25 top-p True arc
```

### 评测

```bash
cd benchmark/LongBench/
python eval.py --model llama-3-8b-1048k --attn_type Full_Flash_Attn
python eval.py --model llama-3-8b-1048k --attn_type RetroInfer

cd benchmark/ruler/
python eval/evaluate.py --data_dir ./ruler_eval_result/gradientai/Llama-3-8B-Instruct-Gradient-1048k/synthetic/131072/Full_Flash_Attn/pred --benchmark synthetic
python eval/evaluate.py --data_dir ./ruler_eval_result/gradientai/Llama-3-8B-Instruct-Gradient-1048k/synthetic/131072/RetroInfer/pred --benchmark synthetic
```

### 预期结果

| 指标 | Full_Flash_Attn | RetroInfer (最佳) |
|------|----------------|-------------------|
| LongBench 准确率 | 上限 | 接近上限（差距量化信息损失） |
| RULER NIAH 准确率 | 上限 | 接近上限（验证聚类检索的有效性） |
| RULER 非 NIAH 准确率 | 上限 | 接近上限 |
| 吞吐量 (tokens/s) | 基线 | ↑↑（长上下文下优势更大） |
| Decode 延迟 (ms/step) | 基线 | ↓↓ |
| 最大上下文长度 | 受 GPU 显存限制 | 受 CPU 内存限制（更大） |
| 缓存命中率 | — | RetroInfer 特有指标 |

---

## 结果汇总模板

### E1: 缓存替换策略

| 策略 | 高敏感 LB acc | 低敏感 LB acc | 多针 RULER acc | 单针 RULER acc | 缓存命中率 | 吞吐量 |
|------|-------------|-------------|---------------|---------------|----------|--------|
| LRU | | | | | | |
| SCLRU | | | | | | |
| ARC | | | | | | |

### E2: 聚类选择方式

| 方式 | 分散任务 acc | 集中任务 acc | 单针 RULER acc | 多针 RULER acc | 吞吐量 | 延迟 |
|------|------------|------------|---------------|---------------|--------|------|
| top-k | | | | | | |
| top-p | | | | | | |

### E3: 聚类复用

| 复用 | 吞吐量 (60K) | 吞吐量 (120K) | 吞吐量 (240K) | 吞吐量 (480K) | 延迟 | LB spot acc | RULER spot acc |
|------|-------------|--------------|--------------|--------------|------|------------|---------------|
| OFF | | | | | | | |
| ON | | | | | | | |

### E4: 整体对比

| 方法 | LB acc | RULER NIAH acc | RULER 其他 acc | 吞吐量 | 延迟 |
|------|--------|---------------|---------------|--------|------|
| Full_Flash_Attn | | | | | |
| RetroInfer | | | | | |

---

## 快速启动：最小验证实验

时间有限时，最少跑以下 5 组命令即可覆盖所有优化维度的对比：

### 1. 三种策略对比（E1 核心）

```bash
cd benchmark/LongBench/
bash pred.sh llama-3-8b-1048k musique RetroInfer fp16 0.018 0.25 top-p True lru
bash pred.sh llama-3-8b-1048k musique RetroInfer fp16 0.018 0.25 top-p True sclru
bash pred.sh llama-3-8b-1048k musique RetroInfer fp16 0.018 0.25 top-p True arc

cd benchmark/ruler/
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multikey_1 fp16 0.018 0.25 top-p True lru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multikey_1 fp16 0.018 0.25 top-p True sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_multikey_1 fp16 0.018 0.25 top-p True arc
```

### 2. top-k vs top-p（E2 核心：准确率 + 吞吐量）

```bash
cd benchmark/LongBench/
bash pred.sh llama-3-8b-1048k gov_report RetroInfer fp16 0.018 0.25 top-k True sclru
bash pred.sh llama-3-8b-1048k gov_report RetroInfer fp16 0.018 0.25 top-p True sclru

cd benchmark/ruler/
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_single_1 fp16 0.018 0.25 top-k True sclru
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic RetroInfer 131072 niah_single_1 fp16 0.018 0.25 top-p True sclru
```

### 3. 复用开关（E3 核心：吞吐量）

```bash
cd throughput_eval/
# 修改 run_different_lengths.sh 顶部 CLUSTER_REUSE="False" → 运行
bash run_different_lengths.sh
# 修改 run_different_lengths.sh 顶部 CLUSTER_REUSE="True" → 运行
bash run_different_lengths.sh
```

### 4. 全注意力基线（E4）

```bash
cd benchmark/LongBench/
bash pred.sh llama-3-8b-1048k musique    Full_Flash_Attn fp16 0.018 0.25 top-p True arc
bash pred.sh llama-3-8b-1048k gov_report Full_Flash_Attn fp16 0.018 0.25 top-p True arc

cd benchmark/ruler/
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic Full_Flash_Attn 131072 niah_single_1    fp16 0.018 0.25 top-p True arc
bash ruler_run.sh gradientai/Llama-3-8B-Instruct-Gradient-1048k synthetic Full_Flash_Attn 131072 niah_multikey_1  fp16 0.018 0.25 top-p True arc
```

### 5. 评测

```bash
cd benchmark/LongBench/
python eval.py --model llama-3-8b-1048k --attn_type Full_Flash_Attn
python eval.py --model llama-3-8b-1048k --attn_type RetroInfer

cd benchmark/ruler/
python eval/evaluate.py --data_dir ./ruler_eval_result/gradientai/Llama-3-8B-Instruct-Gradient-1048k/synthetic/131072/Full_Flash_Attn/pred --benchmark synthetic
python eval/evaluate.py --data_dir ./ruler_eval_result/gradientai/Llama-3-8B-Instruct-Gradient-1048k/synthetic/131072/RetroInfer/pred --benchmark synthetic
```
