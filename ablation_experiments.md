# RetroInfer 消融实验方案

## 实验总览

| 实验 | 消融对象 | 固定配置 | 变化量 | 主要指标 | 一键脚本 |
|------|---------|---------|--------|---------|---------|
| E1 | 缓存替换策略 | top-p + 复用 | LRU / SCLRU / ARC | 命中率 + 准确率 | `run_e1_eviction.sh` |
| E2 | 聚类选择方式 | SCLRU + 复用 | top-k / top-p | 吞吐量 + 准确率 | `run_e2_selection.sh` |
| E3 | 聚类复用 | SCLRU + top-p | True / False | 吞吐量 + 复用命中率 + 准确率 | `run_e3_reuse.sh` |
| E4 | Top-p 阈值 | SCLRU + 复用 | top_p ∈ {0.3, 0.4, 0.5, 0.6} | 准确率 + 吞吐量 | `run_e4_topp.sh` |
| E5 | 参数交互 | — | 2×2×2 组合 | 准确率 + 命中率 + 吞吐量 | `run_e5_interact.sh` |

**公共参数**: `budget_ratio=0.018`, `estimate_ratio=0.25`, `dtype=fp16`, `model=llama-3-8b-1048k`, `top_p=0.4`

实验室脚本使用方式参见 [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)。
所有脚本位于 `benchmark/`，通过环境变量控制参数（冷却时间、模型、跳过部分测试等）。

---

## RULER 任务说明

RULER 在 **131072 tokens** 上下文中测试检索能力，与 RetroInfer 的 k-means 聚类检索机制高度相关。

| 任务 | 针的数量 | 特点 |
|------|---------|------|
| `niah_single_1` | 1 key, 1 value | repeat haystack（最简单） |
| `niah_single_2` | 1 key, 1 value | essay haystack |
| `niah_single_3` | 1 key, 1 value | essay + UUID（最困难） |
| `niah_multikey_1` | 4 key | 多针分散检索 |
| `niah_multikey_2` | 1 key | needle 噪声干扰 |
| `niah_multikey_3` | 1 key | needle 噪声 + UUID |
| `niah_multivalue` | 1 key, 4 value | 关联多值检索 |
| `niah_multiquery` | 1 key, 4 query | 多次查询同上下文 |

非 NIAH 任务：`vt`（变量追踪）、`cwe`（常见词）、`fwe`（高频词）、`qa_1`/`qa_2`（问答）。

---

## E1: 缓存替换策略 (LRU vs SCLRU vs ARC)

### 目的

ARC 维护 4 个链表自适应平衡 recency/frequency。当 GPU 缓存放不下所有热点聚类时，更好的替换策略能保留真正需要的聚类，减少 CPU→GPU 重载。

### 任务选择

**LongBench 高敏感**（缓存压力大）：`musique`, `gov_report`, `passage_count`
**LongBench 低敏感**（对照组）：`passage_retrieval_en`, `trec`, `triviaqa`
**RULER 高敏感**（多针）：`niah_multikey_1`
**RULER 低敏感**（对照组）：`niah_single_1`

### 运行

```bash
bash benchmark/run_e1_eviction.sh
# 或只跑 LongBench:  LONG_ONLY=1 bash benchmark/run_e1_eviction.sh
# 或只跑 RULER:      RULER_ONLY=1 bash benchmark/run_e1_eviction.sh
```

### 预期

| 指标 | LRU | SCLRU | ARC |
|------|-----|-------|-----|
| 高敏感 LongBench 准确率 | 基线 | ↑ | ↑↑ |
| 低敏感 LongBench 准确率 | — | ≈LRU | ≈LRU |
| 多针 RULER 准确率 | 基线 | ↑ | ↑↑ |
| 单针 RULER 准确率 | — | ≈LRU | ≈LRU |
| 缓存命中率 | 基线 | ↑ | ↑↑ |

关键验证：ARC 在高敏感任务上命中率显著优于 LRU；低敏感/单针任务三策略无差异。

---

## E2: 聚类选择方式 (Top-k vs Top-p)

### 目的

Top-k 固定检索 `nprobe` 个聚类，Top-p 按累积注意力概率动态决定检索数量。核心假设：注意力集中时 Top-p 自动减少检索提升吞吐，注意力分散时自动增加检索保持准确率。

### 任务选择

**LongBench 注意力分散**：`gov_report`, `musique`, `passage_count`
**LongBench 注意力集中**：`passage_retrieval_en`, `narrativeqa`, `qasper`
**RULER 注意力集中**：`niah_single_1`
**RULER 注意力分散**：`niah_multikey_1`

### 运行

```bash
bash benchmark/run_e2_selection.sh
# 跳过吞吐量:  SKIP_THROUGHPUT=1 bash benchmark/run_e2_selection.sh
```

### 预期

| 指标 | Top-k | Top-p |
|------|-------|-------|
| 注意力分散任务准确率 | 基线 | ≈Top-k |
| 注意力集中任务准确率 | 基线 | ≈Top-k |
| 吞吐量 — 注意力集中 | 基线 | ↑ |
| 吞吐量 — 注意力分散 | 基线 | ≈Top-k |

---

## E3: 聚类复用 (Reuse ON vs OFF)

### 目的

连续 decode 步中若 query 余弦相似度 > 0.95 则复用上一步聚类选择，跳过 `batch_gemm_softmax`。纯性能优化，核心指标是吞吐量和复用命中率。

### 运行

```bash
bash benchmark/run_e3_reuse.sh
```

### 预期

| 指标 | Reuse OFF | Reuse ON |
|------|-----------|----------|
| 吞吐量 (tokens/s) | 基线 | ↑ |
| LongBench / RULER 准确率 | 基线 | ≈OFF |
| 复用命中率 | — | > 80% |

---

## E4: Top-p 阈值敏感性

### 目的

验证 `top_p ∈ {0.3, 0.4, 0.5, 0.6}` 的敏感性。top_p 越大检索越多聚类，准确率上升但吞吐下降。

### 任务选择

LongBench: `gov_report`（注意力分散）, `passage_retrieval_en`（注意力集中）
RULER: `niah_single_1`（单针）, `niah_multikey_1`（多针）

### 运行

```bash
bash benchmark/run_e4_topp.sh
```

### 预期

| top_p | 集中任务 acc | 分散任务 acc | 吞吐量 |
|-------|------------|------------|--------|
| 0.3 | ≈0.4 | 可能下降 | ↑ |
| 0.4 | 基线 | 基线 | 基线 |
| 0.5 | ≈0.4 | ≈0.4 | ↓ |
| 0.6 | ≈0.4 | ≈0.4 | ↓↓

---

## E5: 参数交互效应

### 目的

验证 E1-E4 独立消融假设：reuse 改访问模式→影响缓存替换；top-p 减少检索→减轻缓存压力。选取 2×2×2 抽样（LRU/ARC × top-k/top-p × Reuse ON/OFF）。

### 任务选择

LongBench: `musique`（高缓存压力）, `gov_report`（注意力分散）
RULER: `niah_multikey_1`（4针分散）, `niah_single_1`（单针集中）

### 运行

```bash
bash benchmark/run_e5_interact.sh
```

### 预期

| 指标 | 预期 |
|------|------|
| LRU vs ARC 高压力任务 | ARC > LRU，top-k 下差距 > top-p 下差距 |
| Reuse 交互 | ARC 下 Reuse 吞吐提升 < LRU 下 |
| top-k vs top-p 单针 | top-p 吞吐 > top-k |

---

## 结果汇总模板

### E1: 缓存替换策略

| 策略 | 高敏感 LB acc | 低敏感 LB acc | 多针 RULER acc | 单针 RULER acc | 缓存命中率 |
|------|-------------|-------------|---------------|---------------|----------|
| LRU | | | | | |
| SCLRU | | | | | |
| ARC | | | | | |

### E2: 聚类选择方式

| 方式 | 分散任务 acc | 集中任务 acc | 单针 RULER acc | 多针 RULER acc | 吞吐量 | 延迟 |
|------|------------|------------|---------------|---------------|--------|------|
| top-k | | | | | | |
| top-p | | | | | | |

### E3: 聚类复用

| 复用 | 吞吐量 60K/120K/240K/480K | 延迟 | LB acc | RULER acc | 复用命中率 |
|------|--------------------------|------|--------|-----------|----------|
| OFF | | | | | — |
| ON | | | | | |

### E4: Top-p 阈值敏感性

| top_p | gov_report acc | passage_retrieval acc | niah_s1 acc | niah_mk1 acc | 吞吐量 |
|-------|---------------|----------------------|------------|-------------|--------|
| 0.3 | | | | | |
| 0.4 | | | | | |
| 0.5 | | | | | |
| 0.6 | | | | | |

### E5: 参数交互效应

| ev × sel × reuse | musique acc | gov_report acc | niah_mk1 acc | niah_s1 acc | 命中率 | 吞吐量 |
|------------------|------------|---------------|-------------|------------|------|--------|
| LRU × top-k × ON | | | | | | |
| LRU × top-k × OFF | | | | | | |
| LRU × top-p × ON | | | | | | |
| LRU × top-p × OFF | | | | | | |
| ARC × top-k × ON | | | | | | |
| ARC × top-k × OFF | | | | | | |
| ARC × top-p × ON | | | | | | |
| ARC × top-p × OFF | | | | | | |

