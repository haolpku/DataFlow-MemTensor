# 20B 数据处理 Pipeline 设计

> 从"能跑通的 demo pipeline"到"能出 20B token"的工程设计。
> 核心问题不是"算子会不会写",而是**阶段怎么切、怎么并行、怎么保质、怎么可复现**。

---

## 0. 设计原则(先讲为什么这么设计)

1. **分阶段落盘,不做一条龙**。20B 规模下,任何一步崩了都不能从头再来。每阶段产物独立落盘(沿用 `FileStorage` 的 `step()` 机制),可断点续跑、可单独重跑、可审计。
2. **贵的操作尽量靠后**。LLM 生成是最贵的一步。所以**先用便宜的规则/校验把题筛干净**,再花钱生成 CoT,而不是先生成一堆再扔。
3. **能复用真实 CoT 就不生成**。§源台账里"自带 CoT"的数据走轻路径(清洗+校验),只有"仅题+答案"的才生成。
4. **可验证优先于形式复杂**。答案能不能被 `math_verify` 校验,是数据能不能用的地基。

---

## 1. 全景:七阶段流水线

```
[S1 采集入库]  拉 HF 题库 → 归一 schema → 分片落盘
      │
[S2 清洗去重]  格式清洗 + 题面近似去重 + PII/语言过滤
      │
[S3 难度分层]  打 L1-L4 标签 + 可解性评估 → 按配比抽样
      │
[S4 去污染]★   10-gram 黑名单扫描(评测集) → 命中即剔除     ← 涨分为真的红线
      │
      ├──[自带CoT]──→ 清洗/格式归一 ────────┐
[S5 CoT获取]                                 │
      └──[仅题+答案]→ LLM 生成 CoT(多解)──┘
      │
[S6 校验过滤]  math_verify 答案校验 + Pass@k 双向过滤 + 格式/长度过滤
      │         (不通过 → Failure Pool)
[S7 出库打包]  去重(解级) + Data Card + manifest + token 统计 → 主训练集
```

★ = 最关键、最容易被跳过的一步。

---

## 2. 逐阶段设计

### S1 采集入库
- **做什么**:从 HF 拉题库(`FileStorage` 支持 `hf:` 前缀直读),统一归一到 `{id, problem, golden_answer, source, has_cot, cot?}` schema。
- **算子**:新建 `SourceIngestOperator`(读 HF/本地 → 归一列名)。
- **并行**:按数据集分片,每个数据集一个 worker,并行拉。
- **落盘**:`s1_ingest/<dataset>_shardNN.jsonl`。

### S2 清洗去重
- **做什么**:去空题/超长/乱码;题面近似去重(MinHash/SimHash);非英文或含 PII 的剔除。
- **算子**:复用上游 `reasoning_question_filter`(合法性/可解性)+ 新建 `DedupOperator`(MinHash LSH)。
- **并行**:去重需全局视野 → 先分片算 MinHash 签名,再全局 LSH 合并(map-reduce)。
- **落盘**:`s2_clean/`。

### S3 难度分层
- **做什么**:给每题打 L1-L4;按方案配比(25/40/25/10)抽样。
- **算子**:**复用上游** `reasoning_question_difficulty_sample_evaluator`(LLM 打难度分)+ `reasoning_question_solvable_sample_evaluator`(可解性)。也可先用来源标签(olympiad→L4, gsm8k→L1)做廉价初分,再对边界题上 LLM。
- **并行**:LLM 打分批量并发。
- **落盘**:`s3_leveled/`(带 `difficulty` 列)。

### S4 去污染 ★(红线)
- **做什么**:对 problem/answer/关键中间步骤做 **10-gram 黑名单扫描**,命中评测集即剔除。
- **算子**:新建 `DecontaminationFilter`(构建评测集 10-gram 集合 → 扫描)。
- **黑名单**:Omni-MATH / AIME 2024-2026 / MATH-500 / GSM8K test / GPQA-Diamond / OlympiadBench / ...(方案 §3.4 全表)。
- **特别注意**:来源自带的评测集要按标签先剔(Big-Math↔Omni-MATH、NuminaMath↔MATH/GSM8K,见源台账)。
- **产出**:`contamination_report`(各评测集命中率,应全 0)。
- **落盘**:`s4_clean/` + 报告。

### S5 CoT 获取(分流)
- **自带 CoT 的**(NuminaMath/OpenR1/OpenMathReasoning):清洗 + 格式归一到 `<think>+\boxed{}`,**不重新生成**。算子:新建 `CoTNormalizeOperator`。
- **仅题+答案的**(Big-Math/DAPO/...):`ReasoningLongCoTGenerator`(本仓库,已有)生成。**每题多解**:L2-L3 采样 ≥3 次、L4 ≥5 次(温度抖动),用 `num_return` 控制。
- **并行**:生成是最贵的一步,用高并发 serving(Ray / 大 `max_workers`),按难度分队列(难题给强模型)。
- **落盘**:`s5_cot/`。

### S6 校验过滤
- **做什么**(按顺序,从便宜到贵):
  1. **格式过滤**:复用上游 `reasoning_answer_formatter_filter`(有没有 `\boxed{}`)、`reasoning_answer_token_length_filter`(长度)。
  2. **答案校验**:`ReasoningCoTAnswerFilter`(本仓库,已有)—— 抽 boxed → `math_verify` 对 golden_answer。**错的进 Failure Pool,不删**。
  3. **Pass@k 双向过滤**:一题多解的通过率——全对(太简单)和全错(不可解)都剔除,保留有区分度的。算子:新建 `PassAtKFilter`。
  4. **n-gram 自重复过滤**:复用上游 `reasoning_answer_ngram_filter`(去掉复读机式 CoT)。
- **落盘**:`s6_verified/` + `failure_pool/`。

### S7 出库打包
- **做什么**:解级去重(同题近似解去重)→ 生成 Data Card(难度分布/来源配比/token 统计)→ manifest(每条来源/难度/license 血缘)→ 分片打包。
- **算子**:新建 `PackagingOperator`。
- **落盘**:`final/` + `data_card.md` + `manifest.jsonl`。

---

## 3. 复用 vs 新建(工程量清单)

| 阶段 | 复用现有 | 需新建 |
|------|----------|--------|
| S1 采集 | FileStorage(hf:) | `SourceIngestOperator` |
| S2 清洗去重 | `reasoning_question_filter` | `DedupOperator`(MinHash LSH) |
| S3 难度分层 | ✅ `reasoning_question_difficulty_sample_evaluator`、`..._solvable_...` | 来源标签初分(轻量脚本) |
| S4 去污染 | — | `DecontaminationFilter`(**优先级最高**) |
| S5 CoT | ✅ `ReasoningLongCoTGenerator`(本仓库) | `CoTNormalizeOperator`、多解采样开关 |
| S6 校验 | ✅ `ReasoningCoTAnswerFilter`(本仓库)、`answer_formatter/token_length/ngram_filter`(上游) | `PassAtKFilter` |
| S7 出库 | — | `PackagingOperator` |

**关键结论**:难度分层、答案校验、格式/长度/ngram 过滤**都有现成算子**。真正要新建的是 **去污染、去重、Pass@k、采集/打包** 这几个——其中**去污染最关键**(demo 完全没做)。

---

## 4. 规模与并发(20B 的工程现实)

| 维度 | demo 现状 | 20B 生产 |
|------|-----------|----------|
| 数据流 | 单机 jsonl,串行 step | **分片**(按数据集/难度切),每片独立跑 |
| LLM serving | `APILLMServing_request` 串行 max_workers | **Ray 批量 serving**(方案 §2.2 已有),难题分队列 |
| 存储 | `./cache_*/` 本地 | 对象存储/分布式 FS,分片 + manifest 血缘 |
| 断点续跑 | step 缓存 | 每阶段 checkpoint,失败重跑单片 |
| 监控 | 无 | 每阶段落盘条数 + 淘汰率 + token 统计仪表盘 |

**成本控制三招**:①复用 CoT 占比最大化(不烧生成钱);②便宜操作(去重/去污染/格式)全部前置,贵的生成靠后;③生成按难度分队列,L1 用小模型、L4 才上强模型。

---

## 5. 数据流量估算(淘汰漏斗)

```
S1 拉题        ~500 万道题(去重前)
S2 清洗去重    → ~350 万道(去重淘汰 ~30%)
S3 难度分层    → 按 25/40/25/10 配比抽样
S4 去污染      → ~340 万道(评测集命中剔除,应 <3%)
S5 生成/复用   → ~1500 万条解(每题多解)
S6 校验过滤    → ~1300 万条(答案错/格式坏淘汰 ~15%)  → Failure Pool ~200 万条
S7 出库        → 20B token 主训练集
```

> 淘汰率是估算,试制阶段(§里程碑)用真实数据校准。

---

## 6. 与 demo 的差距 = 待建清单

demo(`cot_pipeline`)现覆盖 **S0 血缘 + S3 难度打标 + S4 去污染 + S5 生成 + S6 答案校验**,并把淘汰样本落失败池。放量到 20B 仍需补:

- [x] **S4 去污染**(`DecontaminationFilter`,红线,已接入)
- [x] **S3 难度打标**(`DifficultyTagOperator`:来源初分 + LLM 兜底;**按配比抽样已实现**,放量时开 `target_ratio`)
- [x] **S0 血缘标记**(`ProvenanceOperator`:source/synthetic_flag/gen_model/created_at)
- [x] **失败池 + 通过率**(`failure_pool.py`:S6 及各 filter 淘汰样本落盘,即 Failure Pool)
- [x] **判分独立**(interleaved 判分改用独立模型,核查文档 §2)
- [ ] S1 采集器(接 HF 大题库)
- [ ] S2 去重(MinHash LSH)
- [ ] S5 多解采样开关 + 自带CoT 归一算子
- [ ] S6 Pass@k 过滤
- [ ] S7 打包 + Data Card + manifest
- [ ] 分布式 serving + 分片存储 + 监控
- [ ] **数据侧(非代码)**:接真实题源与检索语料重跑,以真实检索命中率/成功率/绑定率建 baseline(核查文档 §一.1 / §七)

---

## 一句话总结

**七阶段漏斗:采集→清洗去重→难度分层→[去污染]→CoT获取(复用/生成)→校验过滤→出库。
便宜操作前置、贵的生成靠后、每阶段落盘可续跑;难度分层和答案校验复用现成算子,
去污染是必须新建且不能跳的红线。**
