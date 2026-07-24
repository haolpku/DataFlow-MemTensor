# 数学 Mid-train 数据生产计划

> 版本 v1 · 对齐《MidTrain 数据方案》(飞书) 的 source dataset
> 定位:**主训练服务"提升数学 benchmark"这一真实目标;evidence / interleaved 作为能力 demo 交付甲方。**

---

## 0. 目标与分工(先说清楚,避免配比浪费)

数据格式决定训练出的能力。本项目有**两个不同性质的目标**,对应两条不同的数据线:

| 线 | 目标 | 主力数据 | 规模定位 |
|----|------|----------|----------|
| **A. 主训练线** | 提升 AIME / MATH-500 / GSM8K 等**数学 benchmark** | **可验证数学 CoT / TIR** | 占绝大部分 token(真花力气) |
| **B. Demo 线** | 给甲方展示"证据推理 / 长程交错思维"能力 | evidence + interleaved | 小规模(几十~几百条,够演示即可) |

**关键判断**:纯数学 benchmark 涨分,**加数学 CoT 的边际收益远高于 evidence/interleaved**。因此 token 预算应向 A 线倾斜;B 线不追求规模,追求"样例质量高、可视化清楚、能讲清方法论"。

> 若后续甲方 KPI 扩到长上下文(如飞书文档提的 AA-LCR ≥+3pp),再把 B 线放量成方案里的 8B 长程部分。当前阶段 B 线只做 demo。

---

## 1. 数据来源总表(源自飞书方案 §3.1 / §3.2 / §3.3)

### 1.1 主力数学题库(A 线核心,§3.2)

| 数据集 | HF 仓库 | License | 体量 | 难度档 | 用途 |
|--------|---------|---------|------|--------|------|
| NuminaMath-CoT | `AI-MO/NuminaMath-CoT` | Apache-2.0 | ~860K | L2-L4 | **中高难主力**(含 AMC/AIME/olympiad) |
| OpenMathReasoning | `nvidia/OpenMathReasoning` | CC-BY-4.0 | 540K题/3.2M CoT/1.7M TIR | L2-L4 | **CoT+TIR 主力**(AIMO-2 冠军) |
| OpenR1-Math-220k | `open-r1/OpenR1-Math-220k` | Apache-2.0 | ~220K | L2-L3 | DeepSeek-R1 蒸馏 CoT |
| OpenMathInstruct-2 | `nvidia/OpenMathInstruct-2` | CC-BY-4.0 | ~14M | L1-L3 | 大规模合成主力 |
| ORCA-Math | `microsoft/orca-math-word-problems-200k` | MIT | ~200K | L1 | 基础应用题 |
| MetaMathQA | `meta-math/MetaMathQA` | MIT | ~395K | L1 | 基础改写增广 |
| DART-Math-Uniform | `hkust-nlp/dart-math-uniform` | MIT | ~591K | L1-L4 | 难度均衡 |
| Big-Math-RL-Verified | `SynthLabsAI/Big-Math-RL-Verified` | Apache-2.0 | ~250K | L4 | **严格可验证**(RL 级) |
| DAPO-Math-17k | `BytedTsinghua-SIA/DAPO-Math-17k` | Apache-2.0 | ~17K | L4 | 高密度难题 |
| DeepScaleR-Preview | `agentica-org/DeepScaleR-Preview-Dataset` | MIT | ~40K | L4 | RL 难题 |
| HARP | `HKUST-NLP/HARP` | MIT | ~5K | L4 | 北美高中竞赛 |
| OpenThoughts3-1.2M | `open-thoughts/OpenThoughts3-1.2M` | Apache-2.0 | ~1.2M | OOD | long-CoT / OOD |

### 1.2 长上下文骨架语料(§3.1,主要用于长程/预训练回流,当前阶段非重点)

| 数据集 | HF 仓库 | License | 体量 |
|--------|---------|---------|------|
| MegaMath | `LLM360/MegaMath` | Apache-2.0 | ~371B |
| Nemotron-CC-Math-4+ | `nvidia/Nemotron-CC-Math` | NVIDIA Open Data | ~52B |
| FineMath-4+ | `HuggingFaceTB/finemath` | ODC-By-1.0 | ~9.6B |
| OpenWebMath | `open-web-math/open-web-math` | ODC-By-1.0 | ~14.7B |
| peS2o (S2ORC) | `allenai/peS2o` | ODC-BY | 海量 |
| parsed_math_pdf | `OpenDCAI/parsed_math_pdf` | 逐文件核验 | ~130.7GB |
| college-mathbooks-2k | `OpenDCAI/college-mathbooks-2k` | 逐文件核验 | ~2.5GB(带 KG) |

### 1.3 长程 / 多跳来源(§3.3,B 线放量时才用)

- **B 数学天然多跳**(核心):`nvidia/OpenMathReasoning`(TIR 按工具边界切)、`open-thoughts/OpenThoughts3-1.2M`(long-CoT 按结论切)、`leanprover-community/mathlib4`(定理依赖图)、`wellecks/naturalproofs-gen`(证明链)
- **A 外部通用多跳**:`hotpot_qa`、`dgslibisey/MuSiQue`、`voidful/2WikiMultihopQA` 等(加干扰改造成长程)
- **C 自合成**:college-KG concept-dependency 反推、章节聚类跨文档链、NuminaMath 难题分步重写

---

## 2. A 线:数学 CoT 主训练数据(真正提 benchmark)

### 2.1 难度分布(对齐 §2.3,容差 ±5pp)

| 难度档 | 占比 | 主要来源 |
|--------|------|----------|
| L1 基础 | 25% | ORCA-Math、MetaMathQA、OpenMathInstruct-2 |
| L2-L3 中等竞赛 | 40% | NuminaMath(AMC/AIME)、OpenR1-Math、OpenMathReasoning |
| L4 高难竞赛 | 25% | NuminaMath olympiad、HARP、Big-Math-RL-Verified、DAPO-Math、DeepScaleR |
| OOD | 10% | OpenThoughts3、DART-Math-Hard |

### 2.2 生产流程(用本项目 `cot_pipeline`)

对**已有 CoT 的数据**(NuminaMath / OpenR1 / OpenMathReasoning):走"整理 + 校验"轻路径,不重新生成。
对**只有题+答案的数据**(Big-Math / DAPO / ORCA):用 `ReasoningLongCoTGenerator` 生成 CoT。
两者统一过 `ReasoningCoTAnswerFilter`(抽 `\boxed{}` → `math_verify` 对 golden_answer)。

```
seed(题+golden_answer)
  → [已有CoT] 直接整理  /  [无CoT] ReasoningLongCoTGenerator 生成 <think>+\boxed{}
  → ReasoningCoTAnswerFilter  (math_verify 校验,错的进 Failure Pool)
  → 按难度档配比抽样
  → 去重 + 污染扫描(§4)
  → 主训练集
```

命令:`python -m dataflow_memtensor.pipelines.cot_pipeline`

### 2.3 提分的关键抓手(比"堆量"更重要)

1. **答案可验证**:每条都能被 `math_verify`/sympy 独立校验,错答案坚决不进(§5.2)。
2. **难度分层**:别全是 GSM8K 级简单题;L4 竞赛题是 AIME/MATH-500 提分的关键。
3. **解法多样性**:同题多解(方案要求 L2-L3 ≥3 解 / L4 ≥5 解),用 Big-Math/DAPO 的 verified 多解。
4. **CoT+TIR 混合**:OpenMathReasoning 的 TIR(带工具调用的解题)对难题有额外增益。
5. **去污染**:见 §4,评测集命中率必须 0%,否则涨分是假的。

---

## 3. B 线:evidence + interleaved Demo(交付甲方展示)

**定位**:不放量,做**高质量小样本 + 清晰可视化**,讲清"证据接地"和"长程工具推理"两种能力。

### 3.1 evidence demo(多步证据推理)

- pipeline:`python -m dataflow_memtensor.pipelines.evidence_pipeline`
- 种子:从 NuminaMath / college-mathbooks-2k 取 20-50 道有多步结构的题
- 产出:每条 `evidences[]`(含**干扰项**)+ `steps[]`(claim 绑 evidence_id)+ grounding 指标
- 展示重点:claim↔evidence 绑定率 ≥95%、去证据答案不可得、**有干扰项需筛选**(不是照抄)

### 3.2 interleaved demo(长程交错思维)

- pipeline:`python -m dataflow_memtensor.pipelines.interleaved_pipeline`
- 种子:20-50 道需要"查定理→计算→验证"的题
- 产出:`(thought → tool → observation)` 真实工具轨迹 + 质量四轴评分
- 展示重点:**真实 sympy/python 执行**、observation 留痕、模型自我纠错轨迹

### 3.3 可视化(已就绪)

`viewer/`(本地网页):三类数据自动识别渲染,evidence 有 claim↔evidence 高亮联动,
interleaved 有 r/a/o 时间线,并可切「训练格式」看真正喂给模型的训练串。
→ **直接用于甲方演示。**

---

## 4. 污染控制(A/B 线都必须做,§3.4 / §5.1)

以下评测集**不得进入训练数据**,对 problem/answer/关键中间步骤做 10-gram 黑名单扫描,命中率 = 0%:

```
Omni-MATH, AIME 2024-2026, MATH-500, GSM8K test, GPQA-Diamond,
OlympiadBench, TheoremQA, MMLU-Pro, BBH, LiveBench, ProcessBench,
GSM-Hard/Plus/Symbolic, LongBench v1/v2, RULER, FRAMES, Bamboogle, FanOutQA
```

> 这是"涨分为真"的前提。当前 pipeline **尚未接入**此扫描,放量前必须补(见 §6 待办)。

---

## 5. Token 预算建议(服务数学 benchmark 目标)

在"只提数学 benchmark"的目标下,建议配比(与飞书 20B 方案的差异:把重心压到可验证数学):

| 部分 | 建议占比 | 说明 |
|------|----------|------|
| 可验证数学 CoT/TIR(A 线) | **80-90%** | 真正提分的主力 |
| evidence + interleaved(B 线) | demo 规模(几百条) | 甲方展示,不占大预算 |
| 长上下文长程(§3.3 放量) | 视甲方是否要 AA-LCR 再定 | 当前不做 |

> 若严格按飞书方案交付(数学 12B + 长程 8B),则 B 线需放量到 8B——但那服务的是长上下文 KPI,不是纯数学分。**两个目标要分开算预算,别混。**

---

## 6. 阶段与待办

### 阶段
1. **试制(now)**:A 线跑 L1-L4 各档小样本验证 cot_pipeline;B 线出 evidence/interleaved demo 给甲方。
2. **放量**:A 线按难度配比拉到目标 token;接入去污染 + 去重 + 分片存储。
3. **验收**:固定 base model + recipe,只换数据,对照公开基线(NuminaMath/OpenMathReasoning 等)看 benchmark 提升。

### 从 demo 到生产的待办(诚实清单)
- [ ] **接真实题库**:seed 从当前手写/小样本换成 §1.1 的 HF 数据集(`FileStorage` 支持 `hf:` 前缀直读)
- [x] **去污染扫描**:接入 §4 的 10-gram 黑名单(`DecontaminationFilter`)
- [x] **难度分层器**:自动给题目打 L1-L4(`DifficultyTagOperator`:来源标签初分 + LLM scorer 兜底)
- [x] **来源/合成血缘**:`ProvenanceOperator` 盖 problem_source/synthetic_flag/gen_model/created_at(核查文档 §5)
- [x] **失败池 + 通过率**:各 filter 淘汰样本落盘 + 打印通过率(核查文档 §3/§七)
- [x] **判分独立**:interleaved 判分改用独立模型(核查文档 §2)
- [ ] **多解生成**:L2-L3 ≥3 解、L4 ≥5 解(Teacher rollout)
- [ ] **去重**:题面/答案近似去重
- [ ] **分片 + manifest**:大规模下记录来源/难度/长度/license 血缘
- [ ] **evidence 接真实检索**:证据从"LLM 脑补"换成从真实语料检索(BM25→FlashRAG),当前 demo 用小 KB

---

## 附:一句话总结

**你要的数学 benchmark 提升 = 主要靠 A 线数学 CoT(80-90% 预算,严格校验+难度分层+去污染)。
evidence / interleaved 是 B 线 demo,做精不做多,配合 `viewer/` 给甲方演示能力即可。**
