# 20B token 数学 CoT 合成计划

> 目标:合成 **20B token 的可验证数学 CoT 数据**,用于提升 AIME / MATH-500 / GSM8K 等数学 benchmark。
> 本文力求**合理且易懂**:先算清楚"20B 到底是多少题",再讲"从哪来、怎么造、怎么保质",最后给里程碑。

---

## 一、先把 20B 换算成"人话"

训练数据的单位是 token,但我们生产的单位是"题"。先建立换算感觉:

| 量 | 估算 | 说明 |
|----|------|------|
| 一条数学 CoT 的长度 | **~1,500 token** | 题面+`<think>`长推理+`\boxed{}`答案。简单题 ~500,竞赛难题 ~4,000,平均按 1.5K |
| 20B ÷ 1.5K | **≈ 1,300 万条** | 这是"最终入库"的条数 |
| 考虑过滤淘汰率 ~40% | **需生成 ≈ 2,200 万条** | 答案错的、格式坏的、污染的都要扔 |
| 去重后独立题目 | **~200-400 万道题** | 每题平均 3-5 个解(多解多样性),题目数远小于条数 |

**一句话**:20B ≈ 1300 万条 CoT ≈ 300 万道题 × 每题多个解。所以核心工作是"**搞到几百万道好题 + 每题生成并校验多个解**"。

---

## 二、题从哪来(source dataset)

不自己凭空造题,**优先复用公开高质量题库**(来自《MidTrain 方案》§3.2)。按难度分档配比:

| 难度档 | 占比 | Token | 主要题库(HF 仓库) |
|--------|------|-------|--------------------|
| **L1 基础** | 25% | 5B | `microsoft/orca-math-word-problems-200k`、`meta-math/MetaMathQA`、`nvidia/OpenMathInstruct-2` |
| **L2-L3 中等竞赛** | 40% | 8B | `AI-MO/NuminaMath-CoT`(含 AMC/AIME)、`open-r1/OpenR1-Math-220k`、`nvidia/OpenMathReasoning` |
| **L4 高难竞赛** | 25% | 5B | NuminaMath olympiad 子集、`HKUST-NLP/HARP`、`SynthLabsAI/Big-Math-RL-Verified`、`BytedTsinghua-SIA/DAPO-Math-17k`、`agentica-org/DeepScaleR-Preview-Dataset` |
| **OOD 泛化** | 10% | 2B | `open-thoughts/OpenThoughts3-1.2M`、DART-Math-Hard |

> 难度分布对齐方案 §2.3(容差 ±5pp)。**L4 竞赛题是 AIME/MATH-500 提分的关键**,别让数据全堆在 L1。

**两类题库,两种处理**:
- **已带 CoT 的**(NuminaMath / OpenR1 / OpenMathReasoning):走轻路径——只做清洗、校验、格式归一,**不重新生成**,省钱且真实。
- **只有题+答案的**(Big-Math / DAPO / ORCA):用 LLM 生成 CoT(见下)。

---

## 三、怎么造(生产流程)

用本仓库的 `cot_pipeline`(`ReasoningLongCoTGenerator` + `ReasoningCoTAnswerFilter`),放量版流程:

```
                           ┌─ 已有CoT → 清洗/格式归一 ─┐
HF题库(题+golden_answer) ─┤                          ├─→ math_verify 校验 ─→ 去重 ─→ 去污染 ─→ 入库
                           └─ 无CoT → LLM生成<think>+boxed ┘        │
                                                              错答案 → Failure Pool(自检/PRM/负样本)
```

**分四步**:

1. **拉题 + 难度打标**:从 HF 拉题(`FileStorage` 支持 `hf:` 前缀直读),按来源/LLM scorer 打 L1-L4 标签,按 §二 配比抽样。
2. **生成 CoT**:无 CoT 的题用 teacher 模型生成 `<think>…</think>` + `\boxed{}`。**每题生成多个解**(L2-L3 ≥3 解、L4 ≥5 解,温度抖动),提升解法多样性。
3. **答案校验**(最关键):抽 `\boxed{}` → `math_verify`/sympy 对比 golden_answer。**错的坚决不进主集**,进 Failure Pool。
4. **去重 + 去污染**:题面/答案近似去重;评测集 10-gram 黑名单扫描(见 §五)。

**成本感觉**:2200 万条 × 平均 1.5K token,teacher 模型推理是大头。建议:
- 已有 CoT 的题占比尽量高(不花生成成本);
- 生成部分用性价比模型(如 DeepSeek / Qwen)批量跑,难题档再上强模型;
- 用 Ray / 高并发 serving(方案 §2.2 已有此基础设施)。

---

## 四、质量抓手(比"堆量"更重要)

同样 20B,质量决定涨不涨分。五个抓手:

1. **答案可验证**:每条都过 `math_verify`,错答案零容忍。这是数据可信的地基。
2. **难度分层要真**:别全是 GSM8K 级。L4 竞赛题(AIME/olympiad)是高分区间的关键。
3. **解法多样性**:同题多解,覆盖不同解题路径(方案要求 L2-L3 ≥3 / L4 ≥5)。
4. **CoT + TIR 混合**:掺入带工具调用的解题轨迹(OpenMathReasoning 的 1.7M TIR),对难题额外增益(AIMO-2 冠军经验)。
5. **Pass@k 双向过滤**:太简单(Pass@1 就 100%)和完全不可解(Pass@k 全 0)的题都剔除,保留有区分度的。

---

## 五、去污染(涨分为真的前提,方案 §3.4 / §5.1)

**训练数据绝不能包含评测集**,否则涨分是假的。对 problem / answer / 关键中间步骤做 **10-gram 黑名单扫描**,命中率必须 = 0%:

```
Omni-MATH, AIME 2024-2026, MATH-500, GSM8K test, GPQA-Diamond,
OlympiadBench, TheoremQA, MMLU-Pro, BBH, LiveBench, ProcessBench,
GSM-Hard/Plus/Symbolic, LongBench, RULER, FRAMES, Bamboogle, FanOutQA
```

> ⚠️ 当前 `cot_pipeline` **尚未接入**此扫描,放量前必须补。这是验收红线。

---

## 六、里程碑(分阶段,先小后大)

| 阶段 | 规模 | 目标 | 门控 |
|------|------|------|------|
| **试制** | ~1B(5%) | 跑通全流程,验证格式/校验/去污染;小样本 continue-train 看信号 | 格式解析 100%、污染 0%、答案校验通过率达标 |
| **中期** | ~10B(50%) | 同 token 对照公开基线,确认方向正确 | 核心数学不退步,主要指标方向为正 |
| **终验** | 20B(100%) | 完整对照评估,报告均值/方差/置信区间 | AIME/MATH-500/GSM8K 平均提升达约定 MDE(方案建议 ≥+1.5pp) |

**验收方式**(方案 §6.2):固定同一 base model + 同 token 数 + 同 recipe + 同评测脚本,**只换数据**,和公开基线(NuminaMath/OpenMathReasoning 等)对照。

---

## 七、交付物清单

- **训练数据**:20B token,JSONL,每条含 `instruction / generated_cot / golden_answer / difficulty / source`
- **Data Card**:难度分布、来源配比、token 统计(raw/candidate/final)
- **污染报告**:各评测集 10-gram 命中率(应全 0)
- **质检报告**:抽检通过率、答案校验通过率
- **Failure Pool**:错答案样本(可作 PRM / 自检训练资产)
- **复跑脚本**:即本仓库 `cot_pipeline` + 配置

---

## 八、与本仓库现状的差距(诚实清单)

当前 `cot_pipeline` 是**能跑通的骨架**(24 条 demo 已验证),放量到 20B 还需补:

- [ ] **接 HF 大题库**:seed 从 demo 的 24 条换成 §二 的百万级题库
- [ ] **难度分层器**:自动打 L1-L4 标签
- [ ] **多解生成**:每题多次采样(现在每题 1 解)
- [ ] **去污染扫描**:10-gram 黑名单(现在没有,§五)
- [ ] **去重**:题面/答案近似去重
- [ ] **Pass@k 过滤**:剔除过易/不可解题
- [ ] **分布式 serving**:Ray/高并发(现在是 API 串行 max_workers)
- [ ] **分片 + manifest**:大规模存储 + 来源/难度/license 血缘

---

## 一句话总结

**20B ≈ 300 万道好题 × 每题多解,经 math_verify 校验 + 去污染后入库。
重心是"搞到分层良好的真实题库 + 严格校验",而不是无脑堆量。质量和去污染决定涨分真假。**
