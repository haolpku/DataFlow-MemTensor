# 20B CoT 数据源台账 (Source Catalog)

> 按**难度档为主轴**分类,配合来源类型、处理方式、license 的结构化台账。
> 目的:一眼看清"每个数据集是什么档、怎么处理、能不能用"。
> 核验状态:标 ✅ 的已联网核验(2026-07 HF 现状);未标的沿用《MidTrain 方案》§3.2,放量前需核验。

---

## 0. 分类逻辑(先讲清怎么选的)

**主轴 = 难度档**(因为目标是提数学 benchmark,而 benchmark 从小学题铺到竞赛题,难度分布是提分关键)。四档配比对齐方案 §2.3:

```
L1 基础 25%  │  L2-L3 中等竞赛 40%  │  L4 高难竞赛 25%  │  OOD 10%
   5B        │        8B            │       5B          │   2B
```

**每个数据集过三关才入选**:①答案可验证 ②难度可分层 ③license 干净。

**处理方式两分类**(决定成本):
- `复用CoT` — 数据自带 CoT,只清洗+校验+格式归一,**不重新生成**(省钱、真实)
- `生成CoT` — 只有题+答案,需 LLM 生成解题过程

---

## 1. L1 基础档 (25% / ~5B)

**定位**:保下限,GSM8K 级应用题与基础运算。要求量大、简单、干净。

| 数据集 | HF 仓库 | 体量 | License | 自带CoT | 处理 | 选择理由 |
|--------|---------|------|---------|---------|------|----------|
| ORCA-Math | `microsoft/orca-math-word-problems-200k` | ~200K | MIT | 是(解题过程) | 复用CoT | 小学应用题主力,量足 |
| MetaMathQA | `meta-math/MetaMathQA` | ~395K | MIT | 是 | 复用CoT | 改写增广,一题多形态 |
| OpenMathInstruct-2 | `nvidia/OpenMathInstruct-2` | ~14M | CC-BY-4.0 | 是 | 复用CoT | 大规模合成,L1 量的兜底池 |

---

## 2. L2-L3 中等竞赛档 (40% / ~8B) — 占比最大,主力

**定位**:高中数学 + 中档竞赛(AMC/AIME 入门)。这是 benchmark 提分的中坚。

| 数据集 | HF 仓库 | 体量 | License | 自带CoT | 处理 | 选择理由 |
|--------|---------|------|---------|---------|------|----------|
| ✅ NuminaMath-CoT | `AI-MO/NuminaMath-CoT` | **859,608 行** | Apache-2.0 | **是(全部)** | 复用CoT | **竞赛数学事实标准**,含 AMC/AIME/olympiad(见下源分布) |
| ✅ OpenR1-Math-220k | `open-r1/OpenR1-Math-220k` | **default 93.7K / 全 450K 行** | Apache-2.0 | **是(每题 2-4 条 R1 轨迹)** | 复用CoT | DeepSeek-R1 蒸馏,**自带 math_verify 校验字段** |
| ✅ OpenMathReasoning (CoT) | `nvidia/OpenMathReasoning` | **306K 题 / 3.2M CoT** | CC-BY-4.0 | **是** | 复用CoT | **AIMO-2 冠军数据** |
| ✅ OpenMathReasoning (TIR) | `nvidia/OpenMathReasoning` | **1.72M TIR** | CC-BY-4.0 | 是(带工具调用) | 复用TIR | **独特价值:带工具的解题轨迹**,补纯 CoT 补不上的难题能力 |

> **NuminaMath-CoT 源分布**(核验所得,便于按难度再筛):
> cn_k12 276K / synthetic_math 168K / orca_math 153K / **olympiads 150K** / synthetic_amc 62K / aops_forum 30K / math 7.5K / gsm8k 7.3K / **amc_aime 4K**
> → olympiads + amc_aime 子集可上提到 L4;gsm8k + cn_k12 偏 L1-L2。

---

## 3. L4 高难竞赛档 (25% / ~5B) — AIME/MATH-500 提分关键

**定位**:olympiad / AIME 级难题。要求**答案经严格验证**(难题最怕答案错)。

| 数据集 | HF 仓库 | 体量 | License | 自带CoT | 处理 | 选择理由 |
|--------|---------|------|---------|---------|------|----------|
| ✅ Big-Math-RL-Verified | `SynthLabsAI/Big-Math-RL-Verified` | **251,122 题** | Apache-2.0 | **否(仅题+答案)** | 生成CoT | **RL 级严格验证**,单一可验证答案;含 olympiads 33K/HARP 3K/Omni-MATH 2.5K |
| DAPO-Math-17k | `BytedTsinghua-SIA/DAPO-Math-17k` | ~17K | Apache-2.0 | 否 | 生成CoT | RL 高密度难题 |
| DeepScaleR-Preview | `agentica-org/DeepScaleR-Preview-Dataset` | ~40K | MIT | 否 | 生成CoT | RL 难题集 |
| HARP | `HKUST-NLP/HARP` | ~5K | MIT | 部分 | 生成/复用 | 北美高中竞赛 |
| NuminaMath olympiad 子集 | `AI-MO/NuminaMath-CoT`(筛 source) | ~150K | Apache-2.0 | 是 | 复用CoT | 从主库切 olympiad 档,免费 |

> ⚠️ Big-Math 含 `Omni-MATH` 2478 条 —— **Omni-MATH 是评测集**,必须在去污染阶段剔除(见 §5 去污染)。

---

## 4. OOD 泛化档 (10% / ~2B)

**定位**:风格/来源不同的题,防过拟合,提泛化。

| 数据集 | HF 仓库 | 体量 | License | 自带CoT | 处理 | 选择理由 |
|--------|---------|------|---------|---------|------|----------|
| OpenThoughts3-1.2M | `open-thoughts/OpenThoughts3-1.2M` | ~1.2M | Apache-2.0 | 是(long-CoT) | 复用CoT | 长推理风格,来源多样 |
| DART-Math-Uniform/Hard | `hkust-nlp/dart-math-uniform` | ~591K | MIT | 是 | 复用CoT | 难度均衡采样,补分布 |

---

## 5. 汇总视图

### 按处理方式(决定成本)
| 处理方式 | 数据集 | 说明 |
|----------|--------|------|
| **复用 CoT**(省钱,优先) | NuminaMath / OpenR1 / OpenMathReasoning / OpenMathInstruct-2 / ORCA / MetaMathQA / OpenThoughts3 / DART | 只清洗校验,不重新生成 |
| **生成 CoT**(需 teacher 模型) | Big-Math / DAPO / DeepScaleR / HARP | 只有题+答案,须补解题过程 |

→ **绝大部分 token 来自"复用 CoT"**,只有 L4 的可验证题库走"生成"。性价比高、AI artifact 少。

### 按 license
全部为 **Apache-2.0 / MIT / CC-BY-4.0**,可商用可训练。CC-BY 类(OpenMathReasoning/OpenMathInstruct)注意保留署名。

### 去污染红线(§方案 3.4)
入库前对 problem/answer/中间步骤做 10-gram 黑名单扫描,以下**必须命中率 0%**:
`Omni-MATH, AIME 2024-2026, MATH-500, GSM8K test, GPQA-Diamond, OlympiadBench, ...`
→ 特别注意:Big-Math 里混有 Omni-MATH、NuminaMath 里混有 math/gsm8k,**都要按来源标签剔除对应评测集部分**。

---

## 6. 放量前待核验清单(诚实标注)

- [x] NuminaMath-CoT / OpenR1-Math-220k / OpenMathReasoning / Big-Math-RL-Verified — 已核验
- [ ] OpenMathInstruct-2 / ORCA-Math / MetaMathQA / DAPO / DeepScaleR / HARP / OpenThoughts3 / DART — 体量/license 未逐个核验
- [ ] 各数据集与评测集的**来源重叠**逐个排查(尤其 Big-Math↔Omni-MATH、NuminaMath↔MATH/GSM8K)
- [ ] 实际 token 数按选定 tokenizer 重新统计(表中为估算)
