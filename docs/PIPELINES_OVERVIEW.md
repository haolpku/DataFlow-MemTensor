# 三条 Demo Pipeline 总览

> 供 review。三条 pipeline 的算子链、输入输出、关键参数、校验规则、真实产物。
> 代码位置:`dataflow_memtensor/pipelines/` + `dataflow_memtensor/operators/` + `dataflow_memtensor/sandbox/`。

## 共性

- 都建在 DataFlow 的 `FileStorage` 上,**每步落盘**(`cache_*/…_step{N}.jsonl`),可断点续跑、可逐步审计。
- LLM 走 `APILLMServing_request`(真实 API,由 `DF_API_KEY` / `DF_API_URL` / `DF_MODEL` 配置)。
- 本次验证:`gpt-4.1-mini`,三条各跑 **24 条**,全部通过。
- 运行方式:`python -m dataflow_memtensor.pipelines.<name>_pipeline`。

---

## ① 长思维链 Long-CoT — `cot_pipeline.py`

**目标**:可验证数学题的长推理(对齐《MidTrain 方案》§4.2)。
是 DataFlow `reasoning_math_pipeline`(11 步)的**精简版**,只留"生成 + 答案校验"两步(去掉题目合成/难度/分类/格式/长度/ngram 等步)。

```
seed(题 + golden_answer)
  → [S0]  ProvenanceOperator         # 盖血缘:problem_source/synthetic_flag/gen_model/created_at
  → [S0.5]DifficultyTagOperator      # 按来源标签打 L1-L4(LLM 兜底未命中的)
  → [S1]  ReasoningLongCoTGenerator  # LLM 生成 <think>长推理</think> + \boxed{答案}
  → [S2]  CoTQualityFilter           # 剔除空壳/复读机/仅复述题面   ┐
  → [S3]  ReasoningCoTAnswerFilter   # 抽 boxed → math_verify 校验   ├─ 淘汰样本 → failure_pool.jsonl
  → [S4]  DecontaminationFilter      # 评测集 10-gram 黑名单(红线) ┘  (带 _drop_stage/_drop_reason)
  → 通过入库(不合格的落失败池,不再静默丢弃)
```

| 算子 | 来源 | 作用 | 关键参数 |
|------|------|------|----------|
| `ProvenanceOperator` | 本仓库 | 盖血缘字段(来源/是否合成/生成模型/时间戳),已有值不覆盖 | `problem_source`,`synthetic_flag`,`gen_model`,`pipeline` |
| `DifficultyTagOperator` | 本仓库 | 来源标签初分 L1-L4 + LLM 兜底,可选按配比抽样 | `llm_serving`(可选),`target_ratio`,`force_llm` |
| `ReasoningLongCoTGenerator` | 本仓库 | 生成 CoT,抽 `\boxed{}` 到 `extracted_answer` | — |
| `CoTQualityFilter` | 本仓库 | 剔除空壳(think过短)/复读(n-gram自重复)/复述题面 | `min_think_chars=120`,`min_distinct_ratio=0.35`,`failure_pool_path` |
| `ReasoningCoTAnswerFilter` | 本仓库 | 答案校验过滤 | `compare_method="math_verify"`,`require_think_tag=True`,`failure_pool_path` |
| `DecontaminationFilter` | 本仓库 | 评测集去污染(红线) | `benchmark_file`,`ngram=10`,`failure_pool_path` |

- **产物字段**:`instruction / generated_cot / extracted_answer / golden_answer / difficulty / problem_source / synthetic_flag / gen_model / created_at`
- **落盘**:`cache_cot/cot_step_step1..6.jsonl`(血缘/难度/生成/质量/校验/去污染)+ `cache_cot/failure_pool.jsonl`(全部被淘汰样本)
- **通过率与失败池**:每个 filter 都打印 `[pass_rate] kept X/Y (pass_rate Z%)`,并把被淘汰行(附 `_drop_stage`/`_drop_reason`)追加进失败池——**交付时通过率与失败池一并给出**(对齐核查文档 §3/§七)。
- **真实结果**:24/24 通过质量+答案校验(`math_verify` 正确识别 `\frac{40}{3}` == `40/3`)。去污染在测试黑名单上验证有效(揪出混入的 gcd、7^100 两道评测题)。

---

## ② 多步证据推理 Evidence — `evidence_pipeline.py`

**目标**:证据接地(对齐 §5.3)。核心是**带干扰项**,逼模型"筛选证据"而非照抄。

```
seed(题 + golden_answer)
  → [S1] ReasoningEvidenceChainGenerator  # LLM 生成 证据簇 + 干扰项 + 推理链(每步 claim 绑 evidence_id)
                                          #   干扰项与真证据混合、打乱、重编号
  → [S2] ReasoningEvidenceGroundingFilter # 绑定率 / 跳数 / 引用真实性 / 干扰未引用 / 答案 校验
  → 通过入库
```

| 算子 | 来源 | 作用 | 关键参数 |
|------|------|------|----------|
| `ReasoningEvidenceChainGenerator` | 本仓库 | 生成 evidences[]+distractors,混合打乱重编号,steps 每步绑 evidence_id | `add_distractors=True`,`shuffle_seed` |
| `ReasoningEvidenceGroundingFilter` | 本仓库 | grounding 过滤 | `min_binding_rate=0.95`,`min_hops=3`,`forbid_citing_distractors=True`,`compare_method="math_verify"` |

**过滤规则**(全过才留):
1. claim-证据绑定率 ≥ 0.95
2. 推理 ≥ 3 跳
3. 每步引用的 evidence_id 真实存在(无悬空引用)
4. **干扰项不被任何 step 引用**(证明模型做了筛选)
5. golden_answer 经 math_verify 校验

- **产物字段**:`question / evidences[] / distractor_ids / steps[]{step,claim,evidence_ids,derivation} / num_hops / claim_binding_rate`
- **落盘**:`cache_evidence/evidence_step_step1.jsonl`(生成)→ `step2`(过滤后)
- **真实结果**:24/24 通过,平均 6.3 证据 / 3 干扰,**0 引用泄漏**。干扰质量高(问"根之积"时混入"根之和"公式作硬负例)。

---

## ③ 长程交错思维 Interleaved — `interleaved_pipeline.py`

**目标**:工具轨迹(对齐 §4.3 / §5.4)。
**串联的是 DataFlow-Agent 现有 4 个算子**,本仓库只新写了 `MathSandboxClient`(真实工具后端)。

```
seed(题 → 包装成 agent 任务)
  → [S1] AgentExploreGenerator(MathSandboxClient)  # LLM 自主 (思考→工具→观察) 轨迹
  → [S2] TrajectoryQualityEvaluator                # LLM-as-judge 四轴打分
  → [S3] TrajectoryFilter                          # 规则门控
  → [S4] TrajectorySelector                        # top-N 多样性选择
```

| 阶段 | 算子 | 来源 | 关键参数 |
|------|------|------|----------|
| S1 | `AgentExploreGenerator` | DataFlow-Agent | `max_steps=8`,`max_workers=4` |
| — | `MathSandboxClient` | **本仓库新建** | 7 工具:search/read/**run_python/sympy_check(真实执行)**/select_evidence/synthesize/finish;检索走可插拔 `BM25Retriever`(50 条语料) |
| S2 | `TrajectoryQualityEvaluator` | DataFlow-Agent | 四轴:goal / efficiency / coherence / tool_use + overall。**判分用独立模型**(`DF_JUDGE_MODEL`/`DF_JUDGE_API_URL`),与生成器不同源(核查文档 §2) |
| S3 | `TrajectoryFilter` | DataFlow-Agent | `require_success=True`,`min_steps=2`,`drop_parse_errors=True`,`drop_invalid_tools=True`,`require_nonempty_answer=True` |
| S4 | `TrajectorySelector` | DataFlow-Agent | `max_selected=50`,`min_depth=2`,`mode="rows"` |

- **产物字段**:`task / steps[]{thought,action{tool,args},observation,ok} / final_answer / success / num_steps / traj_goal_achievement / traj_efficiency / traj_coherence / traj_tool_use / traj_overall / traj_rationale`
- **落盘**:`cache_interleaved/interleaved_step_step1..4.jsonl`(生成/打分/过滤/选择)
- **检索后端**:`MEMTENSOR_CORPUS` 指向语料 → 真实 BM25;不设则用内置 DictRetriever 兜底。
- **真实结果**:24/24 成功。observation 是 **sympy 真算**结果;模型会**自我纠错**(如 sympy_check 失败后改用真实 solve 重算)。

---

## 对比表

| | Long-CoT | Evidence | Interleaved |
|---|---|---|---|
| 训练目标能力 | 一步步算对 | 找证据 / 绑证据 | 调工具 / 多步纠错 |
| 生成算子来源 | 本仓库新建 | 本仓库新建 | **复用 DataFlow-Agent** |
| 校验核心 | math_verify 答案 | 绑定率 + 干扰未引用 + 答案 | 四轴打分 + 规则门控 |
| 工具执行 | 无 | 无 | **真实 sympy / python** |
| 对齐方案章节 | §4.2 | §5.3 | §4.3 / §5.4 |
| 本次 24 条通过 | 24/24 | 24/24 | 24/24 |

---

## 针对核查文档的改进(本轮)

核查文档 §六把"指标可不可信"收敛为三项检查:**来源是否真实、判分是否独立、淘汰了多少**。本轮针对其中的纯代码项落地:

| 核查文档 | 改进 | 落地 |
|---|---|---|
| §2 生成与判分同源 | interleaved 判分改用**独立模型** | `build_judge_llm()`:读 `DF_JUDGE_MODEL`/`DF_JUDGE_API_URL`/`DF_JUDGE_API_KEY`;未配置则回退并**告警**(不再静默同源) |
| §3 / §七 只交付通过样本,无通过率与失败池 | 四个 filter 都把淘汰样本落**失败池** + 打印**通过率** | `operators/failure_pool.py`;每行带 `_drop_stage`/`_drop_reason`,失败池即 PRM/负样本资产 |
| §5 无来源/合成标记字段 | 新增**血缘算子**,pipeline 最前盖字段 | `ProvenanceOperator`:`problem_source`/`synthetic_flag`/`gen_model`/`pipeline`/`created_at`;已有值不覆盖 |
| §一.1 / L4 缺失 | 新增**难度分层器** | `DifficultyTagOperator`:来源标签初分 L1-L4 + LLM 兜底 + 可选配比抽样;L4=0 时显式告警 |

## Review 时仍需知道的局限(诚实标注)

1. **Evidence 的证据仍是 LLM 脑补的**,不是从真实语料检索来的——生产应接真实检索(`retrievers.py` 已留 `FlashRAGRetriever` 接口)。
2. **Interleaved 的 search 检索 50 条小语料**,工具执行(run_python/sympy)是真的,但检索池是 demo 级。
3. **难度分层的来源初分依赖题库自带的来源标签**;demo 自造题无标签,需靠 `DifficultyTagOperator` 的 LLM 兜底,准确度未在真实题库上标定。
4. **判分独立性靠配置保证**:必须实际设置 `DF_JUDGE_MODEL` 指向不同来源的模型才生效;不设只会回退+告警,不会自动变独立。

这些都是"从 demo 到生产"要补的(见 `PIPELINE_DESIGN_20B.md` 待建清单)。
