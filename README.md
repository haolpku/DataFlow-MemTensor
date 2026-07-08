# DataFlow-MemTensor

数学 Mid-train 数据生产扩展包,构建在 [DataFlow](../DataFlow) 与 [DataFlow-Agent](../DataFlow-Agent) 之上。

对齐《MidTrain 数据方案》,提供两类数据的**可 scale** 生产算子与 pipeline:

| 数据类型 | 组件 | 对齐章节 |
|----------|------|----------|
| **多步证据推理** | `ReasoningEvidenceChainGenerator` + `ReasoningEvidenceGroundingFilter` | §3.3 / §5.3 / §6 |
| **长程交错思维** | `MathSandboxClient`(真实 sympy/python 工具 + 可插拔检索)+ DataFlow-Agent 轨迹栈 | §4.3 / §5.4 |

> 这是一个**独立扩展包**,不修改上游框架。算子挂在 DataFlow 的 `OPERATOR_REGISTRY` 上,
> 数据由**真实 LLM** 生成 —— 换 model / 题库 / 检索语料即可放量到 10B。

---

## 目录结构

```
DataFlow-MemTensor/
├── pyproject.toml
├── README.md
├── data/
│   ├── evidence_seed.jsonl        # 证据 pipeline 的种子题
│   └── math_corpus.jsonl          # 检索语料(FlashRAG/BM25 通用 schema: {id, contents})
└── dataflow_memtensor/
    ├── __init__.py                # import 即注册两个算子到 OPERATOR_REGISTRY
    ├── operators/
    │   ├── reasoning_evidence_chain_generator.py    # 生成 evidences[]+steps[](每步绑 evidence_id)
    │   └── reasoning_evidence_grounding_filter.py   # 按 绑定率/跳数/引用真实性/答案校验 过滤
    ├── sandbox/
    │   ├── math_client.py         # MathSandboxClient:7 工具,search/read 走可插拔检索
    │   └── retrievers.py          # DictRetriever / BM25Retriever / FlashRAGRetriever
    └── pipelines/
        ├── evidence_pipeline.py       # 多步证据推理 pipeline
        └── interleaved_pipeline.py    # 长程交错思维 pipeline
```

---

## 安装

依赖上游两个框架(本地源码树或 pip 包均可):

```bash
# 方式 A:两个框架已 pip 安装 (open-dataflow / dataflow-agent)
pip install -e .

# 方式 B:本地源码树未安装,用 PYTHONPATH 指过去
export PYTHONPATH="/path/to/DataFlow:/path/to/DataFlow-Agent:/path/to/DataFlow-MemTensor"
```

本机跑真实检索(BM25)与答案校验的额外依赖:
```bash
pip install rank_bm25 numpy math_verify
```

---

## 可复现数据工作流

两条 pipeline 都建立在 DataFlow 的 `FileStorage` 之上:**每调用一次 `storage.step()`,当前算子把输出落成一个独立的 jsonl 文件**,文件名 `{prefix}_step{N}.jsonl`。这带来三个可复现特性:

1. **每一步都可检查** —— 中间产物全部落盘,不是黑盒;
2. **可断点续跑** —— `FileStorage` 会读上一步的产物作为下一步输入;
3. **可精确定位数据流失** —— 每条样本在哪一步、因什么被过滤,一目了然。

### 数据流与落盘位置

```
                            输入                         每步产物(可检查)
多步证据推理:
  data/evidence_seed.jsonl
        │
        ├─[step1] ReasoningEvidenceChainGenerator  → cache_evidence/evidence_step_step1.jsonl  (LLM 生成,未过滤)
        └─[step2] ReasoningEvidenceGroundingFilter → cache_evidence/evidence_step_step2.jsonl  (grounding+math_verify 过滤后)

长程交错思维:
  SEED_TASKS (或自定义)
        │
        ├─[step1] AgentExploreGenerator      → cache_interleaved/interleaved_step_step1.jsonl  (轨迹生成,真实工具调用)
        ├─[step2] TrajectoryQualityEvaluator → cache_interleaved/interleaved_step_step2.jsonl  (+ traj_overall 打分)
        ├─[step3] TrajectoryFilter           → cache_interleaved/interleaved_step_step3.jsonl  (规则门控后)
        └─[step4] TrajectorySelector         → cache_interleaved/interleaved_step_step4.jsonl  (top-N 选择)
                                              → interleaved_output.jsonl  (最终交付)
```

> 说明:cache 目录落在**运行时的当前目录(cwd)** 下,可用 pipeline 的 `cache_path` 参数改到固定位置。

### 端到端复现步骤

```bash
# 0) 依赖框架(源码树用 PYTHONPATH;绝对路径,避免切目录后失效)
BASE=/path/to
export PYTHONPATH="$BASE/DataFlow:$BASE/DataFlow-Agent:$BASE/DataFlow-MemTensor"
pip install rank_bm25 numpy math_verify     # 本机真实检索 + 答案校验

# 1) LLM API(OpenAI 兼容端点)
export DF_API_KEY=sk-...
export DF_API_URL=http://.../v1/chat/completions
export DF_MODEL=gpt-4.1-mini                 # 或 claude / deepseek / qwen ...

# 2) 准备输入(见 data/README.md 的 schema)
#    data/evidence_seed.jsonl   种子题
#    data/math_corpus.jsonl     检索语料
export MEMTENSOR_CORPUS=$BASE/DataFlow-MemTensor/data/math_corpus.jsonl

# 3) 跑两条 pipeline
python -m dataflow_memtensor.pipelines.evidence_pipeline
python -m dataflow_memtensor.pipelines.interleaved_pipeline

# 4) 检查每一步中间结果
head -1 cache_evidence/evidence_step_step1.jsonl | python -m json.tool      # 生成器原始输出
wc -l cache_evidence/evidence_step_step*.jsonl                              # 每步剩多少条
```

### 逐步追踪一条样本(定位过滤原因)

```python
import json
for s in [1, 2, 3, 4]:
    rows = [json.loads(l) for l in open(f"cache_interleaved/interleaved_step_step{s}.jsonl")]
    print(f"step{s}: {len(rows)} 条")
# 例:某题在 step1 已 success=False,step2 打分 0.4,
#     step3 的 TrafectoryFilter(require_success=True) 将其剔除 —— 未解出的样本不进训练集。
```

### 复现的确定性边界

- **可完全复现**:算子逻辑、过滤规则、落盘结构、检索(BM25 确定性)。相同输入 + 相同中间产物 → 相同过滤/选择结果。
- **不完全复现**:LLM 生成本身随温度/服务端有随机性(证据链措辞、轨迹步数会变)。要更强复现性:`DF_MODEL` 固定、温度设 0(`APILLMServing_request(temperature=0)`)、并保留 cache 目录作为该次运行的快照。

---

## 运行(速查)

两条 pipeline 都需要真实 LLM API(见上节完整步骤):

```bash
export DF_API_KEY=sk-...  DF_API_URL=http://.../v1/chat/completions  DF_MODEL=gpt-4.1-mini

# 多步证据推理: 种子题 → 生成证据链 → grounding+math_verify 过滤
python -m dataflow_memtensor.pipelines.evidence_pipeline

# 长程交错思维: 种子任务 → 轨迹生成 → 打分 → 过滤 → 选择
export MEMTENSOR_CORPUS=data/math_corpus.jsonl   # 挂真实语料走 BM25;不设则用内置 Dict 兜底
python -m dataflow_memtensor.pipelines.interleaved_pipeline
```

---

## 可 scale 到 10B 的关键:检索后端可插拔

`search`/`read` 的知识来源由 `RetrieverABC` 抽象,三档实现覆盖 demo → 生产:

| 后端 | 用途 | 规模 | 依赖 |
|------|------|------|------|
| `DictRetriever` | 零配置兜底 / 单测 | 几条 | 无 |
| `BM25Retriever` | 本机真实检索、验证接口 | ~1e5–1e6 文档 | `rank_bm25` |
| `FlashRAGRetriever` | **生产**:e5/faiss 稠密检索 | 百万级语料 | `flashrag` + GPU 集群 |

切换只需一行(pipeline 里已按 `MEMTENSOR_CORPUS` 自动选择):
```python
from dataflow_memtensor import MathSandboxClient, BM25Retriever, FlashRAGRetriever

# 本机验证
sandbox = MathSandboxClient(retriever=BM25Retriever(corpus_path="data/math_corpus.jsonl"))

# 生产(GPU 集群,百万级语料)
sandbox = MathSandboxClient(retriever=FlashRAGRetriever(
    retrieval_model_path="/models/e5", index_path="/idx/e5.index",
    corpus_path="/corpus/math_10M.jsonl", faiss_gpu=True, topk=5))
```

---

## 从 demo 到 10B 生产:仍需补齐的部分

当前包已把最核心的接口打通(算子挂框架、真实 LLM 生成、真实检索可插拔、真实工具执行)。
放量到 10B tokens 前,以下仍需在**集群**上补齐(本机不验证):

- [ ] **题库**:`data/*_seed.jsonl` 换成 §3.2 的真实题库(NuminaMath / OpenMathReasoning / OpenR1 …),`FileStorage` 支持 `hf:` 前缀直读 HF。
- [ ] **检索语料**:`math_corpus.jsonl` 换成百万级数学语料 + 建 e5/faiss 索引,切 `FlashRAGRetriever`。
- [ ] **分布式 serving**:`APILLMServing_request` 的 `max_workers` 调大,或换 Ray 批量后端。
- [ ] **verifier 栈**:现有 grounding + math_verify + sympy;补 Pass@k 双向过滤、多模型交叉、LLM judge(§6 五层)。
- [ ] **污染控制**:接入 10-gram 黑名单扫描(AIME/MATH-500/GSM8K…),命中率 0%(§5.1)。
- [ ] **分片与血缘**:大规模下 `FileStorage` 分片写 + manifest 记录来源/难度/长度(§5 交付要求)。

---

## 验证记录

用真实 API(`gpt-4.1-mini`)端到端跑通:

| 检查 | 结果 |
|------|------|
| 包 import + 两算子注册到 OPERATOR_REGISTRY | 通过 |
| `MathSandboxClient` 7 工具 + 真实 sympy + `os` 拦截 | 通过 |
| `BM25Retriever` 在 12 文档语料上检索 | 命中合理(vieta/chord_distance/quadratic_min) |
| 证据 pipeline 端到端 | 5/5 通过 grounding+math_verify,绑定率 1.0,≥3 跳 |
| 交错 pipeline 端到端(BM25 真实检索) | 3/3 成功;LLM 自主决策;search 命中语料;答案 -8/12/-12 正确 |

许可证:Apache-2.0(随上游)。
