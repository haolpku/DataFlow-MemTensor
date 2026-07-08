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

## 运行

两条 pipeline 都需要真实 LLM API(OpenAI 兼容端点):

```bash
export DF_API_KEY=sk-...
export DF_API_URL=http://.../v1/chat/completions
export DF_MODEL=gpt-4.1-mini          # 或 claude / deepseek / qwen ...
```

### 1. 多步证据推理

```bash
python -m dataflow_memtensor.pipelines.evidence_pipeline
```
流程:`种子题 → ReasoningEvidenceChainGenerator(LLM 生成证据链) → ReasoningEvidenceGroundingFilter(grounding+math_verify 过滤)`。
输出 `{instruction, evidences[], steps[], generated_golden_answer, num_hops, claim_binding_rate}`,
每条 ≥3 跳、每步 claim 绑 `evidence_id`、绑定率 ≥0.95、答案 math_verify 校验。

### 2. 长程交错思维

```bash
# 默认用内置 DictRetriever(仅演示);挂真实语料走 BM25:
export MEMTENSOR_CORPUS=data/math_corpus.jsonl
python -m dataflow_memtensor.pipelines.interleaved_pipeline
```
流程:`种子任务 → AgentExploreGenerator(MathSandboxClient) → QualityEvaluator → Filter → Selector`。
LLM 在 sandbox 里**自主决定每步调什么工具**(search/read/run_python/sympy_check/…),
`run_python`/`sympy_check` 为**真实执行**,`search`/`read` 从**真实语料检索**。

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
