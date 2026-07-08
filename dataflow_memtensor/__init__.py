"""
DataFlow-MemTensor — 数学 Mid-train 数据生产扩展包.

在 DataFlow / DataFlow-Agent 之上,提供两类数据的可 scale 生产算子与 pipeline:

  1. 多步证据推理 (multi-step evidence reasoning)
       - ReasoningEvidenceChainGenerator : LLM 生成 evidences[] + steps[](每步 claim 绑 evidence_id)
       - ReasoningEvidenceGroundingFilter: 按 绑定率/跳数/证据引用真实性/答案校验 过滤

  2. 长程交错思维 (long-horizon interleaved r/a/o)
       - MathSandboxClient : 真实 sympy/python 工具 + 可插拔检索 (DictRetriever/BM25Retriever/FlashRAGRetriever)
       - 复用 DataFlow-Agent 的 AgentExploreGenerator/Evaluator/Filter/Selector

对齐《MidTrain数据方案》§3.3/§4.3/§5.3/§5.4/§6。

依赖: open-dataflow (提供 OperatorABC/LLMServingABC/OPERATOR_REGISTRY/storage),
       dataflow-agent (提供 SandboxClientABC 与轨迹算子)。
"""

# 导入算子模块 -> 触发 @OPERATOR_REGISTRY.register(),使其可被 pipeline 用名字取到。
from .operators import (
    ReasoningEvidenceChainGenerator,
    ReasoningEvidenceGroundingFilter,
)
from .sandbox import MathSandboxClient
from .sandbox.retrievers import (
    RetrieverABC,
    DictRetriever,
    BM25Retriever,
    FlashRAGRetriever,
)

__all__ = [
    "ReasoningEvidenceChainGenerator",
    "ReasoningEvidenceGroundingFilter",
    "MathSandboxClient",
    "RetrieverABC",
    "DictRetriever",
    "BM25Retriever",
    "FlashRAGRetriever",
]

__version__ = "0.1.0"
