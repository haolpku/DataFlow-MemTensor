"""
evidence_pipeline.py — 多步证据推理数据生产 pipeline (DataFlow-MemTensor).

    seed questions (FileStorage)
        -> ReasoningEvidenceChainGenerator   # LLM 生成 evidences[] + steps[](每步 claim 绑 evidence_id)
        -> ReasoningEvidenceGroundingFilter  # 按 绑定率>=0.95 / >=3跳 / 证据引用真实 / 答案校验 过滤
        -> 多步证据推理数据

对齐《MidTrain数据方案》§3.3 / §5.3 / §6。算子挂在 DataFlow 的 OPERATOR_REGISTRY 上,
题目/证据/推理链由**真实 LLM** 生成,可 scale 到任意题库。

运行(需真实 API):
    export DF_API_KEY=sk-...
    export DF_API_URL=http://.../v1/chat/completions
    export DF_MODEL=gpt-4.1-mini
    python -m dataflow_memtensor.pipelines.evidence_pipeline
"""

import json
import os

from dataflow.utils.storage import FileStorage
from dataflow.core import LLMServingABC

from dataflow_memtensor.operators import (
    ReasoningEvidenceChainGenerator,
    ReasoningEvidenceGroundingFilter,
    ProvenanceOperator,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DEFAULT_SEED = os.path.join(_REPO_ROOT, "data", "evidence_seed.jsonl")


def build_llm():
    """构建真实 API LLM serving。必须设置 DF_API_KEY。"""
    if not os.environ.get("DF_API_KEY"):
        raise EnvironmentError(
            "DF_API_KEY 未设置。请先导出:\n"
            "  export DF_API_KEY=sk-...\n"
            "  export DF_API_URL=http://.../v1/chat/completions\n"
            "  export DF_MODEL=gpt-4.1-mini")
    # 直接从模块文件导入,绕过 dataflow.serving.__init__ 顶层的 torch 依赖。
    try:
        from dataflow.serving import APILLMServing_request
    except ModuleNotFoundError:
        from dataflow.serving.api_llm_serving_request import APILLMServing_request
    return APILLMServing_request(
        api_url=os.environ.get("DF_API_URL", "https://api.openai.com/v1/chat/completions"),
        model_name=os.environ.get("DF_MODEL", "gpt-4o"),
        max_workers=int(os.environ.get("DF_MAX_WORKERS", "50")),
    )


class EvidenceReasoningPipeline:
    def __init__(self, seed_file: str = _DEFAULT_SEED, llm_serving: LLMServingABC = None,
                 cache_path: str = "./cache_evidence", failure_pool_path: str = None):
        self.storage = FileStorage(
            first_entry_file_name=seed_file,
            cache_path=cache_path,
            file_name_prefix="evidence_step",
            cache_type="jsonl",
        )
        self.llm_serving = llm_serving or build_llm()
        # 失败池:grounding 不达标样本落盘,供统计通过率(核查文档 §3/§七)
        self.failure_pool_path = failure_pool_path or os.path.join(cache_path, "failure_pool.jsonl")

        self.provenance = ProvenanceOperator(
            gen_model=os.environ.get("DF_MODEL", ""), pipeline="evidence", synthetic_flag=True,
        )
        self.evidence_generator = ReasoningEvidenceChainGenerator(
            llm_serving=self.llm_serving,
        )
        self.grounding_filter = ReasoningEvidenceGroundingFilter(
            min_binding_rate=0.95,
            min_hops=3,
            check_answer_against="golden_answer",
            compare_method="math_verify",
            failure_pool_path=self.failure_pool_path,
        )

    def forward(self):
        # S0 血缘标记(核查文档 §5)
        self.provenance.run(storage=self.storage.step())
        self.evidence_generator.run(
            storage=self.storage.step(),
            input_key="instruction",
            output_evidences_key="evidences",
            output_steps_key="steps",
            output_answer_key="generated_golden_answer",
        )
        self.grounding_filter.run(
            storage=self.storage.step(),
            input_evidences_key="evidences",
            input_steps_key="steps",
            input_answer_key="generated_golden_answer",
        )


def main():
    pipeline = EvidenceReasoningPipeline()
    pipeline.forward()

    df = pipeline.storage.step().read(output_type="dataframe")
    print(f"\n[done] {len(df)} 条通过 grounding 过滤的多步证据推理样本")
    for _, row in df.iterrows():
        steps = row["steps"]
        if isinstance(steps, str):
            steps = json.loads(steps)
        print(f"\n=== {str(row['instruction'])[:60]} ===")
        print(f"  hops={len(steps)} binding_rate={row.get('claim_binding_rate')} "
              f"answer={row.get('generated_golden_answer')!r}")
        for s in steps:
            print(f"  step {s['step']}: {s['claim'][:55]}  <- {s['evidence_ids']}")


if __name__ == "__main__":
    main()
