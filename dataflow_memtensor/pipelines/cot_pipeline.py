"""
cot_pipeline.py — 长思维链 (long-CoT) 数据生产 pipeline (DataFlow-MemTensor).

DataFlow 的 reasoning_math_pipeline 有 11 步(题过滤→合成→再过滤→难度→分类→答案生成
→格式过滤→长度过滤→答案校验→ngram 过滤)。这里砍成最核心的两步:

    seed questions (FileStorage)
        -> ReasoningLongCoTGenerator   # LLM 生成 <think>长推理</think> + \\boxed{答案}
        -> ReasoningCoTAnswerFilter    # 抽 boxed,与 golden_answer 做 math_verify 过滤
        -> 长 CoT 数据

对齐《MidTrain数据方案》§4.2 可验证数学题。答案可被机器独立校验。

运行(需真实 API):
    export DF_API_KEY=sk-...
    export DF_API_URL=http://.../v1/chat/completions
    export DF_MODEL=gpt-4.1-mini
    python -m dataflow_memtensor.pipelines.cot_pipeline

每步中间结果落在 ./cache_cot/cot_step_step{1,2}.jsonl(step1 生成 / step2 过滤后)。
"""

import json
import os

from dataflow.utils.storage import FileStorage
from dataflow.core import LLMServingABC

from dataflow_memtensor.operators import (
    ReasoningLongCoTGenerator,
    ReasoningCoTAnswerFilter,
    CoTQualityFilter,
    DecontaminationFilter,
    ProvenanceOperator,
    DifficultyTagOperator,
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
    try:
        from dataflow.serving import APILLMServing_request
    except ModuleNotFoundError:
        from dataflow.serving.api_llm_serving_request import APILLMServing_request
    return APILLMServing_request(
        api_url=os.environ.get("DF_API_URL", "https://api.openai.com/v1/chat/completions"),
        model_name=os.environ.get("DF_MODEL", "gpt-4o"),
        max_workers=int(os.environ.get("DF_MAX_WORKERS", "50")),
    )


class LongCoTPipeline:
    def __init__(self, seed_file: str = _DEFAULT_SEED, llm_serving: LLMServingABC = None,
                 cache_path: str = "./cache_cot", decontam_benchmark_file: str = None,
                 failure_pool_path: str = None, tag_difficulty: bool = True):
        self.storage = FileStorage(
            first_entry_file_name=seed_file,
            cache_path=cache_path,
            file_name_prefix="cot_step",
            cache_type="jsonl",
        )
        self.llm_serving = llm_serving or build_llm()
        # 失败池:各 filter 把被淘汰样本(带原因)追加到此文件,供统计通过率/失败池
        # (核查文档 §3/§七)。默认落在 cache 目录下。
        self.failure_pool_path = failure_pool_path or os.path.join(cache_path, "failure_pool.jsonl")
        self.tag_difficulty = tag_difficulty

        # S0 血缘标记(核查文档 §5 最小字段):problem_source / synthetic_flag / gen_model ...
        self.provenance = ProvenanceOperator(
            gen_model=os.environ.get("DF_MODEL", ""),
            pipeline="cot",
            synthetic_flag=True,  # CoT 由 LLM 生成
        )
        # S0.5 难度分层(核查文档 L4 缺失):来源标签初分,可选 LLM 兜底。
        # 这里只打标不抽样(不传 target_ratio),避免 demo 小样本被抽空。
        self.difficulty_tagger = DifficultyTagOperator(llm_serving=self.llm_serving)

        self.cot_generator = ReasoningLongCoTGenerator(llm_serving=self.llm_serving)
        self.quality_filter = CoTQualityFilter(
            min_think_chars=120,
            min_distinct_ratio=0.35,
            max_restate_overlap=0.92,
            failure_pool_path=self.failure_pool_path,
        )
        self.answer_filter = ReasoningCoTAnswerFilter(
            compare_method="math_verify",
            require_think_tag=True,
            failure_pool_path=self.failure_pool_path,
        )
        # 去污染:默认无黑名单时不剔除并告警;放量前传 benchmark_file。
        self.decontam = DecontaminationFilter(
            benchmark_file=decontam_benchmark_file,
            ngram=10, overlap_threshold=0.5,
            failure_pool_path=self.failure_pool_path,
        )

    def forward(self):
        # S0 血缘标记(每条盖 source/synthetic_flag/gen_model/created_at,后续步骤都带着它)
        self.provenance.run(storage=self.storage.step())
        # S0.5 难度分层(按来源标签打 L1-L4;LLM 兜底未命中的)
        if self.tag_difficulty:
            self.difficulty_tagger.run(storage=self.storage.step(), input_key="instruction")
        # S1 生成 CoT
        self.cot_generator.run(
            storage=self.storage.step(),
            input_key="instruction",
            output_key="generated_cot",
            output_answer_key="extracted_answer",
        )
        # S2 质量过滤(空壳/复读/复述题面)—— 便宜,先扔没营养的
        self.quality_filter.run(
            storage=self.storage.step(),
            input_cot_key="generated_cot",
            input_question_key="instruction",
        )
        # S3 答案校验(math_verify)
        self.answer_filter.run(
            storage=self.storage.step(),
            input_cot_key="generated_cot",
            input_answer_key="extracted_answer",
            input_gt_key="golden_answer",
        )
        # S4 去污染(评测集 10-gram 黑名单)—— 红线
        self.decontam.run(storage=self.storage.step())


def main():
    pipeline = LongCoTPipeline()
    pipeline.forward()

    df = pipeline.storage.step().read(output_type="dataframe")
    print(f"\n[done] {len(df)} 条通过答案校验的长 CoT 样本")
    for _, row in df.iterrows():
        cot = str(row.get("generated_cot", ""))
        print(f"\n=== {str(row['instruction'])[:60]} ===")
        print(f"  golden={row.get('golden_answer')!r}  extracted={row.get('extracted_answer')!r}  "
              f"cot_chars={len(cot)}")


if __name__ == "__main__":
    main()
