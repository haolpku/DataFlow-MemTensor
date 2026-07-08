"""
interleaved_pipeline.py — 长程交错思维 (interleaved r/a/o) 数据生产 pipeline (DataFlow-MemTensor).

串联 DataFlow-Agent 现有算子,把 MathSandboxClient(真实 sympy/python 工具 + 可插拔检索)接进去:

    seed tasks (FileStorage)
        -> AgentExploreGenerator(MathSandboxClient)   # 生成 (thought,action,observation) 轨迹
        -> TrajectoryQualityEvaluator                 # LLM-as-judge 四轴打分
        -> TrajectoryFilter                           # 规则门控
        -> TrajectorySelector                         # top-N 多样性选择
        -> interleaved trajectories (jsonl)

对齐《MidTrain数据方案》§4.3 / §5.4:interleaved (r,a,o) + 真实工具调用 + observation 留痕。
LLM 在 sandbox 里自主决定每步调什么工具 —— 换 model / seed / 检索语料即可 scale。

检索后端(search/read 的来源)可选:
  * 默认: DictRetriever(内置几条定理,仅演示);
  * 传 MEMTENSOR_CORPUS=path/to/corpus.jsonl -> 用 BM25Retriever 在真实语料上检索;
  * 生产: FlashRAGRetriever(e5/faiss,百万级语料,GPU 集群),见 sandbox/retrievers.py。

运行(需真实 API):
    export DF_API_KEY=sk-...
    export DF_API_URL=http://.../v1/chat/completions
    export DF_MODEL=gpt-4.1-mini
    # 可选: export MEMTENSOR_CORPUS=data/math_corpus.jsonl
    python -m dataflow_memtensor.pipelines.interleaved_pipeline
"""

import json
import os
import tempfile

import pandas as pd

from dataflow.utils.storage import FileStorage
from dataflow_agent.generate.agent_explore_generator import AgentExploreGenerator
from dataflow_agent.eval.trajectory_quality_evaluator import TrajectoryQualityEvaluator
from dataflow_agent.filter.trajectory_filter import TrajectoryFilter
from dataflow_agent.select.trajectory_selector import TrajectorySelector

from dataflow_memtensor.sandbox import MathSandboxClient
from dataflow_memtensor.sandbox.retrievers import BM25Retriever

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


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
        max_workers=int(os.environ.get("DF_MAX_WORKERS", "16")),
    )


def build_sandbox():
    """按 MEMTENSOR_CORPUS 选择检索后端:有语料走真实 BM25,否则用内置 Dict 兜底。"""
    corpus = os.environ.get("MEMTENSOR_CORPUS")
    if corpus and os.path.exists(corpus):
        retriever = BM25Retriever(corpus_path=corpus)
        print(f"[info] BM25Retriever loaded {len(retriever)} docs from {corpus}")
        return MathSandboxClient(retriever=retriever)
    print("[info] 未设 MEMTENSOR_CORPUS,使用内置 DictRetriever(仅演示,生产请挂真实语料)。")
    return MathSandboxClient()


# 种子题(真实使用时从 HF / 题库读入,可任意扩展)
SEED_TASKS = [
    "Find the product of the two roots of x^2 - 5x - 8 = 0. Recall the theorem, compute, and verify.",
    "Find the exact distance from the center to a chord of length 10 in a circle of radius 13.",
    "Find the minimum value of f(x) = 3x^2 - 12x over the reals; recall calculus facts, compute, verify.",
]


def main(seed_tasks=None, out_path=None):
    seed_tasks = seed_tasks or SEED_TASKS
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "seed_tasks.jsonl")
    pd.DataFrame([{"query": t} for t in seed_tasks]).to_json(
        src, orient="records", lines=True, force_ascii=False)

    storage = FileStorage(
        first_entry_file_name=src,
        cache_path=os.path.join(tmp, "cache"),
        cache_type="jsonl",
    )

    llm = build_llm()
    sandbox = build_sandbox()

    # Stage 1: 生成交错轨迹(真实工具调用)
    AgentExploreGenerator(
        llm_serving=llm, sandbox=sandbox, domain="math",
        max_steps=8, max_workers=4,
    ).run(storage.step(), input_key="query", output_key="trajectory")

    # Stage 2: LLM-as-judge 四轴打分
    TrajectoryQualityEvaluator(
        llm_serving=llm, max_workers=4,
    ).run(storage.step(), input_key="trajectory", output_key="traj_overall")

    # Stage 3: 规则门控
    TrajectoryFilter(
        require_success=True, min_steps=2, drop_parse_errors=True,
        drop_invalid_tools=True, require_nonempty_answer=True,
    ).run(storage.step(), input_key="trajectory")

    # Stage 4: top-N 多样性选择
    TrajectorySelector(
        max_selected=10, min_depth=2, mode="rows",
    ).run(storage.step(), input_key="trajectory", output_key="selected_trajectories")

    df = storage.step().read(output_type="dataframe")
    out_path = out_path or os.path.join(_REPO_ROOT, "interleaved_output.jsonl")
    df.to_json(out_path, orient="records", lines=True, force_ascii=False)

    print(f"\n[done] {len(df)} 条交错轨迹 -> {out_path}")
    for _, row in df.iterrows():
        traj = row["trajectory"]
        if isinstance(traj, str):
            traj = json.loads(traj)
        print(f"\n=== {traj['task'][:70]} ===")
        print(f"  success={traj['success']} steps={traj['num_steps']} "
              f"answer={traj['final_answer']!r} overall={row.get('traj_overall')}")
        for i, step in enumerate(traj["steps"], 1):
            print(f"  step {i}: [{step['action']['tool']}] "
                  f"obs={json.dumps(step['observation'], ensure_ascii=False)[:70]}")
    return out_path


if __name__ == "__main__":
    main()
