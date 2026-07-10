from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

import os
import re
import json
import pandas as pd
from typing import List, Optional


def _normalize(text: str) -> List[str]:
    """小写、去标点、按空白分词 —— 用于 n-gram 比对(与题面表述无关的规范化)。"""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _ngrams(tokens: List[str], n: int):
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


@OPERATOR_REGISTRY.register()
class DecontaminationFilter(OperatorABC):
    """
    评测集去污染过滤器(对齐《MidTrain 方案》§3.4 / §5.1)。

    训练数据绝不能包含评测集,否则"涨分"是假的。本算子对指定字段(题面/答案/
    关键中间步骤)做 **n-gram 黑名单扫描**:把评测集问题切成 n-gram 集合,若一条
    样本与任一评测集问题的 n-gram 重叠比例超过阈值,即判为污染并剔除。

    黑名单来源两种方式(可叠加):
      - benchmark_file: jsonl,每行含评测集问题(字段 problem/question/instruction 之一)
      - benchmark_ngrams: 预构建的 n-gram 集合(内存传入)

    产出:被剔除样本数 + 命中率(应为 0% 才达标),并把命中详情写日志。
    这是放量到 20B 前必须接入的红线算子。
    """

    # 方案 §3.4 点名的评测集/高风险集合(仅作提示与默认清单)
    DEFAULT_BENCHMARKS = [
        "Omni-MATH", "AIME 2024", "AIME 2025", "AIME 2026", "MATH-500",
        "GSM8K test", "GPQA-Diamond", "OlympiadBench", "TheoremQA", "MMLU-Pro",
        "BBH", "LiveBench", "ProcessBench", "GSM-Hard", "GSM-Plus", "GSM-Symbolic",
        "LongBench", "RULER", "FRAMES", "Bamboogle", "FanOutQA",
    ]

    def __init__(self,
                 benchmark_file: Optional[str] = None,
                 ngram: int = 10,
                 overlap_threshold: float = 0.5,
                 check_fields: Optional[List[str]] = None,
                 ):
        self.logger = get_logger()
        self.ngram = int(ngram)
        self.overlap_threshold = float(overlap_threshold)
        self.check_fields = check_fields or ["instruction", "question", "problem"]
        self.benchmark_file = benchmark_file
        self._bench_ngrams = set()
        if benchmark_file and os.path.exists(benchmark_file):
            self._load_benchmark(benchmark_file)

    def _load_benchmark(self, path: str):
        n_q = 0
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            q = o.get("problem") or o.get("question") or o.get("instruction") or ""
            toks = _normalize(q)
            if len(toks) >= self.ngram:
                self._bench_ngrams |= _ngrams(toks, self.ngram)
                n_q += 1
        self.logger.info(f"[DecontaminationFilter] loaded {n_q} benchmark questions "
                         f"-> {len(self._bench_ngrams)} {self.ngram}-grams from {path}")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "评测集去污染:对题面/答案/中间步骤做 n-gram 黑名单扫描,与评测集重叠超阈值即剔除。\n"
                "对齐方案 §3.4/§5.1,要求主训练集命中率 0%。这是放量红线算子。\n"
                "参数:benchmark_file(评测集jsonl)、ngram(默认10)、overlap_threshold(默认0.5)、check_fields。"
            )
        elif lang == "en":
            return (
                "Eval-set decontamination: n-gram blacklist scan over problem/answer/steps; drops "
                "any sample whose overlap with a benchmark question exceeds the threshold. Aligned "
                "with plan §3.4/§5.1 (target 0% contamination). A hard red-line operator for scale."
            )
        return "DecontaminationFilter removes samples overlapping benchmark eval sets."

    def _text_of(self, row) -> str:
        parts = []
        for f in self.check_fields:
            if f in row.index and row.get(f):
                parts.append(str(row.get(f)))
        return " ".join(parts)

    def _is_contaminated(self, row) -> bool:
        if not self._bench_ngrams:
            return False  # 无黑名单则不判(但会在 run 里告警)
        toks = _normalize(self._text_of(row))
        if len(toks) < self.ngram:
            return False
        sample_ngrams = _ngrams(toks, self.ngram)
        if not sample_ngrams:
            return False
        hit = len(sample_ngrams & self._bench_ngrams)
        return (hit / len(sample_ngrams)) >= self.overlap_threshold

    def run(self, storage: DataFlowStorage) -> list:
        dataframe = storage.read("dataframe")
        n_before = len(dataframe)

        if not self._bench_ngrams:
            self.logger.warning(
                "[DecontaminationFilter] 未加载任何评测集黑名单(benchmark_file 为空或不存在)。"
                "本次不剔除任何样本 —— 放量到 20B 前必须提供评测集! "
                f"需覆盖: {', '.join(self.DEFAULT_BENCHMARKS[:6])} ...")
            storage.write(dataframe)
            return []

        mask_clean = ~dataframe.apply(self._is_contaminated, axis=1)
        output = dataframe[mask_clean].reset_index(drop=True)
        n_removed = n_before - len(output)
        rate = (n_removed / n_before * 100) if n_before else 0.0

        self.logger.info(
            f"[DecontaminationFilter] kept {len(output)}/{n_before} rows "
            f"(removed {n_removed} contaminated, hit-rate {rate:.2f}%, "
            f"ngram={self.ngram}, threshold={self.overlap_threshold}).")
        storage.write(output)
        return []
