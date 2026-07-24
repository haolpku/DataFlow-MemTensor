from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

import re
import pandas as pd


def _tokens(text: str):
    return re.sub(r"[^\w\s]", " ", str(text).lower()).split()


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


@OPERATOR_REGISTRY.register()
class CoTQualityFilter(OperatorABC):
    """
    长 CoT 质量过滤器 —— 挡住"看着像 CoT 但没营养"的三类坏样本:

      1. 空壳:<think> 内容过短(低于 min_think_chars),等于没推理;
      2. 复读机:CoT 内 n-gram 自重复率过高(unique/total 低于 min_distinct_ratio);
      3. 仅复述题面:CoT 与题面高度重叠(token Jaccard 高于 max_restate_overlap),
         说明模型只是把题目抄了一遍没真推理。

    挂在 cot_pipeline 的"生成之后、答案校验之前":先扔掉没营养的,再花力气校验答案。
    这是"发现质量问题 → 加一个算子堵住"的典型例子 —— pipeline 可插拔的价值所在。
    """

    def __init__(self,
                 min_think_chars: int = 120,
                 min_distinct_ratio: float = 0.35,
                 max_restate_overlap: float = 0.92,
                 ngram: int = 4,
                 failure_pool_path: str = None,
                 ):
        self.logger = get_logger()
        self.min_think_chars = int(min_think_chars)
        self.min_distinct_ratio = float(min_distinct_ratio)
        self.max_restate_overlap = float(max_restate_overlap)
        self.ngram = int(ngram)
        # 失败池:被剔除样本(带原因)落盘,便于统计通过率(核查文档 §3/§七)。
        # None 则不落盘,保持旧行为。
        self.failure_pool_path = failure_pool_path

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "长 CoT 质量过滤:剔除空壳(think过短)、复读机(n-gram自重复率高)、"
                "仅复述题面(与题面重叠过高)三类坏样本。挂在生成后、答案校验前。\n"
                "参数:min_think_chars(120)、min_distinct_ratio(0.35)、max_restate_overlap(0.92)、ngram(4)。"
            )
        elif lang == "en":
            return (
                "Long-CoT quality filter: drops empty-shell (too-short think), repetitive "
                "(low distinct n-gram ratio), and mere-restatement (high overlap with the problem) "
                "samples. Runs after generation, before answer verification."
            )
        return "CoTQualityFilter drops low-substance chain-of-thought samples."

    @staticmethod
    def _extract_think(cot: str) -> str:
        m = re.search(r"<think>([\s\S]*?)</think>", str(cot))
        return m.group(1).strip() if m else str(cot)

    def _reason(self, row) -> str:
        """返回剔除原因;通过则返回空串。"""
        cot = str(row.get(self.cot_key, ""))
        think = self._extract_think(cot)

        # 1. 空壳
        if len(think) < self.min_think_chars:
            return f"think过短({len(think)}<{self.min_think_chars})"

        toks = _tokens(think)
        # 2. 复读机(n-gram 自重复)
        grams = _ngrams(toks, self.ngram)
        if grams:
            distinct = len(set(grams)) / len(grams)
            if distinct < self.min_distinct_ratio:
                return f"自重复率高(distinct={distinct:.2f}<{self.min_distinct_ratio})"

        # 3. 仅复述题面
        q_toks = set(_tokens(row.get(self.question_key, "")))
        t_toks = set(toks)
        if q_toks and t_toks:
            overlap = len(q_toks & t_toks) / len(q_toks)
            # 题面词几乎全被 think 覆盖、且 think 没带来新词 -> 疑似复述
            new_ratio = len(t_toks - q_toks) / max(len(t_toks), 1)
            if overlap >= self.max_restate_overlap and new_ratio < 0.2:
                return f"疑似复述题面(overlap={overlap:.2f}, new={new_ratio:.2f})"
        return ""

    def run(self,
            storage: DataFlowStorage,
            input_cot_key: str = "generated_cot",
            input_question_key: str = "instruction",
            ) -> list:
        self.cot_key = input_cot_key
        self.question_key = input_question_key

        dataframe = storage.read("dataframe")
        n_before = len(dataframe)
        reasons = dataframe.apply(self._reason, axis=1)
        keep_mask = reasons == ""
        output = dataframe[keep_mask].reset_index(drop=True)

        dropped = reasons[~keep_mask]
        if len(dropped):
            from collections import Counter
            cats = Counter(r.split("(")[0] for r in dropped)
            self.logger.info(f"[CoTQualityFilter] 剔除原因分布: {dict(cats)}")
            # 失败池:落盘被剔除样本 + 原因
            from .failure_pool import dump_rejected
            dump_rejected(dataframe[~keep_mask], self.failure_pool_path,
                          stage="CoTQualityFilter", logger=self.logger,
                          reasons=dropped)
        from .failure_pool import log_pass_rate
        log_pass_rate(self.logger, "CoTQualityFilter", n_before, len(output))
        storage.write(output)
        return [self.cot_key]
