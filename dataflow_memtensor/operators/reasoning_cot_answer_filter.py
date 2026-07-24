from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

import pandas as pd
from typing import Literal


@OPERATOR_REGISTRY.register()
class ReasoningCoTAnswerFilter(OperatorABC):
    """
    Filters long-CoT samples by answer correctness.

    Keeps only rows whose extracted boxed answer matches the ground truth
    (math_verify or exact). A simplified stand-in for the answer-verification
    tail of reasoning_math_pipeline (format + token-length + groundtruth +
    ngram filters collapsed to the single check that matters for correctness).
    """

    def __init__(self,
                 compare_method: Literal["math_verify", "exact"] = "math_verify",
                 require_think_tag: bool = True,
                 failure_pool_path: str = None,
                 ):
        self.logger = get_logger()
        self.compare_method = compare_method
        self.require_think_tag = require_think_tag
        # 失败池:答案错/无 think 标签的样本落盘,便于统计通过率并作负样本资产
        # (核查文档 §3/§七;PIPELINE_DESIGN_20B 的 Failure Pool)。None 则不落盘。
        self.failure_pool_path = failure_pool_path

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子按答案正确性过滤长 CoT 样本:抽取的 boxed 答案与标准答案做 "
                "math_verify/exact 对比,可选要求含 <think> 标签。不符则丢弃。"
            )
        elif lang == "en":
            return (
                "Filters long-CoT samples by answer correctness: the extracted boxed answer "
                "is compared to the ground truth via math_verify/exact; optionally requires a "
                "<think> tag. Non-matching rows are dropped."
            )
        return "ReasoningCoTAnswerFilter keeps only correctly-answered long-CoT rows."

    def _answer_ok(self, answer, reference) -> bool:
        if not str(answer).strip():
            return False
        if self.compare_method == "exact":
            return str(answer).strip() == str(reference).strip()
        try:
            from math_verify import parse, verify
            return bool(verify(parse(str(reference)), parse(str(answer))))
        except Exception:
            return str(answer).strip() == str(reference).strip()

    def run(self,
            storage: DataFlowStorage,
            input_cot_key: str = "generated_cot",
            input_answer_key: str = "extracted_answer",
            input_gt_key: str = "golden_answer",
            ) -> list:
        self.cot_key = input_cot_key
        self.answer_key = input_answer_key
        self.gt_key = input_gt_key

        dataframe = storage.read("dataframe")
        n_before = len(dataframe)

        def _reason(row) -> str:
            """返回淘汰原因;通过则空串。"""
            if self.require_think_tag and "<think>" not in str(row.get(self.cot_key, "")):
                return "缺少<think>标签"
            if not self._answer_ok(row.get(self.answer_key), row.get(self.gt_key)):
                return "答案校验未通过"
            return ""

        reasons = dataframe.apply(_reason, axis=1)
        keep = reasons == ""
        output = dataframe[keep].reset_index(drop=True)

        # 失败池:错答案/坏格式落盘,是 PRM / 自检训练的负样本资产
        from .failure_pool import dump_rejected, log_pass_rate
        dump_rejected(dataframe[~keep], self.failure_pool_path,
                      stage="ReasoningCoTAnswerFilter", logger=self.logger,
                      reasons=reasons[~keep])

        output_file = storage.write(output)
        log_pass_rate(self.logger, "ReasoningCoTAnswerFilter", n_before, len(output))
        self.logger.info(
            f"[CoTAnswerFilter] kept {len(output)}/{n_before} rows "
            f"(compare={self.compare_method}). Saved to {output_file}")
        return [self.cot_key, self.answer_key, self.gt_key]
