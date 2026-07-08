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
                 ):
        self.logger = get_logger()
        self.compare_method = compare_method
        self.require_think_tag = require_think_tag

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

        def _row_ok(row) -> bool:
            if self.require_think_tag and "<think>" not in str(row.get(self.cot_key, "")):
                return False
            return self._answer_ok(row.get(self.answer_key), row.get(self.gt_key))

        keep = dataframe.apply(_row_ok, axis=1)
        output = dataframe[keep].reset_index(drop=True)

        output_file = storage.write(output)
        self.logger.info(
            f"[CoTAnswerFilter] kept {len(output)}/{n_before} rows "
            f"(compare={self.compare_method}). Saved to {output_file}")
        return [self.cot_key, self.answer_key, self.gt_key]
