from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

import pandas as pd
from typing import Optional


@OPERATOR_REGISTRY.register()
class ReasoningEvidenceGroundingFilter(OperatorABC):
    """
    Filters multi-step evidence reasoning records by grounding quality.

    Keeps only rows that satisfy (aligned with the MidTrain plan §5.3 / §6):
      * claim-evidence binding rate >= ``min_binding_rate`` (default 0.95);
      * at least ``min_hops`` reasoning steps (default 3);
      * every step's evidence_ids reference an evidence_id that actually exists
        in the row's evidence cluster (no dangling / hallucinated citations);
      * (optional) the stated golden_answer matches a reference answer via
        math_verify / exact comparison when a ground-truth column is present.

    Rows failing any active check are dropped.
    """

    def __init__(self,
                 min_binding_rate: float = 0.95,
                 min_hops: int = 3,
                 check_answer_against: Optional[str] = "golden_answer",
                 compare_method: str = "math_verify",
                 require_distractors: bool = False,
                 forbid_citing_distractors: bool = True,
                 failure_pool_path: str = None,
                 ):
        self.logger = get_logger()
        self.min_binding_rate = min_binding_rate
        self.min_hops = min_hops
        self.check_answer_against = check_answer_against
        self.compare_method = compare_method
        self.require_distractors = require_distractors
        self.forbid_citing_distractors = forbid_citing_distractors
        # 失败池:grounding 不达标(绑定率低/悬空引用/引用干扰项/答案错)的样本落盘,
        # 便于统计通过率(核查文档 §3/§七)。None 则不落盘,保持旧行为。
        self.failure_pool_path = failure_pool_path

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子按证据接地质量过滤多步证据推理样本:\n"
                "- claim-证据绑定率 >= 阈值(默认0.95)\n"
                "- 推理步数 >= min_hops(默认3)\n"
                "- 每步 evidence_ids 必须指向真实存在的证据(无悬空引用)\n"
                "- 可选:golden_answer 与参考答案做 math_verify/exact 校验\n"
                "不满足任一激活项的样本被丢弃。"
            )
        elif lang == "en":
            return (
                "Filters multi-step evidence reasoning by grounding quality: binding rate "
                ">= threshold (default 0.95), >= min_hops steps, every step's evidence_ids "
                "reference a real evidence entry (no dangling citations), and optionally the "
                "golden_answer matches a reference answer via math_verify/exact."
            )
        else:
            return "ReasoningEvidenceGroundingFilter drops poorly-grounded evidence reasoning rows."

    def _answer_ok(self, answer, reference) -> bool:
        if self.compare_method == "exact":
            return str(answer).strip() == str(reference).strip()
        # math_verify
        try:
            from math_verify import parse, verify
            return bool(verify(parse(str(reference)), parse(str(answer))))
        except Exception:
            # fall back to exact if math_verify unavailable / errors
            return str(answer).strip() == str(reference).strip()

    def _reason(self, row) -> str:
        """返回淘汰原因;通过所有检查则返回空串。"""
        steps = row.get(self.steps_key)
        evidences = row.get(self.evidences_key)
        if not isinstance(steps, list) or len(steps) < self.min_hops:
            return f"跳数不足(<{self.min_hops})"
        if not isinstance(evidences, list) or not evidences:
            return "无证据簇"

        valid_ids = {e.get("evidence_id") for e in evidences if isinstance(e, dict)}
        bound = 0
        for s in steps:
            if not isinstance(s, dict):
                return "step结构非法"
            eids = s.get("evidence_ids") or []
            if not eids:
                continue
            # every cited id must exist
            if any(eid not in valid_ids for eid in eids):
                return "悬空引用(evidence_id不存在)"
            bound += 1
        binding_rate = bound / len(steps) if steps else 0.0
        if binding_rate < self.min_binding_rate:
            return f"绑定率低({binding_rate:.2f}<{self.min_binding_rate})"

        # 干扰项检查:证明模型确实要"筛选"证据,而非全用
        distractor_ids = row.get("distractor_ids") if "distractor_ids" in row.index else None
        distractor_ids = distractor_ids if isinstance(distractor_ids, list) else []
        if self.require_distractors and not distractor_ids:
            return "无干扰项"
        if self.forbid_citing_distractors and distractor_ids:
            cited = {e for s in steps if isinstance(s, dict) for e in (s.get("evidence_ids") or [])}
            # 被引用的证据里不能包含任何干扰项
            if cited & set(distractor_ids):
                return "引用了干扰项"

        if self.check_answer_against and self.check_answer_against in row.index:
            ref = row.get(self.check_answer_against)
            ans = row.get(self.answer_key)
            if ref is not None and str(ref) != "" and not self._answer_ok(ans, ref):
                return "答案校验未通过"
        return ""

    def run(self,
            storage: DataFlowStorage,
            input_evidences_key: str = "evidences",
            input_steps_key: str = "steps",
            input_answer_key: str = "generated_golden_answer",
            ) -> list:
        self.evidences_key = input_evidences_key
        self.steps_key = input_steps_key
        self.answer_key = input_answer_key

        dataframe = storage.read("dataframe")
        n_before = len(dataframe)
        reasons = dataframe.apply(self._reason, axis=1)
        keep_mask = reasons == ""
        output = dataframe[keep_mask].reset_index(drop=True)

        # 失败池:grounding 不达标样本落盘 + 原因分布(核查文档 §3/§七)
        from .failure_pool import dump_rejected, log_pass_rate
        dropped = reasons[~keep_mask]
        if len(dropped):
            from collections import Counter
            cats = Counter(r.split("(")[0] for r in dropped)
            self.logger.info(f"[EvidenceGroundingFilter] 剔除原因分布: {dict(cats)}")
            dump_rejected(dataframe[~keep_mask], self.failure_pool_path,
                          stage="ReasoningEvidenceGroundingFilter", logger=self.logger,
                          reasons=dropped)

        output_file = storage.write(output)
        log_pass_rate(self.logger, "ReasoningEvidenceGroundingFilter", n_before, len(output))
        self.logger.info(
            f"[EvidenceGroundingFilter] kept {len(output)}/{n_before} rows "
            f"(min_binding_rate={self.min_binding_rate}, min_hops={self.min_hops}). "
            f"Saved to {output_file}")
        return [self.evidences_key, self.steps_key, self.answer_key]
