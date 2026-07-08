from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

import json
import pandas as pd
from typing import Optional


_EVIDENCE_SYSTEM_PROMPT = """You turn a math problem into a MULTI-STEP, EVIDENCE-GROUNDED reasoning record.

Produce a JSON object with exactly these fields:
- "evidences": a list of 3-6 objects {"evidence_id": "ev1", "text": "..."} — each a
  single definition, theorem, or given condition needed to solve the problem. Number
  them ev1, ev2, ....
- "steps": a list of >=3 objects {"step": <int>, "claim": "...", "evidence_ids": ["ev1", ...],
  "derivation": "..."}. EVERY step's claim must cite the evidence_id(s) it depends on.
  The final step must state the answer.
- "golden_answer": the final answer as a compact value (number, boxed expression, or
  yes/no). It must be verifiable.

Rules:
- Each claim must be grounded: evidence_ids must be non-empty and reference real
  evidence entries you listed.
- Prefer >=3 hops. Difficulty comes from the reasoning chain, not from prose.
- Respond with ONLY the JSON object, no markdown fences."""


@OPERATOR_REGISTRY.register()
class ReasoningEvidenceChainGenerator(OperatorABC):
    """
    Generates multi-step, evidence-grounded reasoning records from math questions.

    For each question the LLM emits an evidence cluster + a reasoning chain whose every
    step cites the evidence it relies on (aligned with the MidTrain plan §3.3 / §5.3:
    >=3 hops, each claim bound to an evidence_id). The parsed record is written to
    output columns for a downstream grounding/answer filter to verify.
    """

    def __init__(self,
                 llm_serving: LLMServingABC,
                 system_prompt: str = _EVIDENCE_SYSTEM_PROMPT,
                 ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.system_prompt = system_prompt

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子把数学题转成多步证据推理记录:LLM 生成证据簇(evidences)与推理链(steps),\n"
                "每一步 claim 绑定其依赖的 evidence_id(对齐 §5.3:≥3 跳、claim-证据绑定)。\n"
                "输入参数:\n"
                "- llm_serving:LLM 服务实例\n"
                "- system_prompt:约束输出 JSON 结构的系统提示\n"
                "输出参数:\n"
                "- evidences / steps / golden_answer / num_hops / claim_binding_rate 等字段"
            )
        elif lang == "en":
            return (
                "Turns a math question into a multi-step, evidence-grounded reasoning record: "
                "the LLM emits an evidence cluster and a reasoning chain where each step's claim "
                "cites the evidence_id(s) it depends on (aligned with plan §5.3: >=3 hops, "
                "claim-evidence binding).\n"
                "Output: evidences / steps / golden_answer / num_hops / claim_binding_rate."
            )
        else:
            return "ReasoningEvidenceChainGenerator produces evidence-grounded multi-step reasoning."

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_key}")
        for k in (self.output_evidences_key, self.output_steps_key):
            if k in dataframe.columns:
                raise ValueError(f"Column already exists and would be overwritten: {k}")

    @staticmethod
    def _clean_json_block(item) -> str:
        if not isinstance(item, str):
            return "{}"
        return item.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    def _reformat_prompt(self, dataframe):
        questions = dataframe[self.input_key].tolist()
        return [f"Problem:\n{q}\n\nProduce the evidence-grounded reasoning JSON." for q in questions]

    @staticmethod
    def _binding_rate(steps) -> float:
        if not steps:
            return 0.0
        bound = sum(1 for s in steps if isinstance(s, dict) and s.get("evidence_ids"))
        return round(bound / len(steps), 3)

    def run(self,
            storage: DataFlowStorage,
            input_key: str = "instruction",
            output_evidences_key: str = "evidences",
            output_steps_key: str = "steps",
            output_answer_key: str = "generated_golden_answer",
            ):
        self.input_key = input_key
        self.output_evidences_key = output_evidences_key
        self.output_steps_key = output_steps_key
        self.output_answer_key = output_answer_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        prompts = self._reformat_prompt(dataframe)
        responses = self.llm_serving.generate_from_input(prompts, self.system_prompt)

        evidences_col, steps_col, answer_col = [], [], []
        num_hops_col, binding_col = [], []
        for resp in responses:
            try:
                obj = json.loads(self._clean_json_block(resp))
                evidences = obj.get("evidences", [])
                steps = obj.get("steps", [])
                answer = obj.get("golden_answer", "")
            except Exception as e:
                self.logger.warning(f"[EvidenceChainGenerator] parse failed: {e}")
                evidences, steps, answer = [], [], ""
            evidences_col.append(evidences)
            steps_col.append(steps)
            answer_col.append(answer)
            num_hops_col.append(len(steps))
            binding_col.append(self._binding_rate(steps))

        dataframe[self.output_evidences_key] = evidences_col
        dataframe[self.output_steps_key] = steps_col
        dataframe[self.output_answer_key] = answer_col
        dataframe["num_hops"] = num_hops_col
        dataframe["claim_binding_rate"] = binding_col

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return [self.output_evidences_key, self.output_steps_key, self.output_answer_key,
                "num_hops", "claim_binding_rate"]
