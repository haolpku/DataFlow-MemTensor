from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

import pandas as pd


_COT_SYSTEM_PROMPT = """You are an expert mathematician. Solve the problem with a rigorous, \
step-by-step chain of thought.

Format your response EXACTLY as:
<think>
... your full step-by-step reasoning: identify the relevant facts, derive each step, \
carry out the computation, and check the result ...
</think>
Then, on a new line, give the final answer as \\boxed{...} with a single compact value \
(number, exact expression, or yes/no).

Rules:
- The <think> block must contain the actual reasoning, not a restatement of the problem.
- End with exactly one \\boxed{...}. Do not add text after it."""


@OPERATOR_REGISTRY.register()
class ReasoningLongCoTGenerator(OperatorABC):
    """
    Generates long chain-of-thought (CoT) solutions for math questions.

    A simplified take on DataFlow's reasoning_math_pipeline: instead of the full
    11-step question-synthesis + difficulty/category + multi-filter chain, this
    operator does the one thing the CoT data needs — produce a ``<think>...</think>``
    long-reasoning block followed by a ``\\boxed{}`` verifiable answer. A downstream
    answer filter checks the boxed value against the ground truth.

    Output columns: ``generated_cot`` (full <think>+boxed text) and
    ``extracted_answer`` (the boxed value, for the filter).
    """

    def __init__(self,
                 llm_serving: LLMServingABC,
                 system_prompt: str = _COT_SYSTEM_PROMPT,
                 ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.system_prompt = system_prompt

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子为数学题生成长思维链(CoT)解答:<think>长推理</think> + \\boxed{答案}。\n"
                "是 reasoning_math_pipeline 的精简版(只保留生成这一核心步)。\n"
                "输出:generated_cot(完整文本)、extracted_answer(boxed 值,供过滤器校验)。"
            )
        elif lang == "en":
            return (
                "Generates long chain-of-thought solutions (<think>...</think> + \\boxed{answer}) "
                "for math questions — a simplified reasoning_math_pipeline keeping only the "
                "generation step. Output: generated_cot, extracted_answer."
            )
        return "ReasoningLongCoTGenerator produces long-CoT math solutions."

    @staticmethod
    def extract_boxed(text) -> str:
        """Extract the last \\boxed{...} with brace matching."""
        if not isinstance(text, str):
            return ""
        idx = text.rfind(r"\boxed")
        if idx < 0:
            return ""
        i = text.find("{", idx)
        if i < 0:
            return ""
        depth = 0
        out = []
        for c in text[i:]:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            out.append(c)
            if depth == 0 and c == "}":
                break
        return "".join(out)[1:-1].strip()

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_key}")

    def run(self,
            storage: DataFlowStorage,
            input_key: str = "instruction",
            output_key: str = "generated_cot",
            output_answer_key: str = "extracted_answer",
            ):
        self.input_key = input_key
        self.output_key = output_key
        self.output_answer_key = output_answer_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        questions = dataframe[self.input_key].tolist()
        prompts = [f"Problem:\n{q}\n\nSolve it with full chain-of-thought." for q in questions]
        responses = self.llm_serving.generate_from_input(prompts, self.system_prompt)

        dataframe[self.output_key] = responses
        dataframe[self.output_answer_key] = [self.extract_boxed(r) for r in responses]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return [self.output_key, self.output_answer_key]
