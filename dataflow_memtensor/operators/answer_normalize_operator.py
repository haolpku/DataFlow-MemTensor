from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

import re
import json
import pandas as pd
from typing import Optional


def _extract_value(text: str) -> str:
    """从一句话/一段文本里抽出最终答案的紧凑值(数值/分数/根式/表达式/boxed)。
    确定性规则,无需 LLM。抽不出则原样返回。"""
    s = str(text).strip()
    if not s:
        return s

    # 1. 已经是纯值(短、无句子标点、无空格分词过多)-> 原样
    if len(s.split()) <= 2 and not s.endswith("."):
        return s.rstrip(".")

    # 2. \boxed{...} 优先
    m = re.search(r"\\boxed\{([^{}]+)\}", s)
    if m:
        return m.group(1).strip()

    # 值的模式:根式(可带系数)优先于裸数字,避免 "24*sqrt(3)" 只抽到 "24"
    val_pat = (
        r"[-+]?\d*\s*\*?\s*sqrt\([^)]*\)"      # 24*sqrt(3) / sqrt(57)
        r"|[-+]?\d+/\d+"                        # 分数 40/3
        r"|[-+]?\d*\.?\d+%"                     # 百分数
        r"|[-+]?\d*\.?\d+"                      # 整数/小数(带负号)
        r"|\b[A-E]\b"                           # 选择题选项
    )
    # 3. 优先 "is/are/equals <VALUE>"(答案主句),取最后一个这种句式
    #    ——避免题面里 "f(x) = 3x^2" 的等号干扰,故 '=' 不参与这一步
    ms = list(re.finditer(r"(?:\bis\b|\bare\b|\bequals?\b)\s*(" + val_pat + r")", s, re.I))
    if ms:
        return re.sub(r"\s+", "", ms[-1].group(1).strip())

    # 3b. 退而用 "= / :" 句式(取最后一个)
    ms_eq = list(re.finditer(r"(?:=|:)\s*(" + val_pat + r")", s))
    if ms_eq:
        return re.sub(r"\s+", "", ms_eq[-1].group(1).strip())

    # 4. 兜底:全句最后一个"像答案的值"
    m3 = list(re.finditer(val_pat, s))
    if m3:
        return re.sub(r"\s+", "", m3[-1].group(0).strip())

    return s.rstrip(".")


@OPERATOR_REGISTRY.register()
class AnswerNormalizeOperator(OperatorABC):
    """
    把交错轨迹的 final_answer 从"整句"规范成"纯值",原句保留到 final_answer_raw。

    背景:agent 有时把答案写成整句(如 "The minimum value is -12, attained at x=2."),
    使 final_answer 字段格式不统一,下游程序化答案校验会解析失败。本算子:
      - 从 trajectory.final_answer 抽出紧凑值(数值/分数/根式/boxed/选项);
      - 回填 trajectory.final_answer(纯值),原句存 trajectory.final_answer_raw;
      - 顶层也补一个 final_answer 列,方便下游直接用。

    确定性规则(正则),不依赖 LLM,可复现。这是"发现字段不规范 → 加算子统一"的例子。
    """

    def __init__(self, traj_key: str = "trajectory"):
        self.logger = get_logger()
        self.traj_key = traj_key

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "把交错轨迹的 final_answer 从整句规范成纯值(数值/分数/根式/boxed/选项),"
                "原句存 final_answer_raw,顶层补 final_answer 列。确定性正则,便于下游答案校验。"
            )
        elif lang == "en":
            return (
                "Normalizes a trajectory's final_answer from a full sentence to a compact value "
                "(number/fraction/radical/boxed/choice); keeps the sentence in final_answer_raw. "
                "Deterministic regex, no LLM. Makes downstream answer verification robust."
            )
        return "AnswerNormalizeOperator compacts trajectory final_answer to a plain value."

    def run(self, storage: DataFlowStorage, traj_key: str = "trajectory") -> list:
        self.traj_key = traj_key
        dataframe = storage.read("dataframe")
        n_norm = 0
        top_answers = []

        for idx in range(len(dataframe)):
            t = dataframe.iloc[idx][self.traj_key]
            parsed = json.loads(t) if isinstance(t, str) else t
            if not isinstance(parsed, dict):
                top_answers.append(None)
                continue
            raw = parsed.get("final_answer", "")
            val = _extract_value(raw)
            if str(val) != str(raw):
                parsed["final_answer_raw"] = raw
                parsed["final_answer"] = val
                n_norm += 1
            top_answers.append(val)
            # 写回(保持与读入同类型:dict 存 dict,原为 str 则存 str)
            dataframe.iat[idx, dataframe.columns.get_loc(self.traj_key)] = (
                json.dumps(parsed, ensure_ascii=False) if isinstance(t, str) else parsed
            )

        dataframe["final_answer"] = top_answers
        storage.write(dataframe)
        self.logger.info(f"[AnswerNormalizeOperator] normalized {n_norm}/{len(dataframe)} "
                         f"sentence answers -> plain value (raw kept in final_answer_raw).")
        return ["final_answer"]
