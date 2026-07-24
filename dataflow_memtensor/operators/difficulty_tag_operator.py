from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

import re
import json
import pandas as pd
from collections import Counter
from typing import Optional, List


# 来源标签 -> 难度档的先验映射(核查文档 §一.1:题目难度上限受出题环节限制;
# 放量必须让高难档出现)。对齐 DATA_PLAN §2.1 / SOURCE_CATALOG 的分档。
# 命中即用来源初分(便宜),边界/无标签题再交给 LLM 打分(可选)。
_SOURCE_LEVEL_HINTS = [
    # (子串关键词, 难度档) —— 大小写不敏感,按顺序首个命中生效
    ("gsm8k", "L1"), ("orca", "L1"), ("metamath", "L1"), ("grade", "L1"),
    ("cn_k12", "L2"), ("openmathinstruct", "L2"),
    ("amc", "L3"), ("synthetic_amc", "L3"), ("openr1", "L3"),
    ("olympiad", "L4"), ("aime", "L4"), ("imo", "L4"), ("aops", "L4"),
    ("harp", "L4"), ("big-math", "L4"), ("bigmath", "L4"), ("dapo", "L4"),
    ("deepscaler", "L4"), ("omni-math", "L4"), ("putnam", "L4"),
]

_VALID_LEVELS = ("L1", "L2", "L3", "L4")

# 方案 §2.3 默认难度配比(L1/L2/L3/L4 各档目标占比)。L2/L3 合并档在此按 20/20 拆分。
_DEFAULT_RATIO = {"L1": 0.25, "L2": 0.20, "L3": 0.20, "L4": 0.25}  # 其余 0.10 归 OOD/未定

_DIFFICULTY_SYSTEM_PROMPT = """You are a math-competition difficulty rater. Given a math \
problem, output ONLY a JSON object {"level": "L1|L2|L3|L4", "reason": "<=12 words"}.

Rubric:
- L1: elementary / grade-school word problems and basic arithmetic (GSM8K level).
- L2: high-school algebra/geometry, routine techniques.
- L3: mid competition (AMC / early AIME), multi-step but standard.
- L4: hard olympiad / AIME-final / Putnam level, non-obvious insight required.

Respond with ONLY the JSON object, no markdown fences."""


@OPERATOR_REGISTRY.register()
class DifficultyTagOperator(OperatorABC):
    """
    难度分层器:给每题打 L1-L4 标签,并可选按配比抽样(对齐 DATA_PLAN §2.1)。

    核查文档反复强调:难度分布是提分关键,而 demo "题目难度上限就是出题环节的上限"
    (L4=0)。放量前必须有一个能自动分层、并保证高难档出现的算子。本算子两级策略:

      1) **来源初分(便宜、优先)**:按 ``source`` / ``problem_source`` 标签命中
         ``_SOURCE_LEVEL_HINTS``(olympiad/aime/big-math -> L4,gsm8k/orca -> L1 …)。
         真实题库大多自带来源,这一步零成本覆盖大部分题。
      2) **LLM 打分(可选、兜底)**:传入 ``llm_serving`` 时,对来源初分**没命中**
         (或 ``force_llm=True`` 全部)的题用 LLM 按 rubric 打 L1-L4。

    可选 ``target_ratio`` 时,按配比对各档下采样,输出一个分布贴近目标的子集
    (哪一档不够就全保留并告警——高难档"凑不出量"正是核查文档 §三.2 的核心风险,
    这里显式暴露而非悄悄拿低难档补齐)。
    """

    def __init__(self,
                 llm_serving: Optional[LLMServingABC] = None,
                 source_key: str = "problem_source",
                 fallback_source_key: str = "source",
                 output_key: str = "difficulty",
                 force_llm: bool = False,
                 target_ratio: Optional[dict] = None,
                 sample_seed: int = 20260724,
                 system_prompt: str = _DIFFICULTY_SYSTEM_PROMPT,
                 ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.source_key = source_key
        self.fallback_source_key = fallback_source_key
        self.output_key = output_key
        self.force_llm = force_llm
        self.target_ratio = target_ratio
        self.sample_seed = sample_seed
        self.system_prompt = system_prompt

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "难度分层器:先按来源标签初分 L1-L4(olympiad/aime->L4,gsm8k/orca->L1),"
                "未命中的可选用 LLM 按 rubric 打分;可选按配比(25/20/20/25)下采样。"
                "对齐 DATA_PLAN §2.1,解决核查文档 L4 缺失问题。\n"
                "参数:llm_serving(可选)、source_key、force_llm、target_ratio、output_key。"
            )
        elif lang == "en":
            return (
                "Difficulty tagger: source-label heuristic first (olympiad/aime->L4, gsm8k->L1), "
                "optional LLM rubric scoring for the rest, optional ratio-based down-sampling. "
                "Fixes the missing-L4 problem flagged in the review doc."
            )
        return "DifficultyTagOperator tags problems L1-L4 and optionally samples to a target ratio."

    # ------------------------------------------------------------ 来源初分
    def _source_level(self, row) -> str:
        """按来源标签命中难度档;无命中返回空串。"""
        src = ""
        for k in (self.source_key, self.fallback_source_key):
            if k in row.index and row.get(k):
                src += " " + str(row.get(k))
        src = src.lower()
        if not src.strip():
            return ""
        for kw, level in _SOURCE_LEVEL_HINTS:
            if kw in src:
                return level
        return ""

    # ------------------------------------------------------------ LLM 打分
    @staticmethod
    def _parse_level(resp: str) -> str:
        if not isinstance(resp, str):
            return ""
        s = resp.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            obj = json.loads(s)
            lv = str(obj.get("level", "")).upper().strip()
            if lv in _VALID_LEVELS:
                return lv
        except Exception:
            pass
        # 兜底:正则捞一个 L1-L4
        m = re.search(r"\bL[1-4]\b", s.upper())
        return m.group(0) if m else ""

    def _llm_tag(self, dataframe: pd.DataFrame, need_idx: List[int], question_key: str):
        prompts = [f"Problem:\n{dataframe.iloc[i][question_key]}\n\nRate its difficulty."
                   for i in need_idx]
        if not prompts:
            return {}
        responses = self.llm_serving.generate_from_input(prompts, self.system_prompt)
        return {i: self._parse_level(r) for i, r in zip(need_idx, responses)}

    # ------------------------------------------------------------ 配比抽样
    def _sample_to_ratio(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        import random
        ratio = self.target_ratio or _DEFAULT_RATIO
        rng = random.Random(self.sample_seed)
        by_level = {lv: dataframe.index[dataframe[self.output_key] == lv].tolist()
                    for lv in _VALID_LEVELS}
        counts = {lv: len(idxs) for lv, idxs in by_level.items()}
        self.logger.info(f"[DifficultyTag] 分层前分布: {counts}")

        # 以"最紧的一档"为锚:找能同时满足所有档配比的最大总量 N。
        # N * ratio[lv] <= counts[lv]  =>  N <= counts[lv]/ratio[lv]
        feasible_N = []
        for lv, r in ratio.items():
            if r > 0:
                feasible_N.append(counts.get(lv, 0) / r)
        target_N = int(min(feasible_N)) if feasible_N else 0

        if target_N <= 0:
            # 某目标档为 0 —— 高难档凑不出量的显式暴露(核查文档 §三.2)
            missing = [lv for lv, r in ratio.items() if r > 0 and counts.get(lv, 0) == 0]
            self.logger.warning(
                f"[DifficultyTag] 无法按配比抽样:目标档 {missing} 样本为 0。"
                f"这正是'高难档凑不出量'的风险点——返回全集不做配比,请补高难档题源。")
            return dataframe

        keep_idx = []
        for lv, r in ratio.items():
            want = int(round(target_N * r))
            pool = by_level.get(lv, [])
            if want >= len(pool):
                keep_idx.extend(pool)
                if want > len(pool):
                    self.logger.warning(f"[DifficultyTag] {lv} 需 {want} 只有 {len(pool)},全保留。")
            else:
                keep_idx.extend(rng.sample(pool, want))
        out = dataframe.loc[sorted(keep_idx)].reset_index(drop=True)
        self.logger.info(f"[DifficultyTag] 按配比 {ratio} 抽样 -> {len(out)} 条 "
                         f"(分布 {dict(Counter(out[self.output_key]))})")
        return out

    def run(self,
            storage: DataFlowStorage,
            input_key: str = "instruction",
            ) -> list:
        dataframe = storage.read("dataframe")
        n = len(dataframe)
        if input_key not in dataframe.columns:
            # 找一个可用的题面列兜底
            for cand in ("instruction", "question", "problem"):
                if cand in dataframe.columns:
                    input_key = cand
                    break

        # 1) 来源初分
        levels = [self._source_level(dataframe.iloc[i]) for i in range(n)]
        n_by_source = sum(1 for lv in levels if lv)

        # 2) LLM 兜底(可选)
        if self.force_llm and self.llm_serving is not None:
            need_idx = list(range(n))
        elif self.llm_serving is not None:
            need_idx = [i for i, lv in enumerate(levels) if not lv]
        else:
            need_idx = []
        if need_idx:
            self.logger.info(f"[DifficultyTag] LLM 打分 {len(need_idx)} 题(来源未命中/强制)。")
            llm_levels = self._llm_tag(dataframe, need_idx, input_key)
            for i, lv in llm_levels.items():
                if lv:
                    levels[i] = lv

        # 3) 仍无标签的标 "unknown"
        levels = [lv if lv in _VALID_LEVELS else "unknown" for lv in levels]
        dataframe[self.output_key] = levels
        dist = dict(Counter(levels))
        self.logger.info(f"[DifficultyTag] 打标完成:来源初分 {n_by_source}/{n},"
                         f"最终分布 {dist}。")
        if dist.get("L4", 0) == 0:
            self.logger.warning("[DifficultyTag] L4 高难档为 0 —— 核查文档核心风险点,"
                                "放量前必须接入 olympiad/AIME 级题源。")

        # 4) 可选按配比抽样
        if self.target_ratio is not None:
            dataframe = self._sample_to_ratio(dataframe)

        storage.write(dataframe)
        return [self.output_key]
