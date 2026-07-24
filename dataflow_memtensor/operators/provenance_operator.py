from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

import os
import datetime
import pandas as pd
from typing import Optional


@OPERATOR_REGISTRY.register()
class ProvenanceOperator(OperatorABC):
    """
    数据血缘/来源标记算子(对齐核查文档 §5"数据清单最小字段")。

    核查文档指出:本批数据"没有任何来源字段",导致指标可不可信无从追溯——
    "指标可不可信,取决于数据是不是真的,而当前恰恰没有检查这一点"。解决的第一步
    是给每条样本盖上最小血缘字段,让"这条题从哪来、是不是合成的、谁生成的"可追溯。

    本算子在 pipeline **最前端**运行,给每行补齐以下字段(已存在则不覆盖,尊重
    真实题库自带的来源标签):

      - ``problem_source``  : 题目来源(数据集名/HF 仓库/"handwritten"/"demo")
      - ``synthetic_flag``  : 是否为合成数据(题面或解由 LLM 造)。True/False
      - ``gen_model``       : 生成所用模型(如 gpt-4.1-mini),便于区分生成与判分
      - ``pipeline``        : 产出该样本的 pipeline 名(cot/evidence/interleaved)
      - ``created_at``      : UTC 时间戳(ISO8601),标记产出时间
      - ``schema_version``  : 血缘 schema 版本,便于日后演进

    设计要点:
    - **不覆盖已有值**:真实题库自带 ``source`` / ``problem_source`` 时保留原值,
      只填空缺——这样 demo 自造题标 "demo",接入 NuminaMath 后标 "AI-MO/NuminaMath-CoT"。
    - 纯确定性,不调 LLM;放在最前,后续每一步(含失败池)都带着血缘。
    """

    def __init__(self,
                 problem_source: str = None,
                 synthetic_flag: Optional[bool] = None,
                 gen_model: str = None,
                 pipeline: str = None,
                 source_key: str = "problem_source",
                 overwrite: bool = False,
                 schema_version: str = "v1",
                 ):
        self.logger = get_logger()
        # 缺省从环境变量取,便于 CLI / CI 注入而不改代码
        self.problem_source = problem_source or os.environ.get("MEMTENSOR_SOURCE", "demo")
        self.synthetic_flag = synthetic_flag
        self.gen_model = gen_model or os.environ.get("DF_MODEL", "")
        self.pipeline = pipeline or ""
        self.source_key = source_key
        self.overwrite = overwrite
        self.schema_version = schema_version

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "数据血缘标记:给每条样本补 problem_source / synthetic_flag / gen_model / "
                "pipeline / created_at / schema_version 字段(已有则不覆盖)。对齐核查文档 §5 "
                "最小字段要求,让来源真实性、是否合成、生成模型可追溯。放在 pipeline 最前。\n"
                "参数:problem_source、synthetic_flag、gen_model、pipeline、overwrite(默认False)。"
            )
        elif lang == "en":
            return (
                "Stamps each sample with provenance fields (problem_source / synthetic_flag / "
                "gen_model / pipeline / created_at / schema_version) unless already present. "
                "Makes source authenticity, synthetic-vs-real, and the generating model traceable. "
                "Runs at the head of the pipeline."
            )
        return "ProvenanceOperator stamps minimal provenance/lineage fields on each sample."

    def _fill_column(self, dataframe: pd.DataFrame, col: str, value):
        """填列:overwrite=True 全填;否则只填缺失(NaN/空串/列不存在)。"""
        if col not in dataframe.columns:
            dataframe[col] = value
            return
        if self.overwrite:
            dataframe[col] = value
            return
        # 只填空:NaN 或空串
        mask_empty = dataframe[col].isna() | (dataframe[col].astype(str).str.strip() == "")
        dataframe.loc[mask_empty, col] = value

    def run(self, storage: DataFlowStorage) -> list:
        dataframe = storage.read("dataframe")
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        self._fill_column(dataframe, self.source_key, self.problem_source)
        if self.synthetic_flag is not None:
            self._fill_column(dataframe, "synthetic_flag", bool(self.synthetic_flag))
        elif "synthetic_flag" not in dataframe.columns:
            # 未显式指定时:demo/合成场景默认 True(题或解由 LLM 造)
            dataframe["synthetic_flag"] = True
        if self.gen_model:
            self._fill_column(dataframe, "gen_model", self.gen_model)
        if self.pipeline:
            self._fill_column(dataframe, "pipeline", self.pipeline)
        self._fill_column(dataframe, "created_at", now)
        self._fill_column(dataframe, "schema_version", self.schema_version)

        storage.write(dataframe)
        self.logger.info(
            f"[ProvenanceOperator] stamped {len(dataframe)} rows "
            f"(source={self.problem_source!r}, gen_model={self.gen_model!r}, "
            f"pipeline={self.pipeline!r}, synthetic_flag default).")
        return [self.source_key, "synthetic_flag", "gen_model", "pipeline",
                "created_at", "schema_version"]
