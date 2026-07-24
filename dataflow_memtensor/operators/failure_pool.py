"""
failure_pool.py — 被过滤器淘汰样本的落盘与通过率统计。

核查文档 §3 / §七指出:demo 只交付"通过筛选"的样本,却不给
①生成总量与淘汰量 ②被淘汰样本本体。留下 24 条,对应生成 25 条还是 400 条,
是完全不同的质量结论——所以"通过率"与"失败池"必须一并交付。

本模块提供一个跨过滤器复用的 helper:每个 filter 在丢弃样本前,调用
``dump_rejected`` 把被淘汰行(附上 ``_drop_stage`` / ``_drop_reason``)
**追加**到一个 jsonl 失败池文件,并把通过率写进日志。

设计要点:
- 失败池是**旁路产物**,不进 FileStorage 的线性 step 链(FileStorage.write
  会把数据缓冲到 operator_step+1,失败池若走它会污染主数据流)。因此这里直接
  用 pandas 追加写盘,与主 pipeline 解耦。
- ``failure_pool_path`` 为 None 时**完全不落盘**(保持旧行为,不破坏现有调用)。
- 每行都带 ``_drop_stage``(哪个算子丢的)和 ``_drop_reason``(为什么),
  失败池本身就是可分析的资产(PRM / 负样本 / 自检训练)。
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def dump_rejected(
    rejected: pd.DataFrame,
    failure_pool_path: Optional[str],
    stage: str,
    logger=None,
    reasons: Optional[pd.Series] = None,
) -> None:
    """把被淘汰的行追加到失败池 jsonl。

    Args:
        rejected: 被本算子淘汰的行(过滤前的原始列)。
        failure_pool_path: 失败池文件路径;None 则不落盘(旧行为)。
        stage: 淘汰阶段标识(算子名),写入 ``_drop_stage`` 列。
        logger: 可选 logger,用于告警写失败。
        reasons: 可选,与 rejected 行对齐的淘汰原因 Series,写入 ``_drop_reason``。
    """
    if failure_pool_path is None or rejected is None or len(rejected) == 0:
        return

    out = rejected.copy()
    out["_drop_stage"] = stage
    if reasons is not None:
        # reasons 的 index 可能与 rejected 不一致,按位置对齐
        out["_drop_reason"] = list(reasons.values) if hasattr(reasons, "values") else list(reasons)
    elif "_drop_reason" not in out.columns:
        out["_drop_reason"] = ""

    try:
        os.makedirs(os.path.dirname(failure_pool_path) or ".", exist_ok=True)
        # 追加写:失败池跨多个算子累积,每个算子只 append 自己丢的那批。
        # 先序列化成 jsonl 文本(每行一条,末尾带换行),再 append 到文件。
        payload = out.to_json(orient="records", lines=True, force_ascii=False)
        if not payload.endswith("\n"):
            payload += "\n"
        with open(failure_pool_path, "a", encoding="utf-8") as f:
            f.write(payload)
        if logger is not None:
            logger.info(
                f"[failure_pool] {stage}: appended {len(out)} rejected rows -> {failure_pool_path}"
            )
    except Exception as e:  # 失败池写盘失败绝不能拖垮主流程
        if logger is not None:
            logger.warning(f"[failure_pool] 写入失败({stage}): {type(e).__name__}: {e}")


def log_pass_rate(logger, stage: str, n_before: int, n_after: int) -> float:
    """统一的通过率日志。返回通过率(0-1)。"""
    rate = (n_after / n_before) if n_before else 0.0
    dropped = n_before - n_after
    if logger is not None:
        logger.info(
            f"[pass_rate] {stage}: kept {n_after}/{n_before} "
            f"(dropped {dropped}, pass_rate {rate * 100:.1f}%)"
        )
    return rate
