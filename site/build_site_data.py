#!/usr/bin/env python3
"""
build_site_data.py — 把三类真实 demo 数据打包成静态站用的 data.js.

读取 cache_cot / cache_evidence / cache_interleaved 的最终产物,
生成 site/data.js: window.DB = {catalog:[...], samples:{key:[rows]}}.
静态站(GitHub Pages)直接加载,无需服务器。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root


def load(path):
    rows = []
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return rows
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def load_last(prefix):
    """读某类 pipeline 的最后一步产物(step 编号最大的)。"""
    import glob
    files = sorted(glob.glob(os.path.join(ROOT, prefix + "*.jsonl")))
    return load(os.path.relpath(files[-1], ROOT)) if files else []


cot = load_last("cache_cot/cot_step_step")
evidence = load_last("cache_evidence/evidence_step_step")
inter = load_last("cache_interleaved/interleaved_step_step")

catalog = [
    {"key": "long_cot", "title": "长思维链 Long-CoT", "cat": "可验证数学题",
     "color": "#c4b5fd", "kind": "cot",
     "desc": "题目 + <think> 长推理 + \\boxed{} 可验证答案。每条经 math_verify 校验答案正确。对齐 MidTrain 方案 §4.2。",
     "count": len(cot)},
    {"key": "evidence", "title": "多步证据推理 Evidence", "cat": "证据接地",
     "color": "#86efac", "kind": "evidence",
     "desc": "证据簇(含干扰项) + 每步 claim 绑定 evidence_id 的推理链。绑定率≥95%,干扰项不被引用——需筛选证据而非照抄。对齐 §5.3。",
     "count": len(evidence)},
    {"key": "interleaved", "title": "长程交错思维 Interleaved", "cat": "工具轨迹",
     "color": "#93c5fd", "kind": "interleaved",
     "desc": "(思考→工具→观察) 交错轨迹。run_python/sympy_check 真实执行,search 从真实语料检索,含 LLM 质量四轴打分。对齐 §4.3/§5.4。",
     "count": len(inter)},
]

db = {
    "catalog": catalog,
    "samples": {
        "long_cot": cot,
        "evidence": evidence,
        "interleaved": inter,
    },
    "meta": {
        "repo": "haolpku/DataFlow-MemTensor",
        "model": "gpt-4.1-mini",
        "note": "真实 API 合成 + 校验的 demo 数据,各 24 条。",
    },
}

out = os.path.join(ROOT, "site", "data.js")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    f.write("window.DB=" + json.dumps(db, ensure_ascii=False) + ";\n")

print(f"wrote {out}")
print(f"  long_cot={len(cot)}  evidence={len(evidence)}  interleaved={len(inter)}")
