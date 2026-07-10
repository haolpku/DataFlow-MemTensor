# 为什么用 DataFlow 做数据生产

> 用本项目**真实发生的"发现问题 → 加算子 → 质量提升"闭环**,说明 DataFlow 的工程优势。
> 这些不是宣传话术,是本仓库开发过程中实际走过的路。

---

## 核心优势:数据质量问题 = 加一个算子就能堵

传统"一次性脚本"造数据,发现坏数据只能手工挑或重写脚本。DataFlow 把生产拆成**可插拔算子的串联**,每种坏数据模式对应一个可复用算子。本项目实际发生的三个闭环:

### 闭环 ①:发现"证据能照抄" → 加干扰项 + grounding 算子
- **问题**:早期 evidence 数据,证据簇"不多不少正好够用",模型线性重排就能答对 —— grounding 能力根本没训到。
- **优化**:改 `ReasoningEvidenceChainGenerator` 生成硬负例干扰项(混合打乱),加 `ReasoningEvidenceGroundingFilter` 的 `forbid_citing_distractors` 规则。
- **结果**:24 条平均 6.3 证据/3 干扰,**0 引用泄漏**;数据从"照抄"变成"必须筛选证据"。

### 闭环 ②:发现"sandbox 缺函数造成假失败" → 补工具后端
- **问题**:质检发现 8/24 条 interleaved 的 `run_python` 失败,根因是 `MathSandboxClient` 的 builtins 白名单缺 `all`/`any`。
- **优化**:补全 builtins。因为 sandbox 是可插拔后端(`SandboxClientABC` 子类),改一处即可。
- **结果**:工具失败率 33% → 22%,剩下的是模型**真实逻辑错误**(且都自我纠错成功,是高价值数据)。

### 闭环 ③:发现"缺去污染红线" → 加 DecontaminationFilter
- **问题**:pipeline 完全没有评测集去污染 —— 这会让"涨分"变成数据泄漏的假象。
- **优化**:新增 `DecontaminationFilter`(评测集 n-gram 黑名单扫描),挂到链尾。
- **结果**:真实数据验证有效 —— 混入的 2 道评测题(gcd、7^100)被准确揪出。

> **关键**:三次优化都**没有手工改一条数据**,全是"改/加算子"。这就是 pipeline 相比脚本的本质优势。

---

## 六个具体优势

| 优势 | 说明 | 本项目体现 |
|------|------|-----------|
| **算子化** | 每个处理步骤是独立算子,职责单一 | 6 个算子:2 生成 + 4 过滤,各管一件事 |
| **可插拔** | 加/换算子不影响其他步骤 | 补 sandbox、加去污染,都是局部改动 |
| **可组合** | 算子按需串成不同 pipeline | 同一批算子组出 CoT / evidence / interleaved 三条线 |
| **可复用上游** | 直接用 DataFlow 已有算子,不重复造轮子 | 难度分层、ngram 去重等直接复用上游 reasoning 算子 |
| **每步落盘** | `FileStorage` 每步产物独立存,可断点续跑、可审计 | `cache_*/…_step{N}.jsonl`,能逐步看哪步淘汰了什么 |
| **可验证/可复现** | 规则确定、可讲清、可对照 | 答案 math_verify、绑定率、去污染命中率,全部量化可查 |

---

## 三条 pipeline 的调参也是"拧质量"的旋钮

不用改代码,调算子参数即可拧紧/放松质量门槛:

```python
ReasoningEvidenceGroundingFilter(min_binding_rate=0.95, min_hops=3)  # 提到 0.98 / 4 更严
CoTQualityFilter(min_think_chars=120, min_distinct_ratio=0.35)       # 提高门槛挡更多空壳
DecontaminationFilter(ngram=10, overlap_threshold=0.5)              # 调 n-gram 粒度/阈值
```

---

## 放量到 20B 时的可扩展性

同一套算子架构,放量只是"加算子 + 换数据源 + 上并发",不推翻重来。待建算子(见 `PIPELINE_DESIGN_20B.md`)全是"为堵某类质量问题而加":

- `PassAtKFilter` —— 剔除过易/不可解题(多次采样通过率)
- `DedupOperator` —— 题面近似去重(MinHash LSH)
- `DistractorHardnessFilter` —— 判干扰项是否够像(evidence 质量再拧一档)
- 去污染已就绪,放量时喂真实评测集黑名单即可

---

## 一句话总结

**DataFlow 的价值:数据质量不是靠"造完祈祷",而是靠"每发现一类问题,就加一个算子堵住"。
本项目三次真实优化(干扰项 / sandbox / 去污染)全程零手工改数据 —— 这就是算子化、可插拔、可审计的生产管线相比一次性脚本的根本区别。**
