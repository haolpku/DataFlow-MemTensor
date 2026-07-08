# DataFlow-MemTensor · 数据展示站

一个**纯静态**的下钻式数据展示站(仿 OpenDCAI_Data),用于给甲方/对外展示三类数学 Mid-train 数据样本。

- 三类数据:长思维链 Long-CoT / 多步证据推理 Evidence / 长程交错思维 Interleaved,各 24 条真实样本
- 分类首页 → 点卡片下钻 → 每条样本可切「结构视图 / 训练格式」
- evidence 显示干扰证据(橙色「干扰·未用」)+ claim↔evidence 高亮联动
- interleaved 显示 (思考→工具→观察) 时间线 + 质量四轴评分
- 纯静态(`index.html` + `data.js`),适配 GitHub Pages

## 在线访问(GitHub Pages)

启用 Pages 后:`https://haolpku.github.io/DataFlow-MemTensor/site/`

## 本地查看

```bash
python3 -m http.server 8099 --directory site
# 打开 http://127.0.0.1:8099/index.html
```

## 重建 data.js(数据更新后)

`data.js` 由三条 pipeline 的最终产物打包而成:

```bash
# 先跑三条 pipeline 生成 cache_cot / cache_evidence / cache_interleaved
python3 site/build_site_data.py    # -> site/data.js
```

数据来自真实 API(gpt-4.1-mini)合成 + math_verify/sympy/grounding 校验。
