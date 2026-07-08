# data/

数据文件**不随仓库发布**(见 `.gitignore`)。运行 pipeline 前请在此放入以下文件。

## `evidence_seed.jsonl` — 证据 pipeline 种子题

每行一个 JSON,至少包含题面(可选参考答案用于 math_verify 校验):

```json
{"instruction": "Find the product of the two roots of x^2 - 5x - 8 = 0.", "golden_answer": "-8"}
```

生产时换成真实题库(NuminaMath / OpenMathReasoning / OpenR1 …);
`FileStorage` 支持 `hf:` 前缀直读 HuggingFace 数据集。

## `math_corpus.jsonl` — 检索语料(search/read 的来源)

FlashRAG / BM25 通用 schema,每行:

```json
{"id": "vieta", "contents": "Vieta's formulas: for x^2+px+q=0 with roots r1,r2 ..."}
```

- 本机验证:几百~百万行,配 `BM25Retriever(corpus_path=...)`。
- 生产:百万级语料 + e5/faiss 索引,配 `FlashRAGRetriever(...)`。

通过环境变量指向语料:
```bash
export MEMTENSOR_CORPUS=data/math_corpus.jsonl
```
