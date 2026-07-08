"""
Pluggable corpus retrievers for the math sandbox.

The math sandbox's ``search`` / ``read`` tools must retrieve from a *real* corpus,
not a 6-entry in-memory dict. That toy dict is fine for a demo but does not scale:
a 10B-token data run needs the agent to ground its steps in a corpus of millions of
math documents (theorems, textbook sections, paper snippets).

This module defines a minimal ``RetrieverABC`` and three implementations spanning
the demo -> production path:

    DictRetriever      - the old 6-entry dict (kept only as a trivial fallback)
    BM25Retriever      - real lexical retrieval over an arbitrary corpus (rank_bm25);
                         runs locally with zero heavy deps, scales to ~1e5-1e6 docs;
                         good for validating the *interface* on a laptop.
    FlashRAGRetriever  - adapter over DataFlow's FlashRAGServing (e5/faiss dense
                         retrieval); the production backend for million-scale corpora
                         on a GPU cluster. Import-guarded so this module loads without
                         flashrag installed.

All three expose the same two operations the sandbox needs:

    search(query, topk) -> [{"id","text","score"}...]
    read(doc_id)        -> {"id","text"} | None

Swapping the backend is a one-line change in the sandbox / pipeline; the agent-explore
loop is untouched.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


def _tokenize(text: str) -> List[str]:
    # lightweight tokenizer: words + standalone math symbols
    return re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]", str(text).lower())


class RetrieverABC(ABC):
    """Minimal retrieval contract the math sandbox depends on."""

    @abstractmethod
    def search(self, query: str, topk: int = 5) -> List[Dict[str, Any]]:
        """Return up to ``topk`` hits: [{"id","text","score"}...]."""
        raise NotImplementedError

    @abstractmethod
    def read(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Return the full document by id, or None if missing."""
        raise NotImplementedError

    def __len__(self) -> int:
        return 0


# --------------------------------------------------------------------------- #
# 1. DictRetriever — trivial fallback (the old KB dict behaviour)
# --------------------------------------------------------------------------- #
class DictRetriever(RetrieverABC):
    """Keyword-substring retrieval over a small ``{id: text}`` dict.

    Only suitable for demos / tests. Kept so the sandbox has a zero-config default.
    """

    def __init__(self, knowledge: Dict[str, str]):
        self.docs = dict(knowledge)

    def search(self, query: str, topk: int = 5) -> List[Dict[str, Any]]:
        terms = [t for t in _tokenize(query) if t.isalnum()]
        hits = []
        for k, v in self.docs.items():
            hay = (k + " " + v).lower()
            score = sum(1 for t in terms if t in hay)
            if score:
                hits.append({"id": k, "text": v, "score": float(score)})
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:topk]

    def read(self, doc_id: str) -> Optional[Dict[str, Any]]:
        if doc_id in self.docs:
            return {"id": doc_id, "text": self.docs[doc_id]}
        return None

    def __len__(self) -> int:
        return len(self.docs)


# --------------------------------------------------------------------------- #
# 2. BM25Retriever — real lexical retrieval over an arbitrary corpus
# --------------------------------------------------------------------------- #
class BM25Retriever(RetrieverABC):
    """Real BM25 retrieval (rank_bm25) over a jsonl corpus.

    Corpus format: one JSON object per line with at least ``{"id","contents"}``
    (``contents`` or ``text`` accepted — matches FlashRAG's corpus schema, so the
    same corpus file works for both backends).

    Scales to ~1e5-1e6 docs in memory on a laptop; good for validating the sandbox
    <-> retrieval interface before moving to dense retrieval on a cluster.
    """

    def __init__(self, corpus_path: str, text_key: str = "contents", id_key: str = "id"):
        from rank_bm25 import BM25Okapi  # local dep, present on dev machine

        self.corpus_path = corpus_path
        self.text_key = text_key
        self.id_key = id_key
        self.ids: List[str] = []
        self.texts: List[str] = []

        with open(corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):  # skip blanks / comments
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get(text_key) or obj.get("text") or obj.get("contents") or ""
                doc_id = str(obj.get(id_key, i))
                self.ids.append(doc_id)
                self.texts.append(text)

        self._id2idx = {d: i for i, d in enumerate(self.ids)}
        self._bm25 = BM25Okapi([_tokenize(t) for t in self.texts])

    def search(self, query: str, topk: int = 5) -> List[Dict[str, Any]]:
        import numpy as np

        scores = self._bm25.get_scores(_tokenize(query))
        if len(scores) == 0:
            return []
        order = np.argsort(scores)[::-1][:topk]
        out = []
        for idx in order:
            if scores[idx] <= 0:
                continue
            out.append({"id": self.ids[idx], "text": self.texts[idx],
                        "score": float(scores[idx])})
        return out

    def read(self, doc_id: str) -> Optional[Dict[str, Any]]:
        idx = self._id2idx.get(str(doc_id))
        if idx is None:
            return None
        return {"id": self.ids[idx], "text": self.texts[idx]}

    def __len__(self) -> int:
        return len(self.ids)


# --------------------------------------------------------------------------- #
# 3. FlashRAGRetriever — production dense retrieval (e5/faiss) via FlashRAGServing
# --------------------------------------------------------------------------- #
class FlashRAGRetriever(RetrieverABC):
    """Adapter over DataFlow's ``FlashRAGServing`` for million-scale dense retrieval.

    This is the PRODUCTION backend: e5/bge dense embeddings + a prebuilt faiss index
    over a large math corpus. Requires ``flashrag`` + a built index (GPU cluster);
    import is deferred so this module still loads on a laptop without those deps.

    ``read`` is served from the same corpus jsonl loaded into an id->text map, so the
    sandbox's read(doc_id) stays O(1) without round-tripping the retriever.
    """

    def __init__(self,
                 retrieval_model_path: str,
                 index_path: str,
                 corpus_path: str,
                 retrieval_method: str = "e5",
                 faiss_gpu: bool = True,
                 topk: int = 5,
                 **kwargs):
        # Deferred import: only needed when this backend is actually used.
        from dataflow.serving.flash_rag_serving import FlashRAGServing

        self.topk = topk
        self._serving = FlashRAGServing(
            retrieval_method=retrieval_method,
            retrieval_model_path=retrieval_model_path,
            index_path=index_path,
            corpus_path=corpus_path,
            faiss_gpu=faiss_gpu,
            topk=topk,
            **kwargs,
        )
        # id -> text map for read()
        self._id2text: Dict[str, str] = {}
        with open(corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = str(obj.get("id", i))
                self._id2text[doc_id] = obj.get("contents") or obj.get("text") or ""

    def search(self, query: str, topk: Optional[int] = None) -> List[Dict[str, Any]]:
        import asyncio

        k = topk or self.topk
        self._serving.topk = k
        # FlashRAGServing.generate_from_input is async and returns List[List[str]]
        results = asyncio.run(self._serving.generate_from_input([query]))
        docs = results[0] if results else []
        return [{"id": f"hit{i}", "text": t, "score": None} for i, t in enumerate(docs)]

    def read(self, doc_id: str) -> Optional[Dict[str, Any]]:
        if str(doc_id) in self._id2text:
            return {"id": str(doc_id), "text": self._id2text[str(doc_id)]}
        return None

    def __len__(self) -> int:
        return len(self._id2text)
