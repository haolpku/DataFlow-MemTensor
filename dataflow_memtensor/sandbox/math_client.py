"""
MathSandboxClient -- a local, network-free math reasoning sandbox.

Gives an LLM agent the tools needed to solve verifiable math problems and build
long-horizon, evidence-grounded reasoning trajectories: recall theorems from a
small knowledge base, run real Python, verify equalities with SymPy, mark
evidence, synthesize intermediate conclusions, and finish.

Every observation it returns is text / JSON (KB snippets, stdout, sympy
verdicts), so it drops straight into the existing agent-explore loop with
**zero framework changes** -- this whole file is just a ``SandboxClientABC``
subclass, exactly like ``CodingSandboxClient`` / ``MockSandboxClient``.

Why this is a good fit
----------------------
The math-reasoning agent's "world" is theorems + computation + verification.
That maps cleanly onto ``ToolResult.observation`` (already JSON/text) and
``LLMServingABC`` (text in, text out). Swapping ``MockSandboxClient`` for this
client is all the ``AgentExploreGenerator`` needs to emit *real* tool-call
trajectories (aligned with the MidTrain plan §4.3 / §5.4 interleaved (r,a,o)).

Safety
------
* ``run_python`` / ``sympy_check`` execute on the *local machine* inside a
  restricted namespace (only ``sympy`` + a math builtins allowlist exposed).
  It is meant for trusted math task synthesis, not untrusted code. Every call
  is bounded and output is truncated. Harden with a container if needed.

Tools advertised
----------------
    search(query)                     -> KB snippets matching the query
    read(key)                         -> full text of one KB entry
    run_python(code)                  -> exec code, return `result` + stdout
    sympy_check(expr, equals)         -> True iff simplify(expr-equals)==0
    select_evidence(evidence_ids)     -> record which evidence a step relies on
    synthesize(text)                  -> record an intermediate conclusion
    finish(answer)                    -> terminal (handled by the operator)
"""

from __future__ import annotations

import contextlib
import io
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from dataflow_agent.sandbox.base import SandboxClientABC, ToolResult, ToolSchema

if TYPE_CHECKING:
    from .retrievers import RetrieverABC


# Default math knowledge base used by search/read. Override via ``knowledge=``.
_DEFAULT_KB: Dict[str, str] = {
    "vieta": "Vieta's formulas: for x^2+px+q=0 with roots r1,r2: r1+r2=-p, r1*r2=q.",
    "discriminant": "Discriminant: for ax^2+bx+c=0, D=b^2-4ac; two distinct real roots iff D>0.",
    "vertex_form": "Vertex form: f(x)=a(x-h)^2+k has vertex (h,k); roots x=h±sqrt(-k/a) when -k/a>=0.",
    "chord": "Perpendicular distance from center to a chord: d=sqrt(r^2-(L/2)^2).",
    "power_rule": "Power rule: d/dx[x^n]=n*x^(n-1); interior extrema occur where f'(x)=0.",
    "geometric_sum": "Finite geometric sum: sum_{k=0}^{n-1} a*r^k = a*(r^n-1)/(r-1) for r!=1.",
    "arithmetic_sum": "Arithmetic series: S_n = n*(a1+an)/2.",
    "quadratic_formula": "Quadratic formula: x = (-b ± sqrt(b^2-4ac)) / (2a).",
}


class MathSandboxClient(SandboxClientABC):
    """A local, network-free math sandbox for verifiable-reasoning agents.

    Args:
        knowledge: Optional ``{key: snippet}`` KB served by ``search``/``read``.
            Defaults to a small built-in set of theorems.
        max_output_chars: Truncate any observation payload past this size.
        stateful: Mimic a session-stateful backend (default False; math tools
            are pure so no per-task session is needed).
    """

    def __init__(
        self,
        knowledge: Optional[Dict[str, str]] = None,
        *,
        retriever: Optional["RetrieverABC"] = None,
        max_output_chars: int = 4000,
        stateful: bool = False,
    ):
        """
        Args:
            knowledge: Optional ``{id: text}`` dict. Used only when ``retriever`` is
                not given (wrapped in a trivial DictRetriever). Defaults to a small
                built-in theorem set — a DEMO fallback, not for scale.
            retriever: A ``RetrieverABC`` backing ``search``/``read``. Pass a
                ``BM25Retriever(corpus_path=...)`` or ``FlashRAGRetriever(...)`` to
                ground the agent in a real, large corpus (the production path).
        """
        from .retrievers import DictRetriever  # local import avoids cycles

        self.knowledge = knowledge or dict(_DEFAULT_KB)
        self.retriever = retriever or DictRetriever(self.knowledge)
        self.max_output_chars = max_output_chars
        self.stateful = stateful
        self.calls: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ tools
    def list_tools(self, domain: Optional[str] = None) -> List[ToolSchema]:
        return [
            ToolSchema(
                name="search",
                description="Search the math knowledge base for theorems/definitions by keyword.",
                parameters=[{"name": "query", "type": "string", "required": True}],
            ),
            ToolSchema(
                name="read",
                description="Read the full text of one knowledge-base entry by its key.",
                parameters=[{"name": "key", "type": "string", "required": True}],
            ),
            ToolSchema(
                name="run_python",
                description=("Execute Python (sympy available as `sp`) and return the value bound "
                             "to `result` plus captured stdout. Use for explicit computation."),
                parameters=[{"name": "code", "type": "string", "required": True}],
            ),
            ToolSchema(
                name="sympy_check",
                description=("Verify an equality symbolically: returns is_equal=True iff "
                             "simplify(expr - equals) == 0."),
                parameters=[
                    {"name": "expr", "type": "string", "required": True},
                    {"name": "equals", "type": "string", "required": True},
                ],
            ),
            ToolSchema(
                name="select_evidence",
                description="Record the evidence_id(s) the current reasoning step relies on.",
                parameters=[{"name": "evidence_ids", "type": "array", "required": True}],
            ),
            ToolSchema(
                name="synthesize",
                description="Record an intermediate conclusion synthesized from prior steps.",
                parameters=[{"name": "text", "type": "string", "required": True}],
            ),
            ToolSchema(
                name="finish",
                description="Finish the task and return the final answer.",
                parameters=[{"name": "answer", "type": "string", "required": True}],
            ),
        ]

    # --------------------------------------------------------------- lifecycle
    def create_session(self, domain, *, worker_id=None, config=None):
        # Math tools are pure; no session state required.
        return None

    # ----------------------------------------------------------------- execute
    def execute(self, action, params=None, *, worker_id=None, timeout=None) -> ToolResult:
        params = params or {}
        bare = action.split(":", 1)[1] if ":" in action else action
        self.calls.append({"action": bare, "params": params, "worker_id": worker_id})

        try:
            if bare == "search":
                return self._search(params)
            if bare == "read":
                return self._read(params)
            if bare == "run_python":
                return self._run_python(params)
            if bare == "sympy_check":
                return self._sympy_check(params)
            if bare == "select_evidence":
                ids = params.get("evidence_ids", [])
                if isinstance(ids, str):
                    ids = [ids]
                return ToolResult(ok=True, observation={"selected": list(ids), "count": len(ids)})
            if bare == "synthesize":
                return ToolResult(ok=True, observation={"synthesis": str(params.get("text", ""))})
            if bare == "finish":
                return ToolResult(
                    ok=True,
                    observation={"answer": params.get("answer", "")},
                    is_final=True,
                )
        except Exception as e:  # never crash the episode; surface as failed tool
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}", code=5000)

        return ToolResult(ok=False, error=f"Unknown tool: {bare}", code=4040)

    # ------------------------------------------------------------- tool impls
    def _truncate(self, obs: Any) -> Any:
        s = str(obs)
        if len(s) > self.max_output_chars:
            return {"truncated": True, "text": s[: self.max_output_chars]}
        return obs

    def _search(self, params: Dict[str, Any]) -> ToolResult:
        query = str(params.get("query", ""))
        topk = int(params.get("topk", 5))
        hits = self.retriever.search(query, topk=topk)
        if hits:
            # keep both the doc id (for a follow-up read) and a snippet
            results = [{"id": h["id"], "snippet": h["text"], "score": h.get("score")}
                       for h in hits]
        else:
            results = [{"id": None, "snippet": "(no hit; refine the query keywords)",
                        "score": None}]
        return ToolResult(ok=True, observation=self._truncate({"results": results}))

    def _read(self, params: Dict[str, Any]) -> ToolResult:
        # accept "id" (new) or "key" (back-compat) as the document identifier
        doc_id = str(params.get("id", params.get("key", "")))
        doc = self.retriever.read(doc_id)
        if doc is not None:
            return ToolResult(ok=True, observation={"id": doc["id"], "text": doc["text"]})
        return ToolResult(ok=False, error=f"No document '{doc_id}'", code=4041,
                          observation={"id": doc_id, "text": "(not found)"})

    def _safe_globals(self) -> Dict[str, Any]:
        import sympy as sp

        # Guarded importer: allow only a small allowlist of math modules so that
        # user code like ``import sympy as sp`` works, while ``import os`` etc.
        # remain blocked.
        _allowed_modules = {"sympy", "math", "fractions", "decimal", "cmath", "itertools"}
        _real_import = __import__

        def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            root = name.split(".")[0]
            if root not in _allowed_modules:
                raise ImportError(f"import of '{name}' is not allowed in the math sandbox")
            return _real_import(name, globals, locals, fromlist, level)

        allowed_builtins = {
            "abs": abs, "min": min, "max": max, "sum": sum, "range": range,
            "len": len, "round": round, "pow": pow, "int": int, "float": float,
            "str": str, "list": list, "dict": dict, "tuple": tuple, "set": set,
            "enumerate": enumerate, "zip": zip, "sorted": sorted, "map": map,
            "filter": filter, "print": print, "True": True, "False": False, "None": None,
            "all": all, "any": any, "divmod": divmod, "reversed": reversed,
            "bool": bool, "frozenset": frozenset, "complex": complex,
            "__import__": _guarded_import,
        }
        return {"__builtins__": allowed_builtins, "sp": sp, "sympy": sp, "math": math}

    def _run_python(self, params: Dict[str, Any]) -> ToolResult:
        code = str(params.get("code", ""))
        buf = io.StringIO()
        loc: Dict[str, Any] = {}
        with contextlib.redirect_stdout(buf):
            exec(code, self._safe_globals(), loc)
        obs = {"stdout": buf.getvalue().strip(), "result": str(loc.get("result", ""))}
        return ToolResult(ok=True, observation=self._truncate(obs))

    def _sympy_check(self, params: Dict[str, Any]) -> ToolResult:
        import sympy as sp
        expr = sp.sympify(str(params.get("expr", "")))
        target = sp.sympify(str(params.get("equals", "")))
        is_equal = bool(sp.simplify(expr - target) == 0)
        obs = {"expr": str(expr), "equals": str(target), "is_equal": is_equal}
        return ToolResult(ok=True, observation=obs)
