"""AgentCore Platform v1.0 — RET-C2-028 VectorRetrieveNode.

Retrieval against the store-operations knowledge base.

Reads:  normalized_query, retrieval_config, caller_request
Writes: retrieved_chunks, citations, status

Where the settings come from
----------------------------
``retrieval_config`` is seeded into the inner graph's initial state by
``DomainWorkflowGraph._extra_initial_state()`` from the ``retrieval:`` block of
config/config.yaml, after that block has been bounds-checked. The node reads
that field and nothing else.

It deliberately does NOT read ``state["config"]``: the framework never writes a
``config`` key into agent state, so such a read returns ``{}`` on every real
invocation and every declared setting silently reverts to a built-in default.

Where the passages come from
----------------------------
When the caller sends operations passages with the request, those passages are
ranked and cited — this is the path that does real work. When it sends none, the
configured external retriever is used, and failing that the empty baseline,
which makes the pipeline emit the documented "not found" message rather than
inventing a procedure.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel

# Audit trail (free-function API; ships in the SDK wheel only).
# Never use self.emit_trace_event — AttributeError.
from shared.utils.audit_logger import emit_trace_event
from src.services.service import Retriever, StoreOpsKbService

# Mirrors DomainWorkflowGraph.DEFAULT_RETRIEVAL; used only when the node is
# exercised outside the graph (direct execute() calls in tests).
_FALLBACK_RETRIEVAL: Dict[str, Any] = {
    "collection": "ret_store_operations_faq_kb",
    "top_k": 5,
    "score_threshold": 0.68,
    "caller_corpus_score_threshold": 0.25,
    "hybrid_search": True,
}


class VectorRetrieveNode(FunctionNode):
    """Retrieve store-operations passages ranked against the normalized query.

    Constructor accepts an optional external retriever for production wiring:
      VectorRetrieveNode()                       — caller passages, else empty baseline
      VectorRetrieveNode(retriever=my_retriever) — caller passages, else my_retriever
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, retriever: Retriever | None = None) -> None:
        self._service = StoreOpsKbService(external=retriever)

    def execute(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        query = state.get("normalized_query") or state.get("validated_input") or state.get("user_input") or ""
        if not isinstance(query, str) or not query.strip():
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["VectorRetrieveNode: no query to retrieve on"],
            }

        settings = self._settings(state)
        caller_request = state.get("caller_request") or {}
        documents = caller_request.get("documents") if isinstance(caller_request, dict) else None

        # A caller may narrow the search within the operator's own envelope; the
        # values were bounds-checked in PreProcessNode before reaching state.
        top_k = int(caller_request.get("top_k", settings["top_k"]))

        # The two retrievers score on different scales, so each carries its own
        # bar. Applying the vector-store bar to lexical coverage would reject
        # every passage and report "not found" for every question.
        caller_documents = documents if isinstance(documents, list) and documents else None
        default_threshold = (
            settings["caller_corpus_score_threshold"] if caller_documents else settings["score_threshold"]
        )
        score_threshold = float(caller_request.get("score_threshold", default_threshold))

        retriever = self._service.retriever_for(caller_documents)
        raw_chunks: list[dict[str, Any]] = retriever.retrieve(
            query=query.strip(),
            collection=settings["collection"],
            top_k=top_k,
            score_threshold=score_threshold,
            hybrid=settings["hybrid_search"],
        )

        # Enforce the threshold here as well: an external retriever is not
        # required to apply it, and a chunk below the bar must never be cited.
        chunks = [c for c in raw_chunks if isinstance(c, dict) and self._score_of(c) >= score_threshold]

        # Build the deduplicated citation list.
        seen: set[tuple[str, str]] = set()
        citations: list[dict[str, Any]] = []
        for chunk in chunks:
            doc_title = str(chunk.get("source_doc", ""))
            section = str(chunk.get("section", ""))
            key = (doc_title, section)
            if key not in seen:
                seen.add(key)
                citations.append({"doc_title": doc_title, "section": section})

        # Audit trail: record the retrieval side effect (counts and settings
        # only — never the query text, which is caller data).
        emit_trace_event(
            "vector_retrieve",
            {
                "collection": settings["collection"],
                "top_k": top_k,
                "score_threshold": score_threshold,
                "chunks_returned": len(chunks),
            },
            state,
        )

        # An empty result is a graceful pass-through; AnswerGenerateNode emits
        # the documented no-match message.
        return {
            "retrieved_chunks": chunks,
            "citations": citations,
            "status": AgentStatus.SUCCESS.value,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _settings(state: AgentState) -> Dict[str, Any]:
        """Return the retrieval settings seeded into state, or the fallbacks."""
        seeded = state.get("retrieval_config")
        if isinstance(seeded, dict) and seeded:
            return {**_FALLBACK_RETRIEVAL, **seeded}
        return dict(_FALLBACK_RETRIEVAL)

    @staticmethod
    def _score_of(chunk: Dict[str, Any]) -> float:
        """Return a chunk's score as a real number; unusable scores rank as 0.

        A non-numeric or non-finite score must not pass the threshold by
        accident — NaN compares False against the bar, but so does 0.0, and 0.0
        is the honest reading of "this chunk has no usable score".
        """
        raw = chunk.get("score", 0.0)
        if isinstance(raw, bool):
            return 0.0
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if value != value or value in (float("inf"), float("-inf")):
            return 0.0
        return value
