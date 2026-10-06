"""AgentCore Platform v1.0 — RET-C2-028 DomainWorkflowGraph.

Inner graph for the Cat 2 two-layer nested architecture.
Called by StoreOpsQAGraphNode.get_subgraph() in graph.py.

Pipeline (linear VectorRAG domain flow):
  START → query_normalize → vector_retrieve → answer_generate → response_validate → END

Rules enforced:
  ✅ Inherits BaseGraph (fully custom topology — no forced backbone)
  ✅ Implements all 7 BaseGraph ABC methods
  ✅ register_nodes() does NOT call super() (abstract in BaseGraph)
  ✅ Does NOT register initialize / finalize (outer backbone concerns)
  ✅ get_output() designed together with StoreOpsQAGraphNode.merge_output()
  ✅ Each node execute() returns only changed state keys
  ❌ No agenticstar imports
  ❌ Not placed under src/subagents/
"""

from typing import Any

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_caller_request
from src.services.service import Retriever
from src.nodes.answer_generate import AnswerGenerateNode
from src.nodes.query_normalize import QueryNormalizeNode
from src.nodes.response_validate import ResponseValidateNode
from src.nodes.vector_retrieve import VectorRetrieveNode
from src.schemas.state import State
from src.validation import CallerInputError, finite_in_range

# Retrieval defaults, used when config/config.yaml declares no value. Kept in one
# place so the node, the validator and the documentation cannot drift apart.
DEFAULT_RETRIEVAL: dict[str, Any] = {
    "collection": "ret_store_operations_faq_kb",
    "top_k": 5,
    "score_threshold": 0.68,
    "caller_corpus_score_threshold": 0.25,
    "hybrid_search": True,
}

# Operator-settable ranges. Values outside these fail at compile time rather
# than silently degrading: a threshold of 50 would match nothing and a top_k of
# 0 would return nothing, and both would look like "the knowledge base is empty".
_TOP_K_RANGE = (1.0, 100.0)
_SCORE_THRESHOLD_RANGE = (0.0, 1.0)


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for RET-C2-028 Store Operations FAQ Agent.

    Inherits BaseGraph directly for a fully custom node topology.
    Called by StoreOpsQAGraphNode.get_subgraph() in graph.py.

    Pipeline (linear):
      START
        → query_normalize   (QueryNormalizeNode  — expand abbreviations + JP synonyms)
        → vector_retrieve   (VectorRetrieveNode  — hybrid vector + keyword retrieval)
        → answer_generate   (AnswerGenerateNode  — cited synthesis or fallback message)
        → response_validate (ResponseValidateNode — non-empty, citation, hallucination checks)
        → END

    All nodes are FunctionNode subclasses returning partial-dict state updates.
    initialize / finalize are outer backbone concerns — not registered here.
    """

    def __init__(self, config: dict[str, Any] | None = None, retriever: Retriever | None = None) -> None:
        """Build the inner graph, optionally with a production retriever.

        `retriever` is any object satisfying src.services.service.Retriever. It
        is used when the caller supplied no passages of its own; without one the
        graph falls back to the empty baseline and reports no match.
        """
        super().__init__(config)
        self._retriever = retriever

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "ret_c2_028_store_ops_qa_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across inner and outer graph."""
        return State

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """Validate the retrieval settings before the graph compiles.

        Fails CLOSED: a non-finite or out-of-range value raises here rather than
        reaching the retrieval node, where a NaN threshold would compare False
        against every score and quietly return nothing at all — indistinguishable
        from an empty knowledge base.
        """
        self.resolve_retrieval_config()

    def resolve_retrieval_config(self) -> dict[str, Any]:
        """Return the validated retrieval settings for this graph.

        Merges the forwarded `retrieval:` block over the shipped defaults, then
        bounds every numeric. Raises CallerInputError (a ValueError) when a
        declared value cannot be used.
        """
        configurable = self.config.get("configurable") or {}
        declared = configurable.get("retrieval")
        declared = declared if isinstance(declared, dict) else {}

        vector_store = declared.get("vector_store")
        vector_store = vector_store if isinstance(vector_store, dict) else {}
        collection = vector_store.get("collection", DEFAULT_RETRIEVAL["collection"])
        if not isinstance(collection, str) or not collection.strip():
            raise CallerInputError("retrieval.vector_store.collection: must be a non-empty string")

        top_k = int(
            finite_in_range(
                declared.get("top_k", DEFAULT_RETRIEVAL["top_k"]),
                field="retrieval.top_k",
                lo=_TOP_K_RANGE[0],
                hi=_TOP_K_RANGE[1],
                integer=True,
            )
        )
        score_threshold = finite_in_range(
            declared.get("score_threshold", DEFAULT_RETRIEVAL["score_threshold"]),
            field="retrieval.score_threshold",
            lo=_SCORE_THRESHOLD_RANGE[0],
            hi=_SCORE_THRESHOLD_RANGE[1],
        )
        caller_corpus_score_threshold = finite_in_range(
            declared.get(
                "caller_corpus_score_threshold",
                DEFAULT_RETRIEVAL["caller_corpus_score_threshold"],
            ),
            field="retrieval.caller_corpus_score_threshold",
            lo=_SCORE_THRESHOLD_RANGE[0],
            hi=_SCORE_THRESHOLD_RANGE[1],
        )
        hybrid_search = declared.get("hybrid_search", DEFAULT_RETRIEVAL["hybrid_search"])
        if not isinstance(hybrid_search, bool):
            raise CallerInputError("retrieval.hybrid_search: must be true or false")

        return {
            "collection": collection.strip(),
            "top_k": top_k,
            "score_threshold": score_threshold,
            "caller_corpus_score_threshold": caller_corpus_score_threshold,
            "hybrid_search": hybrid_search,
        }

    def _extra_initial_state(self) -> dict[str, Any]:
        """Seed the inner state with the settings and the caller contract.

        The graph boundary forwards only the query string, so both the resolved
        retrieval settings and the validated caller contract are injected here —
        this is the inner half of the bridge described in
        src/graph/context_bridge.py.
        """
        extra: dict[str, Any] = {"retrieval_config": self.resolve_retrieval_config()}
        caller_request = get_caller_request()
        if caller_request:
            extra["caller_request"] = caller_request
        return extra

    # ── Node registration ─────────────────────────────────────────────────────

    def register_nodes(self) -> None:
        """Register all 4 domain nodes.

        No super() call — BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.
        Every key registered here is referenced in add_edges().
        """
        self._nodes["query_normalize"] = QueryNormalizeNode()
        self._nodes["vector_retrieve"] = VectorRetrieveNode(retriever=self._retriever)
        self._nodes["answer_generate"] = AnswerGenerateNode()
        self._nodes["response_validate"] = ResponseValidateNode()

    # ── Edge wiring ───────────────────────────────────────────────────────────

    def add_edges(self) -> None:
        """Wire the linear VectorRAG domain topology.

        Linear: query_normalize → vector_retrieve → answer_generate → response_validate.
        For this template the topology is intentionally linear — no conditional
        branching between domain nodes. route() is implemented to satisfy the ABC
        contract but add_conditional_edges() is not used.
        """
        self._sg.add_edge(START, "query_normalize")
        self._sg.add_edge("query_normalize", "vector_retrieve")
        self._sg.add_edge("vector_retrieve", "answer_generate")
        self._sg.add_edge("answer_generate", "response_validate")
        self._sg.add_edge("response_validate", END)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, state: AgentState) -> str:
        """Conditional routing — required by BaseGraph ABC.

        For this linear topology add_conditional_edges() is not used, so this
        method is never called at runtime. Implemented to satisfy the ABC contract.
        Returns END on ERROR so an unexpected call does not re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "response_validate"

    # ── Output shape ──────────────────────────────────────────────────────────

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by StoreOpsQAGraphNode.merge_output() in graph.py
        as the `sub_result` argument. Both methods are designed together:

            Inner get_output()   emits : "answer", "citations", "confidence_score", "status", "trace_id"
            Outer merge_output() reads : sub_result.get("answer")
                                         sub_result.get("citations")
                                         sub_result.get("confidence_score")
                                         sub_result.get("status")

        Additional fields (trace_id) are surfaced for observability.
        """
        return {
            # the reason must leave the subgraph or the outer graph cannot report it
            "error_code": state.get("error_code"),
            "answer": state.get("answer"),
            "citations": state.get("citations"),
            "confidence_score": state.get("confidence_score"),
            "status": state.get("status"),
            "trace_id": state.get("trace_id"),
        }
