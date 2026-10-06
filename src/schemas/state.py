"""AgentCore Platform v1.0 — RET-C2-028 State schema."""

# State must be a flat TypedDict — never a Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.
#
# Field design follows docs/02_design.md §4 (the design of record).
# Both the outer backbone and the inner domain workflow share this State.
#
# No PII — store staff queries are operational, not personal data.
# No credentials — auth is via InvocationContext in config["configurable"].

from typing import Any, Optional

from framework.schemas.agent_state import AgentState


class State(AgentState):
    """Shared state for RET-C2-028 Store Operations FAQ Agent.

    Outer backbone fields:
      user_input       — raw query from store staff (set by caller)
      validated_input  — validated question (set by PreProcessNode)

    Inner domain pipeline fields (set by domain nodes in DomainWorkflowGraph):
      normalized_query — cleaned/expanded query (QueryNormalizeNode)
      retrieved_chunks — retrieval result list, each dict has keys:
                         chunk_id, text, score, source_doc, section
                         (VectorRetrieveNode)
      citations        — deduplicated source refs, each dict has keys:
                         doc_title, section
                         (VectorRetrieveNode)
      answer           — cited answer string (AnswerGenerateNode)
      confidence_score — top-chunk retrieval score; None on no-match
                         (AnswerGenerateNode)

    Framework-managed fields (inherited from AgentState):
      status           — AgentStatus value (multiple nodes)
      trace_id         — audit trail ID (framework)
      session_id, correlation_id, node_history, error_log, etc.
    """

    # ── Outer backbone fields ──────────────────────────────────────────────
    # user_input is inherited from AgentState; redeclared here for clarity.
    # validated_input is written by PreProcessNode after validation.
    validated_input: str

    # ── Inner domain pipeline fields ───────────────────────────────────────
    normalized_query: str
    retrieved_chunks: list[dict[str, Any]]  # list[dict] — JSON-serializable; no Pydantic
    citations: list[dict[str, Any]]  # list[dict] — deduplicated source refs
    answer: str
    confidence_score: Optional[float]

    # ── Caller contract + runtime configuration ────────────────────────────
    # caller_request  — the VALIDATED caller contract (passages the caller sent
    #                   with this request, plus any bounded retrieval overrides).
    #                   Written by PreProcessNode; carried into the inner graph
    #                   through src/graph/context_bridge.py, because the graph
    #                   boundary does not forward input_context.
    # retrieval_config — retrieval settings resolved from config/config.yaml and
    #                   seeded into the inner graph by DomainWorkflowGraph.
    #                   Plain primitives only, so the state stays msgpack-safe.
    caller_request: dict[str, Any]
    retrieval_config: dict[str, Any]
    error_code: Optional[str]
