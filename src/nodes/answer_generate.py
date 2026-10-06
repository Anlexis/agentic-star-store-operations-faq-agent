"""AgentCore Platform v1.0 — RET-C2-028 AnswerGenerateNode.

Cited answer synthesis from retrieved chunks.

Field names follow docs/02_design.md §3.4 and src/schemas/state.py.
Reads:  normalized_query, retrieved_chunks, citations
Writes: answer, confidence_score, status

Design notes:
- SDK v1.0.0rc1 does NOT include framework.services.llm_client or LLMClient.
  This node uses deterministic cited synthesis from retrieved chunk text.
  # Production wires the real LLM here (inject via config["configurable"])
- No-match fallback: if retrieved_chunks is empty → return the configured
  fallback message WITHOUT any LLM/generation attempt (PB-02 asserts this).
- Confidence score: score of the top retrieved chunk, or None on no-match.
- system_prompt is read from config["configurable"] (not stored in State).
"""

from typing import Any, ClassVar, Dict, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel

# Audit trail (free-function API; ships in the SDK wheel only).
# Never use self.emit_trace_event — AttributeError.
from shared.utils.audit_logger import emit_trace_event

# Default fallback message when no chunks pass the score threshold.
# Must match the prompt template (prompts/ret_28_qa.j2) and PB-02 assertion.
_NO_MATCH_FALLBACK = "該当する手順が見つかりませんでした。店長または本部に確認してください。"


class AnswerGenerateNode(FunctionNode):
    """Synthesize a cited answer from retrieved store-ops KB chunks.

    When retrieved_chunks is non-empty:
      - Assembles a cited answer from chunk text.
      - Each chunk contributes one paragraph with a citation marker
        in the format: [source_doc §section].
      - Prepends a brief answer header.
      - confidence_score = score of the top (first) chunk.
      # Production wires the real LLM here using system_prompt from
      # config["configurable"]["system_prompt"] for full RAG generation.

    When retrieved_chunks is empty:
      - Returns _NO_MATCH_FALLBACK immediately.
      - No LLM/generation attempt is made.
      - confidence_score = None.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        retrieved_chunks: list[Any] = state.get("retrieved_chunks") or []

        # ── No-match path ──────────────────────────────────────────────────
        # PB-02: if retrieved_chunks is empty, return fallback immediately.
        # No LLM call, no synthesis attempt.
        if not retrieved_chunks:
            # Audited explicitly: "no passage cleared the threshold" is a
            # reportable outcome, not an absence of one. The coverage gate only
            # requires one event per node, so a branch that returns early is
            # exactly where an audit trail goes quietly missing.
            emit_trace_event(
                "answer_generated",
                {"answer_len": len(_NO_MATCH_FALLBACK), "chunks_used": 0, "no_match": True},
                state,
            )
            return {
                "answer": _NO_MATCH_FALLBACK,
                "confidence_score": None,
                "status": AgentStatus.SUCCESS.value,
            }

        # ── Cited synthesis path ───────────────────────────────────────────
        # Deterministic assembly from chunk text + citation markers.
        # Production wires the real LLM here.
        # PB-03: the answer is grounded ONLY in retrieved passage text — the
        # raw (possibly adversarial) query is NEVER echoed into the output, so prompt
        # injection in the query cannot leak into the answer. Use a static header.
        lines: list[str] = ["【関連する店舗運営手順】"]
        lines.append("")

        for chunk in retrieved_chunks:
            if not isinstance(chunk, dict):
                continue
            text = str(chunk.get("text", "")).strip()
            source_doc = str(chunk.get("source_doc", ""))
            section = str(chunk.get("section", ""))

            if not text:
                continue

            citation = f"[{source_doc} §{section}]" if section else f"[{source_doc}]"
            lines.append(f"{text} {citation}")
            lines.append("")

        # Trim trailing empty line.
        while lines and lines[-1] == "":
            lines.pop()

        answer = "\n".join(lines)
        if not answer.strip():
            # Should not happen given non-empty chunks, but guard defensively.
            answer = _NO_MATCH_FALLBACK
            emit_trace_event(
                "answer_generated",
                {"answer_len": len(answer), "chunks_used": 0, "no_match": True},
                state,
            )
            return {
                "answer": answer,
                "confidence_score": None,
                "status": AgentStatus.SUCCESS.value,
            }

        # Confidence score = score of the first (highest-ranked) chunk.
        first_chunk = retrieved_chunks[0] if retrieved_chunks else {}
        confidence_score: float | None = None
        raw_score = first_chunk.get("score")
        if raw_score is not None:
            try:
                confidence_score = float(raw_score)
            except (TypeError, ValueError):
                confidence_score = None

        # Audit: record the answer-synthesis outcome (non-sensitive counts only).
        emit_trace_event(
            "answer_generated",
            {"answer_len": len(answer), "chunks_used": len(retrieved_chunks)},
            state,
        )

        return {
            "answer": answer,
            "confidence_score": confidence_score,
            "status": AgentStatus.SUCCESS.value,
        }
