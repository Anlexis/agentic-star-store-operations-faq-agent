"""AgentCore Platform v1.0 — RET-C2-028 ResponseValidateNode.

Validates the generated answer before it exits the inner domain workflow.

Field names follow docs/02_design.md §3.5 and src/schemas/state.py.
Reads:  answer, retrieved_chunks, citations
Writes: status (updated), optionally answer (with appended warning on soft-fail)

Validation checks:
  1. Non-empty check — answer must not be blank.
  2. Citation presence check — if retrieved_chunks was non-empty, answer must
     contain at least one '[' citation marker; soft-fail appends a warning note.
  3. Grounding check — the answer's content must come from the retrieved
     passages. The template's own framing (the section header and the citation
     markers, which this pipeline adds itself) is stripped first, then the share
     of the remaining content tokens that appear in the retrieved text must
     reach _MIN_GROUNDING_RATIO. Catches an answer asserting material the
     passages do not support; it is not a deep semantic check.

On hard validation fail: status=ERROR, answer preserved (not overwritten).
On soft citation fail: status=SUCCESS, answer amended with warning note appended.
On pass: status=SUCCESS, answer unchanged.
"""

import re
from typing import Any, ClassVar, Dict, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from src.services.service import content_tokens

# Audit trail (free-function API; ships in the SDK wheel only).
# Never use self.emit_trace_event — AttributeError.
from shared.utils.audit_logger import emit_trace_event

# Share of the answer's content tokens that must be traceable to the retrieved
# passages. The pipeline assembles answers from passage text, so a grounded
# answer scores near 1.0; the bar sits well below that so legitimate rephrasing
# is not rejected, while a largely invented answer falls under it.
#
# Grounding is measured with the SAME tokenizer the retriever scores with, so
# "grounded" means one thing across the pipeline. The previous implementation
# split on a character range that matched an entire Japanese sentence as ONE
# token, so a correctly-cited answer counted as two overlapping "words" and was
# rejected as a hallucination. It never surfaced because the check only runs
# when passages were retrieved, and the shipped retriever never returned any.
_MIN_GROUNDING_RATIO = 0.5

# Framing this pipeline adds around passage text: the section header and the
# citation markers. Stripped before grounding is measured — they are the
# template's own words, not the passages'.
_TEMPLATE_FRAMING_RE = re.compile(r"【[^】]*】|\[[^\]]*\]")
# Minimum answer length to apply citation check (very short answers may be fallback messages).
_MIN_ANSWER_LEN_FOR_CITATION_CHECK = 30


class ResponseValidateNode(FunctionNode):
    """Validate the generated answer before it leaves the inner graph.

    Sets status=SUCCESS on pass, status=ERROR on hard fail.
    Soft citation fail: appends a warning note but does not fail the run.
    answer is never overwritten on ERROR (preserved for the output gate).
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        answer: str = state.get("answer") or ""
        retrieved_chunks: list[Any] = state.get("retrieved_chunks") or []

        # ── Check 1: Non-empty answer ──────────────────────────────────────
        if not answer or not answer.strip():
            emit_trace_event(
                "response_validated",
                {"outcome": "empty_answer", "answer_len": 0},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["ResponseValidateNode: answer is empty"],
            }

        # ── Check 2: Citation presence (soft-fail) ─────────────────────────
        # Only applied when chunks were retrieved and answer is long enough
        # to expect citations (avoids false-positives on fallback messages).
        citation_warning: str = ""
        if retrieved_chunks and len(answer) >= _MIN_ANSWER_LEN_FOR_CITATION_CHECK:
            if "[" not in answer:
                citation_warning = (
                    "\n\n※ 注意: この回答には出典引用が含まれていません。"
                    "内容を店舗マニュアルで確認することを推奨します。"
                )

        # ── Check 3: Grounding ────────────────────────────────────────────
        if retrieved_chunks:
            chunk_text = " ".join(str(chunk.get("text", "")) for chunk in retrieved_chunks if isinstance(chunk, dict))
            body = _TEMPLATE_FRAMING_RE.sub(" ", answer)
            answer_terms = content_tokens(body)
            grounded_terms = answer_terms & content_tokens(chunk_text)
            ratio = len(grounded_terms) / len(answer_terms) if answer_terms else 1.0

            if answer_terms and ratio < _MIN_GROUNDING_RATIO:
                emit_trace_event(
                    "response_validated",
                    {
                        "outcome": "insufficient_grounding",
                        "grounding_ratio": round(ratio, 3),
                        "required_ratio": _MIN_GROUNDING_RATIO,
                    },
                    state,
                )
                return {
                    "status": AgentStatus.ERROR.value,
                    "error_log": [
                        f"ResponseValidateNode: only {ratio:.0%} of the answer's content is "
                        f"supported by the retrieved passages "
                        f"({_MIN_GROUNDING_RATIO:.0%} required) — answer not sufficiently grounded"
                    ],
                }

        # ── All checks passed ──────────────────────────────────────────────
        final_answer = answer + citation_warning
        emit_trace_event(
            "response_validated",
            {
                "outcome": "passed",
                "answer_len": len(final_answer),
                "citation_warning": bool(citation_warning),
            },
            state,
        )
        result: Dict[str, Any] = {"status": AgentStatus.SUCCESS.value}
        if citation_warning:
            result["answer"] = final_answer
        return result
