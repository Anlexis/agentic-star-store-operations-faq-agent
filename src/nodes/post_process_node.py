"""AgentCore Platform v1.0 — RET-C2-028 PostProcessNode (outer graph post_process slot).

Output gate for the store-operations answer.

This is the OUTER graph's post_process slot. It runs after the inner
``DomainWorkflowGraph`` completes and ``StoreOpsQAGraphNode.merge_output()`` has
mapped the inner answer onto the outer state's ``result`` key.

Two things it must get right
----------------------------
**1. The domain check runs in the sanctioned hook.** ``_security_gate_output``
is ``@final`` on ``FunctionNode``: the framework calls it itself, on the dict a
node RETURNS, and it raises on a credential rather than returning an error dict.
A node cannot call it usefully and must not try — the extension point is
``_extra_security_gate_output(result)``, which this node overrides.

**2. A refusal must WITHHOLD the answer, not merely report one.** The graph's
output resolves as ``formatted_output or result``, with no status check, so
returning an error while leaving ``result`` populated still ships the un-gated
answer inside the error envelope. A falsy ``formatted_output`` is worse than
useless there: ``""`` activates the very fallback it was meant to suppress. On a
violation this node therefore clears every output-bearing field and writes a
TRUTHY withholding notice.

Node contract:
  - Extend FunctionNode; implement execute(state, config=None) -> dict
  - Return ONLY the fields this node changes (never full state)
  - Return AgentStatus enum value strings (`.value`) — never raw enum constants
  - Declare required_trust_level: ClassVar[TrustLevel]
"""

import logging
from typing import Any, ClassVar, Dict, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from framework.security.credential_detector import detect_credentials_in_value

# Audit trail (free-function API; ships in the SDK wheel only).
# Never use self.emit_trace_event — AttributeError.
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, TOO_LONG

logger = logging.getLogger(__name__)

# Fields that can carry answer text or a payload out of the graph. On a refusal
# every one of them is cleared. Listed explicitly so that adding a new
# output-bearing field without revisiting this list shows up as a test failure
# rather than as a silent leak.
_OUTPUT_BEARING_FIELDS: tuple[str, ...] = (
    "result",
    "answer",
    "retrieved_chunks",
    "citations",
)

# Truthy on purpose: an empty string here would fall through to `result`.
_WITHHELD_NOTICE = "回答は安全性チェックにより保留されました。店長または本部に確認してください。"


# Reason code -> the sentence the caller reads. A code with no entry falls
# back to the generic one rather than leaking the code itself.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Verify and release the store-operations answer, or withhold it.

    Input state keys (mapped in by StoreOpsQAGraphNode.merge_output):
        result: str            — the answer string (mapped from inner `answer`)
        answer: str            — inner-graph answer (fallback if `result` absent)
        confidence_score: float | None
        citations: list[dict]
        status: str            — status from the inner graph

    Output state keys (partial dict):
        result: str            — released answer, or "" when withheld
        formatted_output: str  — the released answer, or the withholding notice
        status: str            — SUCCESS, or ERROR when the answer is withheld
        error_log: list        — populated on a refusal; locations only
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # A run declined upstream has nothing to format. Render the reason as
        # the caller-facing body and carry the marker onward.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
                "formatted_output": message,
                "result": message,
            }
        # The inner answer is surfaced on `result` by merge_output(); fall back
        # to the raw `answer` field for direct-execute tests.
        answer = state.get("result")
        if not isinstance(answer, str) or not answer.strip():
            answer = state.get("answer", "")

        # ── Presence check ────────────────────────────────────────────────────
        # An empty answer here is a pipeline error: the inner graph must always
        # emit either a cited answer or the documented no-match message.
        if not isinstance(answer, str) or not answer.strip():
            logger.error("PostProcessNode: no answer present in state at the output gate.")
            emit_trace_event("output_postprocessed", {"outcome": "empty_answer", "answer_len": 0}, state)
            return self._withhold("the pipeline produced no deliverable answer")

        # ── Credential screen ────────────────────────────────────────────────
        # Uses the framework's own detector, so this node's refusal set is
        # exactly the framework's block set. A narrower local pattern list would
        # be a bypass: the framework would raise inside its own gate, the
        # wrapper would discard this node's delta, and the clearing below would
        # be lost along with it.
        if detect_credentials_in_value(answer):
            logger.warning("PostProcessNode: answer withheld — credential pattern in answer text.")
            emit_trace_event(
                "output_gate_violation",
                {"outcome": "credential_in_answer", "field": "result"},
                state,
            )
            return self._withhold("a credential pattern was detected in the answer text")

        logger.info("PostProcessNode: answer released; answer_len=%d", len(answer))
        emit_trace_event(
            "output_postprocessed",
            {"outcome": "released", "answer_len": len(answer)},
            state,
        )

        return {
            "result": answer,
            "formatted_output": answer,
            "status": AgentStatus.SUCCESS.value,
        }

    # ── Domain output hook ────────────────────────────────────────────────────

    def _extra_security_gate_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Domain output check, called by the framework on this node's result.

        The framework's own credential scan has already run over `result` by the
        time this is called. This hook re-checks the fields this node is
        responsible for releasing, so a refusal is enforced on the value that
        actually leaves the node — not only on the value read from state.

        Returns the result unchanged when clean; substitutes the withholding
        delta when not. It does not raise: raising here would be caught by the
        node wrapper, which discards the whole delta and therefore the clearing.
        """
        released = result.get("result")
        if isinstance(released, str) and released and detect_credentials_in_value(released):
            return self._withhold("a credential pattern was detected in the released answer")
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _withhold(reason: str) -> Dict[str, Any]:
        """Return the refusal delta, with every output-bearing field cleared.

        `reason` names a location and a rule. It never carries the matched text:
        echoing the offending value would place the credential straight back
        into the result the framework is scanning.
        """
        withheld: dict[str, Any] = {field: "" for field in _OUTPUT_BEARING_FIELDS}
        withheld["retrieved_chunks"] = []
        withheld["citations"] = []
        withheld["confidence_score"] = None
        withheld["formatted_output"] = _WITHHELD_NOTICE
        withheld["status"] = AgentStatus.ERROR.value
        withheld["error_log"] = [f"PostProcessNode: output withheld — {reason}."]
        return withheld
