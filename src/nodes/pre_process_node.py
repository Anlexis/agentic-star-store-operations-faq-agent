"""AgentCore Platform v1.0 — RET-C2-028 PreProcessNode (outer graph pre_process slot).

Input validation for the store-operations FAQ request.

Two things arrive from the caller and both are validated here, before any domain
code reads them:

  1. ``user_input`` — the store-staff question. Checked for presence, length and
     plain-text-ness, then screened for prompt-injection content.
  2. ``input_context`` — the structured request contract: the operations
     passages the caller wants searched, an optional routing channel, and
     optional retrieval overrides. Every field is checked against an explicit
     bound; the validated result is written to ``caller_request`` and is the only
     form the rest of the pipeline ever sees.

The screening is owned here rather than delegated to the framework's input gate.
That gate covers ``user_input`` but not the context channel, and where it is
configured off nothing would check the request at all — so the node that owns
the caller contract enforces it itself and rejection is proved by calling
``execute()`` directly.

Rejected values are never echoed: an error names the field and the rule.

Node contract:
  - Extend FunctionNode; implement execute(state, config=None) -> dict
  - Return ONLY the fields this node changes (never full state)
  - Return AgentStatus enum value strings (`.value`) — never raw enum constants
  - Declare required_trust_level: ClassVar[TrustLevel]
"""

import json
import logging
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel

# Audit trail (free-function API; ships in the SDK wheel only).
# Never use self.emit_trace_event — AttributeError.
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import INPUT_REJECTED
from src.services.progress import emit_progress
from src.validation import (
    CallerInputError,
    InjectionRefused,
    bounded_text,
    channel_identifier,
    finite_in_range,
    inert_identifier,
    screen_injection,
)

logger = logging.getLogger(__name__)

# Maximum allowed query length (characters). Store-staff questions are short;
# this guards against resource exhaustion from pathological input.
_MAX_INPUT_LENGTH = 4_000

# Bounds for the caller-supplied passage set.
_MAX_DOCUMENTS = 20
_MAX_DOCUMENT_CHARS = 4_000

# Bounds for caller retrieval overrides. These mirror the ranges the retrieval
# config itself is validated against, so a caller can tune within the same
# envelope an operator can — and no further.
_TOP_K_RANGE = (1.0, 20.0)
_SCORE_THRESHOLD_RANGE = (0.0, 1.0)

# Fields the request contract recognises. Anything else is rejected by name
# rather than ignored: silently dropping an unknown key hides a caller mistake,
# and leaving it in place would carry unvalidated data forward.
# Cap on the JSON request envelope accepted through `user_input`. A caller with
# no structured channel carries the whole request in the text field, so the
# question bound alone cannot also bound the passages travelling with it.
_MAX_REQUEST_ENVELOPE_CHARS = 65_536

_CALLER_CONTEXT_KEYS = frozenset({"channel", "documents", "top_k", "score_threshold"})

# Fields the hosting runtime puts on the context channel itself. They are not
# part of the caller contract and this node reads none of them, but refusing
# them would refuse every invocation served that way — the caller cannot remove
# what it never added. They are accepted and ignored, which is sound precisely
# because no constraint is attached to them: ``_validate_context()`` builds its
# contract only from the caller keys above, so a runtime field never reaches
# State or the answer, and there is no promise about it the caller could be
# misled about.
_RUNTIME_CONTEXT_KEYS = frozenset({"conversation_history"})

_ALLOWED_CONTEXT_KEYS = _CALLER_CONTEXT_KEYS | _RUNTIME_CONTEXT_KEYS
_ALLOWED_DOCUMENT_KEYS = frozenset({"id", "text", "source_doc", "section"})


class PreProcessNode(FunctionNode):
    """Validate the store-operations question and the caller request contract.

    Accepts:
        user_input: str — raw store-staff question
        input_context: dict — optional request contract (see module docstring)

    Returns partial dict:
        validated_input: str  — validated question
        caller_request: dict  — validated contract (passages + overrides)
        enriched_context: dict — lightweight routing/source context
        status: str           — AgentStatus.SUCCESS.value

    On validation failure, either way (no validated_input written — the inner
    graph will not run):
        error_log: list[str]  — names the field and the rule, never the value
        status: str           — AgentStatus.SUCCESS.value plus an error_code,
                                when the caller can correct the value and send
                                the request again; AgentStatus.ERROR.value when
                                the content is refused outright (injection).

    Store-staff questions are operational rather than personal data, so no
    personal-data strip runs here.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    @staticmethod
    def _split_envelope(user_input: str) -> tuple[str, Dict[str, Any]]:
        """Split a JSON request envelope carried in ``user_input``.

        A caller that can send only text puts the same request object there
        instead of on the structured channel. It is a transport, not a second
        contract: the question and each context field go through the identical
        validation below. Text that is not a JSON object is an ordinary
        question and comes back unchanged, so only a caller who clearly
        intended an envelope can get an envelope error.

        Raises ``CallerInputError`` for an envelope the caller can correct.
        """
        stripped = user_input.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return user_input, {}
        if len(stripped) > _MAX_REQUEST_ENVELOPE_CHARS:
            # The envelope as a whole is oversized, which is a different fix
            # from a question past its own bound: shorten the passages.
            raise CallerInputError(
                f"user_input exceeds the maximum length of {_MAX_REQUEST_ENVELOPE_CHARS} characters."
            )
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError):
            return user_input, {}
        if not isinstance(parsed, dict):
            return user_input, {}
        question = parsed.get("question")
        if not isinstance(question, str):
            raise CallerInputError("user_input: a JSON request must carry a question")
        return question, {k: v for k, v in parsed.items() if k != "question"}

    def execute(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        raw_context = state.get("input_context", {}) or {}

        # ── Rule 1: non-empty string ──────────────────────────────────────────
        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            return self._reject(
                "user_input is empty or missing. " "A store operations question must be a non-empty string.",
                code="EMPTY_INPUT",
            )

        # A caller with no structured channel carries the whole request in the
        # text field. Read first, so the question is screened as a question on
        # both routes; where both carry the same field the structured channel
        # wins, because it is the declared contract.
        if not isinstance(raw_context, dict):
            return self._reject("input_context: must be an object")
        try:
            user_input, envelope_context = self._split_envelope(user_input)
        except CallerInputError as exc:
            code = "QUESTION_TOO_LONG" if "maximum length" in str(exc) else "INVALID_REQUEST"
            return self._reject(str(exc), code=code)
        if envelope_context:
            raw_context = {**envelope_context, **raw_context}
        if not user_input.strip():
            return self._reject(
                "user_input is empty or missing. " "A store operations question must be a non-empty string.",
                code="EMPTY_INPUT",
            )

        stripped = user_input.strip()

        # ── Rule 2: length check ──────────────────────────────────────────────
        if len(stripped) > _MAX_INPUT_LENGTH:
            return self._reject(
                f"user_input exceeds the maximum length of {_MAX_INPUT_LENGTH} characters.",
                code="QUESTION_TOO_LONG",
            )

        # ── Rule 3: binary / non-text detection ──────────────────────────────
        null_count = stripped.count("\x00")
        if null_count > 0:
            return self._reject("user_input appears to contain binary data. Questions must be plain text (UTF-8).")

        # ── Rule 4: prompt-injection screening on the question ───────────────
        # Terminal. Screening raises InjectionRefused, a distinct type rather
        # than a distinct message, so this stays a refusal however the wording
        # of the message is later edited.
        try:
            screen_injection(stripped, field="user_input")
        except InjectionRefused as exc:
            return self._reject(str(exc), code="")

        # ── Rule 5: the structured request contract ──────────────────────────
        # Both outcomes arrive here as the same exception hierarchy, and the
        # order of the handlers is what separates them: a passage carrying
        # injection content is screened by bounded_text() and refused, while
        # every other contract failure is a value the caller can correct.
        try:
            caller_request = self._validate_context(raw_context)
        except InjectionRefused as exc:
            return self._reject(str(exc), code="")
        except CallerInputError as exc:
            return self._reject(str(exc))

        documents = caller_request.get("documents", [])
        logger.info(
            "PreProcessNode: validated store-ops question (length=%d, passages=%d)",
            len(stripped),
            len(documents),
        )

        # Audit: record that validation passed (non-sensitive counts only).
        emit_trace_event(
            "pre_process_validated",
            {"input_len": len(stripped), "document_count": len(documents)},
            state,
        )

        return {
            "validated_input": stripped,
            "caller_request": caller_request,
            "enriched_context": {
                "source": "StoreOperationsFAQAgent",
                "channel": caller_request.get("channel", "unknown"),
            },
            "status": AgentStatus.SUCCESS.value,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _reject(reason: str, code: str = "INVALID_REQUEST") -> Dict[str, Any]:
        """Build the stop delta. `reason` names a field and a rule only.

        Two ways to stop, and the caller can act on only one of them. A value
        the caller can correct completes the run carrying ``code``, so the
        reason reaches the caller and a corrected request can be sent on the
        same conversation. Content this node refuses outright passes ``code=""``
        and terminates, so a refusal is never presented as something a reworded
        request would get past. Neither path publishes a validated question.
        """
        if code:
            # A value the caller can correct: the run COMPLETES carrying the
            # reason so the request can be sent again on the same conversation.
            emit_progress(INPUT_REJECTED)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": code,
                "error_log": [f"PreProcessNode: {reason}"],
            }
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PreProcessNode: {reason}"],
        }

    def _validate_context(self, raw_context: Any) -> Dict[str, Any]:
        """Validate ``input_context`` into the caller request contract.

        Raises CallerInputError naming the offending field. Never echoes a value.
        """
        if not isinstance(raw_context, dict):
            raise CallerInputError("input_context: must be an object")

        unknown = sorted(set(raw_context) - _ALLOWED_CONTEXT_KEYS)
        if unknown:
            # Field NAMES are caller data too — report a name only when it is
            # itself inert, otherwise report its position.
            raise CallerInputError(f"input_context: unrecognised field {self._safe_name(unknown[0], raw_context)}")

        contract: dict[str, Any] = {}

        if "channel" in raw_context:
            contract["channel"] = channel_identifier(raw_context["channel"], field="input_context.channel")

        if "top_k" in raw_context:
            contract["top_k"] = int(
                finite_in_range(
                    raw_context["top_k"],
                    field="input_context.top_k",
                    lo=_TOP_K_RANGE[0],
                    hi=_TOP_K_RANGE[1],
                    integer=True,
                )
            )

        if "score_threshold" in raw_context:
            contract["score_threshold"] = finite_in_range(
                raw_context["score_threshold"],
                field="input_context.score_threshold",
                lo=_SCORE_THRESHOLD_RANGE[0],
                hi=_SCORE_THRESHOLD_RANGE[1],
            )

        if "documents" in raw_context:
            contract["documents"] = self._validate_documents(raw_context["documents"])

        return contract

    def _validate_documents(self, raw_documents: Any) -> List[Dict[str, Any]]:
        """Validate the caller-supplied passage list."""
        if not isinstance(raw_documents, list):
            raise CallerInputError("input_context.documents: must be a list")
        if len(raw_documents) > _MAX_DOCUMENTS:
            raise CallerInputError(f"input_context.documents: must contain at most {_MAX_DOCUMENTS} entries")

        documents: List[Dict[str, Any]] = []
        for index, entry in enumerate(raw_documents):
            prefix = f"input_context.documents[{index}]"
            if not isinstance(entry, dict):
                raise CallerInputError(f"{prefix}: must be an object")

            unknown = sorted(set(entry) - _ALLOWED_DOCUMENT_KEYS)
            if unknown:
                raise CallerInputError(f"{prefix}: unrecognised field {self._safe_name(unknown[0], entry)}")

            missing = sorted({"id", "text", "source_doc"} - set(entry))
            if missing:
                raise CallerInputError(f"{prefix}: missing required field '{missing[0]}'")

            documents.append(
                {
                    "id": inert_identifier(entry["id"], field=f"{prefix}.id"),
                    "text": bounded_text(entry["text"], field=f"{prefix}.text", max_len=_MAX_DOCUMENT_CHARS),
                    "source_doc": inert_identifier(entry["source_doc"], field=f"{prefix}.source_doc"),
                    "section": (
                        inert_identifier(entry["section"], field=f"{prefix}.section")
                        if entry.get("section") is not None
                        else ""
                    ),
                }
            )

        return documents

    @staticmethod
    def _safe_name(name: Any, container: Any) -> str:
        """Render an unrecognised field name, or its position when unsafe.

        A caller controls its own key names, so a name is echoed only when it is
        an inert identifier. Anything else is reported positionally.
        """
        try:
            inert_identifier(name, field="_")
        except CallerInputError:
            try:
                position = list(container).index(name) + 1
            except (ValueError, TypeError):
                position = 0
            return f"#{position}" if position else "(unnamed)"
        return f"'{name}'"
