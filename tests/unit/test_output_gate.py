"""RET-C2-028 — the output gate withholds rather than merely reporting.

The graph resolves its output as ``formatted_output or result``, with no status
check. Two consequences drive every test here:

  * returning an error while leaving ``result`` populated still ships the
    un-gated answer inside the error envelope;
  * a FALSY ``formatted_output`` activates that fallback, so ``""`` used to mean
    "withheld" produces exactly the leak it was written to prevent.

The credential screen uses the framework's own detector, so the values probed
below are ones the framework itself recognises. A pattern the framework knows
and the template misses is not a smaller net — it is a bypass, because the
framework then raises inside its own gate and the node's clearing is discarded
along with the rest of its delta.
"""

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.security.credential_detector import detect_credentials_in_value
from src.nodes.post_process_node import _OUTPUT_BEARING_FIELDS, PostProcessNode

CLEAN_ANSWER = "【関連する店舗運営手順】\n\n閉店手順: レジ締めを実施する。 [store-ops-manual §3.2]"

# Shapes the framework's detector recognises. Probing with one it does not know
# would report the gate as safe when it is not.
FRAMEWORK_KNOWN_CREDENTIALS = [
    "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
    "sk_live_" + "ABCDEFGHIJKLMNOP1234",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
    "AKIAIOSFODNN7EXAMPLE",
    "Bearer abcdef1234567890abcdef",
    # No inline password: the framework's connection-string pattern matches any
    # such URL, so the fixture exercises it without committing a literal
    # credential — which the repository's own credential scan rightly forbids
    # even in test code.
    "postgresql://ops-db.example.invalid:5432/store_operations",
]


def state(**kwargs):
    base = {"result": CLEAN_ANSWER, "citations": [], "error_log": []}
    base.update(kwargs)
    return base


class TestCleanPath:
    """A control, so a refuse-everything gate cannot pass this suite."""

    def test_clean_answer_is_released(self):
        result = PostProcessNode().execute(state())
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["result"] == CLEAN_ANSWER
        assert result["formatted_output"] == CLEAN_ANSWER

    def test_answer_falls_back_to_the_inner_field(self):
        result = PostProcessNode().execute(state(result=None, answer=CLEAN_ANSWER))
        assert result["result"] == CLEAN_ANSWER


class TestWithholding:
    @pytest.mark.parametrize("secret", FRAMEWORK_KNOWN_CREDENTIALS)
    def test_credential_in_answer_is_withheld(self, secret):
        answer = f"閉店手順: 管理APIキーは {secret} です。"
        result = PostProcessNode().execute(state(result=answer))
        assert result["status"] == AgentStatus.ERROR.value
        assert secret not in str(result)

    @pytest.mark.parametrize("secret", FRAMEWORK_KNOWN_CREDENTIALS)
    def test_every_output_bearing_field_is_explicitly_cleared(self, secret):
        """Each field must be PRESENT and empty — not merely absent.

        The graph merges a node's partial dict into state, so omitting a key
        leaves the previous value untouched. An assertion that accepts an absent
        key therefore passes on a gate that clears nothing at all, which is the
        exact leak this test exists to catch.
        """
        answer = f"閉店手順: {secret}"
        result = PostProcessNode().execute(state(result=answer, answer=answer, retrieved_chunks=[{"text": answer}]))
        for field in _OUTPUT_BEARING_FIELDS:
            assert field in result, f"{field} omitted — the previous value would survive"
            assert not result[field], f"{field} still carries a payload"
        assert secret not in str(result)

    def test_withholding_notice_is_truthy(self):
        """A falsy formatted_output activates the `or result` fallback — the
        exact leak the notice exists to prevent."""
        result = PostProcessNode().execute(state(result=f"x {FRAMEWORK_KNOWN_CREDENTIALS[0]}"))
        assert result["formatted_output"], "notice must be truthy"
        assert isinstance(result["formatted_output"], str)

    def test_resolved_output_is_the_notice_not_the_answer(self):
        """Mirrors how the graph resolves its output: formatted_output or result."""
        answer = f"閉店手順 {FRAMEWORK_KNOWN_CREDENTIALS[0]}"
        result = PostProcessNode().execute(state(result=answer))
        resolved = result.get("formatted_output") or result.get("result")
        assert resolved == result["formatted_output"]
        assert FRAMEWORK_KNOWN_CREDENTIALS[0] not in str(resolved)

    def test_empty_answer_is_withheld_not_released(self):
        result = PostProcessNode().execute(state(result="", answer=""))
        assert result["status"] == AgentStatus.ERROR.value
        assert result["formatted_output"]

    def test_refusal_message_names_a_location_never_the_value(self):
        secret = FRAMEWORK_KNOWN_CREDENTIALS[0]
        result = PostProcessNode().execute(state(result=f"閉店手順 {secret}"))
        joined = " ".join(result["error_log"])
        assert secret not in joined
        assert "withheld" in joined


class TestDetectorParity:
    """The template's refusal set must equal the framework's block set.

    Pinned as a property rather than a fixed list: a local pattern set that
    drifts narrower than the framework's is a containment bypass, and a fixed
    list would not notice the framework widening.
    """

    @pytest.mark.parametrize(
        "text",
        FRAMEWORK_KNOWN_CREDENTIALS
        + [
            CLEAN_ANSWER,
            "閉店手順: レジ締めを実施する。",
            "SKU-48210 の棚卸手順",
            "sk-short",
            "a" * 64,  # a long hex-ish run is NOT a credential pattern
        ],
    )
    def test_withholding_matches_the_framework_detector(self, text):
        answer = f"手順: {text}"
        withheld = PostProcessNode().execute(state(result=answer))["status"] == AgentStatus.ERROR.value
        assert withheld == bool(detect_credentials_in_value(answer))


class TestOutputHook:
    """The domain check also runs where the framework calls it — on the dict the
    node returns — so a refusal is enforced on the value that actually leaves."""

    def test_hook_passes_a_clean_result_through(self):
        clean = {"result": CLEAN_ANSWER, "status": AgentStatus.SUCCESS.value}
        assert PostProcessNode()._extra_security_gate_output(clean) == clean

    def test_hook_withholds_a_credential_bearing_result(self):
        secret = FRAMEWORK_KNOWN_CREDENTIALS[0]
        result = PostProcessNode()._extra_security_gate_output(
            {"result": f"手順 {secret}", "status": AgentStatus.SUCCESS.value}
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert secret not in str(result)

    def test_hook_does_not_raise(self):
        """Raising here would be caught by the node wrapper, which discards the
        whole delta — and with it the clearing."""
        secret = FRAMEWORK_KNOWN_CREDENTIALS[0]
        PostProcessNode()._extra_security_gate_output({"result": f"x {secret}"})
