"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
# Entry points are adapters only — no business logic here.
# For platform-level routing, AgentGateway calls agent.invoke() directly.

import os
import secrets
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from framework.security.credential_detector import detect_credentials_in_value
from shared.secrets import factory as secrets_factory
from src.graph.graph import Graph

app = FastAPI(title="Agent")

# Graph() with no argument loads config/config.yaml itself (see
# src/graph/graph.py:load_runtime_config), so the standalone path runs with the
# same settings the registry would apply rather than with an empty config.
agent = Graph()

# Mirror the registry's conditional checkpointer: memory_enabled or hitl.enabled
# needs one, or those features silently no-op on this path.
_hitl_enabled = agent.config.get("hitl", {}).get("enabled", False)
_needs_checkpointer = agent.config.get("memory_enabled") or _hitl_enabled
agent.compile(checkpointer=MemorySaver() if _needs_checkpointer else None)
agent.provision_secrets(secrets_factory(namespace="ret-c2-028", agent_name="StoreOperationsFAQAgent"))


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""
    # Structured request contract: the operations passages to search, an
    # optional routing channel and optional retrieval overrides. Every field is
    # validated in PreProcessNode before any domain code reads it.
    input_context: Optional[dict[str, Any]] = None


def _bearer_matches(supplied: str, expected: str) -> bool:
    """Constant-time bearer comparison that is safe for non-ASCII header input."""
    return secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode())


def _resolve_standalone_trust(
    current: TrustLevel,
    authorization: str,
    invoke_auth_token: Optional[str],
    internal_runner_token: Optional[str],
) -> TrustLevel:
    """Authenticate standalone callers without allowing external-token elevation.

    Every node in this agent requires VERIFIED_EXTERNAL, so a caller left at
    ANONYMOUS is denied at the trust gate before any work happens. This adapter
    is the standalone equivalent of the platform's auth middleware and is what
    establishes that trust.

    The internal runner token is a distinct, deployment-issued credential: it is
    considered only for an otherwise-anonymous caller and maps exactly to
    INTERNAL. The external token maps to VERIFIED_EXTERNAL. Trust already
    established by middleware is never changed.
    """
    if current is not TrustLevel.ANONYMOUS:
        return current
    if internal_runner_token and _bearer_matches(authorization, internal_runner_token):
        return TrustLevel.INTERNAL
    if invoke_auth_token and _bearer_matches(authorization, invoke_auth_token):
        return TrustLevel.VERIFIED_EXTERNAL
    if internal_runner_token or invoke_auth_token:
        raise HTTPException(status_code=401, detail="Token is invalid or expired.")
    return TrustLevel.ANONYMOUS


# Field names are caller data too: a name is echoed back only when it is itself
# an inert identifier and carries no credential pattern of its own.
def _safe_field_label(name: Any, position: int) -> str:
    if (
        isinstance(name, str)
        and 1 <= len(name) <= 64
        and name.replace("_", "").replace("-", "").replace(".", "").isalnum()
        and not detect_credentials_in_value(name)
    ):
        return f"input_context.{name}"
    return f"input_context field #{position}"


def _reject_credentials_in_context(context: Optional[dict[str, Any]]) -> None:
    """Refuse a request whose context carries a credential-shaped value.

    Without this the request fails anyway, but opaquely: the framework's output
    gate scans every value of every node result, the first node returns the
    caller's context verbatim in its own result, and the caller receives a
    node-level error with a traceback and no indication of which field caused
    it. The request cannot succeed either way, so it is refused here with the
    field named.

    The scan uses the framework's own detector, so this refusal set matches the
    framework's block set exactly — no local approximation that could drift.
    Iterating the top-level fields is equivalent to scanning the whole mapping
    (the detector's dict case is the union over its values); iterating is what
    allows the field to be named.
    """
    if not isinstance(context, dict):
        return
    for position, (name, value) in enumerate(context.items(), start=1):
        if detect_credentials_in_value(value):
            # 400, not 422: pydantic owns 422 and returns a list of error
            # objects there, so reusing it makes client handling ambiguous.
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{_safe_field_label(name, position)} contains a credential-shaped value. " "Remove it and retry."
                ),
            )


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> Any:
    # This adapter is the entry-point auth boundary (the standalone equivalent
    # of the platform's auth middleware). Both values are deployment-level
    # caller credentials, not agent secrets: no InvocationContext exists before
    # this boundary, so per-request secret binding cannot apply.
    trust = _resolve_standalone_trust(
        getattr(request.state, "trust_level", TrustLevel.ANONYMOUS),
        request.headers.get("authorization", ""),
        os.environ.get("INVOKE_AUTH_TOKEN"),
        os.environ.get("STG_INTERNAL_RUNNER_TOKEN"),
    )
    _reject_credentials_in_context(req.input_context)
    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        return agent.invoke(req.input, ctx=ctx, input_context=req.input_context or {})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "StoreOperationsFAQAgent"}
