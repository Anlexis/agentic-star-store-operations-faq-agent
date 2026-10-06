"""AgentCore Platform v1.0 — caller-context bridge across the graph boundary.

Why this exists: ``GraphNode.execute()`` invokes the inner graph as
``subgraph.invoke(<string>, ...)`` and does NOT forward the outer state's
``input_context``. An inner node reading ``state["input_context"]`` therefore
always sees ``{}`` when it runs through the nested graph, even though the caller
supplied data. The two sanctioned subclass hooks bridge it:

    StoreOpsQAGraphNode.extract_input(state)    [runs BEFORE subgraph.invoke]
        -> set_caller_request(state["caller_request"])
    DomainWorkflowGraph._extra_initial_state()  [runs INSIDE subgraph.invoke]
        -> returns {"caller_request": <the stashed contract>}

What crosses the bridge is the VALIDATED caller contract produced by
``PreProcessNode`` — never the raw request body — so the inner graph only ever
receives fields that already passed their type, alphabet and length bounds.

Smuggling the data inside ``validated_input`` is not usable: the framework masks
that field at node boundaries, so caller text can be rewritten between hops.
This channel is not masked, which is precisely why the pre-process node screens
and bounds every field before anything enters it.

A ContextVar keeps the hand-off correct per thread/task, so concurrent
invocations in one process cannot observe each other's context.
"""

from contextvars import ContextVar
from typing import Any, Optional

_CALLER_REQUEST: ContextVar[Optional[dict[str, Any]]] = ContextVar("ret_c2_028_caller_request", default=None)


def set_caller_request(caller_request: Optional[dict[str, Any]]) -> None:
    """Stash the validated caller contract for the imminent inner-graph invoke."""
    _CALLER_REQUEST.set(caller_request or None)


def get_caller_request() -> Optional[dict[str, Any]]:
    """Read (without consuming) the stashed contract; None when unset."""
    return _CALLER_REQUEST.get()
