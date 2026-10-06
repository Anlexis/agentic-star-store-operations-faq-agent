"""AgentCore Platform v1.0 — RET-C2-028 outer graph.

Cat 2 two-layer nested architecture.

Outer backbone (fixed — same as Cat 1, do NOT override add_edges()):
  START → initialize → pre_process → main → {route} → post_process → finalize → END
                                           ↑ (RETRY, max 3)
                                        pre_process

`main` slot = StoreOpsQAGraphNode(GraphNode) → delegates to DomainWorkflowGraph (inner BaseGraph).

Directory layout:
  src/graph/graph.py                 ← outer graph (this file)
  src/graph/domain_workflow_graph.py ← inner graph (VectorRAG domain pipeline)

Rules enforced:
  ✅ RetC2028Agent inherits AgentBaseGraph
  ✅ super().register_nodes() called first (fills initialize + finalize)
  ✅ StoreOpsQAGraphNode assigned to self._nodes["main"]
  ✅ merge_output() returns only changed keys
  ❌ add_edges() NOT overridden on the outer graph
  ❌ No imports of the underlying platform SDK
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import yaml

from framework.schemas.agent_status import AgentStatus
from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.invocation_context import TrustLevel
from src.graph.context_bridge import set_caller_request
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State
from src.services.service import Retriever

if TYPE_CHECKING:  # import cycle at runtime; needed only for the annotation
    from src.graph.domain_workflow_graph import DomainWorkflowGraph

# Repo-root runtime config: src/graph/graph.py -> parents[2] = repo root.
# config/agent.yaml is the static registry manifest (identity keys only);
# every runtime value lives in config/config.yaml.
_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_runtime_config() -> dict[str, Any]:
    """Load config/config.yaml (max_retry, timeout_s, retrieval block).

    Used by the agent constructor, so the backbone's retry settings actually
    apply, and by StoreOpsQAGraphNode._parent_config(), which forwards the
    `retrieval:` block to the inner graph. A missing or unreadable file degrades
    to {} and the retrieval node then falls back to its documented defaults.
    """
    try:
        loaded = yaml.safe_load(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


class StoreOpsQAGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of RetC2028Agent.

    Wraps DomainWorkflowGraph (inner Cat 2 BaseGraph).
    Called by AgentBaseGraph backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()    — instantiate and return DomainWorkflowGraph
      extract_input()   — pull the validated question from outer state
      merge_output()    — map sub_result fields into outer state delta (changed keys only)
      error_strategy    — "propagate": re-raise inner errors as SubgraphError (fail-fast)
    """

    # "propagate": re-raise inner graph exceptions as SubgraphError (default — fail fast).
    # "handle": call on_subgraph_error() instead — use for graceful degradation.
    error_strategy: ClassVar[str] = "propagate"

    # False: HITL interrupts are handled inside the inner graph only.
    # True: surface inner HITL interrupt to the outer caller.
    propagate_hitl: ClassVar[bool] = False

    # Production retriever for the inner graph; None uses the empty baseline.
    # Set by RetC2028Agent.register_nodes() from the agent's own constructor.
    retriever: Optional["Retriever"] = None

    def get_subgraph(self) -> "DomainWorkflowGraph":
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported lazily (inside the method) to avoid
        circular-import risk at module load time.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config(), retriever=self.retriever)

    def execute(self, state: AgentState) -> dict[str, Any]:
        """Skip the inner graph when the request was already found unacceptable.

        A request declined by pre_process has no validated input to act on, so
        running the inner graph would only produce a second, vaguer reason for
        the same rejection - and overwrite the specific one already settled.
        """
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        result: dict[str, Any] = super().execute(state)
        return result

    def extract_input(self, state: AgentState) -> str:
        """Return the string input passed into inner_graph.invoke().

        Also stashes the VALIDATED caller contract on the context bridge: the
        graph boundary forwards only this string, so anything structured the
        caller sent would otherwise be invisible to the inner nodes. Runs
        immediately before subgraph.invoke() — see src/graph/context_bridge.py.
        """
        caller_request = state.get("caller_request")
        set_caller_request(caller_request if isinstance(caller_request, dict) else None)
        return str(state.get("validated_input", state.get("user_input", "")))

    def merge_output(self, state: AgentState, sub_result: dict[str, Any]) -> dict[str, Any]:
        """Map inner graph sub_result back into the outer state delta.

        sub_result is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys — never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  → "answer", "citations", "confidence_score",
                                       "status", "trace_id"
          This merge_output() reads → sub_result.get("answer")         → "result"
                                       sub_result.get("citations")      → "citations"
                                       sub_result.get("confidence_score") → "confidence_score"
                                       sub_result.get("status")         → "status"
        """
        return {
            # Outer reason wins: a reason settled before the inner run is the real
            # one, and a plain sub_result.get() would erase it.
            "error_code": state.get("error_code") or sub_result.get("error_code", ""),
            "result": sub_result.get("answer"),
            "citations": sub_result.get("citations"),
            "confidence_score": sub_result.get("confidence_score"),
            "status": sub_result.get("status"),
        }

    def _parent_config(self) -> dict[str, Any]:
        """Forward the runtime retrieval settings to the inner graph.

        Loads config/config.yaml and forwards the `retrieval:` section under
        config["configurable"]. Returning {} here would silently disconnect
        every declared retrieval value: nothing else populates the inner graph's
        config, so the nodes would fall back to their defaults while the shipped
        config file still advertised the operator's settings.
        """
        runtime = load_runtime_config()
        configurable: dict[str, Any] = {}
        retrieval = runtime.get("retrieval")
        if isinstance(retrieval, dict):
            configurable["retrieval"] = retrieval
        return {"configurable": configurable}


class RetC2028Agent(AgentBaseGraph):
    """Outer graph for RET-C2-028 Store Operations FAQ Agent (Cat 2).

    Inherits AgentBaseGraph directly (L1 Base). Domain logic is fully
    encapsulated in StoreOpsQAGraphNode (main slot), which delegates to
    DomainWorkflowGraph (inner BaseGraph).

    Backbone (fixed — identical to Cat 1):
        START → initialize → pre_process → main → post_process → finalize → END

    register_nodes() is the ONLY override:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process: PreProcessNode (request validation + empty-question rejection)
      - main:        StoreOpsQAGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode (output gate)

    add_edges() is NOT overridden — backbone wiring belongs to the framework.
    """

    # Declare the trust gate on the outer agent class. Store operations
    # questions originate from authenticated store-staff channels, so
    # VERIFIED_EXTERNAL is the minimum caller trust level.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, config: Optional[dict[str, Any]] = None, retriever: Optional["Retriever"] = None) -> None:
        """Build the agent, defaulting to the shipped runtime configuration.

        Callers that pass no config get config/config.yaml rather than {} — an
        empty config would leave the backbone on its built-in retry defaults
        while the shipped file advertised the operator's values.

        `retriever` is the deployment's own knowledge-base client (any object
        satisfying src.services.service.Retriever). It is consulted when the
        caller sent no passages with the request; without one the agent reports
        no match rather than answering from nothing.
        """
        super().__init__(config if config is not None else load_runtime_config())
        self._retriever = retriever

    @property
    def name(self) -> str:
        """Agent identifier registered with AgentRegistry."""
        return "ret_c2_028"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first — it injects the
        framework's default InitializeNode (sets schema_version, session_id,
        trust_level) and FinalizeNode (builds response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        main_node = StoreOpsQAGraphNode()
        main_node.retriever = self._retriever
        self._nodes["main"] = main_node
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.


# Alias: the standalone entry point imports the agent as `Graph`.
Graph = RetC2028Agent
