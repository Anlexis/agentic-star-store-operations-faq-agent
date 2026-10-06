# docs/02_design.md — RET-C2-028 Store Operations FAQ Agent

## 1. Overview

RET-C2-028 is a Cat 2 VectorRAG Q&A agent for retail store operations. Store staff
ask questions about opening/closing procedures, equipment operation, emergency protocols,
and general FAQ. The agent retrieves from a store operations knowledge base and responds
with cited answers grounded exclusively in retrieved chunks.

**Template ID:** RET-C2-028  
**Category:** Cat 2 (domain-specific pipeline)  
**Industry:** RET (Retail)  
**Pattern:** VectorRAG nested (Cat 2 two-layer architecture)  
**L1 Base (framework base class):** `AgentBaseGraph` (outer) + `BaseGraph` (inner) — direct framework inheritance

---

## 2. Architecture

### 2.1 Cat 2 Two-Layer Architecture

This template follows the mandatory Cat 2 nested-graph pattern:

```
Outer backbone (AgentBaseGraph — fixed, do NOT override add_edges()):
  START → initialize → pre_process → main → post_process → finalize → END
                                         ↑ (RETRY, max 3)
                                      pre_process

  pre_process  = PreProcessNode          (request validation)
  main         = StoreOpsQAGraphNode     (GraphNode — delegates to inner graph)
  post_process = PostProcessNode         (output gate)

Inner domain workflow (BaseGraph — fully custom topology):
  START → query_normalize → vector_retrieve → answer_generate → response_validate → END
```

**Outer graph:** `src/graph/graph.py` — `RetC2028Agent(AgentBaseGraph)`  
**Inner graph:** `src/graph/domain_workflow_graph.py` — `DomainWorkflowGraph(BaseGraph)`  
**GraphNode:** `StoreOpsQAGraphNode(GraphNode)` defined in `src/graph/graph.py`

### 2.2 Outer Graph — `RetC2028Agent(AgentBaseGraph)`

- Inherits `AgentBaseGraph` directly (L1 framework)
- Fixed 5-node backbone: initialize → pre_process → main → post_process → finalize
- `register_nodes()` calls `super().register_nodes()` first (fills initialize + finalize)
- `add_edges()` is **NOT** overridden — backbone wiring belongs to the framework
- `pre_process` slot: `PreProcessNode` — validates the question and the caller request contract. A value the caller can correct ends the run by completing it with a reason code; content that is refused outright terminates it (§6.3)
- `main` slot: `StoreOpsQAGraphNode` — wraps `DomainWorkflowGraph`; skips it when the request was already declined
- `post_process` slot: `PostProcessNode` — output gate; renders a declined request's reason, releases the answer, or withholds it

### 2.3 `StoreOpsQAGraphNode(GraphNode)` — main slot

| Method | Contract |
|--------|----------|
| `get_subgraph()` | Instantiate and return `DomainWorkflowGraph(config=self._parent_config())` |
| `execute(state)` | Return `{status: SUCCESS, error_code: <marker>}` without running the inner graph when `error_code` is already set; otherwise delegate to the framework's `GraphNode.execute()` |
| `extract_input(state)` | Return `state.get("validated_input", state.get("user_input", ""))` |
| `merge_output(state, sub_result)` | Map inner sub_result fields into outer state delta (changed keys only) |
| `error_strategy` | `"propagate"` (default — re-raise inner errors as SubgraphError) |

The `execute()` override is what keeps a declined request from being answered
twice over: a request `pre_process` found unacceptable has no validated question
to act on, so running the inner graph could only produce a second, vaguer reason
for the same rejection — and overwrite the specific one already settled.

`merge_output()` maps:

```python
return {
    # Outer reason wins: a reason settled before the inner run is the real one,
    # and a plain sub_result.get() would erase it.
    "error_code":       state.get("error_code") or sub_result.get("error_code", ""),
    "result":           sub_result.get("answer"),
    "citations":        sub_result.get("citations"),
    "confidence_score": sub_result.get("confidence_score"),
    "status":           sub_result.get("status"),
}
```

### 2.4 Inner Graph — `DomainWorkflowGraph(BaseGraph)`

Located at `src/graph/domain_workflow_graph.py` (only accepted path).

Implements all 7 BaseGraph ABC methods:

| Method | Implementation |
|--------|----------------|
| `name` | `"ret_c2_028_store_ops_qa_workflow"` |
| `state_schema` | returns `State` |
| `_validate_config()` | Validates collection, top_k, score_threshold keys; non-fatal pass |
| `register_nodes()` | Registers 4 domain nodes (no `super()` call) |
| `add_edges()` | Linear: START → query_normalize → vector_retrieve → answer_generate → response_validate → END |
| `route()` | Returns END on ERROR status; required by ABC for linear topology |
| `get_output()` | Returns `error_code`, `answer`, `citations`, `confidence_score`, `status`, `trace_id` |

`register_nodes()` does NOT call `super()` — BaseGraph.register_nodes() is abstract.  
Does NOT register initialize / finalize — those are outer backbone concerns.

`get_output()` (designed together with `StoreOpsQAGraphNode.merge_output()`):

```python
return {
    # the reason must leave the subgraph or the outer graph cannot report it
    "error_code":       state.get("error_code"),
    "answer":           state.get("answer"),
    "citations":        state.get("citations"),
    "confidence_score": state.get("confidence_score"),
    "status":           state.get("status"),
    "trace_id":         state.get("trace_id"),
}
```

---

## 3. Domain Node Pipeline

### 3.1 Node Summary

| Node | Class | Location | Role |
|------|-------|----------|------|
| `query_normalize` | `QueryNormalizeNode` | `src/nodes/query_normalize.py` | Deterministic normalize + abbreviation expand |
| `vector_retrieve` | `VectorRetrieveNode` | `src/nodes/vector_retrieve.py` | Hybrid vector+keyword retrieval with score threshold |
| `answer_generate` | `AnswerGenerateNode` | `src/nodes/answer_generate.py` | Cited answer synthesis from retrieved chunks |
| `response_validate` | `ResponseValidateNode` | `src/nodes/response_validate.py` | Non-empty check, citation presence, grounding check |

All nodes: `FunctionNode` subclass, `execute()` returns partial-dict (changed keys only).

### 3.2 QueryNormalizeNode

- Input: `validated_input` (set by `pre_process`)
- Processing: strip whitespace, lowercase, expand common store ops abbreviations (POS, EOD, SKU) and Japanese retail synonyms (閉店 → クロージング/閉店手順)
- Output: `normalized_query` (string)
- No LLM call — stateless and deterministic

### 3.3 VectorRetrieveNode

- Input: `normalized_query`
- Retrieval config (from `config/agent.yaml`):
  - `collection`: `ret_store_operations_faq_kb`
  - `top_k`: 5
  - `score_threshold`: 0.68
  - `hybrid_search`: true (vector + BM25/keyword)
- Filtering: discard chunks with score < 0.68
- Output:
  - `retrieved_chunks` — list of dicts `{chunk_id, text, score, source_doc, section}`
  - `citations` — deduplicated source references `{doc_title, section}`
- No-match handling: `retrieved_chunks = []`, `citations = []`, status = SUCCESS (not ERROR); `AnswerGenerateNode` handles graceful fallback
- Retriever injectable via constructor for testability; empty-result is structurally valid

### 3.4 AnswerGenerateNode

- Input: `normalized_query`, `retrieved_chunks`, `citations`
- Config: `system_prompt_template: prompts/ret_28_qa.j2`, `temperature: 0.1`, `max_tokens: 1500`
- No-match fallback: if `retrieved_chunks` is empty → return configured fallback message without any LLM/generation attempt:
  ```
  "該当する手順が見つかりませんでした。店長または本部に確認してください。"
  ```
- Answer synthesis: deterministic cited answer from chunk text + `[doc_title §section]` citations; `system_prompt` from config["configurable"]; `# Production wires the real LLM here`
- Output: `answer` (cited string), `confidence_score` (top chunk score or None on no-match)

### 3.5 ResponseValidateNode

- Input: `answer`, `retrieved_chunks`, `citations`
- Checks:
  1. Non-empty check — answer must not be blank
  2. Citation presence — if chunks non-empty, answer must contain at least one `[` citation marker
  3. Grounding heuristic — share of the answer's content tokens that also appear in the retrieved
     chunk text, measured after this pipeline's own framing (the section header and the citation
     markers) is stripped; at least half must be traceable
- On pass: `status` = `AgentStatus.SUCCESS.value`
- On a missing citation marker (soft fail): `status` = `AgentStatus.SUCCESS.value`; a warning note is
  appended to `answer`. The answer is still deliverable, so the run is not ended over it.
- On an empty answer or insufficient grounding (hard fail): `status` = `AgentStatus.ERROR.value` with a
  descriptive message; `answer` preserved (not overwritten). Neither is a value the caller could
  correct by rewording, so the run terminates rather than completing — see §6.3.

---

## 4. State Schema

File: `src/schemas/state.py`  
TypedDict, flat, JSON-serializable only. Inherits from `AgentState`.

| Field | Type | Written by | Description |
|-------|------|-----------|-------------|
| `user_input` | `str` | Caller | Raw query from store staff |
| `validated_input` | `str` | `PreProcessNode` | Validated question |
| `normalized_query` | `str` | `QueryNormalizeNode` | Expanded/cleaned query |
| `retrieved_chunks` | `list` | `VectorRetrieveNode` | Retrieval results (chunk_id, text, score, source_doc, section) |
| `citations` | `list` | `VectorRetrieveNode` | Deduplicated source refs (doc_title, section) |
| `answer` | `str` | `AnswerGenerateNode` | Cited answer string |
| `confidence_score` | `float \| None` | `AnswerGenerateNode` | Top-chunk score or None on no-match |
| `status` | `str` | Multiple nodes | AgentStatus value |
| `error_code` | `str \| None` | `PreProcessNode` | Reason marker for a request that was declined but not refused; routes the run past the inner graph and selects the caller-facing sentence (§6.3) |
| `caller_request` | `dict` | `PreProcessNode` | Validated caller contract (passages + retrieval overrides) |
| `retrieval_config` | `dict` | `DomainWorkflowGraph` | Bounds-checked retrieval settings seeded into the inner graph |
| `trace_id` | `str` | Framework | Audit trail ID |

No credentials in State. No PII (store ops queries are operational, not personal data).

**State Constraints (mandatory):**
- Flat TypedDict only (primitives + JSON-serializable types)
- No JWT, API keys, credentials in State
- InvocationContext via `config["configurable"]` only (not in State)
- No Pydantic models, dataclass, arbitrary Python objects (msgpack incompatible)

---

## 5. Configuration Surface

File: `config/agent.yaml`

```yaml
agent_id: "RET-C2-028"

agent:
  id: "RET-C2-028"
  name: "StoreOperationsFAQAgent"
  version: "0.1.0"
  category: "Cat 2"
  industry: "RET"
  module: "src.graph"
  class: "RetC2028Agent"
  required_trust_level: "VERIFIED_EXTERNAL"

llm:
  system_prompt_template: prompts/ret_28_qa.j2
  temperature: 0.1
  max_tokens: 1500

retrieval:
  vector_store:
    collection: ret_store_operations_faq_kb
  top_k: 5
  score_threshold: 0.68
  hybrid_search: true

security:
  s3_gate_enabled: true
```

---

## 6. Security Design

| Concern | Implementation |
|---------|----------------|
| Caller trust | Every node declares `required_trust_level = VERIFIED_EXTERNAL`. The standalone entry point authenticates a bearer token and establishes that level; an unauthenticated caller is denied before any work runs. |
| Request validation | `PreProcessNode` validates the question and every field of `input_context` against explicit bounds — see §6.1. Rejections name the field, never the value, and end the run in one of the two ways set out in §6.3. |
| Credential-shaped context | The entry point screens `input_context` with the framework's own credential detector and refuses with `400` naming the field. Without this the request fails anyway, but as an opaque first-node error the caller cannot act on. |
| Input scanning | The framework's own input gate masks personal data and screens the question fields; the template's screen runs in addition, so the guarantee does not depend on that gate being active. |
| Output gate | `PostProcessNode` releases the answer or withholds it, clearing every output-bearing field. It uses the framework's credential detector, so its refusal set matches the framework's block set exactly. |
| Audit trail | Every boundary node emits a domain event through `emit_trace_event()`, on every return path. |

### 6.1 Caller request contract (`input_context`)

| Field | Type | Bound |
|-------|------|-------|
| `channel` | string | `[a-z0-9_]{1,32}` |
| `documents` | list | at most 20 entries |
| `documents[].id` | string | `[A-Za-z0-9_.-]{1,64}` |
| `documents[].text` | string | at most 4,000 characters; injection-screened |
| `documents[].source_doc` | string | `[A-Za-z0-9_.-]{1,64}` — rendered into the citation marker, so it is an identifier, not a title |
| `documents[].section` | string | `[A-Za-z0-9_.-]{1,64}`, optional |
| `top_k` | integer | 1–20 |
| `score_threshold` | number | 0.0–1.0, finite |

A caller with no structured channel sends one text field, so the same request
object may travel there instead: a JSON object in `user_input` carrying
`question` plus any of the fields above. It is a transport, not a second
contract — the question is screened as a question and each context field
against the bounds in the table, through the identical validation. Text that is
not a JSON object is read as an ordinary question. Where both channels carry the
same field, `input_context` wins, because it is the declared contract. The
envelope as a whole is bounded at 65,536 characters, separately from the
question's own bound, because the question bound cannot also bound the passages.

Unrecognised fields are rejected rather than ignored: ignoring a key is not
stripping it, and it would travel onward unvalidated.

Every numeric passes a finite-and-bounded parser. `NaN` and the infinities parse
as floats but compare `False` against every bound, so a plain range check accepts
them silently and the pipeline then makes its central decision on a value that is
not a number. They are rejected by name.

### 6.2 Output invariant

This template renders no monetary aggregates and imposes no rounding grid, so
identifiers, quantities and decimals pass through byte-identical. The invariant
it does enforce is provenance: every factual statement in the answer carries a
citation, and the answer's content must be traceable to the retrieved passages
(§3.5). Both directions are pinned by tests.

### 6.3 How a rejection ends the run

A request can end without being carried out in two different ways, and the caller
can act on only one of them. The distinction is deliberate, and the two are never
merged: a refusal presented as a correctable value would invite the caller to
reword and retry something this agent will not do.

**Completes — a value the caller can correct.** `PreProcessNode` writes
`status = AgentStatus.SUCCESS.value` together with an `error_code`, and publishes
no `validated_input`. `StoreOpsQAGraphNode.execute()` sees the code and skips the
inner graph, so nothing retrieves or answers. `PostProcessNode` then maps the code
to one fixed sentence from `src/services/failure_message.py` and writes it to both
`result` and `formatted_output`. The caller receives a completed envelope whose
body names what to correct, so the request can be fixed and sent again on the same
conversation instead of the turn being ended.

| `error_code` | Condition | Caller reads |
|--------------|-----------|--------------|
| `EMPTY_INPUT` | `user_input` missing, not a string, empty, or whitespace only | "No question was received. Send the question you want answered." |
| `QUESTION_TOO_LONG` | question longer than 4,000 characters | "The request is too long. Shorten it and send it again." |
| `INVALID_REQUEST` | null bytes in the question, or any `input_context` contract failure from §6.1 — unrecognised field, wrong type, missing required passage field, an out-of-bound count or length, a non-inert identifier, a non-finite or out-of-range number | "A value in the request could not be accepted. Check it against the documented format." |

A code with no sentence of its own falls back to the generic one rather than
leaking the code itself. `error_code` is a routing marker inside the graph, not
part of the caller's envelope — the graph's output carries `output`, `status`,
`trace_id`, `correlation_id` and `node_history`. The reason reaches the caller as
the sentence. A non-terminal progress event is emitted alongside it, carrying the
generic form of the notice only — a progress event leaves the process outside the
output gate, so nothing request-specific is put on it.

**Terminates — `status = AgentStatus.ERROR.value`, no answer.** These are not
values a caller can reword into acceptance, so no reason code is set and no
caller-facing sentence is produced.

| Condition | Enforced in |
|-----------|-------------|
| Chat-template control tokens or an instruction-override directive in the question | `PreProcessNode` → `screen_injection()` |
| The same content in a caller-supplied passage's `text` | `PreProcessNode` → `bounded_text()` |
| No usable question reaching the inner pipeline | `QueryNormalizeNode`, `VectorRetrieveNode` — upstream-invariant guards. A declined request publishes no `validated_input` and the inner graph is skipped, so these fire only when that invariant is broken. |
| Empty answer, or an answer whose content is not traceable to the retrieved passages | `ResponseValidateNode` (§3.5) |
| No deliverable answer at the output gate, or a credential pattern in the answer | `PostProcessNode` — withholds, clearing every output-bearing field |
| A credential pattern anywhere in a node's returned dict | The framework's `@final` output gate, which raises rather than returning |

Screening raises a distinct exception type rather than a distinct message, so the
refusal stays a refusal however the wording of the message is later edited. The
backbone routes an ERROR status straight to `finalize`, so the output gate does
not run and the envelope carries no answer — which is the intended shape: a
refusal must not be dressed as something a reworded request would get past.

---

## 7. Prompt Design

File: `prompts/ret_28_qa.j2`

Jinja2 system prompt. Key constraints:
- Answer ONLY from retrieved chunks — no hallucination
- Every factual statement must cite source: `[doc_title §section]`
- No-match fallback: `"該当する手順が見つかりませんでした。店長または本部に確認してください。"` when no chunks provided
- Tone: clear, concise, actionable — store staff in time-sensitive situations
- Template variables: `{{ query }}`, `{{ retrieved_chunks }}`, `{{ top_k }}`, `{{ score_threshold }}`

---

## 8. Framework Utilization

### Shared Components Used

- `InvocationContext` (correlation_id, session_id, caller_trust_level)
- `emit_trace_event()` from `shared.utils.audit_logger` (audit trail)
- `SecurityViolationError`
- `_security_gate_input()` / `_security_gate_output()` — both `@final` on `FunctionNode`; domain
  checks attach through the `_extra_security_gate_input()` / `_extra_security_gate_output()` hooks
- `detect_credentials_in_value()` from `framework.security.credential_detector`
- `_security_gate_output()` (content safety — mandatory)

### Composition Pattern

- **Pattern:** GraphNode (inner BaseGraph subgraph)
- **Composition target:** `DomainWorkflowGraph` (inner), called via `StoreOpsQAGraphNode.get_subgraph()`
- **Error propagation strategy:** `"propagate"` (default)

### Import Isolation

- The template does NOT import the underlying platform SDK directly
- Import targets: `framework/`, `shared/` and `src/` only

---

## 9. Test Plan Summary

| TC/PB | Scenario |
|-------|----------|
| TC-01 | Exact-match opening procedure query → cited answer |
| TC-02 | Low-confidence vague query → no chunks pass threshold → fallback message |
| TC-03 | Multi-chunk query → top-5 chunks, multiple citations synthesized |
| TC-04 | Japanese abbreviation normalization (POS/EOD/閉店) → retrieval succeeds |
| TC-05 | Emergency protocol query → correct procedure cited within latency budget |
| PB-01 | Empty/whitespace question → declined before retrieval: the run completes carrying `error_code = EMPTY_INPUT` and publishes no validated question |
| PB-02 | Mock empty vector store → fallback message returned, no LLM call made |
| PB-03 | Prompt injection in the question → refused (the run terminates); the phrase never reaches the answer |
| PB-04 | End-to-end through the HTTP entry point: trust boundary, real cited answer from caller passages, validation rejection, credential-shaped context refusal, envelope containment |

---

## 10. Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| L1 base type | `AgentBaseGraph` | `AutonomousBaseGraph` | `AgentBaseGraph` | Fixed VectorRAG pipeline, no autonomous loop needed |
| Composition pattern | Flat slots (Cat 1 style) | GraphNode + inner BaseGraph (Cat 2 nested) | Cat 2 nested | Template is Cat 2: domain-specific multi-step pipeline; mandatory per F-QuynhDN1 directive |
| Rejection the caller can correct | Terminate with ERROR | Complete with SUCCESS and a reason code | Complete with a reason code | Terminating ends the calling surface's turn and surfaces only a failure type, leaving the reason reachable from the audit trail alone. Completing carries a fixed sentence naming what to correct, so the request can be resent on the same conversation. Content that is refused outright still terminates (§6.3), so the two outcomes stay distinguishable. |
| Output-gate refusal | Raise on violation | Return ERROR and clear the output fields | Clear the fields | The graph resolves output as `formatted_output or result` with no status check, so raising still ships the un-gated answer inside the error envelope |
| Answer synthesis | Live LLM call | Deterministic cited assembly | Deterministic | The installed framework exposes no LLM client; the answer is assembled verbatim from retrieved passages, which also makes grounding checkable. A deployment wires a model at this node. |
| Retrieval relevance bar | One threshold for all retrievers | A bar per retriever | A bar per retriever | Lexical coverage and vector cosine similarity are different scales; one number cannot serve both, and reusing the vector bar for the lexical path rejects every passage |
| Inner graph parent | `BaseGraph` | `AgentBaseGraph` | `BaseGraph` | Fully custom 4-node topology; no forced backbone needed |

---

## 11. References

- `docs/03_test_spec.md` — test specification
- Framework wheel: `agenticstar-agentcore` (the version the pipeline installs)
