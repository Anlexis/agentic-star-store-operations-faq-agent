"""AgentCore Platform v1.0 — RET-C2-028 QueryNormalizeNode.

Stateless, deterministic query normalization for store operations FAQ.
No LLM call — pure string processing.

Field names follow docs/02_design.md §3.2 and src/schemas/state.py.
Reads:  validated_input (set by PreProcessNode)
Writes: normalized_query
"""

from typing import Any, ClassVar, Dict, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel

# Audit trail (free-function API; ships in the SDK wheel only).
# Never use self.emit_trace_event — AttributeError.
from shared.utils.audit_logger import emit_trace_event

# Common store operations abbreviations → expansion
# Applied case-insensitively on the lowercased query.
_ABBREVIATIONS: dict[str, str] = {
    "pos": "POS端末（レジ）",
    "eod": "日次締め処理（EOD）",
    "sku": "商品コード（SKU）",
    "bod": "開店処理（BOD）",
    "kpi": "重要業績指標（KPI）",
    "sla": "サービスレベル合意（SLA）",
    "rfid": "RFIDタグ",
}

# Japanese retail synonym expansion — maps known terms to canonical + alternate form
# so downstream retrieval finds more relevant chunks.
_JP_SYNONYMS: dict[str, str] = {
    "閉店": "閉店 クロージング 閉店手順",
    "開店": "開店 オープニング 開店手順",
    "締め": "締め処理 日次締め EOD",
    "棚卸": "棚卸し 在庫カウント 棚卸手順",
    "発注": "発注 注文 オーダー",
    "クレーム": "クレーム 苦情対応 顧客クレーム",
    "緊急": "緊急 緊急対応 緊急手順",
    "火災": "火災 防火 避難手順",
    "停電": "停電 電源喪失 停電時対応",
}


class QueryNormalizeNode(FunctionNode):
    """Normalize and expand the validated store-ops query.

    Steps:
    1. Strip whitespace; reject if empty after strip.
    2. Expand abbreviations (POS, EOD, SKU, …) in lowercase pass.
    3. Expand Japanese retail synonyms (閉店 → クロージング, etc.).
    4. Write normalized_query to State.

    Stateless and deterministic — no LLM, no external call.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Prefer validated_input (written by PreProcessNode).
        # Fall back to user_input for unit-test convenience.
        raw = state.get("validated_input") or state.get("user_input", "")

        if not isinstance(raw, str) or not raw.strip():
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["QueryNormalizeNode: input is empty or missing"],
            }

        query = raw.strip()

        # Step 2: abbreviation expansion (case-insensitive token replace).
        query_lower = query.lower()
        for abbr, expansion in _ABBREVIATIONS.items():
            # Replace whole-word abbreviation (surrounded by space/start/end).
            # Simple approach: check presence and append expansion note.
            if f" {abbr} " in f" {query_lower} ":
                # Append expansion in parentheses if not already present.
                if expansion.split("（")[0] not in query:
                    query = query + f"（{expansion}）"

        # Step 3: Japanese synonym expansion — append alternate forms.
        expanded_terms: list[str] = []
        for term, synonyms in _JP_SYNONYMS.items():
            if term in query and synonyms not in query:
                # Add the synonym string as additional search hints.
                syn_parts = synonyms.split()
                new_parts = [s for s in syn_parts if s != term and s not in query]
                if new_parts:
                    expanded_terms.extend(new_parts)

        if expanded_terms:
            query = query + " " + " ".join(expanded_terms)

        normalized = query.strip()

        # Audit: record the normalization outcome (non-sensitive counts only).
        emit_trace_event("query_normalized", {"query_len": len(normalized)}, state)

        return {
            "normalized_query": normalized,
            "status": AgentStatus.SUCCESS.value,
        }
