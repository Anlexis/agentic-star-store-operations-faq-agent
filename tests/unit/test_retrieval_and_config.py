"""RET-C2-028 — retrieval behaviour and runtime-configuration resolution.

Two properties matter here:

  * a declared setting must REACH the retrieval node. The previous
    implementation read ``state["config"]``, a key the framework never writes,
    so every declared value silently reverted to a built-in default while the
    shipped config file still advertised the operator's numbers;
  * a bad setting must fail the graph rather than degrade it. A NaN threshold
    compares False against every score, so retrieval returns nothing and the
    agent reports "not found" for every question — indistinguishable from an
    empty knowledge base.
"""

import pytest

from framework.schemas.agent_status import AgentStatus
from src.graph.domain_workflow_graph import DEFAULT_RETRIEVAL, DomainWorkflowGraph
from src.nodes.vector_retrieve import VectorRetrieveNode
from src.services.service import (
    CallerCorpusRetriever,
    EmptyRetriever,
    StoreOpsKbService,
    relevance,
)
from src.validation import CallerInputError

CLOSING = "閉店手順: レジ締めを実施し、金庫に売上金を格納してから、防犯システムを起動して施錠する。"
OPENING = "開店手順: 照明を点灯し、POS端末を起動して釣銭を準備する。"
FIRE = "火災時は初期消火を試みず、直ちに避難誘導を行うこと。"

DOCS = [
    {"id": "c1", "text": CLOSING, "source_doc": "store-ops-manual", "section": "3.2"},
    {"id": "c2", "text": OPENING, "source_doc": "store-ops-manual", "section": "3.1"},
    {"id": "c3", "text": FIRE, "source_doc": "emergency-guide", "section": "1.1"},
]


def graph(**retrieval):
    return DomainWorkflowGraph(config={"configurable": {"retrieval": retrieval}} if retrieval else {})


# ── Configuration resolution ──────────────────────────────────────────────────


class TestRetrievalConfigResolution:
    def test_declared_values_are_honoured(self):
        resolved = graph(
            vector_store={"collection": "custom_kb"},
            top_k=9,
            score_threshold=0.42,
            caller_corpus_score_threshold=0.11,
            hybrid_search=False,
        ).resolve_retrieval_config()
        assert resolved == {
            "collection": "custom_kb",
            "top_k": 9,
            "score_threshold": 0.42,
            "caller_corpus_score_threshold": 0.11,
            "hybrid_search": False,
        }

    def test_absent_block_falls_back_to_documented_defaults(self):
        assert graph().resolve_retrieval_config() == DEFAULT_RETRIEVAL

    @pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", float("nan"), float("inf")])
    def test_non_finite_threshold_fails_closed(self, value):
        with pytest.raises(CallerInputError):
            graph(score_threshold=value).resolve_retrieval_config()

    @pytest.mark.parametrize("value", ["NaN", float("inf"), 0, -1, 101, 2.5, True])
    def test_bad_top_k_fails_closed(self, value):
        with pytest.raises(CallerInputError):
            graph(top_k=value).resolve_retrieval_config()

    @pytest.mark.parametrize("value", [-0.01, 1.01, "abc", None])
    def test_out_of_range_threshold_fails_closed(self, value):
        with pytest.raises(CallerInputError):
            graph(score_threshold=value).resolve_retrieval_config()

    @pytest.mark.parametrize("value", ["", "   ", 5, None])
    def test_bad_collection_fails_closed(self, value):
        with pytest.raises(CallerInputError):
            graph(vector_store={"collection": value}).resolve_retrieval_config()

    def test_non_boolean_hybrid_flag_fails_closed(self):
        with pytest.raises(CallerInputError):
            graph(hybrid_search="yes").resolve_retrieval_config()

    def test_compile_rejects_an_unusable_configuration(self):
        """_validate_config runs at compile time, so the failure surfaces before
        any request is served rather than as an empty answer during one."""
        with pytest.raises(CallerInputError):
            graph(score_threshold="NaN").compile()

    def test_settings_are_seeded_into_the_inner_initial_state(self):
        seeded = graph(top_k=7)._extra_initial_state()
        assert seeded["retrieval_config"]["top_k"] == 7


# ── Scoring ───────────────────────────────────────────────────────────────────


class TestRelevance:
    def test_score_is_bounded(self):
        for passage in (CLOSING, OPENING, FIRE):
            assert 0.0 <= relevance("閉店手順を教えてください", passage) <= 1.0

    def test_matching_passage_outranks_a_related_one(self):
        query = "閉店手順を教えてください"
        assert relevance(query, CLOSING) > relevance(query, OPENING)

    def test_unrelated_passage_scores_zero(self):
        assert relevance("閉店手順を教えてください", FIRE) == 0.0

    def test_polite_phrasing_does_not_depress_the_score(self):
        """Grammatical filler carries no retrieval signal. Counting it made a
        politely-phrased question score lower than a terse one against the very
        same passage."""
        terse = relevance("閉店手順", CLOSING)
        polite = relevance("閉店手順を教えてください", CLOSING)
        assert polite >= terse * 0.5

    def test_empty_inputs_score_zero(self):
        assert relevance("", CLOSING) == 0.0
        assert relevance("閉店手順", "") == 0.0


# ── Retriever selection ───────────────────────────────────────────────────────


class TestRetrieverSelection:
    def test_caller_passages_take_precedence(self):
        service = StoreOpsKbService(external=EmptyRetriever())
        assert isinstance(service.retriever_for(DOCS), CallerCorpusRetriever)

    def test_external_used_when_caller_sent_nothing(self):
        external = EmptyRetriever()
        assert StoreOpsKbService(external=external).retriever_for(None) is external

    def test_empty_baseline_when_nothing_is_wired(self):
        assert isinstance(StoreOpsKbService().retriever_for(None), EmptyRetriever)


# ── Node behaviour ────────────────────────────────────────────────────────────


def node_state(**kwargs):
    state = {
        "normalized_query": "閉店手順を教えてください",
        "retrieval_config": dict(DEFAULT_RETRIEVAL),
        "caller_request": {"documents": DOCS},
        "error_log": [],
    }
    state.update(kwargs)
    return state


class TestVectorRetrieveNode:
    def test_returns_the_matching_passage_with_a_citation(self):
        result = VectorRetrieveNode().execute(node_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["retrieved_chunks"]
        assert result["retrieved_chunks"][0]["chunk_id"] == "c1"
        assert {"doc_title": "store-ops-manual", "section": "3.2"} in result["citations"]

    def test_declared_setting_reaches_the_node(self):
        """Raising the bar above what any passage can score must change the
        outcome. If the setting were dead this assertion would not move."""
        strict = dict(DEFAULT_RETRIEVAL, caller_corpus_score_threshold=0.99)
        result = VectorRetrieveNode().execute(node_state(retrieval_config=strict))
        assert result["retrieved_chunks"] == []

    def test_caller_top_k_narrows_the_result(self):
        state = node_state(caller_request={"documents": DOCS, "top_k": 1})
        assert len(VectorRetrieveNode().execute(state)["retrieved_chunks"]) == 1

    def test_empty_query_is_an_error(self):
        result = VectorRetrieveNode().execute(node_state(normalized_query="", validated_input=""))
        assert result["status"] == AgentStatus.ERROR.value

    def test_no_caller_passages_falls_back_to_the_baseline(self):
        result = VectorRetrieveNode().execute(node_state(caller_request={}))
        assert result["retrieved_chunks"] == []

    def test_lexical_and_vector_bars_are_separate(self):
        """One number cannot serve both scales: reusing the vector bar for
        lexical coverage rejects every passage and reports "not found" for every
        question."""
        settings = dict(DEFAULT_RETRIEVAL, score_threshold=0.68, caller_corpus_score_threshold=0.25)
        result = VectorRetrieveNode().execute(node_state(retrieval_config=settings))
        assert result["retrieved_chunks"], "lexical path must not be gated by the vector bar"

    @pytest.mark.parametrize("score", [float("nan"), float("inf"), "abc", None, True])
    def test_unusable_scores_from_an_external_retriever_do_not_pass_the_bar(self, score):
        """NaN compares False against the threshold, but so does an unusable
        score treated as 0.0 — the difference is that 0.0 cannot pass a bar of
        0.0 by accident either."""

        class Odd:
            def retrieve(self, query, collection, top_k, score_threshold, hybrid):
                return [{"chunk_id": "x", "text": CLOSING, "score": score, "source_doc": "d", "section": "1"}]

        state = node_state(caller_request={})
        result = VectorRetrieveNode(retriever=Odd()).execute(state)
        assert result["retrieved_chunks"] == []
