"""AgentCore Platform v1.0 — RET-C2-028 store-operations knowledge-base service.

Service layer: retrieval against the store-operations knowledge base. Holds no
business logic, no routing and no credentials — nodes call this, and this owns
the retrieval mechanics.

Two retrievers ship:

``CallerCorpusRetriever``
    Scores the passages the caller supplied on this request. This is the path
    that does real work: the caller sends its own operations passages, the
    service ranks them against the query, and the answer is assembled from the
    passages that clear the relevance threshold, with a citation for each.

``EmptyRetriever``
    The baseline used when the caller supplied no passages and no external
    vector store is wired. It returns nothing, which makes the pipeline emit the
    documented "not found" message rather than inventing a procedure.

A production deployment substitutes a vector-store client implementing
``Retriever``; the node accepts any object satisfying that protocol.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Protocol

# Character classes used for the lexical overlap score. Japanese store-operations
# text is not whitespace-delimited, so scoring uses two complementary views: word
# tokens for ASCII/alphanumeric terms, and character bigrams for CJK runs.
_WORD_RE = re.compile(r"[0-9A-Za-z_]+")
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿ｦ-ﾟ]+")
# Bigrams made entirely of hiragana are grammatical filler ("を教", "してく"):
# they occur in almost every sentence and carry no retrieval signal, so counting
# them makes a long, politely-phrased question score lower than a terse one
# against the very same passage. Dropped from both sides of the comparison.
_HIRAGANA_ONLY_RE = re.compile(r"^[ぁ-ゟー]+$")


class Retriever(Protocol):
    """Minimal interface for a store-operations passage retriever."""

    def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int,
        score_threshold: float,
        hybrid: bool,
    ) -> list[dict[str, Any]]:
        """Return ranked passage dicts.

        Each dict carries: ``chunk_id``, ``text``, ``score``, ``source_doc``,
        ``section``.
        """
        ...


def content_tokens(text: str) -> set[str]:
    """Return the comparable content-token set for *text*.

    Words are lowercased; CJK runs contribute overlapping character bigrams so
    that "閉店手順" and "閉店の手順" share evidence. Single characters are
    dropped — they match almost anything and would inflate every score.
    """
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens = {w for w in _WORD_RE.findall(normalized) if len(w) > 1}
    for run in _CJK_RE.findall(normalized):
        if len(run) == 1:
            continue
        for i in range(len(run) - 1):
            bigram = run[i : i + 2]
            if _HIRAGANA_ONLY_RE.match(bigram):
                continue
            tokens.add(bigram)
    return tokens


def relevance(query: str, passage: str) -> float:
    """Return a 0.0–1.0 lexical relevance score for *passage* against *query*.

    The score is the share of the query's content tokens that the passage
    covers, so it is bounded by construction and independent of passage length:
    a long passage cannot outrank a short one merely by containing more words.

    This is LEXICAL matching. A passage that answers the question in different
    words scores low and is not retrieved, and the pipeline then reports no
    match rather than guessing — the safe direction for an operations
    procedure. Semantic matching is the job of a vector-store retriever, which a
    deployment supplies through the ``Retriever`` protocol.
    """
    query_tokens = content_tokens(query)
    if not query_tokens:
        return 0.0
    passage_tokens = content_tokens(passage)
    if not passage_tokens:
        return 0.0
    return len(query_tokens & passage_tokens) / len(query_tokens)


class EmptyRetriever:
    """Baseline retriever — matches nothing.

    Used when the caller supplied no passages and no vector store is wired. The
    pipeline treats an empty result as the documented no-match case.
    """

    def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int,
        score_threshold: float,
        hybrid: bool,
    ) -> list[dict[str, Any]]:
        return []


class CallerCorpusRetriever:
    """Ranks the validated passages supplied with the current request.

    The passages have already passed their field contract (identifier alphabet,
    length bounds, injection screening) in the pre-process node, so this class
    only ranks — it never re-admits raw caller input.
    """

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self._documents = documents

    def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int,
        score_threshold: float,
        hybrid: bool,
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for index, document in enumerate(self._documents):
            text = str(document.get("text", ""))
            if not text:
                continue
            score = relevance(query, text)
            if score < score_threshold:
                continue
            scored.append(
                {
                    "chunk_id": str(document.get("id", f"chunk_{index}")),
                    "text": text,
                    "score": score,
                    "source_doc": str(document.get("source_doc", "")),
                    "section": str(document.get("section", "")),
                }
            )

        scored.sort(key=lambda chunk: (-float(chunk["score"]), str(chunk["chunk_id"])))
        return scored[:top_k]


class StoreOpsKbService:
    """Domain entry point for store-operations passage retrieval.

    Selects the retriever for the request: the caller's own passages when the
    request carried any, otherwise the configured external retriever, otherwise
    the empty baseline.
    """

    def __init__(self, external: Retriever | None = None) -> None:
        self._external = external

    def retriever_for(self, documents: list[dict[str, Any]] | None) -> Retriever:
        if documents:
            return CallerCorpusRetriever(documents)
        if self._external is not None:
            return self._external
        return EmptyRetriever()
