"""AgentCore Platform v1.0 — RET-C2-028 caller-input validation helpers.

Every value that arrives from the caller passes through this module before any
domain code reads it. The rules are deliberately narrow and fail CLOSED:

  * numbers  — must parse as a real, finite number inside an explicit range.
               ``float("nan")`` and ``float("inf")`` parse successfully but
               compare False against every bound, so a naive ``lo <= v <= hi``
               check silently accepts them and the pipeline then makes its
               central decision on a value that is not a number. Reject them by
               name instead of by comparison.
  * identifiers — caller strings that are rendered back into the answer (the
               citation labels) are restricted to an inert alphabet, so nothing
               a caller supplies can alter the structure of the rendered text.
  * free text — bounded in length and screened for prompt-injection content,
               both before and after markup is stripped.

Errors name the FIELD and never echo the rejected value.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Final

# ── Identifiers rendered into the answer ──────────────────────────────────────
# Citation labels are identifiers, not titles: a document's human-readable name
# belongs in the passage text, which is rendered as content. Restricting the
# label to this alphabet means no caller value can introduce a bracket, a
# newline or a directive into the citation marker "[doc §section]" that the
# answer is assembled from. Case, "." and "-" are admitted because real document
# and section identifiers use them ("ops-manual.v3", "3.2.1"); whitespace,
# brackets and every non-ASCII character are not.
_INERT_ID_RE: Final = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")

# Channel identifiers are internal routing labels, so they stay lowercase-inert.
_CHANNEL_RE: Final = re.compile(r"^[a-z0-9_]{1,32}$")

# ── Prompt-injection screening ────────────────────────────────────────────────
# Chat-template CONTROL TOKENS are screened as a class, not as individual
# phrases: the delimiters themselves are what a model interprets, so a payload
# needs no recognisable English directive to take effect.
_CONTROL_TOKEN_RE: Final = re.compile(
    r"<\|[^|>]{0,64}\|>"  # <|im_start|>, <|system|>, <|endoftext|>, ...
    r"|\[/?INST\]"  # [INST] / [/INST]
    r"|<</?SYS>>"  # <<SYS>> / <</SYS>>
    r"|<\|?(?:im_start|im_end|endoftext|system)\|?>",
    re.IGNORECASE,
)

# Directive phrases are anchored to an instruction verb followed by an
# instruction object, so ordinary retail prose ("この手順を無視して先に進む場合"
# / "disregard damaged stock") does not trip them.
_DIRECTIVE_RE: Final = re.compile(
    r"(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+|the\s+|your\s+|previous\s+|prior\s+|above\s+)*"
    r"(?:previous\s+|prior\s+|above\s+|earlier\s+)*"
    r"(?:instruction|instructions|rule|rules|prompt|prompts|direction|directions|context)\b"
    r"|you\s+are\s+now\s+(?:a|an)\b"
    r"|(?:reveal|print|output|repeat|show)\s+(?:me\s+)?(?:your\s+|the\s+)"
    r"(?:system\s+prompt|initial\s+prompt|instructions|hidden\s+rules)\b",
    re.IGNORECASE,
)

# Markup that a naive sanitiser would strip. We do not strip it as a defence —
# stripping a control token turns a detectable attack into undetectable plain
# text — we strip it only to build a SECOND view of the string, so a directive
# split across tags ("ig<b>nore all instructions") is caught once re-assembled.
_MARKUP_RE: Final = re.compile(r"<[^<>]{0,200}>")


class CallerInputError(ValueError):
    """Raised when a caller-supplied value fails its contract.

    The message names the field and the rule. It never contains the value.
    """


class InjectionRefused(CallerInputError):
    """Raised when a caller-supplied value carries prompt-injection content.

    A SEPARATE TYPE, not a separate message. Every other contract failure here
    describes a value the caller can correct and resend; this one does not, and
    the two must be distinguishable by something stronger than the wording of a
    message — wording gets reworded, and the distinction would be lost silently
    the next time one of these strings is edited.

    Kept as a subclass so that every existing ``except CallerInputError`` still
    catches it: a handler that does not know about this type keeps its old,
    stricter behaviour rather than letting the refusal escape.
    """


def _describe(field: str, rule: str) -> str:
    return f"{field}: {rule}"


# ── Numbers ───────────────────────────────────────────────────────────────────


def finite_in_range(
    value: Any,
    *,
    field: str,
    lo: float,
    hi: float,
    integer: bool = False,
) -> float:
    """Return *value* as a finite number within [lo, hi], or raise.

    Rejects, by name rather than by comparison: booleans (``True`` is an ``int``
    in Python and would otherwise pass as 1), non-numeric types and strings,
    NaN and ±Infinity, and out-of-range magnitudes.
    """
    if isinstance(value, bool):
        raise CallerInputError(_describe(field, "must be a number, not a boolean"))

    if isinstance(value, (int, float)):
        candidate = float(value)
    elif isinstance(value, str):
        try:
            candidate = float(value.strip())
        except (TypeError, ValueError):
            raise CallerInputError(_describe(field, "must be a number")) from None
    else:
        raise CallerInputError(_describe(field, "must be a number"))

    # NaN and ±Infinity parse fine and compare False against every bound.
    if not math.isfinite(candidate):
        raise CallerInputError(_describe(field, "must be a finite number (NaN and Infinity are rejected)"))

    if not (lo <= candidate <= hi):
        raise CallerInputError(_describe(field, f"must be within [{lo}, {hi}]"))

    if integer:
        if candidate != int(candidate):
            raise CallerInputError(_describe(field, "must be a whole number"))
        return float(int(candidate))

    return candidate


# ── Identifiers and free text ─────────────────────────────────────────────────


def inert_identifier(value: Any, *, field: str) -> str:
    """Return *value* as an inert identifier, or raise."""
    if not isinstance(value, str):
        raise CallerInputError(_describe(field, "must be a string"))
    if not _INERT_ID_RE.match(value):
        raise CallerInputError(
            _describe(field, "must be 1-64 characters from [A-Za-z0-9_.-] (no spaces or punctuation)")
        )
    return value


def channel_identifier(value: Any, *, field: str) -> str:
    """Return *value* as an inert lowercase channel identifier, or raise."""
    if not isinstance(value, str):
        raise CallerInputError(_describe(field, "must be a string"))
    if not _CHANNEL_RE.match(value):
        raise CallerInputError(_describe(field, "must be 1-32 characters from [a-z0-9_]"))
    return value


def screen_injection(value: str, *, field: str) -> None:
    """Raise when *value* carries prompt-injection content.

    The string is screened in two views. The RAW view catches control tokens
    before any markup strip could remove them; the STRIPPED view catches
    directives that were split across markup to evade a single-pass scan.
    Unicode is normalised first so that compatibility forms cannot smuggle a
    delimiter past the pattern.
    """
    raw = unicodedata.normalize("NFKC", value)
    if _CONTROL_TOKEN_RE.search(raw):
        raise InjectionRefused(_describe(field, "contains chat-template control tokens"))
    if _DIRECTIVE_RE.search(raw):
        raise InjectionRefused(_describe(field, "contains prompt-injection directives"))

    stripped = _MARKUP_RE.sub("", raw)
    if stripped != raw:
        if _CONTROL_TOKEN_RE.search(stripped):
            raise InjectionRefused(_describe(field, "contains chat-template control tokens"))
        if _DIRECTIVE_RE.search(stripped):
            raise InjectionRefused(_describe(field, "contains prompt-injection directives"))


def bounded_text(value: Any, *, field: str, max_len: int) -> str:
    """Return *value* as bounded, injection-screened free text, or raise."""
    if not isinstance(value, str):
        raise CallerInputError(_describe(field, "must be a string"))
    text = value.strip()
    if not text:
        raise CallerInputError(_describe(field, "must not be empty"))
    if len(text) > max_len:
        raise CallerInputError(_describe(field, f"must be at most {max_len} characters"))
    if "\x00" in text:
        raise CallerInputError(_describe(field, "must be plain text (null bytes are rejected)"))
    screen_injection(text, field=field)
    return text
