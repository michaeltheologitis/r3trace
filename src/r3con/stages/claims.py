"""After reasoning: split the answer into its claims, and link each to the evidence
that supports it.

An answer alone does not say which of its sentences rests on what. At the
``"sentence"`` evidence level, one call, to reasoning's model with its params, is shown
the question, the answer, and the quotes on the answer's relevant trail, each under an
id, and copies out the answer's claims, each sentence, clause, list item or table row,
which are then found in the answer. Then one call per claim found, in parallel, is
shown the question, the answer, that claim, and the same quotes, each with its
document's name, and names those that directly support the claim and those that are
only background to it: the claim keeps the first, and their documents. At
``"document"``, the one call copies out the claims with the ids of every document whose
round-1 relevance summary, the one written from that document alone, supports each.

None of it fails the run: with no evidence to show, a claims request over the model's
line, or a claims call that fails or is cancelled, there are no claims; a linking
request over the line, or a linking call that fails or is cancelled, leaves its claim
no quotes and no documents.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping, Sequence
from operator import itemgetter
from typing import Any, NamedTuple, cast

import jinja2
from pydantic import BaseModel, create_model

from r3con.config import EvidenceLevel, PromptStage
from r3con.logging_setup import get_logger
from r3con.parallel import parallel_map
from r3con.prompts import load_prompt
from r3con.provenance import Claim, NamedQuote, QuotePlace, RelevantTrail
from r3con.quotes import locate_quotes
from r3con.runs import StageRun
from r3con.runtime.llm import error_line, schema_as_sent, send_until_valid
from r3con.splitting import Splits
from r3con.stages.citing import cited, ids_field, lenient_list
from r3con.stages.relevance import RelevantContext, SnippetRecord

_log = get_logger("claims")

_KIND = "claims"
_LINKING = "linking"
_REWRITE = "Extract the same answer's claims again"
_LINK_AGAIN = "Name the quotes that support the same claim again"
_USER_MESSAGE = jinja2.Template(
    "<question>\n{{ question }}\n</question>\n"
    "<answer>\n{{ answer }}\n</answer>\n"
    "<{{ tag }}>\n"
    "{% for id, piece in evidence.items() %}"
    "### {{ heading }}: {{ id }}\n{{ piece.text }}\n"
    "{% endfor %}"
    "</{{ tag }}>"
)
_LINKING_MESSAGE = jinja2.Template(
    "<question>\n{{ question }}\n</question>\n"
    "<answer>\n{{ answer }}\n</answer>\n"
    "<claim>\n{{ claim }}\n</claim>\n"
    "<quotes>\n"
    "{% for id, quote in quotes.items() %}"
    "### Quote ID: {{ id }}\n(from: {{ quote.name }})\n{{ quote.text }}\n"
    "{% endfor %}"
    "</quotes>"
)


class _Piece(NamedTuple):
    """A piece of evidence shown: its text, its document's id and name, and a quote's
    place."""

    text: str
    document: str
    name: str
    quote: QuotePlace | None


class _Level(NamedTuple):
    """What varies by evidence level: the evidence, how it is shown, and the claims
    call's reply, given the evidence's ids."""

    evidence: Callable[
        [RelevantTrail, RelevantContext, Sequence[str]], dict[str, _Piece]
    ]
    tag: str
    heading: str
    reply: Callable[[Sequence[str]], type[BaseModel]]


def find_claims(
    *,
    question: str,
    answer: str,
    relevant_trail: RelevantTrail,
    relevance: RelevantContext,
    document_ids: Sequence[str],
    evidence_level: EvidenceLevel,
    prompts: Mapping[PromptStage, str],
    model: str,
    max_attempts: int,
    splits: Splits,
    workers: int,
    run: StageRun | None = None,
    **llm_kwargs: Any,
) -> list[Claim]:
    """The claims ``answer`` makes, each with the evidence on its ``relevant_trail``
    that supports it, in the answer's order, as the module says, at most ``workers``
    linking calls under way at once. It never fails the run."""
    level = _LEVELS[evidence_level]
    evidence = level.evidence(relevant_trail, relevance, document_ids)
    if not evidence:
        _log.info("claims: the answer's relevant trail holds no evidence, so no call")
        return []
    send = functools.partial(
        _reply,
        splits=splits,
        max_attempts=max_attempts,
        model=model,
        run=run,
        **llm_kwargs,
    )
    found = send(
        load_prompt("claims", version=prompts["claims"], evidence_level=evidence_level),
        _USER_MESSAGE.render(
            question=question,
            answer=answer,
            tag=level.tag,
            heading=level.heading,
            evidence=evidence,
        ),
        level.reply(list(evidence)),
        kind=_KIND,
        rewrite=_REWRITE,
        warning="claims: none, as their",
    )
    if found is None:
        return []
    if evidence_level == "document":
        return [
            _supported(answer, start, end, claim["document_ids"], evidence, _KIND)
            for start, end, claim in _placed(found["claims"], answer)
        ]
    placed = _placed([{"text": text} for text in found["claims"]], answer)
    system_prompt = load_prompt(_LINKING, version=prompts[_LINKING])
    ids = list(evidence)
    schema = create_model("Linking", direct=ids_field(ids), context=ids_field(ids))

    def link(k: int, span: tuple[int, int, Mapping[str, Any]]) -> Claim:
        start, end, _ = span
        kind = f"{_LINKING}-c{k + 1}"
        user_message = _LINKING_MESSAGE.render(
            question=question, answer=answer, claim=answer[start:end], quotes=evidence
        )
        linked = send(
            system_prompt,
            user_message,
            schema,
            kind=kind,
            rewrite=_LINK_AGAIN,
            warning=f"claims: claim {k + 1} keeps no quotes, as its linking",
        )
        direct = linked["direct"] if linked is not None else []
        return _supported(answer, start, end, direct, evidence, kind)

    return parallel_map(link, placed, max_workers=workers)


def _reply(
    system_prompt: str,
    user_message: str,
    schema: type[BaseModel],
    *,
    kind: str,
    rewrite: str,
    warning: str,
    splits: Splits,
    **llm_call: Any,
) -> dict[str, Any] | None:
    """The reply to ``user_message`` beside ``system_prompt``
    (:func:`~r3con.runtime.llm.send_until_valid`), as a dict; or ``None`` when the
    request is over ``splits``' line, or its call fails or is cancelled, each logged as
    a warning that starts with ``warning``."""
    schema_json = json.dumps(schema_as_sent(schema))
    if estimate := splits.over_line(system_prompt, user_message, schema_json):
        _log.warning(
            "%s request would take about %s tokens, over the %s-token line",
            warning,
            f"{estimate:,}",
            f"{splits.line:,}",
        )
        return None
    try:
        reply = send_until_valid(
            system_prompt,
            user_message,
            schema=schema,
            rewrite=rewrite,
            kind=kind,
            **llm_call,
        )
    # A cancel included: once reasoning has answered, the run finishes with it.
    except Exception as error:  # noqa: BLE001
        _log.warning("%s call failed: %s", warning, error_line(error))
        return None
    return cast(BaseModel, reply).model_dump()


def _quotes(
    trail: RelevantTrail, _relevance: RelevantContext, _ids: Sequence[str]
) -> dict[str, _Piece]:
    """The trail's quotes, as ``q1``, ``q2``, … in order."""
    return {
        f"q{n}": _Piece(
            _quote_text(quote),
            quote["document"],
            quote["name"],
            QuotePlace(
                document=quote["document"], start=quote["start"], end=quote["end"]
            ),
        )
        for n, quote in enumerate(trail["quotes"], start=1)
    }


def _quote_text(quote: NamedQuote) -> str:
    """The quote in the document's words, or the note's when elided past some."""
    if quote["match"] == "elided" or quote["document_text"] is None:
        return quote["text"]
    return quote["document_text"]


def _summaries(
    trail: RelevantTrail, relevance: RelevantContext, document_ids: Sequence[str]
) -> dict[str, _Piece]:
    """Each trail document's summaries (:func:`_shown_summaries`), in order."""
    index = {id_: doc for doc, id_ in enumerate(document_ids)}
    pieces: dict[str, _Piece] = {}
    for document in trail["documents"]:
        id_, name = document["document"], document["name"]
        notes = _shown_summaries(relevance.records, index[id_])
        pieces |= {part: _Piece(note, id_, name, None) for part, note in notes.items()}
    return pieces


def _shown_summaries(
    records: Sequence[Sequence[Sequence[SnippetRecord]]], doc: int
) -> dict[str, str]:
    """Document ``doc``'s non-empty round-1 summaries by id, else its latest round's."""
    for round_ in [*records[:1], *reversed(records[1:])]:
        notes = {r.id: r.summary.strip() for r in round_[doc] if r.summary.strip()}
        if notes:
            return notes
    return {}


def _claim_texts(_ids: Sequence[str]) -> type[BaseModel]:
    """A claims reply that lists the claims' texts alone."""
    return create_model("Claims", claims=lenient_list(str))


def _claims_with_documents(ids: Sequence[str]) -> type[BaseModel]:
    """A claims reply that lists each claim's text with the ids of ``ids`` it cites."""
    claim = create_model("Claim", text=(str, ...), document_ids=ids_field(ids))
    return create_model("Claims", claims=lenient_list(claim))


_LEVELS: dict[EvidenceLevel, _Level] = {
    "sentence": _Level(_quotes, "quotes", "Quote ID", _claim_texts),
    "document": _Level(_summaries, "summaries", "Document ID", _claims_with_documents),
}


def _supported(
    answer: str,
    start: int,
    end: int,
    ids: Sequence[str],
    evidence: Mapping[str, _Piece],
    kind: str,
) -> Claim:
    """The claim ``answer`` makes from ``start`` to ``end``, with the evidence of
    ``ids`` in the order it was shown, less the ids it was not shown, which
    :func:`~r3con.stages.citing.cited` drops, warning as ``kind``."""
    kept = frozenset(cited(ids, evidence, kind))
    pieces = [piece for id_, piece in evidence.items() if id_ in kept]
    return Claim(
        text=answer[start:end],
        start=start,
        end=end,
        documents=list(dict.fromkeys(piece.document for piece in pieces)),
        quotes=[piece.quote for piece in pieces if piece.quote is not None],
    )


def _placed(
    claims: Sequence[Mapping[str, Any]], answer: str
) -> list[tuple[int, int, Mapping[str, Any]]]:
    """Each claim's span in ``answer``, sorted, or the claim dropped with a warning."""
    placed: list[tuple[int, int, Mapping[str, Any]]] = []
    misses: list[tuple[str, Mapping[str, Any]]] = []
    cursor = 0
    # Verbatim from the last one's end, so a sentence said twice lands where it is.
    for claim in claims:
        text = claim["text"].strip()
        start = answer.find(text, cursor) if text else -1
        if start >= 0:
            cursor = start + len(text)
            placed.append((start, cursor, claim))
        elif text:
            misses.append((text, claim))
        else:
            _log.warning("claims: dropped a blank claim")
    # The rest in one call, which normalizes the answer once.
    located = locate_quotes([text for text, _ in misses], answer)
    for (text, claim), quote in zip(misses, located, strict=True):
        start, end = quote["start"], quote["end"]
        if start is None or end is None:
            _log.warning("claims: dropped a claim not found in the answer: %r", text)
        elif any(start < other_end and other < end for other, other_end, _ in placed):
            _log.warning("claims: dropped a claim that overlaps another: %r", text)
        else:
            placed.append((start, end, claim))
    return sorted(placed, key=itemgetter(0))
