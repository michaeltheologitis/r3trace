import json
import logging
import shutil
import threading
from pathlib import Path

import pytest

from r3con import Answer, Cancelled, ask, resume, run, settings, trace
from r3con.config import EVIDENCE_LEVELS
from r3con.provenance import RULES
from tests.fakes import (
    ANSWERING,
    TWO_LINE_MEMOS,
    bad_key,
    citing,
    claims_reply,
    evidence,
    first_line,
    in_json,
    is_later_round,
    linked_claim,
    linking_reply,
    quoting_a_line_a_round,
    shown_evidence,
    stage_of,
    whole_trail,
)

QUESTION = "Who?"
HALLORAN = "The Halloran memo logs two sites."
MERROW = "The Merrow memo logs two sites."
ANSWER = f"{HALLORAN} {MERROW}"
SENTENCE_SYSTEM = """You will be given a question and an answer that is supported by a collection of quotes. Your job is to extract the exact claims made in the answer, word for word (each sentence, clause, list item or table row; split a sentence that makes several claims into its parts, each word for word), so that each claim can be checked against the quotes on its own.
Return exactly one JSON object with the key `claims`, a list of the claims' texts. Output only that JSON.
Preserve the original wording of each claim exactly as it appears in the answer. Include every claim from the answer once, in the same order it appears. Do not list quote IDs.
## Example
<question>
Which bakery sold more loaves in March, and by how much?
</question>
<answer>
Rye & Co. sold more loaves in March: 1,240 against Hearth's 980, a difference of 260. Neither report gives prices.
</answer>
<quotes>
### Quote ID: q1
Loaves sold in March: 1,240.
### Quote ID: q2
All flour is milled on site.
### Quote ID: q3
Loaves sold in March: 980.
### Quote ID: q4
The shop is open seven days a week.
</quotes>
Reply: {"claims": ["Rye & Co. sold more loaves in March: 1,240 against Hearth's 980, a difference of 260.", "Neither report gives prices."]}"""
DOCUMENT_SYSTEM = """You will be given a question and an answer that is supported by a collection of document summaries. Your job is to extract the exact claims made in the answer, word for word (each sentence, clause, list item or table row; split a sentence that makes several claims into its parts, each word for word), along with the IDs of every document whose summary supports them or any part of them.
Return exactly one JSON object with the key `claims`, where each item has `text` and `document_ids`. Output only that JSON.
Preserve the original wording of each claim exactly as it appears in the answer. Include every claim from the answer once, in the same order it appears. Base document_ids only on the provided summaries: cite every document whose summary supports the claim or any part of it, and use an empty array only when no summary supports any part of it. Do not guess or infer unsupported document IDs.
## Example
<question>
Which trial sites reported adverse events?
</question>
<answer>
- Site A: two mild headaches.
- Site C: one case of nausea.
- Site B reported none.
</answer>
<summaries>
### Document ID: 3f9a
Site A's safety log for the trial: two adverse events, both mild headaches.
### Document ID: 7c21
Site B's monthly report: enrollment figures, and a note that no adverse events occurred.
### Document ID: b604
Site C's safety summary: one adverse event, a case of nausea, resolved the same day.
</summaries>
Reply: {"claims": [{"text": "Site A: two mild headaches.", "document_ids": ["3f9a"]}, {"text": "Site C: one case of nausea.", "document_ids": ["b604"]}, {"text": "Site B reported none.", "document_ids": ["7c21"]}]}"""
SYSTEMS = {"sentence": SENTENCE_SYSTEM, "document": DOCUMENT_SYSTEM}
LINKING_SYSTEM = """You will be given a question, an answer to it, one claim taken from that answer, and numbered quotes from source documents. Decide which quotes directly support the claim.

A quote directly supports the claim when it states a fact the claim asserts (a name, number, date, event, or relation in the claim), on its own or together with other quotes you select. A quote that is only on the same topic, gives background, or supports a different part of the answer does not directly support this claim. If the claim only frames the answer (for example "Yes." or a summary of other claims), select the quotes that state the facts it relies on, if any.

Return exactly one JSON object with the keys `direct` and `context`: `direct` lists the IDs of the quotes that directly support the claim; `context` lists the IDs of quotes that are useful background for the claim but do not state what it asserts. Use an empty array when there are none. Output only that JSON."""
QUESTION_AND_ANSWER = """<question>
Who?
</question>
<answer>
The Halloran memo logs two sites. The Merrow memo logs two sites.
</answer>
"""
# The trail of an ANSWER over TWO_LINE_MEMOS quotes each memo's line in its order.
QUOTES = """<quotes>
### Quote ID: q1
Halloran memo
### Quote ID: q2
The Halloran memo logs two sites.
### Quote ID: q3
Merrow memo
### Quote ID: q4
The Merrow memo logs two sites.
### Quote ID: q5
The Calder memo logs two sites.
</quotes>"""
QUOTE_IDS = ["q1", "q2", "q3", "q4", "q5"]
# The same quotes as a linking request shows them, each with its document's name.
NAMED_QUOTES = """<quotes>
### Quote ID: q1
(from: q3/halloran.txt)
Halloran memo
### Quote ID: q2
(from: q3/halloran.txt)
The Halloran memo logs two sites.
### Quote ID: q3
(from: q3/merrow.txt)
Merrow memo
### Quote ID: q4
(from: q3/merrow.txt)
The Merrow memo logs two sites.
### Quote ID: q5
(from: calder.pdf)
The Calder memo logs two sites.
</quotes>"""
CANCELLED = "Cancelled: cancel was set, so no further model call is sent"
CLAIM_KEYS = ("text", "start", "end", "documents", "quotes")
# An answer long enough that the claims request is over a window of WINDOW_TOKENS,
# which every other request of a run over TWO_LINE_MEMOS fits.
LONG_ANSWER = " ".join([HALLORAN] * 1_000)
WINDOW_TOKENS = 8_192
# A claim long enough that its linking request, which shows it beside the answer, is
# over a window of LINKING_WINDOW_TOKENS, which the claims request fits.
LONG_CLAIM = " ".join([MERROW] * 300)
LINKING_WINDOW_TOKENS = 4_800


@pytest.fixture
def quoting_llm(answering_llm, shown_ids):
    """``answering_llm``, each summary quoting a line of its memo a round and citing
    the first summary shown, and the answer ANSWER, citing every summary."""
    return answering_llm.answers(
        relevance=quoting_a_line_a_round(shown_ids),
        reasoning=citing(ANSWER, slice(None)),
    )


def run_memos(llm, tmp_path: Path, **options) -> Answer:
    """A run of QUESTION over TWO_LINE_MEMOS, ``llm`` answering, its folder under
    ``tmp_path / "logs"``."""
    logs = tmp_path / "logs"
    return run(QUESTION, TWO_LINE_MEMOS, completion=llm, logs_dir=logs, **options)


def claimed(llm, tmp_path: Path, *claims, **options) -> Answer:
    """A run of QUESTION over TWO_LINE_MEMOS whose claims are ``claims``, each its
    text and the ids that support it: the claims call splits the answer into them, and
    each one's linking call links it to its ids."""
    llm.answers(claims=claims_reply(*claims), linking=linking_reply(dict(claims)))
    return run_memos(llm, tmp_path, **options)


def place(quote: dict) -> dict:
    """Where ``quote`` is, as a claim names it."""
    return {key: quote[key] for key in ("document", "start", "end")}


def cancelling(cancel: threading.Event, reply):
    """A reply that sets ``cancel`` as it answers with ``reply``."""

    def answer(request):
        cancel.set()
        return reply(request) if callable(reply) else reply

    return answer


def strict(name: str, schema: dict) -> dict:
    """The response format of a request whose reply fits ``schema``, named ``name``."""
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": True},
    }


def strict_object(title: str, properties: dict, **extra) -> dict:
    """A strict JSON schema of an object of ``properties``, all required."""
    return {
        **extra,
        "properties": properties,
        "title": title,
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
    }


def id_list(title: str, ids: list[str]) -> dict:
    """A field listing some of ``ids``."""
    return {"items": {"enum": ids, "type": "string"}, "title": title, "type": "array"}


CLAIM_TEXTS = strict(
    "Claims",
    strict_object(
        "Claims",
        {"claims": {"items": {"type": "string"}, "title": "Claims", "type": "array"}},
    ),
)
"""The schema of a sentence-level claims request: the claims' texts alone."""


def claims_with_ids(ids: list[str]) -> dict:
    """The schema of a document-level claims request, citing ``ids``."""
    claim = strict_object(
        "Claim",
        {
            "text": {"title": "Text", "type": "string"},
            "document_ids": id_list("Document Ids", ids),
        },
    )
    claims = {"items": {"$ref": "#/$defs/Claim"}, "title": "Claims", "type": "array"}
    return strict(
        "Claims",
        strict_object("Claims", {"claims": claims}, **{"$defs": {"Claim": claim}}),
    )


LINKS = strict(
    "Linking",
    strict_object(
        "Linking",
        {
            "direct": id_list("Direct", QUOTE_IDS),
            "context": id_list("Context", QUOTE_IDS),
        },
    ),
)
"""The schema of a linking request over QUOTES."""


@pytest.mark.parametrize("level", EVIDENCE_LEVELS)
def test_a_claims_request_is_the_owners_prompt_then_the_question_answer_and_evidence(
    quoting_llm, tmp_path, level
):
    result = run_memos(quoting_llm, tmp_path, evidence_level=level)
    halloran, merrow, calder = ids = result.document_ids
    summaries = f"""<summaries>
### Document ID: {halloran}
Notes on Halloran memo.
### Document ID: {merrow}
Notes on Merrow memo.
### Document ID: {calder}
Notes on Calder memo.
</summaries>"""
    evidence, schema = {
        "sentence": (QUOTES, CLAIM_TEXTS),
        "document": (summaries, claims_with_ids(ids)),
    }[level]
    [request] = quoting_llm.requests_for("claims")
    assert request["messages"] == [
        {"role": "system", "content": SYSTEMS[level]},
        {"role": "user", "content": QUESTION_AND_ANSWER + evidence},
    ]
    assert request["response_format"] == schema


def test_a_linking_request_shows_the_question_answer_claim_and_named_quotes(
    quoting_llm, tmp_path
):
    claimed(quoting_llm, tmp_path, (HALLORAN, []), (MERROW, []))
    requests = quoting_llm.requests_for("linking")
    [request] = [r for r in requests if linked_claim(r) == MERROW]
    claim = f"<claim>\n{MERROW}\n</claim>\n"
    assert request["messages"] == [
        {"role": "system", "content": LINKING_SYSTEM},
        {"role": "user", "content": QUESTION_AND_ANSWER + claim + NAMED_QUOTES},
    ]
    assert request["response_format"] == LINKS
    assert sorted(map(linked_claim, requests)) == [HALLORAN, MERROW]


def test_a_run_pins_and_records_the_claims_and_linking_prompts(quoting_llm, tmp_path):
    result = run_memos(quoting_llm, tmp_path)
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    pinned = manifest["config"]["prompts"]
    assert (pinned["claims"], pinned["linking"]) == ("v3", "v1")
    read = manifest["sources"]["prompts"]
    assert read["linking"] == str(settings.PROMPTS_DIR / "linking" / "v1.yaml")
    assert read["claims"] == str(settings.PROMPTS_DIR / "claims" / "v3.yaml")


def test_claims_are_split_and_linked_on_reasonings_model_with_its_params(
    quoting_llm, tmp_path
):
    params = {"temperature": 0.3, "seed": 7}
    claimed(quoting_llm, tmp_path, (ANSWER, []), model="openai/gpt-4o", params=params)
    [reasoning] = quoting_llm.requests_for("reasoning")
    [claims] = quoting_llm.requests_for("claims")
    [linking] = quoting_llm.requests_for("linking")
    for request in (claims, linking):
        assert request["model"] == reasoning["model"] == "openai/gpt-4o"
        assert (request["temperature"], request["seed"]) == (0.3, 7)


def test_claims_are_the_answers_own_words_in_its_order_with_what_supports_them(
    quoting_llm, tmp_path
):
    result = claimed(quoting_llm, tmp_path, (MERROW, ["q5", "q3"]), (HALLORAN, ["q2"]))
    quotes = result.provenance["relevant_trail"]["quotes"]
    halloran, merrow, calder = result.document_ids
    assert result.provenance["claims"] == [
        {
            "text": HALLORAN,
            "start": 0,
            "end": len(HALLORAN),
            "documents": [halloran],
            "quotes": [place(quotes[1])],
        },
        {
            "text": MERROW,
            "start": len(HALLORAN) + 1,
            "end": len(ANSWER),
            "documents": [merrow, calder],
            "quotes": [place(quotes[2]), place(quotes[4])],
        },
    ]
    saved = json.loads((result.run_dir / "claims" / "result.json").read_text())
    assert saved["claims"] == result.provenance["claims"]
    assert saved["evidence_trail"] == result.provenance["evidence_trail"]


def test_a_claim_keeps_its_direct_quotes_alone_and_each_call_is_named_for_it(
    quoting_llm, tmp_path
):
    quoting_llm.answers(
        claims=claims_reply((MERROW, []), (HALLORAN, [])),
        linking=linking_reply({HALLORAN: ["q2"], MERROW: ["q4"]}, context=["q1", "q3"]),
    )
    result = run_memos(quoting_llm, tmp_path)
    trail = result.provenance["relevant_trail"]
    quotes = trail["quotes"]
    claims = result.provenance["claims"]
    assert [claim["quotes"] for claim in claims] == [
        [place(quotes[1])],
        [place(quotes[3])],
    ]
    assert result.provenance["evidence_trail"] == evidence(
        "claims", trail["documents"][:2], [quotes[1], quotes[3]]
    )
    saved = json.loads((result.run_dir / "claims" / "result.json").read_text())
    assert saved["claims"] == claims
    assert all(set(claim) == set(CLAIM_KEYS) for claim in saved["claims"])
    calls = json.loads((result.run_dir / "claims" / "calls.json").read_text())
    replies = {call["kind"]: json.loads(call["output"]) for call in calls}
    assert replies == {
        "claims": {"claims": [MERROW, HALLORAN]},
        "linking-c1": {"direct": ["q2"], "context": ["q1", "q3"]},
        "linking-c2": {"direct": ["q4"], "context": ["q1", "q3"]},
    }


def test_the_evidence_trail_is_what_the_claims_cite_in_the_answer_folder_and_event(
    quoting_llm, tmp_path, events_in
):
    result = claimed(quoting_llm, tmp_path, (MERROW, []), (HALLORAN, ["q2"]))
    trail = result.provenance["relevant_trail"]
    claimed_trail = evidence("claims", [trail["documents"][0]], [trail["quotes"][1]])
    assert result.provenance["evidence_trail"] == claimed_trail
    saved = json.loads((result.run_dir / "claims" / "result.json").read_text())
    assert saved["evidence_trail"] == claimed_trail
    [finished] = [
        e for e in events_in(result.run_dir) if e["type"] == "claims_finished"
    ]
    assert finished["evidence_trail"] == claimed_trail
    assert trace(result)["evidence_trail"] == claimed_trail


def test_at_the_document_level_the_evidence_trail_is_the_documents_claims_name(
    quoting_llm, tmp_path
):
    def claiming_merrow(request) -> dict:
        merrow = shown_evidence(request)[1]
        return claims_reply((HALLORAN, []), (MERROW, [merrow]))(request)

    quoting_llm.answers(claims=claiming_merrow)
    result = run_memos(quoting_llm, tmp_path, evidence_level="document")
    documents = result.provenance["relevant_trail"]["documents"]
    assert result.provenance["evidence_trail"] == evidence("claims", [documents[1]], [])
    assert len(quoting_llm.requests_for("claims")) == 1
    assert quoting_llm.requests_for("linking") == []


def test_claims_that_cite_nothing_leave_the_whole_trail_as_the_evidence_trail(
    quoting_llm, tmp_path
):
    result = claimed(quoting_llm, tmp_path, (HALLORAN, []), (MERROW, []))
    assert [claim["documents"] for claim in result.provenance["claims"]] == [[], []]
    assert result.provenance["evidence_trail"] == whole_trail(result.provenance)
    assert trace(result)["evidence_trail"] == whole_trail(result.provenance)


def test_a_sentence_the_answer_repeats_is_claimed_where_each_claim_makes_it(
    quoting_llm, tmp_path
):
    repeated = f"{ANSWER} {HALLORAN}"
    quoting_llm.answers(reasoning=citing(repeated, slice(None)))
    result = claimed(
        quoting_llm, tmp_path, (HALLORAN, ["q2"]), (MERROW, []), (HALLORAN, ["q1"])
    )
    claims = result.provenance["claims"]
    starts = [0, len(HALLORAN) + 1, repeated.rindex(HALLORAN)]
    assert [claim["start"] for claim in claims] == starts
    assert [claim["text"] for claim in claims] == [HALLORAN, MERROW, HALLORAN]


# A claim found again where one is placed is dropped, as is one found there once a
# claim the answer words otherwise is placed before it.
@pytest.mark.parametrize(
    ("claims", "kept"),
    [
        ([HALLORAN, MERROW, HALLORAN], [HALLORAN, MERROW]),
        (
            [MERROW, "The Halloran memo lists two sites.", MERROW],
            [HALLORAN, MERROW],
        ),
    ],
    ids=["repeated", "after-a-located-one"],
)
def test_a_claim_overlapping_one_placed_is_dropped_with_a_warning(
    quoting_llm, tmp_path, caplog, claims, kept
):
    caplog.set_level(logging.WARNING, logger="r3con")
    result = claimed(quoting_llm, tmp_path, *((text, []) for text in claims))
    assert [claim["text"] for claim in result.provenance["claims"]] == kept
    overlapping = claims[-1]
    assert (
        f"claims: dropped a claim that overlaps another: {overlapping!r}"
    ) in caplog.text


def test_an_id_the_call_did_not_show_is_dropped_with_a_warning(
    quoting_llm, tmp_path, caplog
):
    caplog.set_level(logging.WARNING, logger="r3con")
    result = claimed(quoting_llm, tmp_path, (HALLORAN, ["q9", "q2", "a1b2c3"]))
    [claim] = result.provenance["claims"]
    assert claim["quotes"] == [place(result.provenance["relevant_trail"]["quotes"][1])]
    assert (
        "linking-c1: dropped provenance id(s) the call did not show: ['q9', 'a1b2c3']"
    ) in caplog.text


def test_a_claim_the_answer_does_not_make_is_dropped_with_a_warning(
    quoting_llm, tmp_path, caplog
):
    caplog.set_level(logging.WARNING, logger="r3con")
    result = claimed(
        quoting_llm,
        tmp_path,
        (HALLORAN, []),
        ("Calder logs nothing.", ["q5"]),
        (" ", []),
        (MERROW, []),
    )
    assert [claim["text"] for claim in result.provenance["claims"]] == [
        HALLORAN,
        MERROW,
    ]
    assert (
        "claims: dropped a claim not found in the answer: 'Calder logs nothing.'"
    ) in caplog.text
    assert "claims: dropped a blank claim" in caplog.text


@pytest.mark.parametrize(
    ("altered", "found"),
    [
        ("the merrow memo logs two sites", MERROW.removesuffix(".")),
        ("The Merrow memo lists two sites.", MERROW),
    ],
    ids=["normalized", "fuzzy"],
)
def test_a_claim_altered_slightly_is_found_and_linked_in_the_answers_words(
    quoting_llm, tmp_path, altered, found
):
    quoting_llm.answers(
        claims=claims_reply((altered, [])), linking=linking_reply({found: ["q4"]})
    )
    result = run_memos(quoting_llm, tmp_path)
    [claim] = result.provenance["claims"]
    assert claim["text"] == found == ANSWER[claim["start"] : claim["end"]]
    assert claim["quotes"] == [place(result.provenance["relevant_trail"]["quotes"][3])]


def test_an_elided_quote_is_shown_in_the_notes_words_not_the_span_it_skips(
    answering_llm, tmp_path
):
    def eliding(request) -> dict:
        line = request["messages"][1]["content"].splitlines()[1]
        return {"summary": "Notes.", "quotes": [line.replace(" logs ", " ... ")]}

    answering_llm.answers(relevance=eliding, reasoning=citing(ANSWER, slice(None)))
    result = run_memos(answering_llm, tmp_path)
    [elided, *_] = result.provenance["relevant_trail"]["quotes"]
    assert (elided["match"], elided["document_text"]) == ("elided", HALLORAN)
    [request] = answering_llm.requests_for("claims")
    user = request["messages"][1]["content"]
    assert "### Quote ID: q1\nThe Halloran memo ... two sites.\n" in user
    assert HALLORAN not in user.split("<quotes>", 1)[1]
    [linking] = answering_llm.requests_for("linking")
    quotes = linking["messages"][1]["content"].split("<quotes>", 1)[1]
    assert quotes.startswith(
        "\n### Quote ID: q1\n(from: q3/halloran.txt)\nThe Halloran memo ... two sites.\n"
    )
    assert HALLORAN not in quotes


@pytest.mark.parametrize("level", EVIDENCE_LEVELS)
def test_an_answer_whose_trail_holds_no_evidence_makes_no_claims_call(
    answering_llm, tmp_path, level
):
    result = run_memos(answering_llm, tmp_path, evidence_level=level)
    trail = result.provenance["relevant_trail"]
    assert (trail["quotes"], trail["documents"]) == ([], [])
    assert answering_llm.requests_for("claims") == []
    assert answering_llm.requests_for("linking") == []
    assert result.provenance["claims"] == []
    nothing = evidence("trail", [], [])
    assert result.provenance["evidence_trail"] == nothing
    folder = result.run_dir / "claims"
    assert json.loads((folder / "calls.json").read_text()) == []
    saved = json.loads((folder / "result.json").read_text())
    assert (saved["claims"], saved["evidence_trail"]) == ([], nothing)


def test_a_claims_request_over_the_models_line_is_not_sent(
    quoting_llm, window, tmp_path, caplog, events_in
):
    caplog.set_level(logging.WARNING, logger="r3con")
    quoting_llm.answers(reasoning=citing(LONG_ANSWER, slice(None)))
    result = run_memos(quoting_llm, tmp_path, model=window(WINDOW_TOKENS))
    assert result.answer == LONG_ANSWER
    assert result.provenance["relevant_trail"]["quotes"]
    assert quoting_llm.requests_for("claims") == []
    assert quoting_llm.requests_for("linking") == []
    assert result.provenance["claims"] == []
    assert result.provenance["evidence_trail"] == whole_trail(result.provenance)
    assert "claims: none, as their request would take about " in caplog.text
    assert events_in(result.run_dir)[-1]["type"] == "run_finished"


def test_a_linking_request_over_the_models_line_leaves_only_its_claim_unlinked(
    quoting_llm, window, tmp_path, caplog
):
    caplog.set_level(logging.WARNING, logger="r3con")
    quoting_llm.answers(reasoning=citing(f"{HALLORAN} {LONG_CLAIM}", slice(None)))
    result = claimed(
        quoting_llm,
        tmp_path,
        (HALLORAN, ["q2"]),
        (LONG_CLAIM, ["q4"]),
        model=window(LINKING_WINDOW_TOKENS),
    )
    [linked, unlinked] = result.provenance["claims"]
    trail = result.provenance["relevant_trail"]
    assert linked["quotes"] == [place(trail["quotes"][1])]
    assert (unlinked["text"], unlinked["documents"], unlinked["quotes"]) == (
        LONG_CLAIM,
        [],
        [],
    )
    assert [linked_claim(r) for r in quoting_llm.requests_for("linking")] == [HALLORAN]
    assert (
        "claims: claim 2 keeps no quotes, as its linking request would take about "
    ) in caplog.text
    assert result.provenance["evidence_trail"] == evidence(
        "claims", trail["documents"][:1], [trail["quotes"][1]]
    )


@pytest.mark.parametrize(
    ("reply", "sends", "error"),
    [
        (bad_key(), 1, "AuthenticationError: "),
        ("not JSON", 3, "ValidationError: Invalid JSON: expected ident"),
    ],
    ids=["refused", "invalid"],
)
def test_a_claims_call_that_fails_leaves_no_claims_and_the_run_finishes(
    quoting_llm, tmp_path, caplog, events_in, reply, sends, error
):
    caplog.set_level(logging.WARNING, logger="r3con")
    quoting_llm.answers(claims=reply)
    result = run_memos(quoting_llm, tmp_path)
    assert result.answer == ANSWER
    assert result.provenance["claims"] == []
    assert result.provenance["evidence_trail"] == whole_trail(result.provenance)
    assert len(quoting_llm.requests_for("claims")) == sends
    assert quoting_llm.requests_for("linking") == []
    assert f"claims: none, as their call failed: {error}" in caplog.text
    events = events_in(result.run_dir)
    assert [event["type"] for event in events[-3:]] == [
        "claims_started",
        "claims_finished",
        "run_finished",
    ]
    assert (events[-2]["claims"], events[-2]["evidence_trail"]) == (
        [],
        whole_trail(result.provenance),
    )
    assert not (result.run_dir / "claims" / "error.txt").exists()


@pytest.mark.parametrize(
    ("reply", "sends", "error"),
    [
        (bad_key(), 1, "AuthenticationError: "),
        ("not JSON", 3, "ValidationError: Invalid JSON: expected ident"),
    ],
    ids=["refused", "invalid"],
)
def test_a_linking_call_that_fails_leaves_only_its_claim_unlinked(
    quoting_llm, tmp_path, caplog, events_in, reply, sends, error
):
    caplog.set_level(logging.WARNING, logger="r3con")
    links = linking_reply({HALLORAN: ["q2"]})

    def linking(request):
        if linked_claim(request) == HALLORAN:
            return links(request)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    quoting_llm.answers(
        claims=claims_reply((HALLORAN, []), (MERROW, [])), linking=linking
    )
    result = run_memos(quoting_llm, tmp_path)
    trail = result.provenance["relevant_trail"]
    [linked, unlinked] = result.provenance["claims"]
    assert (linked["text"], linked["quotes"]) == (HALLORAN, [place(trail["quotes"][1])])
    assert (unlinked["text"], unlinked["documents"], unlinked["quotes"]) == (
        MERROW,
        [],
        [],
    )
    sent = [r for r in quoting_llm.requests_for("linking") if linked_claim(r) == MERROW]
    assert len(sent) == sends
    assert f"claims: claim 2 keeps no quotes, as its linking call failed: {error}" in (
        caplog.text
    )
    assert result.provenance["evidence_trail"] == evidence(
        "claims", trail["documents"][:1], [trail["quotes"][1]]
    )
    assert events_in(result.run_dir)[-1]["type"] == "run_finished"


def test_a_cancel_during_the_claims_call_leaves_no_claims_and_finishes_the_run(
    quoting_llm, tmp_path, caplog, events_in
):
    caplog.set_level(logging.WARNING, logger="r3con")
    cancel = threading.Event()
    # A reply that does not fit, which the call would send again but for the cancel.
    quoting_llm.answers(claims=cancelling(cancel, "not JSON"))
    result = run_memos(quoting_llm, tmp_path, cancel=cancel)
    assert result.answer == ANSWER
    assert result.provenance["claims"] == []
    assert len(quoting_llm.requests_for("claims")) == 1
    assert f"claims: none, as their call failed: {CANCELLED}" in caplog.text
    assert events_in(result.run_dir)[-1]["type"] == "run_finished"
    saved = json.loads((result.run_dir / "claims" / "result.json").read_text())
    assert saved["claims"] == []
    with pytest.raises(ValueError, match="this run finished"):
        resume(result.run_dir, TWO_LINE_MEMOS, completion=quoting_llm)


def test_a_cancel_during_the_claims_call_of_an_ask_finishes_the_ask(
    quoting_llm, tmp_path, events_in
):
    result = claimed(quoting_llm, tmp_path, (HALLORAN, ["q2"]))
    cancel = threading.Event()
    quoting_llm.answers(claims=cancelling(cancel, "not JSON"))
    asked = ask(result, "Who else?", completion=quoting_llm, cancel=cancel)
    assert (asked.answer, asked.provenance["claims"]) == (ANSWER, [])
    assert events_in(asked.run_dir)[-1]["type"] == "ask_finished"


def test_a_cancel_during_reasoning_still_cancels_the_run(
    quoting_llm, tmp_path, events_in
):
    cancel = threading.Event()
    # A reply that does not fit, which reasoning would send again but for the cancel.
    quoting_llm.answers(reasoning=cancelling(cancel, {"answer": ""}))
    with pytest.raises(Cancelled):
        run_memos(quoting_llm, tmp_path, cancel=cancel)
    [run_dir] = (tmp_path / "logs").iterdir()
    failed = events_in(run_dir)[-1]
    assert (failed["type"], failed["stage"]) == ("run_failed", "reasoning")
    assert quoting_llm.requests_for("claims") == []
    assert not (run_dir / "claims").exists()


def test_claims_turned_off_make_no_call_no_folder_and_no_event(
    quoting_llm, tmp_path, events_in
):
    result = run_memos(quoting_llm, tmp_path, claims=False)
    assert result.provenance["relevant_trail"]["quotes"]
    assert quoting_llm.requests_for("claims") == []
    assert result.provenance["claims"] == trace(result)["claims"] == []
    assert result.provenance["evidence_trail"] == whole_trail(result.provenance)
    assert trace(result)["evidence_trail"] == whole_trail(result.provenance)
    assert not (result.run_dir / "claims").exists()
    assert not any(e["type"].startswith("claims_") for e in events_in(result.run_dir))
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert (manifest["config"]["claims"], manifest["config"]["overrides"]) == (
        False,
        {"claims": False},
    )


def test_a_document_level_request_shows_each_round_1_summary_or_the_latest_instead(
    answering_llm, grown_registry, window, tmp_path
):
    def summary(request) -> str:
        line = first_line(request)
        if is_later_round(request):
            return f"Round 2 notes on {line}."
        return "" if line == "Merrow memo" else f"Round 1 notes on {line}."

    def claiming_a_part_and_merrow(request) -> dict:
        shown = shown_evidence(request)
        return claims_reply((ANSWER, [shown[-1], shown[2]]))(request)

    answering_llm.answers(
        relevance=summary,
        reasoning=citing(ANSWER, slice(None)),
        claims=claiming_a_part_and_merrow,
    )
    documents = ["Halloran memo", grown_registry(40_000), "Merrow memo"]
    result = run(
        QUESTION,
        documents,
        model=window(5_000),
        evidence_level="document",
        completion=answering_llm,
        logs_dir=tmp_path,
    )
    halloran, registry, merrow = result.document_ids
    relevance = json.loads((result.run_dir / "relevance" / "result.json").read_text())
    round_1 = {s["id"]: s["summary"] for s in relevance["rounds"][0]["snippets"]}
    parts = [id_ for id_ in round_1 if id_.startswith(f"{registry}.")]
    assert len(parts) > 1
    shown = [
        (halloran, "Round 1 notes on Halloran memo."),
        *((part, round_1[part]) for part in parts),
        (merrow, "Round 2 notes on Merrow memo."),
    ]
    [request] = answering_llm.requests_for("claims")
    summaries = request["messages"][1]["content"].split("\n<summaries>\n", 1)[1]
    assert summaries == "".join(
        f"### Document ID: {id_}\n{text}\n" for id_, text in shown
    ) + ("</summaries>")
    assert all(text.startswith("Round 1 notes") for _, text in shown[1:-1])
    [claim] = result.provenance["claims"]
    assert (claim["documents"], claim["quotes"]) == ([registry, merrow], [])


@pytest.mark.parametrize("rule", RULES)
def test_trace_gives_the_saved_claims_whose_quotes_the_rules_trail_quotes(
    quoting_llm, tmp_path, rule
):
    result = claimed(quoting_llm, tmp_path, (HALLORAN, ["q2"]), (MERROW, ["q4"]))
    traced = trace(result.run_dir, rule)
    assert traced["claims"] == result.provenance["claims"] != []
    quotes = traced["relevant_trail"]["quotes"]
    places = [place(quote) for quote in quotes]
    assert all(q in places for c in traced["claims"] for q in c["quotes"])
    assert traced["evidence_trail"] == evidence(
        "claims", traced["relevant_trail"]["documents"][:2], [quotes[1], quotes[3]]
    )


def without_an_evidence_trail(folder: Path) -> None:
    """``claims/result.json`` as r3context 0.6.45 and earlier wrote it."""
    in_json("claims/result.json", lambda result: result.pop("evidence_trail"))(folder)


def without_claims(folder: Path) -> None:
    """A folder an r3con made before claims."""
    shutil.rmtree(folder / "claims")


@pytest.mark.parametrize(
    ("older", "claims_kept"),
    [(without_an_evidence_trail, True), (without_claims, False)],
    ids=["claims-without-evidence-trail", "no-claims"],
)
@pytest.mark.parametrize("rule", RULES)
def test_claims_saved_without_an_evidence_trail_or_none_trace_with_the_one_they_make(
    quoting_llm, tmp_path, older, claims_kept, rule
):
    result = claimed(quoting_llm, tmp_path, (MERROW, []), (HALLORAN, ["q2"]))
    asked = ask(result, "Who else?", completion=quoting_llm)
    for folder in (result.run_dir, asked.run_dir):
        assert folder is not None
        before = trace(folder, rule)
        assert before["evidence_trail"]["from"] == "claims"
        older(folder)
        unclaimed = {**before, "claims": [], "evidence_trail": whole_trail(before)}
        assert trace(folder, rule) == (before if claims_kept else unclaimed)


def test_an_ask_finds_the_claims_of_its_own_answer_on_its_own_model(
    quoting_llm, tmp_path, events_in
):
    result = claimed(quoting_llm, tmp_path, (ANSWER, ["q2"]))
    quoting_llm.answers(
        reasoning=citing(MERROW, slice(None)),
        claims=claims_reply((MERROW, [])),
        linking=linking_reply({MERROW: ["q4"]}),
    )
    asked = ask(result, "Who else?", model="openai/gpt-4o", completion=quoting_llm)
    [_, request] = quoting_llm.requests_for("claims")
    [_, linking] = quoting_llm.requests_for("linking")
    assert request["model"] == linking["model"] == "openai/gpt-4o"
    user = request["messages"][1]["content"]
    assert user.startswith(f"<question>\nWho else?\n</question>\n<answer>\n{MERROW}\n")
    [claim] = asked.provenance["claims"]
    assert (claim["text"], claim["documents"]) == (MERROW, [result.document_ids[1]])
    trail = asked.provenance["relevant_trail"]
    assert asked.provenance["evidence_trail"] == evidence(
        "claims", [trail["documents"][1]], [trail["quotes"][3]]
    )
    traced = trace(asked)
    assert (traced["claims"], traced["evidence_trail"]) == (
        asked.provenance["claims"],
        asked.provenance["evidence_trail"],
    )
    [finished] = [e for e in events_in(asked.run_dir) if e["type"] == "claims_finished"]
    assert (finished["claims"], finished["evidence_trail"]) == (
        asked.provenance["claims"],
        asked.provenance["evidence_trail"],
    )
    assert result.provenance["claims"] == trace(result)["claims"] != []


def test_a_run_that_failed_resumes_through_to_its_claims(quoting_llm, tmp_path):
    quoting_llm.answers(parsing=bad_key())
    with pytest.raises(Exception, match="bad key"):
        claimed(quoting_llm, tmp_path, (HALLORAN, ["q2"]))
    [run_dir] = (tmp_path / "logs").iterdir()
    quoting_llm.answers(parsing=ANSWERING["parsing"])
    resumed = resume(run_dir, TWO_LINE_MEMOS, completion=quoting_llm)
    [claim] = resumed.provenance["claims"]
    assert claim["text"] == HALLORAN
    trail = resumed.provenance["relevant_trail"]
    assert resumed.provenance["evidence_trail"] == evidence(
        "claims", [trail["documents"][0]], [trail["quotes"][1]]
    )
    traced = trace(run_dir)
    assert (traced["claims"], traced["evidence_trail"]) == (
        resumed.provenance["claims"],
        resumed.provenance["evidence_trail"],
    )


def test_a_run_stopped_before_its_claims_finished_resumes_from_reasoning(
    quoting_llm, tmp_path
):
    result = claimed(quoting_llm, tmp_path, (HALLORAN, ["q2"]))
    shutil.rmtree(result.run_dir / "claims")
    sent = len(quoting_llm.requests)
    resumed = resume(result.run_dir, TWO_LINE_MEMOS, completion=quoting_llm)
    stages = [stage_of(request) for request in quoting_llm.requests[sent:]]
    assert stages == ["reasoning", "claims", "linking"]
    assert resumed.provenance["claims"] == result.provenance["claims"]


def as_before_claims(manifest: dict) -> None:
    """A run's manifest as r3con wrote one before it made claims."""
    del manifest["config"]["claims"]
    for stage in ("claims", "linking"):
        del manifest["config"]["prompts"][stage]
        del manifest["sources"]["prompts"][stage]


def test_a_run_made_before_claims_is_asked_without_them_and_stays_finished(
    quoting_llm, tmp_path
):
    result = claimed(quoting_llm, tmp_path, (HALLORAN, ["q2"]))
    shutil.rmtree(result.run_dir / "claims")
    in_json("manifest.json", as_before_claims)(result.run_dir)
    sent = len(quoting_llm.requests)
    asked = ask(result, "Who else?", completion=quoting_llm)
    assert [stage_of(r) for r in quoting_llm.requests[sent:]] == ["reasoning"]
    assert asked.provenance["claims"] == []
    assert not (asked.run_dir / "claims").exists()
    with pytest.raises(ValueError, match="this run finished"):
        resume(result.run_dir, TWO_LINE_MEMOS, completion=quoting_llm)


def failed_with_its_claims_prompt_shadowed(llm, tmp_path: Path, **options) -> Path:
    """The folder of a run that failed in parsing, whose claims prompt a ``./prompts``
    overlay now shadows; ``llm`` answering every stage from then on."""
    llm.answers(parsing=bad_key())
    with pytest.raises(Exception, match="bad key"):
        run_memos(llm, tmp_path, **options)
    [run_dir] = (tmp_path / "logs").iterdir()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    pinned = Path(manifest["sources"]["prompts"]["claims"])
    overlay = Path("prompts") / "claims" / pinned.name
    overlay.parent.mkdir(parents=True)
    overlay.write_text(pinned.read_text())
    llm.answers(parsing=ANSWERING["parsing"])
    return run_dir


def test_a_resume_refuses_a_claims_prompt_that_now_resolves_elsewhere(
    quoting_llm, tmp_path
):
    run_dir = failed_with_its_claims_prompt_shadowed(quoting_llm, tmp_path)
    with pytest.raises(ValueError, match=r"claims now reads \S+/prompts/claims/v3"):
        resume(run_dir, TWO_LINE_MEMOS, completion=quoting_llm)


def test_a_resume_with_claims_off_does_not_check_where_its_claims_prompt_is(
    quoting_llm, tmp_path
):
    run_dir = failed_with_its_claims_prompt_shadowed(
        quoting_llm, tmp_path, claims=False
    )
    resumed = resume(run_dir, TWO_LINE_MEMOS, completion=quoting_llm)
    assert (resumed.answer, resumed.provenance["claims"]) == (ANSWER, [])
