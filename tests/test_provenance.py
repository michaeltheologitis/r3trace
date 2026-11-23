import pytest

from r3con.provenance import (
    DEFAULT_RULE,
    build_graph,
    build_provenance,
    evidence_trail,
    with_claims,
)
from r3con.stages.relevance import RelevantContext, SnippetRecord
from tests.fakes import quote

NAMES = {"a": "a.txt", "b": "b.txt", "c": "c.txt"}
NO_PARSE_CITES = [[], [], []]


def snippet(id_: str, *quotes: dict, cites: tuple[str, ...] = ()) -> SnippetRecord:
    return SnippetRecord(
        id=id_, summary=f"Notes on {id_}.", quotes=list(quotes), provenance=list(cites)
    )


def quoted(document: str, start: int, end: int, text: str) -> dict:
    """A quote behind an answer found exactly, as its provenance gives it."""
    name = NAMES[document]
    return {
        "document": document,
        "name": name,
        "start": start,
        "end": end,
        "text": text,
        "document_text": text,
        "match": "exact",
        "score": None,
    }


def named(document: str) -> dict:
    """A document behind an answer, as its provenance gives it."""
    return {"document": document, "name": NAMES[document]}


def run_over(rounds) -> dict:
    """The records of a run over the documents a, b and c with ``rounds``, each a
    round's snippets, as :func:`build_graph` and :func:`build_provenance` take them."""
    relevance = RelevantContext(
        records=[
            [[s for s in round_ if s.id.partition(".")[0] == id_] for id_ in NAMES]
            for round_ in rounds
        ]
    )
    return {"relevance": relevance, "names": NAMES}


def trail_of(
    rounds, summary_ids=(), parse_ids=(), parses=NO_PARSE_CITES, rule=DEFAULT_RULE
):
    """The relevant trail of an answer citing ``summary_ids`` and ``parse_ids`` over
    :func:`run_over` ``rounds``, each document's parse citing what ``parses`` lists
    for it."""
    citations = {"summary_ids": list(summary_ids), "parse_ids": list(parse_ids)}
    return build_provenance(
        citations, parsing_provenance=parses, rule=rule, **run_over(rounds)
    )


def places(quotes: list) -> list[tuple]:
    return [(q["document"], q["start"], q["end"]) for q in quotes]


# Two rounds; the answer cites b's final summary, which cites a's first, and c's parse,
# which cites nothing.
CITED_ROUNDS = [
    [snippet("a", quote("A said x.", 0, 9, "exact")), snippet("b"), snippet("c")],
    [
        snippet("a"),
        snippet("b", quote("B said y.", 10, 19, "exact"), cites=("a",)),
        snippet("c"),
    ],
]


def test_the_graph_holds_every_edge_from_the_answer_back_each_part_under_its_round():
    first = [
        snippet("a"),
        snippet("b.1", quote("B, part one.", 0, 12, "exact")),
        snippet("b.2"),
        snippet("c"),
    ]
    second = [
        snippet("a", cites=("b.2",)),
        snippet("b.1"),
        snippet("b.2"),
        snippet("b.3", cites=("a",)),
        snippet("c"),
    ]
    graph = build_graph(
        {"summary_ids": ["b.3"], "parse_ids": ["c"]},
        parsing_provenance=[[], [], ["a"]],
        **run_over([first, second]),
    )
    assert graph.edges == [
        ["answer", "summary:2:b.3", "cites"],
        ["answer", "parse:c", "cites"],
        ["parse:a", "document:a", "parses"],
        ["parse:b", "document:b", "parses"],
        ["parse:c", "summary:2:a", "cites"],
        ["parse:c", "document:c", "parses"],
        ["summary:2:a", "summary:1:b.2", "cites"],
        ["summary:2:a", "document:a", "summarizes"],
        ["summary:2:b.1", "document:b", "summarizes"],
        ["summary:2:b.2", "document:b", "summarizes"],
        ["summary:2:b.3", "summary:1:a", "cites"],
        ["summary:2:b.3", "document:b", "summarizes"],
        ["summary:2:c", "document:c", "summarizes"],
        ["summary:1:a", "document:a", "summarizes"],
        ["summary:1:b.1", "document:b", "summarizes"],
        ["summary:1:b.2", "document:b", "summarizes"],
        ["summary:1:c", "document:c", "summarizes"],
    ]
    assert graph.quotes["summary:1:b.1"] == [quoted("b", 0, 12, "B, part one.")]


def test_the_default_trail_reaches_a_document_only_through_a_summary_of_it():
    trail = trail_of(CITED_ROUNDS, summary_ids=["b"], parse_ids=["c"])
    assert trail == {
        "cited": {"summary_ids": ["b"], "parse_ids": ["c"]},
        "rule": "through-summaries",
        "nodes": [
            "answer",
            "summary:2:b",
            "parse:c",
            "summary:1:a",
            "document:b",
            "document:a",
        ],
        "edges": [
            ["answer", "summary:2:b", "cites"],
            ["answer", "parse:c", "cites"],
            ["summary:2:b", "summary:1:a", "cites"],
            ["summary:2:b", "document:b", "summarizes"],
            ["summary:1:a", "document:a", "summarizes"],
        ],
        "documents": [named("a"), named("b")],
        "quotes": [quoted("a", 0, 9, "A said x."), quoted("b", 10, 19, "B said y.")],
    }
    assert list(trail) == [
        "cited",
        "rule",
        "nodes",
        "edges",
        "documents",
        "quotes",
    ]


# The trail reaches documents in its own order; documents lists them in document order.
@pytest.mark.parametrize(
    ("rule", "reached", "documents"),
    [
        ("through-summaries", ["document:b", "document:a"], ["a", "b"]),
        (
            "parses-light-documents",
            ["document:b", "document:c", "document:a"],
            ["a", "b", "c"],
        ),
    ],
)
def test_a_cited_parse_that_cites_nothing_reaches_its_document_only_if_parses_do(
    rule, reached, documents
):
    trail = trail_of(CITED_ROUNDS, summary_ids=["b"], parse_ids=["c"], rule=rule)
    assert trail["rule"] == rule
    assert [node for node in trail["nodes"] if node.startswith("document:")] == reached
    assert trail["documents"] == [named(id_) for id_ in documents]
    assert places(trail["quotes"]) == [("a", 0, 9), ("b", 10, 19)]


def test_a_node_reached_twice_is_listed_once_and_both_edges_to_it_followed():
    trail = trail_of(
        CITED_ROUNDS[:1], summary_ids=["a"], parse_ids=["a"], parses=[["a"], [], []]
    )
    assert trail["nodes"] == ["answer", "summary:1:a", "parse:a", "document:a"]
    assert trail["edges"] == [
        ["answer", "summary:1:a", "cites"],
        ["answer", "parse:a", "cites"],
        ["summary:1:a", "document:a", "summarizes"],
        ["parse:a", "summary:1:a", "cites"],
    ]


def test_an_unknown_rule_is_refused_naming_the_known_ones():
    with pytest.raises(ValueError) as refused:
        trail_of(CITED_ROUNDS, rule="through-parses")
    assert str(refused.value) == (
        "no provenance rule is named 'through-parses'; the rules are "
        "'through-summaries', 'parses-light-documents'."
    )


@pytest.mark.parametrize("n_rounds", [0, 2])
def test_an_answer_that_cites_nothing_reaches_only_itself_and_quotes_nothing(n_rounds):
    trail = trail_of(CITED_ROUNDS[:n_rounds])
    assert (trail["documents"], trail["quotes"]) == ([], [])
    assert (trail["nodes"], trail["edges"]) == (["answer"], [])


def test_the_quotes_are_those_of_the_final_round_summaries_cited():
    final = [
        snippet("a", quote("A said x.", 0, 9, "exact")),
        snippet("b", quote("B said y.", 3, 12, "exact")),
        snippet("c", quote("C, it said, saw y.", 4, 22, "exact")),
    ]
    assert trail_of([final], summary_ids=["a", "c"])["quotes"] == [
        quoted("a", 0, 9, "A said x."),
        quoted("c", 4, 22, "C, it said, saw y."),
    ]


def test_the_quotes_follow_the_summaries_cited_back_round_by_round():
    first = [
        snippet("a", quote("A1.", 0, 5, "exact")),
        snippet("b", quote("B1.", 10, 15, "exact")),
        snippet("c", quote("C1.", 20, 25, "exact")),
    ]
    second = [
        snippet("a", quote("A2.", 30, 35, "exact")),
        snippet("b", quote("B2.", 40, 45, "exact"), cites=("c",)),
        snippet("c", quote("C2.", 50, 55, "exact")),
    ]
    third = [
        snippet("a", quote("A3.", 60, 65, "exact"), cites=("b",)),
        snippet("b"),
        snippet("c"),
    ]
    quotes = trail_of([first, second, third], summary_ids=["a"])["quotes"]
    assert places(quotes) == [("a", 60, 65), ("b", 40, 45), ("c", 20, 25)]


def test_the_quotes_follow_a_cited_parse_through_the_summaries_it_cites():
    first = [
        snippet("a", quote("A1.", 0, 5, "exact")),
        snippet("b", quote("B1.", 10, 15, "exact")),
        snippet("c"),
    ]
    second = [
        snippet("a", quote("A2.", 30, 35, "exact"), cites=("b",)),
        snippet("b"),
        snippet("c"),
    ]
    quotes = trail_of([first, second], parse_ids=["c", "b"], parses=[[], ["a"], ["a"]])[
        "quotes"
    ]
    assert places(quotes) == [("a", 30, 35), ("b", 10, 15)]


def test_a_part_is_cited_by_its_part_id_and_quoted_under_its_documents_id():
    first = [
        snippet("a", quote("A1.", 0, 5, "exact")),
        snippet("b.1", quote("B1, part one.", 10, 15, "exact")),
        snippet("b.2", quote("B1, part two.", 5_000, 5_005, "exact")),
        snippet("c", quote("C1.", 20, 25, "exact")),
    ]
    second = [
        snippet("a", quote("A2.", 30, 35, "exact"), cites=("b.2",)),
        snippet("b.1", quote("B2, part one.", 40, 45, "exact")),
        snippet("b.2", quote("B2, part two.", 5_040, 5_045, "exact")),
        snippet("c", quote("C2.", 50, 55, "exact")),
    ]
    quotes = trail_of([first, second], summary_ids=["a", "b.1"])["quotes"]
    assert places(quotes) == [("a", 30, 35), ("b", 40, 45), ("b", 5_000, 5_005)]


def test_a_quote_not_found_in_its_document_is_left_out():
    final = [
        snippet("a", quote("Not in a.", None, None, None), quote("A.", 0, 5, "exact")),
        snippet("b", quote("Not in b.", None, None, None)),
        snippet("c"),
    ]
    quotes = trail_of([final], summary_ids=["a", "b"])["quotes"]
    assert places(quotes) == [("a", 0, 5)]


def test_a_place_matched_alike_more_than_once_is_given_once_as_the_latest_round_copied_it():
    first = [
        snippet("a", quote("As round 1 copied it.", 0, 9, "exact")),
        snippet("b"),
        snippet("c"),
    ]
    second = [
        snippet("a", quote("As round 2 copied it.", 0, 9, "exact")),
        snippet("b", quote("B said y.", 0, 9, "exact"), cites=("a",)),
        snippet("c", cites=("a",)),
    ]
    quotes = trail_of([first, second], summary_ids=["a", "b", "c"])["quotes"]
    assert quotes == [
        quoted("a", 0, 9, "As round 2 copied it."),
        quoted("b", 0, 9, "B said y."),
    ]


WORDS = "A said x."
"""What document a says from 0 to 9."""


def copied(text: str, match: str, score: float | None = None) -> dict:
    """A quote of a's ``WORDS`` as ``text``, found by ``match``."""
    return quote(text, 0, 9, match, WORDS, score)


EXACT = copied(WORDS, "exact")
NORMALIZED = copied("a said X.", "normalized")
ELIDED = copied("A said … x.", "elided")
FUZZY_91 = copied("A said y.", "fuzzy", 91.0)
FUZZY_96 = copied("A says x.", "fuzzy", 96.0)


@pytest.mark.parametrize(
    ("round_1", "round_2", "kept"),
    [
        (EXACT, FUZZY_96, EXACT),
        (NORMALIZED, ELIDED, NORMALIZED),
        (ELIDED, FUZZY_96, ELIDED),
        (FUZZY_96, FUZZY_91, FUZZY_96),
        (FUZZY_96, NORMALIZED, NORMALIZED),
    ],
    ids=[
        "exact-over-fuzzy",
        "normalized-over-elided",
        "elided-over-fuzzy",
        "the-higher-fuzzy-score",
        "a-later-better-match",
    ],
)
def test_a_place_quoted_more_than_once_is_given_once_as_it_was_best_matched(
    round_1, round_2, kept
):
    first = [snippet("a", round_1), snippet("b"), snippet("c")]
    second = [snippet("a", round_2), snippet("b", cites=("a",)), snippet("c")]
    quotes = trail_of([first, second], summary_ids=["a", "b"])["quotes"]
    assert quotes == [{"document": "a", "name": NAMES["a"], **kept}]


def test_quotes_come_in_document_order_then_by_where_they_start():
    final = [
        snippet("a", quote("A, later.", 50, 60, "exact"), quote("A.", 0, 10, "exact")),
        snippet("b", quote("B.", 5, 8, "exact")),
        snippet("c", quote("C.", 1, 2, "exact")),
    ]
    quotes = trail_of([final], summary_ids=["c", "b", "a"])["quotes"]
    assert places(quotes) == [("a", 0, 10), ("a", 50, 60), ("b", 5, 8), ("c", 1, 2)]


def test_a_citation_of_a_note_skipped_in_the_last_round_leads_to_the_one_kept():
    skipped = SnippetRecord(
        id="b", summary="", quotes=[], provenance=[], skipped="schema error"
    )
    rounds = [
        [snippet("a"), snippet("b", quote("B said y.", 10, 19, "exact")), snippet("c")],
        [snippet("a", cites=("b",)), skipped, snippet("c")],
    ]
    trail = trail_of(
        rounds, summary_ids=["a", "b"], parse_ids=["c"], parses=[[], [], ["b"]]
    )
    assert ["answer", "summary:1:b", "cites"] in trail["edges"]
    assert ["parse:c", "summary:1:b", "cites"] in trail["edges"]
    assert "summary:2:b" not in trail["nodes"]
    assert places(trail["quotes"]) == [("b", 10, 19)]


# The answer cites every final summary: its trail quotes a twice, then b and c once.
QUOTED_ROUNDS = [
    [
        snippet(
            "a", quote("A, first.", 0, 9, "exact"), quote("A, then.", 10, 18, "exact")
        ),
        snippet("b", quote("B said y.", 0, 9, "exact")),
        snippet("c", quote("C said z.", 0, 9, "exact")),
    ]
]
EVERY_SUMMARY = ["a", "b", "c"]


def claim(*documents: str, quotes: tuple[tuple[str, int, int], ...] = ()) -> dict:
    """A claim of the answer naming ``documents``, and the ``quotes`` there by place."""
    return {
        "text": "The answer.",
        "start": 0,
        "end": 11,
        "documents": list(documents),
        "quotes": [{"document": d, "start": s, "end": e} for d, s, e in quotes],
    }


def test_the_evidence_trail_holds_each_quote_a_claim_cites_once_in_the_trails_order():
    trail = trail_of(QUOTED_ROUNDS, summary_ids=EVERY_SUMMARY)
    claims = [
        claim("c", quotes=(("c", 0, 9),)),
        claim("a", "c", quotes=(("a", 10, 18), ("c", 0, 9))),
    ]
    assert evidence_trail(trail, claims) == {
        "from": "claims",
        "documents": [named("a"), named("c")],
        "quotes": [quoted("a", 10, 18, "A, then."), quoted("c", 0, 9, "C said z.")],
    }


def test_at_the_document_level_the_evidence_trail_holds_the_documents_claims_name():
    trail = trail_of(
        [[snippet("a"), snippet("b"), snippet("c")]], summary_ids=EVERY_SUMMARY
    )
    claims = [claim("c"), claim("a", "c")]
    assert evidence_trail(trail, claims) == {
        "from": "claims",
        "documents": [named("a"), named("c")],
        "quotes": [],
    }


@pytest.mark.parametrize(
    "claims", [[], [claim(), claim()]], ids=["no-claims", "claims-citing-nothing"]
)
def test_with_no_claim_citing_anything_the_evidence_trail_is_the_whole_trail(claims):
    trail = trail_of(QUOTED_ROUNDS, summary_ids=EVERY_SUMMARY)
    assert evidence_trail(trail, claims) == {
        "from": "trail",
        "documents": trail["documents"],
        "quotes": trail["quotes"],
    }


def test_an_answer_citing_nothing_has_an_empty_evidence_trail_from_its_trail():
    provenance = with_claims(trail_of(CITED_ROUNDS), [])
    assert provenance["evidence_trail"] == {
        "from": "trail",
        "documents": [],
        "quotes": [],
    }


@pytest.mark.parametrize(
    ("rule", "documents"),
    [("through-summaries", ["a", "b"]), ("parses-light-documents", ["a", "b", "c"])],
)
def test_the_evidence_trail_follows_the_relevant_trail_under_any_rule(rule, documents):
    trail = trail_of(CITED_ROUNDS, summary_ids=["b"], parse_ids=["c"], rule=rule)
    whole = evidence_trail(trail, [])
    assert whole["documents"] == [named(id_) for id_ in documents]
    claimed = evidence_trail(trail, [claim("b", quotes=(("b", 10, 19),))])
    assert claimed["documents"] == [named("b")]
    assert places(claimed["quotes"]) == [("b", 10, 19)]


def test_an_answers_provenance_is_its_relevant_trail_its_claims_then_its_evidence():
    trail = trail_of(QUOTED_ROUNDS, summary_ids=EVERY_SUMMARY)
    claims = [claim("b", quotes=(("b", 0, 9),))]
    answered = with_claims(trail, claims)
    assert answered == {
        "relevant_trail": trail,
        "claims": claims,
        "evidence_trail": evidence_trail(trail, claims),
    }
    assert list(answered) == ["relevant_trail", "claims", "evidence_trail"]


def test_the_evidence_trails_entries_are_copies_of_the_relevant_trails():
    trail = trail_of(QUOTED_ROUNDS, summary_ids=EVERY_SUMMARY)
    evidence = evidence_trail(trail, [])
    evidence["quotes"][0]["text"] = "Changed."
    evidence["documents"][0]["name"] = "changed.txt"
    assert trail["quotes"][0]["text"] == "A, first."
    assert trail["documents"][0]["name"] == "a.txt"
