"""r3trace — just-in-time reasoning over a collection of documents.

Ask a question whose evidence is spread across many documents; the apparatus for
answering it is built *per question*, at the moment you ask:

    >>> from r3trace import r3trace
    >>> result = r3trace.run("Which supplier missed the most delivery windows?", docs)
    >>> print(result)                 # the answer
    >>> result.provenance                    # its relevant trail, claims and evidence trail
    >>> result.relevant_context              # what each document contributed
    >>> result.structured_context            # the structured parse behind it

Three moves, in order:

1. **surfacing relevance** — every document is read against the question and written up
   as a **relevance summary**; then re-read in light of the *other* documents'
   summaries, until it is pruned for saying nothing. Together they are the **relevant context**.
   Relevance is not a property a document has; it is a relation between the document,
   the question, and the rest of the collection, so it cannot be settled from a
   document in isolation.
2. **structuring** — a Pydantic schema is proposed for exactly what this question needs,
   then every document not pruned is parsed into instances of it.
3. **reasoning** — one call answers over the structured records and the relevance
   summaries together; or, when the run asks for it or that prompt does not fit the
   model's window, the merged parse is loaded into a sandboxed Python runtime and an
   agent reasons there with code. Either way the answer cites the documents it used.

The whole collection is never placed in a single prompt: each document is read on its
own, in parts when it is too long for the model's window, and information crosses
document boundaries through the relevance summaries.
"""

from importlib import metadata as _metadata

from r3trace import provenance, r3trace, settings
from r3trace.config import RunConfig, load_config
from r3trace.documents import read_documents
from r3trace.pipeline import Answer, Skip, run_pipeline
from r3trace.prompts import load_prompt
from r3trace.r3trace import ask, resume, run, trace
from r3trace.runs import StageRun, TaskLogger
from r3trace.runtime.codeact import CodeActResult, CodeActTurn
from r3trace.runtime.llm import (
    Cancelled,
    EmptyReply,
    UnfitReply,
    UnusableReply,
    litellm_chat_completion,
    litellm_chat_completion_full,
)
from r3trace.stages.reasoning import ReasoningResult, reason
from r3trace.stages.relevance import (
    RelevantContext,
    relevance_snippet,
    render_relevance,
    surface_relevance,
)
from r3trace.stages.structuring.parsing import (
    ParseResult,
    SchemaError,
    check_schema,
    parse_documents,
    parse_one_document,
)
from r3trace.stages.structuring.schema import (
    ProposalAttempt,
    ProposalResult,
    propose_schema,
)

try:
    __version__ = _metadata.version("r3trace")
except _metadata.PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0+unknown"

__all__ = [
    "Answer",
    "Cancelled",
    "CodeActResult",
    "CodeActTurn",
    "EmptyReply",
    "ParseResult",
    "ProposalAttempt",
    "ProposalResult",
    "ReasoningResult",
    "RelevantContext",
    "RunConfig",
    "SchemaError",
    "Skip",
    "StageRun",
    "TaskLogger",
    "UnfitReply",
    "UnusableReply",
    "ask",
    "check_schema",
    "litellm_chat_completion",
    "litellm_chat_completion_full",
    "load_config",
    "load_prompt",
    "parse_documents",
    "parse_one_document",
    "propose_schema",
    "provenance",
    "r3trace",
    "read_documents",
    "reason",
    "relevance_snippet",
    "render_relevance",
    "resume",
    "run",
    "run_pipeline",
    "settings",
    "surface_relevance",
    "trace",
]
