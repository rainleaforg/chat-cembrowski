from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import voyageai
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from chat_cembrowski.data.vectordb import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

from . import authors, llm, nih
from .nih import NIHResult
from .prompts import (
    CLASSIFIER_PROMPT,
    CONDENSE_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    HOSTILE_SYSTEM_PROMPT,
    IDENTITY_ANSWER,
    NIH_FALLBACK_SYSTEM_PROMPT,
    NIH_SYSTEM_PROMPT,
    PERSON_SYSTEM_PROMPT,
    SITE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)

# Works cited alongside an author's bio, most recent first (Phase 2: caps a
# "who is X" answer at a bio plus recent representative work, not a full
# citation dump of everything they've ever co-authored).
AUTHOR_MAX_WORKS = 8

# How many of a person's own "site" bio chunks to pull in for an author
# answer. Small on purpose — the corpus currently has one bio page per person
# (2 chunks), so this is generous headroom, not a real cap in practice.
AUTHOR_BIO_CHUNKS = 3

logger = logging.getLogger(__name__)

# A single prior turn: {"role": "user" | "assistant", "content": str}. Already
# the OpenAI message shape, so it drops straight into a `messages` list.
ChatMessage = dict[str, str]

# Token ceilings for the three kinds of call. These are caps, not reservations
# -- you are billed for what is generated, so headroom is free.
#
# The generosity is deliberate. Thinking models (Gemini 2.5+ and all of 3.x)
# count reasoning tokens against `max_tokens`, and when the budget runs out
# during reasoning the response comes back with empty content and no exception.
# The classifier used to cap at 5, which is ample for the single word it emits
# but is consumed entirely by reasoning on such a model -- yielding an empty
# label that silently defaulted every question to "cembrowski". 512 cannot be
# starved, and a one-word answer still only bills for a handful of tokens.
CLASSIFIER_MAX_TOKENS = 512
CONDENSE_MAX_TOKENS = 512
ANSWER_MAX_TOKENS = 2048

# Shown when the model returns an empty completion. Rare, but a blank string is
# worse than saying so -- it reads as the assistant ignoring the question.
EMPTY_ANSWER_FALLBACK = (
    "I wasn't able to generate an answer for that. Please try rephrasing your "
    "question."
)

# Minimum top-hit Qdrant cosine score to trust the Cembrowski corpus for a
# question classified as Cembrowski-specific. Below this, retrieval is too
# weak to be reliable, so the question is routed to NIH instead.
#
# Measured against the live collection (voyage-multimodal-3.5, page-based
# chunks). On-topic questions score 0.32-0.65; general medical questions that
# belong on the NIH path top out at 0.24. Short on-topic questions sit at the
# bottom of that range -- "tell me about gem 4000" scores 0.33 -- so the
# previous value of 0.4 cut through the middle of the on-topic band and sent
# terse but valid corpus questions to NIH. 0.30 sits in the empty gap between
# the two populations. Re-measure before changing the embedding model or the
# chunking strategy, since both shift the score distribution.
SCORE_THRESHOLD = 0.30

# The labels CLASSIFIER_PROMPT is allowed to emit. "cembrowski" is the default
# and so is never matched for explicitly -- see `_resolve_label`.
KNOWN_LABELS = ("hostile", "site", "medical", "general")

_LABEL_WORD_RE = re.compile(r"[a-z]+")


def _resolve_label(label: str) -> str:
    """
    Map a raw classifier completion onto one of KNOWN_LABELS, or "cembrowski".

    CLASSIFIER_PROMPT asks for exactly one word, but models wrap it in prose,
    so this cannot just compare strings -- nor can it scan for a substring,
    which is what it used to do. "This is not hostile, it is a general
    question." contains "hostile", and hostile was checked first, so a benign
    question got the refusal reply. That is the worst misroute the system can
    produce, and prose that names a label while rejecting it is exactly the
    shape a chatty model emits.

    So: an exact label wins outright; otherwise the completion is tokenised
    and a label is only honoured when it is the *only* one named. Anything
    ambiguous falls back to "cembrowski", the same safe default used for an
    API failure -- the score check in `_route` still catches an off-topic
    question from there.
    """
    exact = label.strip().strip(".!?,:;'\"")
    if exact in KNOWN_LABELS:
        return exact

    named = {w for w in _LABEL_WORD_RE.findall(label) if w in KNOWN_LABELS}
    if len(named) == 1:
        return named.pop()

    if named:
        logger.warning(
            "Classifier completion names %d labels (%s); defaulting to "
            "'cembrowski'. Completion: %r",
            len(named),
            sorted(named),
            label[:200],
        )
    return "cembrowski"


@dataclass
class RetrievedChunk:
    score: float
    source_type: str        # "paper", "document", or "image" (chunk_category/rendering)
    title: str
    text: str
    chunk_index: int
    # "poster" | "paper" | "site" | "document" — the citability tier (see
    # kind stamping in ingestion). Falls back to "poster" if site_path else
    # "document" for points that predate the kind payload field.
    kind: str = "document"
    # Paper-specific
    paper_id: str | None = None
    publication: str | None = None
    year: int | None = None
    page_label: str | None = None
    authors: list[str] | None = None
    page: int | None = None
    # Stamped at ingest time — ties a poster/paper/site chunk to its page on
    # the website. Absent on internal documents and on chunks that predate
    # the stamp, so always optional.
    site_path: str | None = None    # e.g. "/presentation/gem-4000-cartridge-instability"
    poster_id: str | None = None    # e.g. "pos-gem-4000-cartridge-instability"
    # Document-specific
    file_type: str | None = None
    # Image-specific
    caption: str | None = None
    image_type: str | None = None


@dataclass
class SourceRef:
    """
    One entry in a numbered source list, aligned by position with the
    `SOURCE {i}` blocks handed to the model. `index` is 1-based and is what the
    model cites as `[index]`.

    `kind` is "poster" or "paper" (Cembrowski research with a page on the
    site), "site" (a public site/product page), "document" (an internal
    note/code file, no public page), or "nih" (MedlinePlus/PubMed). `url` is
    a site-relative path for poster/paper/site, an absolute URL for NIH, and
    None for internal documents.
    """
    index: int
    kind: str
    title: str
    authors: list[str] = field(default_factory=list)
    url: str | None = None
    page: int | None = None
    publication: str | None = None
    year: int | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "page": self.page,
            "publication": self.publication,
            "year": self.year,
        }


@dataclass
class QueryResult:
    """Answer plus the ordered sources it was grounded in and the route taken."""
    answer: str
    route: str              # "author", "site", "identity", "cembrowski", "nih", "general", or "hostile"
    sources: list[SourceRef] = field(default_factory=list)


@dataclass
class RouteDecision:
    """
    Where a question was routed, plus everything gathered on the way there.

    Routing is separated from answering so it can be measured on its own: it
    costs one cheap classifier call and a Qdrant search, while generating an
    answer costs a full synthesis call. `scripts/eval_routing.py` exercises
    `QueryEngine._route` directly, which makes a whole-corpus routing check
    cheap enough to run on every change rather than once a quarter.
    """
    route: str                      # "author", "site", "identity", "cembrowski", "nih", "general", or "hostile"
    search_question: str            # the condensed, standalone form used for retrieval
    label: str | None = None        # raw classifier label; None when an author match short-circuited it
    matched_author: str | None = None
    chunks: list[RetrievedChunk] = field(default_factory=list)
    top_score: float | None = None  # None when retrieval never ran or returned nothing


class QueryEngine:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        llm_client: OpenAI | None = None,
        voyage_client: voyageai.Client | None = None, # type: ignore
        top_k: int = 10,
        collection_name: str = COLLECTION_NAME,
        llm_config: llm.LLMConfig | None = None,
        openai_client: OpenAI | None = None,
    ) -> None:
        """
        Args:
            llm_client:      Chat client, built from the environment when omitted
                             (see `retrieval/llm.py`). It is an OpenAI-SDK client
                             either way -- OpenRouter speaks the same schema, so
                             the provider difference is a base URL, not a type.
            llm_config:      Provider/model selection; read from env when omitted.
            openai_client:   Deprecated alias for `llm_client`. Kept because the
                             website backend constructs QueryEngine by keyword
                             and would break on deploy without it.
        """
        if voyage_client is None:
            raise ValueError("voyage_client is required.")

        if openai_client is not None:
            logger.warning(
                "QueryEngine parameter 'openai_client' is deprecated and will be "
                "removed in a future version. Use 'llm_client' instead."
            )

        self.qdrant = qdrant_client
        self.llm_config = llm_config or llm.get_config()
        self.llm = llm_client or openai_client or llm.get_llm_client(self.llm_config)
        self.voyage = voyage_client
        self.top_k = top_k
        self.collection_name = collection_name

    def query(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> str:
        """End-to-end RAG query pipeline. See `query_structured` for details."""
        return self.query_structured(question, history).answer

    def query_with_route(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> tuple[str, str]:
        """
        End-to-end RAG query pipeline, returning `(answer, route)`.

        Thin wrapper over `query_structured` for callers that want the route
        label but not the source list. See `query_structured` for the steps.
        """
        result = self.query_structured(question, history)
        return result.answer, result.route

    def query_structured(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> QueryResult:
        """
        End-to-end RAG query pipeline with source routing and structured sources.

        `_route` picks the route (see there for how); this method generates the
        answer for whichever one it chose:

        - "author" — a fuzzy hit on a name known to the corpus. Answers from
          that person's bio plus recent works via metadata filters rather
          than vector search, since author names are never embedded.
        - "site" — classified "site" and retrieval cleared SCORE_THRESHOLD
          against `kind IN ("site", "document")`. Answers from the site/
          internal knowledge chunks `_route` already fetched.
        - "identity" — classified "site" but retrieval came back completely
          empty. Answers with the static IDENTITY_ANSWER, no retrieval.
        - "cembrowski" — classified "cembrowski" and retrieval cleared
          SCORE_THRESHOLD against the unfiltered corpus.
        - "nih" — classified "medical". Answers from MedlinePlus + PubMed.
        - "general" — classified "general", or any of the routes above found
          nothing above SCORE_THRESHOLD. Answers from model knowledge, no
          retrieval, facts only.
        - "hostile" — classified "hostile". A short, civil, non-defensive
          reply, no retrieval.

        The raw `question` (plus `history`) is what's sent to the model for
        answer generation, so phrasing and tone stay natural; only retrieval
        and routing operate on the condensed standalone version.

        Returns a QueryResult whose `sources` are 1-based and aligned by
        position with the `SOURCE {i}` blocks the model was shown, so a `[i]`
        citation in the answer maps straight to `sources[i - 1]`.
        """
        history = history or []
        decision = self._route(question, history)
        return self.answer_from_decision(question, decision, history)

    def answer_from_decision(
        self,
        question: str,
        decision: RouteDecision,
        history: list[ChatMessage] | None = None,
    ) -> QueryResult:
        """
        Generate an answer from a precomputed RouteDecision, without re-routing.

        Used by `query_structured` (which internally calls `_route` and then
        this) and by `scripts/eval_routing.py` to avoid routing twice — the
        eval script measures the route decision separately and then needs the
        answer generated from that exact decision, not a fresh re-route.

        Args:
            question: The raw user question (used for answer generation).
            decision: A RouteDecision already computed by `_route`.
            history: Prior conversation turns (optional).

        Returns a QueryResult whose `route` matches `decision.route` and whose
        `sources` align by position with the `SOURCE {i}` blocks the model was
        shown.
        """
        history = history or []

        if decision.route == "author":
            answer, sources = self._answer_author(
                question, decision.matched_author or "", history
            )
        elif decision.route == "identity":
            return QueryResult(answer=IDENTITY_ANSWER, route="identity", sources=[])
        elif decision.route == "site":
            answer, sources = self._answer_site(question, decision.chunks, history)
        elif decision.route == "cembrowski":
            answer, sources = self._answer_cembrowski(
                question, decision.chunks, history
            )
        elif decision.route == "nih":
            answer, sources = self._answer_nih(
                question, history, decision.search_question
            )
        elif decision.route == "hostile":
            answer, sources = self._answer_hostile(question, history), []
        else:
            answer, sources = self._answer_general(question, history), []

        return QueryResult(answer=answer, route=decision.route, sources=sources)

    def _route(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> RouteDecision:
        """
        Decide which source a question should be answered from, without
        generating anything. See `query_structured` for what each route means.

        Split out from `query_structured` so routing can be evaluated on its
        own -- see RouteDecision.

        Classification runs before author matching for "hostile" and
        "medical" (unlike the old routing, where an author-name hit
        short-circuited classification entirely -- which is how a message
        like "George Cembrowski is a fraud" used to bypass every guard and
        land straight on the corpus route: it still can't, since hostile is
        checked first here). For every other label, author-match still runs
        before the label is trusted -- a question like "Tell me about Mark
        Cervinski's work" can land on "general" from the classifier alone
        (nothing about the wording says "corpus"), and losing the author
        route for it would be its own regression. Measured: this is what it
        takes to keep author routing at 4/4 while still closing the hostile
        bypass.
        """
        history = history or []
        search_question = (
            self._condense_question(question, history) if history else question
        )

        label = self._classify(search_question)

        if label == "hostile":
            return RouteDecision(
                route="hostile", search_question=search_question, label=label
            )

        if label == "medical":
            return RouteDecision(
                route="nih", search_question=search_question, label=label
            )

        # label is "general", "site", or "cembrowski" here -- all three are
        # questions an author-name hit should be allowed to short-circuit.
        matched_author = self._match_author(search_question)
        if matched_author:
            return RouteDecision(
                route="author",
                search_question=search_question,
                label=label,
                matched_author=matched_author,
            )

        # No safety net in this direction, by design: a "general" label skips
        # retrieval entirely. "cembrowski" and "site" both still have to earn
        # their route by clearing SCORE_THRESHOLD below.
        if label == "general":
            return RouteDecision(
                route="general", search_question=search_question, label=label
            )

        if label == "site":
            chunks = self._search(
                self._embed_query(search_question), kind_filter=("site", "document")
            )
            if not chunks:
                return RouteDecision(
                    route="identity", search_question=search_question, label=label
                )
            top_score = chunks[0].score
            route = "site" if top_score >= SCORE_THRESHOLD else "general"
            return RouteDecision(
                route=route,
                search_question=search_question,
                label=label,
                chunks=chunks,
                top_score=top_score,
            )

        # label == "cembrowski": unfiltered search across the whole corpus.
        # Below SCORE_THRESHOLD falls through to "general" rather than NIH --
        # this is the fix for the old "couldn't find NIH information" dead
        # end on questions that were never medical to begin with.
        chunks = self._search(self._embed_query(search_question))
        top_score = chunks[0].score if chunks else None
        route = (
            "cembrowski"
            if top_score is not None and top_score >= SCORE_THRESHOLD
            else "general"
        )

        return RouteDecision(
            route=route,
            search_question=search_question,
            label=label,
            chunks=chunks,
            top_score=top_score,
        )

    def _condense_question(
        self, question: str, history: list[ChatMessage]
    ) -> str:
        """
        Rewrite a follow-up question as a standalone question using the prior
        conversation (cheap LLM call). Used only to drive classification,
        embedding, and search — the original `question` is still what gets
        answered.

        Falls back to the raw question on any API failure or empty response,
        so a condensation hiccup degrades to today's stateless behavior rather
        than breaking the request.
        """
        try:
            response = self.llm.chat.completions.create(
                model=self.llm_config.classifier_model,
                temperature=0,
                max_tokens=CONDENSE_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": CONDENSE_PROMPT},
                    *history, # type: ignore
                    {
                        "role": "user",
                        "content": f"Follow-up question: {question}\n\nStandalone question:",
                    },
                ],
                **llm.completion_extras(
                    self.llm_config, effort=llm.CLASSIFIER_REASONING_EFFORT
                ),
            )
            standalone = (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning(f"Condense call failed ({e}); using the raw question.")
            return question

        if not standalone:
            logger.warning(
                "Condense returned no content "
                f"(model={self.llm_config.classifier_model}); using the raw question."
            )
            return question

        return standalone

    def _classify(self, question: str) -> str:
        """
        Classify a question as "cembrowski", "site", "medical", "general", or
        "hostile" via a cheap LLM call.

        Defaults to "cembrowski" on any API failure or an unrecognized
        label — the retrieval-score check in `_route` still catches
        weak/off-topic matches and falls through to "general", so failing
        open here doesn't strand the reader on the corpus route. (Never
        defaults to "hostile", "site", or "general": those skip or filter
        retrieval, so a wrong guess in that direction would strand the
        answer on the wrong footing rather than just costing a redundant
        Qdrant search.)

        An empty completion is logged rather than quietly falling through the
        label checks. It looks identical to a real "cembrowski" classification
        from the outside, so without the warning a starved thinking model
        (see CLASSIFIER_MAX_TOKENS) would route every question to the corpus
        and give no sign of it.
        """
        try:
            response = self.llm.chat.completions.create(
                model=self.llm_config.classifier_model,
                temperature=0,
                max_tokens=CLASSIFIER_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": CLASSIFIER_PROMPT},
                    {"role": "user", "content": question},
                ],
                **llm.completion_extras(
                    self.llm_config, effort=llm.CLASSIFIER_REASONING_EFFORT
                ),
            )
        except Exception as e:
            logger.warning(f"Classifier call failed ({e}); defaulting to 'cembrowski'.")
            return "cembrowski"

        choice = response.choices[0]
        label = (choice.message.content or "").strip().lower()

        if not label:
            logger.warning(
                "Classifier returned no content "
                f"(model={self.llm_config.classifier_model}, "
                f"finish_reason={choice.finish_reason!r}); defaulting to "
                "'cembrowski'. On a thinking model this means reasoning consumed "
                "the whole max_tokens budget — raise CLASSIFIER_MAX_TOKENS or "
                "lower LLM_REASONING_EFFORT."
            )
            return "cembrowski"

        return _resolve_label(label)

    def _answer_cembrowski(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[ChatMessage] | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> tuple[str, list[SourceRef]]:
        """
        Build a grounded prompt from retrieved chunks and generate an answer,
        returning it alongside the numbered sources it was shown.

        Shared by the "cembrowski", "site", and "author" routes -- they
        differ only in what was retrieved and which system prompt frames the
        answer (see `_answer_site`, `_answer_author`).

        Chunks split by `kind`: `citable` (poster/paper/site with a
        site_path -- a real, clickable link) become numbered SOURCE blocks
        and SourceRefs. `kind == "document"` chunks are internal and are
        never named, even in the background section -- see
        `_build_background_context`. Anything else (an unlinked poster/paper,
        e.g. the textbook) still informs the answer as titled background,
        just without a citation number.
        """
        citable_chunks = [
            c for c in chunks if c.kind in ("poster", "paper", "site") and c.site_path
        ]
        background_chunks = [c for c in chunks if c not in citable_chunks]

        context = self._build_context(citable_chunks)
        background = self._build_background_context(background_chunks)
        sources = self._cembrowski_sources(citable_chunks)

        if context:
            user_content = f"""
Question:
{question}

Context:
{context}
"""
        else:
            # Nothing retrieved has a reader-facing link, so there are no
            # numbered SOURCE blocks at all. Saying so explicitly matters: an
            # empty "Context:" heading followed by a rich background section
            # reads as an oversight, and the model fills the gap by inventing
            # [1], [2], ... that resolve to nothing. This is the common case for
            # the textbook, which is unlinked and is most of the corpus.
            user_content = f"""
Question:
{question}

Context:
(none — no citable sources were retrieved for this question)

There are NO numbered sources for this question. Do not write any bracket
citation such as [1]; there is nothing for it to point at. Answer using the
background material below.
"""
        if background:
            user_content += f"""
Additional background (not citable — do not cite these with a bracket number, use only to inform your answer):
{background}
"""

        return self._generate(system_prompt, user_content, history), sources

    def _answer_site(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[ChatMessage] | None = None,
    ) -> tuple[str, list[SourceRef]]:
        """The "site" route: same retrieval/context shape as `_answer_cembrowski`,
        framed by SITE_SYSTEM_PROMPT instead."""
        return self._answer_cembrowski(
            question, chunks, history, system_prompt=SITE_SYSTEM_PROMPT
        )

    def _answer_general(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> str:
        """The "general" (open-domain) route: no retrieval, no citations,
        answered from the model's own knowledge with a staleness caveat for
        anything time-sensitive -- see GENERAL_SYSTEM_PROMPT."""
        return self._generate(GENERAL_SYSTEM_PROMPT, question, history)

    def _answer_hostile(
        self, question: str, history: list[ChatMessage] | None = None
    ) -> str:
        """The "hostile" route: no retrieval, a short civil reply that
        doesn't repeat or engage with the hostility -- see HOSTILE_SYSTEM_PROMPT."""
        return self._generate(HOSTILE_SYSTEM_PROMPT, question, history)

    def _generate(
        self,
        system_prompt: str,
        user_content: str,
        history: list[ChatMessage] | None = None,
    ) -> str:
        """
        Run one synthesis call and return the answer text.

        Shared by both answering paths, which differ only in their system prompt
        and the context block they assemble. API failures propagate to the
        caller; only an empty completion is handled here, since that comes back
        as a success and would otherwise reach the reader as a blank answer.
        """
        response = self.llm.chat.completions.create(
            model=self.llm_config.chat_model,
            temperature=0.1,
            max_tokens=ANSWER_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                *(history or []), # type: ignore
                {"role": "user", "content": user_content},
            ],
            **llm.completion_extras(self.llm_config),
        )

        choice = response.choices[0]
        answer = (choice.message.content or "").strip()

        if not answer:
            logger.error(
                "Synthesis returned no content "
                f"(model={self.llm_config.chat_model}, "
                f"finish_reason={choice.finish_reason!r})."
            )
            return EMPTY_ANSWER_FALLBACK

        return answer

    def _build_background_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Unnumbered context from non-citable chunks — informs the answer but
        is never assigned a SOURCE number, so it can never be cited.

        `kind == "document"` chunks are internal (design notes, code,
        META.md) and get a neutral header with no title: the model cannot
        name or paraphrase a filename it was never shown, which is the
        structural half of keeping internal documents out of answers (the
        prompt rule in BASE_RULES is the backup). Everything else here is an
        unlinked poster/paper/site chunk (e.g. the textbook) -- not secret,
        just not yet clickable -- so its title is shown.
        """
        if not chunks:
            return ""

        sections = []
        for chunk in chunks:
            if chunk.kind == "document":
                header_lines = ["Internal reference (do not name or cite)"]
            else:
                header_lines = [f"Title: {chunk.title}"]
                if chunk.file_type:
                    header_lines.append(f"Type: {chunk.file_type}")
            if chunk.publication:
                header_lines.append(f"Publication: {chunk.publication}")
            if chunk.year:
                header_lines.append(f"Year: {chunk.year}")
            header = "\n".join(header_lines)
            body = chunk.caption or chunk.text
            sections.append(f"{header}\n\nContent:\n{body}")

        return "\n\n----\n\n".join(sections)

    def _match_author(self, question: str) -> str | None:
        """Fuzzy-match `question` against every author name known to the corpus."""
        try:
            known = authors.get_known_authors(self.qdrant, self.collection_name)
            return authors.match_author(question, known)
        except Exception:
            return None

    def _answer_author(
        self,
        question: str,
        author_name: str,
        history: list[ChatMessage] | None = None,
    ) -> tuple[str, list[SourceRef]]:
        """
        Answer a "who is X" question by grounding it in the person's own
        "site" bio first, then a capped set of their recent work.

        The bio comes from a vector search on the person's name, filtered to
        `kind="site"` -- a payload filter can't find it, since author names
        are never embedded onto bio chunks the way they are onto papers.
        Every distinct work crediting `author_name` is fetched exhaustively
        (a payload filter, not a top-k search), collapsed to one
        representative chunk per work (lowest chunk_index, i.e. first page),
        then capped to the AUTHOR_MAX_WORKS most recent by year -- a bio plus
        recent representative work, not a full citation dump. Bio chunks are
        prepended so they win ties for citation order.
        """
        # Title-gated, because a filtered vector search always returns its top
        # hits: `_match_author` matches every author in the corpus, but only
        # George and Jenna have bio pages, so for anyone else this search
        # returns three unrelated site chunks -- and those carry a site_path,
        # so `_answer_cembrowski` promotes them to numbered SOURCE blocks and
        # cites e.g. the privacy page under "who is Jialin Qiu".
        #
        # SCORE_THRESHOLD does not separate these; measured against the live
        # collection, the unrelated hits score 0.31-0.39, above it. See
        # authors.title_names_author for the numbers and the reasoning.
        bio_chunks = [
            c
            for c in self._search(
                self._embed_query(author_name),
                kind_filter=("site",),
                limit=AUTHOR_BIO_CHUNKS,
            )
            if authors.title_names_author(c.title, author_name)
        ]

        points = authors.fetch_chunks_by_author(
            self.qdrant, self.collection_name, author_name
        )

        best_by_work: dict[str, RetrievedChunk] = {}
        for point in points:
            chunk = self._point_to_chunk(point)
            key = chunk.paper_id or chunk.title
            current = best_by_work.get(key)
            if current is None or chunk.chunk_index < current.chunk_index:
                best_by_work[key] = chunk

        works = sorted(
            best_by_work.values(), key=lambda c: c.year or 0, reverse=True
        )[:AUTHOR_MAX_WORKS]

        return self._answer_cembrowski(
            question, bio_chunks + works, history, system_prompt=PERSON_SYSTEM_PROMPT
        )

    def _cembrowski_sources(
        self, chunks: list[RetrievedChunk]
    ) -> list[SourceRef]:
        """
        One SourceRef per chunk, numbered from 1 in the same order as
        `_build_context` writes the `SOURCE {i}` blocks — so `[i]` in the answer
        maps to `sources[i - 1]`. Callers only pass chunks that already cleared
        the citable check (kind in poster/paper/site with a site_path), so
        `chunk.kind` is used directly rather than re-derived here.
        """
        sources: list[SourceRef] = []
        for i, chunk in enumerate(chunks, start=1):
            sources.append(
                SourceRef(
                    index=i,
                    kind=chunk.kind,
                    title=chunk.title,
                    authors=chunk.authors or [],
                    url=chunk.site_path,
                    page=chunk.page,
                    publication=chunk.publication,
                    year=chunk.year,
                )
            )
        return sources

    def _search_nih(self, question: str) -> list[NIHResult]:
        return nih.search_nih(question)

    def _build_nih_context(self, results: list[NIHResult]) -> str:
        """Build retrieval context for the LLM from NIH (MedlinePlus/PubMed) results."""
        sections = []

        for i, result in enumerate(results, start=1):
            header_lines = [f"Source: {result.source}", f"Title: {result.title}"]
            if result.journal:
                header_lines.append(f"Journal: {result.journal}")
            if result.year:
                header_lines.append(f"Year: {result.year}")
            header_lines.append(f"URL: {result.url}")
            header = "\n".join(header_lines)
            body = result.summary or "(no summary available)"

            sections.append(f"SOURCE {i}\n\n{header}\n\nContent:\n{body}")

        return "\n\n====================\n\n".join(sections)

    def _answer_nih(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
        search_question: str | None = None,
    ) -> tuple[str, list[SourceRef]]:
        """
        Answer a general medical question from NIH (MedlinePlus/PubMed) search
        results, returning the answer alongside the numbered sources shown.

        The question is reduced to keywords before searching -- sending a
        whole sentence to a keyword search is why so many questions used to
        return nothing (see nih.extract_search_terms).
        """
        search_terms = nih.extract_search_terms(
            search_question or question, self.llm, self.llm_config
        )
        results = self._search_nih(search_terms)

        if not results:
            # No dead end: fall through to a medical-safety answer from
            # general knowledge rather than refusing outright.
            return self._generate(NIH_FALLBACK_SYSTEM_PROMPT, question, history), []

        context = self._build_nih_context(results)
        sources = [
            SourceRef(
                index=i,
                kind="nih",
                # The journal, when there is one, moves into the title so the
                # reader still sees it -- `publication` is always the service
                # name (MedlinePlus/PubMed) now, not sometimes overwritten by
                # the journal, so the frontend badge can show which service
                # actually answered instead of always reading "NIH".
                title=f"{result.title} ({result.journal})" if result.journal else result.title,
                url=result.url,
                publication=result.source,
                year=int(result.year) if result.year and result.year.isdigit() else None,
            )
            for i, result in enumerate(results, start=1)
        ]

        user_content = f"""
Question:
{question}

Context:
{context}
"""

        return self._generate(NIH_SYSTEM_PROMPT, user_content, history), sources

    def _embed_query(self, question: str) -> list[float]:
        result = self.voyage.multimodal_embed(
            inputs=[[question]],
            model=EMBEDDING_MODEL,
            input_type="query",
        )
        return result.embeddings[0]

    def _search(
        self,
        query_embedding: list[float],
        kind_filter: tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Vector search, optionally restricted to a payload `kind` allowlist —
        the "site" route filters to `("site", "document")` so internal docs
        stay reachable there without being surfaced on unrelated corpus
        questions. Filtering on `kind` requires it to be indexed
        (`ensure_collection`) or Qdrant 400s.
        """
        query_filter = None
        if kind_filter:
            query_filter = Filter(
                must=[FieldCondition(key="kind", match=MatchAny(any=list(kind_filter)))]
            )

        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit or self.top_k,
            with_payload=True,
        ).points

        return [self._point_to_chunk(point, score=point.score) for point in results]

    def _point_to_chunk(self, point, score: float = 0.0) -> RetrievedChunk:
        """Convert one Qdrant point into a RetrievedChunk, routed by payload shape."""
        payload = point.payload or {}
        chunk_category = payload.get("chunk_category", "text")
        source_type = payload.get("source_type", "paper")
        # Backwards compatibility for points that predate the kind payload
        # field (see scripts/backfill_kind.py): a linked chunk is presumed a
        # poster, an unlinked one an internal document -- the conservative
        # direction, since it only ever hides a title rather than exposing one.
        kind = payload.get("kind") or ("poster" if payload.get("site_path") else "document")

        if chunk_category == "image":
            return RetrievedChunk(
                score=score,
                source_type="image",
                kind=kind,
                title=payload.get("title", "Unknown Title"),
                text=payload.get("text", ""),
                chunk_index=payload.get("chunk_index", -1),
                paper_id=payload.get("paper_id"),
                publication=payload.get("publication"),
                year=payload.get("year"),
                page_label=payload.get("page_label"),
                authors=payload.get("authors"),
                page=payload.get("page"),
                site_path=payload.get("site_path"),
                poster_id=payload.get("poster_id"),
                caption=payload.get("caption"),
                image_type=payload.get("image_type"),
            )
        elif source_type == "document":
            # Covers both data/docs/ documents and data/knowledge/ markdown
            # (site or document kind) -- both are chunked via chunk_document,
            # so both land here regardless of what `kind` they carry.
            return RetrievedChunk(
                score=score,
                source_type="document",
                kind=kind,
                title=payload.get("title", "Unknown Document"),
                text=payload.get("text", ""),
                chunk_index=payload.get("chunk_index", -1),
                file_type=payload.get("file_type"),
                site_path=payload.get("site_path"),
            )
        else:
            return RetrievedChunk(
                score=score,
                source_type="paper",
                kind=kind,
                title=payload.get("title", "Unknown Title"),
                text=payload.get("text", ""),
                chunk_index=payload.get("chunk_index", -1),
                paper_id=payload.get("paper_id"),
                publication=payload.get("publication"),
                year=payload.get("year"),
                page_label=payload.get("page_label"),
                authors=payload.get("authors"),
                page=payload.get("page"),
                site_path=payload.get("site_path"),
                poster_id=payload.get("poster_id"),
            )

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Build retrieval context for the LLM, rendering papers and documents differently."""
        sections = []

        for i, chunk in enumerate(chunks, start=1):
            if chunk.source_type == "document":
                header_lines = [f"Title: {chunk.title}"]
                if chunk.file_type:
                    header_lines.append(f"Type: {chunk.file_type}")
                header = "\n".join(header_lines)
                body = chunk.text

            elif chunk.source_type == "image":
                header_lines = [f"Title: {chunk.title}"]
                if chunk.publication:
                    header_lines.append(f"Publication: {chunk.publication}")
                if chunk.year:
                    header_lines.append(f"Year: {chunk.year}")
                if chunk.page_label:
                    header_lines.append(f"Pages: {chunk.page_label}")
                if chunk.image_type:
                    header_lines.append(f"Image Type: {chunk.image_type}")
                header = "\n".join(header_lines)
                body = chunk.caption or chunk.text

            else:
                header_lines = [f"Title: {chunk.title}"]
                if chunk.publication:
                    header_lines.append(f"Publication: {chunk.publication}")
                if chunk.year:
                    header_lines.append(f"Year: {chunk.year}")
                if chunk.page_label:
                    header_lines.append(f"Pages: {chunk.page_label}")
                header = "\n".join(header_lines)
                body = chunk.text

            sections.append(
                f"SOURCE {i}\n\n{header}\n\nContent:\n{body}"
            )

        return "\n\n====================\n\n".join(sections)