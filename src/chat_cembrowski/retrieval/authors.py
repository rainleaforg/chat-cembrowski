"""
Author-name recognition.

Author names are never part of what gets embedded (see chunker.py's
_build_*_embed_text — only title/year/publication are), so a vector search
has nothing to match "who is X" against. This module answers that question
with a metadata lookup instead: scroll every distinct author name out of
Qdrant, fuzzy-match the question against that list, and (on a hit) fetch
every chunk crediting that person via a payload filter — exhaustive across
the whole corpus, not just whatever lands in a top-k similarity search.
"""

from __future__ import annotations

import re
import time
from typing import Iterator

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from rapidfuzz import fuzz, process, utils

# Long enough to avoid re-scrolling the collection on every question; short
# enough that a poster ingested straight into Qdrant (no backend redeploy —
# see the release flow notes) is recognized within the hour.
CACHE_TTL_SECONDS = 60 * 60

# rapidfuzz token_set_ratio, 0-100. Calibrated against real questions
# mentioning a real corpus author by name (e.g. "Tell me about R. Neill
# Carey's work" scores ~87 against "R. Neill Carey, PhD") while still
# rejecting every unrelated question tried against it, including ones
# containing a standalone "r" (this corpus is statistics-heavy, so "What is
# the r value for this comparison?" is a real, plausible question — see
# get_known_authors/match_author's docstrings for why the token-level check
# below is what actually carries the weight of rejecting those, not this
# threshold alone).
MATCH_THRESHOLD = 85

# A real name always has at least one token longer than a bare initial (e.g.
# "Mei" in "J. Mei", "Xu" in "E. Xu"). This exists because of a real incident:
# one paper's metadata got mangled into a stray "R." author entry (the "R."
# that should have prefixed "Neill Carey, PhD", split off on its own).
# Punctuation-stripping for fuzzy matching reduced "R." to the single
# character "r", which is a near-guaranteed substring/token of *any* question
# of reasonable length — so that one garbled entry silently won the
# author-match check for nearly every question asked, of any topic,
# hijacking retrieval every time (see incident notes / PR discussion).
# Filtering degenerate names out of the candidate pool here — rather than
# only raising the threshold — means a future bad extraction can't do this
# again no matter how the scoring is tuned.
MIN_SIGNIFICANT_TOKEN_LENGTH = 3

_cache: dict[str, tuple[float, list[str]]] = {}


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[\s,.\-]+", name) if t]


def _has_significant_token(name: str) -> bool:
    """True if `name` has at least one token that isn't a bare initial."""
    return any(len(t) >= MIN_SIGNIFICANT_TOKEN_LENGTH for t in _tokens(name))


def get_known_authors(client: QdrantClient, collection_name: str) -> list[str]:
    """Every distinct, non-degenerate author name in the corpus, cached for
    CACHE_TTL_SECONDS. A name with no token longer than a bare initial (e.g.
    a stray "R.") is excluded — see MIN_SIGNIFICANT_TOKEN_LENGTH. This only
    affects which names are searchable via author-match routing; it does not
    touch what's stored in or displayed from Qdrant payloads elsewhere."""
    now = time.time()
    cached = _cache.get(collection_name)
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    names: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            with_payload=["authors"],
            with_vectors=False,
            offset=offset,
        )
        for point in points:
            for name in (point.payload or {}).get("authors") or []:
                # SerpAPI truncates long author lists with "..." — not a name.
                if name and name != "..." and _has_significant_token(name):
                    names.add(name)
        if offset is None:
            break

    result = sorted(names)
    _cache[collection_name] = (now, result)
    return result


def match_author(question: str, known_authors: list[str]) -> str | None:
    """
    Best fuzzy match for `question` among `known_authors`, or None if nothing
    clears MATCH_THRESHOLD.

    token_set_ratio, not partial_ratio: matching is done on the *set of
    words* each side has in common rather than on raw character substrings,
    so a name can't win by aligning with a fragment sitting inside an
    unrelated word (e.g. "r" inside "control" or "your"). `default_process`
    lowercases and strips punctuation on both sides first, since questions
    arrive lowercase but corpus names are stored title-case ("Mark
    Cervinski") — without this, the case mismatch alone was enough to push a
    genuine match below threshold.

    A high aggregate score still isn't enough on its own: this corpus is
    statistics-heavy (correlation coefficients, r-values, CVs), so a
    question can legitimately contain a short token like "r" as a genuine
    standalone word ("What is the r value for this comparison?"). The
    second check below requires the candidate's single most distinctive
    token (its longest word, almost always the surname) to itself be well
    represented in the question — independent defense from the length floor
    already applied when the candidate list was built in get_known_authors.
    """
    if not known_authors:
        return None

    best = process.extractOne(
        question,
        known_authors,
        scorer=fuzz.token_set_ratio,
        processor=utils.default_process,
    )
    if best is None:
        return None

    name, score, _ = best
    if score < MATCH_THRESHOLD:
        return None

    longest_token = max(_tokens(name), key=len, default="")
    if len(longest_token) < MIN_SIGNIFICANT_TOKEN_LENGTH:
        return None

    token_score = fuzz.partial_ratio(
        utils.default_process(longest_token), utils.default_process(question)
    )
    if token_score < MATCH_THRESHOLD:
        return None

    return name


def title_names_author(title: str, author_name: str) -> bool:
    """
    True if `title` names `author_name` -- used to tell a person's own bio
    page apart from an unrelated site page.

    A similarity score cannot make this call. Author names are never embedded
    (see the module docstring), so a name-vs-site-page search scores on prose
    similarity alone, and a filtered vector search always returns its top
    hits. Measured against the live BAPa-V2 collection: a real bio scores
    0.53-0.60 for its own subject, but "Jialin Qiu" -- a co-author with no
    bio page -- still pulls the Security and Privacy page at 0.386, above
    every threshold in the system. There is no gap to threshold on.

    So this checks the title instead, mirroring match_author's second check:
    the name's most distinctive token (its longest, almost always the
    surname) has to be well represented in the title. Exact, and it does not
    drift when the embedding model changes.
    """
    longest_token = max(_tokens(author_name), key=len, default="")
    if len(longest_token) < MIN_SIGNIFICANT_TOKEN_LENGTH:
        return False

    return (
        fuzz.partial_ratio(
            utils.default_process(longest_token), utils.default_process(title)
        )
        >= MATCH_THRESHOLD
    )


def fetch_chunks_by_author(
    client: QdrantClient, collection_name: str, author_name: str
) -> Iterator:
    """Yield every point whose `authors` payload list contains `author_name` exactly."""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="authors", match=MatchValue(value=author_name))]
            ),
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        yield from points
        if offset is None:
            break
