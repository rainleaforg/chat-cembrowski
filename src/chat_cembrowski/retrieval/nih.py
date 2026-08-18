"""
NIH lookup for general medical questions that fall outside the Cembrowski
research corpus.

Two free, keyless NIH/NLM search services are used:
  - MedlinePlus Web Service  (consumer health topics, plain language) — primary
  - PubMed via NCBI E-utilities (biomedical literature abstracts) — fallback

Neither service "answers" a question — they return search results — so callers
should build an LLM context block from the returned NIHResult objects, the
same way query_engine builds context from Qdrant chunks.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional
from xml.etree import ElementTree

import requests
from dotenv import load_dotenv

from . import llm as llm_module
from .prompts import NIH_SEARCH_TERMS_PROMPT

logger = logging.getLogger(__name__)
load_dotenv()

MEDLINEPLUS_URL = "https://wsearch.nlm.nih.gov/ws/query"
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

REQUEST_TIMEOUT = 10
USER_AGENT = "chat-cembrowski/0.1 (NIH lookup for general medical questions)"

NCBI_API_KEY = os.getenv("NCBI_API_KEY")
NCBI_EMAIL = os.getenv("NCBI_EMAIL")

_TAG_RE = re.compile(r"<[^>]+>")

# Simple in-process memo so repeated/similar questions in one process don't
# re-hit the NIH APIs. Keyed on (function, normalized term).
_CACHE: dict[tuple[str, str], list["NIHResult"]] = {}

# Ceiling, not a reservation -- same thinking-model trap as CLASSIFIER_MAX_TOKENS
# in query_engine.py: a starved reasoning budget returns empty content with no
# exception, not an error.
SEARCH_TERMS_MAX_TOKENS = 512

_SEARCH_TERMS_CACHE: dict[str, str] = {}


def extract_search_terms(question: str, llm_client, config: "llm_module.LLMConfig") -> str:
    """
    Extract 2-5 keywords from `question` for the MedlinePlus/PubMed keyword
    search, via the cheap classifier model.

    Sending a whole sentence to a keyword search is why so many questions
    returned nothing: "Who won the last FIFA World Cup?" sent verbatim
    matches nothing MedlinePlus indexes, but "FIFA World Cup" as a term at
    least would. Falls back to the raw question on any API failure or empty
    completion, matching this module's existing degrade-gracefully pattern.
    """
    cache_key = question.strip().lower()
    if cache_key in _SEARCH_TERMS_CACHE:
        return _SEARCH_TERMS_CACHE[cache_key]

    try:
        response = llm_client.chat.completions.create(
            model=config.classifier_model,
            temperature=0,
            max_tokens=SEARCH_TERMS_MAX_TOKENS,
            messages=[
                {"role": "system", "content": NIH_SEARCH_TERMS_PROMPT},
                {"role": "user", "content": question},
            ],
            **llm_module.completion_extras(
                config, effort=llm_module.CLASSIFIER_REASONING_EFFORT
            ),
        )
    except Exception as e:
        logger.warning(f"Search-term extraction failed ({e}); using the raw question.")
        return question

    choice = response.choices[0]
    terms = (choice.message.content or "").strip()

    if not terms:
        logger.warning(
            "Search-term extraction returned no content "
            f"(model={config.classifier_model}, finish_reason={choice.finish_reason!r}); "
            "using the raw question."
        )
        return question

    _SEARCH_TERMS_CACHE[cache_key] = terms
    return terms


@dataclass
class NIHResult:
    """A single result from an NIH search service."""
    source: str            # "MedlinePlus" | "PubMed"
    title: str
    summary: str
    url: str
    journal: Optional[str] = None
    year: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "title": self.title,
            "summary": self.summary,
            "url": self.url,
            "journal": self.journal,
            "year": self.year,
        }


def _strip_tags(text: str | None) -> str:
    """Strip MedlinePlus highlight/HTML markup from a snippet, collapse whitespace."""
    if not text:
        return ""
    return " ".join(_TAG_RE.sub("", text).split())


def _ncbi_params(extra: dict) -> dict:
    """Attach optional NCBI etiquette params (api_key/tool/email) when configured."""
    params = dict(extra)
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    params["tool"] = "chat-cembrowski"
    return params


def search_medlineplus(term: str, max_results: int = 4) -> list[NIHResult]:
    """
    Search MedlinePlus consumer health topics for `term`.

    Returns an empty list (rather than raising) on any network/parse failure,
    so a single flaky request degrades the answer instead of crashing it.
    """
    cache_key = ("medlineplus", term.strip().lower())
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    try:
        response = requests.get(
            MEDLINEPLUS_URL,
            params={"db": "healthTopics", "term": term, "retmax": max_results},
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
    except Exception as e:
        logger.warning(f"MedlinePlus search failed for '{term}': {e}")
        return []

    results: list[NIHResult] = []
    for doc in root.findall(".//document")[:max_results]:
        url = doc.get("url", "")
        title = ""
        summary = ""
        for content in doc.findall("content"):
            name = content.get("name", "")
            if name == "title":
                title = _strip_tags(content.text)
            elif name == "FullSummary" and not summary:
                summary = _strip_tags(content.text)
            elif name == "snippet" and not summary:
                summary = _strip_tags(content.text)

        if not title:
            continue

        results.append(
            NIHResult(source="MedlinePlus", title=title, summary=summary, url=url)
        )

    _CACHE[cache_key] = results
    return results


def search_pubmed(term: str, max_results: int = 3) -> list[NIHResult]:
    """
    Search PubMed (via NCBI E-utilities) for `term` and fetch abstracts.

    Two-step E-utilities flow: esearch for PMIDs, then efetch for abstract XML.
    Returns an empty list on any failure rather than raising.
    """
    cache_key = ("pubmed", term.strip().lower())
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    try:
        esearch_resp = requests.get(
            PUBMED_ESEARCH_URL,
            params=_ncbi_params(
                {
                    "db": "pubmed",
                    "term": term,
                    "retmode": "json",
                    "sort": "relevance",
                    "retmax": max_results,
                }
            ),
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        esearch_resp.raise_for_status()
        id_list = esearch_resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        logger.warning(f"PubMed esearch failed for '{term}': {e}")
        return []

    if not id_list:
        _CACHE[cache_key] = []
        return []

    try:
        efetch_resp = requests.get(
            PUBMED_EFETCH_URL,
            params=_ncbi_params(
                {
                    "db": "pubmed",
                    "id": ",".join(id_list),
                    "rettype": "abstract",
                    "retmode": "xml",
                }
            ),
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        efetch_resp.raise_for_status()
        root = ElementTree.fromstring(efetch_resp.content)
    except Exception as e:
        logger.warning(f"PubMed efetch failed for '{term}': {e}")
        return []

    results: list[NIHResult] = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        if pmid_el is None or title_el is None or not title_el.text:
            continue

        pmid = pmid_el.text
        title = _strip_tags(title_el.text)

        abstract_parts = [
            "".join(el.itertext())
            for el in article.findall(".//AbstractText")
        ]
        summary = _strip_tags(" ".join(abstract_parts))

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else None

        year_el = article.find(".//JournalIssue/PubDate/Year")
        year = year_el.text if year_el is not None else None

        results.append(
            NIHResult(
                source="PubMed",
                title=title,
                summary=summary,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                journal=journal,
                year=year,
            )
        )

    _CACHE[cache_key] = results
    return results


def search_nih(
    term: str,
    medlineplus_max: int = 4,
    pubmed_max: int = 3,
) -> list[NIHResult]:
    """
    Search NIH sources for a general medical question.

    MedlinePlus (plain-language, patient-facing) is tried first since it's the
    better fit for non-technical users. PubMed is a supplement, not a
    fallback: it fires whenever MedlinePlus returns fewer than `medlineplus_max`
    results, not only when MedlinePlus returns none.
    """
    results = search_medlineplus(term, max_results=medlineplus_max)

    if len(results) < medlineplus_max:
        results += search_pubmed(term, max_results=pubmed_max)

    return results
