"""
Routing eval for the query engine.

Runs a labeled question set through `QueryEngine._route` and reports how often
each question reached the source it should have. Routing is the single point of
failure in this system -- a question filed as "general" skips retrieval
entirely, so no amount of good indexing rescues it -- and until this script
existed there was no way to tell a regression from a baseline that was always
soft.

Routing costs one cheap classifier call plus a Qdrant search, with no synthesis,
so a full run is fractions of a cent and can be run on every change. Pass
--full to additionally generate answers and check citation integrity, which is
much slower and costs real money.

Usage:
    uv run scripts/eval_routing.py
    uv run scripts/eval_routing.py --provider openai      # compare providers
    uv run scripts/eval_routing.py --group poster book
    uv run scripts/eval_routing.py --full
    uv run scripts/eval_routing.py --save results.json    # for diffing runs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

QUESTIONS_PATH = Path(__file__).resolve().parent / "eval_questions.json"

ROUTES = ("cembrowski", "site", "nih", "general", "hostile", "author", "identity")

# Routes whose RouteDecision carries a real top-hit Qdrant score (gated by
# SCORE_THRESHOLD) -- used to group the score report below.
SCORED_ROUTES = ("cembrowski", "site")

# Emitted by QueryEngine._classify when the model returns a successful response
# with no content. On a thinking model that means reasoning ate the whole
# max_tokens budget, which silently routes every question to "cembrowski" -- so
# this is counted separately and reported even when accuracy happens to look OK.
EMPTY_LABEL_MARKER = "Classifier returned no content"

CITATION_RE = re.compile(r"\[(\d+)\]")

# --full fires synthesis calls back to back, which exhausts a tokens-per-minute
# allowance that production never approaches (the site answers one question at a
# time). gpt-4.1 on the current OpenAI key is capped at 30k TPM, and each
# synthesis call is ~5k in plus up to 2k out -- so a --full run rate-limits
# within a handful of questions unless it backs off. Mirrors the retry in
# data/ocr.py, which hits the same ceiling for the same reason.
GENERATE_RETRIES = 5
RETRY_BACKOFF_CAP_SECONDS = 60.0


def retry_delay(error: Exception, attempt: int) -> float:
    """
    Seconds to wait before retrying. A rate-limit response says how long to wait
    ("Please try again in 18.756s"); honouring that beats guessing. Everything
    else backs off exponentially.
    """
    hint = re.search(r"try again in ([\d.]+)s", str(error))
    if hint:
        return min(float(hint.group(1)) + 0.5, RETRY_BACKOFF_CAP_SECONDS)
    return min(2.0 * (2 ** (attempt - 1)), RETRY_BACKOFF_CAP_SECONDS)


def answer_with_retry(engine, question: str, decision):
    """
    Generate an answer from a precomputed RouteDecision, backing off through
    rate limits.

    Takes a `decision` already computed by `_route` to avoid routing twice —
    the caller measured that decision separately and needs the answer generated
    from exactly that route, not a fresh re-route.
    """
    for attempt in range(1, GENERATE_RETRIES + 1):
        try:
            return engine.answer_from_decision(question, decision)
        except Exception as e:
            error_msg = str(e).lower()
            # Only retry rate-limit errors; re-raise everything else immediately
            if "rate" not in error_msg and "limit" not in error_msg and "429" not in error_msg:
                raise
            if attempt == GENERATE_RETRIES:
                raise
            delay = retry_delay(e, attempt)
            print(f"      rate limited, retrying in {delay:.1f}s "
                  f"(attempt {attempt}/{GENERATE_RETRIES})")
            time.sleep(delay)


class _EmptyCompletionCounter(logging.Handler):
    """Counts the empty-completion warnings the query engine logs."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if EMPTY_LABEL_MARKER in record.getMessage():
            self.count += 1


@dataclass
class Case:
    group: str
    expected: str
    question: str
    actual: str = ""
    label: str | None = None
    top_score: float | None = None
    matched_author: str | None = None
    elapsed_s: float = 0.0
    # --full only
    answer: str | None = None
    n_sources: int | None = None
    citation_errors: list[str] = field(default_factory=list)
    citation_warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.actual == self.expected


def load_cases(path: Path, groups: list[str] | None) -> list[Case]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for q in data["questions"]:
        expected = q["expected"]
        if expected not in ROUTES:
            sys.exit(
                f"Invalid expected route '{expected}' for question: {q['question'][:60]}...\n"
                f"Expected must be one of {ROUTES}"
            )
        cases.append(Case(group=q["group"], expected=expected, question=q["question"]))
    if groups:
        cases = [c for c in cases if c.group in groups]
    if not cases:
        sys.exit(f"No questions matched groups {groups}.")
    return cases


def check_citations(answer: str, n_sources: int) -> tuple[list[str], list[str]]:
    """
    Verify the answer's bracket citations resolve against the source list.
    Returns `(errors, warnings)`.

    The frontend maps `[i]` to `sources[i - 1]`, so a number past the end of the
    list renders as a dead citation for the reader. That, and a bracket form the
    frontend cannot parse, are hard errors — they are broken output regardless
    of how good the prose is, and they are the failure most likely to appear
    when the synthesis model changes.

    An answer that cites nothing while sources were retrieved is only a warning.
    It is often correct: a book-heavy answer can retrieve one loosely related
    poster in its top-k and rightly decline to attribute anything to it. A
    spurious citation would be worse than none.
    """
    errors: list[str] = []
    warnings: list[str] = []
    cited = [int(n) for n in CITATION_RE.findall(answer)]

    for n in sorted(set(cited)):
        if n < 1 or n > n_sources:
            errors.append(f"[{n}] out of range (only {n_sources} sources)")

    if re.search(r"\[\d+\s*[-,]\s*\d+\]", answer):
        errors.append("citation range/list form (e.g. [1-3] or [1, 3])")

    if n_sources and not cited:
        warnings.append(f"no citations despite {n_sources} source(s)")

    return errors, warnings


def print_report(cases: list[Case], empty_labels: int, full: bool, score_threshold: float) -> bool:
    by_group: dict[str, list[Case]] = {}
    for c in cases:
        by_group.setdefault(c.group, []).append(c)

    print("\n" + "=" * 78)
    print("ROUTING ACCURACY")
    print("=" * 78)
    for group, items in by_group.items():
        n_ok = sum(c.ok for c in items)
        flag = "" if n_ok == len(items) else "   <-- regressions"
        print(f"  {group:10} {n_ok:3}/{len(items):<3}{flag}")

    total_ok = sum(c.ok for c in cases)
    print(f"  {'TOTAL':10} {total_ok:3}/{len(cases):<3}  ({total_ok / len(cases):.1%})")

    misrouted = [c for c in cases if not c.ok]
    if misrouted:
        print("\n" + "-" * 78)
        print("MISROUTED")
        print("-" * 78)
        for c in misrouted:
            score = f"{c.top_score:.3f}" if c.top_score is not None else "-"
            print(f"  {c.expected} -> {c.actual}  (label={c.label}, top_score={score})")
            print(f"    {c.question}")

    confusion = Counter((c.expected, c.actual) for c in cases if not c.ok)
    if confusion:
        print("\n  Confusion: " + ", ".join(
            f"{exp}->{act} x{n}" for (exp, act), n in confusion.most_common()
        ))

    scored = [c for c in cases if c.top_score is not None]
    if scored:
        corpus = [c.top_score for c in scored if c.expected in SCORED_ROUTES]
        other = [c.top_score for c in scored if c.expected not in SCORED_ROUTES]
        print("\n" + "-" * 78)
        print(f"TOP-HIT QDRANT SCORES  (SCORE_THRESHOLD gates cembrowski/site at {score_threshold:.2f})")
        print("-" * 78)
        if corpus:
            print(f"  cembrowski/site questions : min={min(corpus):.3f}  max={max(corpus):.3f}  n={len(corpus)}")
        if other:
            print(f"  other questions           : min={min(other):.3f}  max={max(other):.3f}  n={len(other)}")

    print("\n" + "-" * 78)
    print(f"  empty classifier completions: {empty_labels}")
    if empty_labels:
        print("    ^ every one of these silently defaulted to 'cembrowski'.")
        print("      Raise CLASSIFIER_MAX_TOKENS or lower LLM_REASONING_EFFORT.")
    elapsed = [c.elapsed_s for c in cases]
    print(f"  routing latency: median={sorted(elapsed)[len(elapsed) // 2]:.2f}s  max={max(elapsed):.2f}s")

    citation_failures = [c for c in cases if c.citation_errors]
    if full:
        print("\n" + "-" * 78)
        print("CITATION INTEGRITY")
        print("-" * 78)
        checked = [c for c in cases if c.answer is not None]
        print(f"  {len(checked) - len(citation_failures)}/{len(checked)} answers free of dead citations")
        for c in citation_failures:
            print(f"\n  FAIL {c.question}")
            for err in c.citation_errors:
                print(f"    - {err}")

        warned = [c for c in cases if c.citation_warnings]
        if warned:
            print(f"\n  {len(warned)} answer(s) cited nothing despite having sources "
                  "(review, not necessarily wrong):")
            for c in warned:
                print(f"    - {c.question[:64]} ({c.n_sources} src)")

    print()
    return not misrouted and not empty_labels and not citation_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["openrouter", "openai"],
                        help="Override LLM_PROVIDER for this run, to compare providers on the same set.")
    parser.add_argument("--collection", default=None, help="Qdrant collection (default: the configured one).")
    parser.add_argument("--group", nargs="+", default=None, help="Only run these groups (see scripts/eval_questions.json).")
    parser.add_argument("--full", action="store_true",
                        help="Also generate answers and check citation integrity. Slow, and costs real tokens.")
    parser.add_argument("--save", default=None, help="Write per-question results to this JSON path.")
    parser.add_argument("--questions", default=str(QUESTIONS_PATH), help="Path to the labeled question set.")
    args = parser.parse_args()

    # Must precede the chat_cembrowski imports: llm.get_config() reads the
    # environment, and .env is loaded at module import time.
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    from chat_cembrowski.data.vectordb import get_qdrant_client, get_voyage_client
    from chat_cembrowski.retrieval import llm
    from chat_cembrowski.retrieval.query_engine import QueryEngine, SCORE_THRESHOLD

    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    counter = _EmptyCompletionCounter()
    qe_logger = logging.getLogger("chat_cembrowski.retrieval.query_engine")
    qe_logger.setLevel(logging.WARNING)
    qe_logger.addHandler(counter)

    config = llm.get_config()
    cases = load_cases(Path(args.questions), args.group)

    engine_kwargs = {}
    if args.collection:
        engine_kwargs["collection_name"] = args.collection

    engine = QueryEngine(
        qdrant_client=get_qdrant_client(),
        llm_client=llm.get_llm_client(config),
        voyage_client=get_voyage_client(),
        llm_config=config,
        **engine_kwargs,
    )

    print(f"provider={config.provider}  chat={config.chat_model}  classifier={config.classifier_model}")
    print(f"reasoning_effort={config.reasoning_effort} (classifier pinned to {llm.CLASSIFIER_REASONING_EFFORT})")
    print(f"collection={engine.collection_name}  questions={len(cases)}  full={args.full}\n")

    for i, case in enumerate(cases, start=1):
        start = time.perf_counter()
        decision = engine._route(case.question)
        case.elapsed_s = time.perf_counter() - start
        case.actual = decision.route
        case.label = decision.label
        case.top_score = decision.top_score
        case.matched_author = decision.matched_author

        if args.full:
            result = answer_with_retry(engine, case.question, decision)
            case.answer = result.answer
            case.n_sources = len(result.sources)
            case.citation_errors, case.citation_warnings = check_citations(
                result.answer, len(result.sources)
            )

        mark = "ok  " if case.ok else "FAIL"
        print(f"  [{i:>2}/{len(cases)}] {mark} {case.expected:>10} -> {case.actual:<10} {case.question[:52]}")

    clean = print_report(cases, counter.count, args.full, SCORE_THRESHOLD)

    if args.save:
        Path(args.save).write_text(
            json.dumps(
                {
                    "provider": config.provider,
                    "chat_model": config.chat_model,
                    "classifier_model": config.classifier_model,
                    "reasoning_effort": config.reasoning_effort,
                    "empty_labels": counter.count,
                    "cases": [asdict(c) for c in cases],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {args.save}")

    sys.exit(0 if clean else 1)


if __name__ == "__main__":
    main()
