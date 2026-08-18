# RAG Query System for answering questions based on George Cembrowski's publications and related documents.

## Installation

Install uv if not already available:
```
python -m pip install uv
```
Install project dependencies and the package in editable mode:
```
uv sync
```

## Environment

Create a `.env` file in the project root with the following variables:
```
SERPAPI_KEY=...
OPENROUTER_API_KEY=...       # answer generation (retrieval path)
OPENAI_API_KEY=...           # OCR + paper metadata extraction (data pipeline)
VOYAGE_API_KEY=...
QDRANT_API_KEY=...           # only needed for cloud mode
QDRANT_CLUSTER_ENDPOINT=...  # only needed for cloud mode
```

Accounts needed: SerpAPI (paper fetching), OpenRouter (answer generation), OpenAI (scanned-PDF OCR + paper metadata extraction), Voyage AI (embeddings), Qdrant (vector store — local mode works without an account).

Both LLM keys are required. Only the retrieval path is switchable between providers; the data pipeline calls OpenAI directly for OCR (`data/ocr.py`) and metadata extraction (`data/ingestion.py`) regardless of `LLM_PROVIDER`.

Optional overrides, all with working defaults — see `.env.example`: `LLM_PROVIDER`, `CHAT_MODEL`, `CLASSIFIER_MODEL`, `LLM_REASONING_EFFORT`.

## Data Structure

```
data/
  catalog.csv # Sourcing checklist: every known work, most-cited first
  papers/     # PDF source files
  json/       # Paper metadata + extracted text as JSON
  docs/       # Miscellaneous context documents (txt, md, docx, code files)
  doc_json/   # Document metadata + extracted text as JSON
  images/     # Extracted image files
  image_json/ # Image metadata as JSON
  vectors/    # Local Qdrant storage (when not using cloud)
```

## Design Decisions and Models Used

- **Vector Embeddings**: Voyage AI `voyage-multimodal-3.5` (1024 dimensions) — handles text and image+text pairs in the same vector space
- **Answer Generation**: Google `gemini-3.6-flash` via OpenRouter (see below)
- **Query Classification**: Google `gemini-3.5-flash-lite` via OpenRouter
- **Chunking**: Language-aware recursive chunking — code files use language-specific splitters (Python, JS, TS, C++, Java, etc.), prose and docx use a markdown-aware splitter. `CHUNK_SIZE=1024`, `CHUNK_OVERLAP=128`
- **Retrieval**: Top 10 chunks from Qdrant vector search
- **PDF Parsing**: `pymupdf4llm` — extracts text, tables, and charts as structured markdown with automatic OCR if needed

## Which models the assistant uses, and why

Configured in `src/chat_cembrowski/retrieval/llm.py`.

### Why OpenRouter rather than Gemini directly

OpenRouter charges **no markup on inference** — its per-token prices match Google's own list prices; it makes its money on a 5.5% credit top-up fee. In exchange, changing models becomes an env-var edit rather than a code change, which is the point: the model choice here is expected to move.

Going direct to Google would buy one thing OpenRouter cannot offer — Gemini's Google Search grounding — and we deliberately do not want it. Grounding returns `vertexaisearch.cloud.google.com` redirect URIs that expire instead of real source URLs, obliges the site to render Google's Search Suggestions chips under its Service Terms, and has the model do its own retrieval, which breaks the numbered-source contract the frontend relies on (`[i]` resolves to `sources[i - 1]`). None of that is worth the ~5% and the small latency saving.

### Why Flash and not Pro

The synthesis job is **grounded summarization**: roughly 5k tokens of already-retrieved context, condensed into a short markdown answer with correct bracket citations. Retrieval has already done the hard part. There are no intermediate steps for a reasoning model to work through, so `gemini-3.1-pro` ($2.00/$12.00 per M tokens) would cost 15–55% more than the current GPT-4.1 setup to do work that is not reasoning-bound. `gemini-3.6-flash` ($1.50/$7.50) lands roughly break-even to ~20% cheaper than GPT-4.1.

Note that "Flash" is no longer a budget tier in the 3.x line — its output tokens cost *more* than GPT-4.1's ($7.50 vs $8.00 is close, and `gemini-3.5-flash` is $9.00). The genuinely cheap tier is Flash-Lite.

### Why Flash-Lite for classification

`_classify` picks one of three words against a prompt (`prompts.CLASSIFIER_PROMPT`) that already spells out the decision rule and carries six paired examples. That is pattern matching with in-context examples, not reasoning, and `gemini-3.5-flash-lite` ($0.30/$2.50) handles it at roughly a fifth the price. Measured: 47/47 correct on the routing eval.

### Why thinking is off by default

Gemini 2.5+ and all of 3.x bill reasoning tokens as output *and* count them against `max_tokens`. Two consequences:

1. **Cost** is dominated by the thinking budget, not the model tier — leaving reasoning on moves the bill more than choosing Flash vs. Pro does.
2. **Latency** matters more here than cost. `_classify` is a blocking call on the critical path of *every* question; reasoning adds seconds to first token in a chat UI for a single-word output.

`LLM_REASONING_EFFORT` (default `minimal`) controls synthesis. The classifier is pinned to `minimal` and ignores it, so raising the global setting cannot quietly slow down every query.

There is a sharp edge worth knowing about: because reasoning tokens consume `max_tokens`, a budget that is too small gets spent entirely on thinking and the model returns **empty content with no error**. The classifier's cap used to be 5 tokens — ample for the one word it emits, and instantly fatal on a thinking model, since an empty label silently routes every question to the corpus. The caps in `query_engine.py` (`CLASSIFIER_MAX_TOKENS` and friends) are generous for that reason; they are ceilings, not reservations, so the headroom is free. `_classify` also logs a warning on an empty completion rather than letting it pass as a classification.

### Per-query cost

Measured against this corpus (~770-token classifier call, ~5k-token synthesis call):

| Setup | Per query |
|---|---|
| GPT-4.1 + GPT-4.1-mini (previous) | ~$0.0143 |
| **gemini-3.6-flash + gemini-3.5-flash-lite** | **~$0.011–0.015** |
| gemini-3.1-pro + gemini-3.5-flash-lite | ~$0.016–0.022 |

At realistic site volume this is a difference of a few dollars a month. The deciding factors were routing accuracy and latency, not cost.

### Rolling back

`LLM_PROVIDER=openai` restores GPT-4.1 / GPT-4.1-mini in one line, with no code change.

## Running the Data Pipeline

### Research Papers

Ingestion is split into **discovery** (what has the author published?) and **acquisition** (get the PDF). Discovery is cheap and automatic; acquisition is mostly manual, for reasons covered in [Why acquisition is manual](#why-acquisition-is-manual).

Run from the root of the repository in order:

1. `uv run -m chat_cembrowski.data.ingestion catalog`
   Records every work the author has published, **without downloading anything**. Writes one Paper JSON per work (with no `source_file` yet) and a sourcing checklist to `data/catalog.csv`, sorted most-cited first.

   Costs `ceil(N/100)` SerpAPI searches in total — **4 searches for a 332-work profile**, regardless of `--num-articles`. This is the recommended entry point.

   Optional flags:
   - `--author-id <ID>` — a different Google Scholar Author ID (default: `j8iA0kAAAAAJ`)
   - `--num-articles <N>` — upper bound on works (default: `1000`, i.e. everything). Overshooting is safe; pagination stops when the author's list is exhausted.
   - `--with-pdf-links` — also resolve a public PDF URL per work. **Costs one SerpAPI search per article**, so bound it with `--max-lookups`.
   - `--max-lookups <N>` — cap those paid lookups, spent on the most-cited works first.

   Re-running is safe: the catalog merges rather than overwrites, so refreshing metadata never drops an already-acquired PDF, its extracted text, or its `processed` flag. It also de-duplicates by title, since Scholar keeps separate records for the same work (conference abstract vs. journal version) that would otherwise become duplicate chunks in Qdrant.

   `data/catalog.csv` columns:

   | Column | Notes |
   |---|---|
   | `id`, `title`, `authors`, `year`, `publication` | From Scholar; author lists may be truncated (see step 3) |
   | `cited_by` | Sort key — decides what is worth sourcing first |
   | `have_pdf` | `yes` once a PDF is attached |
   | `pdf_url` | Only populated by `--with-pdf-links` |
   | `scholar_link` | Always present, free — the click-through to the publisher |

2. **Collect PDFs into `data/papers/`.** Work down `catalog.csv` from the top and download in a browser. Filenames don't matter — they get renamed on ingest. For a corpus of this size, asking the author or an institutional-access holder for the PDFs in bulk will out-yield anything else.

3. `uv run -m chat_cembrowski.data.ingestion ingest_local`
   Attaches those PDFs to their catalog rows. Each PDF's title is extracted from its first page via GPT-4.1-mini and fuzzy-matched against catalog entries still awaiting a PDF; on a confident match the existing row is updated rather than a second record being created for the same work. A PDF that matches nothing becomes a standalone Paper.

   This step is also where **authoritative metadata** comes from: SerpAPI truncates long author lists (storing `...` as a sentinel), and the PDF is the better source.

   Optional flag:
   - `--reextract-authors` — re-derive authors from the PDF for *every* registered paper, not just visibly truncated ones. Scholar sometimes returns a short author list with nothing to mark it as incomplete, which no sentinel can detect. Costs one LLM call per paper.

4. `uv run -m chat_cembrowski.data.parser`
   Parses each PDF to markdown and stores it in the Paper JSON.

5. `uv run -m chat_cembrowski.data.image_extractor`
   Extracts images from each PDF, finds captions, and writes ImageRecord JSONs to `data/image_json/`.

6. `uv run -m chat_cembrowski.data.vectordb`
   Chunks, embeds, and upserts everything to Qdrant.

   Optional flag:
   - `--collection <name>` — Qdrant collection name to upsert into (default: `BAPa-V2`)

   Posters linked to a page on the website get their `site_path` stamped automatically at
   ingest time by `PAAN-cembrowski/ingestion-lambda/ingest.py::_stamp_site_path` — no separate
   linking step is needed.

Steps 4–6 skip catalog entries that have no PDF yet, so the pipeline runs cleanly against a partially-sourced corpus — it simply embeds nothing for works you haven't collected. Re-run them as more PDFs arrive.

**Editing content already in the index.** Documents in `data/docs/` are expected to change. Edit the file and re-run `doc_ingestion` then `vectordb` — the edit is detected by content hash, and re-indexing *replaces* the old chunks rather than adding a second copy alongside them. See [Updating and re-indexing](#updating-and-re-indexing).

To force a re-index without editing anything, reset the processed flags first:
```
uv run scripts/reset_processed.py            # papers and documents
uv run scripts/reset_processed.py --docs     # documents only
```

#### Why acquisition is manual

The obvious design — have SerpAPI find each paper's "publicly available" PDF and download it — does not work on this corpus. Measured on a 5-article sample: 3 had a public PDF according to Scholar and **all 3 returned HTTP 403**; the other 2 had no public resource at all. **0 of 5 acquired.**

That isn't a bug to fix in code:

- A browser `User-Agent` changes nothing — ResearchGate and SAGE are running real bot protection, not header sniffing.
- OpenAlex independently resolves two of those works to the *same* blocked publisher URLs.
- The two works with PMC IDs sit outside the PMC open-access subset; Europe PMC returns 404 for both PDF and full-text XML.

Scholar lists these as public because Scholar's own crawler can reach them. A script cannot. Expect this to be **worse** from cloud infrastructure, where datacenter IP ranges are blocked most aggressively.

The cost asymmetry drives the rest of the design. Paginating the full publication list is 4 SerpAPI searches; resolving a public PDF link is one search *per article* — ~332 for this author, against a free-tier quota of 250/month. Spending virtually the entire quota on the step with a near-zero success rate is the wrong trade, so discovery and acquisition are separate commands and only discovery runs by default.

The legacy download path is still available for authors whose work *is* freely reachable:

```
uv run -m chat_cembrowski.data.ingestion --num-articles 50
```

It costs ~1 SerpAPI search per article, accepts PDFs only (every downstream stage opens `source_file` with PyMuPDF), creates no Paper record for a failed download, and prints the failures as a to-collect list at the end. `--interactive` restores the older behaviour of pausing on each failure so you can place the PDF by hand — off by default, since a large run would otherwise stall on stdin hundreds of times.

### Miscellaneous Documents

Place any `.txt`, `.md`, `.docx`, or code files in `data/docs/`, then run:

1. `uv run -m chat_cembrowski.data.doc_ingestion`
   Extracts structured text from each file. `.docx` files are converted to markdown (headings, lists, tables preserved). Code files are split with a language-aware splitter. Saves Document JSONs to `data/doc_json/`.

2. `uv run -m chat_cembrowski.data.vectordb`
   Embeds and upserts the new document chunks alongside any unprocessed papers.

Both steps are idempotent — already-processed files are skipped.

## Querying the System

### Interactive CLI

```
uv run scripts/chat.py
```

Starts an interactive session. Type your question at the prompt; type `exit` or `quit` to stop.

Optional flag:
- `--collection <name>` — Qdrant collection to query (default: `BAPa-V2`)

The CLI checks that the collection exists before starting and will list available collections if it doesn't.

### Batch questions

Edit the questions in `scripts/ask.py`, then run:
```
uv run scripts/ask.py
```

Optional flag:
- `--collection <name>` — Qdrant collection to query (default: `BAPa-V2`)

The system retrieves across papers, images, and documents in a single search. The model cites by bracketed number only — `[1]`, `[2]` — matching the `SOURCE n` blocks it was shown; `QueryResult.sources` is aligned by position, so `[i]` resolves to `sources[i - 1]` and the frontend turns it into a link.

### Routing eval

Routing is the single point of failure in this system: a question misfiled as `general` skips retrieval entirely, so no amount of good indexing rescues it. `scripts/eval_routing.py` runs a labeled question set (`scripts/eval_questions.json`) through `QueryEngine._route` and reports per-route accuracy, a confusion table, top-hit Qdrant scores, and the count of empty classifier completions.

```bash
uv run scripts/eval_routing.py                      # routing only — fractions of a cent, seconds
uv run scripts/eval_routing.py --provider openai    # compare providers on the same set
uv run scripts/eval_routing.py --group poster book  # one or more groups
uv run scripts/eval_routing.py --full               # also generate answers + check citations
uv run scripts/eval_routing.py --save results.json  # for diffing runs
```

Routing costs one cheap classifier call plus a Qdrant search per question and no synthesis, so a full run is cheap enough to do on every change. `--full` additionally generates every answer and verifies citation integrity — that every `[i]` resolves to a real source and no unparseable `[1-3]` forms appear — which is slower and costs real tokens.

`--full` fires synthesis calls back to back and will exhaust a tokens-per-minute allowance that production never approaches, since the site answers one question at a time. The OpenAI key is capped at 30k TPM for `gpt-4.1` (measured on the current organization account as of early 2025; this limit is environment-specific and depends on usage tier) and each call is ~5k in plus up to 2k out, so `--provider openai --full` rate-limits within a few questions; the script backs off on the server's own "try again in Xs" hint and continues, which makes that run slow but reliable.

Exit code is 0 only when routing is perfect, no classifier completion came back empty, and no answer contains a dead citation.

**Re-run this after any change to `CLASSIFIER_PROMPT`, the embedding model, the chunking strategy, `SCORE_THRESHOLD`, or the corpus itself.** New corpus material that the classifier prompt does not describe gets filed as `general` and becomes unreachable — this has happened before, when the textbook was ingested while the prompt still described the corpus as papers about troponin and blood gas analyzers.

Current results, both providers measured on the same 47-question set:

| | Gemini (`3.6-flash` / `3.5-flash-lite`) | GPT (`4.1` / `4.1-mini`) |
|---|---|---|
| Routing | 47/47 | 47/47 |
| Citations free of dead `[i]` | 47/47 | 47/47 |
| Empty classifier completions | 0 | 0 |
| Median routing latency | 0.74s | 0.88s |
| Max routing latency | 1.49s | 3.24s |

Routing accuracy is identical — retrieval is Voyage/Qdrant and unaffected by the model swap, so both see the same top-hit scores (0.473–0.769). The difference that shows up for a user is tail latency.

## Future Steps

- Metadata filtering for retrieval (filter by year, publication, source type)
- Reranker: retrieve top 30, rerank to top 10 with a model like `bge-reranker`
