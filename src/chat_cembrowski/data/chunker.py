"""
This module chunks text (extracted from articles) for embedding and RAG.
Currently implements recursive chunking using LangChain's markdown-aware splitter.
Each chunk will also carry paper metadata (title, authors, etc.) in its payload dict, ready for Qdrant upsert.
"""

import logging
import uuid
from pathlib import Path


from openai import OpenAI

from . import ocr
from .models import Paper, Chunk, ImageRecord, Document
from .parser import preprocess_markdown, parse_pdf_for_pages, render_pdf_pages_as_images, DEFAULT_PAGE_RENDER_ZOOM
from .serialization import load_paper, load_image_records_for_paper


from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 # characters per chunk
CHUNK_OVERLAP = 128

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data/papers"
JSON_DIR = PROJECT_ROOT / "data/json"
IMAGE_JSON_DIR = PROJECT_ROOT / "data/image_json"
PAGE_IMAGES_DIR = PROJECT_ROOT / "data/page_images"

# Fixed namespaces for deriving deterministic chunk IDs (uuid5), so re-running
# the chunker on the same source yields the same point IDs and upserts
# overwrite in place rather than duplicating.
#
# Deterministic IDs handle an updated chunk but not a *removed* one: if an edit
# shortens a source from 10 chunks to 7, points 7-9 keep their old content. The
# other half of the contract is vectordb.delete_points_for(), which clears a
# source's points before re-upserting. Both are required.
PAGE_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "chat_cembrowski.page_chunks")
TEXT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "chat_cembrowski.text_chunks")
DOC_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "chat_cembrowski.doc_chunks")

def _build_payload(paper: Paper, chunk_index: int, chunk_text: str, page_start: int, page_end: int) -> dict:
    """
    Assemble Qdrant payload metadata for a given chunk.
    Args:
        paper: Paper object containing metadata
        chunk_index: Index of the current chunk (0-based)
        chunk_text: Text content of the current chunk
        page_start: Starting page number of the chunk
        page_end: Ending page number of the chunk
    Returns:
        Dictionary payload with paper metadata and chunk info
    """
    return {
        "chunk_category": "text",
        "paper_id": paper.id,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "publication": paper.publication,
        "chunk_index": chunk_index,
        "page_start": page_start,
        "page_end": page_end,
        "page_label": (
            f"p. {page_start}"
            if page_start == page_end
            else f"pp. {page_start}–{page_end}"
        ),
        "text": chunk_text,
    }

# How many characters to overlap between adjacent pages to preserve
# cross-page sentence continuity. Mirrors your chunk overlap strategy.
PAGE_STITCH_OVERLAP = CHUNK_OVERLAP

def chunk_paper(paper: Paper) -> list[Chunk]:
    """
    Splits a paper into overlapping chunks with page-number metadata.

    Strategy:
      1. Parse the PDF into per-page text blocks.
      2. Stitch adjacent pages with a character overlap so sentences that
         span a page break are not severed.
      3. Split the stitched text with RecursiveCharacterTextSplitter.
      4. Map each resulting chunk back to the source page(s) it came from
         by scanning a character-offset index built from the stitched text.

    Args:
        paper: Paper object containing source_file path and metadata.

    Returns:
        List of Chunk objects; each payload includes page_start / page_end.
    """
    pages = parse_pdf_for_pages(DATA_DIR / paper.source_file)
    if not pages:
        logger.warning(f"Paper {paper.id} yielded no page text. Skipping.")
        return []

    # journal_page = pdf_page + page_offset
    # Anchors the first non-empty PDF page to first_page_number, handling blank leading pages.
    page_offset = (
        paper.first_page_number - pages[0]["page"]
        if paper.first_page_number is not None
        else 0
    )

    # ------------------------------------------------------------------
    # 1. Build a single stitched string and record the char offset at
    #    which each page begins.  The overlap is taken from the *tail* of
    #    the previous page so the splitter can see both sides of a break.
    # ------------------------------------------------------------------
    stitched_parts: list[str] = []
    # page_offsets[i] = (char_start, char_end, page_number) in stitched text
    page_offsets: list[tuple[int, int, int]] = []
    cursor = 0

    for i, page in enumerate(pages):
        page_text = preprocess_markdown(page["text"])

        if i > 0 and PAGE_STITCH_OVERLAP > 0:
            # Prepend the tail of the previous page's contribution so the
            # splitter can bridge the boundary.  We do NOT advance cursor
            # for this overlap region — it is intentionally double-counted
            # so the page attribution below stays correct.
            prev_tail = stitched_parts[-1][-PAGE_STITCH_OVERLAP:]
            page_text = prev_tail + page_text

        start = cursor
        end = cursor + len(page_text)
        page_offsets.append((start, end, page["page"]))
        stitched_parts.append(page_text)
        cursor = end

    stitched_text = "".join(stitched_parts)

    # ------------------------------------------------------------------
    # 2. Split
    # ------------------------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    try:
        docs = text_splitter.create_documents([stitched_text])
    except Exception as e:
        logger.error(f"Error splitting text for paper {paper.id}: {e}")
        return []

    # ------------------------------------------------------------------
    # 3. Map each chunk to page(s)
    # ------------------------------------------------------------------
    def page_for_offset(char_offset: int) -> int:
        """Return the 1-based page number that owns char_offset."""
        for start, end, page_num in page_offsets:
            if start <= char_offset < end:
                return page_num
        # Fallback: last page
        return page_offsets[-1][2]

    chunks: list[Chunk] = []
    for i, doc in enumerate(docs):
        chunk_text = doc.page_content.strip()
        if not chunk_text:
            continue

        pos = doc.metadata.get("start_index", 0)
        chunk_start_page = page_for_offset(pos)
        chunk_end_page = page_for_offset(pos + len(chunk_text) - 1)

        chunks.append(
            Chunk(
                id=str(uuid.uuid5(TEXT_ID_NAMESPACE, f"{paper.id}:text:{i}")),
                text=_build_text_embed_text(paper, chunk_text),
                payload=_build_payload(
                    paper, i, chunk_text,
                    chunk_start_page + page_offset,
                    chunk_end_page + page_offset,
                ),
            )
        )

    logger.info(f"Paper {paper.id} split into {len(chunks)} chunks.")
    return chunks


def chunk_paper_text_only(paper: Paper) -> list[Chunk]:
    """
    Splits paper's cleaned text into overlapping chunks with metadata payload.

    Args:
        paper: Paper object containing text and metadata

    Returns:
        List of Chunk objects with text and metadata payload
    """

    if not paper.text or not paper.text.strip():
        logger.warning(f"Paper {paper.id} has empty text. Skipping chunking.")
        return []
    
    cleaned_text = preprocess_markdown(paper.text)

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    try:
        raw_chunks = text_splitter.split_text(cleaned_text)
    except Exception as e:
        logger.error(f"Error splitting text for paper {paper.id}: {e}")
        return []
    
    chunks: list[Chunk] = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        chunks.append(
            Chunk(
                id=str(uuid.uuid5(TEXT_ID_NAMESPACE, f"{paper.id}:text:{i}")),
                text=_build_text_embed_text(paper, chunk_text),
                payload=_build_payload(paper, i, chunk_text, 1, 1),
            )
        )

    logger.info(f"Paper {paper.id} split into {len(chunks)} chunks.")
    return chunks



def _build_text_embed_text(paper: Paper, chunk_text: str) -> str:
    """Prepend paper metadata to chunk text so the embedding carries provenance context."""
    parts: list[str] = []
    if paper.title:
        parts.append(f"Title: {paper.title}")
    if paper.year:
        parts.append(f"Year: {paper.year}")
    if paper.publication:
        parts.append(f"Publication: {paper.publication}")
    parts.append(chunk_text)
    return "\n".join(parts)


def _build_image_embed_text(record: ImageRecord) -> str:
    """
    Compose the text input sent to Voyage alongside the image.
    Caption anchors semantics; paper metadata improves retrieval precision.
    """
    parts: list[str] = []
    if record.caption:
        parts.append(record.caption)
    if record.title:
        parts.append(f"Title: {record.title}")
    if record.year:
        parts.append(f"Year: {record.year}")
    if record.publication:
        parts.append(f"Publication: {record.publication}")
    return "\n".join(parts)


def _build_image_payload(record: ImageRecord, chunk_index: int) -> dict:
    return {
        "chunk_category": "image",
        "paper_id": record.paper_id,
        "title": record.title,
        "authors": record.authors,
        "year": record.year,
        "publication": record.publication,
        "chunk_index": chunk_index,
        "page": record.page,
        "page_label": f"p. {record.page}",
        "source_file": record.source_file,
        "bbox": list(record.bbox),
        "caption": record.caption,
        "image_type": record.image_type,
        "text": _build_image_embed_text(record),
    }


def chunk_paper_images(
    paper: Paper,
    image_json_dir: Path = IMAGE_JSON_DIR,
) -> list[Chunk]:
    """
    Convert all extracted ImageRecords for a paper into Chunk objects.

    Chunk.text is the text input Voyage will receive alongside the image
    (caption + title + year + publication).  Chunk.id matches ImageRecord.id
    so vectordb can locate the image file without an extra lookup.
    """
    records = load_image_records_for_paper(paper.id, image_json_dir)
    if not records:
        logger.info(f"No image records found for paper {paper.id}.")
        return []

    chunks: list[Chunk] = []
    for i, record in enumerate(records):
        chunks.append(
            Chunk(
                id=record.id,
                text=_build_image_embed_text(record),
                payload=_build_image_payload(record, i),
            )
        )

    logger.info(f"Paper {paper.id} produced {len(chunks)} image chunks.")
    return chunks


def _build_page_embed_text(paper: Paper, page_text: str) -> str:
    """
    Compose the text input sent to Voyage alongside the full-page image.
    Paper metadata anchors provenance; the page's own markdown carries the
    text Voyage can't read directly off the rendered image.
    """
    parts: list[str] = []
    if paper.title:
        parts.append(f"Title: {paper.title}")
    if paper.year:
        parts.append(f"Year: {paper.year}")
    if paper.publication:
        parts.append(f"Publication: {paper.publication}")
    if page_text:
        parts.append(page_text)
    return "\n".join(parts)


def _build_page_payload(
    paper: Paper,
    chunk_index: int,
    page_text: str,
    source_file: str,
    journal_page: int,
) -> dict:
    """
    Payload for a whole-page image chunk. Reuses the image chunk_category so
    it flows through the existing multimodal embed/retrieval paths; image_type
    "page" (as opposed to "figure"/"table"/etc.) distinguishes it from cropped
    figure chunks produced by chunk_paper_images(). `text` carries the page's
    markdown so it can stand in as LLM context at query time.
    """
    return {
        "chunk_category": "image",
        "image_type": "page",
        "paper_id": paper.id,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "publication": paper.publication,
        "chunk_index": chunk_index,
        "page": journal_page,
        "page_label": f"p. {journal_page}",
        "source_file": source_file,
        "bbox": None,  # whole page, not a crop
        "caption": None,
        "text": page_text,
    }


def _ocr_fallback_for_page(
    image_path: Path,
    paper_id: str,
    openai_client: OpenAI | None,
) -> str:
    """
    Transcribe a rendered page image via OCR when `parse_pdf_for_pages()`
    returned no text for it (e.g. a scanned page inside an otherwise
    born-digital PDF, which has_text_layer()'s per-document check can't catch).

    Returns "" if no client was supplied or OCR failed after retries — same
    shape as a page with genuinely no content, so callers don't special-case it.
    """
    if openai_client is None:
        logger.warning(
            f"{image_path.name} has no extractable text and no OpenAI client "
            "was supplied for OCR fallback; leaving it blank."
        )
        return ""

    result = ocr.transcribe_image(
        image_path,
        openai_client,
        cache_dir=ocr.OCR_CACHE_DIR / paper_id,
    )
    if result.get("error"):
        logger.warning(f"OCR fallback failed for {image_path.name}: {result['error']}")
    elif result["markdown"]:
        logger.info(f"OCR fallback recovered text for {image_path.name}.")
    return result["markdown"]


def chunk_paper_pages(
    paper: Paper,
    page_images_dir: Path = PAGE_IMAGES_DIR,
    zoom: float = DEFAULT_PAGE_RENDER_ZOOM,
    openai_client: OpenAI | None = None,
) -> list[Chunk]:
    """
    Render every PDF page as a full-page image and pair it with that page's
    markdown text, producing one multimodal Chunk per page.

    This is the primary chunking strategy: rather than embedding parsed text
    and cropped figures as separate chunks, each page's full visual layout
    (body text, figures, tables, formatting) is embedded as a single image
    alongside its markdown, so retrieval can surface a page as a whole unit.

    Args:
        paper: Paper object containing source_file path and metadata.
        page_images_dir: Directory to render page PNGs into.
        zoom: Render scale factor (72 DPI * zoom) passed to render_pdf_pages_as_images.
        openai_client: Used to OCR any individual page `parse_pdf_for_pages()`
            returned no text for (see `_ocr_fallback_for_page`). None disables
            the fallback — such pages are still embedded (as images) but carry
            no text, same as before this fallback existed.

    Returns:
        List of Chunk objects; each payload includes page/page_label and the
        rendered PNG filename (source_file) for the multimodal embed step.
    """
    pdf_path = DATA_DIR / paper.source_file

    pages = parse_pdf_for_pages(pdf_path)
    if not pages:
        logger.warning(f"Paper {paper.id} yielded no page text. Skipping page chunking.")
        return []

    text_by_page = {p["page"]: preprocess_markdown(p["text"]) for p in pages}

    # journal_page = pdf_page + page_offset (same anchoring as chunk_paper / image_extractor)
    page_offset = (
        paper.first_page_number - pages[0]["page"]
        if paper.first_page_number is not None
        else 0
    )

    rendered = render_pdf_pages_as_images(pdf_path, page_images_dir, paper.id, zoom=zoom)
    if not rendered:
        logger.warning(f"Paper {paper.id} yielded no rendered page images. Skipping page chunking.")
        return []

    chunks: list[Chunk] = []
    for i, page_info in enumerate(rendered):
        pdf_page_num = page_info["page"]
        page_text = text_by_page.get(pdf_page_num, "")
        journal_page = pdf_page_num + page_offset

        if not page_text.strip():
            page_text = _ocr_fallback_for_page(
                page_images_dir / page_info["image_file"], paper.id, openai_client
            )

        chunks.append(
            Chunk(
                id=str(uuid.uuid5(PAGE_ID_NAMESPACE, f"{paper.id}:page:{pdf_page_num}")),
                text=_build_page_embed_text(paper, page_text),
                payload=_build_page_payload(
                    paper, i, page_text, page_info["image_file"], journal_page
                ),
            )
        )

    logger.info(f"Paper {paper.id} produced {len(chunks)} page chunks.")
    return chunks


def chunk_scanned_pages(paper: Paper, pages: list[dict]) -> list[Chunk]:
    """
    Build one multimodal Chunk per page of a scanned document.

    The scanned counterpart to `chunk_paper_pages`. That function reads a page's
    markdown out of the PDF's text layer; a scan has none, so the text comes
    from `ocr.ocr_pdf` instead and is passed in here already transcribed. The
    resulting payload is deliberately identical in shape to the born-digital
    page chunks, so scanned and native sources retrieve and cite the same way.

    Page numbering differs in one respect: the number stored is the folio
    *printed on the page*, read off the scan during transcription. A scanned
    book's PDF index rarely matches its printed numbering — front matter is
    numbered separately and a two-up capture halves the count — so the printed
    folio is the only value that makes a citation checkable against a physical
    copy. Pages with no printed folio (title page, part openers) fall back to
    their sequential position.

    Args:
        paper: Paper object supplying title, authors, year, publication.
        pages: Output of `ocr.ocr_pdf` — dicts with 'sheet', 'half', 'sequence',
               'image_file', 'text' and 'printed_page'.

    Returns:
        List of Chunk objects, one per page, ready for embedding and upsert.
    """
    if not pages:
        logger.warning(f"Paper {paper.id} has no transcribed pages. Skipping.")
        return []

    chunks: list[Chunk] = []
    for i, page in enumerate(pages):
        page_text = page.get("text", "").strip()
        if not page_text:
            continue

        page_number = page.get("printed_page") or page["sequence"]

        chunks.append(
            Chunk(
                # Keyed on the source sheet and half rather than the sequence,
                # so re-running after excluding a page or extending the range
                # overwrites each page in place instead of duplicating it.
                id=str(
                    uuid.uuid5(
                        PAGE_ID_NAMESPACE,
                        f"{paper.id}:scan:{page['sheet']}{page['half']}",
                    )
                ),
                text=_build_page_embed_text(paper, page_text),
                payload=_build_page_payload(
                    paper, i, page_text, page["image_file"], page_number
                ),
            )
        )

    logger.info(f"Paper {paper.id} produced {len(chunks)} scanned page chunks.")
    return chunks


# Maps Document.file_type → LangChain Language enum for code-aware splitting.
# Unlisted types fall back to MARKDOWN (good general-purpose prose splitter).
_CODE_LANGUAGE_MAP: dict[str, Language] = {
    "py":   Language.PYTHON,
    "js":   Language.JS,
    "jsx":  Language.JS,
    "ts":   Language.TS,
    "tsx":  Language.TS,
    "c":    Language.C,
    "cpp":  Language.CPP,
    "h":    Language.CPP,
    "hpp":  Language.CPP,
    "java": Language.JAVA,
    "sol":  Language.SOL,
    "md":   Language.MARKDOWN,
}


def _build_doc_embed_text(doc: Document, chunk_text: str) -> str:
    """Prepend document title so the embedding carries provenance context."""
    return f"Title: {doc.title}\n{chunk_text}"


def _build_doc_payload(doc: Document, chunk_index: int, chunk_text: str) -> dict:
    return {
        "source_type": "document",
        "chunk_category": "text",
        "doc_id": doc.id,
        "title": doc.title,
        "file_type": doc.file_type,
        "chunk_index": chunk_index,
        "text": chunk_text,
    }


def chunk_document(doc: Document) -> list[Chunk]:
    """Split a Document into overlapping chunks with metadata payload.

    Code files are split with a language-aware splitter; prose and structured
    files (txt, md, docx) use the markdown splitter which respects headings
    and paragraph boundaries produced by the docx extractor.

    Args:
        doc: Document object with extracted text.

    Returns:
        List of Chunk objects ready for embedding and upsert.
    """
    if not doc.text or not doc.text.strip():
        logger.warning(f"Document '{doc.title}' has empty text. Skipping.")
        return []

    language = _CODE_LANGUAGE_MAP.get(doc.file_type, Language.MARKDOWN)

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    try:
        raw_chunks = text_splitter.split_text(doc.text)
    except Exception as e:
        logger.error(f"Error splitting document '{doc.title}': {e}")
        return []

    chunks: list[Chunk] = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue
        chunks.append(
            Chunk(
                id=str(uuid.uuid5(DOC_ID_NAMESPACE, f"{doc.id}:text:{i}")),
                text=_build_doc_embed_text(doc, chunk_text),
                payload=_build_doc_payload(doc, i, chunk_text),
            )
        )

    logger.info(f"Document '{doc.title}' split into {len(chunks)} chunks.")
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage: load papers from JSON, chunk them, and print chunk info
    paper = load_paper(JSON_DIR / "DaZlHqgIxjgJ.json")
    if not paper:
        logger.error("Failed to load paper for chunking.")
    else:
        try:
            chunks = chunk_paper(paper)
            logger.info(f"Paper '{paper.title}' produced {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"Failed to chunk paper '{paper.title}': {e}")
        

