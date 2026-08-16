#type: ignore
"""
Qdrant client factory, collection management, and batch upsert for Chunk objects.
Uses Voyage AI multimodal embeddings (voyage-multimodal-3.5) for both text and image chunks.
"""

import logging
import os
from itertools import islice
from typing import Iterator
from pathlib import Path
from dotenv import load_dotenv

import PIL.Image
import voyageai
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchValue,
    PayloadField,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from .chunker import Chunk

logger = logging.getLogger(__name__)
load_dotenv()


COLLECTION_NAME = "BAPa-V1"

EMBEDDING_MODEL = "voyage-multimodal-3.5"
VECTOR_DIM = 1024
EMBEDDING_DIMENSIONS = VECTOR_DIM  # alias for query_engine compatibility

TEXT_BATCH_SIZE = 64
IMAGE_BATCH_SIZE = 16    # images are token-heavier; keep batches smaller
PAGE_BATCH_SIZE = 8      # full-page renders are heavier still than cropped figures

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QDRANT_LOCAL_PATH = PROJECT_ROOT / "data" / "vectors"
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
PAGE_IMAGES_DIR = PROJECT_ROOT / "data" / "page_images"
PAGE_RENDER_ZOOM = 2.0   # ~144 DPI; see parser.DEFAULT_PAGE_RENDER_ZOOM


def get_qdrant_client() -> QdrantClient:
    """
    Return a QdrantClient configured from environment variables.

    Local dev  → set nothing (uses embedded local DB at ROOT/data/vectors)
    Production → set QDRANT_CLUSTER_ENDPOINT and QDRANT_API_KEY in your
                 environment / .env
    """
    qdrant_url = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_local_path = os.getenv("QDRANT_LOCAL_PATH", str(QDRANT_LOCAL_PATH))

    if qdrant_url:
        logger.info(f"Connecting to Qdrant Cloud at {qdrant_url}")
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    logger.info(f"Using local Qdrant at '{qdrant_local_path}'")
    return QdrantClient(path=qdrant_local_path)


def get_voyage_client() -> voyageai.Client:
    """Return a Voyage AI client. Reads VOYAGE_API_KEY from environment / .env."""
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY not set in environment.")
    return voyageai.Client(api_key=api_key)


def get_openai_client() -> OpenAI:
    """Return an OpenAI client. Reads OPENAI_API_KEY from environment / .env."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment.")
    return OpenAI(api_key=api_key)


def ensure_collection(
    client: QdrantClient,
    recreate: bool = False,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    Create the collection if it doesn't exist.

    Args:
        client:          Active QdrantClient.
        recreate:        If True, drop and recreate (required when changing VECTOR_DIM
                         or switching embedding models).
        collection_name: Qdrant collection name.
    """
    exists = client.collection_exists(collection_name)

    if exists and recreate:
        logger.warning(f"Dropping existing collection '{collection_name}'.")
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info(
            f"Created collection '{collection_name}' "
            f"(model={EMBEDDING_MODEL}, dim={VECTOR_DIM})."
        )
    else:
        logger.info(f"Collection '{collection_name}' already exists — skipping creation.")

    # retrieval.authors filters on this field (Qdrant scroll with a payload
    # filter) to answer "who is X" questions — that filter 400s without an
    # explicit index. paper_id and doc_id back delete_points_for(), which
    # re-indexing runs on every source before upserting. Called unconditionally
    # (not just on fresh collections) since it's idempotent and existing
    # collections predate these fields.
    for field in ("authors", "paper_id", "doc_id"):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )


def delete_points_for(
    client: QdrantClient,
    key: str,
    value: str,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """
    Delete every point belonging to one source, before it is re-indexed.

    Deterministic chunk IDs make a re-upsert overwrite matching points, but
    they cannot remove points an edit no longer produces: shorten a source
    from 10 chunks to 7 and points 7-9 survive, still retrievable, still
    carrying stale text. Clearing by payload filter first is what makes
    re-indexing a true replacement.

    Returns the number of points that existed before deletion (0 when the
    source is new), which the caller uses to warn about clobbered link
    metadata.

    Args:
        client:          Active QdrantClient.
        key:             Payload field identifying the source ("paper_id"/"doc_id").
        value:           That field's value.
        collection_name: Qdrant collection to delete from.
    """
    condition = Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))])

    try:
        existing = client.count(
            collection_name=collection_name, count_filter=condition, exact=True
        ).count
    except Exception as e:
        logger.warning(f"Could not count existing points for {key}={value}: {e}")
        existing = 0

    if not existing:
        return 0

    # link_posters.py writes site_path/poster_id onto poster chunks *after*
    # ingestion via set_payload. Re-indexing drops them, and the only symptom
    # is a citation quietly losing its link, so say so loudly.
    linked = 0
    try:
        linked = client.count(
            collection_name=collection_name,
            count_filter=Filter(
                must=[FieldCondition(key=key, match=MatchValue(value=value))],
                must_not=[IsEmptyCondition(is_empty=PayloadField(key="site_path"))],
            ),
            exact=True,
        ).count
    except Exception as e:
        logger.debug(f"Could not count linked points for {key}={value}: {e}")

    try:
        client.delete(collection_name=collection_name, points_selector=condition)
        logger.info(f"Deleted {existing} existing point(s) for {key}={value} before re-index.")
    except Exception as e:
        logger.error(f"Failed to delete existing points for {key}={value}: {e}")
        return 0

    if linked:
        logger.warning(
            f"{linked} of those point(s) carried website link metadata "
            "(site_path/poster_id). Re-run 'uv run scripts/link_posters.py --apply' "
            "after this ingestion or their citations will render unlinked."
        )

    return existing


def _batched(iterable, n: int) -> Iterator[list]:
    """Yield successive n-sized batches from an iterable."""
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def _embed_text_batch(vo: voyageai.Client, chunks: list[Chunk]) -> list[list[float]]:
    """Embed a batch of text chunks via the Voyage multimodal endpoint."""
    result = vo.multimodal_embed(
        inputs=[[c.text] for c in chunks],
        model=EMBEDDING_MODEL,
        input_type="document",
    )
    return result.embeddings


def _embed_image_batch(
    vo: voyageai.Client,
    chunks: list[Chunk],
    images_dir: Path,
) -> tuple[list[Chunk], list[list[float]]]:
    """
    Embed a batch of image chunks via the Voyage multimodal endpoint.

    Each input is a [text, PIL.Image] pair where text is the caption + paper
    metadata assembled by the chunker.  Chunks whose image file is missing or
    unreadable are skipped; the returned lists are always parallel.
    """
    valid_chunks: list[Chunk] = []
    inputs = []

    for chunk in chunks:
        image_path = images_dir / chunk.payload["source_file"]
        if not image_path.exists():
            logger.warning(f"Image file not found, skipping: {image_path.name}")
            continue
        try:
            img = PIL.Image.open(image_path)
        except Exception as e:
            logger.warning(f"Failed to open image {image_path.name}: {e}")
            continue

        # Always include text if present; Voyage handles text-only or mixed inputs.
        inputs.append([chunk.text, img] if chunk.text else [img])
        valid_chunks.append(chunk)

    if not inputs:
        return [], []

    result = vo.multimodal_embed(
        inputs=inputs,
        model=EMBEDDING_MODEL,
        input_type="document",
    )
    return valid_chunks, result.embeddings


def _chunks_to_points(
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> list[PointStruct]:
    assert len(chunks) == len(embeddings), (
        f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
    )
    return [
        PointStruct(id=chunk.id, vector=embedding, payload=chunk.payload)
        for chunk, embedding in zip(chunks, embeddings)
    ]


def embed_and_upsert(
    client: QdrantClient,
    vo: voyageai.Client,
    chunks: list[Chunk],
    images_dir: Path = IMAGES_DIR,
    collection_name: str = COLLECTION_NAME,
    image_batch_size: int = IMAGE_BATCH_SIZE,
) -> int:
    """
    Embed and upsert a mixed list of text and image Chunk objects into Qdrant.

    Routes by chunk_category: text chunks are embedded as plain text; image
    chunks are embedded as (text, image) pairs using the Voyage multimodal model.
    This also covers full-page chunks from chunk_paper_pages() — they carry
    chunk_category="image" and a rendered page PNG as their source_file.

    Args:
        client:     Active QdrantClient.
        vo:         Active Voyage AI client.
        chunks:     Output of chunk_paper() + chunk_paper_images(), or
                    chunk_paper_pages(), for one or more papers.
        images_dir: Directory containing the image files referenced by
                    image-category chunks' source_file (extracted figures or
                    rendered page PNGs, depending on chunking strategy).
        collection_name: Qdrant collection to upsert into.
        image_batch_size: Batch size for image-category chunks. Full-page
                    renders are token-heavier than cropped figures, so callers
                    embedding pages should pass a smaller value (e.g. PAGE_BATCH_SIZE).

    Returns:
        Total number of points successfully upserted.
    """
    if not chunks:
        logger.warning("embed_and_upsert called with empty chunk list.")
        return 0

    text_chunks  = [c for c in chunks if c.payload.get("chunk_category") == "text"]
    image_chunks = [c for c in chunks if c.payload.get("chunk_category") == "image"]

    total_upserted = 0

    for batch in _batched(text_chunks, TEXT_BATCH_SIZE):
        try:
            embeddings = _embed_text_batch(vo, batch)
        except Exception as e:
            logger.error(f"Voyage text embedding failed: {e}")
            continue
        points = _chunks_to_points(batch, embeddings)
        try:
            client.upsert(collection_name=collection_name, points=points)
            total_upserted += len(points)
            logger.debug(f"Upserted {len(points)} text points.")
        except Exception as e:
            logger.error(f"Qdrant upsert failed (text batch): {e}")

    for batch in _batched(image_chunks, image_batch_size):
        try:
            valid_chunks, embeddings = _embed_image_batch(vo, batch, images_dir)
        except Exception as e:
            logger.error(f"Voyage image embedding failed: {e}")
            continue
        if not valid_chunks:
            continue
        points = _chunks_to_points(valid_chunks, embeddings)
        try:
            client.upsert(collection_name=collection_name, points=points)
            total_upserted += len(points)
            logger.debug(f"Upserted {len(points)} image points.")
        except Exception as e:
            logger.error(f"Qdrant upsert failed (image batch): {e}")

    logger.info(f"Total upserted: {total_upserted} points.")
    return total_upserted


if __name__ == "__main__":
    import argparse
    from . import ocr
    from .chunker import (
        chunk_paper,
        chunk_paper_images,
        chunk_paper_pages,
        chunk_scanned_pages,
        chunk_document,
    )
    from .serialization import load_papers_from_json, save_paper, load_documents_from_json, save_document

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Embed and upsert chunks into Qdrant.")
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Qdrant collection name (default: {COLLECTION_NAME}).",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "page", "split", "scanned"],
        default="auto",
        help=(
            "Chunking strategy. 'auto' (default) uses 'scanned' for a PDF with "
            "no text layer and 'page' otherwise. 'page' renders each PDF page as "
            "a full image paired with its markdown text; 'scanned' does the same "
            "but transcribes the text with OCR, splitting two-up spreads; "
            "'split' uses the original strategy of separate text chunks + "
            "cropped figure chunks."
        ),
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=PAGE_RENDER_ZOOM,
        help=(
            f"Render scale factor (72 DPI * zoom) for --method page "
            f"(default: {PAGE_RENDER_ZOOM})."
        ),
    )
    parser.add_argument(
        "--ocr-zoom",
        type=float,
        default=ocr.DEFAULT_OCR_ZOOM,
        help=(
            f"Render scale factor for --method scanned (default: "
            f"{ocr.DEFAULT_OCR_ZOOM}). Higher than --zoom because the model has "
            "to read these glyphs rather than being handed a text layer."
        ),
    )
    parser.add_argument(
        "--paper-id",
        default=None,
        help="Process only the paper with this ID. Documents are skipped too.",
    )
    parser.add_argument(
        "--ocr-workers",
        type=int,
        default=ocr.DEFAULT_OCR_WORKERS,
        help=(
            f"For --method scanned: pages transcribed concurrently "
            f"(default: {ocr.DEFAULT_OCR_WORKERS})."
        ),
    )
    parser.add_argument(
        "--first-sheet",
        type=int,
        default=None,
        help="For --method scanned: first PDF sheet to ingest (1-based, inclusive).",
    )
    parser.add_argument(
        "--last-sheet",
        type=int,
        default=None,
        help=(
            "For --method scanned: last PDF sheet to ingest (1-based, inclusive). "
            "Use this to cut loose inserts captured after the work itself."
        ),
    )
    parser.add_argument(
        "--exclude-units",
        default="",
        help=(
            "For --method scanned: comma-separated page units to skip, e.g. "
            "'144R,145'. A unit is a sheet number plus L/R for one half of a "
            "two-up spread."
        ),
    )
    args = parser.parse_args()
    collection_name = args.collection
    exclude_units = {u.strip() for u in args.exclude_units.split(",") if u.strip()}

    client = get_qdrant_client()
    vo = get_voyage_client()
    openai_client = None  # built lazily; only the scanned path needs it

    try:
        ensure_collection(client, recreate=False, collection_name=collection_name)

        for paper in load_papers_from_json():
            if args.paper_id and paper.id != args.paper_id:
                continue

            if paper.processed:
                logger.info(
                    f"Paper '{paper.title}' (ID: {paper.id}) already "
                    "processed — skipping.\n"
                )
                continue

            # Catalog-only entry: the work is known but its PDF has not been
            # acquired. There is nothing to chunk, and PAPERS_DIR / "" would
            # resolve to the directory itself.
            if not paper.has_pdf:
                logger.info(
                    f"Paper '{paper.title}' (ID: {paper.id}) has no PDF yet "
                    "— skipping. Collect it into data/papers/ and run ingest_local.\n"
                )
                continue

            method = args.method
            if method == "auto":
                method = (
                    "page"
                    if ocr.has_text_layer(PAPERS_DIR / paper.source_file)
                    else "scanned"
                )
                logger.info(f"Paper {paper.id}: auto-selected '{method}' chunking.")

            if method == "scanned":
                if openai_client is None:
                    openai_client = get_openai_client()
                pages = ocr.ocr_pdf(
                    PAPERS_DIR / paper.source_file,
                    paper_id=paper.id,
                    images_dir=PAGE_IMAGES_DIR,
                    client=openai_client,
                    zoom=args.ocr_zoom,
                    first_sheet=args.first_sheet,
                    last_sheet=args.last_sheet,
                    exclude_units=exclude_units,
                    max_workers=args.ocr_workers,
                )
                chunks = chunk_scanned_pages(paper, pages)
                upsert_kwargs = dict(images_dir=PAGE_IMAGES_DIR, image_batch_size=PAGE_BATCH_SIZE)
            elif method == "page":
                # has_text_layer() only rules out OCR for the whole document; a
                # document it (correctly) calls born-digital can still have
                # individual scanned/image-only pages, so chunk_paper_pages
                # needs the OCR client too, to fall back per-page rather than
                # embedding a page with no text at all.
                if openai_client is None:
                    openai_client = get_openai_client()
                chunks = chunk_paper_pages(
                    paper, PAGE_IMAGES_DIR, zoom=args.zoom, openai_client=openai_client
                )
                upsert_kwargs = dict(images_dir=PAGE_IMAGES_DIR, image_batch_size=PAGE_BATCH_SIZE)
            else:
                chunks = chunk_paper(paper) + chunk_paper_images(paper)
                upsert_kwargs = dict(images_dir=IMAGES_DIR)

            # Replace, don't accumulate. Ordering is deliberate: delete →
            # upsert → mark processed. A crash anywhere leaves processed False,
            # so the next run redoes the whole record rather than leaving it
            # half-indexed.
            delete_points_for(client, "paper_id", paper.id, collection_name=collection_name)

            if embed_and_upsert(client, vo, chunks, collection_name=collection_name, **upsert_kwargs) > 0:
                logger.info(
                    f"Embedded and upserted chunks for paper "
                    f"'{paper.title}' (ID: {paper.id}).\n"
                )
                paper.processed = True
                save_paper(paper)

        for doc in load_documents_from_json():
            if args.paper_id:
                break  # targeting a single paper; documents are out of scope

            if doc.processed:
                logger.info(f"Document '{doc.title}' already processed — skipping.")
                continue

            chunks = chunk_document(doc)

            delete_points_for(client, "doc_id", doc.id, collection_name=collection_name)

            if embed_and_upsert(client, vo, chunks, collection_name=collection_name) > 0:
                logger.info(f"Embedded and upserted chunks for document '{doc.title}'.")
                doc.processed = True
                save_document(doc)

    finally:
        client.close()
