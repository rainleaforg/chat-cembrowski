"""
Ingest data/knowledge/ (developer-authored site + internal-document markdown)
into Qdrant.

A dedicated script rather than a flag on vectordb.__main__, whose main loop
iterates all 340 local Paper JSONs and would re-embed a stale local corpus
into production — this touches nothing but the knowledge documents.

Usage:
    uv run scripts/ingest_knowledge.py                         # into BAPa-V2
    uv run scripts/ingest_knowledge.py --collection bapa-dev-scratch
"""

import argparse
import logging

from chat_cembrowski.data.chunker import chunk_document
from chat_cembrowski.data.doc_ingestion import (
    KNOWLEDGE_DIR,
    KNOWLEDGE_JSON_DIR,
    ingest_local_docs,
)
from chat_cembrowski.data.serialization import load_documents_from_json, save_document
from chat_cembrowski.data.vectordb import (
    COLLECTION_NAME,
    delete_points_for,
    embed_and_upsert,
    ensure_collection,
    get_qdrant_client,
    get_voyage_client,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest data/knowledge/ into Qdrant.")
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Qdrant collection name (default: {COLLECTION_NAME}).",
    )
    args = parser.parse_args()

    # Re-extracts every file in data/knowledge/ and updates data/knowledge_json/
    # for anything new or edited (by content or by front-matter), clearing
    # processed on those. Idempotent — an unrelated re-run touches nothing.
    ingest_local_docs(docs_dir=KNOWLEDGE_DIR, doc_json_dir=KNOWLEDGE_JSON_DIR)

    client = get_qdrant_client()
    vo = get_voyage_client()

    # doc.processed lives in data/knowledge_json/ and records nothing about
    # *which* collection received the points, so it can only be trusted for
    # the default one. Honouring it for a scratch collection breaks both ways:
    # a scratch run would mark documents processed and the next production run
    # would skip all of them (silently — and ensure_collection creates a
    # missing collection, so a typo'd --collection reads the same), while a
    # scratch run after a production run would ingest nothing at all.
    # So: for any non-default collection, ingest everything and persist nothing.
    is_default_collection = args.collection == COLLECTION_NAME
    if not is_default_collection:
        logger.info(
            "Target collection is '%s', not the default '%s' — re-ingesting "
            "every document and leaving the local processed flags untouched.",
            args.collection,
            COLLECTION_NAME,
        )

    try:
        ensure_collection(client, collection_name=args.collection)

        for doc in load_documents_from_json(KNOWLEDGE_JSON_DIR):
            if doc.processed and is_default_collection:
                logger.info(f"'{doc.title}' unchanged — skipping.")
                continue

            chunks = chunk_document(doc)

            # Same ordering as vectordb.py's __main__: delete -> upsert -> mark
            # processed, so a crash leaves processed False and the next run
            # redoes the whole record instead of leaving it half-indexed.
            delete_points_for(client, "doc_id", doc.id, collection_name=args.collection)

            upserted = embed_and_upsert(
                client, vo, chunks, collection_name=args.collection
            )
            if upserted == 0:
                logger.warning(
                    "'%s' produced no points (%d chunk(s)) — nothing upserted, "
                    "leaving it unprocessed so the next run retries it.",
                    doc.title,
                    len(chunks),
                )
                continue

            if is_default_collection:
                doc.processed = True
                save_document(doc, KNOWLEDGE_JSON_DIR)
            logger.info(f"Ingested '{doc.title}' (kind={doc.kind}).")
    finally:
        client.close()


if __name__ == "__main__":
    main()
