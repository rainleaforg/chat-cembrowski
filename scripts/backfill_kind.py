"""
One-off backfill: stamp `kind` onto every existing Qdrant point for documents
already in Postgres, predating kind-stamping in the ingestion pipeline (see
ingestion-lambda/ingest.py::_stamp_document_metadata, which now does this
automatically on every future ingestion).

Idempotent — reads documents.kind and overwrites Qdrant's payload, so
re-running just re-stamps the same values.

Usage:
    uv run scripts/backfill_kind.py              # dry run: report what would change
    uv run scripts/backfill_kind.py --apply       # write the payload updates

Reads DATABASE_URL and QDRANT_CLUSTER_ENDPOINT / QDRANT_API_KEY from the
environment or the repo's .env, falling back to the sibling backend's .env
(where the production DATABASE_URL lives) if unset there.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

import asyncpg
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _qdrant_client() -> QdrantClient:
    endpoint = os.environ.get("QDRANT_CLUSTER_ENDPOINT")
    if not endpoint:
        raise SystemExit("QDRANT_CLUSTER_ENDPOINT is not set.")
    return QdrantClient(url=endpoint, api_key=os.environ.get("QDRANT_API_KEY") or None)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill `kind` onto existing Qdrant points from Postgres."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the payload updates. Without this, runs a dry run.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to a .env with DATABASE_URL / Qdrant credentials. Defaults to "
        "the repo's .env, then the sibling backend's .env.",
    )
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file, override=True)
    else:
        load_dotenv(PROJECT_ROOT / ".env")
        load_dotenv(PROJECT_ROOT.parent / "PAAN-cembrowski" / "backend" / ".env")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is not set (repo .env or --env-file).")

    conn = await asyncpg.connect(database_url)
    qdrant = _qdrant_client()

    failed: list[str] = []

    try:
        rows = await conn.fetch(
            "select id::text, kind, collection_name from documents where status = 'completed'"
        )
        logger.info("%d completed document row(s) in Postgres.", len(rows))

        # Stamping the payload is only half the job: query_engine's routed
        # searches filter on `kind`, and a Qdrant filter on an unindexed
        # payload field 400s. ensure_collection() creates this index, but it
        # only runs on an ingestion path -- a collection that has only ever
        # been written by the ingestion lambda has the payload and no index.
        # create_payload_index is idempotent, so this is safe to re-run.
        collections = sorted(
            {row["collection_name"] for row in rows if row["collection_name"]}
        )
        for collection_name in collections:
            if not args.apply:
                logger.info("Would ensure a `kind` index on '%s'.", collection_name)
                continue
            try:
                qdrant.create_payload_index(
                    collection_name=collection_name,
                    field_name="kind",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info("Ensured a `kind` index on '%s'.", collection_name)
            except Exception as e:
                failed.append(f"`kind` index on {collection_name!r}: {e}")

        stamped = 0
        by_kind: dict[str, int] = {}
        unmatched: list[str] = []
        skipped: list[str] = []

        # Every row is handled defensively, and nothing below re-raises. Both
        # columns are nullable and collection_name can name a collection that
        # no longer exists -- and on an --apply run an exception escaping this
        # loop would abort after some rows were already stamped, taking the
        # summary with it and leaving the operator no record of which.
        for row in rows:
            collection_name = row["collection_name"]
            kind = row["kind"]

            if not collection_name or not kind:
                skipped.append(
                    f"{row['id']} (kind={kind!r}, collection={collection_name!r})"
                )
                continue

            condition = Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=row["id"]))]
            )

            try:
                count = qdrant.count(
                    collection_name=collection_name, count_filter=condition, exact=True
                ).count
                if count == 0:
                    unmatched.append(f"{row['id']} ({kind})")
                    continue

                if args.apply:
                    qdrant.set_payload(
                        collection_name=collection_name,
                        payload={"kind": kind},
                        points=condition,
                    )
            except Exception as e:
                failed.append(f"{row['id']} (collection={collection_name!r}): {e}")
                continue

            stamped += 1
            by_kind[kind] = by_kind.get(kind, 0) + 1

        logger.info("")
        verb = "Stamped" if args.apply else "Would stamp"
        logger.info("%s %d work(s): %s", verb, stamped, dict(by_kind))

        if unmatched:
            logger.warning(
                "%d Postgres row(s) have no matching Qdrant points (not yet "
                "ingested, or a different collection): %s",
                len(unmatched),
                unmatched,
            )

        if skipped:
            logger.warning(
                "%d Postgres row(s) skipped for a missing kind or "
                "collection_name: %s",
                len(skipped),
                skipped,
            )

        if failed:
            logger.error(
                "%d Qdrant operation(s) failed (missing collection, or a "
                "transport error): %s",
                len(failed),
                failed,
            )

        if not args.apply:
            logger.info("")
            logger.info("Dry run — no changes written. Re-run with --apply to write.")
    finally:
        await conn.close()
        qdrant.close()

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
