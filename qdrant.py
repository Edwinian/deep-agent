"""Qdrant Cloud helpers: create collection and payload indexes."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

load_dotenv(override=True)

# all-MiniLM-L6-v2 embedding dimension
DEFAULT_VECTOR_SIZE = 384
# LangChain Qdrant nests document metadata under payload key "metadata".
DEFAULT_PAYLOAD_FIELDS = ("metadata.source", "metadata.file_id")


def get_client() -> QdrantClient:
    """Return a Qdrant Cloud client (HTTPS :443, REST)."""
    return _client()


def resolve_collection_name(collection_name: str | None = None) -> str:
    """Resolve collection name from argument or ``QDRANT_COLLECTION_NAME``."""
    return _collection_name(collection_name)


def _client() -> QdrantClient:
    url = (os.getenv("QDRANT_URL") or "").strip()
    api_key = (os.getenv("QDRANT_API_KEY") or "").strip()
    if not url:
        raise ValueError("QDRANT_URL is not set")
    if not api_key:
        raise ValueError("QDRANT_API_KEY is not set")
    # Cloud URLs are HTTPS on 443. The client default port is 6333, which
    # often times out from restricted networks / Cloud free-tier endpoints.
    return QdrantClient(
        url=url,
        api_key=api_key,
        port=443,
        https=True,
        prefer_grpc=False,
        check_compatibility=False,
        timeout=60,
    )


def _collection_name(collection_name: str | None = None) -> str:
    name = (collection_name or os.getenv("QDRANT_COLLECTION_NAME") or "").strip()
    if not name:
        raise ValueError(
            "collection_name is required (or set QDRANT_COLLECTION_NAME)"
        )
    return name


def create_collection(
    collection_name: str | None = None,
    *,
    vector_size: int = DEFAULT_VECTOR_SIZE,
    distance: rest.Distance = rest.Distance.COSINE,
) -> rest.CollectionInfo:
    """Create a Qdrant collection for MiniLM-sized dense vectors.

    Uses ``QDRANT_COLLECTION_NAME`` when ``collection_name`` is omitted.
    Idempotent: returns existing collection info if it already exists.
    """
    name = _collection_name(collection_name)
    client = _client()

    if client.collection_exists(name):
        return client.get_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config=rest.VectorParams(size=vector_size, distance=distance),
    )
    return client.get_collection(name)


def index_payload(
    field_name: str,
    *,
    collection_name: str | None = None,
    field_schema: rest.PayloadSchemaType = rest.PayloadSchemaType.KEYWORD,
) -> bool:
    """Create a payload index on ``field_name`` for filterable deletes/upserts.

    Uses ``QDRANT_COLLECTION_NAME`` when ``collection_name`` is omitted.
    Returns True when the index is created or already present.
    """
    if not field_name or not field_name.strip():
        raise ValueError("field_name cannot be empty")

    name = _collection_name(collection_name)
    client = _client()

    if not client.collection_exists(name):
        raise ValueError(f"Collection {name!r} does not exist; call create_collection first")

    try:
        client.create_payload_index(
            collection_name=name,
            field_name=field_name.strip(),
            field_schema=field_schema,
        )
    except Exception as exc:
        # Idempotent when the index already exists.
        message = str(exc).lower()
        if "already exists" not in message and "already exist" not in message:
            raise
    return True


if __name__ == "__main__":
    info = create_collection()
    print(f"collection ready: {_collection_name()}")
    print(f"  vectors={info.config.params.vectors}")

    for field in DEFAULT_PAYLOAD_FIELDS:
        index_payload(field)
        print(f"  payload index: {field} (keyword)")
