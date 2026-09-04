"""Index Lilian Weng blog posts into Qdrant Cloud for RAG retrieval."""

from __future__ import annotations

import sys

from dotenv import load_dotenv
from fastapi import HTTPException

from qdrant_service import QdrantService

URLS = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]


def load_web_documents(urls: list[str] | None = None) -> dict[str, str]:
    """Fetch URLs and index their content in Qdrant Cloud."""
    load_dotenv(override=True)
    service = QdrantService()
    target_urls = urls or URLS
    service.delete_web_documents(target_urls)
    return service.index_web_documents(target_urls)


def main() -> int:
    try:
        result = load_web_documents()
    except HTTPException as exc:
        print(f"Failed to index web documents: {exc.detail}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Failed to index web documents: {exc}", file=sys.stderr)
        return 1

    if result["success"] != "1":
        print(f"Indexing failed: {result['error']}", file=sys.stderr)
        return 1

    print("Indexed web documents successfully.")
    print(f"Indexed {len(URLS)} URLs into Qdrant Cloud.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
