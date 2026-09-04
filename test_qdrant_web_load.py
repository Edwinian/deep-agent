"""Load demo web docs into Qdrant Cloud and verify retrieval.

Usage:
    .venv/bin/python test_qdrant_web_load.py
    .venv/bin/python test_qdrant_web_load.py --query "reward hacking"
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from load_web_documents import URLS, load_web_documents
from qdrant_service import QdrantService

load_dotenv(override=True)

DEFAULT_QUERY = "What are types of reward hacking?"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index Lilian Weng blog posts into Qdrant and smoke-test search."
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Similarity search query after indexing",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip fetch/index; only run search against existing points",
    )
    args = parser.parse_args()

    if not args.skip_load:
        print(f"Indexing {len(URLS)} URLs into Qdrant…")
        for url in URLS:
            print(f"  - {url}")
        result = load_web_documents(URLS)
        if result.get("success") != "1":
            print(f"Index failed: {result.get('error')}", file=sys.stderr)
            return 1
        print("Index complete.\n")
    else:
        print("Skipping load (--skip-load).\n")

    service = QdrantService()
    docs = service.get_documents()
    print(f"Collection: {service.collection_name}")
    print(f"Point count (scrolled): {len(docs)}")

    hits = service.vectorstore.similarity_search(args.query, k=3)
    if not hits:
        print(f"No hits for query: {args.query!r}", file=sys.stderr)
        return 1

    print(f"\nTop {len(hits)} hits for {args.query!r}:")
    for i, doc in enumerate(hits, start=1):
        source = (doc.metadata or {}).get("source", "?")
        preview = " ".join(doc.page_content.split())[:180]
        print(f"\n[{i}] source={source}")
        print(f"    {preview}…")

    print("\nOK — web docs are searchable in Qdrant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
