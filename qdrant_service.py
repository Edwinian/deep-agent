"""Qdrant Cloud vector store used by RAG retrieve / ingest paths."""

from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import HTTPException
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http import models as rest

from constants.model_name import ModelName
from qdrant import (
    DEFAULT_PAYLOAD_FIELDS,
    create_collection,
    get_client,
    index_payload,
    resolve_collection_name,
)

load_dotenv(override=True)

logging.basicConfig(filename="app.log", level=logging.INFO)
logger = logging.getLogger(__name__)

# Stable namespace so logical chunk IDs map to the same Qdrant point UUIDs.
_POINT_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _point_id(logical_id: str) -> str:
    """Map an arbitrary chunk id string to a Qdrant-compatible UUID string."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, logical_id))


def _source_filter(source: str) -> rest.Filter:
    return rest.Filter(
        must=[
            rest.FieldCondition(
                key="metadata.source",
                match=rest.MatchValue(value=source),
            )
        ]
    )


def _file_id_filter(file_id: str) -> rest.Filter:
    return rest.Filter(
        must=[
            rest.FieldCondition(
                key="metadata.file_id",
                match=rest.MatchValue(value=str(file_id)),
            )
        ]
    )


class QdrantService:
    """RAG vector operations against Qdrant Cloud (drop-in for ``ChromaService``)."""

    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
    FILE_EXTENSIONS = [
        ".pdf",
        ".doc",
        ".docx",
        ".txt",
    ]

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model: str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        model_name = (
            embedding_model
            or (os.getenv("EMBEDDING_MODEL") or "").strip()
            or ModelName.ALL_MINI_L6_V2.value
        )
        self.embedding_function = HuggingFaceEmbeddings(model_name=model_name)
        self.collection_name = self.format_collection_name(
            collection_name or resolve_collection_name()
        )
        self.client = get_client()
        create_collection(self.collection_name)
        for field in DEFAULT_PAYLOAD_FIELDS:
            index_payload(field, collection_name=self.collection_name)
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_function,
        )

    def format_collection_name(self, name: str) -> str:
        if not name or not name.strip():
            raise HTTPException(
                status_code=400, detail="Collection name cannot be empty or whitespace."
            )

        formatted_name = name.lower().replace(" ", "_")
        formatted_name = re.sub(r"[^a-z0-9_\-.]", "", formatted_name)
        formatted_name = formatted_name.strip("_-.")

        if not formatted_name:
            raise HTTPException(
                status_code=400,
                detail="Collection name is invalid after formatting.",
            )

        max_len = 63
        if len(formatted_name) > max_len:
            formatted_name = formatted_name[:max_len].rstrip("_-.")

        return formatted_name

    def find_similar_documents(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict[str, str]] = None,
        where_document: Optional[dict[str, str]] = None,
    ) -> List[Document]:
        del where_document  # Chroma-only; not used on Qdrant.
        qdrant_filter = None
        if filter:
            must = [
                rest.FieldCondition(
                    key=f"metadata.{key}",
                    match=rest.MatchValue(value=str(value)),
                )
                for key, value in filter.items()
            ]
            qdrant_filter = rest.Filter(must=must)

        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=qdrant_filter,
        )
        # Cosine similarity scores are typically in [-1, 1]; keep positive matches.
        return [doc for doc, score in results if score >= 0.0]

    def get_documents(self, file_id: Optional[int] = None) -> List[Dict]:
        try:
            scroll_filter = _file_id_filter(str(file_id)) if file_id is not None else None
            points: list[Any] = []
            next_offset = None
            while True:
                batch, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=scroll_filter,
                    limit=100,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points.extend(batch)
                if next_offset is None:
                    break

            documents: list[dict] = []
            for point in points:
                payload = point.payload or {}
                metadata = payload.get("metadata") or {}
                content = payload.get("page_content") or ""
                documents.append(
                    {
                        "id": str(point.id),
                        "content": content,
                        "metadata": metadata,
                    }
                )
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents for file_id {file_id}: {str(e)}")
            return []

    def find_similarity_score(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two strings via the embedding model."""
        if not text1 or not text1.strip():
            raise ValueError("text1 cannot be empty")
        if not text2 or not text2.strip():
            raise ValueError("text2 cannot be empty")

        try:
            embedding1 = self.embedding_function.embed_query(text1)
            embedding2 = self.embedding_function.embed_query(text2)

            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                logger.warning("One of the embeddings has zero norm")
                return 0.0

            similarity = float(np.clip(dot_product / (norm1 * norm2), -1.0, 1.0))
            return similarity
        except Exception as e:
            logger.error(f"Error computing cosine similarity: {str(e)}")
            raise ValueError(f"Failed to compute cosine similarity: {str(e)}") from e

    def _scroll_ids_for_source(self, source: str) -> list[str]:
        ids: list[str] = []
        next_offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=_source_filter(source),
                limit=100,
                offset=next_offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.extend(str(point.id) for point in batch)
            if next_offset is None:
                break
        return ids

    def upsert_documents(
        self,
        documents: List[Document],
        source: str,
    ) -> list[str]:
        """Upsert chunk documents by deterministic chunk ID, then drop stale IDs."""
        if not documents:
            raise ValueError("documents cannot be empty")

        try:
            for document in documents:
                document.metadata = {
                    k: str(v)
                    for k, v in (document.metadata or {}).items()
                    if v is not None
                }
                document.metadata.setdefault("source", source)

            filtered = filter_complex_metadata(documents)

            logical_ids: list[str] = []
            for document in filtered:
                doc_id = getattr(document, "id", None) or (document.metadata or {}).get(
                    "id"
                )
                if not doc_id:
                    raise ValueError(
                        "Missing deterministic chunk ID (document.id or metadata['id'])."
                    )
                logical_id = str(doc_id)
                logical_ids.append(logical_id)
                document.metadata["id"] = logical_id

            point_ids = [_point_id(logical_id) for logical_id in logical_ids]
            self.vectorstore.add_documents(filtered, ids=point_ids)

            existing_ids = self._scroll_ids_for_source(source)
            keep = set(point_ids)
            stale_ids = [doc_id for doc_id in existing_ids if doc_id not in keep]
            if stale_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=rest.PointIdsList(points=stale_ids),
                )

            return point_ids
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error upserting documents for source {source}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upsert documents for source {source}: {str(e)}",
            ) from e

    def delete_document(self, source: str) -> bool:
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest.FilterSelector(filter=_source_filter(source)),
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting documents for source {source}: {str(e)}")
            return False

    def get_all_collections(self) -> List[str]:
        try:
            response = self.client.get_collections()
            return [collection.name for collection in response.collections]
        except Exception as e:
            logger.error(f"Error retrieving collection names: {str(e)}")
            return []

    def get_retriever(self) -> BaseRetriever:
        return self.vectorstore.as_retriever()

    def delete_web_documents(self, urls: list[str] | None = None) -> None:
        """Delete indexed chunks for the given web URLs (or all web_page sources)."""
        if urls:
            for url in urls:
                self.delete_document(url)
            return

        next_offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="metadata.type",
                            match=rest.MatchValue(value="web_page"),
                        )
                    ]
                ),
                limit=100,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            sources = {
                str((point.payload or {}).get("metadata", {}).get("source") or "")
                for point in batch
            }
            for source in sources:
                if source:
                    self.delete_document(source)
            if next_offset is None:
                break

    def index_web_documents(self, urls: list[str]) -> dict[str, str]:
        """Fetch URLs, chunk, and upsert into Qdrant."""
        from rag_pipeline import RagPipeline

        if not urls:
            return {"success": "0", "error": "urls cannot be empty"}

        try:
            pipeline = RagPipeline(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            for url in urls:
                raw_docs = pipeline._get_web_page_documents(url)
                chunks = pipeline._get_chunks(raw_docs, url)
                self.upsert_documents(chunks, source=url)
            return {"success": "1", "error": ""}
        except Exception as e:
            logger.error(f"Error indexing web documents: {str(e)}")
            return {"success": "0", "error": str(e)}
