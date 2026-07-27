import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np
from fastapi import HTTPException
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from constants.model_name import ModelName

# Configure logging
logging.basicConfig(filename="app.log", level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaService:
    PERSIST_DIRECTORY = "./chroma_db"
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
        embedding_model: str = ModelName.ALL_MINI_L6_V2.value,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME")
        self.vectorstore = Chroma(
            collection_name=self.format_collection_name(self.collection_name),
            persist_directory=self.PERSIST_DIRECTORY,
            embedding_function=self.embedding_function,
        )

    def find_similar_documents(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict[str, str]] = None,
        where_document: Optional[dict[str, str]] = None,
    ) -> List[Document]:
        documents = self.get_documents()
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k or len(documents),
            filter=filter,
            where_document=where_document,
        )
        filtered_results = [doc for doc, score in results if score < 2.0]
        return filtered_results

    def get_documents(self, file_id: Optional[int] = None) -> List[Dict]:
        try:
            results = (
                self.vectorstore._collection.get(
                    where={"file_id": file_id}, include=["documents", "metadatas"]
                )
                if file_id
                else self.vectorstore._collection.get(
                    include=["documents", "metadatas"]
                )
            )

            return [
                {
                    "id": results["ids"][i],
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
                for i in range(len(results["ids"]))
            ]
        except Exception as e:
            logger.error(f"Error retrieving documents for file_id {file_id}: {str(e)}")
            return []

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

        MAX_COLLECTION_NAME_LENGTH = 63
        if len(formatted_name) > MAX_COLLECTION_NAME_LENGTH:
            formatted_name = formatted_name[:MAX_COLLECTION_NAME_LENGTH].rstrip("_-.")

        return formatted_name


    def find_similarity_score(self, text1: str, text2: str) -> float:
        """
        Compute the cosine similarity score between two strings.

        Args:
            text1: First string input (e.g., user query)
            text2: Second string input (e.g., model answer)

        Returns:
            Cosine similarity score between -1 and 1, where:
            - 1 indicates identical semantic meaning
            - 0 indicates no similarity
            - -1 indicates opposite meaning

        Raises:
            ValueError: If either input string is empty or embedding fails
        """
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

            similarity = dot_product / (norm1 * norm2)
            similarity = np.clip(similarity, -1.0, 1.0)

            return float(similarity)

        except Exception as e:
            logger.error(f"Error computing cosine similarity: {str(e)}")
            raise ValueError(f"Failed to compute cosine similarity: {str(e)}")

    def upsert_documents(
        self,
        documents: List[Document],
        source: str,
    ) -> list[str]:
        """Upsert chunk documents using the chunk's own ID.

        This assumes ``documents`` are already chunks from the extraction layer
        (e.g. ``rag_pipeline``), each with a deterministic ``id`` and ``source``
        in metadata.

        Update semantics: Chroma keys rows by ID. For each provided chunk ID,
        Chroma inserts if missing, otherwise overwrites the stored text,
        embedding, and metadata. After upsert, we delete stale chunks previously
        stored under the same ``source`` but not present in the current list.
        """
        if not documents:
            raise ValueError("documents cannot be empty")

        try:
            for document in documents:
                document.metadata = {
                    k: str(v)
                    for k, v in (document.metadata or {}).items()
                    if v is not None
                }

            filtered = filter_complex_metadata(documents)

            # Prefer Document.id, fall back to metadata["id"].
            ids: list[str] = []
            for document in filtered:
                doc_id = getattr(document, "id", None) or (document.metadata or {}).get(
                    "id"
                )
                if not doc_id:
                    raise ValueError(
                        "Missing deterministic chunk ID (document.id or metadata['id'])."
                    )
                ids.append(str(doc_id))

            texts = [document.page_content for document in filtered]
            metadatas = [document.metadata for document in filtered]
            embeddings = self.embedding_function.embed_documents(texts)
            self.vectorstore._collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            # Drop leftover chunks if this source now has fewer splits.
            existing = self.vectorstore._collection.get(
                where={"source": source},
                include=[],
            )
            stale_ids = [
                doc_id
                for doc_id in (existing.get("ids") or [])
                if doc_id not in ids
            ]
            if stale_ids:
                self.vectorstore._collection.delete(ids=stale_ids)

            return ids
        except ValueError:
            raise
        except Exception as e:
            logger.error(
                f"Error upserting documents for source {source}: {str(e)}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upsert documents for source {source}: {str(e)}",
            )

    def delete_document(self, source: str) -> bool:
        try:
            self.vectorstore._collection.delete(where={"source": source})
            return True
        except Exception as e:
            logger.error(f"Error deleting documents for source {source}: {str(e)}")
            return False

    def get_all_collections(self) -> List[str]:
        try:
            collections = self.vectorstore._client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            logger.error(f"Error retrieving collection names: {str(e)}")
            return []

    def get_retriever(self) -> BaseRetriever:
        return self.vectorstore.as_retriever()
