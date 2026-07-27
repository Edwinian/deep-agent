import logging
import os
import re
from typing import Dict, List, Optional

import bs4
import numpy as np
import requests
from fastapi import HTTPException
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    WEB_PAGE_TIMEOUT_SECONDS = 20
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
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                " ",
                "",
            ],
        )
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

    def get_doc_splits_from_file(self, file_path: str) -> List[Document]:
        file_extension = os.path.splitext(file_path)[1].lower()
        file_loader_map = {
            ".pdf": UnstructuredPDFLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".doc": UnstructuredWordDocumentLoader,
            ".html": UnstructuredHTMLLoader,
            ".txt": lambda x: [Document(page_content=open(x, "r").read())],
        }
        loader_class = file_loader_map.get(file_extension)

        if not loader_class:
            raise ValueError(f"File loader not found: {file_path}")

        try:
            if file_extension == ".txt":
                documents = loader_class(file_path)
            else:
                loader = loader_class(file_path, mode="elements")
                documents = loader.load()

            processed_docs = []
            current_content = ""

            for doc in documents:
                content = doc.page_content.strip()
                if len(content) < 10:
                    current_content += " " + content
                else:
                    if current_content:
                        processed_docs.append(
                            Document(
                                page_content=current_content.strip(),
                                metadata=doc.metadata,
                            )
                        )
                        current_content = ""
                    processed_docs.append(doc)

            if current_content:
                processed_docs.append(
                    Document(
                        page_content=current_content.strip(),
                        metadata=documents[-1].metadata,
                    )
                )

            splits = self.text_splitter.split_documents(processed_docs)
            return splits
        except Exception as e:
            logger.error(f"Failed to load or split document {file_path}: {str(e)}")
            raise ValueError(f"Failed to load or split document {file_path}: {str(e)}")

    def load_web_page(self, url: str, bs_kwargs: dict | None = None) -> list[Document]:
        """Fetch a URL and return its visible text as a LangChain document."""
        response = requests.get(url, timeout=self.WEB_PAGE_TIMEOUT_SECONDS)
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.text, "html.parser", **(bs_kwargs or {}))
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()
        content_root = soup.find("article") or soup.find("main") or soup.body or soup
        text = content_root.get_text(separator="\n", strip=True)
        return [Document(page_content=text, metadata={"source": url})]

    def get_doc_splits_from_web(
        self,
        urls: list[str],
        *,
        bs_kwargs: dict | None = None,
    ) -> List[Document]:
        if not urls:
            raise ValueError("At least one URL is required")

        try:
            documents: list[Document] = []
            for url in urls:
                if not url or not url.strip():
                    raise ValueError("URL cannot be empty or whitespace")
                documents.extend(self.load_web_page(url.strip(), bs_kwargs=bs_kwargs))

            splits = self.text_splitter.split_documents(documents)
            return splits
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to load or split web documents from {urls}: {str(e)}")
            raise ValueError(f"Failed to load or split web documents: {str(e)}") from e

    def get_valid_splits(self, splits: List[Document]) -> List[Document]:
        valid_splits = []

        for split in splits:
            if split.page_content.startswith("IMAGE_CONTENT:"):
                valid_splits.append(split)
                continue

            try:
                len_check = split.page_content.strip() and len(split.page_content) >= 5
                embedding = self.embedding_function.embed_query(split.page_content)

                if all([len_check, embedding is not None, any(embedding)]):
                    valid_splits.append(split)

            except Exception as e:
                logger.error(f"Failed to process split: {str(e)}")
                continue

        return valid_splits

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
        source_pk: int | str,
    ) -> list[str]:
        """Upsert document chunks with deterministic IDs ``{source_pk}_{chunk_idx}``.

        ``chunk_idx`` is the position of each document in the ``documents`` list
        (0-based split order from the caller, e.g. pipeline output).

        How updates work: Chroma keys rows by ID, not by source document. When a
        source row changes, the caller re-chunks the full row and calls this
        again with the same ``source_pk``. Matching IDs are overwritten in place
        (text + embedding + metadata). Chroma does not detect which chunks
        changed — the whole chunk set for that ``source_pk`` is re-materialized.
        If the new split has fewer chunks, leftover higher ``chunk_idx`` IDs are
        deleted so stale vectors are not left behind.
        """
        if not documents:
            raise ValueError("documents cannot be empty")

        try:
            pk = str(source_pk)
            for chunk_idx, document in enumerate(documents):
                metadata = {
                    k: str(v) if v is not None else ""
                    for k, v in (document.metadata or {}).items()
                }
                metadata["source_pk"] = pk
                metadata["chunk_idx"] = str(chunk_idx)
                document.metadata = metadata

            filtered = filter_complex_metadata(documents)
            ids = [f"{pk}_{document.metadata['chunk_idx']}" for document in filtered]
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
                where={"source_pk": pk},
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
                f"Error upserting documents for source_pk {source_pk}: {str(e)}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upsert documents for source_pk {source_pk}: {str(e)}",
            )

    def index_document(self, file_path: str, file_id: int) -> dict[str, str]:
        try:
            splits = self.get_doc_splits_from_file(file_path)
            valid_splits = self.get_valid_splits(splits)
            response = {
                "success": "0",
                "error": "",
            }

            if not valid_splits:
                response["error"] = "No valid document splits found."
                return response

            for split in valid_splits:
                split.metadata["file_id"] = file_id
                split.metadata = {
                    k: str(v) if v is not None else ""
                    for k, v in split.metadata.items()
                }

            if valid_splits:
                filtered_splits = filter_complex_metadata(valid_splits)
                self.vectorstore.add_documents(filtered_splits)
                response["success"] = "1"

            return response
        except Exception as e:
            logger.error(f"Error indexing document with file_id {file_id}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to index {os.path.basename(file_path)}: {str(e)}",
            )

    def index_web_documents(
        self,
        urls: list[str],
        *,
        bs_kwargs: dict | None = None,
    ) -> dict[str, str]:
        try:
            splits = self.get_doc_splits_from_web(urls, bs_kwargs=bs_kwargs)
            valid_splits = self.get_valid_splits(splits)
            response = {
                "success": "0",
                "error": "",
            }

            if not valid_splits:
                response["error"] = "No valid document splits found."
                return response

            for split in valid_splits:
                split.metadata = {
                    k: str(v) if v is not None else ""
                    for k, v in split.metadata.items()
                }
                split.metadata["document_type"] = "web"

            filtered_splits = filter_complex_metadata(valid_splits)
            self.vectorstore.add_documents(filtered_splits)
            response["success"] = "1"
            return response
        except ValueError as e:
            logger.error(f"Error indexing web documents from {urls}: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error indexing web documents from {urls}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to index web documents: {str(e)}",
            )

    def delete_web_documents(self) -> bool:
        try:
            self.vectorstore._collection.delete(where={"document_type": "web"})
            return True
        except Exception as e:
            logger.error(f"Error deleting web documents: {str(e)}")
            return False

    def delete_document(self, file_id: int) -> bool:
        try:
            self.vectorstore._collection.delete(where={"file_id": file_id})
            return True
        except Exception as e:
            logger.error(f"Error deleting document with file_id {file_id}: {str(e)}")
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
