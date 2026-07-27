"""RAG extraction layer.

Web and file extract for now. Future: poll-based DB extract and CDC push extract.
Transform (chunk/validate) and Chroma load live in ``chroma_service.ChromaService``.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

import bs4
import requests
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(filename="app.log", level=logging.INFO)
logger = logging.getLogger(__name__)


class RagPipeline:
    """Extract documents from sources (web, files; later DB poll / CDC)."""

    WEB_PAGE_TIMEOUT_SECONDS = 20

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
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

    @staticmethod
    def _base_url(url: str) -> str:
        """Return scheme + host with trailing slash, e.g. ``https://digitalcloud.training/``."""
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")
        return f"{parsed.scheme}://{parsed.netloc}/"

    def _get_chunks(self, docs: list[Document], id_prefix: str) -> list[Document]:
        """Split documents and assign deterministic chunk IDs."""
        chunks = self.text_splitter.split_documents(docs)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{id_prefix}-{chunk_idx}"
            chunk.id = chunk_id
            chunk.metadata["id"] = chunk_id
        return chunks

    def _get_web_page_documents(self, url: str, bs_kwargs: dict | None = None) -> list[Document]:
        response = requests.get(url, timeout=self.WEB_PAGE_TIMEOUT_SECONDS)
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.text, "html.parser", **(bs_kwargs or {}))
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()
        content_root = soup.find("article") or soup.find("main") or soup.body or soup
        text = content_root.get_text(separator="\n", strip=True)
        return [Document(page_content=text, metadata={"source": url, "type": "web_page"})]

    def extract_web_page(
        self, url: str, bs_kwargs: dict | None = None
    ) -> list[Document]:
        """Fetch a URL and return its visible text as a LangChain document."""
        docs = self._get_web_page_documents(url, bs_kwargs)
        base_url = self._base_url(url)
        return self._get_chunks(docs, base_url)

    def _get_file_documents(self, file_path: str) -> list[Document]:
        """Load raw LangChain documents from a local file path."""
        file_extension = os.path.splitext(file_path)[1].lower()
        file_loader_map = {
            ".pdf": UnstructuredPDFLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".doc": UnstructuredWordDocumentLoader,
            ".html": UnstructuredHTMLLoader,
            ".txt": lambda x: [
                Document(
                    page_content=open(x, "r", encoding="utf-8", errors="ignore").read()
                )
            ],
        }
        loader_class = file_loader_map.get(file_extension)

        if not loader_class:
            raise ValueError(f"File loader not found: {file_path}")

        if file_extension == ".txt":
            return loader_class(file_path)

        loader = loader_class(file_path, mode="elements")
        documents = loader.load()
        for document in documents:
            document.metadata["type"] = file_extension
        return documents

    def _handle_file_fragments(
        self, documents: list[Document], *, min_fragment_len: int = 10
    ) -> list[Document]:
        """Merge short loaded fragments into surrounding content.

        Some file loaders return many small elements. This buffers fragments
        whose text is shorter than ``min_fragment_len`` and merges them into
        the next larger fragment.
        """
        if not documents:
            return []

        processed_docs: list[Document] = []
        current_content = ""

        for doc in documents:
            content = doc.page_content.strip()
            if len(content) < min_fragment_len:
                current_content += " " + content
                continue

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

        return processed_docs

    def extract_file(self, file_path: str) -> list[Document]:
        """Load a local file and return extracted LangChain documents."""
        try:
            documents = self._get_file_documents(file_path)
            processed_docs = self._handle_file_fragments(documents)
            return self._get_chunks(processed_docs, file_path)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to load document %s: %s", file_path, e)
            return []
