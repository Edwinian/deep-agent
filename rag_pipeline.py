"""RAG extraction layer.

Web, file, poll-based SQLite extract, and CDC push extract.
Transform (chunk/validate) and Chroma load live in ``chroma_service.ChromaService``.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence
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

from db.sqlite_service import SqliteService

logging.basicConfig(filename="app.log", level=logging.INFO)
logger = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

CdcOp = Literal["c", "u", "d", "r", "create", "insert", "update", "delete", "read"]

_CDC_ADD_OPS = frozenset({"c", "r", "create", "insert", "read"})
_CDC_UPDATE_OPS = frozenset({"u", "update"})
_CDC_DELETE_OPS = frozenset({"d", "delete"})


@dataclass
class PollDbData:
    """Result of a DB extract (poll or CDC push) for one collection."""

    add: list[Document] = field(default_factory=list)
    update: list[Document] = field(default_factory=list)
    delete: list[Document] = field(default_factory=list)


@dataclass(frozen=True)
class CdcEvent:
    """One row change in a typical CDC envelope (Debezium-style).

    Expected shape from a CDC bus / Debezium payload::

        {
          "op": "c" | "u" | "d" | "r",
          "before": { ... } | null,
          "after":  { ... } | null,
          "source": { "table": "<table>", ... },
          "ts_ms": 1710000000000
        }

    ``source_table`` may be passed explicitly or taken from ``source["table"]``.
    """

    op: CdcOp
    source_table: str
    after: Mapping[str, Any] | None = None
    before: Mapping[str, Any] | None = None
    ts_ms: int | None = None
    source: Mapping[str, Any] | None = None


class RagPipeline:
    """Extract documents from sources (web, files, DB poll, CDC push)."""

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
    def _quote_ident(name: str) -> str:
        if not _IDENT_RE.match(name):
            raise ValueError(f"Invalid SQL identifier: {name}")
        return f'"{name}"'

    @staticmethod
    def _base_url(url: str) -> str:
        """Return scheme + host with trailing slash, e.g. ``https://digitalcloud.training/``."""
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")
        return f"{parsed.scheme}://{parsed.netloc}/"

    @staticmethod
    def _source_key(source_table: str, source_pk: Any) -> str:
        """Stable Chroma ``source`` value for a source-table row."""
        return f"{source_table}:{source_pk}"

    @staticmethod
    def _row_get(
        row: Mapping[str, Any] | sqlite3.Row,
        key: str,
        default: Any = None,
    ) -> Any:
        try:
            return row[key]
        except (KeyError, IndexError):
            return default

    def _get_chunks(self, docs: list[Document], source: str) -> list[Document]:
        """Split documents and assign deterministic chunk IDs from ``source``."""
        chunks = self.text_splitter.split_documents(docs)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{source}-{chunk_idx}"
            chunk.id = chunk_id
            chunk.metadata = chunk.metadata or {}
            chunk.metadata["id"] = chunk_id
            chunk.metadata["source"] = source
        return chunks

    def _row_to_document(
        self,
        row: Mapping[str, Any] | sqlite3.Row,
        *,
        source_table: str,
        pk_column: str,
        content_columns: Sequence[str],
        doc_type: str = "db",
    ) -> Document:
        """Build one Document from a source row (before chunking)."""
        source_pk = self._row_get(row, pk_column)
        if source_pk is None:
            raise ValueError(
                f"Missing pk column {pk_column!r} for table {source_table!r}"
            )
        source = self._source_key(source_table, source_pk)
        parts: list[str] = []
        for col in content_columns:
            value = self._row_get(row, col)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                parts.append(text)
        page_content = "\n\n".join(parts)
        metadata: dict[str, Any] = {
            "source": source,
            "source_table": source_table,
            "source_pk": str(source_pk),
            "type": doc_type,
        }
        return Document(page_content=page_content, metadata=metadata)

    def _rows_to_chunks(
        self,
        rows: Sequence[Mapping[str, Any] | sqlite3.Row],
        *,
        source_table: str,
        pk_column: str,
        content_columns: Sequence[str],
    ) -> list[Document]:
        chunks: list[Document] = []
        for row in rows:
            doc = self._row_to_document(
                row,
                source_table=source_table,
                pk_column=pk_column,
                content_columns=content_columns,
            )
            if not doc.page_content.strip():
                continue
            source = str(doc.metadata["source"])
            chunks.extend(self._get_chunks([doc], source))
        return chunks

    def _delete_document(
        self,
        *,
        source_table: str,
        source_pk: Any,
        collection: str,
    ) -> Document:
        return Document(
            page_content="",
            metadata={
                "source": self._source_key(source_table, source_pk),
                "source_table": source_table,
                "source_pk": str(source_pk),
                "collection": collection,
                "type": "db_delete",
            },
        )

    def _resolve_content_columns(
        self,
        row_keys: set[str],
        *,
        source_table: str,
        content_columns_by_table: Mapping[str, Sequence[str]],
        pk_column: str,
        updated_at_column: str,
        deleted_at_column: str,
    ) -> list[str]:
        explicit = content_columns_by_table.get(source_table)
        if explicit is not None:
            return list(explicit)
        return self._infer_content_columns(
            row_keys,
            pk_column=pk_column,
            updated_at_column=updated_at_column,
            deleted_at_column=deleted_at_column,
        )

    def _cdc_is_soft_delete(
        self,
        after: Mapping[str, Any] | None,
        *,
        deleted_at_column: str,
    ) -> bool:
        if after is None:
            return False
        if deleted_at_column not in after:
            return False
        return after[deleted_at_column] is not None

    def _get_new_rows_from_db(
        self,
        conn: sqlite3.Connection,
        *,
        source_table: str,
        collection: str,
        pk_column: str = "id",
        deleted_at_column: str | None = "deleted_at",
        rag_sync_table: str = "rag_sync",
    ) -> list[sqlite3.Row]:
        """Rows in source not yet in ``rag_sync`` (and not soft-deleted)."""
        st = self._quote_ident(source_table)
        rs = self._quote_ident(rag_sync_table)
        pk = self._quote_ident(pk_column)
        soft_filter = (
            f"AND s.{self._quote_ident(deleted_at_column)} IS NULL"
            if deleted_at_column
            else ""
        )
        sql = f"""
            SELECT s.*
            FROM {st} AS s
            LEFT JOIN {rs} AS r
              ON r.source_table = ?
             AND r.collection = ?
             AND r.source_pk = s.{pk}
            WHERE r.source_pk IS NULL
              {soft_filter}
        """
        return list(conn.execute(sql, (source_table, collection)).fetchall())

    def _get_update_rows_from_db(
        self,
        conn: sqlite3.Connection,
        *,
        source_table: str,
        collection: str,
        pk_column: str = "id",
        updated_at_column: str = "updated_at",
        deleted_at_column: str | None = "deleted_at",
        rag_sync_table: str = "rag_sync",
    ) -> list[sqlite3.Row]:
        """Synced rows whose source ``updated_at`` is newer than ``synced_at``."""
        st = self._quote_ident(source_table)
        rs = self._quote_ident(rag_sync_table)
        pk = self._quote_ident(pk_column)
        updated_at = self._quote_ident(updated_at_column)
        soft_filter = (
            f"AND s.{self._quote_ident(deleted_at_column)} IS NULL"
            if deleted_at_column
            else ""
        )
        sql = f"""
            SELECT s.*
            FROM {st} AS s
            INNER JOIN {rs} AS r
              ON r.source_table = ?
             AND r.collection = ?
             AND r.source_pk = s.{pk}
            WHERE s.{updated_at} IS NOT NULL
              AND s.{updated_at} > r.synced_at
              {soft_filter}
        """
        return list(conn.execute(sql, (source_table, collection)).fetchall())

    def _get_delete_rows_from_db(
        self,
        conn: sqlite3.Connection,
        *,
        source_table: str,
        collection: str,
        pk_column: str = "id",
        deleted_at_column: str | None = "deleted_at",
        rag_sync_table: str = "rag_sync",
    ) -> list[sqlite3.Row]:
        """Ledger rows to remove from Chroma: soft-deleted or hard-deleted sources.

        Soft delete: source row still exists with ``deleted_at`` set, and is in
        ``rag_sync``. Hard delete: ``rag_sync`` row with no matching source row
        (anti-join). No ``rag_deleted_at`` tombstone — callers should delete the
        Chroma chunks and then delete the ``rag_sync`` ledger row.
        """
        st = self._quote_ident(source_table)
        rs = self._quote_ident(rag_sync_table)
        pk = self._quote_ident(pk_column)
        soft_rows: list[sqlite3.Row] = []
        if deleted_at_column:
            deleted_at = self._quote_ident(deleted_at_column)
            soft_sql = f"""
                SELECT r.source_table AS source_table,
                       r.source_pk AS source_pk,
                       r.collection AS collection,
                       r.synced_at AS synced_at
                FROM {rs} AS r
                INNER JOIN {st} AS s
                  ON s.{pk} = r.source_pk
                WHERE r.source_table = ?
                  AND r.collection = ?
                  AND s.{deleted_at} IS NOT NULL
            """
            soft_rows = list(
                conn.execute(soft_sql, (source_table, collection)).fetchall()
            )
        hard_sql = f"""
            SELECT r.source_table AS source_table,
                   r.source_pk AS source_pk,
                   r.collection AS collection,
                   r.synced_at AS synced_at
            FROM {rs} AS r
            LEFT JOIN {st} AS s
              ON s.{pk} = r.source_pk
            WHERE r.source_table = ?
              AND r.collection = ?
              AND s.{pk} IS NULL
        """
        hard_rows = list(conn.execute(hard_sql, (source_table, collection)).fetchall())

        seen: set[Any] = set()
        merged: list[sqlite3.Row] = []
        for row in soft_rows + hard_rows:
            key = row["source_pk"]
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
        return merged

    def _list_source_tables(
        self,
        conn: sqlite3.Connection,
        *,
        rag_sync_table: str = "rag_sync",
    ) -> list[str]:
        """User tables excluding ``rag_sync`` and SQLite internals."""
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
              AND name != ?
            ORDER BY name
            """,
            (rag_sync_table,),
        ).fetchall()
        return [str(row["name"]) for row in rows]

    def _table_columns(self, conn: sqlite3.Connection, table_name: str) -> set[str]:
        quoted = self._quote_ident(table_name)
        return {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({quoted})").fetchall()
        }

    def _infer_content_columns(
        self,
        columns: set[str],
        *,
        pk_column: str,
        updated_at_column: str,
        deleted_at_column: str,
    ) -> list[str]:
        skip = {pk_column, updated_at_column, deleted_at_column}
        return sorted(col for col in columns if col not in skip)

    def poll_db(
        self,
        conn_string: str,
        *,
        content_columns_by_table: dict[str, Sequence[str]] | None = None,
        pk_column: str = "id",
        updated_at_column: str = "updated_at",
        deleted_at_column: str = "deleted_at",
        rag_sync_table: str = "rag_sync",
    ) -> dict[str, PollDbData]:
        """Poll all SQLite source tables and return deltas keyed by collection.

        Each user table (except ``rag_sync``) is treated as one collection with
        the same name. Does **not** write to Chroma or update ``rag_sync``.

        Expected ``rag_sync`` shape (no ``rag_deleted_at``):

        .. code-block:: sql

            CREATE TABLE rag_sync (
                source_table TEXT NOT NULL,
                source_pk    INTEGER NOT NULL,
                collection   TEXT NOT NULL,
                synced_at    TEXT NOT NULL,
                PRIMARY KEY (source_table, source_pk, collection)
            );
        """
        content_columns_by_table = content_columns_by_table or {}
        db = SqliteService(conn_string)

        with db as conn:
            results: dict[str, PollDbData] = {}
            for source_table in self._list_source_tables(
                conn, rag_sync_table=rag_sync_table
            ):
                collection = source_table
                columns = self._table_columns(conn, source_table)
                if pk_column not in columns:
                    logger.warning(
                        "Skipping table %s: missing pk column %s",
                        source_table,
                        pk_column,
                    )
                    continue

                content_columns = list(
                    content_columns_by_table.get(source_table)
                    or self._infer_content_columns(
                        columns,
                        pk_column=pk_column,
                        updated_at_column=updated_at_column,
                        deleted_at_column=deleted_at_column,
                    )
                )
                if not content_columns:
                    logger.warning(
                        "Skipping table %s: no content columns", source_table
                    )
                    continue
                for col in content_columns:
                    self._quote_ident(col)

                deleted_col = (
                    deleted_at_column if deleted_at_column in columns else None
                )
                has_updated_at = updated_at_column in columns

                new_rows = self._get_new_rows_from_db(
                    conn,
                    source_table=source_table,
                    collection=collection,
                    pk_column=pk_column,
                    deleted_at_column=deleted_col,
                    rag_sync_table=rag_sync_table,
                )
                updated_rows = (
                    self._get_update_rows_from_db(
                        conn,
                        source_table=source_table,
                        collection=collection,
                        pk_column=pk_column,
                        updated_at_column=updated_at_column,
                        deleted_at_column=deleted_col,
                        rag_sync_table=rag_sync_table,
                    )
                    if has_updated_at
                    else []
                )
                delete_rows = self._get_delete_rows_from_db(
                    conn,
                    source_table=source_table,
                    collection=collection,
                    pk_column=pk_column,
                    deleted_at_column=deleted_col,
                    rag_sync_table=rag_sync_table,
                )

                add_docs = self._rows_to_chunks(
                    new_rows,
                    source_table=source_table,
                    pk_column=pk_column,
                    content_columns=content_columns,
                )
                update_docs = self._rows_to_chunks(
                    updated_rows,
                    source_table=source_table,
                    pk_column=pk_column,
                    content_columns=content_columns,
                )
                delete_docs = [
                    self._delete_document(
                        source_table=source_table,
                        source_pk=row["source_pk"],
                        collection=collection,
                    )
                    for row in delete_rows
                ]
                results[collection] = PollDbData(
                    add=add_docs,
                    update=update_docs,
                    delete=delete_docs,
                )
            return results

    @staticmethod
    def _coerce_cdc_event(event: CdcEvent | Mapping[str, Any]) -> CdcEvent:
        """Normalize a Debezium-style CDC dict or :class:`CdcEvent`."""
        if isinstance(event, CdcEvent):
            return event

        source = event.get("source")
        source_map = source if isinstance(source, Mapping) else None
        source_table = event.get("source_table") or (
            source_map.get("table") if source_map else None
        )
        if not source_table:
            raise ValueError(
                "CDC event missing source_table "
                "(expected top-level source_table or source.table)"
            )

        op = event.get("op")
        if not op:
            raise ValueError("CDC event missing op")

        after = event.get("after")
        before = event.get("before")
        ts_ms = event.get("ts_ms")
        return CdcEvent(
            op=op,  # type: ignore[arg-type]
            source_table=str(source_table),
            after=after if isinstance(after, Mapping) else None,
            before=before if isinstance(before, Mapping) else None,
            ts_ms=int(ts_ms) if ts_ms is not None else None,
            source=source_map,
        )

    def push_db(
        self,
        events: Sequence[CdcEvent | Mapping[str, Any]],
        *,
        content_columns_by_table: dict[str, Sequence[str]] | None = None,
        pk_column: str = "id",
        updated_at_column: str = "updated_at",
        deleted_at_column: str = "deleted_at",
    ) -> dict[str, PollDbData]:
        """Convert CDC row-change events into extract deltas keyed by collection.

        Intended to be called from a CDC consumer (e.g. Debezium / Kafka).
        Does **not** write to Chroma or update ``rag_sync``.

        Each event follows a typical CDC envelope::

            {
              "op": "c" | "u" | "d" | "r",   # also: create/insert/update/delete/read
              "before": { ... } | null,      # pre-image (required for deletes)
              "after":  { ... } | null,      # post-image (required for c/u/r)
              "source": { "table": "<name>", ... },
              "ts_ms": 1710000000000
            }

        Soft-deleted updates (``after.deleted_at`` set) are routed to ``delete``.
        Collection key equals ``source_table``.
        """
        content_columns_by_table = content_columns_by_table or {}
        results: dict[str, PollDbData] = {}

        for raw in events:
            event = self._coerce_cdc_event(raw)
            op = str(event.op).lower()
            source_table = event.source_table
            collection = source_table
            bucket = results.setdefault(collection, PollDbData())

            if op in _CDC_DELETE_OPS or self._cdc_is_soft_delete(
                event.after, deleted_at_column=deleted_at_column
            ):
                row = event.before if op in _CDC_DELETE_OPS else event.after
                # Hard delete: pk lives on before; soft delete: on after.
                if row is None:
                    row = event.before or event.after
                if row is None or pk_column not in row:
                    logger.warning(
                        "Skipping CDC delete for %s: missing row/%s",
                        source_table,
                        pk_column,
                    )
                    continue
                bucket.delete.append(
                    self._delete_document(
                        source_table=source_table,
                        source_pk=row[pk_column],
                        collection=collection,
                    )
                )
                continue

            if op not in _CDC_ADD_OPS and op not in _CDC_UPDATE_OPS:
                logger.warning(
                    "Skipping CDC event for %s: unknown op %r",
                    source_table,
                    event.op,
                )
                continue

            row = event.after
            if row is None or pk_column not in row:
                logger.warning(
                    "Skipping CDC %s for %s: missing after/%s",
                    op,
                    source_table,
                    pk_column,
                )
                continue

            content_columns = self._resolve_content_columns(
                set(row.keys()),
                source_table=source_table,
                content_columns_by_table=content_columns_by_table,
                pk_column=pk_column,
                updated_at_column=updated_at_column,
                deleted_at_column=deleted_at_column,
            )
            if not content_columns:
                logger.warning(
                    "Skipping CDC %s for %s: no content columns",
                    op,
                    source_table,
                )
                continue

            chunks = self._rows_to_chunks(
                [row],
                source_table=source_table,
                pk_column=pk_column,
                content_columns=content_columns,
            )
            if op in _CDC_ADD_OPS:
                bucket.add.extend(chunks)
            else:
                bucket.update.extend(chunks)

        return results

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
