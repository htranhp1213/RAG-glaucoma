# vector_db_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

Metadata = Dict[str, Any]


@dataclass
class ChromaConfig:
    db_path: str = "./vector_db"
    collection_name: str = "competition_guidelines"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class VectorDB:
    """
    Implements ChromaDB operations:
    - connect
    - create_or_get_collection
    - insert
    - query
    - update
    - upsert
    - delete
    """

    def __init__(self, cfg: ChromaConfig):
        self.cfg = cfg
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[Collection] = None
        self._embedding_fn = SentenceTransformerEmbeddingFunction(model_name=cfg.model_name)

    # ---------- Setup ----------
    def connect(self) -> chromadb.PersistentClient:
        self._client = chromadb.PersistentClient(path=self.cfg.db_path)
        return self._client

    def create_or_get_collection(self, *, metadata: Optional[Dict[str, Any]] = None) -> Collection:
        if self._client is None:
            self.connect()
        assert self._client is not None

        self._collection = self._client.get_or_create_collection(
            name=self.cfg.collection_name,
            embedding_function=self._embedding_fn,
            metadata=metadata,
        )
        return self._collection

    def get_collection(self) -> Collection:
        if self._collection is None:
            self.create_or_get_collection()
        assert self._collection is not None
        return self._collection

    # ---------- Operations ----------
    def insert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Optional[Sequence[Metadata]] = None,
    ) -> None:
        """Insert new records. Will fail if ids already exist."""
        col = self.get_collection()
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]
        col.add(ids=list(ids), documents=list(documents), metadatas=list(metadatas))

    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Optional[Sequence[Metadata]] = None,
    ) -> None:
        """Insert or overwrite records by id."""
        col = self.get_collection()
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]
        col.upsert(ids=list(ids), documents=list(documents), metadatas=list(metadatas))

    def query(
        self,
        query_text: Union[str, List[str]],
        k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """k-NN semantic search."""
        col = self.get_collection()
        if include is None:
            include = ["documents", "metadatas", "distances"]
        query_texts = [query_text] if isinstance(query_text, str) else query_text
        return col.query(query_texts=query_texts, n_results=k, where=where, include=include)

    def update(
        self,
        ids: Sequence[str],
        documents: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[Metadata]] = None,
    ) -> None:
        """
        Update existing records.
        Requires ids + at least one of documents/metadatas.
        """
        col = self.get_collection()
        payload: Dict[str, Any] = {"ids": list(ids)}
        if documents is not None:
            payload["documents"] = list(documents)
        if metadatas is not None:
            payload["metadatas"] = list(metadatas)
        if len(payload) == 1:
            raise ValueError("update() requires documents and/or metadatas.")
        col.update(**payload)

    def delete(self, ids: Optional[Sequence[str]] = None, where: Optional[Dict[str, Any]] = None) -> None:
        """Delete by ids or metadata filter."""
        col = self.get_collection()
        if ids is None and where is None:
            raise ValueError("delete() requires ids or where.")
        col.delete(ids=list(ids) if ids is not None else None, where=where)

    # ---------- Helpers (useful for scripts/tests) ----------
    def get(self, ids: Optional[Sequence[str]] = None, where: Optional[Dict[str, Any]] = None, limit: int = 100) -> Dict[str, Any]:
        col = self.get_collection()
        return col.get(ids=list(ids) if ids is not None else None, where=where, limit=limit)

    def count(self) -> int:
        return self.get_collection().count()

    def reset_collection(self) -> None:
        """Deletes all items in the current collection (safe for tests)."""
        col = self.get_collection()
        n = col.count()
        if n == 0:
            return
        all_ids = col.get(limit=n).get("ids", [])
        if all_ids:
            col.delete(ids=all_ids)
