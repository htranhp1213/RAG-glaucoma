from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from vector_db_utils import VectorDB, ChromaConfig


DOMAIN_QUESTIONS = [
    "What is the primary goal of glaucoma treatment?",
    "How is intraocular pressure managed in glaucoma?",
    "What tests are used to monitor glaucoma progression over time?",
    "When would visual field testing be recommended?",
    "What is the role of optic nerve evaluation in glaucoma care?",
    "How often should patients be followed up after starting glaucoma therapy?",
    "What are common risk factors for glaucoma progression?",
    "What is ocular hypertension and how does it relate to glaucoma risk?",
    "When is laser treatment considered in glaucoma management?",
    "How is angle-closure glaucoma managed differently from open-angle glaucoma?",
]


@dataclass
class CompareConfig:
    db_path: str = "./vector_db"
    collection_name: str = "competition_guidelines"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    out_csv: str = "session4_retrieval_compare.csv"


class DenseRetriever:
    def __init__(self, cfg: CompareConfig):
        db_cfg = ChromaConfig(
            db_path=cfg.db_path,
            collection_name=cfg.collection_name,
            model_name=cfg.embedding_model,
        )
        self.db = VectorDB(db_cfg)
        self.db.connect()
        self.db.create_or_get_collection()

    def invoke(self, question: str, k: int) -> List[Document]:
        res = self.db.get_collection().query(
            query_texts=[question],
            n_results=k,
            where={"modality": "text"},
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        for text, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            meta = dict(meta or {})
            meta["distance"] = float(dist)
            docs.append(Document(page_content=text, metadata=meta))
        return docs


class SparseRetriever:
    def __init__(self, dense: DenseRetriever):
        raw = dense.db.get_collection().get(where={"modality": "text"})

        docs = []
        for text, meta in zip(raw.get("documents", []), raw.get("metadatas", [])):
            docs.append(Document(page_content=text, metadata=meta or {}))

        self.retriever = BM25Retriever.from_documents(docs)

    def invoke(self, question: str, k: int) -> List[Document]:
        self.retriever.k = k
        return self.retriever.invoke(question)


def doc_summary(doc: Document) -> str:
    meta = doc.metadata or {}
    return f"{meta.get('doc_name', 'unknown')}|page={meta.get('page_label', meta.get('page', '?'))}|{doc.page_content[:150].replace(chr(10), ' ')}"


def compare_retrievers(cfg: CompareConfig):
    dense = DenseRetriever(cfg)
    sparse = SparseRetriever(dense)

    rows: List[Dict[str, Any]] = []
    for q in DOMAIN_QUESTIONS:
        dense_docs = dense.invoke(q, k=cfg.top_k)
        sparse_docs = sparse.invoke(q, k=cfg.top_k)

        rows.append(
            {
                "question": q,
                "dense_top1": doc_summary(dense_docs[0]) if dense_docs else "",
                "dense_topk_doc_names": " || ".join((d.metadata or {}).get("doc_name", "unknown") for d in dense_docs),
                "sparse_top1": doc_summary(sparse_docs[0]) if sparse_docs else "",
                "sparse_topk_doc_names": " || ".join((d.metadata or {}).get("doc_name", "unknown") for d in sparse_docs),
            }
        )
    return rows


def save_csv(rows: List[Dict[str, Any]], out_path: str):
    path = Path(out_path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def main():
    cfg = CompareConfig()
    rows = compare_retrievers(cfg)
    save_csv(rows, cfg.out_csv)

    print("\nPreview:")
    for row in rows[:3]:
        print("=" * 100)
        print("Q:", row["question"])
        print("Dense top1:", row["dense_top1"])
        print("Sparse top1:", row["sparse_top1"])


if __name__ == "__main__":
    main()