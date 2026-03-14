from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from transformers import pipeline

from vector_db_utils import VectorDB, ChromaConfig


@dataclass
class EvalConfig:
    db_path: str = "./vector_db"
    collection_name: str = "competition_guidelines"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    eval_dataset_path: str = "glaucoma_eval_dataset_30.json"
    retriever_type: str = "dense"   # "dense" or "sparse"
    out_json: str = "eval_outputs_dense_with_answers.json"

    # local generation model
    generation_model: str = "google/flan-t5-base"
    max_new_tokens: int = 180
    max_context_chars: int = 3500


class DenseRetriever:
    def __init__(self, cfg: EvalConfig):
        db_cfg = ChromaConfig(
            db_path=cfg.db_path,
            collection_name=cfg.collection_name,
            model_name=cfg.embedding_model,
        )
        self.db = VectorDB(db_cfg)
        self.db.connect()
        self.db.create_or_get_collection()

    def invoke(self, question: str, k: int) -> List[Document]:
        res = self.db.query(
            query_text=question,
            k=k,
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
        raw = dense.db.get(where={"modality": "text"}, limit=dense.db.count())

        docs = []
        for text, meta in zip(raw.get("documents", []), raw.get("metadatas", [])):
            docs.append(Document(page_content=text, metadata=meta or {}))

        self.retriever = BM25Retriever.from_documents(docs)

    def invoke(self, question: str, k: int) -> List[Document]:
        self.retriever.k = k
        return self.retriever.invoke(question)


def load_eval_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_generator(cfg: EvalConfig):
    """
    Local text2text generator using Hugging Face Transformers.
    """
    return pipeline(
        task="text2text-generation",
        model=cfg.generation_model,
        tokenizer=cfg.generation_model,
        max_new_tokens=cfg.max_new_tokens,
    )


def trim_contexts(contexts: List[str], max_chars: int) -> str:
    """
    Join contexts while keeping total prompt size manageable.
    """
    joined_parts: List[str] = []
    total = 0

    for i, ctx in enumerate(contexts, start=1):
        piece = f"[Context {i}]\n{ctx.strip()}\n\n"
        if total + len(piece) > max_chars:
            break
        joined_parts.append(piece)
        total += len(piece)

    return "".join(joined_parts).strip()


def generate_answer(question: str, retrieved_contexts: List[str], generator, cfg: EvalConfig) -> str:
    """
    Generate an answer using only retrieved context.
    """
    context_text = trim_contexts(retrieved_contexts, cfg.max_context_chars)

    prompt = f"""
Answer the question using only the provided context.
Do not use outside knowledge.
If the answer cannot be found in the context, say: "I cannot answer from the provided context."

Context:
{context_text}

Question:
{question}

Answer:
""".strip()

    try:
        output = generator(prompt)
        if output and isinstance(output, list):
            # transformers text2text-generation returns list[dict]
            text = output[0].get("generated_text", "").strip()
            return text if text else "I cannot answer from the provided context."
        return "I cannot answer from the provided context."
    except Exception as e:
        return f"[GENERATION_ERROR] {e}"


def run_rag_eval(cfg: EvalConfig) -> List[Dict[str, Any]]:
    eval_data = load_eval_dataset(cfg.eval_dataset_path)

    dense = DenseRetriever(cfg)
    sparse = SparseRetriever(dense)
    generator = build_generator(cfg)

    rows: List[Dict[str, Any]] = []

    for item in eval_data:
        question = item["question"]
        ground_truth = item["ground_truth"]
        difficulty = item.get("difficulty", "")
        source = item.get("source", "")
        row_id = item.get("id", None)

        if cfg.retriever_type == "dense":
            docs = dense.invoke(question, k=cfg.top_k)
        else:
            docs = sparse.invoke(question, k=cfg.top_k)

        retrieved_contexts = [doc.page_content for doc in docs]
        generated_answer = generate_answer(question, retrieved_contexts, generator, cfg)

        rows.append(
            {
                "id": row_id,
                "question": question,
                "ground_truth": ground_truth,
                "difficulty": difficulty,
                "source": source,
                "retrieved_contexts": retrieved_contexts,
                "generated_answer": generated_answer,
                "retriever_type": cfg.retriever_type,
                "top_k": cfg.top_k,
            }
        )

    return rows


def save_json(rows: List[Dict[str, Any]], out_path: str):
    path = Path(out_path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")


def main():
    cfg = EvalConfig()
    rows = run_rag_eval(cfg)
    save_json(rows, cfg.out_json)

    print("\nPreview:")
    for row in rows[:2]:
        print("=" * 100)
        print("Q:", row["question"])
        print("Retriever:", row["retriever_type"])
        print("GT:", row["ground_truth"])
        print("Generated answer:", row["generated_answer"])
        print("Top context snippet:", row["retrieved_contexts"][0][:200] if row["retrieved_contexts"] else "")


if __name__ == "__main__":
    main()