from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from vector_db_utils import VectorDB, ChromaConfig
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)


DEFAULT_PROMPT = """
Answer the question using only the context below.

Rules:
- Give a complete answer in 1-2 sentences.
- Use your own words.
- Be specific and medically accurate.
- Do not copy long text from the context.
- Do not repeat yourself.
- If the answer is not supported by the context, say: I don't know based on the provided context.

Context:
{context}

Question: {question}

Answer:
"""


@dataclass
class RAGConfig:
    db_path: str = "./vector_db"
    collection_name: str = "competition_guidelines"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-base"
    top_k: int = 5
    text_k: int = 3
    image_k: int = 2
    max_new_tokens: int = 200


class ChromaRetriever:
    def __init__(self, cfg: RAGConfig):
        db_cfg = ChromaConfig(
            db_path=cfg.db_path,
            collection_name=cfg.collection_name,
            model_name=cfg.embedding_model,
        )
        self.db = VectorDB(db_cfg)
        self.db.connect()
        self.db.create_or_get_collection()

    def get_relevant_documents(self, question: str, k: int = 5) -> List[Document]:
        res = self.db.query(question, k=k, include=["documents", "metadatas", "distances"])
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

    def get_text_documents(self, question: str, k: int = 2, pool_k: int = 12) -> List[Document]:
        res = self.db.get_collection().query(
            query_texts=[question],
            n_results=pool_k,
            where={"modality": "text"},
            include=["documents", "metadatas", "distances"],
        )

        docs: List[Document] = []
        for text, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            text_clean = (text or "").strip().lower()

            if len(text_clean) < 80:
                continue
            if "all rights reserved" in text_clean:
                continue
            if "notice of rights" in text_clean:
                continue
            if "copyright" in text_clean and len(text_clean) < 400:
                continue
            if text_clean.startswith(". j glaucoma"):
                continue

            meta = dict(meta or {})
            meta["distance"] = float(dist)
            docs.append(Document(page_content=text, metadata=meta))

            if len(docs) >= k:
                break

        return docs

    def get_image_documents(self, question: str, k: int = 2) -> List[Document]:
        res = self.db.get_collection().query(
            query_texts=[question],
            n_results=k,
            where={"modality": "image"},
            include=["documents", "metadatas", "distances"],
        )

        docs: List[Document] = []
        for text, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            meta = dict(meta or {})
            meta["distance"] = float(dist)
            docs.append(Document(page_content=text, metadata=meta))
        return docs


def build_hf_llm(model_name: str, max_new_tokens: int = 200) -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_name_lower = model_name.lower()
    is_seq2seq = ("t5" in model_name_lower) or ("flan" in model_name_lower)

    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            framework="pt",
            max_new_tokens=max_new_tokens,
            truncation=True,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            early_stopping=True,
        )
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            framework="pt",
            max_new_tokens=max_new_tokens,
            truncation=True,
            do_sample=False,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
        )

    return HuggingFacePipeline(pipeline=pipe)


def make_text_query(question: str) -> str:
    q = question.lower()
    for phrase in [
        "in fundus images",
        "fundus images",
        "fundus image",
        "optic disc",
        "retinal",
        "retina",
        "image",
        "images",
        "photo",
        "artifact",
    ]:
        q = q.replace(phrase, " ")
    q = re.sub(r"\s+", " ", q).strip()
    return q if q else question


def make_image_query(question: str) -> str:
    return f"{question} optic disc fundus glaucoma image"


def clean_context_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()

    text = re.sub(r"March \d{4}.*?American Family Physician \d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"www\.[^\s]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Volume \d+, Number \d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\.\s*", "", text)
    text = re.sub(r"\b\d+\b$", "", text).strip()

    return text.strip()


def format_docs_for_generation(docs: List[Document]) -> str:
    parts = []
    seen = set()

    for doc in docs:
        meta = doc.metadata or {}
        modality = meta.get("modality", "text")

        if modality == "image":
            filename = meta.get("filename", "unknown")
            piece = f"Image evidence from {filename}: fundus image centered on the optic disc region, used for glaucoma analysis."
        else:
            piece = clean_context_text(doc.page_content)

        norm = piece.lower()
        if not piece or norm in seen:
            continue
        seen.add(norm)
        parts.append(piece)

    return "\n\n".join(parts)


def format_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    rows = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        rows.append(
            {
                "doc_id": f"Doc {i}",
                "doc_name": meta.get("doc_name", "unknown"),
                "page": meta.get("page_label", meta.get("page", "?")),
                "distance": meta.get("distance"),
                "preview": clean_context_text(doc.page_content)[:180],
                "modality": meta.get("modality", "text"),
            }
        )
    return rows


def clean_answer(answer: str) -> str:
    text = (answer or "").strip()
    text = re.sub(r"\s+", " ", text).strip()

    junk_patterns = [
        r"^\[Doc\s*\d+\]\s*or\s*\[Doc\s*\d+\]\.?$",
        r"^\[Doc\s*\d+\]\.?$",
        r"^No context retrieved\.?$",
        r"^I don't know based on the provided context\.?$",
    ]
    for pat in junk_patterns:
        if re.fullmatch(pat, text, flags=re.IGNORECASE):
            return "I don't know based on the provided context."

    text = text.replace("Context:", "").replace("Question:", "").replace("Answer:", "").strip()

    # Trim repeated sentence fragments
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned_sentences = []
    seen = set()
    for s in sentences:
        s_norm = s.strip().lower()
        if not s_norm or s_norm in seen:
            continue
        seen.add(s_norm)
        cleaned_sentences.append(s.strip())

    text = " ".join(cleaned_sentences).strip()
    return text if text else "I don't know based on the provided context."


def attach_citations(answer: str, docs: List[Document]) -> str:
    answer = (answer or "").strip()

    if answer.lower().startswith("i don't know"):
        return answer

    chosen: List[str] = []

    for i, doc in enumerate(docs, start=1):
        modality = (doc.metadata or {}).get("modality", "text")
        if modality == "text":
            chosen.append(f"[Doc {i}]")
        if len(chosen) >= 2:
            break

    if not chosen:
        for i, _ in enumerate(docs[:2], start=1):
            chosen.append(f"[Doc {i}]")

    cite_str = " ".join(chosen)
    if cite_str and cite_str not in answer:
        answer = f"{answer} {cite_str}"

    return answer


def build_rag_chain(cfg: Optional[RAGConfig] = None):
    cfg = cfg or RAGConfig()
    retriever = ChromaRetriever(cfg)
    llm = build_hf_llm(cfg.llm_model, max_new_tokens=cfg.max_new_tokens)
    prompt = PromptTemplate.from_template(DEFAULT_PROMPT)

    def prepare_inputs(x: Dict[str, Any]) -> Dict[str, Any]:
        question = x["question"]
        q_lower = question.lower()

        image_words = ["image", "fundus", "retina", "retinal", "optic disc", "photo", "artifact"]
        use_images = any(word in q_lower for word in image_words)

        text_query = make_text_query(question)
        image_query = make_image_query(question)

        text_docs = retriever.get_text_documents(
            text_query,
            k=cfg.text_k,
            pool_k=max(cfg.top_k * 3, 12),
        )
        image_docs = retriever.get_image_documents(
            image_query,
            k=cfg.image_k,
        ) if use_images else []

        docs = text_docs + image_docs

        print("DEBUG text query:", text_query)
        print("DEBUG image query:", image_query)
        print("DEBUG total docs:", len(docs))
        print("DEBUG text docs:", len(text_docs))
        print("DEBUG image docs:", len(image_docs))

        return {
            "question": question,
            "docs": docs,
            "text_docs": text_docs,
            "image_docs": image_docs,
            "context": format_docs_for_generation(docs) if docs else "No context retrieved.",
        }

    chain = (
        RunnableLambda(prepare_inputs)
        | RunnablePassthrough.assign(answer=prompt | llm | StrOutputParser())
    )
    return chain


def ask_question(question: str, cfg: Optional[RAGConfig] = None) -> Dict[str, Any]:
    chain = build_rag_chain(cfg)
    out = chain.invoke({"question": question})

    image_sources = []
    for i, doc in enumerate(out["image_docs"], start=1):
        meta = doc.metadata or {}
        image_sources.append(
            {
                "doc_id": f"Image {i}",
                "doc_name": meta.get("doc_name", "unknown"),
                "filename": meta.get("filename", "unknown"),
                "file_path": meta.get("file_path", "unknown"),
                "preview": doc.page_content[:180],
            }
        )

    cleaned = clean_answer(out["answer"])
    final_answer = attach_citations(cleaned, out["docs"])

    return {
        "question": question,
        "answer": final_answer,
        "sources": format_sources(out["docs"]),
        "recommended_images": image_sources,
    }


def main():
    parser = argparse.ArgumentParser(description="OphthoRAG question answering")
    parser.add_argument("--question", required=True, help="Question to ask the RAG system")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--text_k", type=int, default=3)
    parser.add_argument("--image_k", type=int, default=2)
    parser.add_argument("--llm_model", default="google/flan-t5-base")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    cfg = RAGConfig(
        top_k=args.top_k,
        text_k=args.text_k,
        image_k=args.image_k,
        llm_model=args.llm_model,
        max_new_tokens=args.max_new_tokens,
    )

    result = ask_question(args.question, cfg)

    print("\nQUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])

    print(f"\nTOP {len(result['sources'])} RETRIEVED SOURCES:")
    for row in result["sources"]:
        dist = row["distance"]
        dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else "N/A"
        print(
            f"- [{row['doc_id']}] {row['doc_name']} | modality={row['modality']} | "
            f"page={row['page']} | distance={dist_str} | {row['preview']}"
        )

    if result["recommended_images"]:
        print("\nRECOMMENDED IMAGES:")
        for img in result["recommended_images"]:
            print(
                f"- [{img['doc_id']}] {img['filename']} | "
                f"path={img['file_path']} | {img['preview']}"
            )


if __name__ == "__main__":
    main()
