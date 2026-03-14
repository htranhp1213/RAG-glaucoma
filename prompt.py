# from __future__ import annotations

# import argparse
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional

# from vector_db_utils import VectorDB, ChromaConfig

# # LangChain / HF pieces for Session 4
# from langchain_core.documents import Document
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# from langchain_community.llms import HuggingFacePipeline
# from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, pipeline



# # DEFAULT_PROMPT = """
# # You are a helpful assistant for glaucoma-related question answering.

# # Use only the provided context.

# # Instructions:
# # - First, directly answer the question in 1 sentence.
# # - Then explain briefly in 2-4 sentences using the retrieved evidence.
# # - If image context is available, mention the retrieved image artifacts only if they are relevant.
# # - If the text context supports the answer, cite the supporting documents as [Doc X].
# # - Do not output placeholder citations like [Doc X].
# # - You may cite more than one source, such as [Doc 1], [Doc 2].
# # - If the answer is not supported by the context, say: "I don't know based on the provided context."
# # Text Context:
# # {text_context}

# # Image context:
# # {image_context}

# DEFAULT_PROMPT = """
# Answer the question using only the context below.

# Instructions:
# - First give a direct answer in 1 sentence.
# - Then add 1-3 short supporting sentences.
# - Write in your own words.
# - If useful, cite the answer using the exact document labels already shown in the context, for example [Doc 1] or [Doc 2].
# - Do not repeat the instructions.
# - If the answer is not supported by the context, say: I don't know based on the provided context.


# Context:
# {context}

# Question:
# {question}

# Final answer:
# """


# @dataclass
# class RAGConfig:
#     db_path: str = "./vector_db"
#     collection_name: str = "competition_guidelines"
#     embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
#     llm_model: str = "google/flan-t5-base"
#     top_k: int = 5
#     text_k: int = 3
#     image_k: int = 2
#     max_new_tokens: int = 200


# class ChromaRetriever:
#     def __init__(self, cfg: RAGConfig):
#         db_cfg = ChromaConfig(
#             db_path=cfg.db_path,
#             collection_name=cfg.collection_name,
#             model_name=cfg.embedding_model,
#         )
#         self.db = VectorDB(db_cfg)
#         self.db.connect()
#         self.db.create_or_get_collection()

#     def get_relevant_documents(self, question: str, k: int = 5) -> List[Document]:
#         res = self.db.query(question, k=k, include=["documents", "metadatas", "distances"])
#         docs = []
#         for text, meta, dist in zip(
#             res["documents"][0],
#             res["metadatas"][0],
#             res["distances"][0],
#         ):
#             # print("DEBUG META:", meta)
#             meta = dict(meta or {})
#             meta["distance"] = float(dist)
#             docs.append(Document(page_content=text, metadata=meta))
#         return docs
    
#     # NEW: safest text retrieval = query mixed pool, then manually exclude images
#     # def get_text_documents(self, question: str, k: int = 3, pool_k: int = 12) -> List[Document]:
#     #     mixed_docs = self.get_relevant_documents(question, k=pool_k)
#     #     text_docs: List[Document] = []

#     #     for doc in mixed_docs:
#     #         modality = (doc.metadata or {}).get("modality", "text")
#     #         if modality != "image":
#     #             text_docs.append(doc)
#     #         if len(text_docs) >= k:
#     #             break

#     #     return text_docs
#     def get_text_documents(self, question: str, k: int = 3) -> List[Document]:
#         res = self.db.get_collection().query(
#             query_texts=[question],
#             n_results=k,
#             where={"modality": "text"},
#             include=["documents", "metadatas", "distances"],
#         )

#         docs = []
#         for text, meta, dist in zip(
#             res["documents"][0],
#             res["metadatas"][0],
#             res["distances"][0],
#         ):
#             text_clean = (text or "").strip().lower()

#             # skip likely boilerplate chunks
#             if "all rights reserved" in text_clean:
#                 continue
#             if "notice of rights" in text_clean:
#                 continue
#             if "page " in text_clean and len(text_clean) < 250:
#                 continue

#             meta = dict(meta or {})
#             meta["distance"] = float(dist)
#             docs.append(Document(page_content=text, metadata=meta))

#             if len(docs) >= k:
#                 break
#         return docs

#     # NEW: dedicated image retrieval using metadata filter
#     def get_image_documents(self, question: str, k: int = 2) -> List[Document]:
#         res = self.db.get_collection().query(
#             query_texts=[question],
#             n_results=k,
#             where={"modality": "image"},
#             include=["documents", "metadatas", "distances"],
#         )

#         docs = []
#         for text, meta, dist in zip(
#             res["documents"][0],
#             res["metadatas"][0],
#             res["distances"][0],
#         ):
#             meta = dict(meta or {})
#             meta["distance"] = float(dist)
#             docs.append(Document(page_content=text, metadata=meta))
#         return docs


# def build_hf_llm(model_name: str, max_new_tokens: int = 220) -> HuggingFacePipeline:
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model_name_lower = model_name.lower()

#     is_seq2seq = ("t5" in model_name_lower) or ("flan" in model_name_lower)

#     if is_seq2seq:
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#         pipe = pipeline(
#             "text2text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             max_new_tokens=max_new_tokens,
#             truncation=True,
#             do_sample=False,
#         )
#     else:
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token

#         model = AutoModelForCausalLM.from_pretrained(model_name)
#         pipe = pipeline(
#             "text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             max_new_tokens=max_new_tokens,
#             truncation=True,
#             do_sample=False,
#             return_full_text=False,
#             pad_token_id=tokenizer.eos_token_id,
#         )

#     return HuggingFacePipeline(pipeline=pipe)


# def format_docs_with_labels(docs: List[Document]) -> str:
#     parts = []
#     for i, doc in enumerate(docs, start=1):
#         meta = doc.metadata or {}
#         modality = meta.get("modality", "text")

#         if modality == "image":
#             filename = meta.get("filename", "unknown")
#             parts.append(
#                 f"[Doc {i}] Image file: {filename}\n"
#                 # f"[Doc {i}] Type: image | Source: {doc_name} | File: {filename}\n"
#                 # f"Path: {file_path}\n"
#                 f"Description: {doc.page_content}"
#             )
#         else:
#             parts.append(
#                 f"[Doc {i}] {doc.page_content}"
#                 # f"[Doc {i}] Type: text | Source: {doc_name} | Page: {page}\n"
#                 # f"Content: {doc.page_content}"
#             )

#     return "\n\n".join(parts)

# def format_sources(docs: List[Document]) -> List[Dict[str, Any]]:
#     rows = []
#     for i, doc in enumerate(docs, start=1):
#         meta = doc.metadata or {}
#         rows.append(
#             {
#                 "doc_id": f"Doc {i}",
#                 "doc_name": meta.get("doc_name", "unknown"),
#                 "page": meta.get("page_label", meta.get("page", "?")),
#                 "distance": meta.get("distance"),
#                 "preview": doc.page_content[:180],
#                 "modality": meta.get("modality", "text"), 
#             }
#         )
#     return rows


# def build_rag_chain(cfg: Optional[RAGConfig] = None):
#     cfg = cfg or RAGConfig()
#     retriever = ChromaRetriever(cfg)
#     llm = build_hf_llm(cfg.llm_model, max_new_tokens=cfg.max_new_tokens)
#     prompt = PromptTemplate.from_template(DEFAULT_PROMPT)

#     def prepare_inputs(x: Dict[str, Any]) -> Dict[str, Any]:
#         question = x["question"]
#         # docs = retriever.get_relevant_documents(question, k=cfg.top_k)
#         # NEW: retrieve text and image separately
#         q_lower = question.lower()
#         image_words = ["image", "fundus", "retina", "retinal", "optic disc", "photo", "artifact"]
#         use_images = any(word in q_lower for word in image_words)

#         text_docs = retriever.get_text_documents(
#             question,
#             k=cfg.text_k,
#             # pool_k=max(cfg.top_k * 5, 20),
#         )
#         image_docs = retriever.get_image_documents(question, k=cfg.image_k) if use_images else []

#         # NEW: combine both modalities
#         docs = text_docs + image_docs

#         print("DEBUG total docs:", len(docs))
#         print("DEBUG text docs:", len(text_docs))
#         print("DEBUG image docs:", len(image_docs))
#         # text_docs = []
#         # image_docs = []

#         # for doc in docs:
#         #     modality = (doc.metadata or {}).get("modality", "text")
#         #     if modality == "image":
#         #         image_docs.append(doc)
#         #     else:
#         #         text_docs.append(doc)


#         return {
#             "question": question,
#             "docs": docs,
#             "text_docs": text_docs,
#             "image_docs": image_docs,
#             "context": format_docs_with_labels(text_docs) if text_docs else "No text context retrieved.",
#             # "image_context": format_docs_with_labels(image_docs) if image_docs else "No image context retrieved.",
#         }

#     chain = (
#         RunnableLambda(prepare_inputs)
#         | RunnablePassthrough.assign(answer=prompt | llm | StrOutputParser())
#     )
#     return chain


# def ask_question(question: str, cfg: Optional[RAGConfig] = None) -> Dict[str, Any]:
#     chain = build_rag_chain(cfg)
#     out = chain.invoke({"question": question})
#     image_sources = []
#     for i, doc in enumerate(out["image_docs"], start=1):
#         meta = doc.metadata or {}
#         image_sources.append(
#             {
#                 "doc_id": f"Image {i}",
#                 "doc_name": meta.get("doc_name", "unknown"),
#                 "filename": meta.get("filename", "unknown"),
#                 "file_path": meta.get("file_path", "unknown"),
#                 "preview": doc.page_content[:180],
#             }
#         )

#     return {
#         "question": question,
#         "answer": out["answer"],
#         "sources": format_sources(out["docs"]),
#         "recommended_images": image_sources,
#     }


# def main():
#     parser = argparse.ArgumentParser(description="Session 4 LCEL RAG with Chroma + HF")
#     parser.add_argument("--question", required=True, help="Question to ask the RAG system")
#     parser.add_argument("--top_k", type=int, default=3)

#     # Seperate control for each modality
#     parser.add_argument("--text_k", type=int, default=3)
#     parser.add_argument("--image_k", type=int, default=2)

#     parser.add_argument("--llm_model", default="google/flan-t5-base")
#     args = parser.parse_args()

#     cfg = RAGConfig(
#         top_k=args.top_k, 
#         text_k=args.text_k,
#         image_k=args.image_k,
#         llm_model=args.llm_model)
#     result = ask_question(args.question, cfg)

#     print("\nQUESTION:")
#     print(result["question"])

#     print("\nANSWER:")
#     print(result["answer"])

#     print(f"\nTOP {len(result['sources'])} RETRIEVED SOURCES:")
#     for row in result["sources"]:
#         print(
#             f"- [{row['doc_id']}] {row['doc_name']} | page={row['page']} | "
#             f"distance={row['distance']:.4f} | {row['preview']}"
#         )

#     if result["recommended_images"]:
#         print("\nRECOMMENDED IMAGES:")
#         for img in result["recommended_images"]:
#             print(
#                 f"- [{img['doc_id']}] {img['filename']} | "
#                 f"path={img['file_path']} | {img['preview']}"
#             )    




# if __name__ == "__main__":
#     main()

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from vector_db_utils import VectorDB, ChromaConfig

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)


DEFAULT_PROMPT = """
Answer the question using only the context below. Follow this exact format: Direct answer: <one sentence> Explanation: <two or three short sentences> 
Rules: 
- Write in your own words. 
- Do not copy long phrases directly from the context. 
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
    max_new_tokens: int = 160


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

    def get_text_documents(self, question: str, k: int = 3, pool_k: int = 12) -> List[Document]:
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


def build_hf_llm(model_name: str, max_new_tokens: int = 160) -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_name_lower = model_name.lower()
    is_seq2seq = ("t5" in model_name_lower) or ("flan" in model_name_lower)

    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            truncation=True,
            do_sample=False,
        )
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            truncation=True,
            do_sample=False,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
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


def format_docs_for_generation(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        meta = doc.metadata or {}
        modality = meta.get("modality", "text")

        if modality == "image":
            filename = meta.get("filename", "unknown")
            parts.append(
                f"Image evidence from {filename}: fundus image centered on the optic disc region, used for glaucoma analysis."
            )
        else:
            parts.append(doc.page_content)

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
                "preview": doc.page_content[:180],
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
    return text


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
    parser = argparse.ArgumentParser(description="Session 4 LCEL RAG with two-step answer generation")
    parser.add_argument("--question", required=True, help="Question to ask the RAG system")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--text_k", type=int, default=3)
    parser.add_argument("--image_k", type=int, default=2)
    parser.add_argument("--llm_model", default="google/flan-t5-base")
    parser.add_argument("--max_new_tokens", type=int, default=160)
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
        print(
            f"- [{row['doc_id']}] {row['doc_name']} | modality={row['modality']} | "
            f"page={row['page']} | distance={row['distance']:.4f} | {row['preview']}"
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