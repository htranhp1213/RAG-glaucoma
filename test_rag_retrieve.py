from prompt import RAGConfig, ChromaRetriever


def test_get_text_documents_returns_only_text_modality():
    cfg = RAGConfig(text_k=3, image_k=2)
    retriever = ChromaRetriever(cfg)

    docs = retriever.get_text_documents(
        "What is glaucoma and how is it diagnosed?", k=3, pool_k=12
    )

    assert len(docs) > 0
    assert all((d.metadata or {}).get("modality") == "text" for d in docs)


def test_get_image_documents_returns_only_image_modality():
    cfg = RAGConfig(text_k=3, image_k=2)
    retriever = ChromaRetriever(cfg)

    docs = retriever.get_image_documents(
        "What retinal fundus images are available in the dataset?", k=2
    )

    assert len(docs) > 0
    assert all((d.metadata or {}).get("modality") == "image" for d in docs)


def test_text_and_image_retrieval_can_both_work_for_balanced_query():
    cfg = RAGConfig(text_k=3, image_k=2)
    retriever = ChromaRetriever(cfg)

    text_docs = retriever.get_text_documents(
        "What is glaucoma and how is it identified?", k=3, pool_k=12
    )
    image_docs = retriever.get_image_documents(
        "What is glaucoma and how is it identified in fundus images? optic disc fundus glaucoma image",
        k=2,
    )

    assert len(text_docs) > 0
    assert len(image_docs) > 0
    assert all((d.metadata or {}).get("modality") == "text" for d in text_docs)
    assert all((d.metadata or {}).get("modality") == "image" for d in image_docs)