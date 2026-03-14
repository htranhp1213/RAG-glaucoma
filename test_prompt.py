from prompt import (
    RAGConfig,
    ChromaRetriever,
    make_text_query,
    make_image_query,
    format_docs_for_generation,
    clean_answer,
    attach_citations,
)


def test_make_text_query_removes_image_terms():
    q = "What is glaucoma and how is it identified in fundus images?"
    text_q = make_text_query(q)

    assert "fundus images" not in text_q
    assert "image" not in text_q
    assert "optic disc" not in text_q
    assert len(text_q) > 0


def test_make_image_query_adds_image_terms():
    q = "What is glaucoma?"
    image_q = make_image_query(q)

    assert "optic disc" in image_q.lower()
    assert "fundus" in image_q.lower()
    assert "glaucoma" in image_q.lower()


def test_chroma_retriever_returns_docs():
    cfg = RAGConfig()
    retriever = ChromaRetriever(cfg)
    docs = retriever.get_relevant_documents(
        "What is the primary goal of glaucoma treatment?", k=3
    )

    assert len(docs) == 3
    assert all(len(d.page_content) > 0 for d in docs)


def test_format_docs_for_generation_returns_nonempty_string():
    cfg = RAGConfig()
    retriever = ChromaRetriever(cfg)
    docs = retriever.get_relevant_documents("What is intraocular pressure?", k=2)

    context = format_docs_for_generation(docs)

    assert isinstance(context, str)
    assert context.strip() != ""


def test_clean_answer_maps_placeholder_to_idk():
    bad = "[Doc 1] or [Doc 2]."
    cleaned = clean_answer(bad)

    assert cleaned == "I don't know based on the provided context."


def test_attach_citations_adds_doc_labels_to_normal_answer():
    cfg = RAGConfig()
    retriever = ChromaRetriever(cfg)
    docs = retriever.get_relevant_documents(
        "What is the primary goal of glaucoma treatment?", k=2
    )

    answer = "The primary goal is to lower intraocular pressure."
    cited = attach_citations(answer, docs)

    assert "[Doc " in cited


def test_attach_citations_keeps_idk_unchanged():
    answer = "I don't know based on the provided context."
    cited = attach_citations(answer, [])

    assert cited == answer