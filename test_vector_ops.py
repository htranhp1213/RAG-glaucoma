from vector_db_utils import VectorDB, ChromaConfig


def test_vector_db_connect_and_collection():
    cfg = ChromaConfig(
        db_path="./vector_db",
        collection_name="competition_guidelines",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    db = VectorDB(cfg)
    db.connect()
    col = db.create_or_get_collection()

    assert col is not None
    assert db.get_collection() is not None


def test_vector_db_count_is_positive():
    cfg = ChromaConfig(
        db_path="./vector_db",
        collection_name="competition_guidelines",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    db = VectorDB(cfg)
    db.connect()
    db.create_or_get_collection()

    assert db.count() > 0


def test_vector_db_query_returns_expected_shape():
    cfg = ChromaConfig(
        db_path="./vector_db",
        collection_name="competition_guidelines",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    db = VectorDB(cfg)
    db.connect()
    db.create_or_get_collection()

    res = db.query(
        "What is glaucoma?",
        k=3,
        include=["documents", "metadatas", "distances"],
    )

    assert "documents" in res
    assert "metadatas" in res
    assert "distances" in res
    assert len(res["documents"][0]) > 0
    assert len(res["metadatas"][0]) > 0
    assert len(res["distances"][0]) > 0