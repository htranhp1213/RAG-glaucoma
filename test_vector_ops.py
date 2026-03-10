# test_vector_ops.py
import pytest
from vector_db_utils import VectorDB, ChromaConfig


@pytest.fixture()
def db(tmp_path):
    cfg = ChromaConfig(
        db_path=str(tmp_path / "chroma_test_db"),
        collection_name="test_collection",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    vdb = VectorDB(cfg)
    vdb.connect()
    vdb.create_or_get_collection(metadata={"purpose": "unit_test"})
    vdb.reset_collection()
    return vdb


def test_01_connect_create_count_empty(db):
    assert db.count() == 0


def test_02_insert_and_count(db):
    db.insert(
        ids=["a1", "a2"],
        documents=["Glaucoma is an optic neuropathy.", "Lowering IOP is key."],
        metadatas=[{"doc": "d1"}, {"doc": "d1"}],
    )
    assert db.count() == 2


def test_03_query_returns_k(db):
    db.upsert(
        ids=["b1", "b2", "b3"],
        documents=["IOP reduction via drops.", "Visual field monitoring.", "Optic nerve evaluation."],
        metadatas=[{"topic": "treat"}, {"topic": "monitor"}, {"topic": "exam"}],
    )
    res = db.query("How to lower eye pressure?", k=2)
    assert len(res["documents"][0]) == 2
    assert len(res["ids"][0]) == 2


def test_04_update_document(db):
    db.upsert(["c1"], ["Old text about IOP."], [{"doc": "x"}])
    db.update(["c1"], documents=["New text about intraocular pressure lowering."])
    got = db.get(ids=["c1"])
    assert got["documents"][0].startswith("New text")


def test_05_update_metadata(db):
    db.upsert(["d1"], ["Some chunk"], [{"doc": "old"}])
    db.update(["d1"], metadatas=[{"doc": "new", "section": "s1"}])
    got = db.get(ids=["d1"])
    assert got["metadatas"][0]["doc"] == "new"
    assert got["metadatas"][0]["section"] == "s1"


def test_06_upsert_overwrites(db):
    db.upsert(["e1"], ["v1"], [{"v": 1}])
    db.upsert(["e1"], ["v2"], [{"v": 2}])
    got = db.get(ids=["e1"])
    assert got["documents"][0] == "v2"
    assert got["metadatas"][0]["v"] == 2


def test_07_delete_by_id(db):
    db.upsert(["f1", "f2"], ["x", "y"], [{"k": 1}, {"k": 2}])
    db.delete(ids=["f1"])
    assert db.count() == 1
    got = db.get()
    assert got["ids"] == ["f2"]


def test_08_delete_by_where(db):
    db.upsert(
        ["g1", "g2", "g3"],
        ["a", "b", "c"],
        [{"doc": "keep"}, {"doc": "drop"}, {"doc": "drop"}],
    )
    db.delete(where={"doc": "drop"})
    got = db.get()
    assert db.count() == 1
    assert got["metadatas"][0]["doc"] == "keep"


def test_09_persistence_reload(tmp_path):
    path = str(tmp_path / "persist_db")

    cfg = ChromaConfig(db_path=path, collection_name="persist_col")
    db1 = VectorDB(cfg)
    db1.connect()
    db1.create_or_get_collection()
    db1.reset_collection()
    db1.upsert(["p1"], ["Persistent chunk"], [{"doc": "persist"}])
    assert db1.count() == 1

    db2 = VectorDB(cfg)
    db2.connect()
    db2.create_or_get_collection()
    assert db2.count() == 1
    got = db2.get(ids=["p1"])
    assert got["documents"][0] == "Persistent chunk"
