# test_rag_retrieval.py
import pytest
from vector_db_utils import VectorDB, ChromaConfig


@pytest.fixture(scope="session")
def prod_db():
    cfg = ChromaConfig(
        db_path="./vector_db",
        collection_name="competition_guidelines",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    db = VectorDB(cfg)
    db.connect()
    db.create_or_get_collection()
    assert db.count() > 0, "DB is empty. Run: python ingest_to_chroma.py"
    return db


DOMAIN_QUESTIONS = [
    "What is the primary goal of glaucoma treatment?",
    "How is intraocular pressure (IOP) managed in glaucoma?",
    "What tests are used to monitor glaucoma progression over time?",
    "When would visual field testing be recommended?",
    "What is the role of optic nerve evaluation in glaucoma care?",
    "How often should patients be followed up after starting glaucoma therapy?",
    "What are common risk factors for glaucoma progression?",
    "What is ocular hypertension and how does it relate to glaucoma risk?",
    "When is laser treatment considered in glaucoma management?",
    "How is angle-closure glaucoma managed differently from open-angle glaucoma?",
    "What findings suggest worsening glaucoma?",
    "How should clinicians document glaucoma staging or severity?",
    "What is the purpose of gonioscopy in glaucoma evaluation?",
    "What should be considered when a patient is not responding to therapy?",
    "What are typical indications for surgical intervention in glaucoma?",
]


@pytest.mark.parametrize("q", DOMAIN_QUESTIONS)
def test_qa_returns_topk(prod_db, q):
    res = prod_db.query(q, k=5)
    assert len(res["documents"][0]) == 5
    assert all(isinstance(t, str) and len(t) > 0 for t in res["documents"][0])


@pytest.mark.parametrize("q", DOMAIN_QUESTIONS[:10])
def test_metadata_present(prod_db, q):
    res = prod_db.query(q, k=3)
    metas = res["metadatas"][0]
    assert len(metas) == 3
    assert all(isinstance(m, dict) for m in metas)
