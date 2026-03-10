# ingest_to_chroma.py
import json
from typing import Any, Dict, List, Tuple

from vector_db_utils import VectorDB, ChromaConfig

CHUNK_JSONL_PATH = "chunks.jsonl"
TEXT_KEYS = ["text", "chunk", "content"]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_text(obj: Dict[str, Any]) -> str:
    for k in TEXT_KEYS:
        if k in obj and isinstance(obj[k], str):
            return obj[k]
    raise KeyError(f"No text field found. Expected one of {TEXT_KEYS}. Keys found: {list(obj.keys())}")


def _to_chroma_value(v: Any):
    """
    Chroma metadata value must be: str, int, float, bool, list, or None.
    - dict -> JSON string
    - list -> ensure list elements are primitives; otherwise JSON-string the whole list
    - other -> string
    """
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        # if list contains only primitives, keep it; else stringify
        ok = all(x is None or isinstance(x, (str, int, float, bool)) for x in v)
        return v if ok else json.dumps(v, ensure_ascii=False)
    return str(v)


def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    - If there's a nested 'metadata' dict, flatten it one level into meta_flat.
    - Convert any non-allowed values to safe types.
    """
    meta_flat: Dict[str, Any] = {}

    # If chunk has a nested "metadata" field that's a dict, flatten it
    nested = meta.get("metadata")
    if isinstance(nested, dict):
        for k, v in nested.items():
            meta_flat[k] = v

    # Copy the rest of top-level keys except large text fields
    for k, v in meta.items():
        if k in TEXT_KEYS:
            continue
        if k == "metadata" and isinstance(v, dict):
            continue
        meta_flat[k] = v

    # Sanitize values
    return {k: _to_chroma_value(v) for k, v in meta_flat.items()}


def build_records(rows: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for i, obj in enumerate(rows):
        rid = obj.get("id") or f"chunk_{i}"
        ids.append(str(rid))

        docs.append(extract_text(obj))

        metas.append(sanitize_metadata(obj))

    return ids, docs, metas


def main():
    rows = load_jsonl(CHUNK_JSONL_PATH)
    if not rows:
        raise RuntimeError(f"No rows found in {CHUNK_JSONL_PATH}")

    ids, docs, metas = build_records(rows)

    cfg = ChromaConfig(
        db_path="./vector_db",
        collection_name="competition_guidelines",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    db = VectorDB(cfg)
    db.connect()
    db.create_or_get_collection(metadata={"source": "session2_chunk.jsonl"})

    # upsert (rerunnable)
    db.upsert(ids=ids, documents=docs, metadatas=metas)

    print(f"[OK] Ingested {len(ids)} records")
    print("[OK] Total records now:", db.count())


if __name__ == "__main__":
    main()
