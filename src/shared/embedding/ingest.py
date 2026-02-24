from typing import Iterable, Dict, List, Tuple
import logging
from qdrant_client.models import PointStruct, SparseVector
from .utils import sha1, now_s, batched, make_point_id
from .fastembed_provider import FastEmbedProvider
from .qdrant_store import QdrantStore
from configs.settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _portfolio_to_ingest(d: dict) -> dict:
    parent_id = d["parent_id"]
    idx = int(d["index"])
    field = d["field"]

    title = f"{d.get('parent_title', parent_id)}: {field}".strip(": ")
    section_bits = [d.get("parent_type", ""), field]
    if d.get("section"):
        section_bits.append(d["section"])
    section_path = "/".join([str(x) for x in section_bits if x])

    return {
        "doc_id": parent_id,
        "chunk_id": idx,
        "text": (d["text"] or "").strip(),
        "title": title,
        "section_path": section_path,
        "source": "portfolio",
        "tags": d.get("tags", []),
        "extra_payload": {
            "parent_id": parent_id,
            "parent_type": d.get("parent_type"),
            "parent_title": d.get("parent_title"),
            "field": field,
            "index": idx,
            "section": d.get("section"),
            "entities": d.get("entities"),
            "company": d.get("company"),
            "project": d.get("project"),
            "date_start": d.get("date_start"),
            "date_end": d.get("date_end"),
            "last_updated": d.get("last_updated"),
            "links": d.get("links"),
        }
    }

def docs_to_points(provider: FastEmbedProvider, docs: List[dict]) -> List[PointStruct]:
    normed_all = [_portfolio_to_ingest(d) for d in docs]
    normed = [n for n in normed_all if n["text"]]
    dropped = len(normed_all) - len(normed)
    if dropped:
        logger.warning("Skipped %d chunks with empty text", dropped)

    if not normed:
        return []

    texts = [n["text"] for n in normed]
    dense_vectors = provider.embed_passages(texts)
    sparse_vectors = provider.embed_passages_sparse(texts)

    pts: List[PointStruct] = []
    for n, dvec, svec in zip(normed, dense_vectors, sparse_vectors):
        if hasattr(dvec, "tolist"):
            dvec = dvec.tolist()
        dvec = [float(x) for x in dvec]
        name = f"{n['doc_id']}::{n['chunk_id']}"
        pid = make_point_id(name, scheme=settings.embed_id_scheme)

        payload = {
            "doc_id": n["doc_id"],
            "chunk_id": n["chunk_id"],
            "text": n["text"],
            "title": n["title"],
            "section_path": n["section_path"],
            "source": n["source"],
            "tags": n["tags"],
            "ingested_at": now_s(),
            "doc_hash": sha1(str(n["doc_id"])),
            "chunk_hash": sha1(n["text"]),
            **(n["extra_payload"] or {}),
        }
        pts.append(PointStruct(
            id=pid,
            vector={
                "dense": dvec,
                "sparse": SparseVector(indices=svec[0], values=svec[1]),
            },
            payload=payload,
        ))
    return pts

def upsert_from_iter(iterable_docs: Iterable[Dict]) -> Tuple[int, int]:
    store = QdrantStore()
    provider = FastEmbedProvider()
    store.create_or_recreate(recreate=True)
    if hasattr(store, "ensure_collection_dims"):
        store.ensure_collection_dims(expected=384)

    total, failed = 0, 0
    for i, batch in enumerate(batched(iterable_docs, settings.embed_batch_size), start=1):
        try:
            pts = docs_to_points(provider, batch)
            if not pts:
                logger.info("Batch %d had no valid points (all empty text?)", i)
                continue
            store.upsert_points(pts)
            total += len(pts)
            logger.info("Upserted batch %d: +%d (cumulative %d)", i, len(pts), total)
        except Exception as e:
            failed += len(batch)
            logger.error("Failed batch %d with %d docs: %s", i, len(batch), e, exc_info=True)
    return total, failed
