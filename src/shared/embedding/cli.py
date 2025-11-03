"""
Tiny CLI to create collections, upsert from chunk.json, and run ad-hoc searches.
"""

import json
import argparse
from typing import List, Dict
from .ingest import upsert_from_iter
from .retrieval import search_chunks
from .qdrant_store import QdrantStore
from .logging import make_experiment
from configs.settings import settings

REQUIRED_KEYS = ("parent_id", "field", "index", "text")

def _validate_chunks(obj) -> List[Dict]:
    if not isinstance(obj, list):
        # Heuristic: user passed contents.json (nested dict) by mistake
        hint = ""
        if isinstance(obj, dict) and any(k in obj for k in ("metadata","bio","projects","experience","skills","education")):
            hint = (
                "\nIt looks like a nested contents.json. "
                "Run your chunker first to produce a flat chunk.json list, "
                "then call: embedding-cli upsert --chunks chunk.json"
            )
        raise ValueError(f"Expected a JSON array of chunk dicts, got {type(obj).__name__}.{hint}")

    # Schema check for a few samples (first 5)
    for i, d in enumerate(obj[:5]):
        if not isinstance(d, dict):
            raise ValueError(f"Chunk at index {i} is not an object (got {type(d).__name__})")
        missing = [k for k in REQUIRED_KEYS if k not in d]
        if missing:
            raise ValueError(f"Chunk at index {i} missing keys: {missing}. "
                             f"Required keys: {REQUIRED_KEYS}")
    return obj

def main() -> None:
    ap = argparse.ArgumentParser("embedding-cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c1 = sub.add_parser("create", help="Create or recreate the Qdrant collection.")
    c1.add_argument("--recreate", action="store_true", help="Drop and recreate the collection.")

    c2 = sub.add_parser("upsert", help="Embed and upsert chunks from a JSON file (list of dicts).")
    c2.add_argument("--chunks", default="chunk.json", help="Path to chunk.json (flat portfolio chunks)")

    c3 = sub.add_parser("search", help="Search top-k chunks for a natural language query.")
    c3.add_argument("--q", required=True, help="The query text")
    c3.add_argument("--k", type=int, default=None, help="Number of neighbors to return")

    args = ap.parse_args()

    if args.cmd == "create":
        QdrantStore().create_or_recreate(recreate=args.recreate)
        print(f"✅ Collection ready: {settings.embed_collection}")

    elif args.cmd == "upsert":
        with open(args.chunks, "r", encoding="utf-8") as f:
            obj = json.load(f)
        try:
            docs = _validate_chunks(obj)
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
            raise SystemExit(1)

        exp = make_experiment(settings.comet_project_name)
        exp.log_parameters({
            "model": settings.embed_model,
            "dim": settings.embed_dim,
            "metric": settings.embed_metric,
            "collection": settings.embed_collection,
            "batch_size": settings.embed_batch_size
        })

        total, failed = upsert_from_iter(docs)
        exp.log_metrics({"chunks_upserted": total, "chunks_failed": failed})
        print(f"✅ Upserted {total} | ❌ Failed {failed}")

        if failed:
            print("ℹ️ Tip: ensure collection vector size == 384 (bge-small-en-v1.5) and no empty texts.")

    elif args.cmd == "search":
        hits = search_chunks(args.q, k=args.k)
        for h in hits:
            pt = h  # dict from retrieval
            title = pt.get("title") or f"{pt.get('doc_id')}#{pt.get('chunk_id')}"
            print(f"[{pt['score']:.4f}] {title}")
            snippet = (pt['text'] or '').replace("\n"," ")
            print(snippet[:200] + ("..." if len(snippet) > 200 else ""))

if __name__ == "__main__":
    main()
