# local_chunk_test.py
# Run local sanity checks on YOUR contents.json (no Qdrant needed).
import os, sys, json, hashlib
from pathlib import Path
from typing import Optional

# allow "from src.shared.chunking_simple import build_children"
sys.path.append(str(Path(".").resolve()))
from src.shared.chunking_simple import build_children  # noqa

CONTENTS_PATH = Path(os.getenv("CONTENTS_JSON", "contents.json"))

def warn(msg: str):
    print(f"[warn] {msg}")

def sha_first_ids(rows, k=10) -> str:
    ids = [r["id"] for r in rows[:k]]
    return hashlib.sha1("\n".join(ids).encode()).hexdigest()[:12]

def preview(rows, n=5):
    print("\n[preview]")
    for r in rows[:n]:
        sec = f" | section={r['section']}" if r.get("section") else ""
        ents = ", ".join((r.get("entities") or [])[:4])
        print(f"- id={r['id']} | parent={r['parent_title']} | field={r['field']}{sec}")
        print(f"  entities=[{ents}]")
        print(f"  text: {r['text'][:160]}...\n")

def validate_shape(contents: dict):
    """Light schema sanity checks; prints warnings only."""
    if "experience" in contents and isinstance(contents["experience"], list):
        for i, exp in enumerate(contents["experience"]):
            if not isinstance(exp, dict):
                warn(f"experience[{i}] should be an object")
                continue
            # your custom fields allowed to be strings
            for fld in ["company_description", "projects_worked_on", "achievements"]:
                if fld in exp and not isinstance(exp[fld], (str, list)):
                    warn(f"experience[{i}].{fld} should be str or list")
            if "date_range" in exp and isinstance(exp["date_range"], dict):
                for k in ("start", "end"):
                    if exp["date_range"].get(k) and not isinstance(exp["date_range"][k], str):
                        warn(f"experience[{i}].date_range.{k} should be 'YYYY-MM' string")

def main(write_chunks: bool = True, embed_smoke: bool = True):
    if not CONTENTS_PATH.exists():
        raise SystemExit(f"Missing {CONTENTS_PATH.resolve()} — set CONTENTS_JSON env or place contents.json here")

    contents = json.loads(CONTENTS_PATH.read_text(encoding="utf-8"))
    validate_shape(contents)

    rows = build_children(contents)
    if not rows:
        raise SystemExit("No chunks built. Check your contents.json format.")

    print(f"[ok] built {len(rows)} chunks from {CONTENTS_PATH.name}")
    preview(rows, n=5)

    digest = sha_first_ids(rows, k=10)
    print(f"[check] determinism hash (first10 ids) = {digest}")

    if write_chunks:
        out = CONTENTS_PATH.with_name("chunks.json")
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] wrote chunks -> {out.resolve()}")

    if embed_smoke:
        try:
            from fastembed.embedding import TextEmbedding  # type: ignore
            embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            sample_texts = [r["text"] for r in rows[:3]]
            vecs = list(embedder.embed(sample_texts))
            dims = len(vecs[0]) if vecs else 0
            print(f"[emb] embedded {len(sample_texts)} samples | dim={dims} | v0_first5={list(vecs[0])[:5] if vecs else []}")
        except Exception as e:
            print(f"[note] embed smoke skipped ({e}). Install fastembed if you want this check.")

if __name__ == "__main__":
    # Toggle by editing args here if you like
    main(write_chunks=True, embed_smoke=True)
