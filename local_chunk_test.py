
# local_chunk_test.py
# Run local sanity checks on your chunker without touching Qdrant.
import json, hashlib, sys, argparse, os
from pathlib import Path

sys.path.append(str(Path('.').resolve()))  # allow 'src' imports from project root
from src.shared.chunking_simple import build_children

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contents", "-c", default="contents.json")
    ap.add_argument("--write", action="store_true", help="Write chunks.json next to contents")
    args = ap.parse_args()

    p = Path(args.contents)
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = build_children(data)
    print(f"[ok] built {len(rows)} chunks")

    # show a preview
    for r in rows[:3]:
        print(f"- id={r['id']} | parent={r['parent_title']} | field={r['field']} | entities={r.get('entities')[:4]}")
        print(f"  text: {r['text'][:120]}...\n")

    # determinism check: hash of first 10 IDs
    ids = [r["id"] for r in rows[:10]]
    digest = hashlib.sha1("\n".join(ids).encode()).hexdigest()[:12]
    print(f"[check] first10 id hash={digest}")

    if args.write:
        out = p.with_name("chunks.json")
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] wrote chunks -> {out.resolve()}")

if __name__ == "__main__":
    main()
