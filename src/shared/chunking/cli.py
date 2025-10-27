"""This file provides a small CLI to run the chunker on contents.json and write chunks.json."""

import sys
from pathlib import Path
from .io import read_contents, write_chunks
from .builder import build_children

def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint to read contents, build chunks, and write output.
    
    Args:
        argv (list[str] | None): Optional args [src, dst].
        
    Returns:
        None: Prints a short status line.
    """
    argv = argv or sys.argv[1:]
    src = Path(argv[0]) if argv else Path("contents.json")
    dst = Path(argv[1]) if len(argv) > 1 else Path("chunks.json")

    data = read_contents(src)
    rows = build_children(data)
    write_chunks(rows, dst)

    print(f"[OK] wrote {len(rows)} chunks -> {dst.resolve()}")

    # Optional: quick embedding sanity-check (only if fastembed is installed)
    try:
        from fastembed.embedding import TextEmbedding  # type: ignore
        if rows:
            embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            vec = next(iter(embedder.embed([rows[0]['text']])))
            print(f"[OK] embedded 1 sample | dim={len(vec)} | first5={list(vec)[:5]}")
    except Exception:
        # If fastembed is missing or any error occurs, we skip silently.
        pass

if __name__ == "__main__":
    main()
