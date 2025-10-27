"""This file is a small local test harness to load contents.json and print a short summary of produced chunks."""

from pathlib import Path
import json
from src.shared.chunking.builder import build_children

def _load(path: str = "contents.json"):
    """Load a JSON file to a dict for quick local tests.
    
    Args:
        path (str): Path to the JSON file.
        
    Returns:
        Dict: Dict parsed from JSON.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))

if __name__ == "__main__":
    rows = build_children(_load())
    print(f"chunks: {len(rows)}")
    print("sample:", rows[0] if rows else "(no rows)")
