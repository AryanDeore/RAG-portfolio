"""This file handles tiny JSON file I/O helpers for reading contents and writing chunks."""

import json
from pathlib import Path
from typing import Dict, Any, List

def read_contents(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file from disk into a dict.
    
    Args:
        path (str | Path): Path to the JSON file.
        
    Returns:
        Dict[str, Any]: Dict parsed from JSON text.
    """
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))

def write_chunks(rows: List[Dict[str, Any]], path: str | Path) -> None:
    """Write a list of chunk dicts to a file as pretty-printed JSON.
    
    Args:
        rows (List[Dict[str, Any]]): List of chunk dictionaries to write.
        path (str | Path): Path where the JSON file will be written.
    """
    p = Path(path)
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
