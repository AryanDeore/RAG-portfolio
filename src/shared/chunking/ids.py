"""
This file provides deterministic ID helpers for parents and children, enabling stable chunk identifiers across runs.
"""

import hashlib
import uuid

def parent_id(kind: str, name: str) -> str:
    """
    Build a deterministic UUIDv5 parent ID from a simple (kind, name) pair.
    
    Args:
        kind (str): Logical category like "bio", "project", "experience", "skills"
        name (str): A human-readable key such as project title or company name
        
    Returns:
        str: A stable UUIDv5 string suitable for use as a parent_id
    """
    base = f"{kind}::{(name or '').lower().strip()}".strip()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))

def child_id(p_id: str, field: str, idx: int, text: str) -> str:
    """
    Build a deterministic child ID that changes only when content changes.
    
    Args:
        p_id (str): Parent ID
        field (str): Field name
        idx (int): Position index
        text (str): Chunk text
        
    Returns:
        str: An ID composed of parent, field, index, and a short SHA1 of the text for change detection
    """
    sha = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{p_id}:{field}:{idx}:{sha}"
