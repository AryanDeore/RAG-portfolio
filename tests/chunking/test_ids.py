# tests/test_ids.py
# Tests for deterministic parent/child identifiers.

from src.shared.chunking.ids import parent_id, child_id

def test_parent_id_stability():
    """Verify parent_id produces consistent, case-insensitive identifiers for the same parent type and title."""
    a = parent_id("project", "RAG Portfolio")
    b = parent_id("project", "RAG Portfolio")
    c = parent_id("project", "rag portfolio")  # case-insensitive
    assert a == b == c

def test_child_id_changes_with_text():
    """Verify child_id produces different identifiers when text content changes, ensuring content-based uniqueness."""
    pid = parent_id("project", "Demo")
    c1 = child_id(pid, "description", 0, "hello")
    c2 = child_id(pid, "description", 0, "hello!!")
    assert c1 != c2
