# tests/embedding/test_utils.py
# Tests for embedding utilities: hashing, timestamps, batching, and point ID generation.

import importlib.util
from pathlib import Path
# Import utils module directly from file to avoid triggering __init__.py imports
utils_path = Path(__file__).parent.parent.parent / "src" / "shared" / "embedding" / "utils.py"
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

def test_sha1_deterministic():
    """Verify sha1 produces consistent, deterministic hashes for the same input."""
    result1 = utils.sha1("test string")
    result2 = utils.sha1("test string")
    assert result1 == result2
    assert len(result1) == 40  # SHA1 hex digest is 40 chars
    assert isinstance(result1, str)
    assert result1.islower()  # Should be lowercase hex

def test_sha1_different_inputs():
    """Verify sha1 produces different hashes for different inputs."""
    hash1 = utils.sha1("hello")
    hash2 = utils.sha1("world")
    assert hash1 != hash2

def test_now_s_returns_integer():
    """Verify now_s returns an integer timestamp representing seconds since epoch."""
    timestamp = utils.now_s()
    assert isinstance(timestamp, int)
    assert timestamp > 0  # Should be a positive timestamp

def test_batched_exact_multiple():
    """Verify batched correctly groups items when count is an exact multiple of batch size."""
    items = [1, 2, 3, 4, 5, 6]
    batches = list(utils.batched(items, 3))
    assert batches == [[1, 2, 3], [4, 5, 6]]

def test_batched_partial_batch():
    """Verify batched includes remaining items in a final partial batch."""
    items = [1, 2, 3, 4, 5]
    batches = list(utils.batched(items, 3))
    assert batches == [[1, 2, 3], [4, 5]]

def test_batched_single_item():
    """Verify batched handles single-item batches correctly."""
    items = [1, 2, 3]
    batches = list(utils.batched(items, 1))
    assert batches == [[1], [2], [3]]

def test_batched_larger_than_collection():
    """Verify batched handles batch size larger than collection size."""
    items = [1, 2]
    batches = list(utils.batched(items, 5))
    assert batches == [[1, 2]]

def test_batched_empty_iterable():
    """Verify batched handles empty iterables gracefully."""
    batches = list(utils.batched([], 3))
    assert batches == []

def test_make_point_id_uuid_scheme():
    """Verify make_point_id generates deterministic UUIDs for the same input using uuid scheme."""
    name = "doc1::chunk1"
    id1 = utils.make_point_id(name, scheme="uuid")
    id2 = utils.make_point_id(name, scheme="uuid")
    assert id1 == id2
    assert isinstance(id1, str)
    # UUIDv5 format check (has dashes)
    assert len(id1) == 36
    assert id1.count("-") == 4

def test_make_point_id_uuid_deterministic():
    """Verify make_point_id produces same UUID for same name across calls."""
    name = "test::123"
    id1 = utils.make_point_id(name, scheme="uuid")
    id2 = utils.make_point_id(name, scheme="uuid")
    assert id1 == id2

def test_make_point_id_int_scheme():
    """Verify make_point_id generates deterministic integers for the same input using int scheme."""
    name = "doc1::chunk1"
    id1 = utils.make_point_id(name, scheme="int")
    id2 = utils.make_point_id(name, scheme="int")
    assert id1 == id2
    assert isinstance(id1, int)
    assert id1 > 0  # Should be positive

def test_make_point_id_int_deterministic():
    """Verify make_point_id produces same integer for same name across calls."""
    name = "test::456"
    id1 = utils.make_point_id(name, scheme="int")
    id2 = utils.make_point_id(name, scheme="int")
    assert id1 == id2

def test_make_point_id_different_schemes():
    """Verify make_point_id produces different IDs for same name with different schemes."""
    name = "doc1::chunk1"
    uuid_id = utils.make_point_id(name, scheme="uuid")
    int_id = utils.make_point_id(name, scheme="int")
    assert uuid_id != int_id
    assert isinstance(uuid_id, str)
    assert isinstance(int_id, int)

def test_make_point_id_invalid_scheme():
    """Verify make_point_id raises ValueError for unknown scheme."""
    try:
        utils.make_point_id("test", scheme="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown id scheme" in str(e)

