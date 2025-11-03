"""
Small utilities for hashing, time-stamps, and batching to support embedding and upserts.
"""

import hashlib
import time
import uuid
from typing import Iterable, Any

def sha1(s: str) -> str:
    """
    Compute a hex SHA1 digest for deterministic IDs and change detection.

    Args:
        s (str): Input string to hash.

    Returns:
        str: 40-character lowercase hexadecimal SHA1 digest.
    """
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def now_s() -> int:
    """
    Get the current UNIX timestamp in whole seconds.

    Returns:
        int: Seconds since the UNIX epoch.
    """
    return int(time.time())

def batched(iterable: Iterable[Any], n: int) -> Iterable[list[Any]]:
    """
    Yield successive fixed-size lists from an input iterable.

    Args:
        iterable (Iterable[Any]): Source items to be grouped.
        n (int): Maximum size of each batch; must be >= 1.

    Yields:
        list[Any]: Contiguous batch of up to n items.
    """
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def make_point_id(name: str, scheme: str = "uuid"):
    """
    Deterministically map a stable name like "doc_id::chunk_id" to a Qdrant point ID.

    scheme:
      - "uuid" (default): UUIDv5 derived from the name (string id, Qdrant-valid)
      - "int":  60-bit unsigned int from SHA-1 prefix (Qdrant-valid integer id)
    """
    if scheme == "uuid":
        return str(uuid.uuid5(uuid.NAMESPACE_URL, name))
    if scheme == "int":
        # Take first 15 hex chars of SHA-1 -> fits under 2^60 (unsigned int)
        return int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:15], 16)
    raise ValueError(f"Unknown id scheme: {scheme!r} (use 'uuid' or 'int')")