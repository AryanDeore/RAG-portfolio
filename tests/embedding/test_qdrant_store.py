# tests/embedding/test_qdrant_store.py
# Tests for Qdrant client wrapper managing collections, upserts, and search.

import sys
import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from qdrant_client.models import PointStruct, Filter, Distance, VectorParams

# Import qdrant_store directly
root = Path(__file__).parent.parent.parent
qdrant_store_path = root / "src" / "shared" / "embedding" / "qdrant_store.py"
spec = importlib.util.spec_from_file_location("qdrant_store", qdrant_store_path)
qdrant_store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qdrant_store)

def test_qdrant_store_init_uses_settings(monkeypatch):
    """Verify QdrantStore initializes QdrantClient with settings URL and API key."""
    mock_client = Mock()
    mock_client_class = Mock(return_value=mock_client)
    monkeypatch.setattr(qdrant_store, "QdrantClient", mock_client_class)
    mock_settings = Mock()
    mock_settings.qdrant_url = "http://localhost:6333"
    mock_settings.qdrant_url_resolved = "http://localhost:6333"
    mock_settings.qdrant_api_key = "test_key"
    monkeypatch.setattr(qdrant_store, "settings", mock_settings)
    store = qdrant_store.QdrantStore()
    mock_client_class.assert_called_once_with(
        url="http://localhost:6333",
        api_key="test_key"
    )
    assert store.client == mock_client

def test_create_or_recreate_with_recreate_true(monkeypatch):
    """Verify create_or_recreate calls recreate_collection when recreate=True."""
    mock_client = Mock()
    mock_client_class = Mock(return_value=mock_client)
    monkeypatch.setattr(qdrant_store, "QdrantClient", mock_client_class)
    mock_settings = Mock()
    mock_settings.qdrant_url = "http://localhost:6333"
    mock_settings.qdrant_url_resolved = "http://localhost:6333"
    mock_settings.qdrant_api_key = "test_key"
    mock_settings.embed_dim = 384
    mock_settings.embed_metric = "Cosine"
    mock_settings.embed_collection = "test_collection"
    monkeypatch.setattr(qdrant_store, "settings", mock_settings)
    store = qdrant_store.QdrantStore()
    store.create_or_recreate(recreate=True)
    mock_client.recreate_collection.assert_called_once()
    call_args = mock_client.recreate_collection.call_args
    assert call_args.kwargs["collection_name"] == "test_collection"
    # Verify vectors_config was passed (can't check isinstance with mocked VectorParams)
    assert "vectors_config" in call_args.kwargs
    assert call_args.kwargs["vectors_config"] is not None

def test_create_or_recreate_with_recreate_false_success(monkeypatch):
    """Verify create_or_recreate calls create_collection when recreate=False and collection doesn't exist."""
    mock_client = Mock()
    mock_client_class = Mock(return_value=mock_client)
    monkeypatch.setattr(qdrant_store, "QdrantClient", mock_client_class)
    mock_settings = Mock()
    mock_settings.qdrant_url = "http://localhost:6333"
    mock_settings.qdrant_url_resolved = "http://localhost:6333"
    mock_settings.qdrant_api_key = "test_key"
    mock_settings.embed_dim = 384
    mock_settings.embed_metric = "Cosine"
    mock_settings.embed_collection = "test_collection"
    monkeypatch.setattr(qdrant_store, "settings", mock_settings)
    store = qdrant_store.QdrantStore()
    store.create_or_recreate(recreate=False)
    mock_client.create_collection.assert_called_once()
    call_args = mock_client.create_collection.call_args
    assert call_args.kwargs["collection_name"] == "test_collection"

def test_create_or_recreate_with_recreate_false_handles_existing(monkeypatch):
    """Verify create_or_recreate silently handles exception when collection already exists."""
    mock_client = Mock()
    mock_client.create_collection = Mock(side_effect=Exception("Collection already exists"))
    mock_client_class = Mock(return_value=mock_client)
    monkeypatch.setattr(qdrant_store, "QdrantClient", mock_client_class)
    mock_settings = Mock()
    mock_settings.qdrant_url = "http://localhost:6333"
    mock_settings.qdrant_url_resolved = "http://localhost:6333"
    mock_settings.qdrant_api_key = "test_key"
    mock_settings.embed_dim = 384
    mock_settings.embed_metric = "Cosine"
    mock_settings.embed_collection = "test_collection"
    monkeypatch.setattr(qdrant_store, "settings", mock_settings)
    store = qdrant_store.QdrantStore()
    # Should not raise exception
    store.create_or_recreate(recreate=False)
    mock_client.create_collection.assert_called_once()

def test_upsert_points_calls_client_upsert(monkeypatch):
    """Verify upsert_points calls client.upsert with correct collection name and points."""
    mock_client = Mock()
    mock_client_class = Mock(return_value=mock_client)
    monkeypatch.setattr(qdrant_store, "QdrantClient", mock_client_class)
    mock_settings = Mock()
    mock_settings.qdrant_url = "http://localhost:6333"
    mock_settings.qdrant_url_resolved = "http://localhost:6333"
    mock_settings.qdrant_api_key = "test_key"
    mock_settings.embed_collection = "test_collection"
    monkeypatch.setattr(qdrant_store, "settings", mock_settings)
    store = qdrant_store.QdrantStore()
    points = [
        PointStruct(id=1, vector=[0.1, 0.2], payload={"text": "test"}),
        PointStruct(id=2, vector=[0.3, 0.4], payload={"text": "test2"})
    ]
    store.upsert_points(points)
    mock_client.upsert.assert_called_once_with(
        collection_name="test_collection",
        points=points
    )

def test_search_calls_client_search(monkeypatch):
    """Verify search calls client.search with query vector, limit, and optional filter."""
    mock_client = Mock()
    mock_client.search = Mock(return_value=[])
    mock_client_class = Mock(return_value=mock_client)
    monkeypatch.setattr(qdrant_store, "QdrantClient", mock_client_class)
    mock_settings = Mock()
    mock_settings.qdrant_url = "http://localhost:6333"
    mock_settings.qdrant_url_resolved = "http://localhost:6333"
    mock_settings.qdrant_api_key = "test_key"
    mock_settings.embed_collection = "test_collection"
    monkeypatch.setattr(qdrant_store, "settings", mock_settings)
    store = qdrant_store.QdrantStore()
    query_vec = [0.1, 0.2, 0.3]
    filter_obj = Filter()
    result = store.search(query_vec, k=5, filt=filter_obj)
    mock_client.search.assert_called_once_with(
        collection_name="test_collection",
        query_vector=query_vec,
        limit=5,
        query_filter=filter_obj
    )

def test_search_without_filter(monkeypatch):
    """Verify search works correctly when no filter is provided."""
    mock_client = Mock()
    mock_client.search = Mock(return_value=[])
    mock_client_class = Mock(return_value=mock_client)
    monkeypatch.setattr(qdrant_store, "QdrantClient", mock_client_class)
    mock_settings = Mock()
    mock_settings.qdrant_url = "http://localhost:6333"
    mock_settings.qdrant_url_resolved = "http://localhost:6333"
    mock_settings.qdrant_api_key = "test_key"
    mock_settings.embed_collection = "test_collection"
    monkeypatch.setattr(qdrant_store, "settings", mock_settings)
    store = qdrant_store.QdrantStore()
    query_vec = [0.1, 0.2, 0.3]
    result = store.search(query_vec, k=10, filt=None)
    mock_client.search.assert_called_once_with(
        collection_name="test_collection",
        query_vector=query_vec,
        limit=10,
        query_filter=None
    )

