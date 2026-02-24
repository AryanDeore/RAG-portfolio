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
    mock_settings.qdrant_api_key = "test_key"
    monkeypatch.setattr(qdrant_store, "settings", mock_settings)
    store = qdrant_store.QdrantStore()
    mock_client_class.assert_called_once_with(
        url="http://localhost:6333",
        api_key="test_key"
    )
    assert store.client == mock_client

def test_create_or_recreate_with_recreate_true(monkeypatch):
    """Verify create_or_recreate deletes then creates collection when recreate=True."""
    mock_client = Mock()
    mock_client.collection_exists = Mock(return_value=True)
    mock_client_class = Mock(return_value=mock_client)
    monkeypatch.setattr(qdrant_store, "QdrantClient", mock_client_class)
    mock_settings = Mock()
    mock_settings.qdrant_url = "http://localhost:6333"
    mock_settings.qdrant_api_key = "test_key"
    mock_settings.embed_dim = 384
    mock_settings.embed_metric = "Cosine"
    mock_settings.embed_collection = "test_collection"
    monkeypatch.setattr(qdrant_store, "settings", mock_settings)
    store = qdrant_store.QdrantStore()
    store.create_or_recreate(recreate=True)
    mock_client.delete_collection.assert_called_once_with(collection_name="test_collection")
    mock_client.create_collection.assert_called_once()
    call_args = mock_client.create_collection.call_args
    assert call_args.kwargs["collection_name"] == "test_collection"
    assert "vectors_config" in call_args.kwargs
    assert "sparse_vectors_config" in call_args.kwargs

def test_create_or_recreate_with_recreate_false_success(monkeypatch):
    """Verify create_or_recreate calls create_collection when recreate=False and collection doesn't exist."""
    mock_client = Mock()
    mock_client_class = Mock(return_value=mock_client)
    monkeypatch.setattr(qdrant_store, "QdrantClient", mock_client_class)
    mock_settings = Mock()
    mock_settings.qdrant_url = "http://localhost:6333"
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

def test_search_calls_query_points_dense_only(monkeypatch):
    """Verify search uses query_points with dense-only when no sparse_vec provided."""
    mock_client = Mock()
    mock_result = Mock()
    mock_result.points = []
    mock_client.query_points = Mock(return_value=mock_result)
    mock_client.collection_exists = Mock(return_value=True)
    mock_client_class = Mock(return_value=mock_client)
    monkeypatch.setattr(qdrant_store, "QdrantClient", mock_client_class)
    mock_settings = Mock()
    mock_settings.qdrant_url = "http://localhost:6333"
    mock_settings.qdrant_api_key = "test_key"
    mock_settings.embed_collection = "test_collection"
    mock_settings.embed_collection_read = ""
    monkeypatch.setattr(qdrant_store, "settings", mock_settings)
    store = qdrant_store.QdrantStore()
    query_vec = [0.1, 0.2, 0.3]
    result = store.search(query_vec, k=5)
    mock_client.query_points.assert_called_once()
    call_kwargs = mock_client.query_points.call_args.kwargs
    assert call_kwargs["using"] == "dense"
    assert call_kwargs["limit"] == 5

def test_search_calls_query_points_hybrid(monkeypatch):
    """Verify search uses query_points with prefetch+RRF when sparse_vec provided."""
    mock_client = Mock()
    mock_result = Mock()
    mock_result.points = []
    mock_client.query_points = Mock(return_value=mock_result)
    mock_client.collection_exists = Mock(return_value=True)
    mock_client_class = Mock(return_value=mock_client)
    monkeypatch.setattr(qdrant_store, "QdrantClient", mock_client_class)
    mock_settings = Mock()
    mock_settings.qdrant_url = "http://localhost:6333"
    mock_settings.qdrant_api_key = "test_key"
    mock_settings.embed_collection = "test_collection"
    mock_settings.embed_collection_read = ""
    monkeypatch.setattr(qdrant_store, "settings", mock_settings)
    store = qdrant_store.QdrantStore()
    query_vec = [0.1, 0.2, 0.3]
    sparse_vec = ([1, 5, 10], [0.5, 0.3, 0.1])
    result = store.search(query_vec, k=5, sparse_vec=sparse_vec)
    mock_client.query_points.assert_called_once()
    call_kwargs = mock_client.query_points.call_args.kwargs
    assert "prefetch" in call_kwargs
    assert len(call_kwargs["prefetch"]) == 2
    assert call_kwargs["limit"] == 5
