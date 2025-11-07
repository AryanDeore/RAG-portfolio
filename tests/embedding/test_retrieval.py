# tests/embedding/test_retrieval.py
# Tests for KNN retrieval pipeline that embeds queries and searches Qdrant.

import sys
import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Mock dependencies before importing
sys.modules['fastembed'] = MagicMock()
sys.modules['fastembed'].TextEmbedding = MagicMock()

# Set up package structure for relative imports
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))
from types import ModuleType
sys.modules['src'] = ModuleType('src')
sys.modules['src.shared'] = ModuleType('src.shared')
sys.modules['src.shared.embedding'] = ModuleType('src.shared.embedding')
embedding_pkg = sys.modules['src.shared.embedding']

# Import fastembed_provider
fastembed_provider_path = root / "src" / "shared" / "embedding" / "fastembed_provider.py"
fastembed_spec = importlib.util.spec_from_file_location("src.shared.embedding.fastembed_provider", fastembed_provider_path)
fastembed_module = importlib.util.module_from_spec(fastembed_spec)
sys.modules['src.shared.embedding.fastembed_provider'] = fastembed_module
embedding_pkg.fastembed_provider = fastembed_module
fastembed_spec.loader.exec_module(fastembed_module)

# Import qdrant_store
qdrant_store_path = root / "src" / "shared" / "embedding" / "qdrant_store.py"
qdrant_spec = importlib.util.spec_from_file_location("src.shared.embedding.qdrant_store", qdrant_store_path)
qdrant_module = importlib.util.module_from_spec(qdrant_spec)
sys.modules['src.shared.embedding.qdrant_store'] = qdrant_module
embedding_pkg.qdrant_store = qdrant_module
qdrant_spec.loader.exec_module(qdrant_module)

# Now import retrieval
retrieval_path = root / "src" / "shared" / "embedding" / "retrieval.py"
retrieval_spec = importlib.util.spec_from_file_location("src.shared.embedding.retrieval", retrieval_path)
retrieval = importlib.util.module_from_spec(retrieval_spec)
sys.modules['src.shared.embedding.retrieval'] = retrieval
retrieval_spec.loader.exec_module(retrieval)

def test_search_chunks_uses_default_k_from_settings():
    """Verify search_chunks uses default k from settings when None is provided."""
    mock_provider = Mock()
    mock_provider.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
    mock_store = Mock()
    mock_hit = MagicMock()
    mock_hit.id = "test_id"
    mock_hit.score = 0.95
    mock_hit.payload = {"doc_id": "doc1", "chunk_id": 0, "title": "Test", "text": "Test text"}
    mock_store.search = Mock(return_value=[mock_hit])
    with patch("src.shared.embedding.retrieval.FastEmbedProvider", return_value=mock_provider):
        with patch("src.shared.embedding.retrieval.QdrantStore", return_value=mock_store):
            with patch("src.shared.embedding.retrieval.settings") as mock_settings:
                mock_settings.retrieval_k = 5
                result = retrieval.search_chunks("test query", k=None)
                mock_store.search.assert_called_once_with([0.1, 0.2, 0.3], k=5)

def test_search_chunks_uses_provided_k():
    """Verify search_chunks uses provided k value instead of default."""
    mock_provider = Mock()
    mock_provider.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
    mock_store = Mock()
    mock_store.search = Mock(return_value=[])
    with patch("src.shared.embedding.retrieval.FastEmbedProvider", return_value=mock_provider):
        with patch("src.shared.embedding.retrieval.QdrantStore", return_value=mock_store):
            result = retrieval.search_chunks("test query", k=10)
            mock_store.search.assert_called_once_with([0.1, 0.2, 0.3], k=10)

def test_search_chunks_result_structure():
    """Verify search_chunks returns dicts with id, score, doc_id, chunk_id, title, and text fields."""
    mock_provider = Mock()
    mock_provider.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
    mock_store = Mock()
    mock_hit1 = MagicMock()
    mock_hit1.id = "id1"
    mock_hit1.score = 0.9
    mock_hit1.payload = {"doc_id": "doc1", "chunk_id": 0, "title": "Title 1", "text": "Text 1"}
    mock_hit2 = MagicMock()
    mock_hit2.id = "id2"
    mock_hit2.score = 0.8
    mock_hit2.payload = {"doc_id": "doc2", "chunk_id": 1, "title": "Title 2", "text": "Text 2"}
    mock_store.search = Mock(return_value=[mock_hit1, mock_hit2])
    with patch("src.shared.embedding.retrieval.FastEmbedProvider", return_value=mock_provider):
        with patch("src.shared.embedding.retrieval.QdrantStore", return_value=mock_store):
            result = retrieval.search_chunks("test query", k=2)
            assert len(result) == 2
            assert result[0]["id"] == "id1"
            assert result[0]["score"] == 0.9
            assert result[0]["doc_id"] == "doc1"
            assert result[0]["chunk_id"] == 0
            assert result[0]["title"] == "Title 1"
            assert result[0]["text"] == "Text 1"

def test_search_chunks_calls_embed_query():
    """Verify search_chunks calls embed_query with the provided query string."""
    mock_provider = Mock()
    mock_provider.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
    mock_store = Mock()
    mock_store.search = Mock(return_value=[])
    with patch("src.shared.embedding.retrieval.FastEmbedProvider", return_value=mock_provider):
        with patch("src.shared.embedding.retrieval.QdrantStore", return_value=mock_store):
            retrieval.search_chunks("what is python?", k=5)
            mock_provider.embed_query.assert_called_once_with("what is python?")

def test_search_chunks_handles_missing_payload_fields():
    """Verify search_chunks handles missing optional payload fields gracefully using .get()."""
    mock_provider = Mock()
    mock_provider.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
    mock_store = Mock()
    mock_hit = MagicMock()
    mock_hit.id = "id1"
    mock_hit.score = 0.9
    mock_hit.payload = {"doc_id": "doc1"}  # Missing chunk_id, title, text
    mock_store.search = Mock(return_value=[mock_hit])
    with patch("src.shared.embedding.retrieval.FastEmbedProvider", return_value=mock_provider):
        with patch("src.shared.embedding.retrieval.QdrantStore", return_value=mock_store):
            result = retrieval.search_chunks("test", k=1)
            assert result[0]["doc_id"] == "doc1"
            assert result[0]["chunk_id"] is None
            assert result[0]["title"] is None
            assert result[0]["text"] is None

