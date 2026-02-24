# tests/embedding/test_fastembed_provider.py
# Tests for FastEmbed provider wrapper with instruction prefixes.

import sys
import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Mock fastembed before importing
mock_fastembed = MagicMock()
mock_fastembed.TextEmbedding = MagicMock()
mock_fastembed.SparseTextEmbedding = MagicMock()
sys.modules['fastembed'] = mock_fastembed

# Import fastembed_provider directly
root = Path(__file__).parent.parent.parent
fastembed_provider_path = root / "src" / "shared" / "embedding" / "fastembed_provider.py"
spec = importlib.util.spec_from_file_location("fastembed_provider", fastembed_provider_path)
fastembed_provider = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fastembed_provider)

import pytest

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton state before each test."""
    fastembed_provider.FastEmbedProvider._instance = None
    fastembed_provider.FastEmbedProvider._initialized = False
    yield

def test_fastembed_provider_init_uses_settings(monkeypatch):
    """Verify FastEmbedProvider initializes TextEmbedding with settings from config."""
    mock_text_embedding = Mock()
    mock_text_embedding_class = Mock(return_value=mock_text_embedding)
    mock_sparse_embedding = Mock()
    mock_sparse_embedding_class = Mock(return_value=mock_sparse_embedding)
    monkeypatch.setattr(fastembed_provider, "TextEmbedding", mock_text_embedding_class)
    monkeypatch.setattr(fastembed_provider, "SparseTextEmbedding", mock_sparse_embedding_class)
    mock_settings = Mock()
    mock_settings.embed_model = "test-model"
    mock_settings.embed_cache_dir = "/tmp/cache"
    monkeypatch.setattr(fastembed_provider, "settings", mock_settings)
    provider = fastembed_provider.FastEmbedProvider()
    mock_text_embedding_class.assert_called_once_with(
        model_name="test-model",
        cache_dir="/tmp/cache"
    )
    assert provider.model == mock_text_embedding

def test_embed_passages_adds_prefix(monkeypatch):
    """Verify embed_passages adds 'passage: ' prefix to each input text before embedding."""
    mock_text_embedding = Mock()
    mock_text_embedding_class = Mock(return_value=mock_text_embedding)
    mock_sparse_embedding_class = Mock(return_value=Mock())
    monkeypatch.setattr(fastembed_provider, "TextEmbedding", mock_text_embedding_class)
    monkeypatch.setattr(fastembed_provider, "SparseTextEmbedding", mock_sparse_embedding_class)
    mock_settings = Mock()
    mock_settings.embed_model = "test-model"
    mock_settings.embed_cache_dir = "/tmp/cache"
    monkeypatch.setattr(fastembed_provider, "settings", mock_settings)
    mock_text_embedding.embed = Mock(return_value=iter([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    provider = fastembed_provider.FastEmbedProvider()
    result = provider.embed_passages(["text1", "text2"])
    assert len(result) == 2
    mock_text_embedding.embed.assert_called_once_with(["passage: text1", "passage: text2"])

def test_embed_passages_returns_list_of_vectors(monkeypatch):
    """Verify embed_passages returns a list of dense vectors matching input count."""
    mock_text_embedding = Mock()
    mock_text_embedding_class = Mock(return_value=mock_text_embedding)
    mock_sparse_embedding_class = Mock(return_value=Mock())
    monkeypatch.setattr(fastembed_provider, "TextEmbedding", mock_text_embedding_class)
    monkeypatch.setattr(fastembed_provider, "SparseTextEmbedding", mock_sparse_embedding_class)
    mock_settings = Mock()
    mock_settings.embed_model = "test-model"
    mock_settings.embed_cache_dir = "/tmp/cache"
    monkeypatch.setattr(fastembed_provider, "settings", mock_settings)
    mock_text_embedding.embed = Mock(return_value=iter([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
    provider = fastembed_provider.FastEmbedProvider()
    result = provider.embed_passages(["a", "b", "c"])
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == [0.1, 0.2]

def test_embed_query_adds_prefix(monkeypatch):
    """Verify embed_query adds 'query: ' prefix to the input query before embedding."""
    mock_text_embedding = Mock()
    mock_text_embedding_class = Mock(return_value=mock_text_embedding)
    mock_sparse_embedding_class = Mock(return_value=Mock())
    monkeypatch.setattr(fastembed_provider, "TextEmbedding", mock_text_embedding_class)
    monkeypatch.setattr(fastembed_provider, "SparseTextEmbedding", mock_sparse_embedding_class)
    mock_settings = Mock()
    mock_settings.embed_model = "test-model"
    mock_settings.embed_dim = 384
    mock_settings.embed_cache_dir = "/tmp/cache"
    monkeypatch.setattr(fastembed_provider, "settings", mock_settings)
    mock_text_embedding.embed = Mock(return_value=iter([[0.1, 0.2, 0.3]]))
    provider = fastembed_provider.FastEmbedProvider()
    # embed_query now takes optional parent_span
    mock_span = Mock()
    mock_span.span = Mock(return_value=mock_span)
    mock_span.end = Mock()
    result = provider.embed_query("what is python?", parent_span=mock_span)
    mock_text_embedding.embed.assert_called_once_with(["query: what is python?"])

def test_embed_query_returns_single_vector(monkeypatch):
    """Verify embed_query returns a single vector (first element) from embedding result."""
    mock_text_embedding = Mock()
    mock_text_embedding_class = Mock(return_value=mock_text_embedding)
    mock_sparse_embedding_class = Mock(return_value=Mock())
    monkeypatch.setattr(fastembed_provider, "TextEmbedding", mock_text_embedding_class)
    monkeypatch.setattr(fastembed_provider, "SparseTextEmbedding", mock_sparse_embedding_class)
    mock_settings = Mock()
    mock_settings.embed_model = "test-model"
    mock_settings.embed_dim = 384
    mock_settings.embed_cache_dir = "/tmp/cache"
    monkeypatch.setattr(fastembed_provider, "settings", mock_settings)
    mock_text_embedding.embed = Mock(return_value=iter([[0.1, 0.2, 0.3]]))
    provider = fastembed_provider.FastEmbedProvider()
    mock_span = Mock()
    mock_span.span = Mock(return_value=mock_span)
    mock_span.end = Mock()
    result = provider.embed_query("test query", parent_span=mock_span)
    assert isinstance(result, list)
    assert result == [0.1, 0.2, 0.3]
