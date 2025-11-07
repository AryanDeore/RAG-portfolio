# tests/embedding/test_ingest.py
# Tests for portfolio chunk transformation to ingest format.

import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

# Mock dependencies before importing
sys.modules['fastembed'] = MagicMock()
sys.modules['qdrant_client'] = MagicMock()
sys.modules['qdrant_client.models'] = MagicMock()
sys.modules['configs'] = MagicMock()
sys.modules['configs.settings'] = MagicMock()

# Set up package structure for relative imports
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

# Create package structure
from types import ModuleType
embedding_pkg = ModuleType('src.shared.embedding')
sys.modules['src'] = ModuleType('src')
sys.modules['src.shared'] = ModuleType('src.shared')
sys.modules['src.shared.embedding'] = embedding_pkg

# Import utils first to satisfy relative import
utils_path = root / "src" / "shared" / "embedding" / "utils.py"
utils_spec = importlib.util.spec_from_file_location("src.shared.embedding.utils", utils_path)
utils_module = importlib.util.module_from_spec(utils_spec)
sys.modules['src.shared.embedding.utils'] = utils_module
utils_spec.loader.exec_module(utils_module)
embedding_pkg.utils = utils_module

# Mock fastembed_provider and qdrant_store before importing ingest
fastembed_provider_mock = MagicMock()
fastembed_provider_mock.FastEmbedProvider = MagicMock()
sys.modules['src.shared.embedding.fastembed_provider'] = fastembed_provider_mock
embedding_pkg.fastembed_provider = fastembed_provider_mock

qdrant_store_mock = MagicMock()
qdrant_store_mock.QdrantStore = MagicMock()
sys.modules['src.shared.embedding.qdrant_store'] = qdrant_store_mock
embedding_pkg.qdrant_store = qdrant_store_mock

# Now import ingest
ingest_path = root / "src" / "shared" / "embedding" / "ingest.py"
ingest_spec = importlib.util.spec_from_file_location("src.shared.embedding.ingest", ingest_path)
ingest_module = importlib.util.module_from_spec(ingest_spec)
sys.modules['src.shared.embedding.ingest'] = ingest_module
ingest_spec.loader.exec_module(ingest_module)
_portfolio_to_ingest = ingest_module._portfolio_to_ingest

def test_portfolio_to_ingest_basic_transformation():
    """Verify _portfolio_to_ingest transforms chunk dict to ingest format with required fields."""
    chunk = {
        "parent_id": "project_1",
        "index": "0",
        "field": "description",
        "text": "This is a test description."
    }
    result = _portfolio_to_ingest(chunk)
    assert result["doc_id"] == "project_1"
    assert result["chunk_id"] == 0
    assert result["text"] == "This is a test description."
    assert result["source"] == "portfolio"
    assert result["title"] == "project_1: description"

def test_portfolio_to_ingest_title_with_parent_title():
    """Verify _portfolio_to_ingest constructs title from parent_title and field when available."""
    chunk = {
        "parent_id": "project_1",
        "parent_title": "My Project",
        "index": "0",
        "field": "description",
        "text": "Test text"
    }
    result = _portfolio_to_ingest(chunk)
    assert result["title"] == "My Project: description"

def test_portfolio_to_ingest_title_without_parent_title():
    """Verify _portfolio_to_ingest uses parent_id when parent_title is missing."""
    chunk = {
        "parent_id": "project_1",
        "index": "0",
        "field": "description",
        "text": "Test text"
    }
    result = _portfolio_to_ingest(chunk)
    assert result["title"] == "project_1: description"

def test_portfolio_to_ingest_section_path_with_section():
    """Verify _portfolio_to_ingest constructs section_path including section when present."""
    chunk = {
        "parent_id": "project_1",
        "parent_type": "project",
        "index": "0",
        "field": "description",
        "section": "Topic 1",
        "text": "Test text"
    }
    result = _portfolio_to_ingest(chunk)
    assert result["section_path"] == "project/description/Topic 1"

def test_portfolio_to_ingest_section_path_without_section():
    """Verify _portfolio_to_ingest constructs section_path without section when missing."""
    chunk = {
        "parent_id": "project_1",
        "parent_type": "project",
        "index": "0",
        "field": "description",
        "text": "Test text"
    }
    result = _portfolio_to_ingest(chunk)
    assert result["section_path"] == "project/description"

def test_portfolio_to_ingest_extra_payload_includes_all_fields():
    """Verify _portfolio_to_ingest includes all expected fields in extra_payload."""
    chunk = {
        "parent_id": "project_1",
        "parent_type": "project",
        "parent_title": "My Project",
        "index": "0",
        "field": "description",
        "section": "Topic 1",
        "text": "Test text",
        "tags": ["tag1", "tag2"],
        "entities": ["Python", "FastAPI"],
        "company": "Acme Corp",
        "project": "My Project",
        "date_start": "2024-01",
        "date_end": "2024-12",
        "last_updated": "2024-10-14"
    }
    result = _portfolio_to_ingest(chunk)
    extra = result["extra_payload"]
    assert extra["parent_id"] == "project_1"
    assert extra["parent_type"] == "project"
    assert extra["parent_title"] == "My Project"
    assert extra["field"] == "description"
    assert extra["index"] == 0
    assert extra["section"] == "Topic 1"
    assert extra["entities"] == ["Python", "FastAPI"]
    assert extra["company"] == "Acme Corp"
    assert extra["project"] == "My Project"
    assert extra["date_start"] == "2024-01"
    assert extra["date_end"] == "2024-12"
    assert extra["last_updated"] == "2024-10-14"

def test_portfolio_to_ingest_handles_missing_optional_fields():
    """Verify _portfolio_to_ingest handles missing optional fields gracefully."""
    chunk = {
        "parent_id": "project_1",
        "index": "0",
        "field": "description",
        "text": "Test text"
    }
    result = _portfolio_to_ingest(chunk)
    assert result["tags"] == []
    extra = result["extra_payload"]
    assert extra["parent_type"] is None
    assert extra["parent_title"] is None
    assert extra["section"] is None
    assert extra["entities"] is None
    assert extra["company"] is None
    assert extra["project"] is None
    assert extra["date_start"] is None
    assert extra["date_end"] is None
    assert extra["last_updated"] is None

def test_portfolio_to_ingest_strips_text_whitespace():
    """Verify _portfolio_to_ingest strips leading/trailing whitespace from text field."""
    chunk = {
        "parent_id": "project_1",
        "index": "0",
        "field": "description",
        "text": "  Test text with spaces  "
    }
    result = _portfolio_to_ingest(chunk)
    assert result["text"] == "Test text with spaces"

def test_portfolio_to_ingest_handles_empty_text():
    """Verify _portfolio_to_ingest handles empty or None text gracefully."""
    chunk = {
        "parent_id": "project_1",
        "index": "0",
        "field": "description",
        "text": None
    }
    result = _portfolio_to_ingest(chunk)
    assert result["text"] == ""

def test_portfolio_to_ingest_section_path_filters_empty_bits():
    """Verify _portfolio_to_ingest filters out empty strings from section_path construction."""
    chunk = {
        "parent_id": "project_1",
        "parent_type": "",
        "index": "0",
        "field": "description",
        "text": "Test text"
    }
    result = _portfolio_to_ingest(chunk)
    assert result["section_path"] == "description"

