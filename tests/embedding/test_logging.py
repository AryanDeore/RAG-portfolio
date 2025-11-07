# tests/embedding/test_logging.py
# Tests for logging facade that integrates with Comet ML when available.

import os
import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch
# Import logging module directly from file to avoid triggering __init__.py imports
logging_path = Path(__file__).parent.parent.parent / "src" / "shared" / "embedding" / "logging.py"
spec = importlib.util.spec_from_file_location("logging", logging_path)
logging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logging)

def test_noop_log_parameters():
    """Verify Noop.log_parameters accepts any arguments and does nothing."""
    noop = logging.Noop()
    noop.log_parameters({"key": "value"})
    noop.log_parameters(key1="val1", key2="val2")
    # Should not raise any exceptions

def test_noop_log_metrics():
    """Verify Noop.log_metrics accepts any arguments and does nothing."""
    noop = logging.Noop()
    noop.log_metrics({"metric": 1.0})
    noop.log_metrics(accuracy=0.95)
    # Should not raise any exceptions

def test_noop_log_other():
    """Verify Noop.log_other accepts any arguments and does nothing."""
    noop = logging.Noop()
    noop.log_other("some", "data")
    noop.log_other(key="value")
    # Should not raise any exceptions

def test_make_experiment_no_api_key(monkeypatch):
    """Verify make_experiment returns Noop when COMET_API_KEY is not set."""
    monkeypatch.delenv("COMET_API_KEY", raising=False)
    monkeypatch.setattr("os.getenv", lambda key, default=None: None if key == "COMET_API_KEY" else default)
    exp = logging.make_experiment("test_project")
    assert isinstance(exp, logging.Noop)

def test_make_experiment_empty_api_key(monkeypatch):
    """Verify make_experiment returns Noop when COMET_API_KEY is empty string."""
    monkeypatch.setattr("os.getenv", lambda key, default=None: "" if key == "COMET_API_KEY" else default)
    exp = logging.make_experiment("test_project")
    assert isinstance(exp, logging.Noop)

def test_make_experiment_with_api_key_mocked(monkeypatch):
    """Verify make_experiment returns Experiment when COMET_API_KEY is set and comet_ml is available."""
    monkeypatch.setattr("os.getenv", lambda key, default=None: "test_key_123" if key == "COMET_API_KEY" else default)
    mock_experiment = Mock()
    monkeypatch.setattr(logging, "Experiment", mock_experiment)
    exp = logging.make_experiment("test_project")
    assert exp is not None
    mock_experiment.assert_called_once_with(project_name="test_project", auto_output_logging="simple")

def test_make_experiment_comet_unavailable(monkeypatch):
    """Verify make_experiment returns Noop when comet_ml module is not available."""
    monkeypatch.setattr("os.getenv", lambda key, default=None: "test_key" if key == "COMET_API_KEY" else default)
    monkeypatch.setattr(logging, "Experiment", None)
    exp = logging.make_experiment("test_project")
    assert isinstance(exp, logging.Noop)

