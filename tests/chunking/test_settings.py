# tests/test_settings.py
# Smoke tests for Settings behavior (no real secrets dir, env toggles).

import os
import importlib
from configs import settings as settings_module

def test_settings_loads_without_secrets_dir(monkeypatch):
    # Ensure no warning-producing secrets_dir path is set/exists
    monkeypatch.delenv("APP_SECRETS_DIR", raising=False)
    importlib.reload(settings_module)
    s = settings_module.get_settings()
    # Verify settings object loads correctly by checking an existing field
    assert s.debug is not None
    assert isinstance(s.chunk_max_chars_paragraph, int)

def test_env_override_chunk_limits(monkeypatch):
    monkeypatch.setenv("CHUNK_MAX_CHARS_PARAGRAPH", "120")
    # Force reload to pick up env change
    importlib.reload(settings_module)
    s = settings_module.get_settings()
    assert s.chunk_max_chars_paragraph == 120
