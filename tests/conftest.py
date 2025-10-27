# tests/conftest.py
import sys
from pathlib import Path
import pytest

# Ensure project root on sys.path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture
def sample_contents():
    return {"bio": {"summary": "Hi"}}
