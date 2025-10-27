# tests/test_utils.py
# Tests for text utilities: bullet detection, heading parse, packing, normalization.

import importlib
from src.shared.chunking import utils

def test_detect_md_heading():
    # label + colon outside bold
    lab, body = utils.detect_md_heading("**Topic 1:** This is the body.")
    assert lab == "Topic 1"
    assert body == "This is the body."

    # label + dash
    lab2, body2 = utils.detect_md_heading("**Findings** - First, do X.")
    assert lab2 == "Findings"
    assert body2 == "First, do X."

    # no label
    lab3, body3 = utils.detect_md_heading("Plain paragraph.")
    assert lab3 is None
    assert body3 == "Plain paragraph."

def test_looks_and_strip_bullet():
    s = "- hello world"
    assert utils.looks_bullet(s) is True
    assert utils.strip_bullet(s) == "hello world"

def test_pack_sentences_limit_arg():
    # Use explicit max_chars arg so we don't depend on settings
    text = "A short. Another short. Third."
    chunks = utils.pack_sentences(text, max_chars=12)
    # Expect 2+ chunks given tight limit
    assert len(chunks) >= 2
    assert "A short." in " ".join(chunks)

def test_bullet_children_default_no_split():
    # With default settings, one bullet = one child
    children = utils.bullet_children("- A long bullet that should remain a single child by default.")
    assert children == ["A long bullet that should remain a single child by default."]

def test_norm_list_variants():
    assert utils.norm_list(None) == []
    assert utils.norm_list([" A ", " ", "B"]) == ["A", "B"]
    assert utils.norm_list("[A, B, C]") == ["A", "B", "C"]
