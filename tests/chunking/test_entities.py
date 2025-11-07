# tests/test_entities.py
# Tests for entity extraction and tag generation using in-memory catalogs.

import importlib
from src.shared.chunking import entities as ent

def test_extract_entities_with_monkeypatched_catalog(monkeypatch):
    """Verify extract_entities_from_text identifies tech entities using in-memory catalogs and aliases."""
    # Patch catalogs so we don't rely on filesystem JSON
    monkeypatch.setattr(ent, "_TECH_ALIAS", {
        "aws": "AWS",
        "postgresql": "PostgreSQL",
        "next.js": "Next.js",
    }, raising=True)

    monkeypatch.setattr(ent, "_TECH_CATALOG", {
        "aws": ["aws", "amazon web services"],
        "postgresql": ["postgres", "postgresql"],
        "next.js": ["next.js", "nextjs"],
        "rag": ["retrieval-augmented"],
    }, raising=True)

    text = "We deployed on AWS with Postgres. Also built a NextJS front-end and a retrieval-augmented backend."
    found = ent.extract_entities_from_text(text)
    # Note: "postgres" in text maps to Postgres catalog trigger -> PostgreSQL alias
    assert {"AWS", "PostgreSQL", "Next.js", "RAG"} - set([f.upper() for f in []])  # ensure names below
    assert "AWS" in found
    assert "PostgreSQL" in found
    assert "Next.js" in found
    # 'rag' canon normalizes via _normalize_token to "Rag" (title-cased) unless you alias it.
    assert any(x.lower() == "rag" for x in found)

def test_build_project_entities_and_tags(monkeypatch):
    """Verify build_project_entities_and_tags extracts entities from project data and generates auto-tags based on tech stack and features."""
    monkeypatch.setattr(ent, "_TECH_ALIAS", {
        "aws": "AWS", "fastapi": "FastAPI", "next.js": "Next.js"
    }, raising=True)
    monkeypatch.setattr(ent, "_TECH_CATALOG", {
        "aws": ["aws"], "fastapi": ["fastapi"], "next.js": ["nextjs"]
    }, raising=True)

    proj = {
        "title": "X",
        "tech_stack": ["aws", "fastapi"],
        "tags": ["demo"],
        "description": "Using NextJS UI on AWS.",
        "features": ["- Real-time updates"],
        "outcomes": {"impact": "Improved latency"},
    }
    entities, tags = ent.build_project_entities_and_tags(proj)
    assert "AWS" in entities and "FastAPI" in entities and "Next.js" in entities
    # Auto-tags: backend because FastAPI; frontend because Next.js; cloud because AWS; real-time because feature text
    assert set(tags) >= {"demo", "backend", "frontend", "cloud", "real-time"}
