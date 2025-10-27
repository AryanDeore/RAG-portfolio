# tests/test_builder.py
# Integration-ish tests for build_children() on a minimal contents dict.

from src.shared.chunking.builder import build_children

def test_build_children_minimal_contents():
    contents = {
        "metadata": {"last_updated": "2025-10-14"},
        "bio": {"summary": "Hi there."},
        "projects": [
            {
                "title": "RAG Portfolio",
                "tagline": "Search your work, fast.",
                "description": "**Topic 1:** Sentence A. Sentence B.",
                "features": ["- Feature one", "- Feature two"]
            }
        ],
        "experience": [
            {
                "company": "Acme",
                "position": "Engineer",
                "date_range": {"start": "2024-01", "end": "2024-12"},
                "company_description": "**About:** We build things.",
                "achievements": "- Shipped stuff",
                "tech_stack": ["Python", "SQL"],
                "tags": ["backend"]
            }
        ],
        "skills": {"languages": ["Python"], "frameworks": ["FastAPI"], "tools_and_platforms": ["Docker"]}
    }

    rows = build_children(contents)
    assert len(rows) > 0

    # Bio row exists
    assert any(r["parent_type"] == "bio" and r["field"] == "summary" for r in rows)

    # Project pieces with section captured
    proj_desc = [r for r in rows if r["parent_type"] == "project" and r["field"] == "description"]
    assert proj_desc, "Project description chunks should exist"
    assert any("section" in r and r["section"] == "Topic 1" for r in proj_desc)

    # Feature bullets become rows
    features = [r for r in rows if r["parent_type"] == "project" and r["field"] == "feature"]
    assert len(features) == 2

    # Experience rows carry dates
    exp_rows = [r for r in rows if r["parent_type"] == "experience"]
    assert exp_rows and all(r["date_start"] == "2024-01" for r in exp_rows if "date_start" in r)

    # Skills catalog present
    assert any(r["parent_type"] == "skills" and r["field"] == "catalog" for r in rows)
