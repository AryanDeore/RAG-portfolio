# src/shared/chunking_simple.py
# -----------------------------------------------------------------------------
# Example of the final output (written as an ARRAY to chunks.json):
#
# [
#   {
#     "id": "a4d9...:description:0:8c1a5e2b91b3",
#     "parent_id": "a4d9f110-5a8c-51d9-8c2e-2c3e7b9a6f12",
#     "parent_type": "project",
#     "parent_title": "RAG Portfolio",
#     "field": "description",
#     "index": 0,
#     "text": "Lorem ipsum sentence-packed…",
#     "section": "Topic 1",                  # <- detected from **Topic 1:** prefix
#     "entities": ["AWS Lambda","FastAPI","Next.js","PostgreSQL","cloud","full-stack","microservices","real-time"],
#     "tags": ["full-stack","microservices","real-time","cloud"],
#     "company": null,
#     "project": "RAG Portfolio",
#     "date_start": null,
#     "date_end": null,
#     "last_updated": "2025-10-14"
#   }
# ]
# -----------------------------------------------------------------------------

import json
import re
import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# -------------------------- knobs (simple defaults) ---------------------------
MAX_CHARS_PARAGRAPH = 700     # paragraphs are sentence-packed near this size
SPLIT_LONG_BULLETS = False    # False => one bullet = one child (recommended)
MAX_CHARS_BULLET = 700        # used only when SPLIT_LONG_BULLETS=True

# Optional local embeddings (CPU-friendly). Install with: `uv add fastembed`
try:
    from fastembed.embedding import TextEmbedding
    FASTEMBED_AVAILABLE = True
except Exception:
    FASTEMBED_AVAILABLE = False


# ------------------------------ tiny helpers ---------------------------------
def parent_id(kind: str, name: str) -> str:
    """
    Args:
        kind: Section type ("bio" | "project" | "experience" | "skills").
        name: Human-readable key to derive the parent id (e.g., project title).

    Input:
        Strings.

    Output:
        Deterministic UUIDv5 string that stays the same for the same (kind, name).
    """
    base = f"{kind}::{name}".lower().strip()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


def child_id(p_id: str, field: str, idx: int, text: str) -> str:
    """
    Args:
        p_id: Parent id from parent_id().
        field: Child field label (e.g., "feature", "description").
        idx: Child position within that field.
        text: The chunk text.

    Input:
        Strings + int.

    Output:
        Stable child id = parent + field + index + short SHA1(text).
        Lets you skip re-embedding unchanged chunks.
    """
    sha = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{p_id}:{field}:{idx}:{sha}"


def looks_bullet(s: str) -> bool:
    """
    Args:
        s: One line of text.

    Output:
        True if s starts with a bullet prefix like '-', '*', or '•'.
    """
    return bool(re.match(r"^\s*[-*•]\s+", s))


def strip_bullet(s: str) -> str:
    """
    Args:
        s: A possibly bulleted string.

    Output:
        The same string without the bullet prefix.
    """
    return re.sub(r"^\s*[-*•]\s+", "", s).strip()


def para_split(text: Optional[str]) -> List[str]:
    """
    Args:
        text: Raw text that may contain blank lines.

    Input:
        String or None.

    Output:
        List of non-empty paragraphs split on blank lines.
    """
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]


def sent_split(text: str) -> List[str]:
    """
    Args:
        text: A paragraph.

    Input:
        String.

    Output:
        List of sentences using a simple punctuation-based split.
    """
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def pack_sentences(paragraph: str, max_chars: int = MAX_CHARS_PARAGRAPH) -> List[str]:
    """
    Args:
        paragraph: One paragraph string.
        max_chars: Rough size cap per chunk (character-based, not tokens).

    Input:
        String.

    Output:
        List of chunk strings, each ~<= max_chars, preserving sentence boundaries.
    """
    out: List[str] = []
    cur: List[str] = []
    n = 0
    for s in sent_split(paragraph):
        if n + len(s) + 1 <= max_chars or not cur:
            cur.append(s)
            n += len(s) + 1
        else:
            out.append(" ".join(cur))
            cur = [s]
            n = len(s)
    if cur:
        out.append(" ".join(cur))
    return out


def norm_list(x: Any) -> List[str]:
    """
    Args:
        x: Could be a real list or a string like "[A, B]".

    Input:
        Any.

    Output:
        Clean Python list of strings.
    """
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str):
        s = x.strip().strip("[]")
        return [p.strip().strip("'\"") for p in s.split(",") if p.strip()]
    return [str(x).strip()]


def detect_md_heading(paragraph: str):
    """
    Detect headings like:
      - "**Title:** text"
      - "**Title**: text"
      - "**Title: ** text"   (colon inside the bold)
      - "**Title** - text"
    Returns: (section_label or None, body_text)
    """
    s = (paragraph or "").strip()
    if not s:
        return None, ""

    # Case A: **Title** : text   OR   **Title** - text
    m = re.match(r"^\s*\*\*(.+?)\*\*\s*[:\-–]\s*(.*)$", s)
    if m:
        label, rest = m.group(1).strip(), m.group(2).strip()
        return label, (rest if rest else "")

    # Case B: **Title: ** text  (colon inside bold)
    m2 = re.match(r"^\s*\*\*\s*(.+?)\s*[:：]\s*\*\*\s*(.*)$", s)
    if m2:
        label, rest = m2.group(1).strip(), m2.group(2).strip()
        return label, (rest if rest else "")

    return None, s


def bullet_children(text: str) -> List[str]:
    """
    Args:
        text: One bullet text (may contain multiple sentences).

    Input:
        String (possibly starting with a bullet prefix).

    Output:
        List of one or more child strings.
        - If SPLIT_LONG_BULLETS=False (default): one bullet = one child.
        - If True: sentence-pack into multiple children near MAX_CHARS_BULLET.
    """
    clean = strip_bullet(text) if looks_bullet(text) else str(text).strip()
    if not clean:
        return []
    if not SPLIT_LONG_BULLETS:
        return [clean]
    pieces: List[str] = []
    paras = para_split(clean) or [clean]
    for para in paras:
        pieces.extend(pack_sentences(para, max_chars=MAX_CHARS_BULLET))
    return pieces

def normalize_token(tok: str) -> str:
    """
    Canonicalize technology strings (simple, fast).
    Args: tok (str)
    Output: canonical display string (e.g., 'postgres' -> 'PostgreSQL')
    """
    t = (tok or "").strip()
    if not t:
        return t
    low = t.lower()
    alias = {
        "aws": "AWS",
        "amazon web services": "AWS",
        "gcp": "GCP",
        "google cloud platform": "GCP",
        "azure": "Azure",
        "postgres": "PostgreSQL",
        "postgresql": "PostgreSQL",
        "nextjs": "Next.js",
        "next.js": "Next.js",
        "fastapi": "FastAPI",
        "lambda": "AWS Lambda",
        "aws lambda": "AWS Lambda",
        "docker": "Docker",
        "kubernetes": "Kubernetes",
        "dbt": "dbt",
        "sql": "SQL",
        "python": "Python",
    }
    if low in alias:
        return alias[low]
    if "." in t:
        parts = t.split(".")
        return ".".join(p[:1].upper() + p[1:] for p in parts)
    return t[:1].upper() + t[1:]


def extract_entities_from_text(text: str) -> List[str]:
    """
    Lightweight entity harvesting from free text using a small catalog.
    Args: text (str)
    Output: list of canonical entity strings (deduped)
    """
    if not text:
        return []
    s = text.lower()
    catalog = {
        "aws": ["aws", "amazon web services"],
        "gcp": ["gcp", "google cloud platform"],
        "azure": ["azure"],
        "aws lambda": ["aws lambda", " lambda "],
        "postgresql": ["postgres", "postgresql"],
        "fastapi": ["fastapi"],
        "next.js": ["next.js", "nextjs"],
        "docker": ["docker"],
        "kubernetes": ["kubernetes", "k8s"],
        "python": ["python"],
        "sql": [" sql "],  # add spaces to avoid matching words like 'sequel'
        "rag": ["retrieval-augmented", "retrieval augmented", " rag "],
        "vector db": ["qdrant", "pinecone", "weaviate", "vector db", "vectordb"],
    }
    found: List[str] = []
    for canon, triggers in catalog.items():
        for trig in triggers:
            if trig in s:
                found.append(normalize_token(canon))
                break
    return sorted(set(found))


def build_project_entities_and_tags(proj: Dict[str, Any]) -> (List[str], List[str]):
    """
    Build richer entities and tags for a project.
    Args: proj (dict) – one project object from contents.json
    Output: (entities: List[str], tags: List[str])
    """
    declared_tech = norm_list(proj.get("tech_stack"))
    declared_tags = norm_list(proj.get("tags"))

    # normalize declared tech
    norm_tech = [normalize_token(t) for t in declared_tech if t]

    # harvest entities from long text fields (description/architecture/challenges/features/outcomes)
    text_fields = [
        proj.get("description") or "",
        proj.get("problem") or "",
        proj.get("architecture") or "",
        proj.get("challenges") or "",
        " ".join(proj.get("features") or []) if isinstance(proj.get("features"), list) else (proj.get("features") or ""),
        json.dumps(proj.get("outcomes") or {}, ensure_ascii=False),
    ]
    harvested = extract_entities_from_text(" \n ".join(text_fields))

    entities = sorted(set(norm_tech + harvested))

    # simple auto-tags from entities/content
    auto_tags: List[str] = []
    if any(e in entities for e in ["AWS", "AWS Lambda", "GCP", "Azure"]):
        auto_tags.append("cloud")
    if any(e in entities for e in ["Docker", "Kubernetes"]):
        auto_tags.append("devops")
    if "Next.js" in entities:
        auto_tags.append("frontend")
    if "FastAPI" in entities:
        auto_tags.append("backend")
    long_text = (" ".join(text_fields)).lower()
    if "real-time" in long_text or "realtime" in long_text:
        auto_tags.append("real-time")

    tags = sorted(set(declared_tags + auto_tags))
    return entities, tags



# ------------------------------- main builder --------------------------------
def build_children(contents: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Args:
        contents: Loaded dict from your contents.json.

    Input:
        Dict with keys: "metadata", "bio", "projects", "experience", "skills", …

    Output:
        Flat list of child-chunk dicts (ready to write as an ARRAY to chunks.json).
    """
    last_updated = (contents.get("metadata") or {}).get("last_updated")
    rows: List[Dict[str, Any]] = []

    # --- BIO ---
    bio = contents.get("bio") or {}
    if bio:
        pid = parent_id("bio", "bio")
        if bio.get("summary"):
            rows.append(_row(
                pid, "bio", "bio", "summary", 0, bio["summary"],
                entities=[], tags=[], company=None, project=None,
                last_updated=last_updated
            ))

    # --- PROJECTS ---
    for proj in contents.get("projects", []) or []:
        title = proj.get("title") or "project"
        pid = parent_id("project", title)

        # Entities = normalized list of tech_stack + tags (deduped, sorted).
        # This is useful metadata for retrieval filtering/boosting.
        entities, proj_tags = build_project_entities_and_tags(proj)

        def add_field(field: str, text: Optional[str], idx_base: int) -> int:
            """
            Add a paragraph field (description/problem/etc.) as one or more chunks.
            Detects markdown bold headings at paragraph starts and stores them in `section`.
            Returns the next available index.
            """
            if not text:
                return idx_base
            idx = idx_base
            for para in para_split(text):
                # detect '**Topic X:**' style heading and carry it as metadata
                section, body = detect_md_heading(para)
                payload = body if section else para  # set to `para` if you want to keep the heading in text
                for piece in pack_sentences(payload, max_chars=MAX_CHARS_PARAGRAPH):
                    row = _row(
                        pid, "project", title, field, idx, piece,
                        entities=entities, tags=proj_tags,
                        company=None, project=title, last_updated=last_updated
                    )
                    if section:
                        row["section"] = section
                    rows.append(row)
                    idx += 1
            return idx

        i = 0
        i = add_field("tagline", proj.get("tagline"), i)
        i = add_field("description", proj.get("description"), i)
        i = add_field("problem", proj.get("problem"), i)
        i = add_field("architecture", proj.get("architecture"), i)
        i = add_field("challenges", proj.get("challenges"), i)

        outcomes = proj.get("outcomes") or {}
        i = add_field("outcomes.metrics", outcomes.get("metrics"), i)
        i = add_field("outcomes.impact", outcomes.get("impact"), i)

        # Features: one child per bullet (or more if SPLIT_LONG_BULLETS=True)
        for feat in proj.get("features", []) or []:
            for piece in bullet_children(feat):
                rows.append(_row(
                    pid, "project", title, "feature", i, piece,
                    entities=entities, tags=proj_tags,
                    company=None, project=title, last_updated=last_updated
                ))
                i += 1

        # Optional: compact link line to hint presence of live/GitHub
        links = proj.get("links") or {}
        link_text_parts: List[str] = []
        if links.get("live"):
            link_text_parts.append(f"Live: {links['live']}")
        if links.get("github"):
            link_text_parts.append(f"GitHub: {links['github']}")
        if link_text_parts:
            rows.append(_row(
                pid, "project", title, "links", i, " | ".join(link_text_parts),
                entities=entities, tags=proj_tags,
                company=None, project=title, last_updated=last_updated
            ))

    # --- EXPERIENCE ---
    for exp in contents.get("experience", []) or []:
        company = exp.get("company")
        position = exp.get("position")
        title = f"{company} — {position}" if (company or position) else "experience"
        pid = parent_id("experience", company or (title or "experience"))

        entities = sorted(set(norm_list(exp.get("tech_stack")) + norm_list(exp.get("tags"))))
        start = (exp.get("date_range") or {}).get("start")
        end = (exp.get("date_range") or {}).get("end")

        def add_exp_field(field_name: str, text: Optional[str], idx_base: int) -> int:
            """Paragraph-like field with optional **Heading:** detection."""
            if not text:
                return idx_base
            idx = idx_base
            for para in para_split(text):
                section, body = detect_md_heading(para)
                payload = body if section else para
                for piece in pack_sentences(payload, max_chars=MAX_CHARS_PARAGRAPH):
                    row = _row(
                        pid, "experience", title, field_name, idx, piece,
                        entities=entities, tags=(exp.get("tags") or []),
                        company=company, project=None, last_updated=last_updated,
                        start=start, end=end
                    )
                    if section:
                        row["section"] = section
                    rows.append(row)
                    idx += 1
            return idx

        i = 0
        # Original fields (still supported)
        i = add_exp_field("description", exp.get("description"), i)

        # Your custom fields (strings with possible "**Heading:**" prefixes)
        i = add_exp_field("company_description", exp.get("company_description"), i)
        i = add_exp_field("projects_worked_on", exp.get("projects_worked_on"), i)

        # responsibilities can be list or string
        resp = exp.get("responsibilities")
        if isinstance(resp, list):
            for r in resp:
                for piece in bullet_children(r):
                    rows.append(_row(
                        pid, "experience", title, "responsibility", i, piece,
                        entities=entities, tags=(exp.get("tags") or []),
                        company=company, project=None, last_updated=last_updated,
                        start=start, end=end
                    ))
                    i += 1
        elif isinstance(resp, str):
            i = add_exp_field("responsibilities", resp, i)

        # achievements can be list or string (your sample is a string)
        ach = exp.get("achievements")
        if isinstance(ach, list):
            for a in ach:
                for piece in bullet_children(a):
                    rows.append(_row(
                        pid, "experience", title, "achievement", i, piece,
                        entities=entities, tags=(exp.get("tags") or []),
                        company=company, project=None, last_updated=last_updated,
                        start=start, end=end
                    ))
                    i += 1
        elif isinstance(ach, str):
            i = add_exp_field("achievements", ach, i)

        # Tech stack summary line
        stack = norm_list(exp.get("tech_stack"))
        if stack:
            rows.append(_row(
                pid, "experience", title, "tech_stack", i,
                "Technologies used: " + ", ".join(sorted(set(stack))) + ".",
                entities=entities, tags=(exp.get("tags") or []),
                company=company, project=None, last_updated=last_updated,
                start=start, end=end
            ))

    # --- SKILLS (short narrative) ---
    skills = contents.get("skills") or {}
    if skills:
        pid = parent_id("skills", "skills")
        langs = norm_list(skills.get("languages"))
        fws = norm_list(skills.get("frameworks"))
        tools = norm_list(skills.get("tools_and_platforms"))
        entities = sorted(set(langs + fws + tools))
        parts: List[str] = []
        if langs: parts.append(f"Languages: {', '.join(langs)}.")
        if fws: parts.append(f"Frameworks: {', '.join(fws)}.")
        if tools: parts.append(f"Tools & Platforms: {', '.join(tools)}.")
        txt = " ".join(parts) if parts else "Skills catalog."
        rows.append(_row(
            pid, "skills", "skills", "catalog", 0, txt,
            entities=entities, tags=[], company=None, project=None,
            last_updated=last_updated
        ))

    return rows


def _row(
    pid: str, ptype: str, ptitle: str, field: str, idx: int, text: str,
    entities: List[str], tags: List[str], company: Optional[str], project: Optional[str],
    last_updated: Optional[str], start: Optional[str] = None, end: Optional[str] = None
) -> Dict[str, Any]:
    """
    Args:
        pid: Parent id.
        ptype: Parent type ("bio"|"project"|"experience"|"skills").
        ptitle: Parent title (e.g., "RAG Portfolio" or "Company — Role").
        field: Child field name (e.g., "description", "feature").
        idx: Child index (position within the field).
        text: Child text content.
        entities: Normalized skills/tags for retrieval boosts.
        tags: Free-form tags from your JSON.
        company, project: Optional scoping fields for filtering.
        last_updated: ISO date from metadata.
        start, end: Optional YYYY-MM range (experience only).

    Output:
        Child-chunk dict ready for embedding and storage.
    """
    return {
        "id": child_id(pid, field, idx, text),
        "parent_id": pid,
        "parent_type": ptype,
        "parent_title": ptitle,
        "field": field,
        "index": idx,
        "text": text,
        "entities": entities,
        "tags": tags or [],
        "company": company,
        "project": project,
        "date_start": start,
        "date_end": end,
        "last_updated": last_updated,
    }


# ------------------------------- minimal CLI ---------------------------------
if __name__ == "__main__":
    # Load your contents.json (array output preferred for you)
    src = Path("contents.json")
    data = json.loads(src.read_text(encoding="utf-8"))

    # Build child chunks
    rows = build_children(data)

    # Write ARRAY JSON instead of JSONL
    out_path = Path("chunks.json")
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {len(rows)} chunks -> {out_path.resolve()}")

    # Optional: quick embedding sanity-check
    if FASTEMBED_AVAILABLE and rows:
        embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        vec = next(iter(embedder.embed([rows[0]["text"]])))
        print(f"[OK] embedded 1 sample | dim={len(vec)} | first5={list(vec)[:5]}")
    else:
        if not FASTEMBED_AVAILABLE:
            print("[note] fastembed not installed (use `uv add fastembed` if you want embeddings)")
