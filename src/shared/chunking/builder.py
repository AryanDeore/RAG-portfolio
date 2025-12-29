"""This file orchestrates turning contents.json data into flat chunk rows using utils, ids, and entities helpers."""

from typing import Any, Dict, List, Optional
from .ids import parent_id, child_id
from .utils import para_split, detect_md_heading, pack_sentences, bullet_children, norm_list
from .entities import build_project_entities_and_tags

def _row(
    pid: str, ptype: str, ptitle: str, field: str, idx: int, text: str,
    entities: List[str], tags: List[str], company: Optional[str], project: Optional[str],
    last_updated: Optional[str], start: Optional[str] = None, end: Optional[str] = None
) -> Dict[str, Any]:
    """Construct a uniform chunk row dictionary.
    
    Args:
        pid (str): Parent ID.
        ptype (str): Parent type.
        ptitle (str): Parent title.
        field (str): Field name.
        idx (int): Index.
        text (str): Text content.
        entities (List[str]): List of entities.
        tags (List[str]): List of tags.
        company (Optional[str]): Company name.
        project (Optional[str]): Project name.
        last_updated (Optional[str]): Last updated timestamp.
        start (Optional[str]): Start date.
        end (Optional[str]): End date.
        
    Returns:
        Dict[str, Any]: Dict representing a chunk row ready for embedding/storage.
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

def build_children(contents: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pure builder that converts a loaded contents.json dict into flat chunks.
    
    Args:
        contents (Dict[str, Any]): Contents dictionary.
        
    Returns:
        List[Dict[str, Any]]: List of chunk dictionaries; each has id/parent/field/index/text/metadata.
    """
    last_updated = (contents.get("metadata") or {}).get("last_updated")
    rows: List[Dict[str, Any]] = []

    # --- BIO ---
    bio = contents.get("bio") or {}
    if bio and bio.get("summary"):
        pid = parent_id("bio", "bio")
        rows.append(_row(
            pid, "bio", "bio", "summary", 0, bio["summary"],
            entities=[], tags=[], company=None, project=None, last_updated=last_updated
        ))

    # --- PROJECTS ---
    for proj in contents.get("projects", []) or []:
        title = proj.get("title") or "project"
        pid = parent_id("project", title)
        entities, proj_tags = build_project_entities_and_tags(proj)

        def add_field(field: str, text: Optional[str], idx_base: int) -> int:
            """Split a long field into chunks, preserving detected **Heading:** as 'section'.
            
            Args:
                field (str): Field name.
                text (Optional[str]): Text content.
                idx_base (int): Base index.
                
            Returns:
                int: Next index after appending created rows.
            """
            if not text:
                return idx_base
            idx = idx_base
            for para in para_split(text):
                section, body = detect_md_heading(para)
                payload = body if section else para
                # Use hierarchical field name when section is detected
                effective_field = f"{field}:{section}" if section else field
                for piece in pack_sentences(payload):
                    row = _row(
                        pid, "project", title, effective_field, idx, piece,
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

        # Handle features - can be a string or a list
        features = proj.get("features")
        if isinstance(features, str):
            # Treat as a string field with section detection
            i = add_field("features", features, i)
        elif isinstance(features, list):
            # Handle as list of bullet items
            for feat in features:
                for piece in bullet_children(feat):
                    rows.append(_row(
                        pid, "project", title, "feature", i, piece,
                        entities=entities, tags=proj_tags,
                        company=None, project=title, last_updated=last_updated
                    ))
                    i += 1

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
            """Append paragraph-like experience fields as chunks with optional **Heading:** detection.
            
            Args:
                field_name (str): Field name.
                text (Optional[str]): Text content.
                idx_base (int): Base index.
                
            Returns:
                int: Next index after appending created rows.
            """
            if not text:
                return idx_base
            idx = idx_base
            for para in para_split(text):
                section, body = detect_md_heading(para)
                payload = body if section else para
                # Use hierarchical field name when section is detected
                effective_field = f"{field_name}:{section}" if section else field_name
                for piece in pack_sentences(payload):
                    row = _row(
                        pid, "experience", title, effective_field, idx, piece,
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
        i = add_exp_field("description", exp.get("description"), i)
        i = add_exp_field("company_description", exp.get("company_description"), i)
        i = add_exp_field("projects_worked_on", exp.get("projects_worked_on"), i)

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

        stack = norm_list(exp.get("tech_stack"))
        if stack:
            rows.append(_row(
                pid, "experience", title, "tech_stack", i,
                "Technologies used: " + ", ".join(sorted(set(stack))) + ".",
                entities=entities, tags=(exp.get("tags") or []),
                company=company, project=None, last_updated=last_updated,
                start=start, end=end
            ))

    # --- SKILLS ---
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
