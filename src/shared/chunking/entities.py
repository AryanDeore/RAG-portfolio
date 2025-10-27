"""This file builds project entities and tags by loading a token alias map and trigger catalog from JSON, normalizing declared tech, harvesting mentions from free text, and inferring simple auto tags."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from configs.settings import settings
from .utils import norm_list

# Module-level variables for testing - can be monkeypatched
_TECH_ALIAS = {}
_TECH_CATALOG = {}

def _load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file into a Python dict with a safe fallback.
    
    Args:
        path (str): Filesystem path to a JSON file.
        
    Returns:
        Dict[str, Any]: Parsed JSON object or an empty dict on failure.
    """
    try:
        p = Path(path)
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _normalize_token(tok: str, alias_map: Dict[str, str]) -> str:
    """Canonicalize a technology token using an alias map and simple capitalization rules.
    
    Args:
        tok (str): Raw token.
        alias_map (Dict[str, str]): Mapping of lowercase alias->canonical.
        
    Returns:
        str: Canonical display token.
    """
    t = (tok or "").strip()
    if not t:
        return t
    key = t.lower()
    if key in alias_map:
        return alias_map[key]
    if "." in t:
        parts = t.split(".")
        return ".".join(p[:1].upper() + p[1:] for p in parts)
    return t[:1].upper() + t[1:]

def extract_entities_from_text(text: str, catalog: Dict[str, List[str]] = None, alias_map: Dict[str, str] = None) -> List[str]:
    """Harvest canonical entities from free text using trigger phrases defined in a catalog.
    
    Args:
        text (str): Free text to scan.
        catalog (Dict[str, List[str]], optional): Canon->trigger list. Uses module-level _TECH_CATALOG if None.
        alias_map (Dict[str, str], optional): Alias normalization map. Uses module-level _TECH_ALIAS if None.
        
    Returns:
        List[str]: Sorted, deduplicated canonical entity list.
    """
    if not text:
        return []
    
    # Use provided parameters or fall back to module-level variables
    if catalog is None:
        if _TECH_CATALOG:
            catalog = _TECH_CATALOG
        else:
            catalog_raw = _load_json(settings.tech_catalog_path)
            catalog = {k: (v if isinstance(v, list) else []) for k, v in (catalog_raw or {}).items()}
    
    if alias_map is None:
        if _TECH_ALIAS:
            alias_map = _TECH_ALIAS
        else:
            alias_map_raw = _load_json(settings.tech_alias_path)
            alias_map = {k.lower(): v for k, v in alias_map_raw.items()} if isinstance(alias_map_raw, dict) else {}
    
    s = text.lower()
    found: List[str] = []
    for canon, triggers in catalog.items():
        for trig in triggers:
            if trig in s:
                found.append(_normalize_token(canon, alias_map))
                break
    return sorted(set(found))

def build_project_entities_and_tags(proj: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Build a pair (entities, tags) for a project using declared tech/tags and harvested content.
    
    Args:
        proj (Dict[str, Any]): A single project object from contents.json.
        
    Returns:
        Tuple[List[str], List[str]]: (entities, tags) suitable for retrieval and filtering.
    """
    # Use module-level variables if available (for testing), otherwise load from JSON
    if _TECH_ALIAS:
        alias_map_raw = _TECH_ALIAS
    else:
        alias_map_raw = _load_json(settings.tech_alias_path)  # e.g., {"postgresql": "PostgreSQL", ...}
    alias_map: Dict[str, str] = {k.lower(): v for k, v in alias_map_raw.items()} if isinstance(alias_map_raw, dict) else {}

    if _TECH_CATALOG:
        catalog_raw = _TECH_CATALOG
    else:
        catalog_raw = _load_json(settings.tech_catalog_path)  # e.g., {"aws": ["aws", "amazon web services"], ...}
    catalog: Dict[str, List[str]] = {k: (v if isinstance(v, list) else []) for k, v in (catalog_raw or {}).items()}

    declared_tech = norm_list(proj.get("tech_stack"))
    declared_tags = norm_list(proj.get("tags"))

    norm_tech = [_normalize_token(t, alias_map) for t in declared_tech if t]

    text_fields = [
        proj.get("description") or "",
        proj.get("problem") or "",
        proj.get("architecture") or "",
        proj.get("challenges") or "",
        " ".join(proj.get("features") or []) if isinstance(proj.get("features"), list) else (proj.get("features") or ""),
        json.dumps(proj.get("outcomes") or {}, ensure_ascii=False),
    ]
    harvested = extract_entities_from_text(" \n ".join(text_fields), catalog, alias_map)
    entities = sorted(set(norm_tech + harvested))

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
