# This file contains small, pure text utilities for bullets, paragraph/sentence splitting, and simple normalization.

import re
from typing import List, Optional, Any
from configs.settings import settings

# Precompiled regular expressions for performance
_BULLET_RE = re.compile(r"^\s*[-*•]\s+")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_HEADING_CASE_A = re.compile(r"^\s*\*\*(.+?)\*\*\s*[:\-–]\s*(.*)$")
_HEADING_CASE_B = re.compile(r"^\s*\*\*\s*(.+?)\s*[:：]\s*\*\*\s*(.*)$")

def looks_bullet(s: str) -> bool:
    """Check if a line starts with a bullet marker.
    
    Args:
        s (str): A single line of text.
        
    Returns:
        bool: True if it looks like a bullet; False otherwise.
    """
    return bool(_BULLET_RE.match(s or ""))

def strip_bullet(s: str) -> str:
    """Remove a single leading bullet marker from a line of text.
    
    Args:
        s (str): A possibly bulleted string.
        
    Returns:
        str: The text without the bullet prefix.
    """
    return _BULLET_RE.sub("", s or "").strip()

def para_split(text: Optional[str]) -> List[str]:
    """Split raw text into paragraphs using blank lines as separators.
    
    Args:
        text (Optional[str]): Raw text with possible blank lines.
        
    Returns:
        List[str]: Non-empty, trimmed paragraphs.
    """
    if not text:
        return []
    return [p.strip() for p in _PARA_SPLIT_RE.split(text) if p.strip()]

def sent_split(text: str) -> List[str]:
    """Split a paragraph into sentences using punctuation boundaries.
    
    Args:
        text (str): A paragraph.
        
    Returns:
        List[str]: Sentence strings without trailing spaces.
    """
    return [s.strip() for s in _SENT_RE.split(text or "") if s.strip()]

def pack_sentences(paragraph: str, max_chars: Optional[int] = None) -> List[str]:
    """Pack sentences into chunks up to a character limit while preserving sentence boundaries.
    
    Args:
        paragraph (str): The paragraph to split.
        max_chars (Optional[int]): Size cap (defaults to settings).
        
    Returns:
        List[str]: Chunk strings near the max size.
    """
    limit = max_chars or settings.chunk_max_chars_paragraph
    out: List[str] = []
    cur: List[str] = []
    n = 0
    for s in sent_split(paragraph or ""):
        if n + len(s) + 1 <= limit or not cur:
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
    """Normalize a value to a flat list of trimmed strings.
    
    Args:
        x (Any): List/str/other to be normalized.
        
    Returns:
        List[str]: Clean list of strings.
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
    """Detect a bold markdown heading at the start of a paragraph (e.g., "**Title:** body").
    
    Args:
        paragraph (str): Paragraph possibly starting with a heading.
        
    Returns:
        Tuple[Optional[str], str]: (section label or None, remaining body text).
    """
    s = (paragraph or "").strip()
    if not s:
        return None, ""
    m = _HEADING_CASE_A.match(s)
    if m:
        return m.group(1).strip(), (m.group(2) or "").strip()
    m2 = _HEADING_CASE_B.match(s)
    if m2:
        return m2.group(1).strip(), (m2.group(2) or "").strip()
    return None, s

def bullet_children(text: str) -> List[str]:
    """Convert a bullet string into one or more child strings depending on configuration.
    
    Args:
        text (str): Bullet text (with or without a leading marker).
        
    Returns:
        List[str]: One or more child strings derived from the bullet.
    """
    clean = strip_bullet(text) if looks_bullet(text or "") else str(text or "").strip()
    if not clean:
        return []
    if not settings.chunk_split_long_bullets:
        return [clean]
    parts: List[str] = []
    paras = para_split(clean) or [clean]
    for p in paras:
        parts.extend(pack_sentences(p, max_chars=settings.chunk_max_chars_bullet))
    return parts
