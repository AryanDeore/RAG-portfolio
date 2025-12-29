# Chunking Implementation

Transforms structured portfolio data (`contents.json`) into atomic, embeddable chunks with rich metadata for RAG retrieval.

---

## Architecture

**Parent-Child Model:**
- **Parents**: Top-level entities (bio, projects, experience entries, skills)
- **Children**: Atomic chunks derived from parent content
- **Metadata Inheritance**: Children inherit parent context (company, project, dates, entities, tags)

**Why?** Enables structured retrieval (fetch entire sections), atomic search units, and filtering without duplication.

---
Things to keep in mind while chunking. Points should be split with "new para split" which is \n\n

## Components

```
src/shared/chunking/
├── builder.py    # Main orchestration: contents.json → chunks
├── entities.py   # Entity extraction & auto-tagging
├── ids.py        # Deterministic ID generation (UUIDv5 + content hash)
├── utils.py      # Text processing (paragraphs, sentences, bullets)
├── io.py         # JSON I/O helpers
└── cli.py        # Command-line interface
```

---

## Chunking Strategy

### Projects
- Fields: `tagline`, `description`, `problem`, `architecture`, `challenges`, `outcomes.metrics`, `outcomes.impact`, `features[]`, `links`
- Processing: Paragraphs → sentences (max 700 chars), bullets stay atomic
- Special: Markdown headings (`**Section:** body`) extracted as `section` metadata
- Metadata: Entities from tech stack + text harvesting, auto-tags (cloud/devops/frontend/backend/real-time)

### Experience
- Fields: `description`, `company_description`, `projects_worked_on`, `responsibilities[]`, `achievements[]`, `tech_stack`
- Processing: Same as projects (paragraphs → sentences)
- Metadata: Inherits company, date range, normalized tech stack

### Bio & Skills
- Bio: Single chunk for `summary`
- Skills: Single formatted chunk ("Languages: X. Frameworks: Y. Tools: Z.")

---

## ID Generation

**Parent IDs:** `parent_id(kind, name)` → UUIDv5 from `{kind}::{name}`
- Deterministic, stable across runs

**Child IDs:** `child_id(parent_id, field, idx, text)` → `{parent_id}:{field}:{idx}:{sha1_hash}`
- Content hash enables change detection and cache invalidation

---

## Entity Extraction

**Two-step process:**

1. **Normalization** (`tech_alias.json`): Maps aliases → canonical names
   - Example: `{"aws": "AWS", "amazon web services": "AWS"}`

2. **Harvesting** (`tech_catalog.json`): Scans text for trigger phrases
   - Example: `{"aws": ["aws", "amazon web services"]}`

**Auto-tags:** Generated from detected entities (cloud, devops, frontend, backend, real-time)

**Why?** Finds tech mentioned but not listed, enables filtering, ensures consistent naming.

---

## Text Processing

- **Paragraphs**: Split on `\n\n+`
- **Sentences**: Pack into chunks (max 700 chars) at sentence boundaries
- **Bullets**: One chunk per bullet (atomic facts, ideal size)
- **Headings**: Extract `**Heading:** body` patterns as `section` metadata

**Why sentence boundaries?** Preserves semantic meaning, better embedding quality than arbitrary splits.

---

## Data Flow

```
contents.json
  ↓
Load configs (tech_alias.json, tech_catalog.json)
  ↓
For each content type:
  - Create parent ID
  - Extract entities/tags
  - Chunk fields (paragraphs → sentences)
  - Create children with inherited metadata
  ↓
chunks.json (flat list)
```

**Chunk Schema:**
```json
{
  "id": "parent-id:field:0:hash",
  "parent_id": "parent-id",
  "parent_type": "project",
  "parent_title": "Project X",
  "field": "description",
  "index": 0,
  "text": "Chunk content...",
  "entities": ["Python", "FastAPI"],
  "tags": ["backend"],
  "company": null,
  "project": "Project X",
  "date_start": null,
  "date_end": null,
  "last_updated": "2025-01-27",
  "section": "Architecture"  // optional
}
```

---

## Configuration

```python
chunk_max_chars_paragraph: int = 700      # Max chars per chunk
chunk_split_long_bullets: bool = False    # Split bullets if long
chunk_max_chars_bullet: int = 700
tech_alias_path: str = "configs/tech_alias.json"
tech_catalog_path: str = "configs/tech_catalog.json"
```

---

## Usage

**CLI:**
```bash
python -m src.shared.chunking.cli [contents.json] [chunks.json]
```

**Programmatic:**
```python
from src.shared.chunking import build_children
from src.shared.chunking.io import read_contents, write_chunks

contents = read_contents("contents.json")
chunks = build_children(contents)
write_chunks(chunks, "chunks.json")
```

---

## Key Design Decisions

1. **Parent-child hierarchy**: Enables section retrieval + atomic search
2. **Deterministic IDs**: Stable IDs enable incremental updates without re-embedding
3. **Sentence-level chunking**: Preserves semantics vs arbitrary splits
4. **Entity harvesting**: Finds tech mentioned but not listed
5. **Metadata inheritance**: Avoids duplication, enables filtering
6. **Character limits**: Simpler than tokenization, ~100-150 tokens per 700 chars

---

## Limitations & Future Work

**Current:**
- No overlap between chunks
- Character-based limits (less precise than tokens)
- Simple substring-based entity extraction

**Potential:**
- Token-based limits
- Configurable overlap
- Chunk summaries for ranking
- Structured store (SQLite/DuckDB) for aggregations
