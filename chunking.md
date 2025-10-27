Short answer: don’t rely on “pure semantic chunking + summaries” alone. For a portfolio RAG that must answer aggregation, detail, tech-filtered, and time-scoped questions, use a **hybrid, schema-aware, parent→child chunking** approach paired with **multi-index retrieval (dense + sparse + structured)** and light **query routing**.

Here’s a battle-tested blueprint that fits your JSON perfectly.

# 1) Chunking strategy (schema-aware, parent→child)

Think in three layers: *document → section → atomic facts*.

**A. Parents (sections you want whole when needed)**

* One parent node per top-level item:

  * `bio`, each `projects[i]`, each `experience[i]`, and `skills` as a catalog.
* Store rich metadata on the parent (critical for filtering and routing):

  * `type`: project | experience | bio | skills
  * `title`, `company`, `position`, `date_start`, `date_end`, `tech_stack[]`, `tags[]`, `links`, `location`
  * `time_index`: normalized YYYY-MM and YYYY
  * `entities`: normalized skills/tech (e.g., “AWS”, “GCP”, “FastAPI”)

**B. Children (atomic, answerable facts)**
Split each parent into **small, semantically coherent children** (the units the retriever will rank). Target sizes:

* Bullets / short claims: **80–150 tokens**
* Paragraphs: **150–250 tokens**
* Overlap: **~30 tokens** for paragraphs; **0** for bullets

Examples:

* **Project nodes**

  * `tagline` (child)
  * `problem` (child)
  * `architecture` (1–2 children if long)
  * each `feature` bullet (one child per bullet)
  * `challenges` (1–2 children)
  * `outcomes.metrics` (child), `outcomes.impact` (child)
* **Experience nodes**

  * `description` (1 child)
  * each `responsibility` (one child)
  * each `achievement` (one child)
  * `tech_stack` (child list, also normalized to entities)
* **Skills**

  * Keep a **skills catalog** (structured table); also materialize a short **narrative child** like: “I’ve used AWS across Project A (Lambda, S3), Project B (ECS), and at Company X (Glue, Athena).”

**Why this works**

* Detail questions (“Tell me about the RAG Portfolio project”) can pull the **project parent** + top-ranked children (parent-child join).
* Aggregations (“What’s my experience with Python?”) can retrieve **many tiny, tech-tagged children** from different parents and synthesize.
* Time filters work via parent metadata (`date_range`) and are inherited by children.

# 2) What to embed & store in the vector DB

For **each child** store:

* `text`: the atomic fact (what the model should quote)
* `title`: short label (e.g., “Project: RAG Portfolio — Challenge: latency on Lambda cold starts”)
* `parent_id`
* `type`, `entities[]` (normalized tech/skills), `company`, `project`, `date_start/end`, `tags[]`

For **embeddings**, do both:

* **Dense embedding** on `text` *(semantic)*
* **Sparse/BM25 index** on `text + entities + title` *(lexical)*

  * If you don’t run ES/Opensearch, use a local BM25 (e.g., Tantivy, Lucene via Whoosh) or a **hybrid retriever** (dense + SPLADE/BM25).
* Optional but powerful: keep a **very short abstractive synopsis** (≤ 40–60 tokens) for each child focused on *question-answerability* (e.g., “Latency issue solved by provisioned concurrency on Lambda; cost +95th pctl improved 43%”). Embed it too and keep the original as ground truth. (This helps ranking while keeping faithful content.)

# 3) Multi-index retrieval (hybrid)

Use **RRF (Reciprocal Rank Fusion)** or **weighted blending** over:

1. Dense (vector) results (MMR-diversified, k≈8–12)
2. Sparse/BM25 results (k≈12–20)
3. **Structured hits** (see §4): when a query matches an entity/time filter, promote those children.

Post-retrieval **cross-encoder reranking** (e.g., `ms-marco-MiniLM-L-6-v2`) on the top 50 combined is a cheap, high-ROI upgrade for “Tell me everything about X” queries.

# 4) Don’t force RAG to do SQL’s job: add a tiny structured store

Some questions are **lists, filters, or counts** (“What cloud platforms have I used?”, “What did I do at Company X in 2023?”). Answer these from a **structured sidecar** (SQLite/DuckDB or in-memory Pandas), not the LLM. Keep tables:

* `projects(project_id, title, tech_stack[], tags[], start, end, links…)`
* `experience(exp_id, company, position, start, end, tech_stack[], tags[])`
* `skills(skill_name, first_seen, last_seen, contexts[])` (contexts: list of project/exp ids)
* A **many-to-many** table `entity_usage(entity, parent_id, parent_type, role)` (e.g., `entity='AWS'`, `role='lambda|ecs|glue'`)

At query time:

* If intent looks **aggregative / enumerative**, answer from the structured store (deterministic), then **ground with 2–4 retrieved children** as citations/examples.
* If intent looks **narrative**, rely on multi-index RAG.

# 5) Light query routing (no heavy agent needed)

A small intent router (few-shot classifier) directs retrieval:

* **Detail / “tell me about <project>”** → pull the **project parent**, then top 6–10 children; allow longer context (1.5–2k tokens).
* **Aggregation / “What’s my experience with Python?”** → structured store for list + fetch top 4–6 children that evidence breadth.
* **Tech-filtered / “What cloud platforms have I used?”** → structured store (entities table), then bring 3 exemplars from RAG.
* **Time-scoped / “What did I do at Company X (2023)?”** → structured filter on `experience.company` + date, then children under that parent with matching time.

# 6) Practical knobs

* Child chunk sizes: **80–250 tokens**, favor **one fact / claim per child** when possible (bullets = gold).
* Overlap: **0** for bullets; **~30 tokens** for paragraphs.
* Retrieval k: dense=12, sparse=20, **MMR** λ=0.5, RRF top=50 → rerank → pick 8–12.
* Metadata boosts: if query mentions an entity (`AWS`, `GCP`, `Next.js`), apply a **score boost** to children whose `entities[]` contain it; same for `company`, `project`, `date`.
* Normalization: store canonical synonyms (`AWS`↔`Amazon Web Services`, `GCP`↔`Google Cloud`) in `entities[]`.

# 7) How to build children from your JSON (example)

**Project → children**

* `(project_id, type='project', field='problem', text=problem)`
* `(…, field='architecture', text=slice_1)` … `(slice_2 if long)`
* `(…, field='feature', text=features[i])` for each feature
* `(…, field='challenge', text=challenges)` (split if long)
* `(…, field='metrics', text=outcomes.metrics)`
* `(…, field='impact', text=outcomes.impact)`
* Entities: union of `tech_stack[] + tags[]`, normalized

**Experience → children**

* `(exp_id, field='description', text=description)`
* one child per responsibility and per achievement
* Entities: `tech_stack[] + tags[]`
* Dates: inherit to children (`date_start`, `date_end`)

**Skills → structured + one narrative child**

* Structured table powers lists/filters
* Optional short child: “Where/How I used Python” with pointers to parents

# 8) Answer composition

* **Detail answers**: include parent synopsis (2–3 lines), then bullet the most relevant children (challenges, architecture, outcomes), with live links (from metadata) when available.
* **Aggregation answers**: generate a concise list/table (from structured store), then add **2–3 diverse child quotes** as evidence. Cite parent titles and date ranges.

# 9) Summaries: keep them short & targeted

Storing a **short, neutral synopsis per child** is great. Keep it ≤60 tokens, present-tense, and **don’t replace the original text**—use it to *help ranking*; always show original child snippets in the final context.

# 10) Updates & versioning

* Keep `metadata.last_updated` mirrored into every new parent/child batch via `version=2025-10-14`.
* On update, **tombstone** removed children (don’t silently overwrite); keep `parent_version`, `child_version`.
* Cache embeddings keyed by `sha1(text)` to avoid re-embedding unchanged nodes.

---

## TL;DR (recommended stack)

* **Chunking:** schema-aware parent→child; bullets as atomic children; 80–250 tok; small overlap.
* **Indexes:** dense + sparse (BM25) + tiny **structured store** (SQLite/DuckDB).
* **Retrieval:** hybrid (RRF) → cross-encoder rerank → parent-child join for context packing.
* **Routing:** 4 intents → pick query plan.
* **Metadata:** rich facets (project/experience/tech/time/tags) for filtering & score boosts.
* **Summaries:** short, ranking-oriented synopses per child; keep originals for grounding.

This will outperform pure semantic chunking with summaries—especially on *aggregation* (“what have I used?”) and *time-scoped* (“what did I do in 2023 at X?”) queries—while still giving excellent project deep-dives.


What to test (stepwise variants)

Name each variant and change only one thing at a time.

S0 — Baseline: child-only chunks (80–200 tok), dense embeddings only.

S1 — +Parent join: when a child wins, also fetch its parent summary (project/experience header).

S2 — +Hybrid retrieval: dense ⊕ BM25 (RRF or weighted blend).

S3 — +Reranker: cross-encoder reranks top 50 → top 10.

S4 — +Metadata boosts: score boosts for matches on entities, company, project, years.

S5 — +Structured router: for list/aggregation/time queries, answer from a tiny SQLite table first, then pull 2–4 evidential children.

S6 — +Child synopses: add 40–60 token synopses for each child (used for ranking only; keep originals for context).