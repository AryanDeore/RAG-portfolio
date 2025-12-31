# Interactive AI Portfolio

[![Live Demo](https://img.shields.io/badge/Live_Application-Visit_Website-2ea44f?style=for-the-badge&logo=vercel)](https://www.aryandeore.ai/)<br>
[![Stack](https://img.shields.io/badge/Stack-FastAPI_|_Next.js_|_Qdrant_|_LiteLLM_|_Comet_Opik-blue)](https://github.com/AryanDeore/RAG-portfolio)

**An interactive, Retrieval-Augmented Generation (RAG) system that allows users to "interview" my portfolio.**

Instead of presenting static text, this application enables users to ask natural language questions—such as *"What is your experience with MLOps?"* or *"How do you monitor and evaluate LLM performance?"*—and receive grounded, cited answers based on my projects and professional history.

## Architecture

<img width="3367" height="4613" alt="arch-dark" src="https://github.com/user-attachments/assets/0c79f4e6-492f-43d2-9ecb-b0592461675a" />


The logic is divided into three distinct pipelines:

### 1. Ingestion Pipeline:
One of the primary challenges in RAG is that standard recursive text splitters often sever the semantic link between specific entities (e.g., a "Skill") and the larger context (e.g., the "Project" it belongs to).

*   **Strategy:** **Schema-Aware Parent-Child Chunking**.
*   The system respects the JSON schema of the source data. Small "Child" chunks (paragraphs) are indexed for dense vector retrieval, but they remain linked to their "Parent" chunks (full job entries).
*   **Outcome:** When a specific skill is queried, the system retrieves the *entire* job entry, ensuring the LLM maintains full context regarding dates, roles, and company details.
### 2. Retrieval Pipeline:
*   **Guardrails:** All incoming queries are pre-validated via the OpenAI Moderation API to intercept and block harmful inputs before they reach the inference layer.
*   **Embeddings:** **FastEmbed** is utilized to run the `BAAI/bge-small-en-v1.5` model locally on the server. This approach eliminates external embedding API costs and reduces latency.
*   **Vector Search:** **Qdrant** performs a cosine similarity search to identify the most relevant child chunks before fetching their associated parent contexts.
### 3. Generation Pipeline:
* **OpenRouter** allows for model hot-swapping (e.g., GPT-4 to Sonnet 3.5) via configuration changes.
*   **LiteLLM** serves as a unified interface, decoupling the backend logic from specific providers. 
*   **Streaming:** Responses are delivered via Server-Sent Events (SSE) to the Next.js frontend, achieving a **Time-To-First-Token (TTFT) of <1 second**.

## Observability
**Comet Opik** tracks every trace, every interaction within the system. This monitors:
*   **Traces:** Full visibility into retrieval steps, including the exact chunks retrieved and their similarity scores.
*   **Cost Analysis:** Token usage tracking per query to optimize spend (maintaining an average of **~$0.001 per query**).

