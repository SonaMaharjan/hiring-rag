# Hiring RAG Assistant

An agentic, multi-tier Candidate Ranking and Self-RAG Assistant built for hiring managers. It automates job description requirement extraction, enforces explicit manager approval gates, audits candidate PDF resumes for security prompt injections, ranks candidates into flag tiers using multi-factor evaluation, and answers candidate-specific queries with hallucination-free Self-RAG.

---

## Demo Video

Watch the complete walkthrough and live demonstration on YouTube:

[![Hiring RAG Assistant Demo](https://img.youtube.com/vi/oTDO0Yx9sIY/maxresdefault.jpg)](https://youtu.be/oTDO0Yx9sIY)

🔗 **[Watch on YouTube: https://youtu.be/oTDO0Yx9sIY](https://youtu.be/oTDO0Yx9sIY)**

---

## What This Project Does

1. **Job Ingestion & Approval Gate**: A hiring manager pastes a job description. The system automatically extracts key requirements (Education, Language, Certifications, Experience Years). Ranking remains **blocked** until the manager explicitly confirms by typing `approve`.
2. **Resume Ingestion & Security Audit**: Candidate PDF resumes placed in `resumes/` are parsed and scanned by an LLM security checkpoint for prompt injection attacks (e.g., attempts to force top scores or override system instructions). Malicious CVs are isolated in `Prompt Injection Flag/`.
3. **Multi-Factor Candidate Ranking**: Valid candidates are evaluated using:
   - **Cosine Similarity** (30% Full CV + 70% Work Experience embeddings vs. Job Posting requirements).
   - **Skill Overlap Score** (LLM-judged semantic skill matching, 0.0 to 1.0).
   - **Experience Ratio** (Candidate years vs. required years).
   - **Hard-Gate Audit**: LLM contradiction checks categorize candidates into **Tier 0** (0 flags - perfect match), **Tier 1** (1 flag - minor mismatch), and **Tier 2** (2+ flags - major mismatches) with tie-aware ranking.
4. **Self-RAG Candidate Q&A**: Managers can ask specific questions about any candidate. The assistant uses a **Self-RAG loop** (Retrieval → Relevance Evaluation → Context-Bounded Answer → Grounding Verification) to answer strictly from the candidate's CV without hallucinating.

---

## Technology Stack & Tools Used

- **Agent Framework & Backend**: [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/), `google-agents-cli`, FastAPI, A2A Protocol, Uvicorn.
- **Hybrid LLM Architecture**:
  - **Gemini 2.5 Flash** (Cloud API via Google AI Studio / Vertex AI): Handles real-time Chat Agent interaction and tool execution (1 call per chat message).
  - **Llama 3.2** (Local LLM via [Ollama](https://ollama.com/)): Offloads heavy, high-frequency batch processing (CV metadata extraction, prompt injection audits, skill overlap scoring, hard gate audits, Self-RAG relevance grading & grounding checks) to reduce cloud resource usage.
- **Observability & Tracing**: [LangSmith](https://www.langchain.com/langsmith) with `@traceable` instrumentation across all pipelines.
- **Vector Database**: [Chroma DB](https://www.trychroma.com/) (`chromadb` persistent client) storing 3 collections: `cv_profiles`, `cv_experience_profiles`, and `cv_chunks`.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace Inference API with local `sentence-transformers` fallback).
- **Relational Storage**: SQLite (`hiring_rag.db`) for candidate profiles, job posting metadata, session states, and cached score profiles.
- **PDF Extraction**: `pypdf`.

---

## Work in Progress (WIP) & Production Readiness Roadmap

> [!WARNING]
> This project is a **Work in Progress (WIP)** designed as an exploratory agentic RAG system and is **currently far from production-ready**.

### What is Missing for Production Readiness:

1. **Refined & Calibrated Ranking Logic**:
   - *Current State*: The candidate scoring mechanism relies on a fixed heuristic combination of cosine similarity, LLM skill judging, and experience ratios, which can currently feel somewhat arbitrary or simple.
   - *Production Need*: Research and implement advanced data-driven ranking algorithms, such as **Cross-Encoders for fine-grained re-ranking**, **Learning-to-Rank (LTR)** models trained on recruiter feedback, and **LLM Preference Alignment (DPO/RLHF)** to deliver objective, calibrated candidate evaluations.

2. **Enterprise Security & Advanced Threat Mitigation**:
   - *Current State*: Basic regex parsing and single-pass LLM prompt injection detection.
   - *Production Need*: Robust enterprise security layers including sandboxed PDF evaluation, automated PII (Personally Identifiable Information) anonymization/redaction pipelines, rate-limiting guardrails, and adversarial input protection.

3. **Continuous Evaluation & Automated Benchmarking**:
   - *Current State*: Unit test coverage and basic trace logging.
   - *Production Need*: Continuous automated evaluation suites (`agents-cli eval`), LLM-as-a-judge regression testing, dataset synthesis for edge-case CVs, latency SLA tracking, and real-time failure mode clustering (`agents-cli eval analyze`).

4. **Async & Distributed Task Queues**:
   - *Current State*: Resume ingestion and candidate scoring execute synchronously in loops.
   - *Production Need*: Integrate asynchronous worker queues (e.g., Celery, Redis, or GCP Cloud Tasks) to handle concurrent bulk uploads of hundreds of resumes without blocking the API thread.

5. **Advanced Document & Layout Parsing**:
   - *Current State*: Uses basic `pypdf` text extraction.
   - *Production Need*: Upgrade to OCR / layout-aware parsers (e.g., Unstructured, PyMuPDF, or Google DocAI) to parse complex multi-column resumes, tables, scanned images, and styled PDFs cleanly.

6. **Production-Grade Vector Search & Multi-Tenancy**:
   - *Current State*: Uses local disk-based Chroma DB (`chroma_db/`).
   - *Production Need*: Migrate to scalable enterprise vector databases (e.g., Qdrant, Pinecone, or Vertex AI Vector Search) with namespace isolation and tenant partitioning for multi-organization security.

7. **Authentication, Authorization & Data Isolation**:
   - *Current State*: Open FastAPI endpoints without user authentication.
   - *Production Need*: Implement OAuth2 / JWT authentication, Role-Based Access Control (RBAC), organization-level tenant isolation, and payload encryption at rest and in transit.

8. **CI/CD & Cloud Infrastructure**:
   - *Current State*: Local execution via `agents-cli playground`.
   - *Production Need*: Terraform IaC, Docker containerization, Kubernetes / GCP Cloud Run automated deployment, secrets management (GCP Secret Manager), and automated linting/testing in CI pipelines.

---

## Quick Start Guide

### Prerequisites
- **Python 3.11**
- **uv**: `uv tool install google-agents-cli`
- **Ollama** (for local Llama 3.2 hybrid model execution): `ollama pull llama3.2`

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/hiring-rag.git
   cd hiring-rag
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   agents-cli install
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```
   ```env
   GEMINI_API_KEY=your-gemini-api-key-here
   USE_LOCAL_LLM=true
   LOCAL_LLM_URL=http://localhost:11434/v1
   LOCAL_LLM_MODEL=llama3.2
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your-langsmith-api-key-here
   LANGCHAIN_PROJECT=hiring-rag
   ```

4. **Start Local Agent Playground**:
   ```bash
   # Ensure Ollama is running in a separate terminal: ollama serve
   agents-cli playground
   ```
   Access the web interface at `http://127.0.0.1:8080/dev-ui/?app=app`.

---

## Testing

Run unit and integration test suites:
```bash
uv run pytest tests/unit tests/integration
```

---

## License

Apache License 2.0
