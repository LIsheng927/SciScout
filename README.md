# SciScout — Multi-Agent Deep Research System for Scientific Literature

[中文说明 →](./README.zh-CN.md)

A multi-agent research assistant: given a research question, it decomposes the question into
subtasks, concurrently retrieves papers from multiple academic sources, runs reflection-driven
multi-round retrieval, generates a report with inline citation sourcing, and supports multi-turn
follow-up questions on the finished report.

## Architecture

```
User question
   │
   ▼
PlannerAgent ── decomposes into 3-5 searchable subtasks (structured JSON, Pydantic-validated)
   │
   ▼
Concurrent retrieval (ThreadPoolExecutor, capped at 3 concurrent workers)
   ├── ArxivSearchTool        ─┐
   └── SemanticScholarTool    ─┴─→ cross-source dedup by URL + normalized title
   │
   ▼
ReflectionAgent ── decides at runtime whether coverage is sufficient; if not, generates
   │                follow-up subtasks and loops back into retrieval
   ▼ (up to max_rounds)
ReportAgent ── writes a structured report from paper titles/abstracts, every claim tagged with [n]
   │
   ▼
CitationVerifier ── an independent LLM call (temperature=0) that checks whether each citation
   │                is actually supported by the cited abstract
   ▼
FastAPI WebSocket connection, streaming every step's progress to the frontend in real time
   │
   ▼
FollowUpAgent ── the connection stays open after the report; supports multi-turn follow-up
                  questions (short-term memory via a sliding history window)
```

A hybrid dense + BM25 retrieval layer (`app/tools/rag_tool.py`, fused with Reciprocal Rank
Fusion) has been implemented and verified in isolation, but is **not yet wired into** the report
generation path above (report/follow-up currently reason only over titles + abstracts, not full
paper text) — see "Known Limitations" below for why.

## Core Capabilities

- **Multi-agent orchestration loop**: Planner decomposes → concurrent retrieval → Reflection
  decides at runtime whether another retrieval round is needed (an LLM judgment call at runtime,
  not a hardcoded fixed pipeline — this is what distinguishes it from a plain pipeline)
- **Multi-source concurrent retrieval**: arXiv + Semantic Scholar queried concurrently via
  `ThreadPoolExecutor`; deduplicated across sources by normalized title (different sources give
  the same paper different URLs, so URL-only dedup misses cross-source duplicates)
- **Hybrid RAG retrieval**: dense (text-embedding-3-small) + BM25 sparse retrieval, fused via
  Reciprocal Rank Fusion; verified to promote keyword-dense but semantically "less polished"
  passages from rank #3 (dense-only) to rank #1 (hybrid)
- **Cited report generation + citation verification**: every claim is tagged `[n]`; an
  independent `CitationVerifier` call at `temperature=0` re-checks whether each citation is
  actually supported by its abstract
- **Multi-turn follow-up (short-term memory)**: the WebSocket connection stays open after the
  report is generated, supporting continued follow-up questions; conversation history is
  concatenated into the prompt for short-term memory, with a `MAX_HISTORY_TURNS=6` sliding
  window to prevent request cost from growing quadratically with conversation length
- **LoRA fine-tuning + quantitative evaluation**: distilled training data from GPT-4o-mini's
  historical outputs, LoRA fine-tuned (rank=16) on `Qwen2.5-1.5B-Instruct` as a drop-in
  replacement for the Planner's task-decomposition step; evaluated on a 24-example held-out set
  with automated scoring (embedding cosine similarity, JSON validity rate, style-mismatch
  detection) at zero inference-time API cost
- **Real-time progress streaming**: FastAPI WebSocket pushes every pipeline step (decompose /
  search / reflect / generate / verify) to the frontend as it happens, rendered as a live
  timeline

## LoRA Fine-tuning Evaluation Results

On a 24-example held-out validation set, local LoRA model vs. online GPT-4o-mini (teacher):

| Metric | Result |
|---|---|
| JSON format validity rate | 100% |
| Subtask count within valid range | 100% |
| Mean semantic alignment score (embedding cosine, all examples) | 0.833 |
| Mean semantic alignment score (excluding style-mismatch outliers) | 0.854 |
| Mean inference latency | ~11.7s (local RTX 5080 Laptop GPU) |
| Inference-time API cost | $0 (fully local) |

See `backend/scripts/22_eval_benchmark.py` for methodology.

## Tech Stack

**Backend**: Python, FastAPI + WebSocket, Pydantic (structured output validation), OpenAI SDK
(swappable to DashScope/DeepSeek-compatible endpoints), Qdrant (vector store, local persistent
mode), rank-bm25, transformers + peft + trl (LoRA training), ThreadPoolExecutor (concurrent
retrieval)

**Frontend**: React + Vite, native WebSocket, marked (markdown rendering)

## Project Structure

```
backend/
  app/
    agents/       # PlannerAgent, ReflectionAgent, ReportAgent, CitationVerifier, FollowUpAgent
    tools/        # ArxivSearchTool, SemanticScholarTool, RagTool (hybrid retrieval)
    core/         # LLMClient/LocalModelClient (unified ChatClient interface), config
    api/          # WebSocket route
    orchestrator.py   # retrieval-reflection loop orchestration
  scripts/        # 00-24: numbered roughly in build order, from single-concept scratch
                  # scripts to training/evaluation scripts
  data/           # distilled training data, benchmark results
frontend/
  src/App.jsx     # single-page app: kick off research + live progress timeline +
                  # report display + multi-turn follow-up
```

Each script under `scripts/` corresponds to a concept validated at a specific point in the
project's evolution; reading them roughly in order traces the build-up from "a single working
LLM call" to the full multi-agent system.

## Getting Started

**Backend**

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # then set your own OPENAI_API_KEY
uvicorn app.main:app --reload
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

Open the URL the frontend prints and enter a research question.

**Or with Docker**

```bash
cp backend/.env.example backend/.env   # then set your own OPENAI_API_KEY
docker compose up --build
```

Backend runs at `http://localhost:8000`, frontend at `http://localhost:3000`. The vector store
persists in a named Docker volume across container restarts.

**(Optional) Local LoRA fine-tuned model**

```bash
cd backend
pip install -r requirements-train.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128  # match your CUDA version
python -m scripts.18_format_training_data
python -m scripts.19_train_lora
python -m scripts.20_eval_lora
```

Training artifacts (`backend/models/`) are large (a single checkpoint's optimizer state exceeds
GitHub's 100MB per-file limit) and are excluded from version control — regenerate them locally
by running the scripts above.

## Known Limitations

- **RAG hybrid retrieval is not yet wired into the report generation path**: report generation
  and follow-up currently reason only over paper titles + abstracts from the retrieval stage.
  The dense+BM25 hybrid retrieval already implemented and verified in `rag_tool.py` has not been
  connected to `ReportAgent`/`FollowUpAgent` — intentionally, to keep both using the same
  granularity of evidence rather than two inconsistent evidence sources
- **Semantic Scholar's anonymous API is subject to shared rate limiting**: without a registered
  API key, requests share a global anonymous quota and are prone to 429 errors (mitigated with
  exponential backoff retry); a free registered key gives a dedicated quota and is more reliable
- **Multi-turn follow-up history is capped at the most recent 6 turns**: older history is
  dropped outright rather than summarized
- **Short-term memory does not persist across sessions**: conversation history only lives for
  the duration of the current WebSocket connection; it is lost on page refresh or disconnect
