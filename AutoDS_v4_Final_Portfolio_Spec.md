# AutoDS — Portfolio-Optimized Multi-Agent Autonomous Data Scientist

| | |
|---|---|
| **Version** | 5.0 (revised working spec) |
| **Status** | Supersedes the original v4 draft |
| **Context** | Final-year BSc Computer Science capstone · solo build · one–two semesters |
| **Deployment target** | AWS |
| **Model policy** | Open-source models only |

**Delivery priorities, in order:** (1) a working end-to-end demo, (2) methodology an examiner cannot poke holes in, (3) enough genuine complexity to be impressive.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Design Philosophy](#2-design-philosophy)
3. [Key Architectural Decisions](#3-key-architectural-decisions)
4. [Resolved Decisions](#4-resolved-decisions)
5. [Technology Stack](#5-technology-stack)
6. [AI Model Strategy](#6-ai-model-strategy)
7. [Multi-Agent Architecture](#7-multi-agent-architecture)
8. [Methodology — Data-Leakage Prevention](#8-methodology--data-leakage-prevention)
9. [Clustering & Data Mining](#9-clustering--data-mining)
10. [Error Handling & Structured Outputs](#10-error-handling--structured-outputs)
11. [LangGraph Workflow](#11-langgraph-workflow)
12. [Application Layer](#12-application-layer)
13. [Database Design](#13-database-design)
14. [AWS Deployment](#14-aws-deployment)
15. [Deliverables](#15-deliverables)
16. [Build Milestones](#16-build-milestones)
17. [Resume Highlights](#17-resume-highlights)
18. [Success Criteria](#18-success-criteria)

---

## 1. Overview

AutoDS is a portfolio-quality AI application that performs an end-to-end data science workflow from a single CSV upload, using multiple collaborating AI agents orchestrated with LangGraph, deployed on AWS, using open-source models only.

A user uploads a tabular dataset; the system infers its schema, cleans it, explores it, engineers features, trains and evaluates models, explains the best model, critiques its own work, writes a report, and exposes a conversational interface over the results.

**Competencies demonstrated**

- Multi-Agent Systems
- LLM Engineering (open-source models)
- Data Mining
- Machine Learning
- Explainable AI
- RAG Systems
- Full-Stack Development
- MLOps Fundamentals (cloud deployment, cost tracking)

---

## 2. Design Philosophy

The specification balances maximum resume value, strong software-engineering practice, and realistic solo implementation complexity. Four rules govern every design choice:

- **LLM decides, code executes.** LLMs emit small, validated structured decisions (JSON); deterministic Python does the real work.
- **Leakage-safe by construction.** All fitting and resampling happen inside cross-validation folds.
- **Cloud-ready by construction.** No reliance on local disk; configuration and secrets come from the environment.
- **Vertical slice first.** One thin thread runs end-to-end before breadth is added.

---

## 3. Key Architectural Decisions

| # | Decision | Summary |
|---|----------|---------|
| 1 | **Open-source models only** | **Gemma 4** (Apache 2.0) for LLM agents — E4B (small) and 31B (large); open-source sentence-transformers for embeddings. Served via the free Google AI Studio API. |
| 2 | **Single orchestrator** | The standalone Supervisor agent is removed. The Planner writes a plan into shared state; LangGraph conditional edges include/skip optional steps. Routing and retries are graph responsibilities. |
| 3 | **HITL via pipeline split** | Schema detection runs synchronously before the heavy job, so human confirmation happens between two jobs — no mid-run graph freeze/resume in v1. |
| 4 | **AWS hosts the application only** | Models are served externally via the free Google AI Studio API, so **no GPU is needed on AWS**. AWS runs the app on CPU: artifacts in S3, secrets in Secrets Manager, config via env vars, single-EC2 `docker compose` for v1 with a documented ECS/RDS upgrade path. |
| 5 | **Leakage-safe ML** | Unfitted preprocessing pipeline fitted only inside CV; SMOTE inside CV folds via the `imblearn` pipeline. |
| 6 | **Revised clustering** | Planner-selected K-Means / K-Prototypes + PCA 2-D scatter + LLM cluster profiles; dendrogram demoted to an optional illustration. |

---

## 4. Resolved Decisions

All planning decisions are locked. Fallbacks are retained as documented escape hatches if the schedule slips.

| Decision | Choice | Fallback |
|----------|--------|----------|
| HITL mechanism | **Synchronous schema step, then background job** | LangGraph interrupt + checkpointer resume (later upgrade) |
| Async processing | **Celery + Redis** | FastAPI `BackgroundTasks` (drops 2 services) |
| Vector store | **ChromaDB** | `pgvector` in Postgres (one fewer service) |
| LLM serving | **Gemma 4 via the free Google AI Studio API** | Self-host Ollama / vLLM on a GPU |
| Large model | **Gemma 4 31B** | Gemma 4 26B-A4B (MoE) if free-tier rate limits bite |
| Small model | **Gemma 4 E4B** | — |

---

## 5. Technology Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Frontend | Streamlit | Needs sticky sessions behind a load balancer; a non-issue on single EC2 |
| Backend | FastAPI | |
| Orchestration | LangGraph | Conditional edges driven by the Planner |
| Async processing | Celery + Redis | Locked; `BackgroundTasks` is the documented fallback |
| Database | PostgreSQL | Schema managed with Alembic migrations |
| Vector store | ChromaDB | Locked; pgvector is the documented fallback |
| Machine learning | Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM, imbalanced-learn, kmodes | |
| Explainability | SHAP | |
| LLM agents | Gemma 4 (Apache 2.0) — E4B (small), 31B or 26B-A4B (large) | Served via the free Google AI Studio API; self-host (Ollama/vLLM) as fallback |
| Embeddings | sentence-transformers / BGE / E5 | Open source, runs on CPU; keeps the embedding step open too |
| Structured output | Instructor or Outlines + Pydantic | Constrained decoding |
| Storage & config | S3, AWS Secrets Manager, environment variables | 12-factor |
| Deployment | Docker, Docker Compose, EC2 | ECS Fargate + RDS + ElastiCache as future work |

---

## 6. AI Model Strategy

The project uses two kinds of open-source AI models:

- **LLMs** — reasoning/generation models that act as agents.
- **An embedding model** — turns text into vectors for RAG search; runs cheaply on CPU.

### 6.1 Model Assignment

| Agent / Feature | Type | LLM's job (if any) | Model tier |
|-----------------|------|--------------------|-----------|
| Schema Detection | LLM + regex | Infer column meaning, suggest target/task type, semantic PII | Small–mid |
| Planner | LLM | Decide which optional steps run | Small–mid |
| Feature Engineering (strategy) | LLM | Choose encoding/scaling/imputation per column (JSON only) | Small–mid |
| Cluster profiling | LLM | Summarize cluster stats in plain language | Small |
| Critic | LLM | Reasoned critique and recommendations | Mid–large |
| Report | LLM | Write the narrative report | Mid–large |
| Dataset Chat | LLM | Q&A, tool routing (RAG vs pandas), phrasing | Mid–large |
| RAG embeddings | Embedding model | Encode text for semantic search | — |
| Data Cleaning | Code | — | — |
| EDA statistics & charts | Code | — | — |
| Clustering computation | Code | — | — |
| Model Selection | Code | — | — |
| Evaluation | Code | — | — |
| Explainability computation | Code | — | — |
| Final training / `/predict` | Code | — | — |

**Concrete model choices — Gemma 4 (Apache 2.0):**

| Tier | Model | Agents |
|------|-------|--------|
| Small | **Gemma 4 E4B** | Schema, Planner, Feature-Engineering strategy, cluster profiling |
| Large | **Gemma 4 31B** (default, max quality) — or **26B-A4B (MoE)** as fallback | Critic, Report, Chat |

Because serving is free via Google AI Studio, the GPU-cost argument for the MoE no longer applies, so the **31B dense** model is the default large tier for best Critic/Report/Chat quality (85.2 MMLU Pro, 89.2 AIME 2026). Keep **26B-A4B** as the fallback — ~97% of the quality at far lighter compute and likely higher free-tier throughput if rate limits bite. Both share a 256K context and native function-calling (useful for the Chat agent's tool routing).

### 6.2 The "LLM decides, code executes" pattern

Open-source models are less reliable at long outputs, so every LLM's job is kept small: it returns a Pydantic-validated JSON decision, and deterministic code performs the actual work.

```json
{ "age": "standard_scale", "city": "onehot_encode", "income": "median_impute" }
```

Code then constructs the real scikit-learn pipeline from this decision. Invalid output triggers a retry (see [§10](#10-error-handling--structured-outputs)). Use **Instructor** or **Outlines** (constrained/guided decoding) to force valid JSON from open models.

### 6.3 Serving

- **Primary (v1):** the **free Google AI Studio API** serves both Gemma 4 tiers. No GPU and no model hosting — the AWS side stays CPU-only and cheap. The API key is stored in Secrets Manager.
- **Free-tier limits (Gemma 4 31B, per model):** **30 RPM**, **15,000 input TPM**, **14,400 RPD**. RPM/RPD are comfortable (the pipeline is sequential with ML compute between calls); the **binding constraint is 15K input TPM**. Note the trap: the model's 256K context is unusable on the free tier — Critic/Report must send **lean, summarized prompts**, not raw artifacts. The small tasks run on E4B (its own separate limit bucket). *Verify: metering is per-model, and whether a separate output-token limit applies (Report produces long output).*
- **Data-usage policy:** free-tier prompts may be used to improve Google's products — acceptable for demo/example datasets; note it in the writeup and don't send anything sensitive.
- **Fallback (portability / offline demo):** self-host with Ollama or vLLM on a GPU instance — Apache 2.0 makes this unencumbered; both tiers fit a single 24 GB GPU.
- **Embeddings:** open-source sentence-transformers run on CPU, keeping the embedding step open and GPU-free.

---

## 7. Multi-Agent Architecture

### 7.1 Schema Detection Agent

**Type:** LLM + deterministic · **Output:** `schema_report.json`

- Infer column data types (dtype/regex) and semantic meaning (LLM).
- Suggest target variable and task type (LLM).
- Detect PII: regex for emails/phones/SSNs; LLM for semantic cases such as name columns.
- Detect class imbalance (code).
- **Runs synchronously in FastAPI on upload**, enabling the HITL checkpoint without background pausing.

### 7.2 Human-in-the-Loop Checkpoint

The user reviews the suggested target, task type, detected PII, and columns to exclude. Because Schema Detection runs synchronously **before** the heavy job starts, confirmation happens **between two jobs** — no mid-run graph freeze/resume is required. Only after confirmation is the background pipeline (Planner onward) launched.

### 7.3 Planner Agent

**Type:** LLM · **Output:** `planner_report.json`

Writes a plan into shared state deciding which optional steps run (SMOTE, feature selection, sampling) and the recommended model families. The plan is consumed by LangGraph conditional edges.

> The standalone **Supervisor Agent** has been removed. Its routing/retry role is now handled by LangGraph conditional edges plus the retry policy.

### 7.4 Data Cleaning Agent

**Type:** Code · **Output:** `cleaning_report.json`

Missing-value handling, duplicate removal, dtype correction, outlier detection, constant-column removal, PII removal.

### 7.5 EDA Agent

**Type:** Code + LLM (for summaries) · **Output:** `eda_report.json`

- **Standard EDA (code):** summary statistics, correlation, class balance, missing-value analysis.
- **Data-mining EDA (code + LLM):** clustering (see [§9](#9-clustering--data-mining)) with LLM-generated cluster profiles.
- **Visualizations:** histograms, boxplots, correlation heatmaps, target-distribution charts, PCA cluster scatter, and an optional small-sample dendrogram.
- **Academic foundation:** Tan, Steinbach, Karpatne, Kumar — *Introduction to Data Mining*, 2nd ed.

### 7.6 Feature Engineering Agent

**Type:** LLM strategy → code builds pipeline · **Output:** `preprocessing_pipeline.pkl` (unfitted), `feature_report.json`

Must **not** transform the dataset directly. It emits an **unfitted** preprocessing pipeline to prevent leakage; the LLM only chooses strategies (JSON), and code builds the `Pipeline` / `ColumnTransformer`.

- Imputation, encoding, scaling; feature-generation and feature-selection recommendations.
- Ordinal handling: default `OrdinalEncoder`; optional cumulative/thermometer coding, with the chosen encoding recorded.

### 7.7 Model Selection Agent

**Type:** Code · **Output:** `model_candidates.json`

Receives the unfitted pipeline and cleaned dataset.

- **Classification:** Logistic Regression, Random Forest, XGBoost, LightGBM.
- **Regression:** Linear Regression, RF Regressor, XGBoost Regressor, LightGBM Regressor.
- **Cross-validation:** StratifiedKFold (classification) / KFold (regression), 5-fold.
- Pipeline fitting **and SMOTE** occur entirely inside CV folds (see [§8](#8-methodology--data-leakage-prevention)).

### 7.8 Evaluation Agent

**Type:** Code · **Output:** `evaluation_report.json`, `leaderboard.json`

- **Classification:** Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
- **Regression:** MAE, MSE, RMSE, R².

### 7.9 Final Training Agent

**Type:** Code · **Output:** `final_model.pkl` (fitted, stored in S3)

Fits the winning pipeline on the **full dataset** and persists it for serving and explanation. Referenced by `/predict` and the Explainability Agent.

### 7.10 Explainability Agent

**Type:** Code · **Output:** `explainability_report.json`

Runs SHAP on the single best fitted model from Final Training.

- Explainer: `TreeExplainer` for RF/XGB/LightGBM; sample rows if the dataset is large.
- Explains in the **transformed** feature space, with a name-mapping back to original features.
- Generates global feature importance, SHAP summary plot, dependence plots, and local explanations.

### 7.11 Critic Agent

**Type:** LLM · **Output:** `critic_report.json`

Reviews cleaning, feature-engineering, model-selection, evaluation, and explainability outputs. May recommend simpler models, alternative strategies, or additional validation.

### 7.12 Report Agent

**Type:** LLM · **Output:** PDF (WeasyPrint or ReportLab) + Markdown

Produces an executive summary, dataset overview, data-quality report, EDA findings, clustering findings, feature-engineering decisions, model comparison, explainability results, critic feedback, and final recommendations.

### 7.13 Dataset Chat Agent

**Type:** LLM + RAG + tools

RAG over reports, EDA, SHAP, evaluation, and critic text (ChromaDB or pgvector). The agent routes between two tools:

- **RAG retrieval** — for "explain / why" questions (semantic).
- **Pandas query tool** — for "what's the mean / count" questions (arithmetic).

*Example questions:* "Why was transaction amount important?", "Which features drive fraud?", "Explain the model simply.", "What's the average age?" (→ pandas tool).

---

## 8. Methodology — Data-Leakage Prevention

The Feature Engineering Agent outputs an **unfitted** `Pipeline` / `ColumnTransformer`. Fitting happens only inside cross-validation folds (Model Selection Agent), and once more on the full dataset (Final Training Agent).

**SMOTE is applied to training folds only** — use `imblearn.pipeline.Pipeline` (not sklearn's) so the resampler lives inside the fold. Applying SMOTE outside CV silently inflates every score.

This guarantees no train/test contamination, proper cross-validation, and correct evaluation methodology.

---

## 9. Clustering & Data Mining

This replaces the original "hierarchical clustering + dendrogram required" plan, which does not scale past a few hundred rows and mishandles categorical data.

- **Method selected by the Planner from column types:**
  - Numeric-only → **K-Means**, with silhouette score to auto-select `k`.
  - Mixed numeric + categorical → **K-Prototypes** (`kmodes`), purpose-built for mixed data (no fabricated distances).
- **Visualization:** a PCA (or UMAP) 2-D scatter colored by cluster — legible at any dataset size.
- **Cluster profiles:** an LLM writes a plain-language summary per cluster, feeding the Report and RAG chat.
- **Dendrogram:** optional, on a capped numeric sample (~300 rows), clearly labeled as illustrative — preserves the hierarchical-clustering talking point without being load-bearing.
- **Guardrail:** clustering is EDA insight only. Cluster labels are **not** fed back into the predictive model as features (computing them on the full dataset would leak).

---

## 10. Error Handling & Structured Outputs

- Every LLM agent returns a **Pydantic-validated** structured output.
- On invalid or malformed output: retry up to *N* times (constrained decoding first), then fail the job gracefully with a clear status.
- Per-agent status is persisted so the Progress page can reflect it.
- A crashed agent marks the job `failed`; it never hangs silently.
- **Rate limiting:** all LLM calls go through a wrapper enforcing the AI Studio free-tier limits (31B: 30 RPM / 15K input TPM / 14,400 RPD) via a token-bucket limiter with **exponential backoff on HTTP 429**. Prompts are kept compact (summarized JSON, capped feature lists) so Critic/Report stay within the 15K input-TPM cap.

---

## 11. LangGraph Workflow

Optional nodes are included or skipped by conditional edges that read the Planner's plan — the core "dynamic orchestration" novelty.

```text
START
  → Schema Detection            (synchronous, in FastAPI)
  → Human Confirmation          ← job boundary; background pipeline starts after this
  → Planner                     (writes plan to shared state)
  → Cleaning
  → EDA                         (+ clustering, cluster profiles)
  → [conditional] Feature Engineering / SMOTE / Feature Selection / Sampling
  → Model Selection             (fit pipeline inside CV)
  → Evaluation
  → Final Training              (fit best pipeline on full data → final_model.pkl)
  → Explainability              (SHAP on final model)
  → Critic
  → Report
  → Dataset Chat Initialization (embed artifacts)
  → END
```

---

## 12. Application Layer

### 12.1 Streamlit Pages

| Page | Contents |
|------|----------|
| Upload | Upload CSV, view detected schema, confirm task type / target / PII / excluded columns |
| Progress | Current agent, per-agent status, execution timeline |
| Results | EDA charts, cluster scatter, model leaderboard, SHAP visualizations, reports |
| Chat | Dataset Q&A, model explanations, insight exploration |

### 12.2 Backend API

| Method & Path | Purpose |
|---------------|---------|
| `POST /upload` | Store CSV to S3, run synchronous schema detection, return schema |
| `POST /jobs` | After confirmation, launch the background pipeline |
| `GET /jobs/{id}` | Job and per-agent status |
| `GET /jobs/{id}/results` | Results payload |
| `POST /jobs/{id}/chat` | Conversational query |
| `GET /jobs/{id}/report` | Retrieve the report |
| `POST /jobs/{id}/predict` | Predict using `final_model.pkl` from S3 |
| `GET /health` | Load-balancer health check |

---

## 13. Database Design

**Tables:** `users`, `jobs`, `reports`, `artifacts`, `chat_history`, `token_usage`.

- **Auth:** single-user stub for v1 (hardcoded dev user, e.g. `user_id = 1`); real auth is out of scope and stated as such in the writeup.
- **`artifacts`:** stores **S3 keys** (e.g., `jobs/42/final_model.pkl`), not files. Streamlit reads them via short-lived presigned URLs.
- **`token_usage`:** populated by an LLM callback/wrapper logging tokens and cost per agent per job — the "cost-aware AI" talking point.
- **Migrations:** schema managed with Alembic (not create-on-startup).

---

## 14. AWS Deployment

**Required (AWS is ephemeral and multi-container):**

1. **Artifacts → S3.** No agent writes to local disk; all `.pkl` files, plots, and reports go to S3; the DB stores keys.
2. **Uploaded CSVs → S3.** Celery reads them back from S3.
3. **Secrets → AWS Secrets Manager / SSM.** LLM keys and DB passwords are never committed or baked into images.
4. **Config via env vars.** DB URL, Redis URL, S3 bucket, model endpoint — the same image runs locally and on AWS.
5. **Alembic migrations** for schema changes on a real database.
6. **`/health` endpoint and graceful failure** for load-balancer checks.

**Deployment shape (recommended for a solo student):**

- **v1:** a single EC2 instance running `docker compose up`, plus S3 and Secrets Manager. Compose networking "just works"; cheapest and simplest.
- **Future work (resume talking point):** ECS Fargate + RDS (managed Postgres) + ElastiCache (managed Redis).
- **No GPU on AWS:** models are served by the external Google AI Studio API, so AWS runs CPU-only instances — dramatically cheaper and simpler.
- **Cost control:** small CPU instances (t3.medium+; SHAP/XGBoost are memory-hungry), a billing alarm, and stopping the instance when not demoing.
- **API key:** the Google AI Studio key lives in Secrets Manager alongside the DB credentials.
- **Vector store on AWS:** ChromaDB needs a persistent EBS volume; pgvector avoids a service and is easier to deploy (the documented fallback).
- **Streamlit:** needs sticky sessions behind a load balancer (a non-issue on single EC2).

---

## 15. Deliverables

**Software** — Streamlit frontend, FastAPI backend, LangGraph workflow, agent implementations, PostgreSQL schema (Alembic), vector-store integration, S3 + Secrets Manager integration, Docker Compose setup, README, example datasets.

**Documentation** — architecture diagram, agent-interaction diagram, database ER diagram, API documentation, setup guide, AWS deployment guide.

**Testing**

- High coverage on the deterministic core (cleaning, pipeline construction, CV, metrics, SHAP wiring).
- LLM agents tested with mocked responses (outputs are non-deterministic).
- Integration tests for the end-to-end thread.
- No blanket 80%-of-all-code target; coverage is concentrated where it is meaningful.

---

## 16. Build Milestones

| Milestone | Scope |
|-----------|-------|
| **M1 — Vertical slice** *(highest priority)* | `upload → synchronous schema → confirm → clean → train one model (in CV) → evaluate → simple report`, through the real stack (FastAPI + Celery + Streamlit + Postgres + S3). De-risks the hardest integration first. |
| **M2 — EDA + clustering** | K-Means / K-Prototypes + PCA scatter + cluster profiles |
| **M3 — Feature Engineering** | LLM strategy → pipeline, SMOTE-in-CV, full model roster |
| **M4 — Explainability** | Final Training + SHAP |
| **M5 — Critic + Report** | LLM agents with structured outputs |
| **M6 — RAG Chat** | Embeddings + retrieval + pandas tool |
| **M7 — AWS hardening** | S3, Secrets Manager, Alembic, health checks, billing alarm |

---

## 17. Resume Highlights

- Multi-agent AI system using LangGraph with dynamic, Planner-driven orchestration.
- Open-source **Gemma 4** LLMs (Apache 2.0) with structured-output engineering (constrained decoding + Pydantic).
- Human-in-the-loop workflow.
- Data-mining techniques (K-Means / K-Prototypes clustering, cluster profiling).
- Data-leakage-safe ML pipelines (fitting and SMOTE inside cross-validation).
- Explainable AI using SHAP.
- RAG dataset chat with tool routing (semantic + structured queries).
- FastAPI backend, Streamlit frontend, Celery + Redis async processing.
- PostgreSQL persistence with cost/token tracking.
- Dockerized and deployed on AWS (S3 artifacts, Secrets Manager, EC2).

---

## 18. Success Criteria

The application should:

- Accept any tabular dataset.
- Automatically perform analysis and build and evaluate ML models (leakage-safe).
- Explain model behavior with SHAP.
- Generate reports (PDF + Markdown).
- Support conversational querying (RAG + pandas tool).
- Demonstrate real, dynamic multi-agent orchestration.
- Run via `docker compose up` locally and be deployable on AWS.
