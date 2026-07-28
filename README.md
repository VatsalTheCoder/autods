# AutoDS — Multi-Agent Autonomous Data Scientist

Upload a CSV. Get back cleaned data, exploratory analysis, clustering, a trained and
cross-validated model, SHAP explanations of its behaviour, a written report, and a chat interface
to ask questions about all of it.

A dozen specialised agents — cleaning, EDA, planning, feature engineering, modelling, critique,
reporting — coordinated by LangGraph, using open-source LLMs only.

> **Status: planning complete, implementation starting.**
> The architecture and build plan are settled; code begins at Section 0 of the build plan.
> This README describes the intended system and will be updated as sections land.

---

## Why it's built this way

**LLMs decide, code executes.** No LLM ever transforms your data. Instead it emits a small,
schema-validated JSON decision — *"scale `age`, one-hot `city`, median-impute `income`"* — and
ordinary Python builds the real scikit-learn pipeline from it. A JSON dictionary can be validated
against the actual column list; generated pandas code cannot.

**Leakage-safe by construction.** Feature engineering hands over an *unfitted* pipeline. It is
only ever fitted inside cross-validation folds, and SMOTE runs inside the folds too (via
`imblearn`'s pipeline, not scikit-learn's). Fitting preprocessing before splitting silently
inflates every metric you report — nothing crashes, the scores are just wrong. This is the
methodology claim the project is built on.

**Human in the loop, without freezing the graph.** Schema detection runs synchronously during
upload, so the user confirms the target column, task type, and PII flags *between two jobs*
rather than mid-run. No workflow pause-and-resume machinery needed.

**Dynamic orchestration.** A Planner agent writes a plan into shared state; LangGraph conditional
edges then include or skip optional steps. Different datasets genuinely take different routes
through the graph.

---

## Architecture

```
upload → schema detection → [human confirms] → planner → cleaning → EDA + clustering
      → [conditional] feature engineering / SMOTE / feature selection
      → model selection (fit inside CV) → evaluation → final training
      → SHAP explainability → critic → report → chat initialisation
```

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI |
| Orchestration | LangGraph |
| Async | Celery + Redis |
| Database | PostgreSQL + Alembic |
| Vector store | ChromaDB |
| ML | scikit-learn, XGBoost, LightGBM, imbalanced-learn, kmodes |
| Explainability | SHAP |
| LLMs | Open-source models, structured output via Pydantic |
| Infrastructure | Docker Compose, S3, AWS Secrets Manager, EC2 |

---

## Documentation

| Document | Contents |
|---|---|
| [AutoDS_v4_Final_Portfolio_Spec.md](AutoDS_v4_Final_Portfolio_Spec.md) | Full architecture, agent designs, and design decisions |
| [BUILD_PLAN.md](BUILD_PLAN.md) | 13 sections across ~24 weeks, with exit criteria for each |

---

## Build progress

- [x] **Section 0** — Skeleton: Docker Compose, config, S3 abstraction, health check
- [x] **Section 1** — Upload: CSV to S3, jobs and artifacts tables
- [x] **Section 2** — LLM client: structured output, rate limiting, token accounting
- [x] **Section 3** — Schema detection and the human checkpoint
- [x] **Section 4** — Background worker: Celery, LangGraph, progress tracking
- [x] **Section 5** — Vertical slice: end-to-end demo *(milestone M1)*
- [x] **Section 6** — EDA and clustering
- [x] **Section 7** — Feature engineering
- [x] **Section 8** — Final training, SHAP, prediction *(milestone M4)*
- [x] **Section 9** — Critic and report *(milestone M5)*
- [x] **Section 10** — RAG chat *(milestone M6)*
- [ ] **Section 11** — AWS deployment
- [ ] **Section 12** — Testing and documentation

---

## Running it

Requires Docker. On macOS, either Docker Desktop or Colima works:

```bash
brew install colima docker docker-compose
colima start --cpu 4 --memory 4 --disk 40
```

Then bring up the stack:

```bash
make up          # starts postgres, redis, minio, api, worker, ui
make health      # confirm everything is reachable
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| API docs | http://localhost:8000/docs |
| MinIO console | http://localhost:9001 (`minioadmin` / `minioadmin`) |

Common commands — run `make` on its own for the full list:

```bash
make test        # run the test suite
make lint        # the same checks CI runs
make logs        # follow logs from all services
make down        # stop (data is preserved)
make clean       # stop and DELETE all data volumes
```

Configuration comes entirely from environment variables. `docker compose` supplies
them; copy `.env.example` to `.env` only if you run something directly on your host.

There is an example dataset at `data/examples/customer_churn.csv` — 500 rows with mixed
types, missing values, PII-shaped columns and a binary target.

> **Do not keep this project in an iCloud-synced folder** (Desktop or Documents, if you
> have iCloud Drive enabled). Containers cannot read files through the macOS file
> provider and fail with `OSError: [Errno 35] Resource deadlock avoided`.

---

## Context

Final-year BSc Computer Science capstone project. Solo build.
