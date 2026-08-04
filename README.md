# AutoDS — Multi-Agent Autonomous Data Scientist

Upload a CSV. Get back cleaned data, exploratory analysis, clustering, a trained and
cross-validated model, SHAP explanations of its behaviour, a written report, and a chat interface
to ask questions about all of it.

A dozen specialised agents — cleaning, EDA, planning, feature engineering, modelling, critique,
reporting — coordinated by LangGraph. The language model makes decisions; ordinary Python carries
them out.

> **Status: 11 of 13 sections merged.** Milestones M1–M6 delivered. The pipeline runs end to end
> against a live model — a 500-row classification job completes in about 60 seconds and a
> 100,000-row one in about four minutes. What remains is deployment (Section 11 — the AWS
> guide, production compose file and scripts are written and the code changes are done and
> tested, but nothing has been run against a real AWS account) and the concentrated docs and
> testing pass (Section 12).
>
> See [the progress report](https://claude.ai/code/artifact/2fb6fa9c-b8bc-448e-b653-3145e3c02794)
> for measured figures and what running it on real data turned up.

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
| Vector store | pgvector, in the Postgres already running |
| ML | scikit-learn, XGBoost, LightGBM, imbalanced-learn, kmodes |
| Explainability | SHAP |
| LLMs | Gemini via Google AI Studio, structured output via Pydantic |
| Infrastructure | Docker Compose, S3, AWS Secrets Manager, EC2 |

Two of those differ from the spec, deliberately, and both are one setting away from being changed
back:

**pgvector rather than ChromaDB.** The spec locks ChromaDB and documents pgvector as the sanctioned
fallback. Taking the fallback removes a container and, on AWS, removes the persistent EBS volume
that container would have needed — the embeddings live in the database that is already running.

**Gemini rather than Gemma.** The spec's `gemma-3-*` model ids no longer exist (404), and the
smallest Gemma now served is a 26B MoE. Measured on the feature-strategy prompt:
`gemini-3.1-flash-lite` answered in 1.6s and succeeded 3 times out of 3; `gemma-4-26b-a4b-it` took
42s and failed 4 times out of 5 against the 60-second timeout, silently degrading every agent to
its deterministic fallback. Model ids are environment variables precisely so a self-hosted or Gemma
deployment needs no code change — which is the property the spec was protecting.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/](docs/README.md) | **Start here** — architecture, data model, API reference, runbook |
| [docs/architecture.md](docs/architecture.md) | Containers, request path, the pipeline graph, which agents call the model, where leakage is prevented |
| [docs/data-model.md](docs/data-model.md) | The seven tables and what each is for |
| [docs/api.md](docs/api.md) | All 25 endpoints, plus what the OpenAPI schema cannot tell you |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Bringing the stack up from cold, measured demo timings, troubleshooting |
| [docs/related-work.md](docs/related-work.md) | Positioning against the similarly-named CHI '21 system by Wang et al. |
| [AutoDS_v4_Final_Portfolio_Spec.md](AutoDS_v4_Final_Portfolio_Spec.md) | The original specification — see the two documented departures above |
| [BUILD_PLAN.md](BUILD_PLAN.md) | 13 sections across ~24 weeks, with exit criteria for each |

---

## Build progress

- [x] **Section 0** — Skeleton: Docker Compose, config, S3 abstraction, health check
- [x] **Section 1** — Upload: CSV to S3, jobs and artifacts tables
- [x] **Section 2** — LLM client: structured output, rate limiting, token accounting
- [x] **Section 3** — Schema detection and the human checkpoint
- [x] **Section 4** — Background worker: Celery, LangGraph, progress tracking
- [x] **Section 5** — Vertical slice: end-to-end demo *(milestone M1)*
- [x] **Section 6** — EDA and clustering *(milestone M2)*
- [x] **Section 7** — Feature engineering *(milestone M3)*
- [x] **Section 8** — Final training, SHAP, prediction *(milestone M4)*
- [x] **Section 9** — Critic and report *(milestone M5)*
- [x] **Section 10** — RAG chat *(milestone M6)*
- [ ] **Section 11** — AWS deployment *(milestone M7)* — code, compose file, scripts and guide done; not yet run on real AWS
- [ ] **Section 12** — Testing and documentation — end-to-end tests run in CI and the diagrams are written; consolidation pass outstanding

This list is maintained by hand in three places — here, the app's landing page, and the progress
report — so if they ever disagree, the repository is the one to trust.

---

## Running it

Requires Docker. On macOS, either Docker Desktop or Colima works:

```bash
brew install colima docker docker-compose
colima start --cpu 4 --memory 4 --disk 40
```

Then bring up the stack:

```bash
cp .env.example .env
$EDITOR .env     # set GOOGLE_API_KEY -- see below, this one matters

make up          # starts postgres, redis, minio, api, worker, ui
make migrate     # NOT automatic; the first upload fails without it
make health      # confirm everything is reachable
```

**`make migrate` is a separate step and nothing reminds you.** Postgres starts empty and the health
check reports `database: true` against it, because it checks connectivity rather than schema. So
the stack looks fine and the *first upload* is what fails.

**The key is needed even under Docker.** Compose reads `.env` to interpolate `GOOGLE_API_KEY` but
does not otherwise inject the file, so without it the stack still starts and every agent quietly
takes its deterministic fallback — which looks exactly like a working system producing worse
output. The confirmation screen after an upload shows `llm_enriched: true` when the model really
ran.

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

Configuration comes entirely from environment variables. `docker compose` supplies most of them
directly; `.env` is where the API key and any per-machine overrides live, and it is gitignored.
[docs/RUNBOOK.md](docs/RUNBOOK.md) covers the cold start in more detail, including the failure
modes that look like bugs and are not.

There are two example datasets, one for each task type:

| File | Rows | Target | What it exercises |
|---|---|---|---|
| `data/examples/customer_churn.csv` | 500 | `churned` (binary) | Mixed types, missing values, PII-shaped columns, class imbalance |
| `data/examples/house_prices.csv` | 606 | `sale_price` (continuous) | An identifier to drop, an ordinal grade, a date, 8-level categorical, ~12% missing, exact duplicates |

Both are synthetic. The house-prices generator is committed at
`scripts/make_house_prices.py` rather than only its output, so the generative
relationship is inspectable — you can see that the signal is real, the noise deliberate,
and nothing cherry-picked to flatter a model. Regenerate with
`docker compose run --rm api python scripts/make_house_prices.py`.

> **Do not keep this project in an iCloud-synced folder** (Desktop or Documents, if you
> have iCloud Drive enabled). Containers cannot read files through the macOS file
> provider and fail with `OSError: [Errno 35] Resource deadlock avoided`.

### Sweeping a set of datasets

Every dataset shape takes a different route through the pipeline — a continuous target skips
resampling, a numeric-only frame clusters with K-Means rather than K-Prototypes, a wide frame is
where the report agent's prompt budget gives out. Each route has tests, but nothing ran *a set* of
shapes back to back until now, so a change that breaks one shape while the demo dataset keeps
passing had nothing to trip over.

```bash
make sweep                                   # everything in data/examples
make sweep paths=data/wide_telemetry.csv     # or specific files and directories
```

It drives each CSV through upload → confirm → pipeline and writes one row per dataset to
`sweep_results/`: status, wall time, which optional steps the planner skipped, **which agents fell
back to their deterministic path**, the winning model and its score. That fallback column is the
one worth watching — a run can complete, render a report, and be entirely the fallback output, and
nothing else in the system says so.

```
| Dataset             | Rows × Cols | Task           | Status    | Time | Best model         | Score | Skipped                     | Fell back |
| customer_churn.csv  | 500 × 10    | classification | completed | 72s  | LogisticRegression | 0.782 | sampling, feature_selection | none      |
| house_prices.csv    | 606 × 13    | regression     | completed | 104s | LinearRegression   | 0.908 | sampling, feature_selection | none      |
| wide_telemetry.csv  | 500 × 122   | classification | completed | 78s  | LogisticRegression | 0.711 | sampling                    | none      |
```

A sweep reports; it does not pass or fail, and it is deliberately not part of `make check` — it
calls a live model and takes minutes per dataset. Wall times move by tens of seconds between runs
on free-tier latency alone; the scores do not. Per-dataset targets and exclusions go in
`scripts/sweep_manifest.json`, which also lists the dataset shapes not yet covered.

---

## Context

Final-year BSc Computer Science capstone project. Solo build.
