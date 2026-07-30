# Architecture

Every diagram here was generated from the running system rather than drawn from
the spec — the pipeline graph from `PIPELINE_NODES` and `OPTIONAL_NODES`, the
tables from `information_schema`, the endpoints from the live OpenAPI document.
Where the spec and the code disagree, this describes the code.

- [The containers](#the-containers)
- [The request path](#the-request-path)
- [The pipeline graph](#the-pipeline-graph)
- [Who calls the model, and when](#who-calls-the-model-and-when)
- [Where the leakage guarantee lives](#where-the-leakage-guarantee-lives)

Related: [data model](data-model.md) · [API reference](api.md) · [runbook](RUNBOOK.md)

---

## The containers

Six services, all started by one `docker compose up`.

```mermaid
flowchart TD
  user([Browser]) --> ui["<b>ui</b> · Streamlit 8501"]
  ui -->|HTTP only| api["<b>api</b> · FastAPI 8000"]
  api -->|enqueue| redis[("<b>redis</b> · 6379")]
  redis -->|consume| worker["<b>worker</b> · Celery + LangGraph"]

  api --> pg[("<b>postgres</b> · pgvector 5432")]
  api --> minio[("<b>minio</b> · S3 API 9000")]
  worker --> pg
  worker --> minio

  api -. "1 call · schema detection" .-> gemini{{"Google AI Studio"}}
  worker -. "5 calls per run" .-> gemini

  classDef store fill:#E3EDE8,stroke:#1D6F5C,color:#123f34;
  classDef ext fill:#F3EAD3,stroke:#8C6714,color:#5c4409;
  class pg,minio,redis store
  class gemini ext
```

The UI never touches the database or object storage: it is an HTTP client of the
API and nothing else. That is what makes the API the only thing that has to be
deployed for a programmatic user, and it is why the Streamlit layer can be
replaced without touching anything below it.

**Storage is an abstraction, not MinIO.** `app/core/storage.py` speaks the S3 API,
and `S3_ENDPOINT_URL` is the only thing pointing it at the local container. Unset
that variable on AWS and the same code talks to real S3 through the instance's IAM
role — which is what "cloud-ready by construction" actually cashes out to.

**The vector store is the database.** The spec locks ChromaDB and documents
pgvector as the sanctioned fallback; this takes the fallback, so embeddings live
in `run_chunks` alongside everything else. One less container, and on AWS one less
persistent volume.

---

## The request path

The human checkpoint is the load-bearing part of this diagram. Schema detection
runs **synchronously inside the upload request**, so the user confirms between two
jobs rather than mid-run — no workflow pause-and-resume machinery is needed, and
no running graph is ever suspended.

```mermaid
sequenceDiagram
  autonumber
  actor U as User
  participant UI as Streamlit
  participant API as FastAPI
  participant S3 as Object storage
  participant DB as Postgres
  participant Q as Redis
  participant W as Celery worker

  U->>UI: choose a CSV
  UI->>API: POST /upload
  API->>API: validate, then parse
  API->>DB: insert job (UPLOADED)
  API->>S3: store the raw file
  API->>API: schema detection (≈2s, live LLM)
  API->>DB: store schema report
  API-->>UI: job id + detected schema

  Note over U,UI: the human checkpoint —<br/>target, task type, PII exclusions

  U->>UI: confirm
  UI->>API: POST /jobs
  API->>DB: store confirmed schema, status QUEUED
  API->>Q: enqueue
  API-->>UI: 200

  Q->>W: deliver
  loop each graph node
    W->>DB: mark running / completed / skipped
    W->>S3: write artifacts
  end
  W->>DB: status COMPLETED

  loop while running
    UI->>API: GET /jobs/{id}
    API-->>UI: per-node status
  end
```

Confirming *is* launching: `POST /jobs` writes `QUEUED` and dispatches in the same
request. The status is committed before the dispatch, so a worker that picks the
job up instantly cannot have its `RUNNING` clobbered back to `QUEUED` by the tail
of the request that created it.

---

## The pipeline graph

Fourteen nodes. Twelve always run; two are conditional, and a node routed past is
recorded as **skipped with a reason** rather than left looking unreached.

```mermaid
flowchart TD
  START([START]) --> planner
  planner --> cleaning --> eda

  eda -->|plan.run_sampling| sampling
  eda -.->|otherwise| feature_strategy
  sampling --> feature_strategy

  feature_strategy -->|plan.run_feature_selection| feature_selection
  feature_strategy -.->|otherwise| preprocessing
  feature_selection --> preprocessing

  preprocessing --> modeling --> evaluation --> final_training
  final_training --> explainability --> critic --> report --> chat_index --> END([END])

  classDef opt fill:#F3EAD3,stroke:#8C6714,stroke-width:1.5px,color:#5c4409;
  class sampling,feature_selection opt
```

The graph is **built by walking `PIPELINE_NODES`**, not by wiring edges by hand.
That list is also what the Progress page renders, so the roadmap the user watches
and the graph the worker executes cannot drift apart — they are the same object.
The builder raises if two optional nodes are ever made adjacent, because the
router can only skip one at a time and the failure would otherwise be a silently
unreachable node.

| Node | Always runs | What it produces |
|---|---|---|
| `planner` | ✅ | The plan the conditional edges read |
| `cleaning` | ✅ | Duplicate and unusable-row removal, dropped-column report |
| `eda` | ✅ | Statistics, six charts, clustering with LLM group profiles |
| `sampling` | conditional | A stratified subset, when the planner asks for one |
| `feature_strategy` | ✅ | The LLM's per-column preparation decisions |
| `feature_selection` | conditional | *How many* features to keep — the selector itself is a pipeline step |
| `preprocessing` | ✅ | An **unfitted** scikit-learn recipe |
| `modeling` | ✅ | Four candidates scored on identical folds |
| `evaluation` | ✅ | Cross-validated metrics and per-fold detail |
| `final_training` | ✅ | The one full-data fit, which is the served model |
| `explainability` | ✅ | SHAP in the user's own column names |
| `critic` | ✅ | Measured checks first, then the model's commentary |
| `report` | ✅ | Markdown, and a best-effort PDF |
| `chat_index` | ✅ | Passages embedded into `run_chunks` |

**Why `eda` sits where it does.** After cleaning, so the charts describe the data
the model actually saw; before preprocessing, because it is purely descriptive.
Nothing downstream reads its output, which is exactly what lets it fail without
taking the model with it.

**Why `sampling` sits after `eda`.** So the charts always describe every uploaded
row and only the modelling sees a subset.

**Why `feature_strategy` and `preprocessing` are two nodes.** One decides, the
other builds. Splitting them means the LLM's choice lands as its own step on the
Progress page, and means a strategy artifact exists to read even if building the
recipe from it later fails.

---

## Who calls the model, and when

Seven agents exist; **six calls happen in a normal run**, and the seventh is the
chat, which runs on demand afterwards. Measured identically across all three live
runs.

```mermaid
flowchart LR
  subgraph req["In the upload request"]
    sd["schema_detection<br/><i>small tier</i>"]
  end

  subgraph run["In the worker"]
    pl["planner<br/><i>small</i>"]
    cp["cluster_profiles<br/><i>small</i>"]
    fs["feature_strategy<br/><i>small</i>"]
    cr["critic<br/><i>large</i>"]
    rw["report_writer<br/><i>large</i>"]
  end

  subgraph after["On demand"]
    ch["dataset_chat<br/><i>small</i>"]
  end

  sd --> pl --> cp --> fs --> cr --> rw -.-> ch

  classDef big fill:#E3EDE8,stroke:#1D6F5C,color:#123f34;
  class cr,rw big
```

**Every one of them degrades to a deterministic fallback.** With no API key the
pipeline still completes: detection profiles the file without semantic meanings,
the planner takes defaults, feature strategy uses dtype rules, and the critic
still reports everything it measured rather than judged. This is why the test
suite runs offline and why a rate limit does not fail a run.

**The two large-tier agents are ~75% of the wall clock.** On the churn dataset,
21.0s and 27.6s of 63.5s. Everything deterministic is sub-second. The one
exception is EDA, whose cost scales with the data rather than the prose — 113s of
a 250s run at 100,000 rows.

**Rate limits step down rather than fail.** Hitting the free tier's large-tier
limit mid-run logs `Rate limited on the large tier; stepping down` and completes
on the small model, with shorter and plainer prose. Verified under a full
pipeline, not only in a unit test.

---

## Where the leakage guarantee lives

This is the claim the project is built on, so it is worth saying exactly which
code enforces it.

```mermaid
flowchart TD
  fs["feature_strategy<br/>decides"] --> pp["preprocessing<br/>builds an <b>unfitted</b> recipe"]
  pp --> cv{"cross_validate_model"}
  cv -->|"clone() per fold"| f1["fold 1<br/>fit on train only"]
  cv -->|"clone() per fold"| f2["fold 2 … k"]
  f1 --> sc["scores from held-out rows"]
  f2 --> sc
  pp -.->|"never fitted here"| x(["the caller's object<br/>stays unfitted"])

  classDef danger fill:#F6E4DE,stroke:#9B3B22,color:#5e2415;
  class x danger
```

`build_preprocessor` returns something that has never seen data. `cross_validate_model`
clones it per fold and fits each clone on training rows only. SMOTE is a step
inside an `imblearn` pipeline, so it is applied during `fit` and skipped during
`predict` — each fold trains balanced and is scored on the real skew.

`final_training` is the one place a full-dataset fit is correct, because that
model is the one served. No score is recomputed there: the figure in the artifact
is the cross-validated one, named `cv_score`, since a model fitted on every row
has no unseen data left to be measured against.

**The assertions were checked by breaking them** (30 July 2026). Three leaks were
introduced into `cross_validate_model` and reverted; all three were caught. The
useful result: dropping the `clone` is *not* caught by the row-counting spy —
every fold still refits on 80 training rows, so the log looks perfect — and is
caught only by the two other proofs. The overlap between them is load-bearing.
See the docstring in `tests/test_leakage.py`.
