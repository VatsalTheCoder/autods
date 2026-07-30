# API reference

Twenty-five endpoints, listed from the live OpenAPI document. FastAPI serves an
interactive version at **http://localhost:8000/docs** — that is the authority and
it is generated from the code; this page is the map of it, with the things the
schema cannot tell you.

Related: [architecture](architecture.md) · [data model](data-model.md) · [runbook](RUNBOOK.md)

---

## The shape of a run

Four calls get you from a CSV to a trained model:

```bash
# 1. Upload. Schema detection runs inside this request, so it takes ~2s.
curl -F "file=@data/examples/customer_churn.csv" http://localhost:8000/upload

# 2. Confirm — this is the human checkpoint, and confirming launches the run.
curl -X POST http://localhost:8000/jobs \
  -H 'Content-Type: application/json' \
  -d '{"job_id": 42, "target_column": "churned", "task_type": "classification"}'

# 3. Poll until status is completed (60-90s for a small dataset).
curl http://localhost:8000/jobs/42

# 4. Read the results.
curl http://localhost:8000/jobs/42/evaluation
```

**Omitting `columns` at step 2 is safe.** It means "no opinion", not "no
exclusions" — the PII columns detection flagged stay excluded. To model a flagged
column deliberately, send it explicitly with `exclude: false`.

---

## Lifecycle

| Method | Path | Notes |
|---|---|---|
| `POST` | `/upload` | Multipart CSV. Validates, stores, and runs schema detection synchronously. Returns the job id, a preview and the detected schema. |
| `POST` | `/jobs` | Confirm the schema. **Launches the pipeline.** Unknown fields are a `422`, deliberately. |
| `GET` | `/jobs` | Newest first. `?limit=` defaults to 50. |
| `GET` | `/jobs/{id}` | The job plus its per-node status — one poll gives the Progress page everything. |
| `GET` | `/jobs/{id}/schema` | The detected schema, for a returning user to confirm. |

## Results

Each returns the artifact of the node named, or `404` if that node has not run.

| Method | Path | Notes |
|---|---|---|
| `GET` | `/jobs/{id}/cleaning` | Rows removed, columns dropped and **why** each was dropped. |
| `GET` | `/jobs/{id}/eda` | Descriptive statistics and the chart list. |
| `GET` | `/jobs/{id}/clustering` | Groups found, and how well separated. |
| `GET` | `/jobs/{id}/features` | The per-column decisions, and anything the code overruled. |
| `GET` | `/jobs/{id}/leaderboard` | Every candidate, ranked on identical folds. |
| `GET` | `/jobs/{id}/evaluation` | Cross-validated metrics with per-fold detail. |
| `GET` | `/jobs/{id}/explainability` | SHAP in the user's own column names, with the additivity error. |
| `GET` | `/jobs/{id}/critic` | Findings worst-first, each flagged measured or judged. |
| `GET` | `/jobs/{id}/report` | Markdown. The authoritative document. |
| `GET` | `/jobs/{id}/report/pdf` | Best-effort. See the caveat below. |
| `GET` | `/jobs/{id}/model` | What is served, and which columns it expects. |

## Prediction

| Method | Path | Notes |
|---|---|---|
| `POST` | `/jobs/{id}/predict` | `{"rows": [{...}]}`. Returns a prediction per row, plus `probabilities` for classification and `{}` for regression. |

Send the columns `/jobs/{id}/model` reports — the recipe's inputs, not the raw
file's. Excluded and dropped columns are not expected.

## Chat

| Method | Path | Notes |
|---|---|---|
| `GET` | `/jobs/{id}/chat/status` | `ready` and `indexed_passages`. Check before offering an input box. |
| `POST` | `/jobs/{id}/chat` | `{"question": "..."}` — note **`question`**, not `message`. |
| `GET` | `/jobs/{id}/chat` | The conversation so far. |

Every answer carries a `route`: `rag` when it came from a retrieved passage,
`pandas` when it was computed from the data. Attribution is the point — a
retrieved sentence and a computed number are different kinds of claim.

## Artifacts and health

| Method | Path | Notes |
|---|---|---|
| `GET` | `/jobs/{id}/artifacts` | Registry listing. Does not touch object storage. |
| `GET` | `/jobs/{id}/artifacts/{name}/content` | Streams the bytes with the recorded content type. |
| `GET` | `/jobs/{id}/artifacts/{name}/link` | A temporary presigned URL instead. |
| `GET` | `/health` | Liveness plus dependencies. |
| `GET` | `/` | Root. |

---

## Things the OpenAPI schema will not tell you

**`/health` reports `database: true` against an empty database.** It checks
connectivity, not schema. A stack that has never been migrated looks healthy and
fails on the first upload.

**Redis is not in the health response**, because the API does not talk to it. A
dead Redis presents as jobs sitting in `queued` forever, not as an unhealthy API.

**`/report/pdf` returns `404` for two different situations** — the run has not
reached the report yet, *and* rendering failed on a run that otherwise completed.
Only the detail string distinguishes them. PDF rendering is best-effort by design
so that a font problem cannot discard a finished analysis, and the Markdown at
`/report` is authoritative either way. A client cannot currently tell "not yet"
from "never"; this is a known gap.

**Unknown fields on `POST /jobs` are rejected rather than ignored.** `exclude` is
easy to guess wrong — `include` is the obvious alternative and it is *inverted* —
and Pydantic's default of ignoring unknown keys turned that typo into a `200` and
a model trained on the column the caller meant to withhold.

**A 5xx from an agent does not fail the run.** Transient provider failures are
retried within a time budget, and a rate limit steps the model down a tier rather
than losing the work. With no API key at all every agent takes a deterministic
fallback and the run still completes.

**Upload is capped at 200 MB** (`MAX_UPLOAD_MB`), and the Streamlit UI enforces
the same number before the API sees the file. Larger datasets need subsampling
first — the request parses the whole file in memory, twice, so raising the cap
alone will not work.
