# Runbook — bringing AutoDS up from cold

Written by rehearsing it, not from memory. Every timing here is measured from a
real run against live Gemini on 30 July 2026 (job ids 2938 and 2939), on an
Apple Silicon Mac with Docker Desktop.

If you are demoing this, read [Before a demo](#before-a-demo) as well — there
are two failure modes that look like bugs and are not.

---

## Prerequisites

- Docker Desktop, running. Six services come up; SHAP and XGBoost are the
  memory-hungry parts, which is why the build plan specifies t3.medium as the
  EC2 floor. The runs below were done with Docker Desktop's default allocation
  and did not come close to failing, but I have not probed where the floor
  actually is.
- A Google AI Studio API key. The free tier is sufficient — see
  [Rate limits](#rate-limits-what-actually-happens) for what that costs you.
- **Do not clone into an iCloud-synced folder** (Desktop or Documents with
  iCloud Drive on). Containers cannot read through the macOS file provider and
  fail with `OSError: [Errno 35] Resource deadlock avoided`. Use `~/projects`
  or similar.

## Cold start

```bash
git clone git@github.com:VatsalTheCoder/autods.git
cd autods

# The key. Compose loads .env for interpolation but does not inject it into
# containers -- docker-compose.yml passes GOOGLE_API_KEY through explicitly.
# Without this the stack still starts and every agent silently takes its
# deterministic fallback, which looks exactly like the key not working.
cp .env.example .env
$EDITOR .env          # set GOOGLE_API_KEY=...

make up               # first run pulls and builds images; allow several minutes
make migrate          # NOT automatic -- see below
make health           # expect "healthy" with database and storage both true
```

`make health` returns:

```json
{"status": "healthy", "environment": "local", "version": "0.1.0",
 "dependencies": {"database": true, "storage": true}}
```

Note that Redis is not in that list — the API does not talk to it. A dead Redis
shows up as jobs sitting in `queued` forever, not as an unhealthy API.

**`make migrate` is a separate step and nothing reminds you.** Postgres starts
empty, and the health check reports `database: true` on an empty database — it
checks connectivity, not schema. So the stack looks fine and the *first upload*
is what fails, with a missing-table error in the API logs. Run `make migrate`
after every `git pull` that adds a migration too; there are 5 revisions as of
this writing.

Then open:

| | |
|---|---|
| UI | http://localhost:8501 |
| API docs | http://localhost:8000/docs |
| MinIO console | http://localhost:9001 (`minioadmin` / `minioadmin`) |

## Verify the key is actually live

Worth doing before a demo, because the deterministic fallback is silent by
design and a stale key looks like a working system producing worse output.

```bash
docker compose run --rm --no-deps api python -c "
from app.core.llm.factory import get_optional_llm
from app.core.llm.base import ModelTier, user
c = get_optional_llm()
print(type(c).__name__)
print(c.complete([user('Reply with exactly: OK')], tier=ModelTier.SMALL).text)
"
```

Expect `GeminiLLM` and `OK`. If you get `NoneType`, the key is missing from the
container's environment — check `.env` is at the repo root and that you ran
`make up` (not just `docker compose start`) since editing it.

The stronger signal is on the confirmation screen after an upload:
**`llm_enriched: true`** in the schema report means the LLM pass ran. `false`
means it was skipped or failed and you are looking at deterministic-only
detection.

## The demo path, with real numbers

Upload → confirm → watch Progress → read Results → ask the Chat page something.

Both example datasets, end to end, measured:

| | `customer_churn.csv` | `house_prices.csv` |
|---|---|---|
| Rows × columns | 500 × 10 | 606 × 13 |
| Target | `churned` (classification) | `sale_price` (regression) |
| Upload + schema detection | **1.8s** | **2.1s** |
| Confirm → queued | 0.5s | 0.1s |
| **Pipeline wall clock** | **63.5s** | **87.7s** |
| LLM calls | 6 | 6 |
| Tokens (in / out) | 7,922 / 2,478 | 8,760 / 2,674 |
| Cost | $0.00 | $0.00 |
| Result | `completed` | `completed`, r² 0.907 |

Schema detection is synchronous inside the upload request — that ~2s is a live
LLM call, and it is why the confirmation screen can render without a second
round trip. On both runs it picked the correct target and task type unaided,
and flagged the PII columns (`customer_id`, `email`; `agent_email`).

**Budget 60–90 seconds of dead air per run.** Where it goes:

| Stage | churn | house prices |
|---|---|---|
| planner | 2.0s | 1.1s |
| cleaning | 0.0s | 0.0s |
| eda | 4.2s | 5.1s |
| feature_strategy | 2.0s | 2.4s |
| preprocessing → final_training | ~1.1s | ~1.6s |
| explainability | 0.8s | 0.6s |
| **critic** | **21.0s** | **36.8s** |
| **report** | **27.6s** | **36.7s** |
| chat_index | 2.2s | 1.7s |

The critic and the report writer are ~75% of the run. They are the two
large-tier agents writing the most prose; everything deterministic is
sub-second. `sampling` and `feature_selection` were skipped on both datasets by
the planner's conditional edges — that is the dynamic orchestration working, not
a stage failing.

Talk over the Progress page while those two run. It polls `GET /jobs/{id}` and
shows per-node status, so there is something moving on screen.

## Rate limits: what actually happens

The house-prices run hit the free tier's limit on the large tier mid-pipeline
and **completed anyway**:

```
Rate limited on the large tier; stepping down
POST .../gemini-3.1-flash-lite:generateContent "HTTP/1.1 200 OK"
```

Both the critic and the report writer ran on `small` for that job where the
churn run had them on `large`. The output is shorter and a little plainer; the
run does not fail and does not lose its prose. This is the intended behaviour
and it is verified under a full pipeline, not just in a unit test.

Two consequences for a demo:

- **Back-to-back runs will step down.** If you run both example datasets in
  quick succession, expect the second to be small-tier throughout. Leave a
  couple of minutes between them if you want the examiner to see large-tier
  prose, or just explain the fallback — it is a better story than a nicer
  paragraph.
- **Cost is genuinely $0.00, not a broken counter.** `PRICE_PER_MILLION` in
  [`app/core/llm/usage.py`](../app/core/llm/usage.py) is zero for both tiers
  because the free AI Studio tier is free. The column and the arithmetic exist
  so that a paid endpoint is a change to that one table. Say this before someone
  asks whether cost tracking works.

## Before a demo

- [ ] `make up && make migrate && make health`
- [ ] Verify the key is live (above) and that a fresh upload shows
      `llm_enriched: true`
- [ ] Do one throwaway run end to end. It warms the image, the model tiers and
      your own memory of the click path.
- [ ] Leave a few minutes before the real run, so you start with the large tier
      un-throttled.
- [ ] Know that the PDF button 404s until the report node finishes. The
      Markdown report at `/jobs/{id}/report` is the authoritative version and
      appears at the same time.

## Troubleshooting

**A code change had no effect.** The worker never reloads. `./app` is mounted so
uvicorn restarts itself for API changes, but Celery loaded its modules at
process start. After any change to pipeline or agent code:

```bash
docker compose restart worker
```

After a dependency or `docker-compose.yml` change, `docker compose up -d` —
`restart` reuses the old container definition and will not pick up a new mount
or a new package.

**The app lost its theme after switching branches.** `.streamlit/` is bind-mounted
into the ui container, and `git checkout` deletes and recreates that directory
whenever you move between a branch that has it and one that does not. The mount
then points at an inode that no longer exists, so the container sees an *empty*
`.streamlit/` and silently falls back to stock Streamlit — no error, no warning,
just default grey where the theme should be.

```bash
docker compose up -d --force-recreate ui
```

`restart` is not enough; the mount has to be re-resolved. Confirm with:

```bash
docker compose exec ui python -c "
from streamlit import config; config.get_config_options(force_reparse=True)
print(config.get_option('theme.primaryColor'))"
```

`#1D6F5C` means the theme is live; `None` means the mount is stale.

**The logs are unreadable.** `DEBUG=true` with `ENVIRONMENT=local` turns on
SQLAlchemy echo ([`app/core/db.py:27`](../app/core/db.py#L27)), which logs every
statement including full embedding vectors from the chat index — a single run
produces tens of thousands of lines. For a demo where you might show logs, set
`DEBUG=false` in `.env` and `make up`. To follow just the pipeline:

```bash
docker compose logs -f worker | grep -v "sqlalchemy.engine"
```

**The first upload fails.** Almost always missing migrations. `make migrate`.

**A job sits in `queued`.** The worker is not consuming. `docker compose ps
worker` and `docker compose logs worker --tail 50`. A worker that died mid-job
leaves the job `running` forever; there is no reaper, so re-upload rather than
waiting.

**Was PII modelled?** It should not be. Detection flags PII and sets
`exclude: true` on those columns, and confirming without a `columns` array
inherits that — so the safe choice is the default whether you go through the UI
or straight at the API. To model a flagged column deliberately, send it with
`exclude: false`.

The one exception is a PII-flagged column chosen as the *target*: that is never
inherited as excluded, since it is the column being predicted. It stays marked
`is_pii: true` — the flag is information, the exclusion is policy.

(Before the fix, an omitted array meant *no exclusions* and silently discarded
what detection had decided. A live run modelled an `agent_email` column that way
and the critic remarked on the model's reliance on it.)

## Shutdown

```bash
make down     # stops everything, data preserved
make clean    # stops and DELETES all volumes -- Postgres, MinIO, Redis
```

`make down` between demo days is fine and cheap. `make clean` means re-running
`make migrate` and losing every previous job.
