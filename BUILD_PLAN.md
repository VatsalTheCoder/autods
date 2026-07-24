# AutoDS — Build Plan

| | |
|---|---|
| **Companion to** | `AutoDS_v4_Final_Portfolio_Spec.md` (v5.0) |
| **Purpose** | Break the spec into sections you can actually sit down and code |
| **Assumed budget** | ~24 working weeks, solo |

Week counts are rough weights, not promises. The one rule that matters: **Sections 0–5 go in order** — each genuinely needs the one before it. **Sections 6–10 are independent** — reorder or drop them freely if time gets tight.

---

## What we're building

A website where someone drags in a spreadsheet, and a few minutes later gets back: cleaned data, charts describing it, a trained model that predicts something, an explanation of *why* the model predicts what it does, a written report, and a chat box to ask questions about all of it.

"Multi-agent" means we don't write one giant program. We write a dozen small specialists — one cleans, one explores, one trains models, one criticises the work, one writes the report — and LangGraph coordinates them.

---

## The idea behind the ordering

There are two ways to build this.

**The tempting way:** perfect the cleaning agent. Then perfect the EDA agent. Then the modelling agent. Wire it all together at the end.

**Why that fails:** you find out in month five whether the pieces connect. They never do first try. The background worker can't read the file the API uploaded. Streamlit can't reach the backend inside Docker. A database migration conflicts. Each of those is a multi-day fight, and you hit all of them at once, at the worst moment, with a deadline.

**What we do instead:** build a pathetically thin version that runs end to end, first. Upload a CSV → clean it badly → train one boring model → print a one-page report. Not impressive. But it proves the wiring works. After that, every remaining task is swapping a weak part for a strong one, inside a system you already know runs.

That's a "vertical slice" — a thin cut through every layer, instead of one complete layer at a time. It's the entire logic of the section order below.

---

## Two changes from the spec's milestone list

**1. The spec's M1 is secretly a two-month milestone.** It says "through the real stack (FastAPI + Celery + Streamlit + Postgres + S3)" — but that stack doesn't exist yet and is ~6 weeks of work on its own. Sections 0–4 are that plumbing, pulled out and made visible, so Section 5 is honestly a 2-week job instead of a hidden cliff.

**2. "AWS hardening" moved to the front.** The spec puts it at M7, but §2 of the spec also says "cloud-ready by construction." Writing artifacts to S3 and reading config from environment variables costs nothing in week 1 and is miserable to retrofit into forty files in week 20. So that groundwork sits in Section 0, and only the real deploy work — EC2, Secrets Manager, billing alarm — stays at the end.

---

# Phase A — The spine (weeks 1–8)

Six sections, strictly in order. None is impressive alone. All of them are what makes the impressive parts possible.

---

## Section 0 — Skeleton · ~1 week

**What it is:** no features at all. Just getting five containers to start up and see each other.

**Why it's a whole section:** "make five Docker services talk to each other" is a real week of work. Much better for that week to be *only* about that, rather than tangled up with debugging machine-learning code at the same time.

**Build:**
- `git init`; repo layout (`app/api`, `app/agents`, `app/core`, `app/ml`, `ui/`, `tests/`, `alembic/`)
- Dependency management (`pyproject.toml` + lockfile, or pinned `requirements.txt`)
- `docker-compose.yml` with five services: `api`, `worker`, `ui`, `postgres`, `redis`
- `app/core/config.py` — every setting read from environment variables, `.env.example` committed
- `app/core/storage.py` — all S3 access behind one interface (use MinIO locally if you prefer). **From here on, no other file in the project ever writes artifacts to the hard drive.**
- Alembic set up against Postgres
- `GET /health`, and a Streamlit page that successfully calls it
- A `Makefile` for `up`, `test`, `migrate`, `fmt`

**Done when:** `docker compose up` gives five healthy containers, the Streamlit page shows a value it fetched from FastAPI, and `alembic upgrade head` runs clean.

**In plain English, the work is:**
- Set up folders so the code has an obvious place to live
- Write a recipe that builds an identical mini-computer for the app to run inside, so it behaves the same on your Mac as on a server
- Write one file listing every mini-computer the project needs and how they find each other
- Put every setting — passwords, addresses, bucket names — into environment variables, so nothing is written into the code itself
- Write one module that saves and loads files from cloud storage, and make a rule that nothing else in the project is ever allowed to touch files directly
- Add a web address that answers "am I working?", and a webpage that asks it and shows the answer

> ✅ **Complete** — six containers healthy (a MinIO storage container was added to the five planned), `/health` returning 200, and the API → queue → worker round-trip already working.

---

## Section 1 — Upload · ~1 week

**What it is:** the first thing that actually happens. Pick a CSV, it lands in S3, a row appears in the database saying "job 1 exists."

**Why it matters:** small, but it's the first time all three storage systems — S3, Postgres, the browser — do something real together.

**Build:**
- Migration 1: `users`, `jobs`, `artifacts` tables (with the hardcoded `user_id = 1` dev stub)
- `POST /upload` — validate the CSV, stream it to S3, insert a `jobs` row, return a `job_id`
- Helpers to save/load artifacts, and presigned URLs so Streamlit can read them
- Streamlit Upload page: file picker → shows the `job_id` and a preview of the data

**Done when:** you upload a CSV in the browser and can see both the object in S3 and the row in Postgres.

**In plain English, the work is:**
- Create three database tables: one for users, one for uploaded jobs, one listing the files each job produces
- Add a web address the browser can send a file to
- When a file arrives: check it really is a valid spreadsheet, put it in storage, write down in the database that it happened, and hand back a ticket number
- Build the upload page — a file picker, and a preview of the first few rows so the user can see their data arrived intact
- Add a way to give the browser a temporary, expiring link so it can read files straight from storage instead of routing everything through the backend

---

## Section 2 — The LLM client · ~1 week

**What it is:** one shared module that every AI agent will call. Build it *before* writing any agent.

**Why before the agents:** every agent needs the same four things — send a prompt, force the reply into a valid JSON shape, retry when the model returns garbage, and stay under the API rate limit. Write your first agent first, and all four get baked into that agent, then copy-pasted into the next eight. Change the rate limiter later and you're editing nine files.

It also gets you a `FakeLLM` — a stand-in returning canned answers — which is what lets your tests run without network. Tests that call a real LLM are slow, cost money, and give a different answer every run, which makes them useless as tests.

**Build:**
- A client for the serving API, with the model tier (small vs large) selectable
- Structured output: Instructor/Outlines + Pydantic, retrying N times on invalid output
- A rate limiter (token bucket) with exponential backoff when the API returns 429
- Migration 2: `token_usage` table, plus a callback logging tokens and cost per agent per job
- `FakeLLM` implementing the same interface for tests

**Done when:** a test round-trips a Pydantic schema through the real client; a second test suite exercises the same code paths through `FakeLLM` with zero network calls; and a deliberately malformed response visibly triggers the retry path.

**In plain English, the work is:**
- Write one wrapper that knows how to talk to the AI model, so no other part of the project has to
- Make it insist the AI replies in a fixed shape, and automatically ask again when the reply comes back malformed
- Add a queue that deliberately slows requests down, so you never get temporarily blocked for sending too many too fast
- Record how many words each request used and what it cost, in a database table
- Build a fake version that returns pre-written answers, so tests can run instantly, for free, and give the same result every time

> ⚠️ **Check this first.** The model names, benchmark scores, and free-tier quotas in spec §6.3 are unverified — the spec itself flags them. Confirm the real requests-per-minute and input-tokens-per-minute limits from live provider docs before you size the rate limiter. Spec §10 designs the entire prompt-shrinking strategy around the input-token cap being the binding constraint, so if that number is wrong, Section 9 is built on sand.

---

## Section 3 — Schema detection & the human checkpoint · ~1.5 weeks

**What it is:** the first real agent. It reads the uploaded CSV and works out what's in each column, which column is probably the thing to predict, whether it's a classification or regression problem, and whether any columns hold personal data.

Then it **stops and asks the user**: "I think you want to predict `churn`, and I think `email` is PII — correct?" The user fixes it before anything expensive runs.

**The design trick worth understanding:** pausing a running workflow to wait for a human is genuinely hard. So schema detection runs *immediately and synchronously during upload*, before any background job starts. The user confirms. *Then* the long job launches. You get the human checkpoint without ever freezing and resuming a workflow — the pause happens in the gap between two separate jobs. (This is spec §7.1–7.2; it's a good call and worth keeping.)

**Build:**
- Deterministic profiling: dtypes, cardinality, null rates, regex PII (email/phone/SSN), class-imbalance check
- LLM pass: what each column means, suggested target, task type, semantic PII like name columns
- `SchemaReport` Pydantic model saved as `schema_report.json`
- Hook schema detection into `POST /upload` **synchronously**
- Streamlit confirmation screen: editable target, task type, PII flags, excluded columns
- `POST /jobs` — accepts and stores the confirmed schema

**Done when:** upload → schema appears in a few seconds → you edit and confirm it → the confirmed version is saved against the job.

**In plain English, the work is:**
- Write code that inspects every column and reports what kind of data it holds, how many blanks it has, and how many different values appear
- Add pattern matching that spots emails, phone numbers and ID numbers automatically
- Ask the AI what each column actually *means*, which one looks worth predicting, and whether any column holds someone's name
- Run all of this the instant the file is uploaded, while the user is still sat watching, rather than later in a background job
- Build a screen that shows every guess with everything editable, so the user can correct it
- Save whatever the user confirmed, and use that from then on

---

## Section 4 — The background worker · ~1.5 weeks

**What it is:** training models takes minutes, and a browser request can't take minutes — it times out. So the work moves to a background worker, and the browser polls a progress page.

**The trick:** build all of this with **fake work inside it**. The workflow nodes literally just sleep a few seconds and report "done." Same principle as Section 0 — prove the distributed execution works before putting anything valuable inside it. When the progress bar ticks through five sleeping nodes, your infrastructure is correct.

**Build:**
- Celery worker pulling jobs from Redis; `POST /jobs` queues the pipeline task
- The worker reads the CSV back **from S3**, never from local disk
- A LangGraph `PipelineState` model holding everything passed between nodes
- A graph of 2–3 placeholder nodes that sleep and write status
- Status saved per node; any crash marks the job `failed` with a readable message (spec §10)
- `GET /jobs/{id}` returning job status and per-agent status
- Streamlit Progress page polling it

**Done when:** you confirm a job, watch the Progress page tick through the nodes, and a deliberately thrown exception shows up as `failed` with a clear reason instead of hanging.

**In plain English, the work is:**
- Set up a second program that sits waiting, picks jobs off a queue, and does the slow work while the website stays responsive
- Have it fetch the spreadsheet back out of storage, because it cannot see the backend's disk — they are separate machines as far as the code is concerned
- Define the shared clipboard that gets handed from one step to the next
- Wire up a chain of pretend steps that just wait a few seconds and say "done"
- Record which step is currently running and save it where the website can read it
- Add an address the browser can keep asking "how's it going?", and a page that displays it
- Break a step on purpose and check the job reports a clear failure instead of hanging forever

---

## Section 5 — The vertical slice · ~2 weeks

**What it is:** swap the sleeping placeholder nodes for real ones. This is spec M1.

**Build:**
- **Planner** (LLM, minimal for now): outputs a plan JSON that toggles one or two flags
- **Cleaning** (code): missing values, duplicates, dtype fixes, constant columns, drop PII columns
- **Preprocessing** (code): an unfitted `ColumnTransformer` with a hardcoded strategy — the LLM does *not* choose strategies yet, that's Section 7
- **One model**, trained inside 5-fold cross-validation, with the pipeline **fitted only inside the folds** (see the warning below — get this right now and never break it)
- **Evaluation**: the metric set for the task type → `evaluation_report.json`
- **A simple Markdown report**
- Streamlit Results page showing the metrics and the report

**Done when — the most important checkpoint in the project:** a CSV uploaded in the browser produces a cross-validated score and a readable report, with every artifact in S3 and every status in Postgres. The whole product is demoable. It's a weak version — one model, hardcoded preprocessing, plain-text report — but from here everything is just upgrading parts of a machine that already runs.

**In plain English, the work is:**
- Write the cleaning code: fill in blanks, drop duplicate rows, fix columns stored as the wrong type, delete columns that never change, remove personal data
- Write out a *recipe* describing how each column should be prepared — but deliberately do not run it yet
- Split the data into 5 chunks. For each chunk in turn: apply the recipe using only the other 4, train a model on those 4, then test on the held-out chunk
- Score the results with the right measures for the kind of problem it is
- Write the numbers out as a short report and show it on a results page
- Ask the AI for a small plan, and wire up one or two on/off switches it controls

> ⚠️ The split-into-5 step is where the whole project's credibility sits. The recipe must be applied *separately inside each chunk*, never once over everything up front. See the final section of this document for why.

---

# Phase B — The capabilities (weeks 8–19)

Where the actual portfolio value lives. Each section is self-contained and visibly improves the demo when finished. Run out of time and you can drop one and still have a working product — that's the whole payoff of doing Phase A first.

---

## Section 6 — EDA & clustering · ~2 weeks · spec M2

**What it is:** charts describing the data, plus clustering — automatically finding natural groupings of rows ("these customers behave similarly"). An LLM then describes each group in plain English.

**Build:**
- Statistics (code): summary stats, correlations, class balance, missingness
- Charts: histograms, boxplots, correlation heatmap, target distribution → PNGs to S3
- Planner picks K-Means (numeric-only data) or K-Prototypes (mixed data), per spec §9
- Silhouette score to auto-pick the number of clusters; PCA 2-D scatter coloured by cluster
- LLM (small tier) writes a profile per cluster
- Optional: a dendrogram on a ~300-row sample, clearly labelled as illustrative only

> ⚠️ **Enforce in code:** cluster labels must never become model features. Computing them over the whole dataset and feeding them to the model leaks test information into training (spec §9).

**Done when:** the Results page shows the charts, the cluster scatter, and a readable description of each cluster.

**In plain English, the work is:**
- Work out the basic numbers for every column: average, spread, how many blanks, and which columns move together
- Draw the standard charts and save them as images into storage
- Automatically sort rows into groups of similar records
- Try several different group counts and keep whichever splits the data most cleanly
- Squash the data down to two dimensions so those groups can actually be drawn on a flat chart
- Ask the AI to describe each group in a sentence or two, in language a non-technical person understands
- Make sure the group labels are used for *insight only* and never fed into the model as an input

---

## Section 7 — Feature engineering · ~2.5 weeks · spec M3

**What it is:** the best demonstration of the project's core design idea. The LLM is asked, per column, "how should this be handled — scale it? one-hot encode it? fill blanks with the median?" It replies with a small JSON object. Then **ordinary Python code** reads that JSON and builds the real scikit-learn pipeline.

**Why that split matters:** the LLM never touches the data. It only makes small decisions that code carries out. An LLM writing a JSON dictionary is something you can validate against the real column list. An LLM writing pandas code that transforms your dataset is something you cannot.

**Build:**
- LLM emits per-column strategy JSON, validated against the actual columns so invented column names get rejected
- Code builds the `Pipeline` / `ColumnTransformer` from it — **still unfitted**
- Ordinal encoding handled, with the choice recorded
- SMOTE through `imblearn.pipeline.Pipeline` so the resampling happens **inside** each fold
- Feature selection and sampling as conditional branches reading the Planner's plan — this is your "dynamic orchestration" claim, so make sure a skipped step is *visibly* skipped in the UI
- Full model roster: LogReg/LinReg, Random Forest, XGBoost, LightGBM
- `leaderboard.json`

**Done when:** two datasets with different shapes visibly take different routes through the graph, and the leaderboard ranks the full model roster.

**In plain English, the work is:**
- Ask the AI, column by column, how that column should be prepared — and have it answer as a small table of decisions, not as code
- Check its answer only mentions columns that genuinely exist, and reject it if it invents one
- Turn that table of decisions into a real preparation recipe — still without running it
- Add the step that balances rare cases, and put it in exactly the right place: applied to the training portion only, never to the portion being tested on
- Let the plan switch optional steps on and off, and make the website visibly show when a step was skipped
- Train four different types of model and rank them on a leaderboard

---

## Section 8 — Final training, SHAP & prediction · ~2 weeks · spec M4

**What it is:** retrain the winning model on all the data, save it, then run SHAP — a technique that shows how much each feature pushed each prediction up or down. This is your explainable-AI claim.

**Build:**
- Final Training: refit the winning pipeline on the full dataset → `final_model.pkl` to S3
- SHAP on that model; `TreeExplainer` for tree models, sample rows if the data is big
- Global importance, summary plot, dependence plots, local explanations
- `POST /jobs/{id}/predict`, loading the model from S3

> ⚠️ **Budget real time for the name mapping.** SHAP explains the model in terms of *transformed* features with names like `cat__city_London` and `num__income`. Users need to see `city` and `income`. Mapping back is unglamorous, surprisingly fiddly, and the single most likely task here to eat three unplanned days (spec §7.10).

**Done when:** SHAP plots appear on the Results page with human-readable feature names, and a live prediction works against the saved model.

**In plain English, the work is:**
- Take whichever model won, retrain it on all the data this time, and save it into storage
- Run the explanation tool over it to work out how much each column pushed each prediction up or down
- Translate the machine's internal column names back into the ones the user recognises — the fiddly bit, budget real time for it
- Draw two kinds of chart: which columns matter most overall, and why one specific row got the answer it got
- Add a web address that takes new data and returns a live prediction from the saved model

---

## Section 9 — Critic & report · ~2 weeks · spec M5

**What it is:** an LLM reviews the whole pipeline's work and critiques it; another writes the final PDF.

**The catch:** these are your biggest prompts, and the free API tier caps input tokens per minute. You can't dump every artifact into the prompt. Summarise first — round the numbers, cap the feature lists, truncate the tables.

**Build:**
- Critic agent reading **summarised** cleaning/FE/evaluation/explainability JSON, not raw artifacts
- Reusable prompt-shrinking helpers
- Report agent → Markdown, then PDF via WeasyPrint or ReportLab
- `GET /jobs/{id}/report`

**Done when:** a multi-section PDF downloads from the browser — and you've tested it on a dataset with 100+ columns, because that's where the token cap breaks, not on your tidy 8-column demo file.

**In plain English, the work is:**
- Write helpers that squeeze all the results down into a short summary — round the numbers, cut long lists short, trim big tables
- Ask the AI to review the whole pipeline's work and honestly point out its weaknesses
- Ask the AI to write the final report, section by section
- Convert that report into a proper PDF
- Add a web address to download it
- Test it on a spreadsheet with a hundred-plus columns, because that is where it will break — a tidy little demo file will pass and tell you nothing

---

## Section 10 — RAG chat · ~2.5 weeks · spec M6

**What it is:** a chat box over the results. Two kinds of question need two different tools:

- *"Why was transaction amount important?"* — a meaning question, answered by searching the report text. That's RAG.
- *"What's the average age?"* — arithmetic. No amount of text search answers this correctly, so it goes to a pandas query instead.

The agent decides which tool to use. Getting that routing right is the interesting part.

**Build:**
- Chunk and embed the report/EDA/SHAP/critic/cluster text with a CPU sentence-transformer
- Store in ChromaDB (keep the pgvector fallback in mind — see the AWS note in Section 11)
- Retrieval plus grounded answer generation
- A pandas query tool for arithmetic
- Routing between the two
- `POST /jobs/{id}/chat`, a `chat_history` table, and a Streamlit Chat page

**Done when:** "why was X important?" routes to RAG, "what's the average age?" routes to pandas, and both answer correctly on a dataset you know the answers for.

**In plain English, the work is:**
- Chop all the reports into paragraphs, and convert each into a list of numbers that captures its meaning — so you can search by *meaning* rather than by exact wording
- Store those so they can be searched quickly
- When a question comes in, find the handful of most relevant paragraphs and hand them to the AI along with the question, so its answer is grounded in real output rather than invented
- Build a second tool that runs actual calculations over the spreadsheet
- Write the decision that picks which of the two tools a question needs
- Save the conversation history, and build the chat page

---

# Phase C — Closing

## Section 11 — AWS deployment · ~2 weeks · spec M7

**What it is:** put it on a real server, with real secrets management and a billing alarm so you don't get a surprise charge.

**Why it's short:** because Section 0 did the groundwork. Nothing writes to local disk and all config comes from environment variables, so deploying is mostly "run the same containers on a bigger machine." That's what "cloud-ready by construction" cashes out to.

**Build:**
- EC2 instance (t3.medium minimum — SHAP and XGBoost eat memory), running `docker compose up`
- Secrets Manager for the API key and database password
- Migrations run against the real database
- Billing alarm, and a written stop/start procedure for demo days
- A persistent EBS volume for ChromaDB — **or** switch to the pgvector fallback, which avoids the problem entirely by removing a service
- Document ECS Fargate + RDS + ElastiCache as future work

**Done when:** there's a URL an examiner can open, and a runbook for bringing it up from cold.

**In plain English, the work is:**
- Rent a computer from Amazon and get the containers running on it, the same way they run on your Mac
- Move the passwords and API keys out of files and into Amazon's secure store
- Point the storage module at real Amazon storage instead of the local stand-in — one setting
- Run the database setup against the real database
- Set an alarm that emails you if the bill goes above a few pounds, and write down how to switch the machine off when you are not demoing
- Write instructions for starting the whole thing from nothing, so you are not reconstructing it from memory the night before a demo

---

## Section 12 — Testing & docs · ongoing, ~1.5 weeks concentrated

Write tests **inside each section as you go** — this slot is the consolidation pass, not "write all the tests at the end."

- Concentrate coverage on the deterministic core: cleaning, pipeline construction, CV wiring, metrics, SHAP wiring. Spec §15 explicitly rejects a blanket 80% target — coverage goes where it's meaningful.
- LLM agents tested against the `FakeLLM` from Section 2
- One end-to-end integration test
- **The leakage test is the most valuable test in the repo** — see below
- Architecture diagram, agent-interaction diagram, database ER diagram, API docs, setup guide, AWS deployment guide
- README with example datasets

**In plain English, the work is:**
- Write tests as you go, not at the end — each section adds its own
- Test the boring, predictable parts hardest: cleaning, recipe building, the scoring maths. Those are the parts that can be checked exactly
- Write the one test that matters most: proving the preparation recipe really is unused when it leaves the feature-engineering step
- Use the fake AI from Section 2 to test the AI-driven parts, so tests stay fast and repeatable
- Write one test that runs the entire thing start to finish
- Draw the diagrams, write the setup guide, and include a couple of example spreadsheets so someone else can actually run it

---

# The one thing to actually be careful about

Everything else in this plan is negotiable. This isn't.

When you evaluate a model you split data into training and test parts. If *anything* learned from the test data leaks into training, your scores come out inflated and meaningless. Two classic ways it happens:

- **Scaling or imputing across the whole dataset before splitting.** Your training data has now quietly absorbed the average of the test data.
- **Running SMOTE before splitting.** SMOTE invents extra rows for the minority class; do it first and synthetic copies of test rows end up sitting in your training set.

Both silently inflate every number you report. Nothing crashes. You just get a great-looking score that is a lie.

**The fix** (spec §8): feature engineering hands over an **unfitted** pipeline — a recipe, not a cooked meal — and it's only ever fitted *inside* each cross-validation fold. SMOTE goes inside the fold too, which is why you use `imblearn`'s pipeline rather than scikit-learn's.

Get it right in Section 5, never break it, and write the test in Section 12 that asserts the pipeline is genuinely unfitted when it leaves feature engineering. This is the claim an examiner is most likely to probe, and the one most student projects get wrong.

---

# Schedule at a glance

| Section | Weeks | Cumulative | Milestone |
|---|---|---|---|
| 0 · Skeleton | 1 | 1 | |
| 1 · Upload | 1 | 2 | |
| 2 · LLM client | 1 | 3 | |
| 3 · Schema & human checkpoint | 1.5 | 4.5 | |
| 4 · Background worker | 1.5 | 6 | |
| 5 · Vertical slice | 2 | **8** | **M1 — demoable end-to-end** |
| 6 · EDA & clustering | 2 | 10 | M2 |
| 7 · Feature engineering | 2.5 | 12.5 | M3 |
| 8 · Final training & SHAP | 2 | 14.5 | M4 |
| 9 · Critic & report | 2 | 16.5 | M5 |
| 10 · RAG chat | 2.5 | 19 | M6 |
| 11 · AWS deployment | 2 | 21 | M7 |
| 12 · Testing & docs | 1.5 | 22.5 | |
| — buffer | 1.5 | 24 | |

---

# If you fall behind

**Cut in this order:** dendrogram → the optional conditional steps (feature selection, sampling) → the pandas tool in chat → the Critic agent.

**Simplify infrastructure** using the spec's §4 fallback table: Celery → FastAPI `BackgroundTasks` (drops two services), ChromaDB → pgvector (drops one). Each removes a whole class of deployment problem.

**Never cut** the leakage-safe cross-validation (Section 5), SHAP (Section 8), or the AWS deployment (Section 11). Those three are what the project is sold on.
