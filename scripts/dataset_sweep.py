"""Run many datasets through the whole pipeline and record what each one did.

Every dataset shape takes a different route through this system. A numeric-only
frame clusters with K-Means and a mixed one with K-Prototypes; a continuous
target skips resampling entirely; a wide frame is where the report agent's
prompt budget gives out. Those routes were each built and tested on their own,
but nothing has ever run *a set* of shapes back to back and put the outcomes
side by side -- so a change that quietly breaks one shape while the demo dataset
keeps passing has, until now, had nothing to trip over.

This is that thing. Point it at a directory of CSVs and it drives each one
through upload -> confirm -> pipeline in-process, then writes one row per
dataset saying: did it finish, how long it took, which optional steps the
planner switched on, which agents got a real model and which fell back to their
deterministic path, and what the winning model scored.

**It is a script, not a test, and the difference is deliberate.** The test suite
is hermetic by construction -- ``tests/conftest.py`` strips ``GOOGLE_API_KEY``
out of the environment before collection so nothing can reach a live model by
accident. That is right for tests and exactly wrong here: the fallback columns
below are only interesting when a real key is present, because what they measure
is how often the live model failed to answer and the deterministic path stood in
for it. So this lives in ``scripts/`` and reads the key like the app does.

Usage::

    # inside the stack, where Postgres and object storage are reachable
    docker compose exec api python scripts/dataset_sweep.py data/examples

    # with per-dataset targets rather than schema detection's guess
    docker compose exec api python scripts/dataset_sweep.py data/sweep \\
        --manifest scripts/sweep_manifest.json --out /tmp/sweep

Nothing here asserts. A sweep does not pass or fail -- it produces a table you
read, and a JSON file the next sweep can be diffed against.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ``scripts/`` is not on the path when the file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.api.main import app  # noqa: E402
from app.api.routes import upload as upload_routes  # noqa: E402
from app.core.db import SessionLocal, database_healthy  # noqa: E402
from app.core.storage import storage_healthy  # noqa: E402
from app.models.agent_run import AgentRun  # noqa: E402
from app.models.job import Job  # noqa: E402
from app.models.token_usage import TokenUsage  # noqa: E402
from app.services.artifacts import (  # noqa: E402
    CLEANING_ARTIFACT,
    CLUSTERING_ARTIFACT,
    CRITIC_ARTIFACT,
    EVALUATION_ARTIFACT,
    FEATURE_ARTIFACT,
    LEADERBOARD_ARTIFACT,
    NARRATIVE_ARTIFACT,
    PLANNER_ARTIFACT,
    PREPROCESSING_ARTIFACT,
    SCHEMA_ARTIFACT,
    load_json_artifact,
)
from app.worker.pipeline import run_pipeline  # noqa: E402

# The optional steps. A sweep's most useful single column is which of these
# fired, because "different datasets genuinely take different routes" is the
# dynamic-orchestration claim and this is the only place it is observed rather
# than asserted.
OPTIONAL_STEPS = ("sampling", "feature_selection")

# The API runs with SQL echo on, which is right when you are debugging one
# request and useless across a dozen multi-minute jobs: it buries the progress
# line and the table under tens of thousands of statements. Quietened by
# default, restored by ``--verbose``.
NOISY_LOGGERS = ("sqlalchemy.engine", "httpx", "google_genai", "app.core.storage")

# Where each agent records whether it got an answer from the model or fell back.
# The names differ per contract -- ``llm_enriched`` on the schema report,
# ``strategy_source`` on the preprocessing spec -- so they are mapped once here
# rather than special-cased at every read.
LLM_SOURCES: dict[str, tuple[str, str]] = {
    "schema": (SCHEMA_ARTIFACT, "llm_enriched"),
    "planner": (PLANNER_ARTIFACT, "source"),
    "feature_strategy": (FEATURE_ARTIFACT, "source"),
    "preprocessing": (PREPROCESSING_ARTIFACT, "strategy_source"),
    "critic": (CRITIC_ARTIFACT, "source"),
    "report": (NARRATIVE_ARTIFACT, "source"),
}


@dataclass(slots=True)
class SweepCase:
    """One dataset and how to confirm it at the human checkpoint."""

    path: Path
    target: str | None = None
    task_type: str | None = None
    exclude: list[str] = field(default_factory=list)
    # Only sent when set. Naming a time column switches cross-validation to
    # time-ordered folds; on a build without that support the API rejects the
    # field outright, which the record will show as a confirm failure rather
    # than a silently random split.
    time_column: str | None = None
    note: str = ""

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class RunRecord:
    """Everything worth knowing about one dataset's trip through the pipeline."""

    dataset: str
    note: str = ""
    job_id: int | None = None
    status: str = "not started"
    error: str = ""
    wall_seconds: float = 0.0
    n_rows: int | None = None
    n_columns: int | None = None
    target: str | None = None
    task_type: str | None = None
    # Per-node status and duration, which is where a slow shape shows itself.
    nodes: list[dict[str, Any]] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)
    failed_node: str | None = None
    llm_used: dict[str, bool] = field(default_factory=dict)
    cv_strategy: str | None = None
    primary_metric: str | None = None
    best_model: str | None = None
    best_score: float | None = None
    clustering: str | None = None
    columns_dropped: int | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ---- Reading the outcome back ----------------------------------------------


def _artifact(job_id: int, name: str) -> dict | None:
    with SessionLocal() as db:
        return load_json_artifact(db, job_id, name)


def _collect_nodes(record: RunRecord, job_id: int) -> None:
    """Per-node status and wall time, from the rows the Progress page reads.

    Reading the same table the UI reads is the point: if a node is missing here
    it is missing from the user's Progress page too.
    """
    with SessionLocal() as db:
        runs = list(
            db.execute(
                select(AgentRun).where(AgentRun.job_id == job_id).order_by(AgentRun.sequence)
            ).scalars()
        )

    for run in runs:
        seconds = None
        if run.started_at and run.finished_at:
            seconds = round((run.finished_at - run.started_at).total_seconds(), 2)
        record.nodes.append({"name": run.name, "status": str(run.status), "seconds": seconds})

        if str(run.status) == "skipped":
            record.skipped_steps.append(run.name)
        elif str(run.status) == "failed" and record.failed_node is None:
            # First failure only. Later nodes fail as a consequence of this one,
            # and listing all of them buries the cause.
            record.failed_node = run.name
            if run.error_message:
                record.warnings.append(f"{run.name}: {run.error_message}")


def _collect_llm_sources(record: RunRecord, job_id: int) -> None:
    """Which agents got a live answer and which took the deterministic path.

    A run with a valid key where half of these are ``False`` is the signal worth
    catching: the pipeline completes, the report renders, and the output is
    quietly the fallback rather than the model's. Nothing else surfaces that.
    """
    for agent, (artifact_name, key) in LLM_SOURCES.items():
        payload = _artifact(job_id, artifact_name)
        if payload is None:
            continue
        value = payload.get(key)
        record.llm_used[agent] = value is True or value == "llm"


def _collect_results(record: RunRecord, job_id: int) -> None:
    """Scores, route and cost -- whatever of it exists.

    Every read is optional on purpose. A job that failed at modelling still has
    a cleaning report worth recording, and a sweep that refused to summarise a
    partial run would throw away its most interesting rows.
    """
    if evaluation := _artifact(job_id, EVALUATION_ARTIFACT):
        record.cv_strategy = evaluation.get("cv_strategy")
        record.primary_metric = evaluation.get("primary_metric")
        record.warnings.extend(evaluation.get("warnings", []))

    if leaderboard := _artifact(job_id, LEADERBOARD_ARTIFACT):
        record.cv_strategy = record.cv_strategy or leaderboard.get("cv_strategy")
        entries = leaderboard.get("entries", [])
        scored = [e for e in entries if not e.get("error")]
        if scored:
            best = min(scored, key=lambda e: e.get("rank", 0))
            record.best_model = best.get("model_name")
            record.best_score = best.get("score")
        # A model that errored while others succeeded is invisible in the
        # leaderboard's top line, and is exactly the kind of shape-specific
        # breakage this sweep exists to find.
        for entry in entries:
            if entry.get("error"):
                record.warnings.append(f"{entry.get('model_name')}: {entry['error']}")

    if clustering := _artifact(job_id, CLUSTERING_ARTIFACT):
        record.clustering = f"{clustering.get('method')} k={clustering.get('k')}"
        record.warnings.extend(clustering.get("warnings", []))

    if cleaning := _artifact(job_id, CLEANING_ARTIFACT):
        before = cleaning.get("n_columns_before")
        after = cleaning.get("n_columns_after")
        if before is not None and after is not None:
            record.columns_dropped = before - after

    with SessionLocal() as db:
        usage = list(db.execute(select(TokenUsage).where(TokenUsage.job_id == job_id)).scalars())
    record.llm_calls = len(usage)
    record.input_tokens = sum(u.input_tokens for u in usage)
    record.output_tokens = sum(u.output_tokens for u in usage)
    record.cost_usd = round(float(sum(u.cost_usd for u in usage)), 6)


# ---- Running one dataset ----------------------------------------------------


@contextmanager
def _only_one_pipeline_per_job():
    """Stop ``POST /jobs`` from queueing the run the sweep is about to do itself.

    Confirming a job enqueues the Celery task, and the worker container picks it
    up immediately. The sweep then calls ``run_pipeline`` in-process for the same
    job id, so **two pipelines run the same job at once** -- against one row in
    ``jobs`` and one set of rows in ``agent_runs``. They race: the second one's
    ``init_agent_runs`` deletes the rows the first is mid-way through updating,
    and the loser dies with ``StaleDataError: expected to update 1 row(s); 0 were
    matched``. Which one loses is a coin toss, so it reads as a flaky dataset
    rather than as a harness fault.

    It also quietly doubles the token spend and makes every recorded artifact
    ambiguous about which of the two runs produced it.

    The integration tests neutralise the same call for the same reason. Patching
    the name in the route's namespace rather than in ``worker.tasks`` is what
    makes it take effect: the route imported the function directly.
    """
    original = upload_routes.enqueue_pipeline
    upload_routes.enqueue_pipeline = lambda job_id: None
    try:
        yield
    finally:
        upload_routes.enqueue_pipeline = original


def run_case(case: SweepCase, client: TestClient) -> RunRecord:
    """Drive one dataset from upload to finished job, recording as it goes.

    Never raises. A dataset that breaks the harness itself is a result, so the
    traceback is recorded and the sweep moves to the next one -- the alternative
    is a twelve-dataset run dying on the third and telling you nothing about the
    remaining nine.
    """
    record = RunRecord(dataset=case.name, note=case.note)
    started = time.monotonic()

    try:
        data = case.path.read_bytes()
        upload = client.post("/upload", files={"file": (case.name, data, "text/csv")})
        # Any 2xx, not ``== 200``: upload answers 201 Created, and a harness
        # that reads a successful upload as a rejection reports every dataset
        # as broken while the stack is working perfectly.
        if not _ok(upload):
            record.status = "upload rejected"
            record.error = _detail(upload)
            return record

        payload = upload.json()
        record.job_id = payload["job_id"]
        record.n_rows = payload["preview"]["n_rows"]
        record.n_columns = payload["preview"]["n_columns"]

        schema = payload["schema_report"]
        target = case.target or schema.get("suggested_target")
        task_type = case.task_type or schema.get("task_type")
        if not target or not task_type:
            # Detection declining to guess is a finding, not a harness error:
            # the user would be staring at an unfilled checkpoint.
            record.status = "no target"
            record.error = (
                "schema detection suggested no target or task type, and the "
                "manifest does not supply one"
            )
            return record
        record.target = target
        record.task_type = task_type

        confirm_body: dict[str, Any] = {
            "job_id": record.job_id,
            "target_column": target,
            "task_type": task_type,
        }
        if case.exclude:
            confirm_body["columns"] = [{"name": name, "exclude": True} for name in case.exclude]
        if case.time_column:
            confirm_body["time_column"] = case.time_column

        with _only_one_pipeline_per_job():
            confirm = client.post("/jobs", json=confirm_body)
            if not _ok(confirm):
                record.status = "confirm rejected"
                record.error = _detail(confirm)
                return record

            # Synchronous on purpose. Celery would be more faithful to
            # production, but the sweep wants the run's wall time and its
            # outcome in one place, and polling a queue for a dozen multi-minute
            # jobs adds failure modes that belong to the harness rather than to
            # the datasets.
            run_pipeline(record.job_id)

    except Exception:  # noqa: BLE001 -- a broken dataset must not end the sweep
        record.status = "harness error"
        record.error = traceback.format_exc(limit=3)
        return record
    finally:
        record.wall_seconds = round(time.monotonic() - started, 1)

    with SessionLocal() as db:
        job = db.get(Job, record.job_id)
        record.status = str(job.status) if job else "vanished"
        if job and job.error_message:
            record.error = job.error_message

    _collect_nodes(record, record.job_id)
    _collect_llm_sources(record, record.job_id)
    _collect_results(record, record.job_id)
    return record


def _ok(response) -> bool:
    return 200 <= response.status_code < 300


def _detail(response) -> str:
    """The rejection reason, without assuming the response has one.

    Note the two-step read rather than ``.get("detail", response.text)``: a
    default argument is evaluated whether or not it is used, so the one-liner
    touches ``.text`` on every response including the ones that parsed fine.
    """
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        payload = None
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])[:400]
    return str(getattr(response, "text", payload))[:400]


# ---- Choosing what to run ---------------------------------------------------


def load_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    """Per-dataset overrides, keyed by filename.

    Optional throughout. Without it every dataset is confirmed with whatever
    schema detection suggested, which is itself worth sweeping -- it exercises
    the same guess the user is offered at the checkpoint.
    """
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    entries = payload["datasets"] if isinstance(payload, dict) else payload
    return {entry["file"]: entry for entry in entries}


def discover_cases(paths: list[Path], manifest: dict[str, dict[str, Any]]) -> list[SweepCase]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.glob("*.csv")))
        elif path.is_file() and path.suffix.lower() == ".csv":
            files.append(path)
        # A path that does not exist is dropped here and reported by ``main``.
        # Turning a typo into a case would produce a row reading "harness
        # error", which looks like the dataset's fault rather than the
        # command line's.

    cases = []
    for file in files:
        entry = manifest.get(file.name, {})
        cases.append(
            SweepCase(
                path=file,
                target=entry.get("target"),
                task_type=entry.get("task_type"),
                exclude=entry.get("exclude", []),
                time_column=entry.get("time_column"),
                note=entry.get("note", ""),
            )
        )
    return cases


# ---- Reporting --------------------------------------------------------------


def _fallbacks(record: RunRecord) -> str:
    """The agents that did *not* reach the model, which is the readable half."""
    fell_back = sorted(name for name, used in record.llm_used.items() if not used)
    if not record.llm_used:
        return "—"
    return ", ".join(fell_back) if fell_back else "none"


def summary_table(records: list[RunRecord]) -> str:
    header = (
        "| Dataset | Rows × Cols | Task | Status | Time | Best model | Score | "
        "Skipped | Fell back | Clustering |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in records:
        shape = f"{r.n_rows:,} × {r.n_columns}" if r.n_rows is not None else "—"
        score = f"{r.best_score:.3f}" if r.best_score is not None else "—"
        status = r.status if not r.failed_node else f"{r.status} @ {r.failed_node}"
        rows.append(
            f"| {r.dataset} | {shape} | {r.task_type or '—'} | {status} | "
            f"{r.wall_seconds:.0f}s | {r.best_model or '—'} | {score} | "
            f"{', '.join(r.skipped_steps) or 'none'} | {_fallbacks(r)} | "
            f"{r.clustering or '—'} |"
        )
    return header + "\n".join(rows)


def write_outputs(records: list[RunRecord], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    json_path = out_dir / f"sweep-{stamp}.json"
    json_path.write_text(
        json.dumps(
            {
                "run_at": stamp,
                "n_datasets": len(records),
                "records": [r.as_dict() for r in records],
            },
            indent=2,
            default=str,
        )
    )

    md_path = out_dir / f"sweep-{stamp}.md"
    completed = sum(1 for r in records if r.status == "completed")
    total_cost = sum(r.cost_usd for r in records)
    md_path.write_text(
        f"# Dataset sweep — {stamp}\n\n"
        f"{completed} of {len(records)} datasets completed. "
        f"{sum(r.llm_calls for r in records)} LLM calls, "
        f"${total_cost:.4f}.\n\n"
        + summary_table(records)
        + "\n\n## Warnings\n\n"
        + (
            "\n".join(f"- **{r.dataset}**: {w}" for r in records for w in dict.fromkeys(r.warnings))
            or "None."
        )
        + "\n"
    )
    return json_path, md_path


# ---- Entry point ------------------------------------------------------------


def _quieten() -> None:
    """Turn off the per-statement logging so the table is the visible output.

    Raising the logger's level is not enough for SQL. ``create_engine(echo=...)``
    attaches an ``InstanceLogger`` that carries its own echo flag and emits
    regardless of what level the ``sqlalchemy.engine`` logger is set to, so the
    engine's own switch has to be thrown as well.
    """
    from app.core.db import engine

    engine.echo = False
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="CSV files, or directories to take every *.csv from.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSON of per-dataset target/task/exclusions. Without it, schema "
        "detection's suggestion is confirmed as-is.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("sweep_results"), help="Where to write the results."
    )
    parser.add_argument("--verbose", action="store_true", help="Leave the SQL and HTTP logging on.")
    args = parser.parse_args(argv)

    if not args.verbose:
        _quieten()

    if not (database_healthy() and storage_healthy()):
        print(
            "Postgres and object storage must be reachable — start them with `make up`.",
            file=sys.stderr,
        )
        return 2

    # Checked before anything runs: a mistyped path would otherwise produce a
    # sweep that is quietly one dataset short, and a short sweep looks exactly
    # like a complete one.
    if missing := [p for p in args.paths if not p.exists()]:
        print(f"No such path: {', '.join(str(p) for p in missing)}", file=sys.stderr)
        return 2

    cases = discover_cases(args.paths, load_manifest(args.manifest))
    if not cases:
        print("No CSVs found in the given paths.", file=sys.stderr)
        return 2

    client = TestClient(app)
    records = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case.name} ... ", end="", flush=True)
        record = run_case(case, client)
        records.append(record)
        print(f"{record.status} in {record.wall_seconds:.0f}s")
        if record.error:
            print(f"    {record.error.splitlines()[-1][:200]}")

    json_path, md_path = write_outputs(records, args.out)
    print("\n" + summary_table(records))
    print(f"\nWritten to {json_path} and {md_path}")

    # Zero even when datasets fail: a sweep reports, it does not gate. Only a
    # harness that could not run at all is an error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
