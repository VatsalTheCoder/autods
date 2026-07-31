"""The sweep harness's own logic, which is the part that can be wrong quietly.

``scripts/dataset_sweep.py`` mostly drives infrastructure, and that half is
covered by actually running it. What is tested here is the half that *decides*
things -- which files become cases, which agents get reported as having fallen
back, and whether a dataset that explodes ends the whole sweep. That last one
matters most: a harness that dies on the third of twelve datasets tells you
nothing about the remaining nine, and it would fail in exactly the situation you
built it for.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "dataset_sweep.py"


def _load_module():
    """Import the script by path -- ``scripts/`` is not a package.

    The module has to be in ``sys.modules`` *before* it executes: its
    ``@dataclass(slots=True)`` decorators resolve annotations by looking their
    own module up there, and find ``None`` if it is not yet registered.
    """
    spec = importlib.util.spec_from_file_location("dataset_sweep", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sweep = _load_module()


class TestChoosingWhatToRun:
    def test_a_directory_contributes_every_csv_and_nothing_else(self, tmp_path):
        (tmp_path / "a.csv").write_text("x\n1\n")
        (tmp_path / "b.csv").write_text("x\n1\n")
        (tmp_path / "notes.md").write_text("not a dataset")
        (tmp_path / "model.pkl").write_bytes(b"\x00")

        cases = sweep.discover_cases([tmp_path], {})

        assert [c.name for c in cases] == ["a.csv", "b.csv"]

    def test_a_named_file_is_taken_without_a_manifest_entry(self, tmp_path):
        path = tmp_path / "one.csv"
        path.write_text("x\n1\n")

        cases = sweep.discover_cases([path], {})

        assert len(cases) == 1
        # No entry means schema detection's suggestion is confirmed as-is,
        # which is a deliberate mode rather than an incomplete case.
        assert cases[0].target is None
        assert cases[0].task_type is None

    def test_manifest_fields_land_on_the_matching_case_only(self, tmp_path):
        (tmp_path / "churn.csv").write_text("x\n1\n")
        (tmp_path / "other.csv").write_text("x\n1\n")
        manifest = {
            "churn.csv": {
                "file": "churn.csv",
                "target": "churned",
                "task_type": "classification",
                "exclude": ["customer_id"],
                "note": "baseline",
            }
        }

        cases = {c.name: c for c in sweep.discover_cases([tmp_path], manifest)}

        assert cases["churn.csv"].target == "churned"
        assert cases["churn.csv"].exclude == ["customer_id"]
        assert cases["churn.csv"].note == "baseline"
        assert cases["other.csv"].target is None

    def test_the_committed_manifest_parses_and_names_real_files(self):
        """The manifest is hand-edited, so a typo'd filename is a live risk.

        An entry that matches nothing is silently ignored by ``discover_cases``:
        the dataset still runs, just with detection's guess instead of the
        target you carefully chose. That is the failure this catches.
        """
        manifest_path = SCRIPT.parent / "sweep_manifest.json"
        entries = sweep.load_manifest(manifest_path)

        assert entries, "the committed manifest lists no datasets"
        data_dir = SCRIPT.parent.parent / "data"
        for name in entries:
            matches = list(data_dir.rglob(name))
            assert matches, f"manifest names {name}, which is not under data/"

    def test_a_manifest_may_be_a_bare_list(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps([{"file": "a.csv", "target": "y"}]))

        assert sweep.load_manifest(path)["a.csv"]["target"] == "y"

    def test_no_manifest_is_not_an_error(self):
        assert sweep.load_manifest(None) == {}


class TestReportingFallbacks:
    """The column that says the run completed but the model never answered."""

    def test_agents_that_missed_the_model_are_the_ones_named(self):
        record = sweep.RunRecord(dataset="d.csv")
        record.llm_used = {"planner": True, "critic": False, "report": False}

        # The agents that *fell back* are listed, not the ones that worked:
        # on a healthy run this column is short, and it grows as things break.
        assert sweep._fallbacks(record) == "critic, report"

    def test_a_fully_live_run_says_so_rather_than_going_blank(self):
        record = sweep.RunRecord(dataset="d.csv")
        record.llm_used = {"planner": True, "critic": True}

        assert sweep._fallbacks(record) == "none"

    def test_no_evidence_is_not_reported_as_success(self):
        """A job that never reached these agents must not read as 'none'.

        'none fell back' and 'nothing ran' are opposite outcomes and the table
        has one column for both, so they need different glyphs.
        """
        assert sweep._fallbacks(sweep.RunRecord(dataset="d.csv")) == "—"


class TestOneBadDatasetDoesNotEndTheSweep:
    def test_a_client_that_raises_is_recorded_and_survived(self):
        class ExplodingClient:
            def post(self, *args, **kwargs):
                raise RuntimeError("connection reset")

        record = sweep.run_case(sweep.SweepCase(path=SCRIPT), ExplodingClient())

        assert record.status == "harness error"
        assert "connection reset" in record.error

    def test_a_rejected_upload_is_a_row_not_an_exception(self, tmp_path):
        path = tmp_path / "huge.csv"
        path.write_text("x\n1\n")

        class RejectingClient:
            def post(self, *args, **kwargs):
                class Response:
                    status_code = 413

                    @staticmethod
                    def json():
                        return {"detail": "File exceeds the 200 MB limit."}

                return Response()

        record = sweep.run_case(sweep.SweepCase(path=path), RejectingClient())

        assert record.status == "upload rejected"
        assert "200 MB" in record.error
        # The size ceiling is a known finding, so it must survive into the
        # table rather than aborting the run that would have measured it.
        assert record.job_id is None

    def test_detection_declining_to_guess_is_a_finding(self, tmp_path):
        """No target and no manifest entry means an unfillable checkpoint."""
        path = tmp_path / "opaque.csv"
        path.write_text("a,b\n1,2\n")

        class NoTargetClient:
            def post(self, *args, **kwargs):
                class Response:
                    status_code = 200

                    @staticmethod
                    def json():
                        return {
                            "job_id": 7,
                            "preview": {"n_rows": 1, "n_columns": 2},
                            "schema_report": {"suggested_target": None, "task_type": None},
                        }

                return Response()

        record = sweep.run_case(sweep.SweepCase(path=path), NoTargetClient())

        assert record.status == "no target"
        assert record.job_id == 7


class TestTheTable:
    def test_every_dataset_gets_a_row_including_the_broken_ones(self):
        records = [
            sweep.RunRecord(
                dataset="good.csv",
                status="completed",
                n_rows=500,
                n_columns=10,
                task_type="classification",
                best_model="XGBoost",
                best_score=0.913,
                skipped_steps=["sampling"],
                clustering="kprototypes k=3",
            ),
            sweep.RunRecord(dataset="broken.csv", status="failed", failed_node="modeling"),
        ]

        table = sweep.summary_table(records)
        lines = table.strip().splitlines()

        assert len(lines) == 4  # header, separator, two rows
        assert "0.913" in table
        assert "500 × 10" in table
        # The failing node is what makes a failure row useful.
        assert "failed @ modeling" in table

    def test_a_missing_score_does_not_break_the_row(self):
        record = sweep.RunRecord(dataset="d.csv", status="failed")

        assert "—" in sweep.summary_table([record])


class TestWritingResults:
    def test_both_files_are_written_and_the_json_round_trips(self, tmp_path):
        records = [
            sweep.RunRecord(dataset="a.csv", status="completed", cost_usd=0.0012, llm_calls=6)
        ]

        json_path, md_path = sweep.write_outputs(records, tmp_path / "out")

        payload = json.loads(json_path.read_text())
        assert payload["n_datasets"] == 1
        assert payload["records"][0]["dataset"] == "a.csv"
        # The JSON is what the next sweep gets diffed against, so every field of
        # the record has to be in it, not just the ones the table shows.
        assert set(payload["records"][0]) == set(sweep.RunRecord.__dataclass_fields__)

        markdown = md_path.read_text()
        assert "1 of 1 datasets completed" in markdown
        assert "$0.0012" in markdown

    def test_warnings_are_deduplicated_per_dataset(self, tmp_path):
        record = sweep.RunRecord(dataset="a.csv", warnings=["same thing", "same thing"])

        _, md_path = sweep.write_outputs([record], tmp_path / "out")

        assert md_path.read_text().count("same thing") == 1


@pytest.mark.parametrize("missing", ["no such dir", "nope.csv"])
def test_paths_that_match_nothing_yield_no_cases(missing):
    assert sweep.discover_cases([Path(missing)], {}) == []
