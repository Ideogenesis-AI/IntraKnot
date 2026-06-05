# Copyright (C) 2026 Changkai Zhang.
#
# This file is part of IntraKnot.
#
# IntraKnot is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# IntraKnot is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with IntraKnot. If not, see <https://www.gnu.org/licenses/>.


"""Tests for src/intraknot/collect.py."""

import csv as _csv
from pathlib import Path

import pytest

from intraknot.collect import collect_campaign, collect_run
from intraknot.status import MainStatus, RunState, write_status
from helpers import make_run_dir as _make_run_dir_full


def _make_run_dir(tmp_path: Path, run_id: str, state: RunState, energy: float = -1.23) -> Path:
    """Thin wrapper around the shared `make_run_dir` for collect-specific use."""
    return _make_run_dir_full(tmp_path, run_id, state=state, energy=energy)


class TestCollectRun:
    def test_collects_completed_run(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run01", RunState.COMPLETED, energy=-42.5)
        summary = collect_run(run_dir)
        assert summary["run_id"] == "run01"
        assert summary["state"] == "completed"
        assert summary["energy"] == pytest.approx(-42.5)

    def test_writes_summary_files(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run02", RunState.FAILED)
        collect_run(run_dir)
        assert (run_dir / "summary" / "status.json").exists()
        assert (run_dir / "summary" / "info.json").exists()

    def test_missing_observables_does_not_crash(self, tmp_path):
        run_dir = tmp_path / "run03"
        (run_dir / "main" / "attempts" / "attempt_01").mkdir(parents=True)
        (run_dir / "summary").mkdir()
        write_status(
            run_dir / "main" / "status.json",
            MainStatus(state=RunState.RUNNING, current_attempt="attempt_01"),
        )
        summary = collect_run(run_dir)
        assert summary["state"] == "running"

    def test_no_current_attempt(self, tmp_path):
        run_dir = tmp_path / "run04"
        (run_dir / "main").mkdir(parents=True)
        (run_dir / "summary").mkdir()
        write_status(run_dir / "main" / "status.json", MainStatus(state=RunState.PENDING))
        summary = collect_run(run_dir)
        assert summary["state"] == "pending"


class TestCollectCampaign:
    def _make_runs_csv(self, campaign_dir: Path, run_ids: list[str]) -> None:
        import csv
        campaign_dir.mkdir(parents=True, exist_ok=True)
        with open(campaign_dir / "runs.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "status"])
            for rid in run_ids:
                writer.writerow([rid, "pending"])

    def test_collects_all_runs(self, tmp_path):
        campaign_dir = tmp_path / "campaigns" / "c1"
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        self._make_runs_csv(campaign_dir, ["r1", "r2"])
        _make_run_dir(runs_root, "r1", RunState.COMPLETED)
        _make_run_dir(runs_root, "r2", RunState.FAILED)

        summaries = collect_campaign(campaign_dir, runs_root)
        assert len(summaries) == 2
        states = {s["run_id"]: s["state"] for s in summaries}
        assert states["r1"] == "completed"
        assert states["r2"] == "failed"

    def test_updates_runs_csv(self, tmp_path):
        import csv as _csv
        campaign_dir = tmp_path / "campaigns" / "c2"
        runs_root = tmp_path / "runs2"
        runs_root.mkdir()
        self._make_runs_csv(campaign_dir, ["r1"])
        _make_run_dir(runs_root, "r1", RunState.COMPLETED)

        collect_campaign(campaign_dir, runs_root)

        with open(campaign_dir / "runs.csv") as f:
            rows = list(_csv.DictReader(f))
        assert rows[0]["status"] == "completed"

    def test_missing_run_dir_is_reported(self, tmp_path):
        campaign_dir = tmp_path / "campaigns" / "c3"
        runs_root = tmp_path / "runs3"
        runs_root.mkdir()
        self._make_runs_csv(campaign_dir, ["ghost"])

        summaries = collect_campaign(campaign_dir, runs_root)
        assert summaries[0]["state"] == "missing"

    def test_empty_csv_returns_empty_list(self, tmp_path):
        campaign_dir = tmp_path / "campaigns" / "c4"
        campaign_dir.mkdir(parents=True)
        import csv as _csv
        with open(campaign_dir / "runs.csv", "w") as f:
            _csv.writer(f).writerow(["run_id", "status"])

        summaries = collect_campaign(campaign_dir, tmp_path / "runs4")
        assert summaries == []
