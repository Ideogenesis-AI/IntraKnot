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


"""Tests for src/intraknot/resume.py."""

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from intraknot.resume import find_resumable_runs, is_resumable, resume_campaign, resume_run
from intraknot.status import FailureReason, RunState
from helpers import make_machine as _make_machine, make_run_dir as _make_run_dir


class TestIsResumable:
    def test_failed_restartable_is_true(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "r1", RunState.FAILED, restartable=True)
        assert is_resumable(run_dir) is True

    def test_failed_not_restartable_is_false(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "r2", RunState.FAILED, restartable=False)
        assert is_resumable(run_dir) is False

    def test_completed_is_false(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "r3", RunState.COMPLETED, restartable=False)
        assert is_resumable(run_dir) is False

    def test_running_is_false(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "r4", RunState.RUNNING, restartable=True)
        assert is_resumable(run_dir) is False

    def test_missing_status_file_returns_false(self, tmp_path):
        run_dir = tmp_path / "empty"
        run_dir.mkdir()
        assert is_resumable(run_dir) is False

    def test_not_converged_is_resumable(self, tmp_path):
        run_dir = _make_run_dir(
            tmp_path, "r5", RunState.FAILED,
            restartable=True, reason=FailureReason.NOT_CONVERGED,
        )
        assert is_resumable(run_dir) is True


class TestResumeRun:
    def test_raises_when_not_resumable(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "r1", RunState.COMPLETED, restartable=False)
        with pytest.raises(ValueError, match="not resumable"):
            resume_run(run_dir, _make_machine(), submit=False)

    def test_returns_predicted_attempt_path_without_submit(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "r1", RunState.FAILED, restartable=True)
        attempt = resume_run(run_dir, _make_machine(), submit=False)
        # make_run_dir already created attempt_01, so the predicted next attempt is _02.
        assert attempt.name == "attempt_02"
        # The directory must NOT be pre-created; the runner creates it when the job starts.
        assert not attempt.exists()

    def test_calls_sbatch_when_submit(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "r1", RunState.FAILED, restartable=True)
        with patch("intraknot.resume.submit_job") as mock_submit:
            mock_submit.return_value = "99999"
            resume_run(run_dir, _make_machine(), submit=True)
            mock_submit.assert_called_once_with(run_dir)


class TestResumeCampaign:
    def _make_csv(self, campaign_dir: Path, run_ids: list[str]) -> None:
        campaign_dir.mkdir(parents=True, exist_ok=True)
        with open(campaign_dir / "runs.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "status"])
            for rid in run_ids:
                writer.writerow([rid, "failed"])

    def test_resumes_all_resumable(self, tmp_path):
        campaign_dir = tmp_path / "c1"
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        self._make_csv(campaign_dir, ["r1", "r2"])
        _make_run_dir(runs_root, "r1", RunState.FAILED, restartable=True)
        _make_run_dir(runs_root, "r2", RunState.FAILED, restartable=True)

        with patch("intraknot.resume.submit_job") as mock_sub:
            mock_sub.return_value = "42"
            resumed = resume_campaign(campaign_dir, runs_root, _make_machine(), submit=True)
        assert len(resumed) == 2

    def test_skips_non_resumable(self, tmp_path):
        campaign_dir = tmp_path / "c2"
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        self._make_csv(campaign_dir, ["r1", "r2"])
        _make_run_dir(runs_root, "r1", RunState.FAILED, restartable=True)
        _make_run_dir(runs_root, "r2", RunState.COMPLETED, restartable=False)

        resumed = resume_campaign(campaign_dir, runs_root, _make_machine(), submit=False)
        assert len(resumed) == 1

    def test_empty_campaign_returns_empty_list(self, tmp_path):
        campaign_dir = tmp_path / "empty"
        campaign_dir.mkdir()
        with open(campaign_dir / "runs.csv", "w") as f:
            csv.writer(f).writerow(["run_id", "status"])
        resumed = resume_campaign(campaign_dir, tmp_path, _make_machine(), submit=False)
        assert resumed == []


class TestFindResumableRuns:
    def test_returns_run_ids(self, tmp_path):
        campaign_dir = tmp_path / "c1"
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        campaign_dir.mkdir()
        with open(campaign_dir / "runs.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run_id", "status"])
            w.writerow(["r1", "failed"])
            w.writerow(["r2", "completed"])
        _make_run_dir(runs_root, "r1", RunState.FAILED, restartable=True)
        _make_run_dir(runs_root, "r2", RunState.COMPLETED, restartable=False)

        ids = find_resumable_runs(campaign_dir, runs_root)
        assert ids == ["r1"]
