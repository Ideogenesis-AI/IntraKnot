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


"""Tests for src/intraknot/cli.py — CLI commands and helper functions."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from intraknot.cli import _is_intraknot_source_project, main
from intraknot.status import FailureReason, MainStatus, RunState, write_status


# ---------------------------------------------------------------------------
# _is_intraknot_source_project
# ---------------------------------------------------------------------------

class TestIsIntraknotSourceProject:
    def test_true_when_name_matches(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "intraknot"\n')
        assert _is_intraknot_source_project(tmp_path) is True

    def test_false_when_name_differs(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "myproject"\n')
        assert _is_intraknot_source_project(tmp_path) is False

    def test_false_when_no_pyproject(self, tmp_path):
        assert _is_intraknot_source_project(tmp_path) is False

    def test_false_on_malformed_toml(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("not valid toml {{{")
        assert _is_intraknot_source_project(tmp_path) is False

    def test_false_when_project_section_absent(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[build-system]\nrequires = []\n')
        assert _is_intraknot_source_project(tmp_path) is False


# ---------------------------------------------------------------------------
# iknot init — .gitignore behaviour
# ---------------------------------------------------------------------------

class TestCmdInit:
    def test_creates_gitignore_in_intraknot_project(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "pyproject.toml").write_text('[project]\nname = "intraknot"\n')
            result = runner.invoke(main, ["init"])
            assert result.exit_code == 0, result.output
            for subdir in ("campaigns", "runs", "notebooks", "configs"):
                assert (tmpdir / subdir / ".gitignore").exists(), (
                    f"{subdir}/.gitignore should exist in the IntraKnot source project"
                )

    def test_no_gitignore_in_user_project(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "pyproject.toml").write_text('[project]\nname = "myproject"\n')
            result = runner.invoke(main, ["init"])
            assert result.exit_code == 0, result.output
            for subdir in ("campaigns", "runs", "notebooks", "configs"):
                assert (tmpdir / subdir).is_dir(), f"{subdir}/ should be created"
                assert not (tmpdir / subdir / ".gitignore").exists(), (
                    f"{subdir}/.gitignore should NOT exist in a user project"
                )

    def test_no_gitignore_without_pyproject(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as tmpdir:
            tmpdir = Path(tmpdir)
            result = runner.invoke(main, ["init"])
            assert result.exit_code == 0, result.output
            assert not (tmpdir / "campaigns" / ".gitignore").exists()

    def test_creates_config_templates(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as tmpdir:
            tmpdir = Path(tmpdir)
            result = runner.invoke(main, ["init"])
            assert result.exit_code == 0, result.output
            assert (tmpdir / "configs" / "slurm.toml").exists()
            assert (tmpdir / "configs" / "paths.toml").exists()
            # machines.yaml is no longer created by init; cluster.yaml is
            # written by `iknot machine sync` instead.
            assert not (tmpdir / "configs" / "machines.yaml").exists()

    def test_appends_iknot_state_to_root_gitignore(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as tmpdir:
            tmpdir = Path(tmpdir)
            result = runner.invoke(main, ["init"])
            assert result.exit_code == 0, result.output
            assert ".iknot_state" in (tmpdir / ".gitignore").read_text()

    def test_does_not_duplicate_iknot_state_entry(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / ".gitignore").write_text(".iknot_state\n")
            runner.invoke(main, ["init"])
            text = (tmpdir / ".gitignore").read_text()
            assert text.count(".iknot_state") == 1

    def test_custom_roots_respected(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as tmpdir:
            tmpdir = Path(tmpdir)
            result = runner.invoke(
                main,
                ["init", "--campaigns-root", "camp", "--runs-root", "sim",
                 "--notebooks-root", "nb"],
            )
            assert result.exit_code == 0, result.output
            assert (tmpdir / "camp").is_dir()
            assert (tmpdir / "sim").is_dir()
            assert (tmpdir / "nb").is_dir()


# ---------------------------------------------------------------------------
# iknot run start
# ---------------------------------------------------------------------------

class TestRunStart:
    def _make_run_dir(self, base: Path, run_id: str = "my_run") -> Path:
        run_dir = base / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "algorithm").mkdir()
        (run_dir / "algorithm" / "run_dmrg.py").write_text("# dummy\n")
        # Provide a minimal slurm.toml so write_slurm_script can render the script.
        (run_dir / "slurm.toml").write_text(
            '[basic]\naccount = "acc"\n'
            '[main]\npartition = "cpu"\ntime = "01:00:00"\nmem = "4000"\n'
            'ntasks = 1\nnodes = 1\ncpus_per_task = 1\n'
            '[exec]\ntime = "00:30:00"\nmem = "2000"\nntasks = 1\n'
            'nodes = 1\ncpus_per_task = 1\n'
        )
        return run_dir

    def _machine(self):
        from intraknot.config import MachineConfig, PathsConfig
        return MachineConfig(paths=PathsConfig(command="python"))

    def test_invokes_slurm_script_directly(self, tmp_path):
        self._make_run_dir(tmp_path)
        runner = CliRunner()
        with patch("intraknot.cli._load_machine", return_value=self._machine()), \
             patch("intraknot.cli.subprocess.run") as mock_run:
            result = runner.invoke(
                main,
                ["run", "start", "my_run",
                 "--runs-root", str(tmp_path / "runs")],
            )
        assert result.exit_code == 0, result.output
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "sh"
        assert "submit.slurm" in cmd[1]

    def test_slurm_env_vars_stubbed(self, tmp_path):
        self._make_run_dir(tmp_path)
        runner = CliRunner()
        with patch("intraknot.cli._load_machine", return_value=self._machine()), \
             patch("intraknot.cli.subprocess.run") as mock_run:
            runner.invoke(
                main,
                ["run", "start", "my_run",
                 "--runs-root", str(tmp_path / "runs")],
            )
        _, kwargs = mock_run.call_args
        assert kwargs["env"]["SLURM_JOB_ID"] == "local"
        assert kwargs["env"]["SLURM_NODELIST"] == "localhost"

    def test_missing_run_dir_exits_nonzero(self, tmp_path):
        runner = CliRunner()
        with patch("intraknot.cli._load_machine", return_value=self._machine()):
            result = runner.invoke(
                main,
                ["run", "start", "nonexistent",
                 "--runs-root", str(tmp_path / "runs")],
            )
        assert result.exit_code != 0

    def test_runner_exit_code_propagated(self, tmp_path):
        import subprocess
        self._make_run_dir(tmp_path)
        runner = CliRunner()
        with patch("intraknot.cli._load_machine", return_value=self._machine()), \
             patch("intraknot.cli.subprocess.run",
                   side_effect=subprocess.CalledProcessError(2, "bash")):
            result = runner.invoke(
                main,
                ["run", "start", "my_run",
                 "--runs-root", str(tmp_path / "runs")],
            )
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# iknot status
# ---------------------------------------------------------------------------

class TestCmdStatus:
    def _write_main_status(self, runs_root: Path, run_id: str, **kwargs) -> Path:
        run_dir = runs_root / run_id
        (run_dir / "main").mkdir(parents=True)
        status = MainStatus(state=RunState.COMPLETED, **kwargs)
        write_status(run_dir / "main" / "status.json", status)
        return run_dir

    def test_shows_state_reason_restartable(self, tmp_path):
        runs_root = tmp_path / "runs"
        self._write_main_status(
            runs_root, "r1",
            current_attempt="attempt_01",
            reason=FailureReason.CONVERGED,
            restartable=False,
        )
        runner = CliRunner()
        result = runner.invoke(
            main, ["status", "r1", "--runs-root", str(runs_root)]
        )
        assert result.exit_code == 0, result.output
        assert "completed" in result.output
        assert "converged" in result.output
        assert "attempt_01" in result.output
        assert "False" in result.output

    def test_shows_hostname_when_set(self, tmp_path):
        runs_root = tmp_path / "runs"
        self._write_main_status(
            runs_root, "r2",
            hostname="node07.hpc",
        )
        runner = CliRunner()
        result = runner.invoke(
            main, ["status", "r2", "--runs-root", str(runs_root)]
        )
        assert "node07.hpc" in result.output

    def test_shows_dash_when_hostname_absent(self, tmp_path):
        runs_root = tmp_path / "runs"
        self._write_main_status(runs_root, "r3")
        runner = CliRunner()
        result = runner.invoke(
            main, ["status", "r3", "--runs-root", str(runs_root)]
        )
        assert "Hostname        : —" in result.output

    def test_missing_run_reports_no_status(self, tmp_path):
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            main, ["status", "ghost", "--runs-root", str(runs_root)]
        )
        assert result.exit_code == 0
        assert "No status found" in result.output


# ---------------------------------------------------------------------------
# iknot run start — scan exit code propagation
# ---------------------------------------------------------------------------

class TestRunStartScanExitCode:
    def _make_run_dir(self, base: Path, run_id: str) -> Path:
        run_dir = base / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "algorithm").mkdir()
        (run_dir / "algorithm" / "run_dmrg.py").write_text("# dummy\n")
        (run_dir / "slurm.toml").write_text(
            '[basic]\naccount = "acc"\n'
            '[main]\npartition = "cpu"\ntime = "01:00:00"\nmem = "4000"\n'
            'ntasks = 1\nnodes = 1\ncpus_per_task = 1\n'
            '[exec]\ntime = "00:30:00"\nmem = "2000"\nntasks = 1\n'
            'nodes = 1\ncpus_per_task = 1\n'
        )
        return run_dir

    def _machine(self):
        from intraknot.config import MachineConfig, PathsConfig
        return MachineConfig(paths=PathsConfig(command="python"))

    def _make_campaign(self, base: Path, run_ids: list) -> tuple:
        import csv
        campaigns_root = base / "campaigns"
        campaign_dir = campaigns_root / "c1"
        campaign_dir.mkdir(parents=True)
        (campaign_dir / "campaign.yaml").write_text("algorithm: dmrg\n")
        with open(campaign_dir / "runs.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "scan_id", "status"])
            for rid in run_ids:
                writer.writerow([rid, "scan1", "pending"])
        return campaigns_root, campaign_dir

    def test_scan_last_nonzero_code_propagated(self, tmp_path):
        import subprocess
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        campaigns_root, _ = self._make_campaign(tmp_path, ["r1", "r2"])
        self._make_run_dir(runs_root, "r1")
        self._make_run_dir(runs_root, "r2")

        call_count = [0]

        def fake_run(cmd, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise subprocess.CalledProcessError(3, "bash")

        runner = CliRunner()
        with patch("intraknot.cli._load_machine", return_value=self._machine()), \
             patch("intraknot.cli.subprocess.run", side_effect=fake_run):
            result = runner.invoke(
                main,
                ["run", "start", "--scan", "scan1",
                 "--campaign", "c1",
                 "--campaigns-root", str(campaigns_root),
                 "--runs-root", str(runs_root)],
            )
        assert result.exit_code == 3


# ---------------------------------------------------------------------------
# iknot run delete — all-campaign sweep with --delete-dir
# ---------------------------------------------------------------------------

class TestRunDeleteAllCampaigns:
    def _make_campaigns_with_run(self, base: Path, run_id: str) -> tuple:
        import csv
        campaigns_root = base / "campaigns"
        runs_root = base / "runs"
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True)
        # Register the run in two different campaigns.
        for cid in ("c1", "c2"):
            campaign_dir = campaigns_root / cid
            campaign_dir.mkdir(parents=True)
            with open(campaign_dir / "runs.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["run_id", "scan_id", "status"])
                writer.writerow([run_id, "", "pending"])
        return campaigns_root, runs_root

    def test_delete_dir_removes_from_all_campaigns(self, tmp_path):
        import csv
        run_id = "my_run"
        campaigns_root, runs_root = self._make_campaigns_with_run(tmp_path, run_id)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "delete", run_id,
             "--delete-dir", "--yes",
             "--campaigns-root", str(campaigns_root),
             "--runs-root", str(runs_root)],
        )
        assert result.exit_code == 0, result.output
        # Run directory must be gone.
        assert not (runs_root / run_id).exists()
        # Run must be absent from both campaign CSVs.
        for cid in ("c1", "c2"):
            with open(campaigns_root / cid / "runs.csv", newline="") as f:
                ids = [r[0] for r in csv.reader(f) if r and r[0] != "run_id"]
            assert run_id not in ids
        # Output must name each cleaned campaign.
        assert "c1" in result.output
        assert "c2" in result.output

    def test_delete_without_delete_dir_removes_from_one_campaign(self, tmp_path):
        import csv
        run_id = "my_run"
        campaigns_root, runs_root = self._make_campaigns_with_run(tmp_path, run_id)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "delete", run_id,
             "--campaign", "c1",
             "--campaigns-root", str(campaigns_root),
             "--runs-root", str(runs_root)],
        )
        assert result.exit_code == 0, result.output
        # Run directory must still exist.
        assert (runs_root / run_id).exists()
        # Run must be gone from c1 but still in c2.
        with open(campaigns_root / "c1" / "runs.csv", newline="") as f:
            ids_c1 = [r[0] for r in csv.reader(f) if r and r[0] != "run_id"]
        with open(campaigns_root / "c2" / "runs.csv", newline="") as f:
            ids_c2 = [r[0] for r in csv.reader(f) if r and r[0] != "run_id"]
        assert run_id not in ids_c1
        assert run_id in ids_c2
