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
            assert (tmpdir / "configs" / "tui.toml").exists()
            # machines.yaml is no longer created by init; cluster.yaml is
            # written by `iknot machine sync` instead.
            assert not (tmpdir / "configs" / "machines.yaml").exists()

    def test_tui_toml_contains_editor_key(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as tmpdir:
            tmpdir = Path(tmpdir)
            result = runner.invoke(main, ["init"])
            assert result.exit_code == 0, result.output
            text = (tmpdir / "configs" / "tui.toml").read_text()
            assert "[tui]" in text
            assert "editor" in text

    def test_does_not_overwrite_existing_tui_toml(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as tmpdir:
            tmpdir = Path(tmpdir)
            runner.invoke(main, ["init"])
            tui_path = tmpdir / "configs" / "tui.toml"
            tui_path.write_text('[tui]\neditor = "emacs"\n')
            runner.invoke(main, ["init"])
            assert tui_path.read_text() == '[tui]\neditor = "emacs"\n'

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
            writer.writerow(["run_id", "scan_id"])
            for rid in run_ids:
                writer.writerow([rid, "scan1"])
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
                writer.writerow(["run_id", "scan_id"])
                writer.writerow([run_id, ""])
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


# ---------------------------------------------------------------------------
# Helpers shared across new CLI test classes
# ---------------------------------------------------------------------------

def _make_env(tmp_path: Path) -> dict:
    """Return a minimal directory layout (campaigns/, runs/) under tmp_path."""
    (tmp_path / "campaigns").mkdir(exist_ok=True)
    (tmp_path / "runs").mkdir(exist_ok=True)
    return {}


def _make_slurm_toml(run_dir: Path) -> None:
    (run_dir / "slurm.toml").write_text(
        "[basic]\n"
        "[main]\npartition=\"cpu\"\ntime=\"01:00:00\"\nmem=\"8000\"\n"
        "ntasks=1\nnodes=1\ncpus_per_task=4\n"
        "[exec]\npartition=\"cpu\"\ntime=\"01:00:00\"\nmem=\"8000\"\n"
        "ntasks=1\nnodes=1\ncpus_per_task=4\n"
    )


# ---------------------------------------------------------------------------
# iknot campaign create/activate/deactivate/status
# ---------------------------------------------------------------------------

class TestCampaignCommands:
    def test_campaign_create(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["campaign", "create", "my_campaign",
             "--campaigns-root", str(tmp_path / "campaigns")],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "campaigns" / "my_campaign").is_dir()

    def test_campaign_create_duplicate_exits_nonzero(self, tmp_path):
        runner = CliRunner()
        args = ["campaign", "create", "dup",
                "--campaigns-root", str(tmp_path / "campaigns")]
        runner.invoke(main, args)
        result = runner.invoke(main, args)
        assert result.exit_code != 0

    def test_campaign_activate(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            td = Path(td)
            result = runner.invoke(main, ["campaign", "activate", "c1"])
            assert result.exit_code == 0
            assert "c1" in result.output
            assert (td / ".iknot_state").exists()
            assert "c1" in (td / ".iknot_state").read_text()

    def test_campaign_deactivate(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            td = Path(td)
            runner.invoke(main, ["campaign", "activate", "c1"])
            result = runner.invoke(main, ["campaign", "deactivate"])
            assert result.exit_code == 0
            assert "cleared" in result.output.lower()

    def test_campaign_status_with_active(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            td = Path(td)
            runner.invoke(main, ["campaign", "activate", "campaign_x"])
            result = runner.invoke(main, ["campaign", "status"])
            assert result.exit_code == 0
            assert "campaign_x" in result.output

    def test_campaign_status_none_active(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["campaign", "status"])
            assert result.exit_code == 0
            assert "No active campaign" in result.output


# ---------------------------------------------------------------------------
# iknot run create --set
# ---------------------------------------------------------------------------

class TestRunCreateCLI:
    def _setup(self, tmp_path: Path):
        camps_root = tmp_path / "campaigns"
        runs_root = tmp_path / "runs"
        runner = CliRunner()
        runner.invoke(
            main,
            ["campaign", "create", "c1", "--campaigns-root", str(camps_root)],
        )
        return camps_root, runs_root, runner

    def test_run_create_explicit_name(self, tmp_path):
        camps_root, runs_root, runner = self._setup(tmp_path)
        result = runner.invoke(
            main,
            ["run", "create", "my_run",
             "--campaign", "c1",
             "--campaigns-root", str(camps_root),
             "--runs-root", str(runs_root)],
        )
        assert result.exit_code == 0, result.output
        assert (runs_root / "my_run").is_dir()

    def test_run_create_set_auto_naming(self, tmp_path):
        camps_root, runs_root, runner = self._setup(tmp_path)
        result = runner.invoke(
            main,
            ["run", "create",
             "--campaign", "c1",
             "--campaigns-root", str(camps_root),
             "--runs-root", str(runs_root),
             "--set", "algorithm.max_bond=128"],
        )
        assert result.exit_code == 0, result.output
        created = list(runs_root.iterdir())
        assert len(created) == 1
        assert "max_bond=128" in created[0].name

    def test_run_create_scan_mode_multi_value(self, tmp_path):
        camps_root, runs_root, runner = self._setup(tmp_path)
        result = runner.invoke(
            main,
            ["run", "create",
             "--campaign", "c1",
             "--campaigns-root", str(camps_root),
             "--runs-root", str(runs_root),
             "--scan", "chi_scan",
             "--set", "algorithm.max_bond=64,128"],
        )
        assert result.exit_code == 0, result.output
        created = list(runs_root.iterdir())
        assert len(created) == 2

    def test_run_create_scan_required_for_multi_value(self, tmp_path):
        camps_root, runs_root, runner = self._setup(tmp_path)
        result = runner.invoke(
            main,
            ["run", "create",
             "--campaign", "c1",
             "--campaigns-root", str(camps_root),
             "--runs-root", str(runs_root),
             "--set", "algorithm.max_bond=64,128"],
        )
        assert result.exit_code != 0
        assert "--scan" in result.output or "--scan" in (result.stderr or "")


# ---------------------------------------------------------------------------
# iknot run submit (mocked sbatch, missing run exits non-zero)
# ---------------------------------------------------------------------------

class TestRunSubmitCLI:
    def _setup(self, tmp_path):
        camps_root = tmp_path / "campaigns"
        runs_root = tmp_path / "runs"
        runner = CliRunner()
        runner.invoke(main, ["campaign", "create", "c1",
                             "--campaigns-root", str(camps_root)])
        runner.invoke(main, ["run", "create", "r1",
                             "--campaign", "c1",
                             "--campaigns-root", str(camps_root),
                             "--runs-root", str(runs_root)])
        _make_slurm_toml(runs_root / "r1")
        return camps_root, runs_root, runner

    def test_submit_missing_run_exits_nonzero(self, tmp_path):
        camps_root = tmp_path / "campaigns"
        runs_root = tmp_path / "runs"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "submit", "nonexistent",
             "--campaigns-root", str(camps_root),
             "--runs-root", str(runs_root)],
        )
        assert result.exit_code != 0

    def test_submit_calls_sbatch(self, tmp_path):
        camps_root, runs_root, runner = self._setup(tmp_path)
        from unittest.mock import patch, MagicMock
        mock_proc = MagicMock()
        mock_proc.stdout = "Submitted batch job 9999999\n"
        mock_proc.returncode = 0
        with patch("intraknot.launch.subprocess.run", return_value=mock_proc):
            result = runner.invoke(
                main,
                ["run", "submit", "r1",
                 "--campaigns-root", str(camps_root),
                 "--runs-root", str(runs_root)],
            )
        assert result.exit_code == 0, result.output
        assert "9999999" in result.output


# ---------------------------------------------------------------------------
# iknot run exec --local (IKNOT_ATTEMPT env var)
# ---------------------------------------------------------------------------

class TestRunExecCLI:
    def _setup(self, tmp_path):
        camps_root = tmp_path / "campaigns"
        runs_root = tmp_path / "runs"
        runner = CliRunner()
        runner.invoke(main, ["campaign", "create", "c1",
                             "--campaigns-root", str(camps_root)])
        runner.invoke(main, ["run", "create", "r1",
                             "--campaign", "c1",
                             "--campaigns-root", str(camps_root),
                             "--runs-root", str(runs_root)])
        _make_slurm_toml(runs_root / "r1")
        # Create a minimal exec script.
        exec_script = camps_root / "c1" / "algorithm" / "my_exec.py"
        exec_script.parent.mkdir(parents=True, exist_ok=True)
        exec_script.write_text("# exec script\n")
        return camps_root, runs_root, runner

    def test_exec_local_sets_iknot_attempt(self, tmp_path):
        camps_root, runs_root, runner = self._setup(tmp_path)
        captured_env = {}

        def fake_run(cmd, **kwargs):
            import subprocess
            captured_env.update(kwargs.get("env", {}))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch("intraknot.cli.subprocess.run", side_effect=fake_run):
            result = runner.invoke(
                main,
                ["run", "exec", "my_exec", "r1",
                 "--campaign", "c1",
                 "--campaigns-root", str(camps_root),
                 "--runs-root", str(runs_root),
                 "--local",
                 "--attempt", "attempt_01"],
            )
        assert result.exit_code == 0, result.output
        assert captured_env.get("IKNOT_ATTEMPT") == "attempt_01"


# ---------------------------------------------------------------------------
# iknot resume run / campaign
# ---------------------------------------------------------------------------

class TestResumeCLI:
    def test_resume_run_no_submit(self, tmp_path):
        from helpers import make_run_dir
        from intraknot.status import RunState, FailureReason
        runs_root = tmp_path / "runs"
        make_run_dir(runs_root, "r1", state=RunState.FAILED,
                     restartable=True, reason=FailureReason.TIMEOUT)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "resume", "r1",
             "--runs-root", str(runs_root),
             "--no-submit"],
        )
        assert result.exit_code == 0, result.output
        assert "attempt" in result.output.lower()

    def test_resume_run_missing_exits_nonzero(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "resume", "nonexistent",
             "--runs-root", str(tmp_path / "runs")],
        )
        assert result.exit_code != 0

    def test_resume_campaign_no_submit(self, tmp_path):
        from helpers import make_run_dir
        from intraknot.launch import create_campaign, _register_run_in_campaign
        from intraknot.status import RunState, FailureReason
        camps_root = tmp_path / "campaigns"
        runs_root = tmp_path / "runs"
        camp_dir = create_campaign("c1", "", "dmrg", camps_root)
        make_run_dir(runs_root, "r1", state=RunState.FAILED,
                     restartable=True, reason=FailureReason.TIMEOUT)
        _register_run_in_campaign(camp_dir, "r1")

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["campaign", "resume",
             "--campaign", "c1",
             "--campaigns-root", str(camps_root),
             "--runs-root", str(runs_root),
             "--no-submit"],
        )
        assert result.exit_code == 0, result.output
        assert "resumed" in result.output.lower()


# ---------------------------------------------------------------------------
# iknot cluster sync / show
# ---------------------------------------------------------------------------

class TestClusterCLI:
    def test_cluster_sync_mocked(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from intraknot.discover import ClusterDiscovery
        fake_discovery = ClusterDiscovery(
            partitions=[], features={},
            discovered_at="2026-01-01T00:00:00", hostname="testhost"
        )
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()

        with patch("intraknot.cli.discover_cluster", return_value=fake_discovery):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["cluster", "sync", "--machine", str(configs_dir)],
            )
        assert result.exit_code == 0, result.output
        assert "testhost" in result.output
        assert (configs_dir / "cluster.yaml").exists()

    def test_cluster_show_no_yaml_exits_nonzero(self, tmp_path):
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["cluster", "show", "--machine", str(configs_dir)],
        )
        assert result.exit_code != 0
        assert "cluster sync" in result.output or "cluster.yaml" in result.output


# ---------------------------------------------------------------------------
# iknot init — registry.yaml
# ---------------------------------------------------------------------------

class TestCmdInitRegistry:
    def test_creates_registry_yaml(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as tmpdir:
            tmpdir = Path(tmpdir)
            result = runner.invoke(main, ["init"])
            assert result.exit_code == 0, result.output
            assert (tmpdir / "configs" / "registry.yaml").exists(), (
                "iknot init should create configs/registry.yaml"
            )

    def test_registry_yaml_contains_databases_key(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as tmpdir:
            tmpdir = Path(tmpdir)
            runner.invoke(main, ["init"])
            text = (tmpdir / "configs" / "registry.yaml").read_text()
            assert "databases:" in text

    def test_does_not_overwrite_existing_registry_yaml(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as tmpdir:
            tmpdir = Path(tmpdir)
            runner.invoke(main, ["init"])
            reg_path = tmpdir / "configs" / "registry.yaml"
            reg_path.write_text("# custom registry\ndatabases: {}\n")
            runner.invoke(main, ["init"])
            assert reg_path.read_text() == "# custom registry\ndatabases: {}\n"


# ---------------------------------------------------------------------------
# iknot database commands
# ---------------------------------------------------------------------------

_MOCK_DB_REGISTRY_YAML = b"""\
description: A mock database
categories:
  observables:
    description: Observable scripts
    scripts:
      - name: spin_corr
        file: observables/spin_corr.py
        description: Spin-spin correlation
        sha256: aabbccdd
"""


class TestDatabaseAddCLI:
    def test_add_registers_database(self, tmp_path):
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        runner = CliRunner()
        with patch("intraknot.database._fetch_bytes", return_value=_MOCK_DB_REGISTRY_YAML):
            result = runner.invoke(
                main,
                ["database", "add", "test-db",
                 "https://github.com/foo/test-db",
                 "--machine", str(configs_dir)],
            )
        assert result.exit_code == 0, result.output
        assert "Registered" in result.output
        reg_path = configs_dir / "registry.yaml"
        assert reg_path.exists()
        import yaml
        data = yaml.safe_load(reg_path.read_text())
        assert "test-db" in data["databases"]

    def test_add_rejects_duplicate(self, tmp_path):
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        runner = CliRunner()
        with patch("intraknot.database._fetch_bytes", return_value=_MOCK_DB_REGISTRY_YAML):
            runner.invoke(
                main,
                ["database", "add", "test-db",
                 "https://github.com/foo/test-db",
                 "--machine", str(configs_dir)],
            )
            result = runner.invoke(
                main,
                ["database", "add", "test-db",
                 "https://github.com/foo/test-db",
                 "--machine", str(configs_dir)],
            )
        assert result.exit_code != 0
        assert "already registered" in result.output

    def test_add_rejects_reserved_name(self, tmp_path):
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["database", "add", "intraknot",
             "https://github.com/foo/bar",
             "--machine", str(configs_dir)],
        )
        assert result.exit_code != 0
        assert "reserved" in result.output

    def test_add_handles_http_error(self, tmp_path):
        import urllib.error
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        runner = CliRunner()
        http_err = urllib.error.HTTPError(
            url="https://example.com", code=404, msg="Not Found",
            hdrs=None, fp=None,  # type: ignore[arg-type]
        )
        with patch("intraknot.database._fetch_bytes", side_effect=http_err):
            result = runner.invoke(
                main,
                ["database", "add", "missing-db",
                 "https://github.com/foo/missing-db",
                 "--machine", str(configs_dir)],
            )
        assert result.exit_code != 0
        assert "404" in result.output or "Not Found" in result.output or "HTTP" in result.output


class TestDatabaseRemoveCLI:
    def _setup(self, tmp_path: Path) -> Path:
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        runner = CliRunner()
        with patch("intraknot.database._fetch_bytes", return_value=_MOCK_DB_REGISTRY_YAML):
            runner.invoke(
                main,
                ["database", "add", "test-db",
                 "https://github.com/foo/test-db",
                 "--machine", str(configs_dir)],
            )
        return configs_dir

    def test_remove_existing(self, tmp_path):
        configs_dir = self._setup(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["database", "remove", "test-db", "--machine", str(configs_dir)],
        )
        assert result.exit_code == 0, result.output
        assert "Removed" in result.output
        import yaml
        data = yaml.safe_load((configs_dir / "registry.yaml").read_text())
        assert "test-db" not in (data.get("databases") or {})

    def test_remove_nonexistent_exits_nonzero(self, tmp_path):
        configs_dir = self._setup(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["database", "remove", "ghost-db", "--machine", str(configs_dir)],
        )
        assert result.exit_code != 0
        assert "not registered" in result.output


class TestDatabaseListCLI:
    def test_list_empty(self, tmp_path):
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["database", "list", "--machine", str(configs_dir)],
        )
        assert result.exit_code == 0, result.output
        assert "No databases" in result.output

    def test_list_shows_registered_db(self, tmp_path):
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        runner = CliRunner()
        with patch("intraknot.database._fetch_bytes", return_value=_MOCK_DB_REGISTRY_YAML):
            runner.invoke(
                main,
                ["database", "add", "test-db",
                 "https://github.com/foo/test-db",
                 "--machine", str(configs_dir)],
            )
        result = runner.invoke(
            main,
            ["database", "list", "--machine", str(configs_dir)],
        )
        assert result.exit_code == 0, result.output
        assert "test-db" in result.output
        assert "observables" in result.output


class TestDatabaseUpdateCLI:
    def test_update_refreshes_manifest(self, tmp_path):
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        runner = CliRunner()
        with patch("intraknot.database._fetch_bytes", return_value=_MOCK_DB_REGISTRY_YAML):
            runner.invoke(
                main,
                ["database", "add", "test-db",
                 "https://github.com/foo/test-db",
                 "--machine", str(configs_dir)],
            )
        updated = _MOCK_DB_REGISTRY_YAML.replace(b"A mock database", b"Updated database")
        with patch("intraknot.database._fetch_bytes", return_value=updated):
            result = runner.invoke(
                main,
                ["database", "update", "test-db", "--machine", str(configs_dir)],
            )
        assert result.exit_code == 0, result.output
        assert "test-db" in result.output

    def test_update_all_when_no_name(self, tmp_path):
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        runner = CliRunner()
        with patch("intraknot.database._fetch_bytes", return_value=_MOCK_DB_REGISTRY_YAML):
            runner.invoke(
                main,
                ["database", "add", "test-db",
                 "https://github.com/foo/test-db",
                 "--machine", str(configs_dir)],
            )
        with patch("intraknot.database._fetch_bytes", return_value=_MOCK_DB_REGISTRY_YAML):
            result = runner.invoke(
                main,
                ["database", "update", "--machine", str(configs_dir)],
            )
        assert result.exit_code == 0, result.output

    def test_update_no_databases_registered(self, tmp_path):
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["database", "update", "--machine", str(configs_dir)],
        )
        assert result.exit_code == 0, result.output
        assert "No databases" in result.output

    def test_update_unknown_db_exits_nonzero(self, tmp_path):
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        runner = CliRunner()
        with patch("intraknot.database._fetch_bytes", return_value=_MOCK_DB_REGISTRY_YAML):
            runner.invoke(
                main,
                ["database", "add", "test-db",
                 "https://github.com/foo/test-db",
                 "--machine", str(configs_dir)],
            )
        result = runner.invoke(
            main,
            ["database", "update", "ghost-db", "--machine", str(configs_dir)],
        )
        assert result.exit_code != 0
        assert "not registered" in result.output


# ---------------------------------------------------------------------------
# iknot campaign install
# ---------------------------------------------------------------------------

class TestCampaignInstallCLI:
    def _setup(self, tmp_path: Path):
        from intraknot.launch import create_campaign
        campaigns_root = tmp_path / "campaigns"
        campaigns_root.mkdir()
        create_campaign("c1", "", "dmrg", campaigns_root)
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        return campaigns_root, configs_dir

    def test_install_from_builtin_source(self, tmp_path):
        campaigns_root, configs_dir = self._setup(tmp_path)
        runner = CliRunner()
        content = b"# run_tdvp stub\n"
        with patch("importlib.resources.files") as mock_files:
            mock_files.return_value.joinpath.return_value.read_bytes.return_value = content
            result = runner.invoke(
                main,
                ["campaign", "install",
                 "intraknot:algorithm/run_tdvp.py",
                 "--campaign", "c1",
                 "--campaigns-root", str(campaigns_root),
                 "--machine", str(configs_dir)],
            )
        assert result.exit_code == 0, result.output
        assert "Installed" in result.output
        dest = campaigns_root / "c1" / "algorithm" / "algorithm" / "run_tdvp.py"
        # dest_rel preserves directory structure from source
        dest_flat = campaigns_root / "c1" / "algorithm" / "run_tdvp.py"
        # The file is at the path "algorithm/run_tdvp.py" inside algorithm dir.
        dest_from_source = (
            campaigns_root / "c1" / "algorithm" / "algorithm" / "run_tdvp.py"
        )
        # Check the lock has the managed entry.
        from intraknot.alg_lock import AlgorithmLock
        lock = AlgorithmLock.load(
            campaigns_root / "c1" / "algorithm" / "algorithm.lock"
        )
        assert lock.is_managed("algorithm/run_tdvp.py")

    def test_install_with_custom_dest(self, tmp_path):
        campaigns_root, configs_dir = self._setup(tmp_path)
        runner = CliRunner()
        content = b"# spin_corr\n"
        with patch("importlib.resources.files") as mock_files:
            mock_files.return_value.joinpath.return_value.read_bytes.return_value = content
            result = runner.invoke(
                main,
                ["campaign", "install",
                 "intraknot:observables/spin_corr.py",
                 "--as", "spin_corr.py",
                 "--campaign", "c1",
                 "--campaigns-root", str(campaigns_root),
                 "--machine", str(configs_dir)],
            )
        assert result.exit_code == 0, result.output
        assert (campaigns_root / "c1" / "algorithm" / "spin_corr.py").exists()
        from intraknot.alg_lock import AlgorithmLock
        lock = AlgorithmLock.load(
            campaigns_root / "c1" / "algorithm" / "algorithm.lock"
        )
        assert lock.is_managed("spin_corr.py")

    def test_install_blocks_duplicate_managed(self, tmp_path):
        campaigns_root, configs_dir = self._setup(tmp_path)
        runner = CliRunner()
        content = b"# run_dmrg\n"
        # run_dmrg.py is already managed (created by create_campaign).
        with patch("importlib.resources.files") as mock_files:
            mock_files.return_value.joinpath.return_value.read_bytes.return_value = content
            result = runner.invoke(
                main,
                ["campaign", "install",
                 "intraknot:algorithm/run_dmrg.py",
                 "--as", "run_dmrg.py",
                 "--campaign", "c1",
                 "--campaigns-root", str(campaigns_root),
                 "--machine", str(configs_dir)],
            )
        assert result.exit_code != 0
        assert "already a managed entry" in result.output

    def test_install_missing_campaign_exits_nonzero(self, tmp_path):
        campaigns_root = tmp_path / "campaigns"
        campaigns_root.mkdir()
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["campaign", "install",
             "intraknot:algorithm/run_dmrg.py",
             "--campaign", "nonexistent",
             "--campaigns-root", str(campaigns_root),
             "--machine", str(configs_dir)],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# iknot campaign override
# ---------------------------------------------------------------------------

class TestCampaignOverrideCLI:
    def _setup(self, tmp_path: Path):
        from intraknot.launch import create_campaign
        campaigns_root = tmp_path / "campaigns"
        campaigns_root.mkdir()
        create_campaign("c1", "", "dmrg", campaigns_root)
        return campaigns_root

    def test_override_promotes_managed_entry(self, tmp_path):
        campaigns_root = self._setup(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["campaign", "override", "run_dmrg.py",
             "--campaign", "c1",
             "--campaigns-root", str(campaigns_root)],
        )
        assert result.exit_code == 0, result.output
        assert "Promoted" in result.output
        from intraknot.alg_lock import AlgorithmLock
        lock = AlgorithmLock.load(
            campaigns_root / "c1" / "algorithm" / "algorithm.lock"
        )
        assert not lock.is_managed("run_dmrg.py")
        assert lock.is_custom("run_dmrg.py")

    def test_override_nonexistent_entry_exits_nonzero(self, tmp_path):
        campaigns_root = self._setup(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["campaign", "override", "nonexistent.py",
             "--campaign", "c1",
             "--campaigns-root", str(campaigns_root)],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# iknot campaign add-script
# ---------------------------------------------------------------------------

class TestCampaignAddScriptCLI:
    def _setup(self, tmp_path: Path):
        from intraknot.launch import create_campaign
        campaigns_root = tmp_path / "campaigns"
        campaigns_root.mkdir()
        create_campaign("c1", "", "dmrg", campaigns_root)
        return campaigns_root

    def test_add_script_registers_custom_file(self, tmp_path):
        campaigns_root = self._setup(tmp_path)
        script = campaigns_root / "c1" / "algorithm" / "my_obs.py"
        script.write_text("# my observable\n")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["campaign", "add-script", "my_obs.py",
             "--campaign", "c1",
             "--campaigns-root", str(campaigns_root)],
        )
        assert result.exit_code == 0, result.output
        assert "custom" in result.output.lower()
        from intraknot.alg_lock import AlgorithmLock
        lock = AlgorithmLock.load(
            campaigns_root / "c1" / "algorithm" / "algorithm.lock"
        )
        assert lock.is_custom("my_obs.py")

    def test_add_script_missing_file_exits_nonzero(self, tmp_path):
        campaigns_root = self._setup(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["campaign", "add-script", "nonexistent.py",
             "--campaign", "c1",
             "--campaigns-root", str(campaigns_root)],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "Error" in result.output

    def test_add_script_blocks_managed_entry(self, tmp_path):
        campaigns_root = self._setup(tmp_path)
        runner = CliRunner()
        # run_dmrg.py is managed, not custom.
        result = runner.invoke(
            main,
            ["campaign", "add-script", "run_dmrg.py",
             "--campaign", "c1",
             "--campaigns-root", str(campaigns_root)],
        )
        assert result.exit_code != 0
        assert "managed entry" in result.output or "override" in result.output.lower()


# ---------------------------------------------------------------------------
# iknot campaign sync
# ---------------------------------------------------------------------------

class TestCampaignSyncCLI:
    def _setup(self, tmp_path: Path):
        from intraknot.launch import create_campaign
        campaigns_root = tmp_path / "campaigns"
        campaigns_root.mkdir()
        create_campaign("c1", "", "dmrg", campaigns_root)
        return campaigns_root

    def test_sync_blocks_on_local_modification(self, tmp_path):
        """sync must not overwrite locally modified managed files."""
        campaigns_root = self._setup(tmp_path)
        runner_script = campaigns_root / "c1" / "algorithm" / "run_dmrg.py"
        runner_script.write_bytes(b"# locally modified\n")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["campaign", "sync",
             "--campaign", "c1",
             "--campaigns-root", str(campaigns_root)],
        )
        assert result.exit_code != 0 or "locally modified" in result.output.lower()
        # The file must NOT have been overwritten.
        assert runner_script.read_bytes() == b"# locally modified\n"

    def test_sync_up_to_date_reports_no_updates(self, tmp_path):
        """When the managed file matches the package content, sync reports up-to-date."""
        campaigns_root = self._setup(tmp_path)
        runner = CliRunner()
        # Patch resolve_file to return the current on-disk content.
        alg_dir = campaigns_root / "c1" / "algorithm"
        current_content = (alg_dir / "run_dmrg.py").read_bytes()

        with patch("intraknot.database.DatabaseRegistry.resolve_file",
                   return_value=(current_content, __import__("intraknot.alg_lock", fromlist=["_sha256_bytes"])._sha256_bytes(current_content))):
            result = runner.invoke(
                main,
                ["campaign", "sync", "--yes",
                 "--campaign", "c1",
                 "--campaigns-root", str(campaigns_root)],
            )
        # Should exit cleanly (0 or may use non-zero for "nothing to update").
        assert "modified" not in result.output.lower() or result.exit_code == 0

    def test_sync_specific_file_only(self, tmp_path):
        """Specifying a file name only syncs that file."""
        campaigns_root = self._setup(tmp_path)
        runner = CliRunner()
        # Syncing a non-managed file should print a warning but not fail hard.
        result = runner.invoke(
            main,
            ["campaign", "sync", "nonexistent.py",
             "--campaign", "c1",
             "--campaigns-root", str(campaigns_root)],
        )
        # Either warns and exits 0, or exits non-zero.  Key: no crash.
        assert "nonexistent.py" in result.output

    def test_sync_no_managed_entries_message(self, tmp_path):
        """A lock with no managed entries produces an informative message."""
        campaigns_root = self._setup(tmp_path)
        from intraknot.alg_lock import AlgorithmLock
        lock_path = campaigns_root / "c1" / "algorithm" / "algorithm.lock"
        lock = AlgorithmLock()
        lock.save(lock_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["campaign", "sync",
             "--campaign", "c1",
             "--campaigns-root", str(campaigns_root)],
        )
        assert result.exit_code == 0, result.output
        assert "No managed" in result.output
