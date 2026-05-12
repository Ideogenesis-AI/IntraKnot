"""Tests for src/intraknot/cli.py — CLI commands and helper functions."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from intraknot.cli import _is_intraknot_source_project, main


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
            assert (tmpdir / "configs" / "machines.yaml").exists()

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
        (run_dir / "algorithm").mkdir(parents=True)
        (run_dir / "algorithm" / "run_dmrg.py").write_text("# dummy\n")
        return run_dir

    def test_invokes_start_run(self, tmp_path):
        run_dir = self._make_run_dir(tmp_path)
        runner = CliRunner()
        with patch("intraknot.launch.subprocess.run") as mock_run:
            result = runner.invoke(
                main,
                ["run", "start",
                 "--id", "my_run",
                 "--runs-root", str(tmp_path / "runs"),
                 "--python", "python"],
            )
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()

    def test_explicit_python_flag_overrides_machine(self, tmp_path):
        run_dir = self._make_run_dir(tmp_path)
        runner = CliRunner()
        with patch("intraknot.launch.subprocess.run") as mock_run:
            runner.invoke(
                main,
                ["run", "start",
                 "--id", "my_run",
                 "--runs-root", str(tmp_path / "runs"),
                 "--python", "uv run"],
            )
        cmd = mock_run.call_args[0][0]
        assert cmd[:2] == ["uv", "run"]

    def test_missing_run_dir_exits_nonzero(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "start",
             "--id", "nonexistent",
             "--runs-root", str(tmp_path / "runs"),
             "--python", "python"],
        )
        assert result.exit_code != 0

    def test_runner_exit_code_propagated(self, tmp_path):
        import subprocess
        self._make_run_dir(tmp_path)
        runner = CliRunner()
        with patch(
            "intraknot.launch.subprocess.run",
            side_effect=subprocess.CalledProcessError(2, "python"),
        ):
            result = runner.invoke(
                main,
                ["run", "start",
                 "--id", "my_run",
                 "--runs-root", str(tmp_path / "runs"),
                 "--python", "python"],
            )
        assert result.exit_code == 2
