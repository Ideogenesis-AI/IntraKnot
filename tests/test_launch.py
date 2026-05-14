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


"""Tests for src/intraknot/launch.py."""

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from intraknot.config import MachineConfig, PathsConfig
from intraknot.launch import (
    _dump_toml,
    _find_exec_script,
    _render_slurm_header,
    create_attempt,
    create_campaign,
    prepare_exec,
    update_current,
    write_exec_slurm_script,
    write_slurm_script,
)
from intraknot.status import MainStatus, RunState, read_status


# ---------------------------------------------------------------------------
# _dump_toml
# ---------------------------------------------------------------------------

class TestDumpToml:
    def test_scalar_values(self):
        text = _dump_toml({"key": "val", "n": 5, "flag": True})
        assert 'key = "val"' in text
        assert "n = 5" in text
        assert "flag = true" in text

    def test_nested_section(self):
        text = _dump_toml({"model": {"lx": 32}})
        assert "[model]" in text
        assert "lx = 32" in text

    def test_list_value(self):
        text = _dump_toml({"obs": ["energy", "entropy"]})
        assert '"energy"' in text
        assert '"entropy"' in text


# ---------------------------------------------------------------------------
# _render_slurm_header
# ---------------------------------------------------------------------------

class TestRenderSlurmHeader:
    def test_basic_directives(self):
        text = _render_slurm_header({
            "--job-name": "myjob",
            "--partition": "cluster",
        })
        assert "#SBATCH --job-name=myjob" in text
        assert "#SBATCH --partition=cluster" in text

    def test_empty_values_skipped(self):
        text = _render_slurm_header({
            "--job-name": "myjob",
            "--constraint": "",        # should be omitted
            "--mail-user": "",         # should be omitted
            "--partition": "cluster",
        })
        assert "--constraint" not in text
        assert "--mail-user" not in text
        assert "#SBATCH --job-name=myjob" in text

    def test_none_values_skipped(self):
        text = _render_slurm_header({"--account": None, "--time": "04:00:00"})
        assert "--account" not in text
        assert "#SBATCH --time=04:00:00" in text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_machine() -> MachineConfig:
    return MachineConfig(paths=PathsConfig(python="uv run"))


def _write_slurm_toml(run_dir: Path, **overrides) -> None:
    """Write a minimal slurm.toml into run_dir for use in tests."""
    fields = dict(
        account="testproject",
        partition="cpu",
        time="02:00:00",
        mem="8000",
        ntasks=1,
        nodes=1,
        cpus_per_task=4,
    )
    fields.update(overrides)
    content = (
        '[basic]\naccount = "{account}"\n\n'
        '[main]\npartition = "{partition}"\ntime = "{time}"\n'
        'mem = "{mem}"\nntasks = {ntasks}\nnodes = {nodes}\n'
        'cpus_per_task = {cpus_per_task}\n\n'
        '[exec]\npartition = "{partition}"\ntime = "01:00:00"\n'
        'mem = "4000"\nntasks = 1\nnodes = 1\ncpus_per_task = 2\n'
    ).format(**fields)
    (run_dir / "slurm.toml").write_text(content)


def _minimal_config_toml(tmp_path: Path) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(
        "[geometry]\nlattice = \"chain\"\nlx = 16\n"
        "[model]\ncategory = \"bosonic\"\nlabel = \"Heisenberg\"\n"
        "symmetry = \"U1\"\nspin = 0.5\nJ = 1.0\n"
    )
    return p


# ---------------------------------------------------------------------------
# create_campaign
# ---------------------------------------------------------------------------

class TestCreateCampaign:
    def test_creates_directory_structure(self, tmp_path):
        campaigns_root = tmp_path / "campaigns"
        campaigns_root.mkdir()
        camp_dir = create_campaign("heis_test", "Test", "dmrg", campaigns_root)
        assert camp_dir.is_dir()
        assert (camp_dir / "campaign.yaml").exists()
        assert (camp_dir / "defaults.toml").exists()
        assert (camp_dir / "runs.csv").exists()
        assert (camp_dir / "notes.md").exists()
        assert (camp_dir / "algorithm" / "run_dmrg.py").exists()

    def test_creates_slurm_toml(self, tmp_path):
        campaigns_root = tmp_path / "campaigns"
        campaigns_root.mkdir()
        camp_dir = create_campaign("c1", "", "dmrg", campaigns_root)
        assert (camp_dir / "slurm.toml").exists()
        text = (camp_dir / "slurm.toml").read_text()
        assert "[basic]" in text
        assert "[main]" in text
        assert "[exec]" in text

    def test_copies_slurm_toml_from_configs(self, tmp_path):
        campaigns_root = tmp_path / "campaigns"
        campaigns_root.mkdir()
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        (configs_dir / "slurm.toml").write_text(
            '[basic]\naccount = "myacct"\n[main]\ntime = "48:00:00"\n[exec]\ntime = "01:00:00"\n'
        )
        camp_dir = create_campaign(
            "c1", "", "dmrg", campaigns_root, configs_dir=configs_dir
        )
        text = (camp_dir / "slurm.toml").read_text()
        assert 'account = "myacct"' in text
        assert '48:00:00' in text

    def test_runs_csv_has_header(self, tmp_path):
        campaigns_root = tmp_path / "campaigns"
        campaigns_root.mkdir()
        camp_dir = create_campaign("c1", "", "dmrg", campaigns_root)
        with open(camp_dir / "runs.csv") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "run_id" in header
        assert "status" in header

    def test_raises_if_exists(self, tmp_path):
        campaigns_root = tmp_path / "campaigns"
        campaigns_root.mkdir()
        create_campaign("c1", "", "dmrg", campaigns_root)
        with pytest.raises(FileExistsError):
            create_campaign("c1", "", "dmrg", campaigns_root)


# ---------------------------------------------------------------------------
# create_attempt
# ---------------------------------------------------------------------------

class TestCreateAttempt:
    def test_creates_first_attempt(self, tmp_path):
        run_dir = tmp_path / "run01"
        (run_dir / "main" / "attempts").mkdir(parents=True)
        attempt_dir = create_attempt(run_dir)
        assert attempt_dir.name == "attempt_01"
        assert attempt_dir.is_dir()

    def test_increments_index(self, tmp_path):
        run_dir = tmp_path / "run01"
        (run_dir / "main" / "attempts" / "attempt_01").mkdir(parents=True)
        attempt_dir = create_attempt(run_dir)
        assert attempt_dir.name == "attempt_02"

    def test_updates_current(self, tmp_path):
        run_dir = tmp_path / "run01"
        (run_dir / "main" / "attempts").mkdir(parents=True)
        create_attempt(run_dir)
        current = run_dir / "main" / "current"
        txt = run_dir / "main" / "current.txt"
        assert current.is_symlink() or txt.exists()


# ---------------------------------------------------------------------------
# update_current
# ---------------------------------------------------------------------------

class TestUpdateCurrent:
    def test_creates_symlink(self, tmp_path):
        run_dir = tmp_path / "run01"
        (run_dir / "main" / "attempts" / "attempt_01").mkdir(parents=True)
        update_current(run_dir, "attempt_01")
        current = run_dir / "main" / "current"
        assert current.is_symlink() or (run_dir / "main" / "current.txt").exists()


# ---------------------------------------------------------------------------
# write_slurm_script
# ---------------------------------------------------------------------------

class TestWriteSlurmScript:
    def test_writes_to_main_dir(self, tmp_path):
        run_dir = tmp_path / "run01"
        (run_dir / "main" / "logs").mkdir(parents=True)
        _write_slurm_toml(run_dir)
        machine = _make_machine()
        script = write_slurm_script(run_dir, machine, "run01")
        assert script == run_dir / "main" / "submit.slurm"
        assert script.exists()

    def test_script_content(self, tmp_path):
        run_dir = tmp_path / "run01"
        (run_dir / "main" / "logs").mkdir(parents=True)
        _write_slurm_toml(run_dir)
        machine = _make_machine()
        script = write_slurm_script(run_dir, machine, "run01")
        text = script.read_text()
        assert "#!/usr/bin/env bash" in text
        assert "#SBATCH --job-name=run01" in text
        assert "uv run" in text
        assert str(run_dir) in text

    def test_reads_account_from_slurm_toml(self, tmp_path):
        run_dir = tmp_path / "myrun"
        (run_dir / "main" / "logs").mkdir(parents=True)
        _write_slurm_toml(run_dir, account="testproject", partition="cpu",
                          time="02:00:00")
        machine = _make_machine()
        script = write_slurm_script(run_dir, machine)
        text = script.read_text()
        assert "testproject" in text
        assert "cpu" in text
        assert "02:00:00" in text

    def test_creates_logs_dir_if_absent(self, tmp_path):
        run_dir = tmp_path / "run01"
        run_dir.mkdir()
        _write_slurm_toml(run_dir)
        machine = _make_machine()
        write_slurm_script(run_dir, machine, "run01")
        assert (run_dir / "main" / "logs").is_dir()

    def test_includes_error_log_line(self, tmp_path):
        run_dir = tmp_path / "run01"
        (run_dir / "main" / "logs").mkdir(parents=True)
        _write_slurm_toml(run_dir)
        text = write_slurm_script(run_dir, _make_machine(), "run01").read_text()
        assert "--error=" in text
        assert "slurm-%j.err" in text

    def test_omits_constraint_when_empty(self, tmp_path):
        run_dir = tmp_path / "run01"
        (run_dir / "main" / "logs").mkdir(parents=True)
        _write_slurm_toml(run_dir)  # constraint not set → ""
        text = write_slurm_script(run_dir, _make_machine(), "run01").read_text()
        assert "--constraint" not in text

    def test_includes_constraint_when_set(self, tmp_path):
        run_dir = tmp_path / "run01"
        (run_dir / "main" / "logs").mkdir(parents=True)
        (run_dir / "slurm.toml").write_text(
            '[basic]\naccount = "proj"\n'
            '[main]\npartition = "cpu"\nconstraint = "x86-64-v4&fast-io"\n'
            'time = "04:00:00"\nmem = "16000"\nntasks = 1\nnodes = 1\n'
            'cpus_per_task = 8\n'
            '[exec]\ntime = "01:00:00"\nmem = "4000"\nntasks = 1\n'
            'nodes = 1\ncpus_per_task = 2\n'
        )
        text = write_slurm_script(run_dir, _make_machine(), "run01").read_text()
        assert "--constraint=x86-64-v4&fast-io" in text

    def test_no_submit_dir_created(self, tmp_path):
        run_dir = tmp_path / "run01"
        run_dir.mkdir()
        _write_slurm_toml(run_dir)
        write_slurm_script(run_dir, _make_machine(), "run01")
        assert not (run_dir / "submit").exists()


# ---------------------------------------------------------------------------
# _find_exec_script
# ---------------------------------------------------------------------------

class TestFindExecScript:
    def test_finds_in_run_algorithm(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        campaign_dir = tmp_path / "campaign"
        campaign_dir.mkdir()
        script = run_dir / "algorithm" / "compute_sf.py"
        script.parent.mkdir()
        script.write_text("# script\n")
        found = _find_exec_script("compute_sf", run_dir, campaign_dir)
        assert found == script

    def test_finds_in_campaign_algorithm(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        campaign_dir = tmp_path / "campaign"
        script = campaign_dir / "algorithm" / "compute_sf.py"
        script.parent.mkdir(parents=True)
        script.write_text("# script\n")
        found = _find_exec_script("compute_sf", run_dir, campaign_dir)
        assert found == script

    def test_raises_when_not_found(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        campaign_dir = tmp_path / "campaign"
        campaign_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="compute_missing"):
            _find_exec_script("compute_missing", run_dir, campaign_dir)

    def test_run_takes_priority_over_campaign(self, tmp_path):
        run_dir = tmp_path / "run"
        campaign_dir = tmp_path / "campaign"
        run_script = run_dir / "algorithm" / "compute_sf.py"
        run_script.parent.mkdir(parents=True)
        run_script.write_text("# run version\n")
        campaign_script = campaign_dir / "algorithm" / "compute_sf.py"
        campaign_script.parent.mkdir(parents=True)
        campaign_script.write_text("# campaign version\n")
        found = _find_exec_script("compute_sf", run_dir, campaign_dir)
        assert found == run_script


# ---------------------------------------------------------------------------
# prepare_exec
# ---------------------------------------------------------------------------

class TestPrepareExec:
    def test_creates_slot_and_logs(self, tmp_path):
        run_dir = tmp_path / "run"
        campaign_dir = tmp_path / "campaign"
        script = campaign_dir / "algorithm" / "compute_sf.py"
        script.parent.mkdir(parents=True)
        script.write_text("# script\n")
        (run_dir / "algorithm").mkdir(parents=True)
        slot = prepare_exec(run_dir, "compute_sf", campaign_dir)
        assert slot == run_dir / "exec" / "compute_sf"
        assert slot.is_dir()
        assert (slot / "logs").is_dir()

    def test_copies_script_to_algorithm(self, tmp_path):
        run_dir = tmp_path / "run"
        campaign_dir = tmp_path / "campaign"
        script = campaign_dir / "algorithm" / "entangle.py"
        script.parent.mkdir(parents=True)
        script.write_text("# entangle\n")
        (run_dir / "algorithm").mkdir(parents=True)
        prepare_exec(run_dir, "entangle", campaign_dir)
        assert (run_dir / "algorithm" / "entangle.py").exists()

    def test_does_not_overwrite_existing_script(self, tmp_path):
        run_dir = tmp_path / "run"
        campaign_dir = tmp_path / "campaign"
        existing = run_dir / "algorithm" / "compute_sf.py"
        existing.parent.mkdir(parents=True)
        existing.write_text("# run version\n")
        campaign_script = campaign_dir / "algorithm" / "compute_sf.py"
        campaign_script.parent.mkdir(parents=True)
        campaign_script.write_text("# campaign version\n")
        prepare_exec(run_dir, "compute_sf", campaign_dir)
        assert existing.read_text() == "# run version\n"


# ---------------------------------------------------------------------------
# write_exec_slurm_script
# ---------------------------------------------------------------------------

class TestWriteExecSlurmScript:
    def test_writes_to_exec_slot(self, tmp_path):
        run_dir = tmp_path / "run01"
        (run_dir / "exec" / "compute_sf" / "logs").mkdir(parents=True)
        _write_slurm_toml(run_dir)
        machine = _make_machine()
        script = write_exec_slurm_script(run_dir, "compute_sf", machine)
        assert script == run_dir / "exec" / "compute_sf" / "submit.slurm"
        assert script.exists()

    def test_exec_script_uses_exec_section(self, tmp_path):
        run_dir = tmp_path / "run01"
        (run_dir / "exec" / "compute_sf" / "logs").mkdir(parents=True)
        (run_dir / "slurm.toml").write_text(
            '[basic]\naccount = "proj"\n'
            '[main]\npartition = "long"\ntime = "24:00:00"\nmem = "200000"\n'
            'ntasks = 1\nnodes = 1\ncpus_per_task = 32\n'
            '[exec]\npartition = "short"\ntime = "02:00:00"\nmem = "16000"\n'
            'ntasks = 1\nnodes = 1\ncpus_per_task = 8\n'
        )
        text = write_exec_slurm_script(run_dir, "compute_sf", _make_machine()).read_text()
        # Should use exec section values, not main section
        assert "short" in text
        assert "02:00:00" in text
        assert "16000" in text
        assert "24:00:00" not in text

    def test_script_calls_correct_python_file(self, tmp_path):
        run_dir = tmp_path / "run01"
        (run_dir / "exec" / "compute_sf" / "logs").mkdir(parents=True)
        _write_slurm_toml(run_dir)
        text = write_exec_slurm_script(run_dir, "compute_sf", _make_machine()).read_text()
        assert "compute_sf.py" in text
        assert "--run-dir" in text
