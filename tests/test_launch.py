"""Tests for src/intraknot/launch.py."""

import csv
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from intraknot.config import MachineConfig, PathsConfig, SlurmConfig
from intraknot.launch import (
    _dump_toml,
    create_attempt,
    create_campaign,
    update_current,
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
# Helpers
# ---------------------------------------------------------------------------

def _make_machine() -> MachineConfig:
    return MachineConfig(
        slurm=SlurmConfig(
            account="testproject",
            partition="cpu",
            default_time="02:00:00",
            default_mem="8G",
            default_cpus_per_task=4,
        ),
        paths=PathsConfig(python="uv run"),
    )


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
    def test_writes_file(self, tmp_path):
        run_dir = tmp_path / "run01"
        (run_dir / "submit").mkdir(parents=True)
        (run_dir / "logs").mkdir()
        machine = _make_machine()
        script = write_slurm_script(run_dir, machine, "run01")
        assert script.exists()
        text = script.read_text()
        assert "#!/usr/bin/env bash" in text
        assert "#SBATCH --job-name=run01" in text
        assert "uv run" in text
        assert str(run_dir) in text

    def test_uses_machine_config(self, tmp_path):
        run_dir = tmp_path / "myrun"
        (run_dir / "submit").mkdir(parents=True)
        (run_dir / "logs").mkdir()
        machine = _make_machine()
        script = write_slurm_script(run_dir, machine)
        text = script.read_text()
        assert "testproject" in text
        assert "cpu" in text
        assert "02:00:00" in text
