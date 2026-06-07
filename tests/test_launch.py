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
import yaml

from intraknot.launch import (
    _dump_toml,
    _find_exec_script,
    _register_run_in_campaign,
    _render_slurm_header,
    create_attempt,
    create_campaign,
    create_run,
    delete_run,
    prepare_exec,
    read_runs_by_filter,
    remove_run_from_all_campaigns,
    update_current,
    write_array_slurm_script,
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
    return MachineConfig(paths=PathsConfig(command="uv run"))


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
        assert "scan_id" in header
        assert "status" not in header

    def test_raises_if_exists(self, tmp_path):
        campaigns_root = tmp_path / "campaigns"
        campaigns_root.mkdir()
        create_campaign("c1", "", "dmrg", campaigns_root)
        with pytest.raises(FileExistsError):
            create_campaign("c1", "", "dmrg", campaigns_root)


# ---------------------------------------------------------------------------
# _register_run_in_campaign
# ---------------------------------------------------------------------------

class TestRegisterRunInCampaign:
    def test_no_duplicate_on_double_register(self, tmp_path):
        campaign_dir = tmp_path / "c1"
        campaign_dir.mkdir()
        _register_run_in_campaign(campaign_dir, "run01")
        _register_run_in_campaign(campaign_dir, "run01")  # second call must be a no-op
        with open(campaign_dir / "runs.csv", newline="") as f:
            rows = list(csv.reader(f))
        data_rows = [r for r in rows if r and r[0] != "run_id"]
        assert len(data_rows) == 1

    def test_different_run_ids_both_registered(self, tmp_path):
        campaign_dir = tmp_path / "c1"
        campaign_dir.mkdir()
        _register_run_in_campaign(campaign_dir, "run01")
        _register_run_in_campaign(campaign_dir, "run02")
        with open(campaign_dir / "runs.csv", newline="") as f:
            rows = list(csv.reader(f))
        data_rows = [r for r in rows if r and r[0] != "run_id"]
        assert len(data_rows) == 2


# ---------------------------------------------------------------------------
# create_run — manifest contents
# ---------------------------------------------------------------------------

class TestCreateRunManifest:
    def _setup(self, tmp_path: Path):
        campaigns_root = tmp_path / "campaigns"
        runs_root = tmp_path / "runs"
        create_campaign("mycampaign", "", "dmrg", campaigns_root)
        return campaigns_root, runs_root

    def test_manifest_has_required_keys(self, tmp_path):
        campaigns_root, runs_root = self._setup(tmp_path)
        run_dir = create_run(
            "run01", "mycampaign", None,
            runs_root=runs_root, campaigns_root=campaigns_root,
        )
        manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
        assert manifest["run_id"] == "run01"
        assert manifest["algorithm"] == "dmrg"
        assert "created_at" in manifest

    def test_manifest_has_uuid(self, tmp_path):
        import uuid
        campaigns_root, runs_root = self._setup(tmp_path)
        run_dir = create_run(
            "run01", "mycampaign", None,
            runs_root=runs_root, campaigns_root=campaigns_root,
        )
        manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
        assert "uuid" in manifest
        # Value must be a valid UUID4.
        parsed = uuid.UUID(manifest["uuid"])
        assert parsed.version == 4

    def test_manifest_uuid_matches_supplied_run_uuid(self, tmp_path):
        import uuid
        campaigns_root, runs_root = self._setup(tmp_path)
        fixed_uuid = uuid.UUID("12345678-1234-4abc-89de-f01234567890")
        run_dir = create_run(
            "run01", "mycampaign", None,
            runs_root=runs_root, campaigns_root=campaigns_root,
            run_uuid=fixed_uuid,
        )
        manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
        assert manifest["uuid"] == str(fixed_uuid)

    def test_manifest_has_no_campaign_key(self, tmp_path):
        campaigns_root, runs_root = self._setup(tmp_path)
        run_dir = create_run(
            "run01", "mycampaign", None,
            runs_root=runs_root, campaigns_root=campaigns_root,
        )
        manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
        assert "campaign" not in manifest

    def test_manifest_has_no_status_key(self, tmp_path):
        campaigns_root, runs_root = self._setup(tmp_path)
        run_dir = create_run(
            "run01", "mycampaign", None,
            runs_root=runs_root, campaigns_root=campaigns_root,
        )
        manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
        assert "status" not in manifest

    def test_manifest_has_no_machine_key(self, tmp_path):
        campaigns_root, runs_root = self._setup(tmp_path)
        run_dir = create_run(
            "run01", "mycampaign", None,
            runs_root=runs_root, campaigns_root=campaigns_root,
        )
        manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
        assert "machine" not in manifest


# ---------------------------------------------------------------------------
# remove_run_from_all_campaigns
# ---------------------------------------------------------------------------

class TestRemoveRunFromAllCampaigns:
    def _make_campaign_csv(self, campaigns_root: Path, campaign_id: str, run_ids: list) -> Path:
        campaign_dir = campaigns_root / campaign_id
        campaign_dir.mkdir(parents=True, exist_ok=True)
        with open(campaign_dir / "runs.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "scan_id"])
            for rid in run_ids:
                writer.writerow([rid, ""])
        return campaign_dir

    def test_removes_from_all_matching_campaigns(self, tmp_path):
        campaigns_root = tmp_path / "campaigns"
        self._make_campaign_csv(campaigns_root, "c1", ["run_a", "run_b"])
        self._make_campaign_csv(campaigns_root, "c2", ["run_a", "run_c"])
        self._make_campaign_csv(campaigns_root, "c3", ["run_c"])

        removed = remove_run_from_all_campaigns(campaigns_root, "run_a")

        assert sorted(removed) == ["c1", "c2"]
        # run_a must be gone from c1 and c2.
        for cid in ("c1", "c2"):
            with open(campaigns_root / cid / "runs.csv", newline="") as f:
                ids = [r[0] for r in csv.reader(f) if r and r[0] != "run_id"]
            assert "run_a" not in ids
        # c3 was unaffected.
        with open(campaigns_root / "c3" / "runs.csv", newline="") as f:
            ids = [r[0] for r in csv.reader(f) if r and r[0] != "run_id"]
        assert "run_c" in ids

    def test_returns_empty_when_run_not_found(self, tmp_path):
        campaigns_root = tmp_path / "campaigns"
        self._make_campaign_csv(campaigns_root, "c1", ["run_x"])
        removed = remove_run_from_all_campaigns(campaigns_root, "run_ghost")
        assert removed == []

    def test_returns_empty_when_campaigns_root_absent(self, tmp_path):
        removed = remove_run_from_all_campaigns(tmp_path / "nonexistent", "run_a")
        assert removed == []


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


# ---------------------------------------------------------------------------
# create_run — additional coverage
# ---------------------------------------------------------------------------

class TestCreateRunExtended:
    def _setup(self, tmp_path: Path):
        campaigns_root = tmp_path / "campaigns"
        runs_root = tmp_path / "runs"
        create_campaign("mycampaign", "", "dmrg", campaigns_root)
        return campaigns_root, runs_root

    def test_create_run_with_config_src(self, tmp_path):
        campaigns_root, runs_root = self._setup(tmp_path)
        config_file = _minimal_config_toml(tmp_path)
        run_dir = create_run(
            "run_cfg", "mycampaign", config_file,
            runs_root=runs_root, campaigns_root=campaigns_root,
        )
        cfg_out = run_dir / "config.toml"
        assert cfg_out.exists()
        import tomllib
        with open(cfg_out, "rb") as f:
            data = tomllib.load(f)
        assert data["geometry"]["lx"] == 16

    def test_create_run_both_config_src_and_overrides_raises(self, tmp_path):
        campaigns_root, runs_root = self._setup(tmp_path)
        config_file = _minimal_config_toml(tmp_path)
        with pytest.raises(ValueError, match="at most one"):
            create_run(
                "run_conflict", "mycampaign", config_file,
                runs_root=runs_root, campaigns_root=campaigns_root,
                overrides={"model": {"J": 2.0}},
            )

    def test_create_run_scan_id_in_runs_csv(self, tmp_path):
        campaigns_root, runs_root = self._setup(tmp_path)
        create_run(
            "run_scan", "mycampaign", None,
            runs_root=runs_root, campaigns_root=campaigns_root,
            scan_id="chi_scan",
        )
        with open(campaigns_root / "mycampaign" / "runs.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        matching = [r for r in rows if r["run_id"] == "run_scan"]
        assert len(matching) == 1
        assert matching[0]["scan_id"] == "chi_scan"


# ---------------------------------------------------------------------------
# read_runs_by_filter
# ---------------------------------------------------------------------------

class TestReadRunsByFilter:
    def _make_csv(self, campaign_dir: Path, rows: list) -> None:
        """Write a two-column runs.csv (run_id, scan_id)."""
        campaign_dir.mkdir(parents=True, exist_ok=True)
        with open(campaign_dir / "runs.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "scan_id"])
            for r in rows:
                writer.writerow(r)

    def _write_live_status(self, runs_root: Path, run_id: str, state: str) -> None:
        """Write a minimal main/status.json so live status filtering works."""
        import json
        main_dir = runs_root / run_id / "main"
        main_dir.mkdir(parents=True, exist_ok=True)
        (main_dir / "status.json").write_text(json.dumps({"state": state}))

    def test_no_filter_returns_all(self, tmp_path):
        cd = tmp_path / "c1"
        self._make_csv(cd, [["r1", "s1"], ["r2", "s1"]])
        ids = read_runs_by_filter(cd)
        assert ids == ["r1", "r2"]

    def test_scan_id_filter(self, tmp_path):
        cd = tmp_path / "c1"
        self._make_csv(cd, [["r1", "s1"], ["r2", "s2"]])
        ids = read_runs_by_filter(cd, scan_id="s1")
        assert ids == ["r1"]

    def test_status_filter(self, tmp_path):
        runs_root = tmp_path / "runs"
        cd = tmp_path / "c1"
        self._make_csv(cd, [["r1", ""], ["r2", ""]])
        self._write_live_status(runs_root, "r1", "pending")
        self._write_live_status(runs_root, "r2", "failed")
        ids = read_runs_by_filter(cd, status="failed", runs_root=runs_root)
        assert ids == ["r2"]

    def test_status_filter_no_status_json_treated_as_pending(self, tmp_path):
        runs_root = tmp_path / "runs"
        cd = tmp_path / "c1"
        self._make_csv(cd, [["r1", ""], ["r2", ""]])
        # r1 has no main/status.json — should be treated as "pending".
        self._write_live_status(runs_root, "r2", "failed")
        ids = read_runs_by_filter(cd, status="pending", runs_root=runs_root)
        assert ids == ["r1"]

    def test_combined_filter(self, tmp_path):
        runs_root = tmp_path / "runs"
        cd = tmp_path / "c1"
        self._make_csv(cd, [["r1", "s1"], ["r2", "s1"], ["r3", "s2"]])
        self._write_live_status(runs_root, "r1", "pending")
        self._write_live_status(runs_root, "r2", "failed")
        self._write_live_status(runs_root, "r3", "failed")
        ids = read_runs_by_filter(cd, scan_id="s1", status="failed", runs_root=runs_root)
        assert ids == ["r2"]

    def test_missing_csv_returns_empty(self, tmp_path):
        ids = read_runs_by_filter(tmp_path / "nonexistent")
        assert ids == []


# ---------------------------------------------------------------------------
# delete_run
# ---------------------------------------------------------------------------

class TestDeleteRun:
    def _make_campaign_csv(self, campaigns_root: Path, campaign_id: str, run_ids: list) -> Path:
        campaign_dir = campaigns_root / campaign_id
        campaign_dir.mkdir(parents=True, exist_ok=True)
        with open(campaign_dir / "runs.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "scan_id"])
            for rid in run_ids:
                writer.writerow([rid, ""])
        return campaign_dir

    def test_deregisters_from_campaign_csv(self, tmp_path):
        campaigns_root = tmp_path / "campaigns"
        runs_root = tmp_path / "runs"
        run_dir = runs_root / "my_run"
        run_dir.mkdir(parents=True)
        campaign_dir = self._make_campaign_csv(campaigns_root, "c1", ["my_run"])

        delete_run(run_dir, campaign_dir=campaign_dir)

        with open(campaign_dir / "runs.csv", newline="") as f:
            ids = [r[0] for r in csv.reader(f) if r and r[0] != "run_id"]
        assert "my_run" not in ids
        assert run_dir.exists()

    def test_delete_dir_removes_directory(self, tmp_path):
        campaigns_root = tmp_path / "campaigns"
        runs_root = tmp_path / "runs"
        run_dir = runs_root / "my_run"
        run_dir.mkdir(parents=True)
        self._make_campaign_csv(campaigns_root, "c1", ["my_run"])
        self._make_campaign_csv(campaigns_root, "c2", ["my_run"])

        delete_run(run_dir, delete_dir=True, campaigns_root=campaigns_root)

        assert not run_dir.exists()


# ---------------------------------------------------------------------------
# write_array_slurm_script
# ---------------------------------------------------------------------------

class TestWriteArraySlurmScript:
    def test_creates_file_with_array_directive(self, tmp_path):
        campaigns_root = tmp_path / "campaigns"
        runs_root = tmp_path / "runs"
        camp_dir = create_campaign("c1", "", "dmrg", campaigns_root)

        script = write_array_slurm_script(camp_dir, runs_root, _make_machine(), "1-5")

        assert script.name == "submit_array.slurm"
        assert script.exists()
        text = script.read_text()
        assert "#SBATCH --array=1-5" in text

    def test_script_uses_row_number_for_run_id_lookup(self, tmp_path):
        campaigns_root = tmp_path / "campaigns"
        runs_root = tmp_path / "runs"
        camp_dir = create_campaign("c1", "", "dmrg", campaigns_root)

        script = write_array_slurm_script(camp_dir, runs_root, _make_machine(), "1-3")
        text = script.read_text()

        # After the fix the awk expr is NR-1==id (row-based), not $1==id.
        assert "NR-1==id" in text
        assert "print $1" in text
