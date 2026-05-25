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


"""Tests for src/intraknot/config.py."""

import pytest

from intraknot.config import (
    MachineConfig,
    PathsConfig,
    SlurmBasicConfig,
    SlurmJobConfig,
    SlurmTomlConfig,
    _merge_defaults,
    load_campaign_defaults,
    load_config,
    load_machine_config,
    load_slurm_toml,
    write_data_gitignore,
    write_machines_yaml,
    write_paths_toml,
    write_slurm_toml,
)


# ---------------------------------------------------------------------------
# load_config dispatch
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_toml(self, tmp_path):
        f = tmp_path / "cfg.toml"
        f.write_text('[section]\nkey = "value"\n')
        data = load_config(f)
        assert data["section"]["key"] == "value"

    def test_loads_yaml(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text("section:\n  key: value\n")
        data = load_config(f)
        assert data["section"]["key"] == "value"

    def test_loads_yml_extension(self, tmp_path):
        f = tmp_path / "cfg.yml"
        f.write_text("machines: []\n")
        data = load_config(f)
        assert data["machines"] == []

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        data = load_config(f)
        assert data == {}

    def test_unknown_extension_raises(self, tmp_path):
        f = tmp_path / "cfg.json"
        f.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported"):
            load_config(f)


# ---------------------------------------------------------------------------
# load_slurm_toml
# ---------------------------------------------------------------------------

class TestLoadSlurmToml:
    def test_returns_defaults_for_missing_file(self, tmp_path):
        cfg = load_slurm_toml(tmp_path / "nonexistent.toml")
        assert isinstance(cfg, SlurmTomlConfig)
        assert cfg.basic.account == ""
        assert cfg.main.cpus_per_task == 8
        assert cfg.exec_.time == "04:00:00"

    def test_reads_all_sections(self, tmp_path):
        p = tmp_path / "slurm.toml"
        p.write_text(
            '[basic]\naccount = "proj"\nmail_type = "ALL"\nmail_user = "x@y.com"\n'
            '[main]\npartition = "cluster"\nconstraint = "fast"\n'
            'time = "24:00:00"\nmem = "300000"\nntasks = 1\nnodes = 1\n'
            'cpus_per_task = 64\n'
            '[exec]\npartition = "short"\ntime = "01:00:00"\nmem = "8000"\n'
            'ntasks = 1\nnodes = 1\ncpus_per_task = 4\n'
        )
        cfg = load_slurm_toml(p)
        assert cfg.basic.account == "proj"
        assert cfg.basic.mail_type == "ALL"
        assert cfg.basic.mail_user == "x@y.com"
        assert cfg.main.partition == "cluster"
        assert cfg.main.constraint == "fast"
        assert cfg.main.time == "24:00:00"
        assert cfg.main.mem == "300000"
        assert cfg.main.cpus_per_task == 64
        assert cfg.exec_.partition == "short"
        assert cfg.exec_.time == "01:00:00"
        assert cfg.exec_.cpus_per_task == 4

    def test_partial_section_uses_defaults(self, tmp_path):
        p = tmp_path / "slurm.toml"
        p.write_text('[main]\ntime = "48:00:00"\n')
        cfg = load_slurm_toml(p)
        assert cfg.main.time == "48:00:00"
        assert cfg.main.cpus_per_task == 8  # dataclass default
        assert cfg.exec_.time == "04:00:00"  # exec section absent, uses default

    def test_mem_coerced_to_str(self, tmp_path):
        p = tmp_path / "slurm.toml"
        p.write_text('[main]\nmem = 300000\n')
        cfg = load_slurm_toml(p)
        assert cfg.main.mem == "300000"
        assert isinstance(cfg.main.mem, str)


# ---------------------------------------------------------------------------
# load_machine_config
# ---------------------------------------------------------------------------

class TestLoadMachineConfig:
    def _write_paths(self, d, **kwargs):
        defaults = dict(
            project_root="/scratch/runs",
            scratch_root="/scratch",
            scratch_node="/tmp/$USER",
            command="uv run",
        )
        defaults.update(kwargs)
        (d / "paths.toml").write_text(
            "[paths]\n"
            + "\n".join(f'{k} = "{v}"' for k, v in defaults.items())
        )

    def test_reads_paths(self, tmp_path):
        self._write_paths(tmp_path)
        mc = load_machine_config(tmp_path)
        assert mc.paths.project_root == "/scratch/runs"
        assert mc.paths.scratch_root == "/scratch"
        assert mc.paths.scratch_node == "/tmp/$USER"
        assert mc.paths.command == "uv run"

    def test_missing_files_give_defaults(self, tmp_path):
        mc = load_machine_config(tmp_path)
        assert isinstance(mc, MachineConfig)
        assert mc.paths.command == "uv run"

    def test_no_slurm_field_on_machine_config(self, tmp_path):
        mc = load_machine_config(tmp_path)
        assert not hasattr(mc, "slurm")


# ---------------------------------------------------------------------------
# load_campaign_defaults
# ---------------------------------------------------------------------------

class TestLoadCampaignDefaults:
    def test_returns_empty_when_absent(self, tmp_path):
        d = load_campaign_defaults(tmp_path)
        assert d == {}

    def test_reads_defaults_toml(self, tmp_path):
        (tmp_path / "defaults.toml").write_text(
            "[algorithm]\nmax_bond = 64\n[output]\nsave_state = true\n"
        )
        d = load_campaign_defaults(tmp_path)
        assert d["algorithm"]["max_bond"] == 64
        assert d["output"]["save_state"] is True


# ---------------------------------------------------------------------------
# Template generators
# ---------------------------------------------------------------------------

class TestTemplateGenerators:
    def test_write_slurm_toml_has_three_sections(self, tmp_path):
        p = tmp_path / "slurm.toml"
        write_slurm_toml(p)
        assert p.exists()
        text = p.read_text()
        assert "[basic]" in text
        assert "[main]" in text
        assert "[exec]" in text
        assert "account" in text
        assert "cpus_per_task" in text

    def test_write_slurm_toml_is_loadable(self, tmp_path):
        p = tmp_path / "slurm.toml"
        write_slurm_toml(p)
        cfg = load_slurm_toml(p)
        assert isinstance(cfg, SlurmTomlConfig)

    def test_write_paths_toml(self, tmp_path):
        p = tmp_path / "paths.toml"
        write_paths_toml(p)
        text = p.read_text()
        assert "[paths]" in text
        assert "project_root" in text
        assert "scratch_root" in text
        assert "scratch_node" in text
        assert "command" in text
        assert "uv run" in text

    def test_write_machines_yaml(self, tmp_path):
        p = tmp_path / "machines.yaml"
        write_machines_yaml(p)
        assert p.exists()
        assert "machines" in p.read_text()

    def test_write_data_gitignore(self, tmp_path):
        d = tmp_path / "data"
        write_data_gitignore(d)
        gi = d / ".gitignore"
        assert gi.exists()
        assert "*" in gi.read_text()
        assert "!.gitignore" not in gi.read_text()


# ---------------------------------------------------------------------------
# _merge_defaults
# ---------------------------------------------------------------------------

class TestMergeDefaults:
    def test_adds_missing_sections(self):
        user = {"geometry": {"lx": 32}, "model": {"label": "Heisenberg"}}
        defaults = {"algorithm": {"max_bond": 64}, "output": {"save_state": True}}
        merged = _merge_defaults(user, defaults)
        assert merged["geometry"]["lx"] == 32
        assert merged["model"]["label"] == "Heisenberg"
        assert merged["algorithm"]["max_bond"] == 64
        assert merged["output"]["save_state"] is True

    def test_geometry_and_model_from_defaults_when_user_has_none(self):
        defaults = {
            "geometry": {"lattice": "chain", "lx": 20},
            "model": {"label": "Heisenberg", "category": "bosonic"},
            "algorithm": {"max_bond": 64},
        }
        merged = _merge_defaults(None, defaults)
        assert merged["geometry"]["lx"] == 20
        assert merged["model"]["label"] == "Heisenberg"
        assert merged["algorithm"]["max_bond"] == 64

    def test_run_geometry_overrides_default_geometry(self):
        user = {"geometry": {"lx": 128, "bcx": "OBC"}}
        defaults = {"geometry": {"lx": 64, "bcx": "PBC"}}
        merged = _merge_defaults(user, defaults)
        assert merged["geometry"]["lx"] == 128
        assert merged["geometry"]["bcx"] == "OBC"

    def test_run_values_win(self):
        user = {"algorithm": {"max_bond": 128}}
        defaults = {"algorithm": {"max_bond": 64, "n_sweeps": 10}}
        merged = _merge_defaults(user, defaults)
        assert merged["algorithm"]["max_bond"] == 128
        assert merged["algorithm"]["n_sweeps"] == 10

    def test_no_defaults_is_identity(self):
        user = {"geometry": {}, "model": {}}
        merged = _merge_defaults(user, {})
        assert merged == user

    def test_none_user_cfg_returns_defaults(self):
        defaults = {"algorithm": {"max_bond": 64}}
        merged = _merge_defaults(None, defaults)
        assert merged["algorithm"]["max_bond"] == 64
