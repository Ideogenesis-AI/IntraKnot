"""Tests for src/intraknot/config.py."""

import textwrap

import pytest

from intraknot.config import (
    MachineConfig,
    PathsConfig,
    SlurmConfig,
    _merge_defaults,
    load_campaign_defaults,
    load_config,
    load_machine_config,
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
# load_machine_config
# ---------------------------------------------------------------------------

class TestLoadMachineConfig:
    def _write_slurm(self, d, **kwargs):
        defaults = dict(
            account="myproject",
            partition="standard",
            default_time="08:00:00",
            default_mem="32G",
            default_cpus_per_task=16,
        )
        defaults.update(kwargs)
        (d / "slurm.toml").write_text(
            "[slurm]\n"
            + "\n".join(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}"
                        for k, v in defaults.items())
        )

    def _write_paths(self, d, **kwargs):
        defaults = dict(
            run_root="/scratch/runs",
            scratch_root="/scratch",
            node_local_scratch="/tmp/$USER",
            python="uv run",
        )
        defaults.update(kwargs)
        (d / "paths.toml").write_text(
            "[paths]\n"
            + "\n".join(f'{k} = "{v}"' for k, v in defaults.items())
        )

    def test_reads_slurm_and_paths(self, tmp_path):
        self._write_slurm(tmp_path)
        self._write_paths(tmp_path)
        mc = load_machine_config(tmp_path)
        assert mc.slurm.account == "myproject"
        assert mc.slurm.partition == "standard"
        assert mc.slurm.default_cpus_per_task == 16
        assert mc.paths.run_root == "/scratch/runs"
        assert mc.paths.python == "uv run"

    def test_missing_files_give_defaults(self, tmp_path):
        mc = load_machine_config(tmp_path)
        assert isinstance(mc, MachineConfig)
        assert mc.slurm.partition == ""
        assert mc.paths.python == "uv run"

    def test_partial_slurm_uses_defaults_for_missing_keys(self, tmp_path):
        (tmp_path / "slurm.toml").write_text('[slurm]\naccount = "proj"\n')
        mc = load_machine_config(tmp_path)
        assert mc.slurm.account == "proj"
        assert mc.slurm.default_time == "04:00:00"  # default


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
    def test_write_slurm_toml(self, tmp_path):
        p = tmp_path / "slurm.toml"
        write_slurm_toml(p)
        assert p.exists()
        text = p.read_text()
        assert "[slurm]" in text
        assert "account" in text

    def test_write_paths_toml(self, tmp_path):
        p = tmp_path / "paths.toml"
        write_paths_toml(p)
        text = p.read_text()
        assert "[paths]" in text
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
        assert "!.gitignore" in gi.read_text()


# ---------------------------------------------------------------------------
# _merge_defaults
# ---------------------------------------------------------------------------

class TestMergeDefaults:
    def test_adds_missing_sections(self):
        user = {"model": {"geometry": {"lx": 32}}}
        defaults = {"algorithm": {"max_bond": 64}, "output": {"save_state": True}}
        merged = _merge_defaults(user, defaults)
        assert merged["model"]["geometry"]["lx"] == 32
        assert merged["algorithm"]["max_bond"] == 64
        assert merged["output"]["save_state"] is True

    def test_run_values_win(self):
        user = {"algorithm": {"max_bond": 128}}
        defaults = {"algorithm": {"max_bond": 64, "n_sweeps": 10}}
        merged = _merge_defaults(user, defaults)
        assert merged["algorithm"]["max_bond"] == 128
        assert merged["algorithm"]["n_sweeps"] == 10

    def test_no_defaults_is_identity(self):
        user = {"model": {}}
        merged = _merge_defaults(user, {})
        assert merged == user

    def test_none_user_cfg_returns_defaults(self):
        defaults = {"algorithm": {"max_bond": 64}}
        merged = _merge_defaults(None, defaults)
        assert merged["algorithm"]["max_bond"] == 64
