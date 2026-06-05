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
    _infer_typed_value,
    _merge_defaults,
    iter_scan_combinations,
    load_campaign_defaults,
    load_cluster_discovery,
    load_config,
    load_machine_config,
    load_run_config,
    load_slurm_toml,
    make_run_id,
    parse_set_overrides,
    write_data_gitignore,
    write_paths_toml,
    write_slurm_toml,
    write_tui_toml,
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

    def test_write_data_gitignore(self, tmp_path):
        d = tmp_path / "data"
        write_data_gitignore(d)
        gi = d / ".gitignore"
        assert gi.exists()
        assert "*" in gi.read_text()
        assert "!.gitignore" not in gi.read_text()


# ---------------------------------------------------------------------------
# load_cluster_discovery
# ---------------------------------------------------------------------------

class TestLoadClusterDiscovery:
    def test_returns_none_when_absent(self, tmp_path):
        assert load_cluster_discovery(tmp_path) is None

    def test_loads_cluster_yaml(self, tmp_path):
        (tmp_path / "cluster.yaml").write_text(
            "discovered_at: '2026-05-26T00:00:00+00:00'\n"
            "hostname: 'login.example.com'\n"
            "partitions:\n"
            "  - name: cpu\n"
            "    state: up\n"
            "    default: true\n"
            "    time_limit: '7-00:00:00'\n"
            "    node_groups:\n"
            "      - nodes: 'node[01-04]'\n"
            "        cpus: 32\n"
            "        mem_mb: 128000\n"
            "        gres: ''\n"
            "        features: [epyc]\n"
            "features:\n"
            "  epyc:\n"
            "    partitions: [cpu]\n"
            "    node_count: 4\n"
        )
        disc = load_cluster_discovery(tmp_path)
        assert disc is not None
        assert disc.hostname == "login.example.com"
        assert len(disc.partitions) == 1
        p = disc.partitions[0]
        assert p.name == "cpu"
        assert p.default is True
        assert len(p.node_groups) == 1
        g = p.node_groups[0]
        assert g.nodes == "node[01-04]"
        assert g.cpus == 32
        assert g.mem_mb == 128000
        assert g.features == ["epyc"]
        assert disc.features["epyc"].node_count == 4
        assert disc.features["epyc"].partitions == ["cpu"]


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

    def test_empty_dict_user_cfg_does_not_drop_defaults(self):
        """Empty dict must not be treated as falsy — the bug was `if user_cfg:`."""
        defaults = {"algorithm": {"max_bond": 64}}
        merged = _merge_defaults({}, defaults)
        assert merged["algorithm"]["max_bond"] == 64

    def test_new_section_in_defaults_merged_automatically(self):
        """Any section in defaults is merged, not only the four hardcoded ones."""
        defaults = {"optimizer": {"lr": 0.01}}
        merged = _merge_defaults(None, defaults)
        assert merged["optimizer"]["lr"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# _infer_typed_value
# ---------------------------------------------------------------------------

class TestInferTypedValue:
    def test_bool_true(self):
        assert _infer_typed_value("true", False) is True
        assert _infer_typed_value("yes", False) is True
        assert _infer_typed_value("1", False) is True

    def test_bool_false(self):
        assert _infer_typed_value("false", True) is False
        assert _infer_typed_value("no", True) is False
        assert _infer_typed_value("0", True) is False

    def test_bool_invalid(self):
        with pytest.raises(ValueError, match="bool"):
            _infer_typed_value("maybe", True)

    def test_int(self):
        assert _infer_typed_value("42", 0) == 42
        assert isinstance(_infer_typed_value("10", 1), int)

    def test_int_invalid(self):
        with pytest.raises(ValueError, match="int"):
            _infer_typed_value("3.14", 0)

    def test_float(self):
        assert _infer_typed_value("3.14", 0.0) == pytest.approx(3.14)

    def test_float_invalid(self):
        with pytest.raises(ValueError, match="float"):
            _infer_typed_value("abc", 1.0)

    def test_str_passthrough(self):
        assert _infer_typed_value("hello", "world") == "hello"

    def test_list_ints(self):
        parsed = _infer_typed_value("1,2,3", [0])
        assert parsed == [1, 2, 3]

    def test_list_floats(self):
        parsed = _infer_typed_value("0.5,1.0", [0.0])
        assert parsed == pytest.approx([0.5, 1.0])

    def test_list_strings(self):
        parsed = _infer_typed_value("a,b,c", ["x"])
        assert parsed == ["a", "b", "c"]

    def test_list_empty_reference_treats_as_str(self):
        """Empty list reference → infer elements as strings."""
        parsed = _infer_typed_value("x,y", [])
        assert parsed == ["x", "y"]

    def test_exception_chain_preserved_on_int(self):
        """ValueError raised from int conversion should chain the original."""
        try:
            _infer_typed_value("notanint", 0)
        except ValueError as exc:
            assert exc.__cause__ is not None


# ---------------------------------------------------------------------------
# parse_set_overrides
# ---------------------------------------------------------------------------

class TestParseSetOverrides:
    def _defaults(self):
        return {"algorithm": {"max_bond": 64, "n_sweeps": 10}}

    def test_basic_override(self):
        ov = parse_set_overrides(["algorithm.max_bond=128"], self._defaults())
        assert ov["algorithm"]["max_bond"] == 128

    def test_multi_override(self):
        ov = parse_set_overrides(
            ["algorithm.max_bond=256", "algorithm.n_sweeps=20"],
            self._defaults(),
        )
        assert ov["algorithm"]["max_bond"] == 256
        assert ov["algorithm"]["n_sweeps"] == 20

    def test_multi_value_raises(self):
        with pytest.raises(ValueError):
            parse_set_overrides(["algorithm.max_bond=64,128"], self._defaults())


# ---------------------------------------------------------------------------
# iter_scan_combinations
# ---------------------------------------------------------------------------

class TestIterScanCombinations:
    def _defaults(self):
        return {"algorithm": {"max_bond": 64, "n_sweeps": 10}}

    def test_single_value_yields_one_item(self):
        combos = list(iter_scan_combinations(["algorithm.max_bond=128"], self._defaults()))
        assert len(combos) == 1
        nested, flat = combos[0]
        assert nested["algorithm"]["max_bond"] == 128
        assert flat["max_bond"] == 128

    def test_multi_value_cartesian_product(self):
        combos = list(
            iter_scan_combinations(["algorithm.max_bond=64,128,256"], self._defaults())
        )
        assert len(combos) == 3
        bonds = [c[0]["algorithm"]["max_bond"] for c in combos]
        assert bonds == [64, 128, 256]

    def test_two_multi_value_cross_product(self):
        combos = list(
            iter_scan_combinations(
                ["algorithm.max_bond=64,128", "algorithm.n_sweeps=5,10"],
                self._defaults(),
            )
        )
        assert len(combos) == 4

    def test_malformed_arg_raises(self):
        with pytest.raises(ValueError):
            list(iter_scan_combinations(["badarg"], self._defaults()))


# ---------------------------------------------------------------------------
# write_tui_toml
# ---------------------------------------------------------------------------

class TestWriteTuiToml:
    def test_creates_file_with_tui_section(self, tmp_path):
        p = tmp_path / "tui.toml"
        write_tui_toml(p)
        assert p.exists()
        text = p.read_text()
        assert "[tui]" in text
        assert "editor" in text

    def test_is_valid_toml(self, tmp_path):
        import tomllib
        p = tmp_path / "tui.toml"
        write_tui_toml(p)
        with open(p, "rb") as f:
            data = tomllib.load(f)
        assert "tui" in data


# ---------------------------------------------------------------------------
# make_run_id
# ---------------------------------------------------------------------------

class TestMakeRunId:
    def _cfg(self, lx: int = 20, max_bond: int = 64, label: str = "Heisenberg") -> dict:
        return {
            "geometry": {"lattice": "chain", "lx": lx},
            "model": {"label": label},
            "algorithm": {"engine": "dmrg", "max_bond": max_bond},
        }

    def test_basic_1d_id(self):
        rid = make_run_id(self._cfg(), {}, "abcd1234")
        assert "dmrg" in rid
        assert "heisenberg" in rid
        assert "chain_len=20" in rid
        assert "abcd1234" in rid

    def test_2d_lattice(self):
        cfg = {
            "geometry": {"lattice": "square", "lx": 4, "ly": 4},
            "model": {"label": "Hubbard"},
            "algorithm": {"engine": "dmrg", "max_bond": 128},
        }
        rid = make_run_id(cfg, {}, "deadbeef")
        assert "square_cell=4x4" in rid

    def test_overrides_appear_in_id(self):
        rid = make_run_id(self._cfg(max_bond=128), {"max_bond": 128}, "00000000")
        assert "max_bond=128" in rid

    def test_uuid8_suffix(self):
        rid = make_run_id(self._cfg(), {}, "cafe1234")
        assert rid.endswith("cafe1234")


# ---------------------------------------------------------------------------
# load_run_config
# ---------------------------------------------------------------------------

class TestLoadRunConfig:
    def test_reads_config_toml(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            '[model]\nlabel = "Heisenberg"\n[algorithm]\nmax_bond = 64\n'
        )
        cfg = load_run_config(tmp_path)
        assert cfg["model"]["label"] == "Heisenberg"
        assert cfg["algorithm"]["max_bond"] == 64

    def test_raises_when_absent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_run_config(tmp_path)


# ---------------------------------------------------------------------------
# gres field in SlurmJobConfig
# ---------------------------------------------------------------------------

class TestSlurmJobConfigGres:
    def test_default_gres_is_empty_string(self):
        cfg = SlurmJobConfig()
        assert cfg.gres == ""

    def test_gres_round_trip_via_load(self, tmp_path):
        p = tmp_path / "slurm.toml"
        p.write_text('[basic]\n[main]\ngres = "gpu:a100:2"\n[exec]\n')
        cfg = load_slurm_toml(p)
        assert cfg.main.gres == "gpu:a100:2"

    def test_gres_emitted_in_script(self, tmp_path):
        from intraknot.launch import write_slurm_script
        from helpers import make_machine
        run_dir = tmp_path / "run01"
        (run_dir / "main" / "logs").mkdir(parents=True)
        p = run_dir / "slurm.toml"
        p.write_text(
            '[basic]\n'
            '[main]\npartition="gpu"\ntime="02:00:00"\nmem="32000"\n'
            'ntasks=1\nnodes=1\ncpus_per_task=8\ngres="gpu:a100:1"\n'
            '[exec]\n'
        )
        script = write_slurm_script(run_dir, make_machine(), "run01")
        assert "--gres=gpu:a100:1" in script.read_text()
