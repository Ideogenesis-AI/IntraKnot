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


"""Config loaders and template generators for IntraKnot.

Two namespaces are kept strictly separate:

    configs/slurm.toml + configs/paths.toml
        Machine and scheduler settings — where and how to run.

    runs/<run_id>/config.toml
        Scientific simulation configuration — what to run (Alice-compatible).

File format dispatch: `.toml` files are loaded with stdlib `tomllib`;
`.yaml` / `.yml` files are loaded with `pyyaml`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SlurmConfig:
    """Slurm scheduler settings read from `configs/slurm.toml`.

    Parameters
    ----------
    account:
        Slurm account name.
    partition:
        Target partition (queue).
    default_time:
        Default walltime string, e.g. `"04:00:00"`.
    default_mem:
        Default memory per node, e.g. `"16G"`.
    default_cpus_per_task:
        Default number of CPUs per Slurm task.
    """

    account: str = ""
    partition: str = ""
    default_time: str = "04:00:00"
    default_mem: str = "16G"
    default_cpus_per_task: int = 8


@dataclass
class PathsConfig:
    """Filesystem path settings read from `configs/paths.toml`.

    Parameters
    ----------
    run_root:
        Root directory where run subdirectories are created.
    scratch_root:
        Fast scratch filesystem root (used by jobs for temporary data).
    node_local_scratch:
        Per-node local scratch (typically `/tmp/$USER`).
    python:
        Command used to invoke the runner script from Slurm.
        Defaults to `"uv run"`.
    """

    run_root: str = ""
    scratch_root: str = ""
    node_local_scratch: str = "/tmp/$USER"
    python: str = "uv run"


@dataclass
class MachineConfig:
    """Combined machine configuration assembled from `slurm.toml` + `paths.toml`.

    Parameters
    ----------
    slurm:
        Slurm scheduler settings.
    paths:
        Filesystem path settings.
    """

    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


# ---------------------------------------------------------------------------
# Generic loader (dispatch by extension)
# ---------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    """Load a TOML or YAML config file and return the parsed dict.

    Dispatches by file extension: `.toml` → `tomllib`; `.yaml` / `.yml` →
    `pyyaml`. All other extensions raise `ValueError`.

    Parameters
    ----------
    path:
        Path to the config file.

    Returns
    -------
    dict
        Parsed contents.

    Raises
    ------
    ValueError
        For unsupported file extensions.
    FileNotFoundError
        If `path` does not exist.
    """
    suffix = path.suffix.lower()
    if suffix == ".toml":
        with open(path, "rb") as f:
            return tomllib.load(f)
    if suffix in (".yaml", ".yml"):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    raise ValueError(f"Unsupported config file extension: {path.suffix!r}")


# ---------------------------------------------------------------------------
# Machine config loaders
# ---------------------------------------------------------------------------

def load_machine_config(configs_dir: Path) -> MachineConfig:
    """Assemble a `MachineConfig` from `configs/slurm.toml` and `configs/paths.toml`.

    Missing keys are filled with dataclass defaults so that the caller never
    needs to guard against absent optional fields.

    Parameters
    ----------
    configs_dir:
        Directory containing `slurm.toml` and `paths.toml`.

    Returns
    -------
    MachineConfig
        Populated machine configuration.
    """
    slurm_cfg = SlurmConfig()
    paths_cfg = PathsConfig()

    slurm_path = configs_dir / "slurm.toml"
    if slurm_path.exists():
        raw = load_config(slurm_path).get("slurm", {})
        slurm_cfg = SlurmConfig(
            account=raw.get("account", slurm_cfg.account),
            partition=raw.get("partition", slurm_cfg.partition),
            default_time=raw.get("default_time", slurm_cfg.default_time),
            default_mem=raw.get("default_mem", slurm_cfg.default_mem),
            default_cpus_per_task=raw.get("default_cpus_per_task", slurm_cfg.default_cpus_per_task),
        )

    paths_path = configs_dir / "paths.toml"
    if paths_path.exists():
        raw = load_config(paths_path).get("paths", {})
        paths_cfg = PathsConfig(
            run_root=raw.get("run_root", paths_cfg.run_root),
            scratch_root=raw.get("scratch_root", paths_cfg.scratch_root),
            node_local_scratch=raw.get("node_local_scratch", paths_cfg.node_local_scratch),
            python=raw.get("python", paths_cfg.python),
        )

    return MachineConfig(slurm=slurm_cfg, paths=paths_cfg)


def load_campaign_defaults(campaign_dir: Path) -> Dict[str, Any]:
    """Load optional `defaults.toml` from a campaign directory.

    Returns an empty dict if the file is absent, so callers can always safely
    call `defaults.get("algorithm", {})` etc.

    Parameters
    ----------
    campaign_dir:
        Campaign directory (e.g. `campaigns/heisenberg_dmrg_chi_scan/`).

    Returns
    -------
    dict
        Parsed TOML contents, or `{}` if `defaults.toml` does not exist.
    """
    path = campaign_dir / "defaults.toml"
    if not path.exists():
        return {}
    return load_config(path)


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    """Load the scientific configuration from `runs/<run_id>/config.toml`.

    The returned dict is Alice-compatible: `config["model"]` can be passed
    directly to `alice.build_interaction()`, and `config["algorithm"]` feeds
    `dmrg.Options`.

    Parameters
    ----------
    run_dir:
        Run directory containing `config.toml`.

    Returns
    -------
    dict
        Full parsed TOML contents.

    Raises
    ------
    FileNotFoundError
        If `config.toml` is absent from `run_dir`.
    """
    path = run_dir / "config.toml"
    if not path.exists():
        raise FileNotFoundError(f"config.toml not found in {run_dir}")
    return load_config(path)


# ---------------------------------------------------------------------------
# Template generators (called by `iknot init`)
# ---------------------------------------------------------------------------

_MACHINES_YAML_TEMPLATE = """\
# configs/machines.yaml
# Registry of known HPC clusters. This file is for human reference only;
# machine-specific submission settings live in slurm.toml and paths.toml.
#
# machines:
#   - name: cluster_a
#     hostname: login.cluster-a.example.com
#     notes: Primary production cluster.
#   - name: local
#     hostname: localhost
#     notes: Local development machine.
machines: []
"""

_SLURM_TOML_TEMPLATE = """\
# configs/slurm.toml
# Slurm scheduler settings. Fill in the values for your cluster.

[slurm]
account           = ""          # Slurm account / project code
partition         = ""          # Target partition (queue)
default_time      = "04:00:00"  # Default walltime (HH:MM:SS)
default_mem       = "16G"       # Default memory per node
default_cpus_per_task = 8       # Default CPUs per task
"""

_PATHS_TOML_TEMPLATE = """\
# configs/paths.toml
# Filesystem path settings. Fill in the paths for your cluster.

[paths]
run_root           = ""          # Root directory for run subdirectories
scratch_root       = ""          # Fast scratch filesystem root
node_local_scratch = "/tmp/$USER"  # Per-node local scratch
python             = "uv run"    # Command used to invoke the runner script
"""

_GITIGNORE_CONTENT = "*\n"


def write_machines_yaml(path: Path) -> None:
    """Write a skeleton `machines.yaml` registry to `path`.

    Parameters
    ----------
    path:
        Destination file path (typically `configs/machines.yaml`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_MACHINES_YAML_TEMPLATE)


def write_slurm_toml(path: Path) -> None:
    """Write a placeholder `slurm.toml` template to `path`.

    Parameters
    ----------
    path:
        Destination file path (typically `configs/slurm.toml`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_SLURM_TOML_TEMPLATE)


def write_paths_toml(path: Path) -> None:
    """Write a placeholder `paths.toml` template to `path`.

    Parameters
    ----------
    path:
        Destination file path (typically `configs/paths.toml`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_PATHS_TOML_TEMPLATE)


def write_data_gitignore(directory: Path) -> None:
    """Write a `.gitignore` that excludes all directory contents except itself.

    This is used for `configs/`, `campaigns/`, `runs/`, and `notebooks/` so
    that data and credentials are never accidentally committed.

    Parameters
    ----------
    directory:
        Target directory. Created if absent.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".gitignore").write_text(_GITIGNORE_CONTENT)


def _merge_defaults(
    user_cfg: Optional[Dict[str, Any]],
    campaign_defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge campaign-level defaults with a run-level config dict.

    The campaign provides fallback `[model]`, `[algorithm]`, and `[output]`
    sections. Run-level keys take precedence over campaign defaults at every
    level. This allows a campaign's `defaults.toml` to define the full model
    for a parameter study, with individual runs overriding only the keys that
    differ.

    Parameters
    ----------
    user_cfg:
        Run-level TOML dict. If `None`, the campaign defaults alone form the
        starting configuration.
    campaign_defaults:
        Dict loaded from `campaign/defaults.toml`.

    Returns
    -------
    dict
        Merged configuration dict.
    """
    merged: Dict[str, Any] = {}
    if user_cfg:
        merged.update(user_cfg)

    # Apply campaign defaults for sections absent in the run config.
    for section in ("model", "algorithm", "output"):
        if section in campaign_defaults and section not in merged:
            merged[section] = dict(campaign_defaults[section])
        elif section in campaign_defaults and section in merged:
            # Merge key-by-key: run values win.
            base = dict(campaign_defaults[section])
            base.update(merged[section])
            merged[section] = base

    return merged
