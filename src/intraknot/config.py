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

    configs/paths.toml
        Machine filesystem settings — where to find the Python interpreter,
        scratch directories, etc.

    slurm.toml  (lives in configs/, campaigns/, and runs/)
        Slurm scheduling settings — account, partition, resource requests.
        The file at configs/ is the master template; it is copied verbatim to
        each campaign on creation, and from there to each run. Users edit the
        copies to customise settings at the desired granularity.

    runs/<run_id>/config.toml
        Scientific simulation configuration — what to run (Alice-compatible).

File format dispatch: `.toml` files are loaded with stdlib `tomllib`;
`.yaml` / `.yml` files are loaded with `pyyaml`.
"""

from __future__ import annotations

import itertools
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SlurmBasicConfig:
    """Fields in the `[basic]` section of `slurm.toml`.

    These are shared by every job (primary and exec) in a run.

    Parameters
    ----------
    account:
        Slurm account / project code.
    mail_type:
        `--mail-type` value, e.g. `"ALL"` or `"END,FAIL"`. Leave empty to
        omit the directive entirely.
    mail_user:
        Email address for job notifications. Leave empty to omit.
    """

    account: str = ""
    mail_type: str = ""
    mail_user: str = ""


@dataclass
class SlurmJobConfig:
    """Fields in a `[main]` or `[exec]` section of `slurm.toml`.

    Parameters
    ----------
    partition:
        Target Slurm partition (queue).
    constraint:
        Node constraint string passed to `-C`. Leave empty to omit.
    time:
        Walltime string, e.g. `"504:00:00"`.
    mem:
        Memory per node passed verbatim to `--mem` (e.g. `"300000"` for
        300 000 MB, or `"300G"`).
    ntasks:
        `--ntasks` value. Typically 1 for threaded (non-MPI) jobs.
    nodes:
        `--nodes` value.
    cpus_per_task:
        `--cpus-per-task` value (number of CPU threads).
    """

    partition: str = ""
    constraint: str = ""
    time: str = "04:00:00"
    mem: str = "16000"
    ntasks: int = 1
    nodes: int = 1
    cpus_per_task: int = 8


@dataclass
class SlurmTomlConfig:
    """Full contents of a `slurm.toml` file.

    Parameters
    ----------
    basic:
        Shared account and mail settings.
    main:
        Resource settings for the primary (DMRG) job.
    exec_:
        Resource settings for exec (follow-up) jobs.
    """

    basic: SlurmBasicConfig = field(default_factory=SlurmBasicConfig)
    main: SlurmJobConfig = field(default_factory=SlurmJobConfig)
    exec_: SlurmJobConfig = field(default_factory=SlurmJobConfig)


@dataclass
class PathsConfig:
    """Filesystem path settings read from `configs/paths.toml`.

    Parameters
    ----------
    project_root:
        Root directory where run subdirectories are created.
    scratch_root:
        Fast scratch filesystem root (used by jobs for temporary data).
    scratch_node:
        Per-node local scratch (typically `/tmp/$USER`).
    command:
        Command used to invoke the runner script from Slurm.
        Defaults to `"uv run"`.
    """

    project_root: str = ""
    scratch_root: str = ""
    scratch_node: str = "/tmp/$USER"
    command: str = "uv run"


@dataclass
class MachineConfig:
    """Machine configuration assembled from `configs/paths.toml`.

    Slurm scheduling settings are no longer part of `MachineConfig`; they
    live in `slurm.toml` at the run level and are loaded on demand by
    `load_slurm_toml`.

    Parameters
    ----------
    paths:
        Filesystem path settings.
    """

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
# Slurm TOML loader
# ---------------------------------------------------------------------------

def load_slurm_toml(path: Path) -> SlurmTomlConfig:
    """Load a `slurm.toml` file into a `SlurmTomlConfig`.

    Returns a default-filled `SlurmTomlConfig` when the file is absent so
    callers never need to guard against a missing `slurm.toml`.

    Parameters
    ----------
    path:
        Path to the `slurm.toml` file.

    Returns
    -------
    SlurmTomlConfig
        Parsed Slurm settings with dataclass defaults for missing keys.
    """
    if not path.exists():
        return SlurmTomlConfig()

    raw = load_config(path)
    basic_raw = raw.get("basic", {})
    main_raw = raw.get("main", {})
    exec_raw = raw.get("exec", {})

    basic = SlurmBasicConfig(
        account=basic_raw.get("account", ""),
        mail_type=basic_raw.get("mail_type", ""),
        mail_user=basic_raw.get("mail_user", ""),
    )

    def _job(d: dict) -> SlurmJobConfig:
        return SlurmJobConfig(
            partition=d.get("partition", ""),
            constraint=d.get("constraint", ""),
            time=d.get("time", "04:00:00"),
            mem=str(d.get("mem", "16000")),
            ntasks=int(d.get("ntasks", 1)),
            nodes=int(d.get("nodes", 1)),
            cpus_per_task=int(d.get("cpus_per_task", 8)),
        )

    return SlurmTomlConfig(basic=basic, main=_job(main_raw), exec_=_job(exec_raw))


# ---------------------------------------------------------------------------
# Machine config loader
# ---------------------------------------------------------------------------

def load_machine_config(configs_dir: Path) -> MachineConfig:
    """Assemble a `MachineConfig` from `configs/paths.toml`.

    Missing keys are filled with dataclass defaults.

    Parameters
    ----------
    configs_dir:
        Directory containing `paths.toml`.

    Returns
    -------
    MachineConfig
        Populated machine configuration.
    """
    paths_cfg = PathsConfig()

    paths_path = configs_dir / "paths.toml"
    if paths_path.exists():
        raw = load_config(paths_path).get("paths", {})
        paths_cfg = PathsConfig(
            project_root=raw.get("project_root", paths_cfg.project_root),
            scratch_root=raw.get("scratch_root", paths_cfg.scratch_root),
            scratch_node=raw.get("scratch_node", paths_cfg.scratch_node),
            command=raw.get("command", paths_cfg.command),
        )

    return MachineConfig(paths=paths_cfg)


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
# Template generators (called by `iknot init` and campaign/run creation)
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
# slurm.toml
# Slurm scheduling settings.
#
# This file is the master template. It is copied verbatim to each campaign
# on creation, and from there to each run. Edit the campaign copy for
# campaign-wide defaults; edit the run copy for per-run overrides.
#
# Fields marked "_init_" must be set before submitting any job.
# Optional fields (mail_type, mail_user) can be left as "" to omit the
# corresponding #SBATCH directive entirely.

[basic]
account   = "_init_"    # Slurm account / project code
mail_type = ""          # --mail-type (e.g. "ALL", "END,FAIL"); leave "" to omit
mail_user = ""          # email address for notifications; leave "" to omit

[main]
partition     = "_init_"    # target partition (queue)
constraint    = "_init_"    # node constraint (-C); set "" to omit
time          = "04:00:00"  # walltime (HH:MM:SS)
mem           = "16000"     # memory per node (MB integer or e.g. "16G")
ntasks        = 1           # --ntasks (1 for threaded jobs)
nodes         = 1           # --nodes
cpus_per_task = 8           # number of CPU threads

[exec]
partition     = "_init_"    # can differ from [main] for lighter follow-up jobs
constraint    = "_init_"    # set "" to omit
time          = "01:00:00"
mem           = "8000"
ntasks        = 1
nodes         = 1
cpus_per_task = 4
"""

_PATHS_TOML_TEMPLATE = """\
# configs/paths.toml
# Filesystem path settings. Fill in the paths for your cluster.

[paths]
project_root = ""           # Root directory for run subdirectories
scratch_root = ""           # Fast scratch filesystem root
scratch_node = "/tmp/$USER" # Per-node local scratch
command      = "uv run"     # Command used to invoke the runner script
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
    """Write the `slurm.toml` template to `path`.

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


def _format_value(v: Any) -> str:
    """Format a scalar config value for use in a run ID.

    Floats are rendered with Python's `g`-format to suppress trailing zeros.
    All other types are converted with `str`.

    Parameters
    ----------
    v:
        Scalar value (int, float, bool, or str).

    Returns
    -------
    str
        Human-readable, filesystem-safe representation.
    """
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _infer_typed_value(raw: str, reference: Any) -> Any:
    """Parse `raw` into the same Python type as `reference`.

    Parameters
    ----------
    raw:
        Raw string from a `--set` argument value.
    reference:
        The existing default value for this key, used to determine the target
        type.

    Returns
    -------
    Any
        Parsed value with the same type as `reference`.

    Raises
    ------
    ValueError
        When `raw` cannot be converted to the type of `reference`.
    """
    if isinstance(reference, bool):
        if raw.lower() in ("true", "1", "yes"):
            return True
        if raw.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"Cannot parse {raw!r} as bool")
    if isinstance(reference, int):
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"Cannot parse {raw!r} as int")
    if isinstance(reference, float):
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"Cannot parse {raw!r} as float")
    # str or unknown: return as-is
    return raw


def _parse_one_set_arg(
    arg: str,
    defaults: Dict[str, Any],
) -> Tuple[str, str, List[Any]]:
    """Parse a single `--set` argument string.

    Parameters
    ----------
    arg:
        A string of the form `"section.key=value"` or
        `"section.key=v1,v2,v3"`.
    defaults:
        Campaign defaults dict, used for type inference.

    Returns
    -------
    tuple[str, str, list]
        `(section, key, [typed_value, ...])` — always a list, length >= 1.

    Raises
    ------
    ValueError
        For malformed arguments or unknown keys.
    """
    if "=" not in arg:
        raise ValueError(
            f"Invalid --set argument {arg!r}: expected 'section.key=value'."
        )
    lhs, rhs = arg.split("=", 1)
    parts = lhs.split(".")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --set key {lhs!r}: expected 'section.key' (exactly one dot)."
        )
    section, key = parts

    # Look up the default for type inference.
    section_defaults = defaults.get(section, {})
    if key not in section_defaults:
        raise ValueError(
            f"Unknown key {lhs!r}: not found in campaign defaults."
        )
    reference = section_defaults[key]

    raw_values = rhs.split(",")
    typed_values = [_infer_typed_value(v.strip(), reference) for v in raw_values]
    return section, key, typed_values


def parse_set_overrides(
    set_args: List[str],
    defaults: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Parse a list of `--set section.key=value` strings into a config dict.

    Only accepts single-valued arguments (no comma-separated lists). Use
    `iter_scan_combinations` when multi-valued arguments are present.

    Parameters
    ----------
    set_args:
        List of strings of the form `"section.key=value"`.
    defaults:
        Campaign defaults dict, used for type inference and key validation.

    Returns
    -------
    dict
        Nested `{section: {key: typed_value}}` suitable for passing to
        `_merge_defaults` as the `user_cfg` argument.

    Raises
    ------
    ValueError
        For malformed arguments, unknown keys, or multi-valued entries.
    """
    overrides: Dict[str, Dict[str, Any]] = {}
    for arg in set_args:
        section, key, values = _parse_one_set_arg(arg, defaults)
        if len(values) > 1:
            raise ValueError(
                f"--set {arg!r} contains multiple values. "
                "Use iter_scan_combinations for multi-valued arguments."
            )
        overrides.setdefault(section, {})[key] = values[0]
    return overrides


def iter_scan_combinations(
    set_args: List[str],
    defaults: Dict[str, Any],
) -> Iterator[Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]]:
    """Yield one override dict per combination of `--set` argument values.

    When all `--set` arguments have exactly one value this yields a single
    item. When any argument has comma-separated values (e.g.
    `"algorithm.max_bond=64,128,256"`) the cartesian product over all
    multi-valued arguments is produced.

    Parameters
    ----------
    set_args:
        List of strings of the form `"section.key=value"` or
        `"section.key=v1,v2,v3"`.
    defaults:
        Campaign defaults dict, used for type inference and key validation.

    Yields
    ------
    tuple[dict, dict]
        `(nested_overrides, flat_overrides)` where `nested_overrides` is
        `{section: {key: value}}` (suitable for `_merge_defaults`) and
        `flat_overrides` is `{leafkey: value}` (suitable for `make_run_id`).

    Raises
    ------
    ValueError
        For malformed arguments or unknown keys.
    """
    parsed: List[Tuple[str, str, List[Any]]] = [
        _parse_one_set_arg(arg, defaults) for arg in set_args
    ]

    # Build the list of (section, key) axes and their value lists for the
    # cartesian product.
    keys: List[Tuple[str, str]] = [(s, k) for s, k, _ in parsed]
    value_lists: List[List[Any]] = [vs for _, _, vs in parsed]

    for combo in itertools.product(*value_lists):
        nested: Dict[str, Dict[str, Any]] = {}
        flat: Dict[str, Any] = {}
        for (section, key), value in zip(keys, combo):
            nested.setdefault(section, {})[key] = value
            flat[key] = value
        yield nested, flat


def make_run_id(campaign_id: str, flat_overrides: Dict[str, Any]) -> str:
    """Build an auto-generated run identifier from campaign name and overrides.

    Keys are sorted alphabetically. Floats are formatted with Python's
    `g`-format. The separator between key and value is `=`; pairs are
    joined by `_`.

    Parameters
    ----------
    campaign_id:
        Campaign identifier (e.g. `"test_heisenberg_dmrg"`).
    flat_overrides:
        `{leafkey: value}` mapping of the overridden parameters.

    Returns
    -------
    str
        Run ID such as `"test_heisenberg_dmrg_lx=40_max_bond=128"`.
    """
    if not flat_overrides:
        return campaign_id
    pairs = "_".join(
        f"{k}={_format_value(v)}" for k, v in sorted(flat_overrides.items())
    )
    return f"{campaign_id}_{pairs}"


def _merge_defaults(
    user_cfg: Optional[Dict[str, Any]],
    campaign_defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge campaign-level defaults with a run-level config dict.

    The campaign provides fallback `[geometry]`, `[model]`, `[algorithm]`,
    and `[output]` sections. Run-level keys take precedence over campaign
    defaults at every section. This allows a campaign's `defaults.toml` to
    define the full physical model for a parameter study, with individual
    runs overriding only the keys that differ.

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
    for section in ("geometry", "model", "algorithm", "output"):
        if section in campaign_defaults and section not in merged:
            merged[section] = dict(campaign_defaults[section])
        elif section in campaign_defaults and section in merged:
            # Merge key-by-key: run values win.
            base = dict(campaign_defaults[section])
            base.update(merged[section])
            merged[section] = base

    return merged
