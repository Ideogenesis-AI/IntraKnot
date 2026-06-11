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

Three namespaces are kept strictly separate:

    configs/paths.toml
        Machine filesystem settings — where to find the Python interpreter,
        scratch directories, etc.

    slurm.toml  (lives in configs/, campaigns/, and runs/)
        Slurm scheduling settings — account, partition, resource requests.
        The file at configs/ is the master template; it is copied verbatim to
        each campaign on creation, and from there to each run. Users edit the
        copies to customise settings at the desired granularity.

    configs/cluster.yaml
        Auto-generated cluster topology written by `iknot cluster sync`.
        Records the hostname, discovery timestamp, per-partition node groups
        (hardware specs in Slurm bracket notation), and a top-level
        feature/constraint index. Never edit this file by hand.

    runs/<run_id>/config.toml
        Scientific simulation configuration — what to run (Alice-compatible).

File format dispatch: `.toml` files are loaded with stdlib `tomllib`;
`.yaml` / `.yml` files are loaded with `pyyaml`.
"""

from __future__ import annotations

import itertools
import os
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
    threads_per_core:
        `--threads-per-core` value. Set to `0` to omit the directive.
    gres:
        Generic resource string passed verbatim to `--gres`
        (e.g. `"gpu:a100:2"`). Leave empty to omit.
    """

    partition: str = ""
    constraint: str = ""
    time: str = "04:00:00"
    mem: str = "16000"
    ntasks: int = 1
    nodes: int = 1
    cpus_per_task: int = 8
    threads_per_core: int = 0
    gres: str = ""


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
# Cluster discovery dataclasses (populated by `iknot cluster sync`)
# ---------------------------------------------------------------------------

@dataclass
class NodeGroup:
    """A group of nodes with identical hardware specs within a partition.

    Nodes are grouped by name prefix first, then split by `(cpus, mem_mb,
    gres)` so that every entry in the group has uniform hardware. The node
    list is stored in Slurm bracket notation (e.g. `"th-cl-hua[01-10]"`).

    Parameters
    ----------
    nodes:
        Slurm bracket-notation node list, e.g. `"th-cl-hua[01-10]"`.
    cpus:
        Number of CPU cores per node.
    mem_mb:
        Memory per node in MiB.
    gres:
        Generic Resource string (e.g. `"gpu:a100:4"`), or `""` if none.
    features:
        Union of all feature/constraint labels present on nodes in this
        group. Labels are admin-assigned and may not be perfectly uniform
        across the group.
    """

    nodes: str
    cpus: int
    mem_mb: int
    gres: str
    features: List[str]


@dataclass
class PartitionInfo:
    """Metadata for one Slurm partition discovered by `iknot cluster sync`.

    Parameters
    ----------
    name:
        Partition name.
    state:
        Availability state: `"up"`, `"down"`, or `"drain"`.
    default:
        `True` when this is the cluster's default partition.
    time_limit:
        Maximum walltime string as reported by Slurm (e.g. `"21-00:00:00"`).
    node_groups:
        Hardware-uniform node groups within this partition.
    """

    name: str
    state: str
    default: bool
    time_limit: str
    node_groups: List[NodeGroup]


@dataclass
class FeatureInfo:
    """Summary of a single Slurm feature (constraint) across all partitions.

    Parameters
    ----------
    partitions:
        Names of partitions that contain at least one node tagged with this
        feature.
    node_count:
        Total number of nodes tagged with this feature across all partitions.
        Nodes belonging to multiple partitions are counted once per
        partition entry.
    """

    partitions: List[str]
    node_count: int


@dataclass
class ClusterDiscovery:
    """Full cluster topology snapshot written to `configs/cluster.yaml`.

    Parameters
    ----------
    hostname:
        Login-node hostname as returned by `socket.getfqdn()`.
    discovered_at:
        ISO-8601 timestamp of when `iknot cluster sync` was run.
    partitions:
        Ordered list of discovered partitions.
    features:
        Mapping from feature/constraint name to its `FeatureInfo` summary.
    """

    hostname: str
    discovered_at: str
    partitions: List[PartitionInfo]
    features: Dict[str, FeatureInfo]


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

    _defaults = SlurmJobConfig()

    def _job(d: dict) -> SlurmJobConfig:
        return SlurmJobConfig(
            partition=d.get("partition", _defaults.partition),
            constraint=d.get("constraint", _defaults.constraint),
            time=d.get("time", _defaults.time),
            mem=str(d.get("mem", _defaults.mem)),
            ntasks=int(d.get("ntasks", _defaults.ntasks)),
            nodes=int(d.get("nodes", _defaults.nodes)),
            cpus_per_task=int(d.get("cpus_per_task", _defaults.cpus_per_task)),
            gres=d.get("gres", _defaults.gres),
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


def load_cluster_discovery(configs_dir: Path) -> Optional[ClusterDiscovery]:
    """Load `configs/cluster.yaml` into a `ClusterDiscovery`, or return `None`.

    Returns `None` when `cluster.yaml` is absent (i.e. `iknot cluster sync`
    has not yet been run) so callers can distinguish "missing" from "empty".

    Parameters
    ----------
    configs_dir:
        Directory containing `cluster.yaml`.

    Returns
    -------
    ClusterDiscovery or None
        Parsed discovery snapshot, or `None` if the file does not exist.
    """
    path = configs_dir / "cluster.yaml"
    if not path.exists():
        return None

    raw = load_config(path)

    partitions: List[PartitionInfo] = []
    for p in raw.get("partitions", []):
        groups: List[NodeGroup] = []
        for g in p.get("node_groups", []):
            groups.append(NodeGroup(
                nodes=g["nodes"],
                cpus=int(g["cpus"]),
                mem_mb=int(g["mem_mb"]),
                gres=g.get("gres", ""),
                features=list(g.get("features", [])),
            ))
        partitions.append(PartitionInfo(
            name=p["name"],
            state=p.get("state", ""),
            default=bool(p.get("default", False)),
            time_limit=p.get("time_limit", ""),
            node_groups=groups,
        ))

    features: Dict[str, FeatureInfo] = {}
    for fname, fdata in raw.get("features", {}).items():
        features[fname] = FeatureInfo(
            partitions=list(fdata.get("partitions", [])),
            node_count=int(fdata.get("node_count", 0)),
        )

    return ClusterDiscovery(
        hostname=raw.get("hostname", ""),
        discovered_at=raw.get("discovered_at", ""),
        partitions=partitions,
        features=features,
    )


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
ntasks           = 1           # --ntasks (1 for threaded jobs)
nodes            = 1           # --nodes
cpus_per_task    = 8           # number of CPU threads
threads_per_core = 1           # --threads-per-core; set 0 to omit

[exec]
partition        = "_init_"    # can differ from [main] for lighter follow-up jobs
constraint       = "_init_"    # set "" to omit
time             = "01:00:00"
mem              = "8000"
ntasks           = 1
nodes            = 1
cpus_per_task    = 4
threads_per_core = 1           # --threads-per-core; set 0 to omit
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

_TUI_TOML_TEMPLATE = """\
# configs/tui.toml
# Settings for `iknot tui`.

[tui]
# Editor used to open log files (e.g. alice.log).
# The dashboard suspends, launches this editor in the foreground, then resumes.
editor = "vi"
"""

_REGISTRY_YAML_TEMPLATE = """\
# configs/registry.yaml
# Unified algorithm database registry.
# Managed by iknot database commands — do not edit manually.
#
# Add a database:  iknot database add <name> <url>
# Update all:      iknot database update

databases:
  intraknot-database:
    url: https://github.com/Ideogenesis-AI/intraknot-database
    registry_url: https://raw.githubusercontent.com/Ideogenesis-AI/intraknot-database/main/registry.yaml
    fetched_at: ''
    description: ''
"""

_GITIGNORE_CONTENT = "*\n"


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


def write_tui_toml(path: Path) -> None:
    """Write the `tui.toml` template to `path`.

    Parameters
    ----------
    path:
        Destination file path (typically `configs/tui.toml`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TUI_TOML_TEMPLATE)


def write_registry_yaml(path: Path) -> None:
    """Write the `registry.yaml` template to `path`.

    The template pre-registers the `intraknot-database` entry with an empty
    `fetched_at` field. Running `iknot database update` will populate it once
    the repository is live.

    Parameters
    ----------
    path:
        Destination file path (typically `configs/registry.yaml`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_REGISTRY_YAML_TEMPLATE)


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
        except ValueError as e:
            raise ValueError(f"Cannot parse {raw!r} as int") from e
    if isinstance(reference, float):
        try:
            return float(raw)
        except ValueError as e:
            raise ValueError(f"Cannot parse {raw!r} as float") from e
    if isinstance(reference, list):
        # Split on commas and apply element-wise inference using the type of
        # the first element (or str when the reference list is empty).
        element_ref = reference[0] if reference else ""
        return [_infer_typed_value(item.strip(), element_ref) for item in raw.split(",")]
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


def resolve_active_campaign() -> Optional[str]:
    """Return the active campaign identifier, or `None` if none is set.

    Resolution order:

    1. `INTRAKNOT_CAMPAIGN` environment variable.
    2. `active_campaign` key in `.iknot_state` TOML at the current directory.
    3. `None` — no active campaign.
    """
    env_val = os.environ.get("INTRAKNOT_CAMPAIGN")
    if env_val:
        return env_val
    state_file = Path.cwd() / ".iknot_state"
    if state_file.exists():
        try:
            with open(state_file, "rb") as f:
                state = tomllib.load(f)
            return state.get("active_campaign")
        except Exception:
            pass
    return None


def make_run_id(merged_cfg: Dict[str, Any], flat_overrides: Dict[str, Any], uuid8: str) -> str:
    """Build an auto-generated run identifier from merged config and overrides.

    The identifier follows the format:

        <engine>_<model_label>_<lattice_descriptor>[_<overrides>]_<uuid8>

    where `lattice_descriptor` is `<lattice>_len=<lx>` for 1-D geometries
    and `<lattice>_cell=<lx>x<ly>` for 2-D geometries (when `ly > 1`).
    Override pairs are sorted alphabetically; floats use `g`-format.

    Parameters
    ----------
    merged_cfg:
        Fully merged run configuration (campaign defaults + run overrides).
    flat_overrides:
        `{leafkey: value}` mapping of the explicitly overridden parameters.
        Omitted from the ID when empty.
    uuid8:
        8-character hex suffix (typically the first 8 chars of a UUID4
        with hyphens removed, e.g. `"a3f7b291"`).

    Returns
    -------
    str
        Run ID such as `"dmrg_heisenberg_chain_len=20_max_bond=128_a3f7b291"`.
    """
    geo = merged_cfg.get("geometry", {})
    alg = merged_cfg.get("algorithm", {})
    mdl = merged_cfg.get("model", {})

    engine = str(alg.get("engine", "run")).lower()
    label = str(mdl.get("label", "unknown")).lower()

    lattice = str(geo.get("lattice", "chain")).lower()
    lx = geo.get("lx", 0)
    ly = geo.get("ly", None)
    if ly is not None and int(ly) > 1:
        lattice_desc = f"{lattice}_cell={lx}x{ly}"
    else:
        lattice_desc = f"{lattice}_len={lx}"

    parts: List[str] = [engine, label, lattice_desc]
    if flat_overrides:
        overrides_str = "_".join(
            f"{k}={_format_value(v)}" for k, v in sorted(flat_overrides.items())
        )
        parts.append(overrides_str)
    parts.append(uuid8)

    return "_".join(parts)


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
    if user_cfg is not None:
        merged.update(user_cfg)

    # Apply campaign defaults for every section present in defaults.toml.
    # Iterating over campaign_defaults.keys() means new sections (e.g.
    # [optimizer]) are merged automatically without a code change here.
    for section in campaign_defaults:
        if section not in merged:
            merged[section] = dict(campaign_defaults[section])
        else:
            # Merge key-by-key: run values win.
            base = dict(campaign_defaults[section])
            base.update(merged[section])
            merged[section] = base

    return merged
