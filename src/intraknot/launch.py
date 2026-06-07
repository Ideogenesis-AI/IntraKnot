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


"""Directory creation, algorithm file management, and Slurm job submission."""

from __future__ import annotations

import csv
import datetime
import importlib.resources
import shutil
import subprocess
import uuid as _uuid_lib
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .config import (
    MachineConfig,
    SlurmJobConfig,
    SlurmTomlConfig,
    _merge_defaults,
    load_campaign_defaults,
    load_config,
    load_slurm_toml,
    write_slurm_toml,
)
from .status import MainStatus, RunState, write_status


# ---------------------------------------------------------------------------
# Minimal TOML writer
# ---------------------------------------------------------------------------

def _toml_value(v: object) -> str:
    """Serialize a scalar or list value to a TOML-compatible string."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        # Escape backslashes and double quotes.
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(v, list):
        items = ", ".join(_toml_value(item) for item in v)
        return f"[{items}]"
    raise TypeError(f"Cannot serialize type {type(v).__name__!r} to TOML")


def _dump_toml(data: dict, _prefix: str = "") -> str:
    """Serialize a nested dict to a TOML string.

    Handles two levels of nesting (e.g. `[model.geometry]`) as used in
    IntraKnot configs. Inline tables and arrays-of-tables are not supported.

    Parameters
    ----------
    data:
        Config dict to serialize.
    _prefix:
        Internal use; dot-separated prefix for nested table headers.

    Returns
    -------
    str
        TOML text.
    """
    lines: list[str] = []
    deferred: list[tuple[str, dict]] = []

    for key, value in data.items():
        full_key = f"{_prefix}.{key}" if _prefix else key
        if isinstance(value, dict):
            deferred.append((full_key, value))
        else:
            lines.append(f"{key} = {_toml_value(value)}")

    header_block = "\n".join(lines) + "\n" if lines else ""

    sub_blocks = []
    for full_key, sub in deferred:
        sub_blocks.append(f"[{full_key}]\n{_dump_toml(sub, _prefix=full_key)}")

    return header_block + "\n".join(sub_blocks)


# ---------------------------------------------------------------------------
# Slurm script rendering
# ---------------------------------------------------------------------------

def _render_slurm_header(directives: Dict[str, str]) -> str:
    """Build `#SBATCH` directive lines from a flag-to-value mapping.

    Entries whose value is empty or `None` are silently skipped, making
    optional directives (e.g. `--constraint`, `--mail-type`) easy to omit.

    Parameters
    ----------
    directives:
        Ordered mapping of `--flag` (or `-f`) to value string.

    Returns
    -------
    str
        Newline-separated `#SBATCH` lines (no trailing newline).
    """
    lines = []
    for flag, value in directives.items():
        if value:
            lines.append(f"#SBATCH {flag}={value}")
    return "\n".join(lines)


def _build_single_script(
    run_dir: Path,
    slurm: SlurmTomlConfig,
    python_cmd: str,
    run_id: str,
) -> str:
    """Render the Slurm submit script for a single primary job.

    Parameters
    ----------
    run_dir:
        Absolute path to the run directory.
    slurm:
        Slurm configuration loaded from `slurm.toml`.
    python_cmd:
        Command used to invoke the runner (e.g. `"uv run"`).
    run_id:
        Run identifier used as the Slurm job name.

    Returns
    -------
    str
        Complete bash script text.
    """
    log_dir = run_dir / "main" / "logs"
    header = _render_slurm_header({
        "--job-name": run_id[:64],
        "--account": slurm.basic.account,
        "--partition": slurm.main.partition,
        "--constraint": slurm.main.constraint,
        "--gres": slurm.main.gres,
        "--time": slurm.main.time,
        "--mem": str(slurm.main.mem),
        "--ntasks": str(slurm.main.ntasks),
        "--nodes": str(slurm.main.nodes),
        "--cpus-per-task": str(slurm.main.cpus_per_task),
        "--threads-per-core": (
            str(slurm.main.threads_per_core) if slurm.main.threads_per_core else ""
        ),
        "--output": f"{log_dir}/slurm-%j.out",
        "--error": f"{log_dir}/slurm-%j.err",
        "--mail-type": slurm.basic.mail_type,
        "--mail-user": slurm.basic.mail_user,
    })
    return (
        "#!/usr/bin/env bash\n"
        f"{header}\n"
        "\n"
        "set -euo pipefail\n"
        "\n"
        f'RUN_DIR="{run_dir}"\n'
        "\n"
        'echo "Starting DMRG run: $RUN_DIR"\n'
        'echo "SLURM_JOB_ID: $SLURM_JOB_ID"\n'
        'echo "SLURM_NODELIST: $SLURM_NODELIST"\n'
        "\n"
        'cd "$RUN_DIR"\n'
        f'{python_cmd} "$RUN_DIR/algorithm/run_dmrg.py" --run-dir "$RUN_DIR"\n'
    )


def _build_array_script(
    campaign_dir: Path,
    runs_root: Path,
    slurm: SlurmTomlConfig,
    python_cmd: str,
    campaign_id: str,
    array_range: str,
) -> str:
    """Render the Slurm array submit script for a campaign.

    Parameters
    ----------
    campaign_dir:
        Absolute path to the campaign directory.
    runs_root:
        Absolute path to the runs/ directory.
    slurm:
        Slurm configuration loaded from the campaign's `slurm.toml`.
    python_cmd:
        Command used to invoke the runner.
    campaign_id:
        Campaign identifier used as the Slurm job name.
    array_range:
        Slurm array range string, e.g. `"1-10"` or `"1,3,5"`.

    Returns
    -------
    str
        Complete bash script text.
    """
    log_dir = campaign_dir / "logs"
    header = _render_slurm_header({
        "--job-name": campaign_id[:64],
        "--account": slurm.basic.account,
        "--partition": slurm.main.partition,
        "--constraint": slurm.main.constraint,
        "--gres": slurm.main.gres,
        "--time": slurm.main.time,
        "--mem": str(slurm.main.mem),
        "--ntasks": str(slurm.main.ntasks),
        "--nodes": str(slurm.main.nodes),
        "--cpus-per-task": str(slurm.main.cpus_per_task),
        "--threads-per-core": (
            str(slurm.main.threads_per_core) if slurm.main.threads_per_core else ""
        ),
        "--output": f"{log_dir}/slurm-%A_%a.out",
        "--error": f"{log_dir}/slurm-%A_%a.err",
        "--array": array_range,
        "--mail-type": slurm.basic.mail_type,
        "--mail-user": slurm.basic.mail_user,
    })
    return (
        "#!/usr/bin/env bash\n"
        f"{header}\n"
        "\n"
        "set -euo pipefail\n"
        "\n"
        f'CAMPAIGN_DIR="{campaign_dir}"\n'
        f'RUNS_ROOT="{runs_root}"\n'
        "\n"
        "# Resolve run_id from runs.csv using SLURM_ARRAY_TASK_ID.\n"
        "# Row number minus the header row equals the 1-based array task index.\n"
        "RUN_ID=$(awk -F',' -v id=\"$SLURM_ARRAY_TASK_ID\" "
        "'NR>1 && NR-1==id {print $1}' \\\n"
        '    "$CAMPAIGN_DIR/runs.csv")\n'
        "\n"
        'if [[ -z "$RUN_ID" ]]; then\n'
        '    echo "ERROR: no run_id found for array_id=$SLURM_ARRAY_TASK_ID" >&2\n'
        "    exit 1\n"
        "fi\n"
        "\n"
        'RUN_DIR="$RUNS_ROOT/$RUN_ID"\n'
        "\n"
        'echo "Starting DMRG run: $RUN_DIR"\n'
        'echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID  RUN_ID: $RUN_ID"\n'
        'echo "SLURM_NODELIST: $SLURM_NODELIST"\n'
        "\n"
        'cd "$RUN_DIR"\n'
        f'{python_cmd} "$RUN_DIR/algorithm/run_dmrg.py" --run-dir "$RUN_DIR"\n'
    )


def _build_exec_script(
    run_dir: Path,
    script_name: str,
    slurm: SlurmTomlConfig,
    python_cmd: str,
    run_id: str,
) -> str:
    """Render the Slurm submit script for an exec (follow-up) job.

    Parameters
    ----------
    run_dir:
        Absolute path to the run directory.
    script_name:
        Script identifier (filename without `.py` extension).
    slurm:
        Slurm configuration loaded from `slurm.toml`.
    python_cmd:
        Command used to invoke the runner.
    run_id:
        Run identifier used as the Slurm job name prefix.

    Returns
    -------
    str
        Complete bash script text.
    """
    log_dir = run_dir / "exec" / script_name / "logs"
    job_name = f"{run_id[:48]}_{script_name}"[:64]
    header = _render_slurm_header({
        "--job-name": job_name,
        "--account": slurm.basic.account,
        "--partition": slurm.exec_.partition,
        "--constraint": slurm.exec_.constraint,
        "--gres": slurm.exec_.gres,
        "--time": slurm.exec_.time,
        "--mem": str(slurm.exec_.mem),
        "--ntasks": str(slurm.exec_.ntasks),
        "--nodes": str(slurm.exec_.nodes),
        "--cpus-per-task": str(slurm.exec_.cpus_per_task),
        "--threads-per-core": (
            str(slurm.exec_.threads_per_core) if slurm.exec_.threads_per_core else ""
        ),
        "--output": f"{log_dir}/slurm-%j.out",
        "--error": f"{log_dir}/slurm-%j.err",
        "--mail-type": slurm.basic.mail_type,
        "--mail-user": slurm.basic.mail_user,
    })
    return (
        "#!/usr/bin/env bash\n"
        f"{header}\n"
        "\n"
        "set -euo pipefail\n"
        "\n"
        f'RUN_DIR="{run_dir}"\n'
        "\n"
        f'echo "Starting exec job: {script_name}"\n'
        'echo "Run dir: $RUN_DIR"\n'
        'echo "SLURM_JOB_ID: $SLURM_JOB_ID"\n'
        'echo "SLURM_NODELIST: $SLURM_NODELIST"\n'
        "\n"
        'cd "$RUN_DIR"\n'
        f'{python_cmd} "$RUN_DIR/algorithm/{script_name}.py" --run-dir "$RUN_DIR"\n'
    )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _algorithm_source_path(algorithm: str) -> Path:
    """Return the path of the bundled runner script for `algorithm`.

    Uses `importlib.resources` to locate the script inside the installed
    `intraknot.algorithm` subpackage.

    Parameters
    ----------
    algorithm:
        Algorithm name, e.g. `"dmrg"`.

    Returns
    -------
    Path
        Absolute path to `run_<algorithm>.py` inside the package.

    Raises
    ------
    FileNotFoundError
        If no runner script for `algorithm` exists.
    """
    # Fast path: real file on disk (editable installs and extracted wheels).
    direct = Path(__file__).parent / "algorithm" / f"run_{algorithm}.py"
    if direct.exists():
        return direct

    # Fallback: package may be inside a zip. Read bytes through
    # importlib.resources and materialise them at the expected path so the
    # returned path is always persistent (as_file temp paths are deleted when
    # the context manager exits, which is before callers can use them).
    try:
        data = (
            importlib.resources.files("intraknot.algorithm") / f"run_{algorithm}.py"
        ).read_bytes()
    except (FileNotFoundError, TypeError, AttributeError):
        raise FileNotFoundError(
            f"No runner script found for algorithm {algorithm!r}. "
            f"Expected: {direct}"
        )
    direct.parent.mkdir(parents=True, exist_ok=True)
    direct.write_bytes(data)
    return direct


def _copy_algorithm(src: Path, dest_dir: Path) -> None:
    """Copy a runner script into `dest_dir/algorithm/`.

    Parameters
    ----------
    src:
        Source runner script path.
    dest_dir:
        Destination parent (campaign or run directory).
    """
    alg_dir = dest_dir / "algorithm"
    alg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, alg_dir / src.name)


def _find_exec_script(
    script_name: str,
    run_dir: Path,
    campaign_dir: Path,
) -> Path:
    """Locate an exec script by searching the run, campaign, then package.

    Search order:

    1. `<run_dir>/algorithm/<script_name>.py` — already promoted to the run.
    2. `<campaign_dir>/algorithm/<script_name>.py` — campaign-level custom
       script.
    3. `intraknot.algorithm/<script_name>.py` — bundled package script.

    Parameters
    ----------
    script_name:
        Script filename without the `.py` extension.
    run_dir:
        Root of the run directory.
    campaign_dir:
        Root of the owning campaign directory.

    Returns
    -------
    Path
        Absolute path to the located script.

    Raises
    ------
    FileNotFoundError
        If the script cannot be found in any of the three locations.
    """
    candidates = [
        run_dir / "algorithm" / f"{script_name}.py",
        campaign_dir / "algorithm" / f"{script_name}.py",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Fall back to the bundled package (editable and extracted-wheel installs).
    here = Path(__file__).parent
    bundled = here / "algorithm" / f"{script_name}.py"
    if bundled.exists():
        return bundled

    # Zip-based distribution: materialise the resource at the expected path.
    try:
        data = (
            importlib.resources.files("intraknot.algorithm") / f"{script_name}.py"
        ).read_bytes()
    except (FileNotFoundError, TypeError, AttributeError):
        pass
    else:
        bundled.parent.mkdir(parents=True, exist_ok=True)
        bundled.write_bytes(data)
        return bundled

    raise FileNotFoundError(
        f"Exec script {script_name!r} not found. "
        f"Searched: {candidates[0]}, {candidates[1]}, and the intraknot package."
    )


def update_current(run_dir: Path, attempt_name: str) -> None:
    """Update `main/current` to point to `attempt_name`.

    Tries a symlink first; falls back to `current.txt` on filesystems that do
    not support symlinks.

    Parameters
    ----------
    run_dir:
        Root of the run directory.
    attempt_name:
        Bare attempt name, e.g. `"attempt_01"`.
    """
    current = run_dir / "main" / "current"
    target = Path("attempts") / attempt_name
    try:
        if current.is_symlink() or current.exists():
            current.unlink()
        current.symlink_to(target)
    except (OSError, NotImplementedError):
        (run_dir / "main" / "current.txt").write_text(attempt_name + "\n")


# ---------------------------------------------------------------------------
# Campaign creation
# ---------------------------------------------------------------------------

def create_campaign(
    campaign_id: str,
    description: str,
    algorithm: str,
    campaigns_root: Path,
    configs_dir: Optional[Path] = None,
) -> Path:
    """Create a new campaign directory with all standard files.

    Writes `campaign.yaml`, a template `defaults.toml`, an empty `runs.csv`
    (header only), a blank `notes.md`, and copies the algorithm runner to
    `<campaign_id>/algorithm/`. A `slurm.toml` is copied from `configs_dir`
    if provided and the file exists there, otherwise the default template is
    written directly.

    Parameters
    ----------
    campaign_id:
        Unique campaign identifier (used as directory name).
    description:
        Human-readable description of the campaign's scientific purpose.
    algorithm:
        Algorithm name whose runner script will be copied (e.g. `"dmrg"`).
    campaigns_root:
        Parent directory where the campaign subdirectory is created.
    configs_dir:
        Path to the project `configs/` directory. When given and
        `configs_dir/slurm.toml` exists, that file is copied into the
        campaign. Otherwise the built-in template is written.

    Returns
    -------
    Path
        Absolute path to the newly created campaign directory.

    Raises
    ------
    FileExistsError
        If a campaign with `campaign_id` already exists.
    """
    campaign_dir = campaigns_root / campaign_id
    if campaign_dir.exists():
        raise FileExistsError(f"Campaign already exists: {campaign_dir}")
    campaign_dir.mkdir(parents=True)

    # campaign.yaml — identity record (YAML).
    campaign_meta = {
        "campaign_id": campaign_id,
        "description": description,
        "algorithm": algorithm,
        "created_at": datetime.date.today().isoformat(),
    }
    (campaign_dir / "campaign.yaml").write_text(
        yaml.dump(campaign_meta, default_flow_style=False, sort_keys=False)
    )

    # defaults.toml — template for geometry, model, algorithm, and output defaults.
    (campaign_dir / "defaults.toml").write_text(
        "# Campaign-level default settings.\n"
        "# Values here are inherited by all runs and can be overridden per-run.\n\n"
        "[geometry]\n"
        "lattice = \"_init_\"\n"
        "lx      = 0\n"
        "bcx     = \"OBC\"\n"
        "n2x     = true\n\n"
        "[model]\n"
        "category = \"_init_\"  # 'bosonic', 'fermionic', or 'conductor'\n"
        "label    = \"_init_\"\n"
        "symmetry = \"_init_\"\n\n"
        "[algorithm]\n"
        "engine       = \"dmrg\"\n"
        "scheme       = \"1sp\"   # '2s' or '1sp'\n"
        "max_bond     = 64\n"
        "n_sweeps     = 20\n"
        "e_tol        = 1.0e-8\n"
        "trunc_thresh = 1.0e-15\n"
        "init         = \"random\"   # 'product', 'random', or 'resume'\n"
        "seed         = 42\n"
        "expand_k     = 8\n"
        "expand_alpha = 16\n\n"
        "[output]\n"
        "save_state      = true\n"
        "observables     = [\"energy\", \"entropy\"]\n"
    )

    # slurm.toml — copy from configs/ or write built-in template.
    slurm_dest = campaign_dir / "slurm.toml"
    global_slurm = configs_dir / "slurm.toml" if configs_dir else None
    if global_slurm and global_slurm.exists():
        shutil.copy2(global_slurm, slurm_dest)
    else:
        write_slurm_toml(slurm_dest)

    # runs.csv — three-column parameter table.
    with open(campaign_dir / "runs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "scan_id", "status"])

    # notes.md — blank.
    (campaign_dir / "notes.md").write_text(f"# {campaign_id}\n\n")

    # Logs directory for array job output.
    (campaign_dir / "logs").mkdir()

    # Copy algorithm runner.
    src = _algorithm_source_path(algorithm)
    _copy_algorithm(src, campaign_dir)

    return campaign_dir


# ---------------------------------------------------------------------------
# Run creation
# ---------------------------------------------------------------------------

def create_run(
    run_id: str,
    campaign_id: str,
    config_src: Optional[Path],
    runs_root: Path,
    campaigns_root: Path,
    scan_id: str = "",
    overrides: Optional[Dict[str, Any]] = None,
    run_uuid: Optional[_uuid_lib.UUID] = None,
) -> Path:
    """Create a new run directory with all standard files and subdirectories.

    Merges `config_src` (if given) with the campaign's `defaults.toml` to
    produce the run's `config.toml`. The `slurm.toml` is copied verbatim from
    the campaign directory; edit it before submitting to override Slurm
    resource settings for this specific run.

    Exactly one of `config_src` and `overrides` should be provided. When both
    are `None` the run is built from campaign defaults alone. When both are
    non-`None` a `ValueError` is raised.

    Parameters
    ----------
    run_id:
        Unique run identifier (used as directory name).
    campaign_id:
        Associated campaign. The campaign must already exist under
        `campaigns_root`.
    config_src:
        Path to a TOML file containing per-run overrides (typically at least
        `[model]`). Pass `None` to rely entirely on the campaign's
        `defaults.toml` or `overrides`.
    runs_root:
        Parent directory where the run subdirectory is created.
    campaigns_root:
        Parent directory containing campaign subdirectories.
    scan_id:
        Optional scan identifier. When non-empty, recorded in the campaign's
        `runs.csv` so that runs belonging to the same scan can be selected
        together with `--scan`.
    overrides:
        In-memory override dict in the same nested structure as a parsed TOML
        config (`{section: {key: value}}`). Takes precedence over `config_src`
        when provided. Mutually exclusive with `config_src`.
    run_uuid:
        UUID for this run. When `None`, a new UUID4 is generated. The full
        UUID is stored in `manifest.yaml`; the auto-name caller is responsible
        for deriving `uuid8` from this value before calling `make_run_id`.

    Returns
    -------
    Path
        Absolute path to the newly created run directory.

    Raises
    ------
    FileExistsError
        If a run with `run_id` already exists.
    FileNotFoundError
        If `config_src` does not exist or the campaign directory is absent.
    ValueError
        If both `config_src` and `overrides` are provided.
    """
    if config_src is not None and overrides is not None:
        raise ValueError("Provide at most one of config_src and overrides.")

    run_dir = runs_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}")

    campaign_dir = campaigns_root / campaign_id
    if not campaign_dir.exists():
        raise FileNotFoundError(f"Campaign not found: {campaign_dir}")

    # Load and merge physics configs.
    if overrides is not None:
        user_cfg: Optional[Dict[str, Any]] = overrides
    else:
        user_cfg = load_config(config_src) if config_src is not None else None
    defaults = load_campaign_defaults(campaign_dir)
    merged_cfg = _merge_defaults(user_cfg, defaults)

    # Create directory skeleton.
    run_dir.mkdir(parents=True)
    (run_dir / "main" / "attempts").mkdir(parents=True)
    (run_dir / "main" / "logs").mkdir(parents=True)
    (run_dir / "summary").mkdir()

    # Write merged config.toml (source of truth for this run's science).
    (run_dir / "config.toml").write_text(_dump_toml(merged_cfg))

    # Copy slurm.toml from campaign (straight copy; user edits for per-run overrides).
    campaign_slurm = campaign_dir / "slurm.toml"
    if campaign_slurm.exists():
        shutil.copy2(campaign_slurm, run_dir / "slurm.toml")
    else:
        write_slurm_toml(run_dir / "slurm.toml")

    # manifest.yaml — identity record (YAML).
    campaign_yaml = load_config(campaign_dir / "campaign.yaml")
    algorithm = campaign_yaml.get("algorithm", "dmrg")
    if run_uuid is None:
        run_uuid = _uuid_lib.uuid4()
    manifest = {
        "run_id": run_id,
        "uuid": str(run_uuid),
        "algorithm": algorithm,
        "created_at": datetime.date.today().isoformat(),
    }
    (run_dir / "manifest.yaml").write_text(
        yaml.dump(manifest, default_flow_style=False, sort_keys=False)
    )

    # Initialise main/status.json as pending.
    write_status(
        run_dir / "main" / "status.json",
        MainStatus(state=RunState.PENDING),
    )

    # Copy algorithm runner from campaign.
    campaign_alg_dir = campaign_dir / "algorithm"
    runner = next(campaign_alg_dir.glob(f"run_{algorithm}.py"), None)
    if runner is None:
        # Fall back to bundled source.
        runner = _algorithm_source_path(algorithm)
    _copy_algorithm(runner, run_dir)

    # Register the run in the campaign's runs.csv.
    _register_run_in_campaign(campaign_dir, run_id, scan_id=scan_id)

    return run_dir


# ---------------------------------------------------------------------------
# Campaign run registry helpers
# ---------------------------------------------------------------------------

def _register_run_in_campaign(
    campaign_dir: Path,
    run_id: str,
    scan_id: str = "",
) -> None:
    """Append a row to the campaign's `runs.csv`.

    Creates the file with a three-column header if it does not yet exist.
    When an existing file has only the legacy two-column header (`run_id`,
    `status`), a two-column row is appended for backward compatibility.

    Parameters
    ----------
    campaign_dir:
        Campaign directory containing (or to receive) `runs.csv`.
    run_id:
        Run identifier to register.
    scan_id:
        Scan identifier to record in the `scan_id` column. Ignored when the
        file uses the legacy two-column format.
    """
    runs_csv = campaign_dir / "runs.csv"
    if not runs_csv.exists():
        with open(runs_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "scan_id", "status"])

    # Detect the existing header to preserve backward compatibility.
    with open(runs_csv, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            header = []

    has_scan_id_col = "scan_id" in header

    # Guard against duplicate entries: read existing run_ids and skip if
    # this run_id is already registered.
    with open(runs_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        existing_ids = {row[0] for row in reader if row}
    if run_id in existing_ids:
        return

    with open(runs_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if has_scan_id_col:
            writer.writerow([run_id, scan_id, "pending"])
        else:
            writer.writerow([run_id, "pending"])


def remove_run_from_campaign(campaign_dir: Path, run_id: str) -> bool:
    """Remove a run from the campaign's `runs.csv`.

    Rewrites `runs.csv` without the row for `run_id`. Does nothing if
    the file does not exist or `run_id` is not present.

    Parameters
    ----------
    campaign_dir:
        Campaign directory containing `runs.csv`.
    run_id:
        Run identifier to remove.

    Returns
    -------
    bool
        `True` if a row was removed, `False` if `run_id` was not found.
    """
    runs_csv = campaign_dir / "runs.csv"
    if not runs_csv.exists():
        return False

    with open(runs_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or ["run_id", "status"])
        rows = list(reader)

    kept = [r for r in rows if r.get("run_id", "").strip() != run_id]
    if len(kept) == len(rows):
        return False

    with open(runs_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    return True


def remove_run_from_all_campaigns(campaigns_root: Path, run_id: str) -> List[str]:
    """Remove a run from every campaign's `runs.csv` under `campaigns_root`.

    Scans all subdirectories of `campaigns_root` and calls
    `remove_run_from_campaign` on each one that is a directory.  Subdirectories
    that do not contain a `runs.csv`, or whose CSV does not list `run_id`, are
    silently skipped.

    Parameters
    ----------
    campaigns_root:
        Parent directory whose subdirectories are the campaign directories.
    run_id:
        Run identifier to remove from every registry found.

    Returns
    -------
    list[str]
        Names of campaign directories from which the run was removed, in
        sorted order.
    """
    removed_from: List[str] = []
    if not campaigns_root.exists():
        return removed_from
    for d in sorted(campaigns_root.iterdir()):
        if d.is_dir():
            if remove_run_from_campaign(d, run_id):
                removed_from.append(d.name)
    return removed_from


def read_runs_by_filter(
    campaign_dir: Path,
    scan_id: Optional[str] = None,
    status: Optional[str] = None,
) -> List[str]:
    """Return run IDs from `runs.csv` matching optional scan and status filters.

    Parameters
    ----------
    campaign_dir:
        Campaign directory containing `runs.csv`.
    scan_id:
        When given, only rows whose `scan_id` column equals this value are
        returned. Ignored when the file uses the legacy two-column format.
    status:
        When given, only rows whose `status` column equals this value are
        returned.

    Returns
    -------
    list[str]
        Ordered list of matching `run_id` values.
    """
    runs_csv = campaign_dir / "runs.csv"
    if not runs_csv.exists():
        return []

    with open(runs_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    matches: List[str] = []
    for row in rows:
        run_id = row.get("run_id", "").strip()
        if not run_id:
            continue
        if scan_id is not None and row.get("scan_id", "").strip() != scan_id:
            continue
        if status is not None and row.get("status", "").strip() != status:
            continue
        matches.append(run_id)
    return matches


def delete_run(
    run_dir: Path,
    campaign_dir: Optional[Path] = None,
    *,
    delete_dir: bool = False,
    campaigns_root: Optional[Path] = None,
) -> bool:
    """Deregister a run from its campaign(s) and optionally delete the directory.

    When `delete_dir` is `True` and `campaigns_root` is given, the run is
    removed from every campaign's `runs.csv` before the directory is deleted.
    When `delete_dir` is `False` (or `campaigns_root` is not given), only the
    single `campaign_dir` registry is updated.

    Parameters
    ----------
    run_dir:
        Root of the run directory.
    campaign_dir:
        Campaign directory whose `runs.csv` to update. Used only when
        `campaigns_root` is not given. Pass `None` to skip the CSV update.
    delete_dir:
        If `True`, delete `run_dir` from disk after deregistering.
    campaigns_root:
        When given together with `delete_dir=True`, all campaigns under this
        directory are scanned and the run is removed from each one. Takes
        precedence over `campaign_dir` in that case.

    Returns
    -------
    bool
        `True` if the run was found and removed from at least one `runs.csv`,
        or if no campaign arguments were supplied. `False` when a
        `campaign_dir` was supplied but `run_id` was absent from its CSV.

    Raises
    ------
    FileNotFoundError
        If `run_dir` does not exist.
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_id = run_dir.name
    if delete_dir and campaigns_root is not None:
        # N:M case: sweep all campaigns before removing the directory.
        cleaned = remove_run_from_all_campaigns(campaigns_root, run_id)
        removed = bool(cleaned)
    elif campaign_dir is not None and campaign_dir.exists():
        removed = remove_run_from_campaign(campaign_dir, run_id)
    else:
        removed = True

    if delete_dir:
        shutil.rmtree(run_dir)

    return removed


# ---------------------------------------------------------------------------
# Attempt creation
# ---------------------------------------------------------------------------

def create_attempt(run_dir: Path) -> Path:
    """Create the next `attempt_NN` directory under `main/attempts/`.

    The attempt index is one more than the highest existing index. Creates
    the directory and updates `main/current`.

    Parameters
    ----------
    run_dir:
        Root of the run directory.

    Returns
    -------
    Path
        Absolute path to the newly created attempt directory.
    """
    attempts_root = run_dir / "main" / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        d.name for d in attempts_root.iterdir()
        if d.is_dir() and d.name.startswith("attempt_")
    )
    next_idx = int(existing[-1].split("_")[1]) + 1 if existing else 1
    name = f"attempt_{next_idx:02d}"
    attempt_dir = attempts_root / name
    attempt_dir.mkdir()
    update_current(run_dir, name)
    return attempt_dir


# ---------------------------------------------------------------------------
# Slurm script rendering and submission — primary job
# ---------------------------------------------------------------------------

def write_slurm_script(
    run_dir: Path,
    machine: MachineConfig,
    run_id: Optional[str] = None,
) -> Path:
    """Render and write `main/submit.slurm` for a single-run primary job.

    Reads Slurm settings from `run_dir/slurm.toml`. Creates `main/logs/` if
    it does not yet exist.

    Parameters
    ----------
    run_dir:
        Root of the run directory.
    machine:
        Machine configuration supplying the Python command.
    run_id:
        Run identifier used as the Slurm job name. Defaults to the directory
        name of `run_dir`.

    Returns
    -------
    Path
        Path to the written `submit.slurm` file.
    """
    if run_id is None:
        run_id = run_dir.name

    run_dir = run_dir.resolve()
    log_dir = run_dir / "main" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    slurm = load_slurm_toml(run_dir / "slurm.toml")
    script = _build_single_script(run_dir, slurm, machine.paths.command, run_id)

    out = run_dir / "main" / "submit.slurm"
    out.write_text(script)
    return out


def write_array_slurm_script(
    campaign_dir: Path,
    runs_root: Path,
    machine: MachineConfig,
    array_range: str,
    campaign_id: Optional[str] = None,
) -> Path:
    """Render and write `submit_array.slurm` for a campaign-level array job.

    Reads Slurm settings from `campaign_dir/slurm.toml`.

    Parameters
    ----------
    campaign_dir:
        Campaign directory.
    runs_root:
        Root directory containing run subdirectories.
    machine:
        Machine configuration supplying the Python command.
    array_range:
        Slurm array range string, e.g. `"1-10"` or `"1,3,5"`.
    campaign_id:
        Campaign identifier for the job name. Defaults to the directory name.

    Returns
    -------
    Path
        Path to the written `submit_array.slurm` file.
    """
    if campaign_id is None:
        campaign_id = campaign_dir.name

    campaign_dir = campaign_dir.resolve()
    runs_root = runs_root.resolve()

    slurm = load_slurm_toml(campaign_dir / "slurm.toml")
    script = _build_array_script(
        campaign_dir, runs_root, slurm, machine.paths.command, campaign_id, array_range
    )

    out = campaign_dir / "submit_array.slurm"
    out.write_text(script)
    return out


def submit_job(run_dir: Path) -> str:
    """Submit the primary Slurm job for a run and record the job ID.

    Calls `sbatch main/submit.slurm` from within `run_dir`. The assigned
    job ID is written to `main/job_id.txt`.

    Parameters
    ----------
    run_dir:
        Root of the run directory. Must contain `main/submit.slurm`.

    Returns
    -------
    str
        Slurm job ID string (e.g. `"12345678"`).

    Raises
    ------
    FileNotFoundError
        If `main/submit.slurm` does not exist.
    subprocess.CalledProcessError
        If `sbatch` exits with a non-zero status.
    """
    script = run_dir / "main" / "submit.slurm"
    if not script.exists():
        raise FileNotFoundError(f"Slurm script not found: {script}")

    proc = subprocess.run(
        ["sbatch", str(script)],
        capture_output=True,
        text=True,
        check=True,
    )
    # sbatch output: "Submitted batch job 12345678"
    parts = proc.stdout.strip().split()
    if not parts or parts[0:3] != ["Submitted", "batch", "job"]:
        raise RuntimeError(
            f"Unexpected sbatch output: {proc.stdout.strip()!r}"
        )
    job_id = parts[-1]
    (run_dir / "main" / "job_id.txt").write_text(job_id + "\n")
    return job_id


# ---------------------------------------------------------------------------
# Exec job infrastructure
# ---------------------------------------------------------------------------

def prepare_exec(
    run_dir: Path,
    script_name: str,
    campaign_dir: Path,
) -> Path:
    """Locate an exec script, promote it to the run's `algorithm/` directory,
    and create the `exec/<script_name>/` slot with its `logs/` subdirectory.

    Parameters
    ----------
    run_dir:
        Root of the run directory.
    script_name:
        Script filename without the `.py` extension.
    campaign_dir:
        Root of the owning campaign directory (used for script search).

    Returns
    -------
    Path
        Absolute path to the `exec/<script_name>/` slot directory.
    """
    # Locate and promote the script.
    src = _find_exec_script(script_name, run_dir, campaign_dir)
    dest = run_dir / "algorithm" / f"{script_name}.py"
    if not dest.exists():
        shutil.copy2(src, dest)

    # Create the exec slot.
    slot = run_dir / "exec" / script_name
    (slot / "logs").mkdir(parents=True, exist_ok=True)
    return slot


def write_exec_slurm_script(
    run_dir: Path,
    script_name: str,
    machine: MachineConfig,
) -> Path:
    """Render and write `exec/<script_name>/submit.slurm`.

    Uses the `[exec]` section of `run_dir/slurm.toml` for resource settings.

    Parameters
    ----------
    run_dir:
        Root of the run directory.
    script_name:
        Script filename without the `.py` extension.
    machine:
        Machine configuration supplying the Python command.

    Returns
    -------
    Path
        Path to the written `submit.slurm` file.
    """
    run_dir = run_dir.resolve()
    run_id = run_dir.name

    slurm = load_slurm_toml(run_dir / "slurm.toml")
    script = _build_exec_script(run_dir, script_name, slurm, machine.paths.command, run_id)

    out = run_dir / "exec" / script_name / "submit.slurm"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(script)
    return out


def submit_exec_job(run_dir: Path, script_name: str) -> str:
    """Submit an exec job to Slurm and record the job ID.

    Calls `sbatch exec/<script_name>/submit.slurm`. The job ID is written to
    `exec/<script_name>/job_id.txt`.

    Parameters
    ----------
    run_dir:
        Root of the run directory.
    script_name:
        Script filename without the `.py` extension.

    Returns
    -------
    str
        Slurm job ID string.

    Raises
    ------
    FileNotFoundError
        If the submit script does not exist (call `write_exec_slurm_script`
        first).
    subprocess.CalledProcessError
        If `sbatch` exits with a non-zero status.
    """
    script = run_dir / "exec" / script_name / "submit.slurm"
    if not script.exists():
        raise FileNotFoundError(f"Exec Slurm script not found: {script}")

    proc = subprocess.run(
        ["sbatch", str(script)],
        capture_output=True,
        text=True,
        check=True,
    )
    parts = proc.stdout.strip().split()
    if not parts or parts[0:3] != ["Submitted", "batch", "job"]:
        raise RuntimeError(
            f"Unexpected sbatch output: {proc.stdout.strip()!r}"
        )
    job_id = parts[-1]
    (run_dir / "exec" / script_name / "job_id.txt").write_text(job_id + "\n")
    return job_id
