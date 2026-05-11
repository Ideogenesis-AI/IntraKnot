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
from pathlib import Path
from typing import Optional

import yaml

from .config import MachineConfig, _merge_defaults, load_campaign_defaults, load_config
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
    IntraKnot configs.  Inline tables and arrays-of-tables are not supported.

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
# Slurm script template (embedded; no separate templates/ directory)
# ---------------------------------------------------------------------------

# Variables filled by str.format_map:
#   python        — command to invoke the runner (e.g. "uv run")
#   run_dir       — absolute path to the run directory
#   job_name      — short job name (run_id)
#   account       — Slurm account
#   partition     — Slurm partition
#   time          — walltime string
#   mem           — memory string
#   cpus_per_task — number of CPUs
#   log_dir       — directory for Slurm log files
_SLURM_SINGLE_TEMPLATE = """\
#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --output={log_dir}/slurm-%j.out

set -euo pipefail

RUN_DIR="{run_dir}"

echo "Starting DMRG run: $RUN_DIR"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"

cd "$RUN_DIR"
{python} "$RUN_DIR/algorithm/run_dmrg.py" --run-dir "$RUN_DIR"
"""

# Array job template — each task maps SLURM_ARRAY_TASK_ID to a run directory
# via the campaign runs.csv file.
#   campaign_dir  — absolute path to the campaign directory
#   runs_root     — absolute path to the runs/ directory
#   python, account, partition, time, mem, cpus_per_task — same as above
_SLURM_ARRAY_TEMPLATE = """\
#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --output={campaign_dir}/logs/slurm-%A_%a.out
#SBATCH --array={array_range}

set -euo pipefail

CAMPAIGN_DIR="{campaign_dir}"
RUNS_ROOT="{runs_root}"

# Resolve run_id from runs.csv using SLURM_ARRAY_TASK_ID.
RUN_ID=$(awk -F',' -v id="$SLURM_ARRAY_TASK_ID" 'NR>1 && $1==id {{print $2}}' \\
    "$CAMPAIGN_DIR/runs.csv")

if [[ -z "$RUN_ID" ]]; then
    echo "ERROR: no run_id found for array_id=$SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

RUN_DIR="$RUNS_ROOT/$RUN_ID"

echo "Starting DMRG run: $RUN_DIR"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID  RUN_ID: $RUN_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"

cd "$RUN_DIR"
{python} "$RUN_DIR/algorithm/run_dmrg.py" --run-dir "$RUN_DIR"
"""


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
    try:
        ref = importlib.resources.files("intraknot.algorithm") / f"run_{algorithm}.py"
        with importlib.resources.as_file(ref) as p:
            src = Path(p)
        if not src.exists():
            raise FileNotFoundError(src)
        return src
    except (FileNotFoundError, TypeError):
        # Fall back to path relative to this file for editable installs.
        here = Path(__file__).parent
        src = here / "algorithm" / f"run_{algorithm}.py"
        if not src.exists():
            raise FileNotFoundError(
                f"No runner script found for algorithm {algorithm!r}. "
                f"Expected: {src}"
            )
        return src


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
) -> Path:
    """Create a new campaign directory with all standard files.

    Writes `campaign.yaml`, a template `defaults.toml`, an empty `runs.csv`
    (header only), a blank `notes.md`, and copies the algorithm runner to
    `<campaign_id>/algorithm/`.

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

    # defaults.toml — template for algorithm + output defaults.
    (campaign_dir / "defaults.toml").write_text(
        "# Campaign-level default settings.\n"
        "# Values here are inherited by all runs and can be overridden per-run.\n\n"
        "[algorithm]\n"
        "name         = \"dmrg\"\n"
        "scheme       = \"2s\"\n"
        "max_bond     = 64\n"
        "n_sweeps     = 20\n"
        "e_tol        = 1.0e-8\n"
        "trunc_thresh = 1.0e-15\n"
        "init         = \"random\"\n\n"
        "[output]\n"
        "save_state      = true\n"
        "save_checkpoint = true\n"
        "observables     = [\"energy\", \"entropy\"]\n"
    )

    # runs.csv — parameter table (array_id column is optional; omit for now).
    with open(campaign_dir / "runs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "status"])

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
    config_src: Path,
    runs_root: Path,
    campaigns_root: Path,
    machine: Optional[MachineConfig] = None,
) -> Path:
    """Create a new run directory with all standard files and subdirectories.

    Copies `config_src` into the run directory as `config.toml`, merging in
    the campaign's `defaults.toml` for any missing `[algorithm]` or `[output]`
    sections.  Also copies the campaign's algorithm runner script.

    Parameters
    ----------
    run_id:
        Unique run identifier (used as directory name).
    campaign_id:
        Associated campaign.  The campaign must already exist under
        `campaigns_root`.
    config_src:
        Path to a TOML file containing at least a `[model]` section.
    runs_root:
        Parent directory where the run subdirectory is created.
    campaigns_root:
        Parent directory containing campaign subdirectories.
    machine:
        Machine config; used to record the machine name in `manifest.yaml`.
        Pass `None` when no machine config is available.

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
    """
    run_dir = runs_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}")

    campaign_dir = campaigns_root / campaign_id
    if not campaign_dir.exists():
        raise FileNotFoundError(f"Campaign not found: {campaign_dir}")

    # Load and merge configs.
    user_cfg = load_config(config_src)
    defaults = load_campaign_defaults(campaign_dir)
    merged_cfg = _merge_defaults(user_cfg, defaults)

    # Create directory skeleton.
    run_dir.mkdir(parents=True)
    (run_dir / "submit").mkdir()
    (run_dir / "logs").mkdir()
    (run_dir / "main" / "attempts").mkdir(parents=True)
    (run_dir / "summary").mkdir()

    # Write merged config.toml (source of truth for this run's science).
    (run_dir / "config.toml").write_text(_dump_toml(merged_cfg))

    # manifest.yaml — identity record (YAML).
    campaign_yaml = load_config(campaign_dir / "campaign.yaml")
    algorithm = campaign_yaml.get("algorithm", "dmrg")
    manifest = {
        "run_id": run_id,
        "campaign": campaign_id,
        "algorithm": algorithm,
        "status": RunState.PENDING.value,
        "created_at": datetime.date.today().isoformat(),
        "machine": machine.slurm.account if machine else "",
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

    return run_dir


# ---------------------------------------------------------------------------
# Attempt creation
# ---------------------------------------------------------------------------

def create_attempt(run_dir: Path) -> Path:
    """Create the next `attempt_NN` directory under `main/attempts/`.

    The attempt index is one more than the highest existing index.  Creates
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
# Slurm script rendering and submission
# ---------------------------------------------------------------------------

def write_slurm_script(
    run_dir: Path,
    machine: MachineConfig,
    run_id: Optional[str] = None,
) -> Path:
    """Render and write `submit/submit.slurm` for a single-run job.

    Parameters
    ----------
    run_dir:
        Root of the run directory.
    machine:
        Machine configuration supplying Slurm and path settings.
    run_id:
        Run identifier used as the Slurm job name.  Defaults to the directory
        name of `run_dir`.

    Returns
    -------
    Path
        Path to the written `submit.slurm` file.
    """
    if run_id is None:
        run_id = run_dir.name

    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    script = _SLURM_SINGLE_TEMPLATE.format_map({
        "job_name": run_id[:64],  # Slurm truncates names >64 chars.
        "account": machine.slurm.account,
        "partition": machine.slurm.partition,
        "time": machine.slurm.default_time,
        "mem": machine.slurm.default_mem,
        "cpus_per_task": machine.slurm.default_cpus_per_task,
        "log_dir": log_dir,
        "run_dir": run_dir,
        "python": machine.paths.python,
    })

    out = run_dir / "submit" / "submit.slurm"
    out.parent.mkdir(parents=True, exist_ok=True)
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

    Parameters
    ----------
    campaign_dir:
        Campaign directory.
    runs_root:
        Root directory containing run subdirectories.
    machine:
        Machine configuration.
    array_range:
        Slurm array range string, e.g. `"1-10"` or `"1,3,5"`.
    campaign_id:
        Campaign identifier for the job name.  Defaults to the directory name.

    Returns
    -------
    Path
        Path to the written `submit_array.slurm` file.
    """
    if campaign_id is None:
        campaign_id = campaign_dir.name

    script = _SLURM_ARRAY_TEMPLATE.format_map({
        "job_name": campaign_id[:64],
        "account": machine.slurm.account,
        "partition": machine.slurm.partition,
        "time": machine.slurm.default_time,
        "mem": machine.slurm.default_mem,
        "cpus_per_task": machine.slurm.default_cpus_per_task,
        "campaign_dir": campaign_dir,
        "runs_root": runs_root,
        "python": machine.paths.python,
        "array_range": array_range,
    })

    out = campaign_dir / "submit_array.slurm"
    out.write_text(script)
    return out


def submit_job(run_dir: Path) -> str:
    """Submit the Slurm job for a run and record the job ID.

    Calls `sbatch submit/submit.slurm` from within `run_dir`.  The assigned
    job ID is written to `submit/job_id.txt`.

    Parameters
    ----------
    run_dir:
        Root of the run directory.  Must contain `submit/submit.slurm`.

    Returns
    -------
    str
        Slurm job ID string (e.g. `"12345678"`).

    Raises
    ------
    FileNotFoundError
        If `submit/submit.slurm` does not exist.
    subprocess.CalledProcessError
        If `sbatch` exits with a non-zero status.
    """
    script = run_dir / "submit" / "submit.slurm"
    if not script.exists():
        raise FileNotFoundError(f"Slurm script not found: {script}")

    result = subprocess.run(
        ["sbatch", str(script)],
        capture_output=True,
        text=True,
        check=True,
    )
    # sbatch output: "Submitted batch job 12345678"
    job_id = result.stdout.strip().split()[-1]
    (run_dir / "submit" / "job_id.txt").write_text(job_id + "\n")
    return job_id
