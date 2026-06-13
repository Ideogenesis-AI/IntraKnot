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


"""New-attempt creation for failed or interrupted runs (break-point resumption)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from .config import MachineConfig
from .launch import submit_job, write_slurm_script
from .status import RETRYABLE_STATES, MainStatus, read_status


def _next_attempt_path(run_dir: Path) -> Path:
    """Return the path of the next attempt directory without creating it.

    Computes the next attempt index so callers can display or log the
    expected path before the job actually runs.

    Parameters
    ----------
    run_dir:
        Root of the run directory.

    Returns
    -------
    Path
        Predicted absolute path, e.g. ``.../main/attempts/attempt_02``.
    """
    attempts_root = run_dir / "main" / "attempts"
    existing = (
        sorted(
            d.name for d in attempts_root.iterdir()
            if d.is_dir() and d.name.startswith("attempt_")
        )
        if attempts_root.exists()
        else []
    )
    next_idx = int(existing[-1].split("_")[1]) + 1 if existing else 1
    return attempts_root / f"attempt_{next_idx:02d}"


def is_resumable(run_dir: Path) -> bool:
    """Return `True` if a new attempt may be created for this run.

    Reads `main/status.json` and checks that:

    - `state` is in `RETRYABLE_STATES` (currently only `failed`)
    - `restartable` is `True`

    Parameters
    ----------
    run_dir:
        Root of the run directory.

    Returns
    -------
    bool
        `True` when resumption is safe.
    """
    status_path = run_dir / "main" / "status.json"
    if not status_path.exists():
        return False
    try:
        status = read_status(status_path)
    except (KeyError, ValueError):
        return False

    if not isinstance(status, MainStatus):
        return False

    return status.state in RETRYABLE_STATES and status.restartable


def resume_run(
    run_dir: Path,
    machine: MachineConfig,
    submit: bool = True,
) -> Path:
    """Create a new attempt for a failed run and optionally submit it.

    Parameters
    ----------
    run_dir:
        Root of the run directory.
    machine:
        Machine configuration used to render the Slurm script.
    submit:
        If `True` (default), call `sbatch` after writing the script.

    Returns
    -------
    Path
        Predicted path to the attempt directory that the runner will create
        (e.g. ``.../main/attempts/attempt_02``). The directory does not exist
        yet at return time; it is created by the runner when the job starts.

    Raises
    ------
    ValueError
        If the run is not resumable according to `is_resumable`.
    """
    if not is_resumable(run_dir):
        raise ValueError(
            f"Run {run_dir.name!r} is not resumable. "
            "Check main/status.json: state must be 'failed' and restartable must be true."
        )

    # Predict the next attempt path for display purposes only.
    # Do NOT pre-create the directory here: the runner (run_dmrg.py) is
    # responsible for creating the attempt directory when the job starts.
    # Pre-creating it causes the runner to skip it and create one extra empty
    # attempt directory.
    next_attempt = _next_attempt_path(run_dir)
    write_slurm_script(run_dir, machine)
    if submit:
        submit_job(run_dir)
    return next_attempt


def resume_campaign(
    campaign_dir: Path,
    runs_root: Path,
    machine: MachineConfig,
    submit: bool = True,
) -> List[Path]:
    """Resume all resumable runs listed in a campaign's `runs.csv`.

    Parameters
    ----------
    campaign_dir:
        Campaign directory containing `runs.csv`.
    runs_root:
        Root directory containing run subdirectories.
    machine:
        Machine configuration.
    submit:
        If `True` (default), submit each new attempt to Slurm.

    Returns
    -------
    list[Path]
        Paths to newly created attempt directories (one per resumed run).
    """
    runs_csv = campaign_dir / "runs.csv"
    if not runs_csv.exists():
        return []

    with open(runs_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    resumed: List[Path] = []
    for row in rows:
        run_id = row.get("run_id", "").strip()
        if not run_id:
            continue
        run_dir = runs_root / run_id
        if not run_dir.exists():
            continue
        if is_resumable(run_dir):
            attempt = resume_run(run_dir, machine, submit=submit)
            resumed.append(attempt)

    return resumed


def find_resumable_runs(
    campaign_dir: Path,
    runs_root: Path,
) -> List[str]:
    """Return the list of run IDs in a campaign that are currently resumable.

    Does not modify any files.

    Parameters
    ----------
    campaign_dir:
        Campaign directory containing `runs.csv`.
    runs_root:
        Root directory containing run subdirectories.

    Returns
    -------
    list[str]
        Run IDs for which `is_resumable` returns `True`.
    """
    runs_csv = campaign_dir / "runs.csv"
    if not runs_csv.exists():
        return []

    with open(runs_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    return [
        row["run_id"]
        for row in rows
        if row.get("run_id", "").strip()
        and (runs_root / row["run_id"]).exists()
        and is_resumable(runs_root / row["run_id"])
    ]
