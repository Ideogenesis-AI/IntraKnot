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


"""Shared test fixtures and helpers.

Public names imported by test modules:

- `machine()` — pytest fixture returning a `MachineConfig` with `uv run`.
- `make_run_dir()` — module-level helper for building a run directory that
  carries a status file, a stub algorithm script, and a minimal `slurm.toml`.
- `write_slurm_toml()` — write a minimal `slurm.toml` into an arbitrary dir.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intraknot.config import MachineConfig, PathsConfig
from intraknot.status import (
    FailureReason,
    MainStatus,
    RunState,
    write_status,
)


# ---------------------------------------------------------------------------
# Shared fixture: machine config
# ---------------------------------------------------------------------------

@pytest.fixture()
def machine() -> MachineConfig:
    """A `MachineConfig` with the `uv run` Python command."""
    return MachineConfig(paths=PathsConfig(command="uv run"))


# ---------------------------------------------------------------------------
# Module-level helpers (plain functions, importable from tests)
# ---------------------------------------------------------------------------

def make_machine(command: str = "uv run") -> MachineConfig:
    """Return a `MachineConfig` with `command` as the Python command."""
    return MachineConfig(paths=PathsConfig(command=command))


def write_slurm_toml(dest: Path, **overrides) -> None:
    """Write a minimal `slurm.toml` into `dest`.

    Parameters
    ----------
    dest:
        Directory to write `slurm.toml` into.
    **overrides:
        Field overrides applied on top of the defaults.
    """
    fields = dict(
        account="testproject",
        partition="cpu",
        constraint="",
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
    (dest / "slurm.toml").write_text(content)


def make_run_dir(
    base: Path,
    run_id: str,
    state: RunState = RunState.FAILED,
    restartable: bool = True,
    reason: FailureReason = FailureReason.TIMEOUT,
    energy: float = -1.23,
) -> Path:
    """Create a realistic run directory for testing.

    Creates the standard directory skeleton (main/attempts, main/logs,
    algorithm/), writes a `MainStatus` to `main/status.json`, writes a stub
    `info.json` with observables, and links `main/current` to `attempt_01`.
    Also writes a minimal `slurm.toml` so Slurm-script writers work.

    Parameters
    ----------
    base:
        Parent directory (typically `tmp_path` or `tmp_path / "runs"`).
    run_id:
        Run identifier used as the directory name.
    state:
        `RunState` for the `main/status.json` file.
    restartable:
        `restartable` flag in the status.
    reason:
        `FailureReason` for the status (ignored for non-failure states).
    energy:
        Energy value written to `info.json`.

    Returns
    -------
    Path
        Absolute path to the created run directory.
    """
    run_dir = base / run_id
    (run_dir / "main" / "attempts" / "attempt_01").mkdir(parents=True)
    (run_dir / "main" / "logs").mkdir(parents=True)
    (run_dir / "algorithm").mkdir()
    (run_dir / "algorithm" / "run_dmrg.py").write_text("# stub\n")

    write_slurm_toml(run_dir)

    status_reason = reason if state == RunState.FAILED else (
        FailureReason.CONVERGED if state == RunState.COMPLETED else None
    )
    write_status(
        run_dir / "main" / "status.json",
        MainStatus(
            state=state,
            current_attempt="attempt_01",
            reason=status_reason,
            restartable=restartable,
        ),
    )

    obs = {
        "energy": energy,
        "converged": state == RunState.COMPLETED,
        "n_sweeps": 5,
    }
    (run_dir / "main" / "attempts" / "attempt_01" / "info.json").write_text(
        json.dumps(obs)
    )

    current = run_dir / "main" / "current"
    try:
        current.symlink_to(Path("attempts") / "attempt_01")
    except OSError:
        (run_dir / "main" / "current.txt").write_text("attempt_01\n")

    return run_dir
