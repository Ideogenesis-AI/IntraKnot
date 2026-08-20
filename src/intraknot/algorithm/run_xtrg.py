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


"""IntraKnot-aware XTRG runner.

This script is a standalone entry point executed by Slurm from within a run
directory. It reads `config.toml`, builds the Hamiltonian via Alice, runs
XTRG, and writes its outputs to `main/` (checkpoints and archived artifacts)
and to a fresh attempt directory (logs and per-attempt status).

Usage
-----
    uv run run_xtrg.py --run-dir /path/to/runs/my_run

The script resolves the next attempt automatically by inspecting
`main/attempts/` and incrementing the highest existing index. Alice owns
XTRG checkpoint and artifact persistence inside `main/`. IntraKnot supplies
lifecycle bookkeeping and writes small, analysis-friendly JSON and CSV
projections of Alice's thermodynamic history.

XTRG cooling and resumption
---------------------------
An XTRG schedule is a fixed number of doubling steps: Alice writes
`thermal.ckpt` after every completed cooling step and `xtrg.ckpt` for the
most recent density-matrix snapshot, then removes `xtrg.ckpt` on successful
completion (`run()` always completes its full step schedule on any clean
return, so there is no "budget exhausted, not finished" case the way there
is for DMRG's sweep count — `xtrg.ckpt` survives only if a prior attempt
crashed or was killed mid-step). `checkpoint_dir` and `artifacts_dir` are
both set to `main/` (Alice >= 0.2.5 decouples the two, but there is no
reason here to keep them apart), shared across every attempt of this run,
so every checkpoint is already exactly where the next attempt looks — no
forwarding between attempt directories is needed.

Alice >= 0.2.6 accepts a starting state at any step covered by
`thermal.ckpt`, which makes each archived `artifacts/step_XX.ckpt` a
genuine restart point. This runner therefore picks its starting state from
the first of these that applies:

1. `main/artifacts/step_XX.ckpt` for an explicit
   `[algorithm] resume_from_step`, which re-cools a segment a previous
   attempt already covered (e.g. at a larger `max_bond`). It outranks
   `xtrg.ckpt`, since overriding the automatic choice is the point.
2. `main/xtrg.ckpt`, left behind by an attempt that crashed or was killed
   mid-step.
3. The highest archived step at or below `n_steps`, which continues a run
   that already finished a shorter schedule — raise `n_steps` in
   `config.toml` and submit again, and the cooling picks up where the
   earlier run stopped instead of restarting.
4. Nothing, in which case ρ(τ₀) is built fresh via a Taylor expansion
   (`thermal_mpo`).

Two properties of `n_steps` and τ₀ matter when continuing a run. `n_steps`
counts cooling steps from τ₀, so it is the absolute step index to stop at,
not a number of steps to add; and τ₀ anchors the whole β grid, so it must
match the value the recorded history was built with. Both are checked
against `main/thermal.ckpt` before any expensive work, and a violation ends
the attempt as `invalid` with reason `checkpoint_incompatible`.

Continuing from a step below the end of the recorded history is allowed —
that is what re-cooling a segment means — but Alice then truncates the
later `thermal.ckpt` entries and overwrites the `step_XX.ckpt` archives
past that step. Before handing over, this runner copies `thermal.ckpt` to
`main/thermal_old.ckpt`, and removes that copy only once the replacement
history has been written and validated, so an interrupted continuation
never leaves the run without a readable series.

Outputs
-------
Written to `main/` (shared across every attempt of this run):

thermal.ckpt
    Native thermodynamic `alice.algorithm.xtrg.Summary`, written atomically
    by Alice after every completed cooling step.
thermal_old.ckpt
    Copy of the `thermal.ckpt` a continuation is about to truncate, kept
    only while that continuation is in flight (see above). Absent from a
    run that has never been continued from an earlier step.
xtrg.ckpt
    Native density-matrix `alice.algorithm.xtrg.Artifact` for the most
    recent cooling step. Alice removes it after successful completion.
artifacts/step_XX.ckpt
    Archived density-matrix artifacts, controlled by
    `algorithm.save_artifacts` and `algorithm.save_artifacts_since`, and the
    only way to continue a run that has already finished — a run archiving
    nothing can only be recomputed from τ₀.

Written to `main/attempts/attempt_NN/` (one attempt's own execution):

alice.log
    Alice logging output from this attempt (DEBUG and above, timestamped).
iknot.log
    Combined log: IntraKnot bookkeeping messages plus Alice output (via
    log propagation to the root logger).
info.json
    Key scalar results: algorithm, alice_version, system_size, finished,
    n_steps, start_step and resumed_from (which step this attempt started
    at and which file it came from), free_energies_per_site (full curve, one
    entry per cooling step), max_bond_dim, bond_dims. Other thermodynamic
    observables are only kept in thermodynamics.csv, not duplicated here.
thermodynamics.csv
    Per-step thermodynamic history: step, beta, temperature, log_z,
    free_energy_per_site, energy_per_site, specific_heat_per_site,
    entropy_per_site, discarded_weight. Already the full history from step 0
    even on a resumed attempt, since Alice reconstructs it from the shared
    `thermal.ckpt`.
status.json
    AttemptStatus record written by IntraKnot (not by Alice).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import math
import shutil
import socket
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import alice
from alice import build_hamiltonian, build_interaction
from alice.algorithm import xtrg
from alice.network.thermal import thermal_mpo

# IntraKnot status helpers — imported from the installed package when run via
# `uv run`, or resolved by climbing the directory tree for direct invocation.
try:
    from intraknot.status import (
        AttemptStatus,
        FailureReason,
        MainStatus,
        RunState,
        write_status,
    )
except ImportError:
    # Fallback: add src/ of the project root so the runner works when invoked
    # directly without `uv run` / an installed editable package.
    _here = Path(__file__).resolve()
    # Climb from  <project>/campaigns/<id>/algorithm/  or
    #             <project>/runs/<id>/algorithm/
    # to find src/intraknot/.
    for _parent in _here.parents:
        _candidate = _parent / "src" / "intraknot"
        if _candidate.is_dir():
            sys.path.insert(0, str(_parent / "src"))
            break
    from intraknot.status import (  # type: ignore[no-redef]
        AttemptStatus,
        FailureReason,
        MainStatus,
        RunState,
        write_status,
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class _ShortLevelFormatter(logging.Formatter):
    """Formatter that abbreviates WARNING to WARN so the bracketed level tag
    is always exactly 6 characters (`[WARN ]`, `[INFO ]`, …) rather than
    overflowing to `[WARNING]`.
    """

    _ABBREV: Dict[str, str] = {"WARNING": "WARN"}

    def format(self, record: logging.LogRecord) -> str:
        original = record.levelname
        record.levelname = self._ABBREV.get(original, original)
        formatted = super().format(record)
        record.levelname = original
        return formatted


class _NonFiniteValue(RuntimeError):
    """Raised when XTRG returns a NaN or infinite scalar observable."""


class _EngineMismatch(ValueError):
    """Raised when `config.toml` selects an engine other than XTRG."""


class _CheckpointMissing(FileNotFoundError):
    """Raised when a checkpoint this attempt must resume from is absent.

    Either `[algorithm] resume_from_step` names an archived step that was
    never written, or a starting state past step 0 has no `thermal.ckpt`
    beside it to recover the β / log Z history from. Neither is fixable by
    running the same attempt again, so both are reported as `invalid`.
    """


class _CheckpointIncompatible(ValueError):
    """Raised when the recovered history disagrees with the starting state.

    Mirrors the consistency conditions Alice's `xtrg.run` enforces on a
    resumed run, checked here first so the failure is classified as a
    checkpoint problem rather than a generic bad-parameter one, and so it
    is reported before the Hamiltonian is built.
    """


# ---------------------------------------------------------------------------
# Attempt directory resolution
# ---------------------------------------------------------------------------

def _resolve_attempt_dir(run_dir: Path) -> Tuple[Path, str]:
    """Return the path and name of the next attempt directory.

    Creates `main/attempts/` if absent. The next attempt index is one more
    than the highest existing `attempt_NN` directory. Names that do not parse
    as an integer suffix (e.g. `attempt_backup`) are ignored so they cannot
    abort attempt numbering.

    Parameters
    ----------
    run_dir:
        Root of the run directory.

    Returns
    -------
    attempt_dir, attempt_name
        Absolute path and bare name (e.g. `"attempt_02"`).
    """
    attempts_root = run_dir / "main" / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)

    indices = []
    for path in attempts_root.iterdir():
        if path.is_dir() and path.name.startswith("attempt_"):
            try:
                indices.append(int(path.name.removeprefix("attempt_")))
            except ValueError:
                continue
    name = f"attempt_{max(indices, default=0) + 1:02d}"
    return attempts_root / name, name


def _find_resume_checkpoint(run_dir: Path) -> Optional[Path]:
    """Return `main/xtrg.ckpt` if present, else `None`.

    `checkpoint_dir` is set to `main/` (see `run` below), shared across
    every attempt of this run, so there is exactly one place to look rather
    than scanning `attempts/*`. Unlike DMRG, XTRG's `run()` always completes
    its full step schedule on any clean return (there is no "budget
    exhausted, not finished" case), so `xtrg.ckpt` is the only resumable
    state — it survives only if a prior attempt crashed or was killed
    mid-step; Alice always removes it on success.

    Parameters
    ----------
    run_dir:
        Root of the run directory.

    Returns
    -------
    Path | None
        Path to `main/xtrg.ckpt`, or `None` if it does not exist.
    """
    ckpt = run_dir / "main" / "xtrg.ckpt"
    return ckpt if ckpt.exists() else None


def _archived_artifact_path(run_dir: Path, step: int) -> Path:
    """Return the path Alice archives the artifact of `step` at.

    Parameters
    ----------
    run_dir:
        Root of the run directory.
    step:
        Cooling step index.

    Returns
    -------
    Path
        `main/artifacts/step_XX.ckpt`, whether or not it exists.
    """
    return run_dir / "main" / "artifacts" / f"step_{step:02d}.ckpt"


def _latest_archived_step(run_dir: Path, n_steps: int) -> Optional[int]:
    """Return the highest archived step index not past `n_steps`.

    Steps beyond `n_steps` are skipped rather than selected and rejected
    later: `n_steps` is the absolute step index to stop at, so Alice raises
    on a starting state past it. Lowering `n_steps` between attempts should
    therefore shorten the schedule, not break the run.

    Names that do not parse as `step_<int>.ckpt` are ignored, which also
    skips Alice's transient `step_XX_lock.ckpt` write-lock files.

    Parameters
    ----------
    run_dir:
        Root of the run directory.
    n_steps:
        Absolute step index this attempt stops at.

    Returns
    -------
    int | None
        Highest usable archived step, or `None` when `main/artifacts/` holds
        no archive at or below `n_steps`.
    """
    artifacts_dir = run_dir / "main" / "artifacts"
    if not artifacts_dir.is_dir():
        return None

    steps = []
    for path in artifacts_dir.glob("step_*.ckpt"):
        try:
            step = int(path.stem.removeprefix("step_"))
        except ValueError:
            continue
        if 0 <= step <= n_steps:
            steps.append(step)
    return max(steps) if steps else None


def _resolve_start_checkpoint(
    run_dir: Path,
    n_steps: int,
    resume_from_step: Optional[int],
) -> Tuple[Optional[Path], str]:
    """Choose the checkpoint this attempt starts its cooling schedule from.

    Alice >= 0.2.6 accepts a starting state at any step covered by
    `thermal.ckpt`, so an archived `step_XX.ckpt` is a genuine restart point
    and not only a record. The candidates are tried in this order:

    1. `main/artifacts/step_XX.ckpt` for an explicit
       `[algorithm] resume_from_step`. It outranks `xtrg.ckpt` because
       overriding the automatic choice is the whole point of the setting:
       re-cooling a segment that a previous attempt already covered.
    2. `main/xtrg.ckpt`, left behind by an attempt that crashed or was
       killed mid-step (Alice removes it on success).
    3. The highest archived step at or below `n_steps`, which continues a
       run that already finished a shorter schedule.
    4. Nothing, so the caller builds ρ(τ₀) from scratch.

    Parameters
    ----------
    run_dir:
        Root of the run directory.
    n_steps:
        Absolute step index this attempt stops at.
    resume_from_step:
        Explicitly requested starting step, or `None` to choose
        automatically.

    Returns
    -------
    Path | None
        Checkpoint to load the starting state from, or `None` for a fresh
        ρ(τ₀) build.
    str
        Short provenance label recorded in `info.json`: `"fresh"`,
        `"xtrg.ckpt"`, or `"artifacts/step_XX.ckpt"`.

    Raises
    ------
    _CheckpointMissing
        If `resume_from_step` names a step that was never archived.
    """
    if resume_from_step is not None:
        path = _archived_artifact_path(run_dir, resume_from_step)
        if not path.exists():
            raise _CheckpointMissing(
                f"algorithm.resume_from_step = {resume_from_step} requires "
                f"{path}, but no such archive exists; check that the earlier "
                "attempts ran with algorithm.save_artifacts enabled"
            )
        return path, f"artifacts/{path.name}"

    live = _find_resume_checkpoint(run_dir)
    if live is not None:
        return live, live.name

    step = _latest_archived_step(run_dir, n_steps)
    if step is not None:
        path = _archived_artifact_path(run_dir, step)
        return path, f"artifacts/{path.name}"

    return None, "fresh"


def _parse_resume_from_step(cfg_algo: Dict[str, Any], n_steps: int) -> Optional[int]:
    """Read and validate `[algorithm] resume_from_step`.

    The key is IntraKnot's own: Alice's `Options.from_toml` ignores keys it
    does not recognize, so it rides along in the same `[algorithm]` section
    without disturbing the options it builds.

    Parameters
    ----------
    cfg_algo:
        `config["algorithm"]` dict.
    n_steps:
        Absolute step index this attempt stops at.

    Returns
    -------
    int | None
        Requested starting step, or `None` when the key is absent.

    Raises
    ------
    ValueError
        If the value is not an integer in `0 … n_steps`.
    """
    value = cfg_algo.get("resume_from_step")
    if value is None:
        return None
    # bool is a subclass of int, and `resume_from_step = true` is far more
    # likely a typo for a step index than a deliberate step 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"algorithm.resume_from_step must be an integer, got {value!r}"
        )
    if value < 0:
        raise ValueError(
            f"algorithm.resume_from_step must be non-negative, got {value}"
        )
    if value > n_steps:
        raise ValueError(
            f"algorithm.resume_from_step = {value} is past algorithm.n_steps = "
            f"{n_steps}; n_steps counts cooling steps from tau_0, so it is the "
            "absolute step index to stop at, not a number of steps to add"
        )
    return value


def _update_current(run_dir: Path, attempt_name: str) -> None:
    """Update `main/current` symlink (or `main/current.txt`) to point to the given attempt.

    Tries a symlink first; falls back to `current.txt` on filesystems that
    do not support symlinks.

    Parameters
    ----------
    run_dir:
        Root of the run directory.
    attempt_name:
        Bare attempt name, e.g. `"attempt_02"`.
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
# Option and history validation
# ---------------------------------------------------------------------------

def _validate_config(cfg_algo: Dict[str, Any]) -> None:
    """Reject orchestration settings inconsistent with this runner.

    `[algorithm] engine` decides which runner the submit script invokes, so
    a mismatch here means the wrong runner was dispatched. Failing fast is
    safer than silently running a different algorithm.

    Parameters
    ----------
    cfg_algo:
        `config["algorithm"]` dict.

    Raises
    ------
    _EngineMismatch
        If `engine` is set to anything other than `"xtrg"`.
    """
    engine = str(cfg_algo.get("engine", "xtrg")).lower()
    if engine != "xtrg":
        raise _EngineMismatch(
            f"run_xtrg.py requires algorithm.engine='xtrg', got {engine!r}"
        )


def _validate_options(opts: xtrg.Options) -> None:
    """Validate option ranges not checked by Alice `Options.__post_init__`.

    Alice already normalizes scheme aliases and rejects a negative
    `save_artifacts_since`. The remaining bounds below would otherwise
    surface as opaque numerical failures after a long job has started.

    Parameters
    ----------
    opts:
        XTRG options parsed from the `[algorithm]` section.

    Raises
    ------
    ValueError
        If a numeric option is non-finite, non-positive, or otherwise
        outside the range this runner accepts.
    """
    if not math.isfinite(opts.tau_0) or opts.tau_0 <= 0.0:
        raise ValueError(f"algorithm.tau_0 must be finite and positive, got {opts.tau_0}")
    if opts.n_steps < 1:
        raise ValueError(f"algorithm.n_steps must be at least 1, got {opts.n_steps}")
    if opts.taylor_order < 1:
        raise ValueError(
            f"algorithm.taylor_order must be at least 1, got {opts.taylor_order}"
        )
    if opts.max_bond is not None and opts.max_bond < 1:
        raise ValueError(
            f"algorithm.max_bond must be positive or omitted, got {opts.max_bond}"
        )
    if not math.isfinite(opts.trunc_thresh) or opts.trunc_thresh < 0.0:
        raise ValueError(
            "algorithm.trunc_thresh must be finite and non-negative, "
            f"got {opts.trunc_thresh}"
        )
    if opts.n_sweeps < 1:
        raise ValueError(f"algorithm.n_sweeps must be at least 1, got {opts.n_sweeps}")
    if opts.env_window < 1:
        raise ValueError(f"algorithm.env_window must be at least 1, got {opts.env_window}")
    if opts.expand_k < 1:
        raise ValueError(f"algorithm.expand_k must be at least 1, got {opts.expand_k}")
    if opts.expand_alpha is not None and opts.expand_alpha < 1:
        raise ValueError(
            "algorithm.expand_alpha must be positive or omitted, "
            f"got {opts.expand_alpha}"
        )


def _load_history(run_dir: Path) -> Optional[xtrg.Summary]:
    """Load `main/thermal.ckpt` if a previous attempt wrote one.

    The file holds the β / log Z history Alice recovers a resumed run from,
    and is small (a handful of float lists), so reading it up front to
    validate the starting state costs nothing worth avoiding.

    Parameters
    ----------
    run_dir:
        Root of the run directory.

    Returns
    -------
    xtrg.Summary | None
        Recorded thermodynamic history, or `None` when the file is absent.
    """
    path = run_dir / "main" / "thermal.ckpt"
    if not path.exists():
        return None
    return xtrg.Summary.load(path)


def _validate_resumption(
    run_dir: Path,
    history: Optional[xtrg.Summary],
    state: xtrg.Artifact,
    opts: xtrg.Options,
) -> None:
    """Check the starting state against the recorded history and `n_steps`.

    Repeats the conditions Alice's `xtrg.run` enforces on a resumed run, so
    that an inconsistency is reported as a checkpoint problem — rather than
    as a generic bad-parameter failure — and is reported before the
    Hamiltonian is built. A state at step 0 needs no history: Alice recovers
    nothing in that case and simply records ρ(τ₀) as the first grid point.

    Parameters
    ----------
    run_dir:
        Root of the run directory, named in error messages.
    history:
        Recorded history from `main/thermal.ckpt`, or `None` if absent.
    state:
        Starting density-matrix snapshot for this attempt.
    opts:
        XTRG options this attempt runs with.

    Raises
    ------
    _CheckpointMissing
        If `state.step > 0` but no `thermal.ckpt` exists to recover the
        β / log Z history for the steps already taken.
    _CheckpointIncompatible
        If the state is past `opts.n_steps`, or if the history stops before
        `state.step`, disagrees with `state.beta` there, or was built with a
        different τ₀.
    """
    main = run_dir / "main"
    if state.step > opts.n_steps:
        raise _CheckpointIncompatible(
            f"starting state is at step {state.step}, past algorithm.n_steps = "
            f"{opts.n_steps}; n_steps counts cooling steps from tau_0, so it is "
            "the absolute step index to stop at, not a number of steps to add"
        )
    if state.step == 0:
        return

    if history is None:
        raise _CheckpointMissing(
            f"resuming at step {state.step} requires {main / 'thermal.ckpt'} to "
            "recover the beta / log Z history of the steps already taken, but "
            "no such file exists"
        )

    betas = [float(beta) for beta in history.betas]
    if len(betas) <= state.step:
        raise _CheckpointIncompatible(
            f"{main / 'thermal.ckpt'} only reaches step {len(betas) - 1}, but the "
            f"starting state is at step {state.step}"
        )
    if not math.isclose(betas[state.step], state.beta):
        raise _CheckpointIncompatible(
            f"{main / 'thermal.ckpt'} records beta = {betas[state.step]:.6g} at step "
            f"{state.step}, but the starting state has beta = {state.beta:.6g}"
        )
    if not math.isclose(betas[0], opts.tau_0):
        raise _CheckpointIncompatible(
            f"{main / 'thermal.ckpt'} was built with tau_0 = {betas[0]:.6g}, but "
            f"algorithm.tau_0 = {opts.tau_0:.6g}; continuing a run requires the "
            "same tau_0, since it anchors the whole beta grid"
        )


def _backup_history(run_dir: Path) -> None:
    """Copy `main/thermal.ckpt` to `main/thermal_old.ckpt`.

    Called when this attempt starts below the last step the history records,
    which means Alice will truncate the later entries and recompute them —
    overwriting both `thermal.ckpt` and the `step_XX.ckpt` archives past the
    starting step. The copy keeps the pre-continuation series readable until
    the new one is confirmed good (see `run`).

    A copy rather than a rename: Alice reads `thermal.ckpt` in place to
    recover the history prefix, so moving it away would break the very run
    this backup protects. An existing `thermal_old.ckpt` is left untouched,
    since it comes from an earlier crashed continuation and is the older —
    hence more original — of the two.

    Parameters
    ----------
    run_dir:
        Root of the run directory.
    """
    source = run_dir / "main" / "thermal.ckpt"
    backup = run_dir / "main" / "thermal_old.ckpt"
    if backup.exists():
        logger.info("Keeping the existing history backup at %s", backup)
        return
    shutil.copy2(source, backup)
    logger.info("Copied the history about to be truncated to %s", backup)


def _validate_summary(summary: xtrg.Summary, artifact: xtrg.Artifact) -> None:
    """Ensure Alice's thermodynamic history and final artifact are aligned.

    A finished XTRG run records `n_steps + 1` thermodynamic samples (Taylor
    init plus one sample after each squaring) and `n_steps` discarded
    weights (one per squaring). The returned artifact must match the last
    cooling step.

    Parameters
    ----------
    summary:
        Thermodynamic history returned by `xtrg.run`.
    artifact:
        Final density-matrix snapshot returned by `xtrg.run`.

    Raises
    ------
    _NonFiniteValue
        If any recorded scalar is NaN or infinite.
    RuntimeError
        If series lengths or the final artifact disagree with `n_steps`.
    """
    expected = summary.n_steps + 1
    series = {
        "betas": summary.betas,
        "log_z": summary.log_z,
        "free_energies": summary.free_energies,
        "energies": summary.energies,
        "specific_heats": summary.specific_heats,
        "entropies": summary.entropies,
    }
    for name, values in series.items():
        if len(values) != expected:
            raise RuntimeError(
                f"XTRG returned {len(values)} {name} values; expected {expected}"
            )
        if not all(math.isfinite(float(value)) for value in values):
            raise _NonFiniteValue(f"XTRG returned a non-finite value in {name}")

    if len(summary.discarded_weights) != summary.n_steps:
        raise RuntimeError(
            "XTRG returned "
            f"{len(summary.discarded_weights)} discarded weights; "
            f"expected {summary.n_steps}"
        )
    if not all(math.isfinite(float(value)) for value in summary.discarded_weights):
        raise _NonFiniteValue("XTRG returned a non-finite discarded weight")
    if artifact.step != summary.n_steps:
        raise RuntimeError(
            f"final artifact step {artifact.step} does not match summary step "
            f"{summary.n_steps}"
        )
    if not math.isclose(artifact.beta, summary.betas[-1], rel_tol=1e-12):
        raise RuntimeError(
            f"final artifact beta {artifact.beta} does not match summary beta "
            f"{summary.betas[-1]}"
        )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_observables(
    attempt_dir: Path,
    summary: xtrg.Summary,
    artifact: xtrg.Artifact,
    L: int,
    *,
    start_step: int,
    resumed_from: str,
) -> None:
    """Write `info.json` to the attempt directory.

    Parameters
    ----------
    attempt_dir:
        Attempt output directory.
    summary:
        Completed XTRG thermodynamic summary.
    artifact:
        Final density-matrix snapshot (bond dimensions).
    L:
        Chain length.
    start_step:
        Cooling step this attempt started from; 0 for a fresh ρ(τ₀).
    resumed_from:
        Provenance label of the starting state, as returned by
        `_resolve_start_checkpoint`.
    """
    bond_dims = artifact.rho.bond_dims
    obs: Dict[str, Any] = {
        "algorithm": "xtrg",
        "alice_version": alice.__version__,
        "system_size": L,
        "finished": summary.finished,
        "n_steps": summary.n_steps,
        # Provenance of this attempt's starting state: which steps it
        # actually computed, and which file it picked them up from.
        "start_step": start_step,
        "resumed_from": resumed_from,
        # Full free-energy-per-site curve across the cooling schedule (one
        # entry per row of thermodynamics.csv); the other thermodynamic
        # observables are only kept in thermodynamics.csv, not duplicated
        # here.
        "free_energies_per_site": summary.free_energies,
        "max_bond_dim": max(bond_dims) if bond_dims else 0,
        "bond_dims": bond_dims,
    }
    # allow_nan=False makes a missed non-finite value fail loudly here rather
    # than writing a JSON file that later analysis cannot parse.
    (attempt_dir / "info.json").write_text(
        json.dumps(obs, indent=2, allow_nan=False) + "\n"
    )


def _write_thermodynamics(attempt_dir: Path, summary: xtrg.Summary) -> None:
    """Write `thermodynamics.csv` to the attempt directory.

    Step 0 is the Taylor-initialized ρ(τ₀) and has no discarded weight.
    Later rows use the discarded weight of the squaring that produced them.

    Parameters
    ----------
    attempt_dir:
        Attempt output directory.
    summary:
        Completed XTRG thermodynamic summary.
    """
    path = attempt_dir / "thermodynamics.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "beta",
            "temperature",
            "log_z",
            "free_energy_per_site",
            "energy_per_site",
            "specific_heat_per_site",
            "entropy_per_site",
            "discarded_weight",
        ])
        for step, beta in enumerate(summary.betas):
            discarded_weight = (
                0.0 if step == 0 else summary.discarded_weights[step - 1]
            )
            writer.writerow([
                step,
                f"{beta:.16g}",
                f"{1.0 / beta:.16g}",
                f"{summary.log_z[step]:.16g}",
                f"{summary.free_energies[step]:.16g}",
                f"{summary.energies[step]:.16g}",
                f"{summary.specific_heats[step]:.16g}",
                f"{summary.entropies[step]:.16g}",
                f"{discarded_weight:.16g}",
            ])


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(run_dir: Path) -> None:
    """Execute one XTRG attempt for the given run directory.

    Writes all outputs to a newly created `main/attempts/attempt_NN/`
    directory and updates `main/status.json` and `main/current`.

    Parameters
    ----------
    run_dir:
        Root of the run directory (must contain `config.toml`).
    """
    run_dir = run_dir.resolve()
    config_path = run_dir / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.toml not found in {run_dir}")

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    cfg_model = {
        "geometry": cfg.get("geometry", {}),
        "model":    cfg.get("model", {}),
        "plugin":   cfg.get("plugin", {}),
    }
    cfg_algo = cfg.get("algorithm", {})

    # Resolve attempt directory and announce intent.
    attempt_dir, attempt_name = _resolve_attempt_dir(run_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    _update_current(run_dir, attempt_name)

    # Configure Alice's own logger: stream (INFO+) and alice.log (DEBUG+).
    alice.configure_logging(log_file=str(attempt_dir / "alice.log"))
    # Replace the formatter on Alice's file handler so alice.log also uses the
    # abbreviated WARN tag. Alice attaches handlers in configure_logging(), so
    # we patch them here without modifying the Alice package.
    _log_fmt = dict(
        fmt="%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for _h in logging.getLogger("alice").handlers:
        if isinstance(_h, logging.FileHandler):
            _h.setFormatter(_ShortLevelFormatter(**_log_fmt))
    # Attach a second file handler to the root logger so that all records
    # (IntraKnot's own + Alice's via propagation) also land in iknot.log.
    iknot_log = attempt_dir / "iknot.log"
    file_handler = logging.FileHandler(iknot_log)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(_ShortLevelFormatter(**_log_fmt))
    logging.getLogger().addHandler(file_handler)
    # The root logger's default level is WARNING, which silently drops INFO
    # records from this runner before they reach any handler. Set the runner
    # logger's level explicitly so its messages propagate to the root handlers.
    logger.setLevel(logging.INFO)

    logger.info("IntraKnot XTRG runner")
    logger.info("  run_dir    : %s", run_dir)
    logger.info("  attempt    : %s", attempt_name)

    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    hostname = socket.gethostname()

    # Mark main as running, recording which node is executing the attempt.
    main_status_path = run_dir / "main" / "status.json"
    write_status(
        main_status_path,
        MainStatus(
            state=RunState.RUNNING,
            current_attempt=attempt_name,
            restartable=False,
            nodename=hostname.split(".")[0],
            hostname=hostname,
        ),
    )

    attempt_status_path = attempt_dir / "status.json"
    write_status(
        attempt_status_path,
        AttemptStatus(state=RunState.RUNNING, started_at=started_at),
    )

    end_state = RunState.FAILED
    end_reason: Optional[FailureReason] = FailureReason.SCHEDULER_FAILURE
    # Retrying resumes from main/xtrg.ckpt when one exists (see
    # _resolve_start_checkpoint below), so this is a genuine continuation,
    # not a restart from tau_0.
    restartable = True

    try:
        _validate_config(cfg_algo)

        # Build XTRG options from the `[algorithm]` section. Override
        # checkpoint_dir and artifacts_dir so Alice writes thermal.ckpt,
        # xtrg.ckpt, and archived artifacts to main/, shared across every
        # attempt of this run instead of nested under this attempt's own
        # directory.
        opts = xtrg.Options.from_toml(cfg_algo)
        _validate_options(opts)
        resume_from_step = _parse_resume_from_step(cfg_algo, opts.n_steps)
        opts.checkpoint_dir = str(run_dir / "main")
        opts.artifacts_dir = str(run_dir / "main" / "artifacts")

        # Resolve where this attempt starts from and check it against the
        # recorded history before any of the expensive work below, so a
        # mismatched checkpoint fails in seconds rather than after the
        # Hamiltonian and rho are in memory.
        start_path, start_source = _resolve_start_checkpoint(
            run_dir, opts.n_steps, resume_from_step,
        )
        history = _load_history(run_dir)
        state: Optional[xtrg.Artifact] = None
        if start_path is not None:
            state = xtrg.Artifact.load(start_path)
            _validate_resumption(run_dir, history, state, opts)

        # Build Hamiltonian.
        interactions, spc, geo = build_interaction(cfg_model)
        mpo = build_hamiltonian(interactions, geo.L, spc)
        L = geo.L

        # Build the starting density-matrix state: resume from the checkpoint
        # resolved above, or build rho(tau_0) fresh via a Taylor expansion.
        if state is not None:
            start_step = state.step
            logger.info(
                "Resuming from %s (step %d / %d, beta=%.6g)",
                start_path, start_step, opts.n_steps, state.beta,
            )
            # Alice's run() recovers the beta/log Z history for step > 0 by
            # reading thermal.ckpt from opts.checkpoint_dir; since that's
            # already the shared main/ directory, no copying is needed.
        else:
            if history is not None:
                # A finished history with nothing to square further means the
                # earlier attempts archived no artifact (save_artifacts off,
                # or save_artifacts_since past n_steps). The cooling schedule
                # is about to be redone from tau_0 rather than continued,
                # which is the opposite of what raising n_steps suggests.
                logger.warning(
                    "main/thermal.ckpt exists but no archived artifact at or "
                    "below step %d is available to continue from; rebuilding "
                    "rho(tau_0) and recomputing the whole schedule. Enable "
                    "algorithm.save_artifacts to make future runs continuable.",
                    opts.n_steps,
                )
            logger.info(
                "Building initial state: rho(tau_0=%.6g) via Taylor expansion "
                "(order %d)", opts.tau_0, opts.taylor_order,
            )
            rho0 = thermal_mpo(mpo, opts.tau_0, opts.taylor_order, spc)
            state = xtrg.Artifact(rho=rho0, beta=opts.tau_0, step=0)
            start_step = 0

        # Preserve the recorded series when this attempt starts below its
        # end: Alice truncates the later entries and recomputes them, both
        # in thermal.ckpt and in the step_XX.ckpt archives.
        if history is not None and len(history.betas) - 1 > start_step:
            _backup_history(run_dir)

        # --- Run XTRG ---
        summary, artifact = xtrg.run(state, opts)
        _validate_summary(summary, artifact)

        # The replacement history is now on disk and validated, so the backup
        # of the series it superseded has served its purpose. Removed here
        # and nowhere else: a failed attempt must leave it behind.
        (run_dir / "main" / "thermal_old.ckpt").unlink(missing_ok=True)

        # Write observables and thermodynamic history.
        _write_observables(
            attempt_dir, summary, artifact, L,
            start_step=start_step,
            resumed_from=start_source,
        )
        _write_thermodynamics(attempt_dir, summary)

        # Determine final status. `Summary.finished` is True only for the
        # summary returned by a completed `run()` call (Alice raises on
        # mid-run interruption rather than returning a partial summary, so
        # this is always True on this success path; the False branch below
        # only matters if that ever changes). Should it ever be False,
        # `restartable=True` is now accurate: the next attempt will resume
        # from `main/xtrg.ckpt` via `_find_resume_checkpoint` above rather
        # than restarting the cooling schedule from tau_0.
        if summary.finished:
            end_state = RunState.COMPLETED
            end_reason = FailureReason.FINISHED
            restartable = False
        else:
            end_state = RunState.FAILED
            end_reason = FailureReason.NOT_FINISHED
            restartable = True

    except _EngineMismatch:
        logger.exception("Engine mismatch: wrong runner dispatched")
        end_state = RunState.INVALID
        end_reason = FailureReason.BAD_PARAMETERS
        restartable = False
    # Both checkpoint clauses must precede the generic FileNotFoundError /
    # ValueError clause below, which would otherwise absorb them and report
    # a checkpoint problem as a bad-parameter one.
    except _CheckpointMissing:
        logger.exception("Required XTRG checkpoint missing")
        end_state = RunState.INVALID
        end_reason = FailureReason.CHECKPOINT_MISSING
        restartable = False
    except _CheckpointIncompatible:
        logger.exception("XTRG checkpoint inconsistent with recorded history")
        end_state = RunState.INVALID
        end_reason = FailureReason.CHECKPOINT_INCOMPATIBLE
        restartable = False
    except _NonFiniteValue:
        logger.exception("Non-finite XTRG observable")
        end_reason = FailureReason.NAN_DETECTED
        restartable = False
    except MemoryError:
        logger.exception("Out of memory")
        end_reason = FailureReason.OUT_OF_MEMORY
        restartable = True
    except (FileNotFoundError, KeyError, TypeError, ValueError, NotImplementedError):
        logger.exception("Invalid XTRG configuration")
        end_state = RunState.INVALID
        end_reason = FailureReason.BAD_PARAMETERS
        restartable = False
    except RuntimeError:
        logger.exception("XTRG numerical failure")
        end_reason = FailureReason.LINEAR_ALGEBRA_ERROR
        restartable = False
    except Exception:
        logger.exception("Unhandled exception during XTRG")
        end_reason = FailureReason.SCHEDULER_FAILURE
        restartable = True

    ended_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Write attempt status.
    write_status(
        attempt_status_path,
        AttemptStatus(
            state=end_state,
            reason=end_reason,
            restartable=restartable,
            started_at=started_at,
            ended_at=ended_at,
        ),
    )

    # Update main status, preserving the hostname recorded at start.
    write_status(
        main_status_path,
        MainStatus(
            state=end_state,
            current_attempt=attempt_name,
            reason=end_reason,
            restartable=restartable,
            nodename=hostname.split(".")[0],
            hostname=hostname,
        ),
    )

    logger.info(
        "Attempt finished: state=%s reason=%s",
        end_state.value,
        end_reason.value if end_reason else None,
    )

    if end_state != RunState.COMPLETED:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "IntraKnot XTRG runner.\n\n"
            "Runs one XTRG attempt for the specified run directory, "
            "reading scientific configuration from config.toml and writing "
            "outputs to main/attempts/attempt_NN/.\n\n"
            "Example:\n"
            "  uv run run_xtrg.py --run-dir /scratch/user/runs/heis_L64_xtrg"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="Root of the run directory (must contain config.toml).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.run_dir)
