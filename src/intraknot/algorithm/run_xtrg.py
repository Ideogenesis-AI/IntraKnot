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
XTRG, and writes all outputs to the attempt directory.

Usage
-----
    uv run run_xtrg.py --run-dir /path/to/runs/my_run

The script resolves the next attempt automatically by inspecting
`main/attempts/` and incrementing the highest existing index. Alice owns
XTRG checkpoint and artifact persistence inside that attempt directory.
IntraKnot supplies lifecycle bookkeeping and writes small, analysis-friendly
JSON and CSV projections of Alice's thermodynamic history.

XTRG cooling and resumption
---------------------------
An XTRG schedule is a fixed number of doubling steps: Alice writes
`thermal.ckpt` after every completed cooling step and `progress.ckpt` for
the most recent density-matrix snapshot, then removes `progress.ckpt` on
successful completion. If an earlier attempt for this run was interrupted,
this runner resumes automatically: it locates the most recent
`progress.ckpt` across all prior attempts, loads it as the starting
`alice.algorithm.xtrg.Artifact`, copies the matching `thermal.ckpt`
alongside it into the new attempt directory (Alice reads that file to
recover the β/log Z history when resuming past step 0), and continues
squaring from that step onward rather than rebuilding ρ(τ₀) from scratch.
When no prior `progress.ckpt` exists, ρ(τ₀) is built fresh via a Taylor
expansion (`thermal_mpo`).

Outputs (written to `main/attempts/attempt_NN/`)
------------------------------------------------
alice.log
    Alice logging output from this attempt (DEBUG and above, timestamped).
iknot.log
    Combined log: IntraKnot bookkeeping messages plus Alice output (via
    log propagation to the root logger).
thermal.ckpt
    Native thermodynamic `alice.algorithm.xtrg.Summary`, written atomically
    by Alice after every completed cooling step.
progress.ckpt
    Native density-matrix `alice.algorithm.xtrg.Artifact` for the most
    recent cooling step. Alice removes it after successful completion.
artifacts/step_XX.ckpt
    Optional archived density-matrix artifacts, controlled by
    `algorithm.save_artifacts` and `algorithm.save_artifacts_since`.
info.json
    Key scalar results: beta, free energy, energy, specific heat, entropy,
    finished, n_steps, max_bond_dim.
thermodynamics.csv
    Per-step thermodynamic history: step, beta, temperature, log_z,
    free_energy_per_site, energy_per_site, specific_heat_per_site,
    entropy_per_site, discarded_weight.
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


def _find_latest_progress(run_dir: Path) -> Optional[Path]:
    """Return the most recent `progress.ckpt` across all prior attempts, or `None`.

    Iterates attempt directories in reverse order and returns the first
    checkpoint found, so the latest attempt is preferred. A `progress.ckpt`
    is present only when a prior attempt was interrupted mid-schedule;
    Alice deletes it after a successful `run()` call.

    Parameters
    ----------
    run_dir:
        Root of the run directory.

    Returns
    -------
    Path | None
        Path to the checkpoint file, or `None` if no checkpoint exists.
    """
    attempts_root = run_dir / "main" / "attempts"
    if not attempts_root.exists():
        return None

    for attempt_dir in sorted(attempts_root.iterdir(), reverse=True):
        ckpt = attempt_dir / "progress.ckpt"
        if ckpt.exists():
            return ckpt
    return None


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
) -> None:
    """Write `info.json` to the attempt directory.

    Parameters
    ----------
    attempt_dir:
        Attempt output directory.
    summary:
        Completed XTRG thermodynamic summary.
    artifact:
        Final density-matrix snapshot (bond dimensions and last beta).
    L:
        Chain length.
    """
    bond_dims = artifact.rho.bond_dims
    beta = artifact.beta
    obs: Dict[str, Any] = {
        "algorithm": "xtrg",
        "alice_version": alice.__version__,
        "system_size": L,
        "finished": summary.finished,
        "n_steps": summary.n_steps,
        "beta": beta,
        "temperature": 1.0 / beta,
        "log_z": summary.log_z[-1],
        "free_energy_per_site": summary.free_energies[-1],
        "energy_per_site": summary.energies[-1],
        "specific_heat_per_site": summary.specific_heats[-1],
        "entropy_per_site": summary.entropies[-1],
        "discarded_weight": (
            summary.discarded_weights[-1] if summary.discarded_weights else 0.0
        ),
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
    # Retrying resumes from the latest progress.ckpt when one exists (see
    # _find_latest_progress below), so this is a genuine continuation, not a
    # restart from tau_0.
    restartable = True

    # Locate any prior progress checkpoint for resume support, before the
    # try block so a lookup failure cannot be mistaken for an Alice error.
    # Exclude the current (freshly created, empty) attempt dir from the search.
    prior_progress = _find_latest_progress(run_dir)
    if prior_progress is not None and prior_progress.parent == attempt_dir:
        prior_progress = None

    try:
        _validate_config(cfg_algo)

        # Build Hamiltonian.
        interactions, spc, geo = build_interaction(cfg_model)
        mpo = build_hamiltonian(interactions, geo.L, spc)
        L = geo.L

        # Build XTRG options from the `[algorithm]` section. Override
        # checkpoint_dir so Alice writes thermal.ckpt / progress.ckpt
        # directly into the attempt directory.
        opts = xtrg.Options.from_toml(cfg_algo)
        _validate_options(opts)
        opts.checkpoint_dir = str(attempt_dir)

        # Build the starting density-matrix state: resume from the latest
        # progress.ckpt left by an interrupted prior attempt, or build
        # rho(tau_0) fresh via a Taylor expansion.
        if prior_progress is not None:
            state = xtrg.Artifact.load(prior_progress)
            logger.info(
                "Resuming from %s (step %d / %d, beta=%.6g)",
                prior_progress, state.step, opts.n_steps, state.beta,
            )
            # Alice's run() recovers the beta/log Z history for step > 0 by
            # reading thermal.ckpt from opts.checkpoint_dir, so the file
            # written alongside the resumed progress.ckpt must be copied
            # into this attempt's (freshly created, otherwise empty) directory.
            if state.step > 0:
                prior_thermal = prior_progress.parent / "thermal.ckpt"
                if not prior_thermal.exists():
                    raise FileNotFoundError(
                        f"progress.ckpt at {prior_progress} is at step {state.step} "
                        f"but no matching thermal.ckpt was found at {prior_thermal}"
                    )
                shutil.copy2(prior_thermal, attempt_dir / "thermal.ckpt")
        else:
            logger.info(
                "Building initial state: rho(tau_0=%.6g) via Taylor expansion "
                "(order %d)", opts.tau_0, opts.taylor_order,
            )
            rho0 = thermal_mpo(mpo, opts.tau_0, opts.taylor_order, spc)
            state = xtrg.Artifact(rho=rho0, beta=opts.tau_0, step=0)

        # --- Run XTRG ---
        summary, artifact = xtrg.run(state, opts)
        _validate_summary(summary, artifact)

        # Write observables and thermodynamic history.
        _write_observables(attempt_dir, summary, artifact, L)
        _write_thermodynamics(attempt_dir, summary)

        # Determine final status. `Summary.finished` is True only for the
        # summary returned by a completed `run()` call (Alice raises on
        # mid-run interruption rather than returning a partial summary, so
        # this is always True on this success path; the False branch below
        # only matters if that ever changes). Should it ever be False,
        # `restartable=True` is now accurate: the next attempt will resume
        # from `progress.ckpt` via `_find_latest_progress` above rather than
        # restarting the cooling schedule from tau_0.
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
