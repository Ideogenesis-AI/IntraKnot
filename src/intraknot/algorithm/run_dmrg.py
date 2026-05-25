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


"""IntraKnot-aware DMRG runner.

This script is a standalone entry point executed by Slurm from within a run
directory. It reads `config.toml`, builds the Hamiltonian via Alice, runs
DMRG, and writes all outputs to the attempt directory.

Usage
-----
    uv run run_dmrg.py --run-dir /path/to/runs/my_run

The script resolves the next attempt automatically by inspecting
`main/attempts/` and incrementing the highest existing index. If a previous
attempt left a `dmrg.ckpt`, the MPS state is loaded from it and DMRG
continues from the last completed sweep; otherwise a fresh MPS is initialized
via `alice.init_mps`.

MPS initialisation
------------------
`alice.init_mps` is used for all fresh starts. The `[algorithm]` section
controls the initial bond dimension and random seed:

    [algorithm]
    init     = "random"   # "random" (default) or "product"
    max_bond = 32         # bond dimension for init="random"; ignored for "product"
    seed     = 42         # random seed for init="random"

- `init = "product"` calls `init_mps(..., bond_dim=1)` — a deterministic
  product state; recommended as the starting point for 2-site or CBE DMRG.
- `init = "random"` calls `init_mps(..., bond_dim=max_bond)` — a random MPS
  pre-populated with the correct symmetry structure.

DMRG checkpointing
------------------
Alice writes `dmrg.ckpt` atomically after every completed sweep, mirroring
the convention of `configure_logging`. IntraKnot sets `checkpoint_dir` to
the current attempt directory so the per-sweep checkpoint always lands there.
On resumption (`init = "resume"`), IntraKnot loads `dmrg.ckpt` from the most
recent prior attempt and passes `summary.state` to `dmrg.run` as the initial
MPS; the remaining sweep budget is reduced by `summary.n_sweeps` so the total
sweep count stays consistent with the original target.

Outputs (written to `main/attempts/attempt_NN/`)
------------------------------------------------
alice.log
    Alice logging output from this attempt (DEBUG and above, timestamped).
iknot.log
    Combined log: IntraKnot bookkeeping messages plus Alice output (via
    log propagation to the root logger).
dmrg.ckpt
    PyTorch checkpoint written by Alice after every sweep (atomic rename from
    `dmrg_lock.ckpt`). Loadable via `dmrg.Summary.load`.
state.ckpt
    Final canonical state file; written by IntraKnot on successful
    completion using `summary.save`.
info.json
    Key scalar results: energy, energy_per_site, converged, n_sweeps,
    max_bond_dim.
conv.csv
    Per-sweep diagnostics: sweep, energy, delta_energy, discarded_weight,
    converged.
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
import socket
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import alice
from alice import MPS, build_hamiltonian, build_interaction, init_mps
from alice import dmrg
from nicole import load_space

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
# Attempt directory resolution
# ---------------------------------------------------------------------------

def _resolve_attempt_dir(run_dir: Path) -> Tuple[Path, str]:
    """Return the path and name of the next attempt directory.

    Creates `main/attempts/` if absent. The next attempt index is one more
    than the highest existing `attempt_NN` directory.

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

    existing = sorted(
        d.name for d in attempts_root.iterdir()
        if d.is_dir() and d.name.startswith("attempt_")
    )
    if existing:
        last_idx = int(existing[-1].split("_")[1])
        next_idx = last_idx + 1
    else:
        next_idx = 1

    name = f"attempt_{next_idx:02d}"
    return attempts_root / name, name


def _find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    """Return the most recent `dmrg.ckpt` across all prior attempts, or `None`.

    Iterates attempt directories in reverse order and returns the first
    checkpoint found, so the latest attempt is preferred.

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
        ckpt = attempt_dir / "dmrg.ckpt"
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
# Physical-space helper
# ---------------------------------------------------------------------------

def _load_space_from_cfg(cfg_model: Dict[str, Any]) -> Tuple[Any, Dict]:
    """Call `load_space` with the parameters implied by the model config.

    `build_interaction` returns `(interactions, Spc, geo)` but does not expose
    the operator dict `Op` needed by `alice.init_mps`. This helper derives
    the correct `load_space` call from the model config so the caller can
    obtain `(Spc, Op)` for MPS initialisation.

    Supported categories: `"bosonic"` (Spin), `"fermionic"` (Ferm),
    `"conductor"` (Band).

    Parameters
    ----------
    cfg_model:
        Alice-compatible model config dict with `"geometry"` and `"model"`
        sub-keys (i.e. the dict passed directly to `build_interaction`).

    Returns
    -------
    Spc, Op
        Physical `Index` and operator dict, as returned by `load_space`.

    Raises
    ------
    ValueError
        For unknown `category` values.
    """
    model_cfg = cfg_model.get("model", {})
    category = model_cfg.get("category", "bosonic").lower()
    symmetry = model_cfg.get("symmetry", "U1")

    if category == "bosonic":
        spin = model_cfg.get("spin", 0.5)
        return load_space("Spin", symmetry, {"J": spin})
    if category == "fermionic":
        return load_space("Ferm", symmetry)
    if category == "conductor":
        return load_space("Band", symmetry)

    raise ValueError(
        f"Unknown model category {category!r}. "
        "Expected 'bosonic', 'fermionic', or 'conductor'."
    )


# ---------------------------------------------------------------------------
# MPS initialisation
# ---------------------------------------------------------------------------

def _init_mps(
    cfg_model: Dict[str, Any],
    cfg_algo: Dict[str, Any],
    L: int,
    prior_checkpoint: Optional[Path],
) -> Tuple[MPS, int]:
    """Initialise the MPS for a DMRG run.

    Three strategies controlled by `cfg_algo["init"]`:

    - `"product"` — deterministic product state via `init_mps(..., bond_dim=1)`.
      Bond dimension grows during DMRG. Best paired with 2-site or CBE DMRG.
    - `"random"` — random MPS via `init_mps(..., bond_dim=max_bond)`.
    - `"resume"` — load `summary.state` from `prior_checkpoint` and reduce
      the sweep budget by the number of sweeps already completed.

    Falls back to `"random"` if `"resume"` is requested but no checkpoint
    exists.

    Parameters
    ----------
    cfg_model:
        Alice-compatible model config dict (`{"geometry": ..., "model": ...}`),
        used to initialise the physical Hilbert space.
    cfg_algo:
        `config["algorithm"]` dict.
    L:
        Chain length.
    prior_checkpoint:
        Path to `dmrg.ckpt` from a previous attempt, or `None`.

    Returns
    -------
    mps, sweeps_done
        The initial MPS in right-canonical form (`center = 0`) and the
        number of DMRG sweeps already completed (non-zero only when resuming).
    """
    init_strategy = cfg_algo.get("init", "random")
    bond_dim = cfg_algo.get("max_bond", 32)
    seed = cfg_algo.get("seed", 42)

    if init_strategy == "resume":
        if prior_checkpoint is not None:
            logger.info("Resuming from checkpoint: %s", prior_checkpoint)
            prev = dmrg.Summary.load(prior_checkpoint)
            mps = prev.state
            mps.canonical(0)
            logger.info(
                "  loaded %d sweeps, energy=%.10g, converged=%s",
                prev.n_sweeps, prev.energy, prev.converged,
            )
            return mps, prev.n_sweeps
        logger.warning(
            "init=resume requested but no prior checkpoint found; "
            "falling back to random initialisation."
        )

    Spc, Op = _load_space_from_cfg(cfg_model)

    if init_strategy == "product":
        logger.info("Initialising product-state MPS (bond_dim=1)")
        mps = init_mps(L, Spc, Op, bond_dim=1, seed=seed)
    else:
        logger.info("Initialising random MPS: bond_dim=%d, seed=%d", bond_dim, seed)
        mps = init_mps(L, Spc, Op, bond_dim=bond_dim, seed=seed)

    return mps, 0


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_observables(
    attempt_dir: Path,
    summary: dmrg.Summary,
    L: int,
) -> None:
    """Write `info.json` to the attempt directory.

    Parameters
    ----------
    attempt_dir:
        Attempt output directory.
    summary:
        Completed DMRG summary.
    L:
        Chain length (for per-site energy).
    """
    obs: Dict[str, Any] = {
        "energy": summary.energy,
        "energy_per_site": summary.energy / L if L > 0 else float("nan"),
        "converged": summary.converged,
        "n_sweeps": summary.n_sweeps,
        "max_bond_dim": max(summary.bond_dims) if summary.bond_dims else 0,
        "bond_dims": summary.bond_dims,
    }
    (attempt_dir / "info.json").write_text(
        json.dumps(obs, indent=2) + "\n"
    )


def _write_convergence(attempt_dir: Path, summary: dmrg.Summary) -> None:
    """Write `conv.csv` to the attempt directory.

    Parameters
    ----------
    attempt_dir:
        Attempt output directory.
    summary:
        Completed DMRG summary.
    """
    path = attempt_dir / "conv.csv"
    energies: List[float] = summary.energies
    dw: List[float] = summary.discarded_weights

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sweep", "energy", "delta_energy", "discarded_weight", "converged"])
        for i, e in enumerate(energies):
            delta = abs(e - energies[i - 1]) if i > 0 else math.nan
            # converged only on the last sweep if summary.converged is True.
            conv = (i == len(energies) - 1) and summary.converged
            writer.writerow([
                i + 1,
                f"{e:.12g}",
                f"{delta:.6e}" if not math.isnan(delta) else "nan",
                f"{dw[i]:.6e}" if i < len(dw) else "0.0",
                conv,
            ])


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(run_dir: Path) -> None:
    """Execute one DMRG attempt for the given run directory.

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
    }
    cfg_algo = cfg.get("algorithm", {})
    cfg_output = cfg.get("output", {})

    # Resolve attempt directory and announce intent.
    attempt_dir, attempt_name = _resolve_attempt_dir(run_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    _update_current(run_dir, attempt_name)

    # Configure Alice's own logger: stream (INFO+) and alice.log (DEBUG+).
    alice.configure_logging(log_file=str(attempt_dir / "alice.log"))
    # Attach a second file handler to the root logger so that all records
    # (IntraKnot's own + Alice's via propagation) also land in iknot.log.
    iknot_log = attempt_dir / "iknot.log"
    file_handler = logging.FileHandler(iknot_log)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)

    logger.info("IntraKnot DMRG runner")
    logger.info("  run_dir    : %s", run_dir)
    logger.info("  attempt    : %s", attempt_name)

    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Mark main as running, recording which node is executing the attempt.
    main_status_path = run_dir / "main" / "status.json"
    write_status(
        main_status_path,
        MainStatus(
            state=RunState.RUNNING,
            current_attempt=attempt_name,
            restartable=False,
            hostname=socket.gethostname(),
        ),
    )

    attempt_status_path = attempt_dir / "status.json"
    write_status(
        attempt_status_path,
        AttemptStatus(state=RunState.RUNNING, started_at=started_at),
    )

    # Locate any prior checkpoint for resume support.
    # Exclude the current (empty) attempt dir from the search.
    prior_checkpoint = _find_latest_checkpoint(run_dir)
    if prior_checkpoint and prior_checkpoint.parent == attempt_dir:
        prior_checkpoint = None

    summary: Optional[dmrg.Summary] = None
    end_state = RunState.FAILED
    end_reason: Optional[FailureReason] = FailureReason.SCHEDULER_FAILURE
    restartable = True

    try:
        # Build Hamiltonian.
        interactions, spc, geo = build_interaction(cfg_model)
        mpo = build_hamiltonian(interactions, geo.L, spc)
        L = geo.L
        logger.info("  chain length: %d", L)

        # Initialise MPS (fresh or resumed from checkpoint).
        mps, sweeps_done = _init_mps(cfg_model, cfg_algo, L, prior_checkpoint)

        # Build DMRG options from the `[algorithm]` section. Override
        # checkpoint_dir so Alice writes its per-sweep dmrg.ckpt directly
        # into the attempt directory.
        opts = dmrg.Options.from_toml(cfg_algo)
        opts.checkpoint_dir = str(attempt_dir)

        # Reduce the sweep budget when resuming so the total sweep count
        # relative to the original n_sweeps target stays consistent.
        if sweeps_done > 0:
            remaining = max(1, opts.n_sweeps - sweeps_done)
            logger.info(
                "  resuming: %d sweep(s) already done, %d remaining",
                sweeps_done, remaining,
            )
            opts.n_sweeps = remaining

        # --- Run DMRG ---
        summary = dmrg.run(mps, mpo, opts)
        # dmrg.ckpt is already written by Alice into attempt_dir after each sweep.

        # Save final canonical state.
        if cfg_output.get("save_state", True):
            state_path = attempt_dir / "state.ckpt"
            summary.save(state_path)
            logger.info("Saved final state: %s", state_path)

        # Write observables and convergence table.
        _write_observables(attempt_dir, summary, L)
        _write_convergence(attempt_dir, summary)

        # Determine final status.
        if summary.converged:
            end_state = RunState.COMPLETED
            end_reason = FailureReason.CONVERGED
            restartable = False
        else:
            end_state = RunState.FAILED
            end_reason = FailureReason.NOT_CONVERGED
            restartable = True

    except MemoryError:
        logger.exception("Out of memory")
        end_reason = FailureReason.OUT_OF_MEMORY
        restartable = True
    except Exception:
        logger.exception("Unhandled exception during DMRG")
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
            hostname=socket.gethostname(),
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
            "IntraKnot DMRG runner.\n\n"
            "Runs one DMRG attempt for the specified run directory, "
            "reading scientific configuration from config.toml and writing "
            "outputs to main/attempts/attempt_NN/.\n\n"
            "Example:\n"
            "  uv run run_dmrg.py --run-dir /scratch/user/runs/heis_L64_chi128_g1.0"
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
