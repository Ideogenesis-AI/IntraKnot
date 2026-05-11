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
directory.  It reads `config.toml`, builds the Hamiltonian via Alice, runs
DMRG, and writes all outputs to the attempt directory.

Usage
-----
    uv run run_dmrg.py --run-dir /path/to/runs/my_run

The script resolves the next attempt automatically by inspecting
`main/attempts/` and incrementing the highest existing index.  If a previous
attempt left a `dmrg.ckpt`, the MPS is loaded from it; otherwise a fresh
random MPS is initialised.

Outputs (written to `main/attempts/attempt_NN/`)
------------------------------------------------
log.txt
    Alice logging output from this attempt.
dmrg.ckpt
    `torch.save` snapshot of the DMRG `Summary` (written after each run,
    including failed ones where possible).
state.ckpt
    Same as `dmrg.ckpt` on successful completion; the canonical
    final-state file.
observables.json
    Key scalar results: energy, energy_per_site, converged, n_sweeps,
    max_bond_dim.
convergence.csv
    Per-sweep diagnostics: sweep, energy, delta_energy, discarded_weight,
    bond_dim, converged.
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
import sys
import tomllib
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

import alice
from alice import MPS, build_hamiltonian, build_interaction
from alice import dmrg
from nicole import Direction, Tensor, load_space
from nicole.index import Index, Sector

# IntraKnot status helpers (imported from the installed package when run from
# within the project, or from a relative path if the package is not on sys.path).
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

    Creates `main/attempts/` if absent.  The next attempt index is one more
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
    """Return the most recent `dmrg.ckpt` across all attempts, or `None`.

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
# MPS initialisation
# ---------------------------------------------------------------------------

def _build_random_mps(spc: Any, L: int, bond_dim: int, symmetry: str, seed: int) -> MPS:
    """Construct a random MPS for the given physical space.

    Distributes `bond_dim` states evenly across charge sectors.  The MPS is
    placed in right-canonical form with `center = 0`.

    Parameters
    ----------
    spc:
        Physical index (Nicole `Index`) for one site.
    L:
        Chain length.
    bond_dim:
        Total bond dimension distributed across sectors.
    symmetry:
        `"U1"` or `"SU2"` (or any Abelian / non-Abelian label understood by
        the symmetry group stored in `spc`).
    seed:
        Base random seed for reproducibility.

    Returns
    -------
    MPS
        Right-canonical random MPS.
    """
    # Determine which charge sectors to include in the bond index.
    # Use ±6 for U1; 0..6 for SU2 (non-negative multiplet labels).
    Smax = 6
    if symmetry.upper() == "SU2":
        bond_charges = list(range(0, Smax + 1))
    else:
        bond_charges = list(range(-Smax, Smax + 1))

    dim_per_sector = max(1, bond_dim // len(bond_charges))
    bulk = Index(
        direction=Direction.IN,
        group=spc.group,
        sectors=tuple(Sector(charge=q, dim=dim_per_sector) for q in bond_charges),
    )
    # Vacuum bond (dim-1, charge-0) for the boundaries.
    vac = Index([Sector(0, 1)], direction=Direction.OUT, group=spc.group)

    tensors = []
    for i in range(L):
        l_idx = vac if i == 0 else bulk
        r_idx = (vac if i == L - 1 else bulk).flip()
        T = Tensor.random(
            [l_idx, r_idx, spc],
            seed=seed + i,
            itags=[f"A{i:02d}", f"A{i + 1:02d}", f"s{i:02d}"],
        )
        tensors.append(T)

    mps = MPS(tensors, center=None)
    mps.canonical(0)
    return mps


def _init_mps(
    cfg_model: Dict[str, Any],
    cfg_algo: Dict[str, Any],
    spc: Any,
    L: int,
    checkpoint: Optional[Path],
) -> MPS:
    """Initialise the MPS for a DMRG run.

    Loads from `checkpoint` if provided and `init` is `"resume"`.
    Otherwise builds a fresh random MPS.

    Parameters
    ----------
    cfg_model:
        `config["model"]` dict (Alice-compatible).
    cfg_algo:
        `config["algorithm"]` dict.
    spc:
        Physical index for one site.
    L:
        Chain length.
    checkpoint:
        Path to a `dmrg.ckpt` file from a previous attempt, or `None`.

    Returns
    -------
    MPS
        Initial MPS in right-canonical form with `center = 0`.
    """
    init_strategy = cfg_algo.get("init", "random")
    bond_dim = cfg_algo.get("max_bond", 32)
    symmetry = cfg_model.get("model", {}).get("symmetry", "U1")
    seed = cfg_algo.get("seed", 42)

    if init_strategy == "resume" and checkpoint is not None:
        logger.info("Resuming from checkpoint: %s", checkpoint)
        data = torch.load(checkpoint, weights_only=True)
        summary = dmrg.Summary.deserialize(data)
        mps = summary.state
        mps.canonical(0)
        return mps

    if init_strategy == "resume" and checkpoint is None:
        logger.warning(
            "init=resume requested but no checkpoint found; falling back to random."
        )

    logger.info("Initialising random MPS: bond_dim=%d, symmetry=%s, seed=%d",
                bond_dim, symmetry, seed)
    return _build_random_mps(spc, L, bond_dim=bond_dim, symmetry=symmetry, seed=seed)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_observables(
    attempt_dir: Path,
    summary: dmrg.Summary,
    L: int,
    cfg_output: Dict[str, Any],
) -> None:
    """Write `observables.json` to the attempt directory.

    Parameters
    ----------
    attempt_dir:
        Attempt output directory.
    summary:
        Completed DMRG summary.
    L:
        Chain length (for per-site energy).
    cfg_output:
        `config["output"]` dict (controls which observables are included).
    """
    obs: Dict[str, Any] = {
        "energy": summary.energy,
        "energy_per_site": summary.energy / L if L > 0 else float("nan"),
        "converged": summary.converged,
        "n_sweeps": summary.n_sweeps,
        "max_bond_dim": max(summary.bond_dims) if summary.bond_dims else 0,
        "bond_dims": summary.bond_dims,
    }
    (attempt_dir / "observables.json").write_text(
        json.dumps(obs, indent=2) + "\n"
    )


def _write_convergence(attempt_dir: Path, summary: dmrg.Summary) -> None:
    """Write `convergence.csv` to the attempt directory.

    Parameters
    ----------
    attempt_dir:
        Attempt output directory.
    summary:
        Completed DMRG summary.
    """
    path = attempt_dir / "convergence.csv"
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

    cfg_model = cfg.get("model", {})
    cfg_algo = cfg.get("algorithm", {})
    cfg_output = cfg.get("output", {})

    # Resolve attempt directory and announce intent.
    attempt_dir, attempt_name = _resolve_attempt_dir(run_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    _update_current(run_dir, attempt_name)

    # Set up logging to both the attempt log file and stderr.
    alice.configure_logging(log_dir=attempt_dir)
    log_file = attempt_dir / "log.txt"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(file_handler)

    logger.info("IntraKnot DMRG runner")
    logger.info("  run_dir    : %s", run_dir)
    logger.info("  attempt    : %s", attempt_name)

    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Mark main as running.
    main_status_path = run_dir / "main" / "status.json"
    write_status(
        main_status_path,
        MainStatus(
            state=RunState.RUNNING,
            current_attempt=attempt_name,
            restartable=False,
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

        # Initialise MPS.
        mps = _init_mps(cfg_model, cfg_algo, spc, L, prior_checkpoint)

        # Build DMRG options from the [algorithm] section.
        opts = dmrg.Options.from_toml(cfg_algo)

        # --- Run DMRG ---
        summary = dmrg.run(mps, mpo, opts)

        # Save checkpoint (always).
        if cfg_output.get("save_checkpoint", True):
            ckpt_path = attempt_dir / "dmrg.ckpt"
            torch.save(summary.serialize(), ckpt_path)
            logger.info("Saved checkpoint: %s", ckpt_path)

        # Save final state.
        if cfg_output.get("save_state", True):
            state_path = attempt_dir / "state.ckpt"
            torch.save(summary.serialize(), state_path)
            logger.info("Saved final state: %s", state_path)

        # Write observables and convergence table.
        _write_observables(attempt_dir, summary, L, cfg_output)
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

    # Update main status.
    write_status(
        main_status_path,
        MainStatus(
            state=end_state,
            current_attempt=attempt_name,
            reason=end_reason,
            restartable=restartable,
        ),
    )

    logger.info("Attempt finished: state=%s reason=%s", end_state.value, end_reason.value if end_reason else None)

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
