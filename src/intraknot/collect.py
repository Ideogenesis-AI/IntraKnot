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


"""Result and status collection from run directories."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def _read_json(path: Path) -> Optional[dict]:
    """Read a JSON file, returning `None` if the file is absent or invalid."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _current_attempt_dir(run_dir: Path) -> Optional[Path]:
    """Resolve `main/current` to an absolute attempt directory path.

    Reads the symlink target or falls back to `main/current.txt`.

    Parameters
    ----------
    run_dir:
        Root of the run directory.

    Returns
    -------
    Path | None
        Absolute path to the current attempt directory, or `None` if not set.
    """
    main_dir = run_dir / "main"
    current = main_dir / "current"

    if current.is_symlink():
        target = (main_dir / current.readlink()).resolve()
        return target if target.exists() else None

    txt = main_dir / "current.txt"
    if txt.exists():
        name = txt.read_text().strip()
        candidate = main_dir / "attempts" / name
        return candidate if candidate.exists() else None

    return None


def collect_run(run_dir: Path) -> dict:
    """Collect status and observables from a run directory into `summary/`.

    Reads `main/status.json` and `main/current/info.json`, then writes
    compact copies to `summary/status.json` and `summary/info.json`.
    All fields that cannot be read are reported as `null`.

    Parameters
    ----------
    run_dir:
        Root of the run directory.

    Returns
    -------
    dict
        Summary dict with at least `run_id`, `state`, `reason`, `restartable`,
        `current_attempt`, `energy`, `converged`.
    """
    run_id = run_dir.name
    summary_dir = run_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    # Read main status.
    main_status = _read_json(run_dir / "main" / "status.json") or {}
    state = main_status.get("state", "unknown")
    reason = main_status.get("reason")
    restartable = main_status.get("restartable", False)
    current_attempt = main_status.get("current_attempt")

    # Read observables from current attempt.
    obs: dict = {}
    attempt_dir = _current_attempt_dir(run_dir)
    if attempt_dir is not None:
        obs = _read_json(attempt_dir / "info.json") or {}

    # Write summary/status.json.
    status_summary = {
        "run_id": run_id,
        "state": state,
        "reason": reason,
        "current_attempt": current_attempt,
        "restartable": restartable,
    }
    (summary_dir / "status.json").write_text(
        json.dumps(status_summary, indent=2) + "\n"
    )

    # Write summary/info.json.
    obs_summary = {"run_id": run_id, **obs}
    (summary_dir / "info.json").write_text(
        json.dumps(obs_summary, indent=2) + "\n"
    )

    return {**status_summary, **obs_summary}


def collect_campaign(
    campaign_dir: Path,
    runs_root: Path,
) -> List[Dict]:
    """Collect results for all runs listed in a campaign's `runs.csv`.

    Calls `collect_run` for each run, then updates the `status` column in
    `runs.csv` in-place.

    Parameters
    ----------
    campaign_dir:
        Campaign directory containing `runs.csv`.
    runs_root:
        Root directory containing run subdirectories.

    Returns
    -------
    list[dict]
        One summary dict per run (same as `collect_run` returns).
    """
    runs_csv = campaign_dir / "runs.csv"
    if not runs_csv.exists():
        return []

    # Read existing rows.
    with open(runs_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ["run_id", "status"]
        rows = list(reader)

    summaries: List[Dict] = []
    status_map: Dict[str, str] = {}

    for row in rows:
        run_id = row.get("run_id", "").strip()
        if not run_id:
            continue
        run_dir = runs_root / run_id
        if not run_dir.exists():
            status_map[run_id] = "missing"
            summaries.append({"run_id": run_id, "state": "missing"})
            continue
        summary = collect_run(run_dir)
        status_map[run_id] = summary.get("state", "unknown")
        summaries.append(summary)

    # Update runs.csv with fresh statuses.
    if "status" not in fieldnames:
        fieldnames = list(fieldnames) + ["status"]

    updated_rows = []
    for row in rows:
        run_id = row.get("run_id", "").strip()
        if run_id in status_map:
            row = dict(row)
            row["status"] = status_map[run_id]
        updated_rows.append(row)

    with open(runs_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    return summaries
