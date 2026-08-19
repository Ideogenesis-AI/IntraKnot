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


"""Status model: enums, dataclasses, and JSON serialization helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union


class RunState(str, Enum):
    """Valid execution states for a run, main unit, or attempt."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALID = "invalid"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class FailureReason(str, Enum):
    """Reason codes for terminal or failure states.

    `CONVERGED` and `FINISHED` are success reasons used when `state=completed`:
    `CONVERGED` for algorithms with a numerical convergence criterion (DMRG),
    `FINISHED` for algorithms that instead run a fixed schedule to completion
    (XTRG's cooling steps). Their failure-mode counterparts, `NOT_CONVERGED`
    and `NOT_FINISHED`, follow the same split. The remaining values describe
    failure modes; some are retryable (`TIMEOUT`, `OUT_OF_MEMORY`,
    `SCHEDULER_FAILURE`) and some are not (`BAD_PARAMETERS`,
    `CHECKPOINT_INCOMPATIBLE`).
    """

    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
    FINISHED = "finished"
    NOT_FINISHED = "not_finished"
    MAX_SWEEPS_REACHED = "max_sweeps_reached"
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    NAN_DETECTED = "nan_detected"
    BAD_PARAMETERS = "bad_parameters"
    CHECKPOINT_MISSING = "checkpoint_missing"
    CHECKPOINT_INCOMPATIBLE = "checkpoint_incompatible"
    LINEAR_ALGEBRA_ERROR = "linear_algebra_error"
    SCHEDULER_FAILURE = "scheduler_failure"


# States that are considered final (no further execution expected).
TERMINAL_STATES: frozenset[RunState] = frozenset({
    RunState.COMPLETED,
    RunState.FAILED,
    RunState.INVALID,
    RunState.SKIPPED,
    RunState.CANCELLED,
})

# States where a new attempt may be created (execution failed but science unchanged).
RETRYABLE_STATES: frozenset[RunState] = frozenset({RunState.FAILED})

# Failure reasons that are safe to retry with the same config.
RETRYABLE_REASONS: frozenset[FailureReason] = frozenset({
    FailureReason.TIMEOUT,
    FailureReason.OUT_OF_MEMORY,
    FailureReason.SCHEDULER_FAILURE,
    FailureReason.CHECKPOINT_MISSING,
    # NOT_CONVERGED is retryable: the run can continue sweeping from its last
    # checkpoint until convergence is reached.
    FailureReason.NOT_CONVERGED,
    # NOT_FINISHED is retryable: the run can continue its cooling schedule
    # from its last checkpoint (XTRG's xtrg.ckpt + thermal.ckpt).
    FailureReason.NOT_FINISHED,
})


@dataclass
class AttemptStatus:
    """Status record for a single attempt directory.

    Written to `attempt_NN/status.json` by the algorithm runner and read
    by `collect` and `retry` to decide whether to create a new attempt.

    Parameters
    ----------
    state:
        Current execution state.
    reason:
        Human-readable reason code; `None` while still running.
    restartable:
        Whether a new attempt with the same config is safe to create.
    started_at:
        ISO-8601 timestamp when the attempt started; `None` if not yet started.
    ended_at:
        ISO-8601 timestamp when the attempt finished; `None` if still running.
    """

    state: RunState
    reason: Optional[FailureReason] = None
    restartable: bool = False
    started_at: Optional[str] = None
    ended_at: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        d = asdict(self)
        # Convert enums to their string values for clean JSON.
        d["state"] = self.state.value
        if self.reason is not None:
            d["reason"] = self.reason.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AttemptStatus:
        """Deserialize from a plain dict loaded from JSON."""
        return cls(
            state=RunState(d["state"]),
            reason=FailureReason(d["reason"]) if d.get("reason") else None,
            restartable=d.get("restartable", False),
            started_at=d.get("started_at"),
            ended_at=d.get("ended_at"),
        )


@dataclass
class MainStatus:
    """Status record for a `main/` unit.

    Written to `main/status.json`; summarizes the overall primary calculation
    rather than any individual attempt.

    Parameters
    ----------
    state:
        Aggregated state of the main calculation.
    current_attempt:
        Name of the most recent attempt (e.g. `"attempt_02"`); `None` if no
        attempt has been created yet.
    reason:
        Reason code from the latest attempt; `None` while pending or running.
    restartable:
        Whether `retry` may create a new attempt.
    nodename:
        Short node name (FQDN with domain suffix stripped) of the node where
        the most recent attempt ran; `None` until the runner starts.
    hostname:
        Full hostname of the node where the most recent attempt ran; `None`
        until the runner starts.
    """

    state: RunState
    current_attempt: Optional[str] = None
    reason: Optional[FailureReason] = None
    restartable: bool = False
    nodename: Optional[str] = None
    hostname: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        d = asdict(self)
        d["state"] = self.state.value
        if self.reason is not None:
            d["reason"] = self.reason.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> MainStatus:
        """Deserialize from a plain dict loaded from JSON."""
        return cls(
            state=RunState(d["state"]),
            current_attempt=d.get("current_attempt"),
            reason=FailureReason(d["reason"]) if d.get("reason") else None,
            restartable=d.get("restartable", False),
            nodename=d.get("nodename"),
            hostname=d.get("hostname"),
        )


def read_status(path: Path) -> Union[AttemptStatus, MainStatus]:
    """Read a `status.json` file and return the appropriate status object.

    The type is inferred from the keys present: `MainStatus` contains
    `current_attempt`; `AttemptStatus` contains `started_at`.

    Parameters
    ----------
    path:
        Path to the `status.json` file.

    Returns
    -------
    AttemptStatus | MainStatus
        Deserialized status object.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    KeyError
        If the JSON is missing the required `state` key.
    """
    with open(path) as f:
        d = json.load(f)
    if "current_attempt" in d:
        return MainStatus.from_dict(d)
    return AttemptStatus.from_dict(d)


def write_status(path: Path, status: Union[AttemptStatus, MainStatus]) -> None:
    """Write a status object to a `status.json` file.

    Creates parent directories if they do not exist. Writes atomically by
    serializing to a string first to avoid partial writes on failure.

    Parameters
    ----------
    path:
        Destination path (typically `main/status.json` or
        `attempt_NN/status.json`).
    status:
        The status object to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status.to_dict(), indent=2) + "\n")
