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


"""Tests for src/intraknot/status.py."""

import json

import pytest

from intraknot.status import (
    AttemptStatus,
    FailureReason,
    MainStatus,
    RETRYABLE_REASONS,
    RunState,
    read_status,
    write_status,
)


class TestRunState:
    def test_values_are_strings(self):
        for state in RunState:
            assert isinstance(state.value, str)

    def test_round_trip(self):
        assert RunState("completed") == RunState.COMPLETED
        assert RunState("failed") == RunState.FAILED


class TestFailureReason:
    def test_values_are_strings(self):
        for reason in FailureReason:
            assert isinstance(reason.value, str)

    def test_not_converged_is_retryable(self):
        assert FailureReason.NOT_CONVERGED in RETRYABLE_REASONS


class TestAttemptStatus:
    def test_default_construction(self):
        s = AttemptStatus(state=RunState.PENDING)
        assert s.state == RunState.PENDING
        assert s.reason is None
        assert s.restartable is False
        assert s.started_at is None
        assert s.ended_at is None

    def test_to_dict_state_is_string(self):
        s = AttemptStatus(state=RunState.RUNNING)
        d = s.to_dict()
        assert d["state"] == "running"

    def test_to_dict_reason_is_string(self):
        s = AttemptStatus(state=RunState.FAILED, reason=FailureReason.TIMEOUT)
        d = s.to_dict()
        assert d["reason"] == "timeout"

    def test_to_dict_reason_none_stays_none(self):
        s = AttemptStatus(state=RunState.PENDING)
        d = s.to_dict()
        assert d["reason"] is None

    def test_from_dict_round_trip(self):
        s = AttemptStatus(
            state=RunState.COMPLETED,
            reason=FailureReason.CONVERGED,
            restartable=False,
            started_at="2026-01-01T00:00:00+00:00",
            ended_at="2026-01-01T01:00:00+00:00",
        )
        s2 = AttemptStatus.from_dict(s.to_dict())
        assert s2.state == s.state
        assert s2.reason == s.reason
        assert s2.restartable == s.restartable
        assert s2.started_at == s.started_at
        assert s2.ended_at == s.ended_at

    def test_from_dict_minimal(self):
        d = {"state": "pending"}
        s = AttemptStatus.from_dict(d)
        assert s.state == RunState.PENDING
        assert s.reason is None


class TestMainStatus:
    def test_default_construction(self):
        s = MainStatus(state=RunState.PENDING)
        assert s.state == RunState.PENDING
        assert s.current_attempt is None
        assert s.reason is None
        assert s.restartable is False
        assert s.hostname is None

    def test_to_dict_state_is_string(self):
        s = MainStatus(state=RunState.RUNNING, current_attempt="attempt_01")
        d = s.to_dict()
        assert d["state"] == "running"
        assert d["current_attempt"] == "attempt_01"

    def test_from_dict_round_trip(self):
        s = MainStatus(
            state=RunState.FAILED,
            current_attempt="attempt_02",
            reason=FailureReason.TIMEOUT,
            restartable=True,
            hostname="node42.cluster",
        )
        s2 = MainStatus.from_dict(s.to_dict())
        assert s2.state == s.state
        assert s2.current_attempt == s.current_attempt
        assert s2.reason == s.reason
        assert s2.restartable == s.restartable
        assert s2.hostname == "node42.cluster"


class TestReadWriteStatus:
    def test_write_and_read_attempt_status(self, tmp_path):
        s = AttemptStatus(
            state=RunState.COMPLETED,
            reason=FailureReason.CONVERGED,
            restartable=False,
            started_at="2026-01-01T00:00:00+00:00",
        )
        path = tmp_path / "status.json"
        write_status(path, s)
        assert path.exists()
        s2 = read_status(path)
        assert isinstance(s2, AttemptStatus)
        assert s2.state == RunState.COMPLETED
        assert s2.reason == FailureReason.CONVERGED

    def test_write_and_read_main_status(self, tmp_path):
        s = MainStatus(
            state=RunState.RUNNING,
            current_attempt="attempt_01",
            restartable=False,
        )
        path = tmp_path / "status.json"
        write_status(path, s)
        s2 = read_status(path)
        assert isinstance(s2, MainStatus)
        assert s2.state == RunState.RUNNING
        assert s2.current_attempt == "attempt_01"

    def test_write_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "status.json"
        write_status(path, AttemptStatus(state=RunState.PENDING))
        assert path.exists()

    def test_written_json_is_valid(self, tmp_path):
        s = MainStatus(state=RunState.PENDING)
        path = tmp_path / "status.json"
        write_status(path, s)
        data = json.loads(path.read_text())
        assert "state" in data
