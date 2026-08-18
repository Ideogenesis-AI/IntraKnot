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


"""Tests for src/intraknot/algorithm/run_xtrg.py."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _write_config_toml(run_dir: Path, extra: str = "", engine: str = "xtrg") -> None:
    """Write a minimal `config.toml` into `run_dir`.

    Parameters
    ----------
    run_dir:
        Run directory that will receive `config.toml`.
    extra:
        Additional TOML text appended verbatim (e.g. a `[plugin]` section).
    engine:
        Value of `[algorithm] engine`.
    """
    (run_dir / "config.toml").write_text(
        '[geometry]\nlattice = "chain"\nlx = 8\n'
        '[model]\ncategory = "bosonic"\nlabel = "Heisenberg"\n'
        'symmetry = "U1"\nspin = 0.5\nJ = 1.0\n'
        f'[algorithm]\nengine = "{engine}"\nn_steps = 2\ntau_0 = 0.001\n'
        + extra
    )


def _mock_opts(**overrides) -> MagicMock:
    """Return a MagicMock that behaves like `xtrg.Options`."""
    opts = MagicMock()
    opts.tau_0 = 0.001
    opts.n_steps = 2
    opts.taylor_order = 10
    opts.max_bond = 32
    opts.trunc_thresh = 1e-15
    opts.n_sweeps = 4
    opts.env_window = 2
    opts.expand_k = 4
    opts.expand_alpha = None
    for key, value in overrides.items():
        setattr(opts, key, value)
    return opts


def _mock_summary(*, converged: bool = True, n_steps: int = 2) -> MagicMock:
    """Return a MagicMock that behaves like an `xtrg.Summary`.

    Parameters
    ----------
    converged:
        Value for `summary.finished`.
    n_steps:
        Number of squaring steps; thermodynamic series have length `n_steps + 1`.
    """
    s = MagicMock()
    s.finished = converged
    s.n_steps = n_steps
    s.betas = [0.001 * 2 ** i for i in range(n_steps + 1)]
    s.log_z = [1.0 - 0.1 * i for i in range(n_steps + 1)]
    s.free_energies = [-1.0 - 0.1 * i for i in range(n_steps + 1)]
    s.energies = [-0.5 - 0.1 * i for i in range(n_steps + 1)]
    s.specific_heats = [0.1 * (i + 1) for i in range(n_steps + 1)]
    s.entropies = [0.4 + 0.1 * i for i in range(n_steps + 1)]
    s.discarded_weights = [0.01 / (i + 1) for i in range(n_steps)]
    return s


def _mock_artifact(*, n_steps: int = 2) -> MagicMock:
    """Return a MagicMock that behaves like an `xtrg.Artifact`."""
    a = MagicMock()
    a.step = n_steps
    a.beta = 0.001 * 2 ** n_steps
    a.rho.bond_dims = [2, 4, 4, 2]
    return a


def _run(run_dir: Path, *, converged: bool = True) -> MagicMock:
    """Call `run_xtrg.run()` with all external (Alice) calls mocked.

    Returns the mock standing in for `build_interaction` so callers can
    inspect the argument it received.

    Parameters
    ----------
    run_dir:
        Run directory containing a pre-written `config.toml`.
    converged:
        Whether the mock XTRG summary reports convergence. If `False`, `run()`
        will call `sys.exit(1)` at the end — the caller must handle that.

    Returns
    -------
    MagicMock
        The mock standing in for `build_interaction`.
    """
    from intraknot.algorithm import run_xtrg

    mock_geo = MagicMock()
    mock_geo.L = 8

    with (
        patch.object(run_xtrg, "alice") as mock_alice,
        patch.object(run_xtrg, "logging") as mock_logging,
        patch.object(run_xtrg, "build_interaction") as mock_bi,
        patch.object(run_xtrg, "build_hamiltonian", return_value=MagicMock()),
        patch.object(run_xtrg, "thermal_mpo", return_value=MagicMock()),
        patch.object(run_xtrg, "xtrg") as mock_xtrg,
        patch.object(run_xtrg, "write_status"),
    ):
        mock_alice.__version__ = "0.0.0"
        mock_logging.INFO = 20
        mock_bi.return_value = ([], MagicMock(), mock_geo)
        mock_xtrg.Options.from_toml.return_value = _mock_opts()
        mock_xtrg.run.return_value = (
            _mock_summary(converged=converged),
            _mock_artifact(),
        )

        if converged:
            run_xtrg.run(run_dir)
        else:
            with pytest.raises(SystemExit):
                run_xtrg.run(run_dir)

    return mock_bi


# ---------------------------------------------------------------------------
# _resolve_attempt_dir
# ---------------------------------------------------------------------------

class TestAttemptDirectory:
    """Tests for `_resolve_attempt_dir`."""

    def test_first_attempt_is_01(self, tmp_path):
        from intraknot.algorithm.run_xtrg import _resolve_attempt_dir
        path, name = _resolve_attempt_dir(tmp_path)
        assert name == "attempt_01"
        assert path == tmp_path / "main" / "attempts" / "attempt_01"

    def test_increments_beyond_existing(self, tmp_path):
        from intraknot.algorithm.run_xtrg import _resolve_attempt_dir
        (tmp_path / "main" / "attempts" / "attempt_01").mkdir(parents=True)
        (tmp_path / "main" / "attempts" / "attempt_02").mkdir()
        _, name = _resolve_attempt_dir(tmp_path)
        assert name == "attempt_03"

    def test_creates_attempts_root_if_absent(self, tmp_path):
        from intraknot.algorithm.run_xtrg import _resolve_attempt_dir
        assert not (tmp_path / "main").exists()
        _resolve_attempt_dir(tmp_path)
        assert (tmp_path / "main" / "attempts").is_dir()

    def test_non_attempt_dirs_ignored(self, tmp_path):
        """Directories not matching `attempt_NN` must not affect the index."""
        from intraknot.algorithm.run_xtrg import _resolve_attempt_dir
        root = tmp_path / "main" / "attempts"
        root.mkdir(parents=True)
        (root / "logs").mkdir()
        (root / "attempt_01").mkdir()
        _, name = _resolve_attempt_dir(tmp_path)
        assert name == "attempt_02"

    def test_non_numeric_attempt_suffix_ignored(self, tmp_path):
        from intraknot.algorithm.run_xtrg import _resolve_attempt_dir
        root = tmp_path / "main" / "attempts"
        root.mkdir(parents=True)
        (root / "attempt_backup").mkdir()
        (root / "attempt_01").mkdir()
        _, name = _resolve_attempt_dir(tmp_path)
        assert name == "attempt_02"


# ---------------------------------------------------------------------------
# _find_latest_progress
# ---------------------------------------------------------------------------

class TestFindLatestProgress:
    """Tests for `_find_latest_progress`."""

    def test_returns_none_when_no_attempts_root(self, tmp_path):
        from intraknot.algorithm.run_xtrg import _find_latest_progress
        assert _find_latest_progress(tmp_path) is None

    def test_returns_none_when_no_ckpt_in_any_attempt(self, tmp_path):
        from intraknot.algorithm.run_xtrg import _find_latest_progress
        (tmp_path / "main" / "attempts" / "attempt_01").mkdir(parents=True)
        assert _find_latest_progress(tmp_path) is None

    def test_returns_ckpt_from_single_attempt(self, tmp_path):
        from intraknot.algorithm.run_xtrg import _find_latest_progress
        a1 = tmp_path / "main" / "attempts" / "attempt_01"
        a1.mkdir(parents=True)
        (a1 / "progress.ckpt").write_bytes(b"")
        assert _find_latest_progress(tmp_path) == a1 / "progress.ckpt"

    def test_prefers_later_attempt_over_earlier(self, tmp_path):
        from intraknot.algorithm.run_xtrg import _find_latest_progress
        root = tmp_path / "main" / "attempts"
        a1 = root / "attempt_01"
        a1.mkdir(parents=True)
        (a1 / "progress.ckpt").write_bytes(b"old")
        a2 = root / "attempt_02"
        a2.mkdir()
        (a2 / "progress.ckpt").write_bytes(b"new")
        assert _find_latest_progress(tmp_path) == a2 / "progress.ckpt"

    def test_skips_attempt_without_ckpt(self, tmp_path):
        """Returns the most recent attempt that *has* a checkpoint."""
        from intraknot.algorithm.run_xtrg import _find_latest_progress
        root = tmp_path / "main" / "attempts"
        a1 = root / "attempt_01"
        a1.mkdir(parents=True)
        (a1 / "progress.ckpt").write_bytes(b"")
        (root / "attempt_02").mkdir()  # no ckpt
        assert _find_latest_progress(tmp_path) == a1 / "progress.ckpt"


# ---------------------------------------------------------------------------
# _validate_config / _validate_options / _validate_summary
# ---------------------------------------------------------------------------

class TestValidateConfig:
    """Tests for `_validate_config`.

    `[algorithm] engine` decides which runner the submit script invokes, so
    this runner must refuse a config that selects a different engine.
    """

    def test_accepts_xtrg_engine(self):
        from intraknot.algorithm.run_xtrg import _validate_config
        _validate_config({"engine": "xtrg"})

    def test_defaults_to_xtrg(self):
        from intraknot.algorithm.run_xtrg import _validate_config
        _validate_config({})

    def test_rejects_other_engine(self):
        from intraknot.algorithm.run_xtrg import _EngineMismatch, _validate_config
        with pytest.raises(_EngineMismatch, match="engine"):
            _validate_config({"engine": "dmrg"})

    def test_run_marks_engine_mismatch_invalid(self, tmp_path):
        """A mis-dispatched run must not be retried with the same config."""
        from intraknot.algorithm import run_xtrg
        from intraknot.status import FailureReason, RunState

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir, engine="dmrg")

        mock_geo = MagicMock()
        mock_geo.L = 8
        calls: list = []

        with (
            patch.object(run_xtrg, "alice") as mock_alice,
            patch.object(run_xtrg, "logging") as ml,
            patch.object(
                run_xtrg, "build_interaction",
                return_value=([], MagicMock(), mock_geo),
            ),
            patch.object(run_xtrg, "build_hamiltonian", return_value=MagicMock()),
            patch.object(run_xtrg, "thermal_mpo", return_value=MagicMock()),
            patch.object(run_xtrg, "xtrg") as mock_xtrg,
            patch.object(
                run_xtrg, "write_status",
                side_effect=lambda p, s: calls.append((p, s)),
            ),
        ):
            mock_alice.__version__ = "0.0.0"
            ml.INFO = 20
            with pytest.raises(SystemExit):
                run_xtrg.run(run_dir)

        mock_xtrg.run.assert_not_called()
        main_calls = [(p, s) for p, s in calls if "attempt" not in str(p)]
        assert main_calls[-1][1].state == RunState.INVALID
        assert main_calls[-1][1].reason == FailureReason.BAD_PARAMETERS
        assert main_calls[-1][1].restartable is False


class TestValidateOptions:
    """Tests for `_validate_options`."""

    def test_accepts_defaults(self):
        from intraknot.algorithm.run_xtrg import _validate_options
        _validate_options(_mock_opts())

    def test_rejects_nonpositive_tau_0(self):
        from intraknot.algorithm.run_xtrg import _validate_options
        with pytest.raises(ValueError, match="tau_0"):
            _validate_options(_mock_opts(tau_0=0.0))

    def test_rejects_n_steps_below_one(self):
        from intraknot.algorithm.run_xtrg import _validate_options
        with pytest.raises(ValueError, match="n_steps"):
            _validate_options(_mock_opts(n_steps=0))

    def test_rejects_nonpositive_max_bond(self):
        from intraknot.algorithm.run_xtrg import _validate_options
        with pytest.raises(ValueError, match="max_bond"):
            _validate_options(_mock_opts(max_bond=0))

    def test_accepts_omitted_max_bond(self):
        from intraknot.algorithm.run_xtrg import _validate_options
        _validate_options(_mock_opts(max_bond=None))


class TestValidateSummary:
    """Tests for `_validate_summary`."""

    def test_accepts_aligned_history(self):
        from intraknot.algorithm.run_xtrg import _validate_summary
        _validate_summary(_mock_summary(), _mock_artifact())

    def test_rejects_short_series(self):
        from intraknot.algorithm.run_xtrg import _validate_summary
        summary = _mock_summary()
        summary.energies = summary.energies[:-1]
        with pytest.raises(RuntimeError, match="energies"):
            _validate_summary(summary, _mock_artifact())

    def test_rejects_nonfinite_observable(self):
        from intraknot.algorithm.run_xtrg import _NonFiniteValue, _validate_summary
        summary = _mock_summary()
        summary.log_z[-1] = math.nan
        with pytest.raises(_NonFiniteValue, match="log_z"):
            _validate_summary(summary, _mock_artifact())

    def test_rejects_mismatched_artifact_step(self):
        from intraknot.algorithm.run_xtrg import _validate_summary
        artifact = _mock_artifact()
        artifact.step = 99
        with pytest.raises(RuntimeError, match="artifact step"):
            _validate_summary(_mock_summary(), artifact)


# ---------------------------------------------------------------------------
# _write_observables
# ---------------------------------------------------------------------------

class TestWriteObservables:
    """Tests for `_write_observables`."""

    def test_creates_info_json(self, tmp_path):
        from intraknot.algorithm import run_xtrg
        from intraknot.algorithm.run_xtrg import _write_observables
        with patch.object(run_xtrg, "alice") as mock_alice:
            mock_alice.__version__ = "0.0.0"
            _write_observables(tmp_path, _mock_summary(), _mock_artifact(), L=8)
        assert (tmp_path / "info.json").exists()

    def test_scalar_fields(self, tmp_path):
        from intraknot.algorithm import run_xtrg
        from intraknot.algorithm.run_xtrg import _write_observables
        summary = _mock_summary()
        artifact = _mock_artifact()
        with patch.object(run_xtrg, "alice") as mock_alice:
            mock_alice.__version__ = "9.9.9"
            _write_observables(tmp_path, summary, artifact, L=8)
        data = json.loads((tmp_path / "info.json").read_text())
        assert data["algorithm"] == "xtrg"
        assert data["alice_version"] == "9.9.9"
        assert data["system_size"] == 8
        assert data["finished"] is True
        assert data["n_steps"] == 2
        assert data["beta"] == pytest.approx(artifact.beta)
        assert data["temperature"] == pytest.approx(1.0 / artifact.beta)
        assert data["log_z"] == pytest.approx(summary.log_z[-1])
        assert data["energy_per_site"] == pytest.approx(summary.energies[-1])
        assert data["max_bond_dim"] == 4
        assert data["bond_dims"] == [2, 4, 4, 2]


# ---------------------------------------------------------------------------
# _write_thermodynamics
# ---------------------------------------------------------------------------

class TestWriteThermodynamics:
    """Tests for `_write_thermodynamics`."""

    def _read_csv(self, path: Path) -> list:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def test_creates_thermodynamics_csv(self, tmp_path):
        from intraknot.algorithm.run_xtrg import _write_thermodynamics
        _write_thermodynamics(tmp_path, _mock_summary())
        assert (tmp_path / "thermodynamics.csv").exists()

    def test_header_columns(self, tmp_path):
        from intraknot.algorithm.run_xtrg import _write_thermodynamics
        _write_thermodynamics(tmp_path, _mock_summary())
        with open(tmp_path / "thermodynamics.csv", newline="") as f:
            header = next(csv.reader(f))
        assert header == [
            "step",
            "beta",
            "temperature",
            "log_z",
            "free_energy_per_site",
            "energy_per_site",
            "specific_heat_per_site",
            "entropy_per_site",
            "discarded_weight",
        ]

    def test_row_count_matches_betas(self, tmp_path):
        from intraknot.algorithm.run_xtrg import _write_thermodynamics
        summary = _mock_summary()
        _write_thermodynamics(tmp_path, summary)
        rows = self._read_csv(tmp_path / "thermodynamics.csv")
        assert len(rows) == len(summary.betas)

    def test_first_row_has_zero_discarded_weight(self, tmp_path):
        from intraknot.algorithm.run_xtrg import _write_thermodynamics
        _write_thermodynamics(tmp_path, _mock_summary())
        rows = self._read_csv(tmp_path / "thermodynamics.csv")
        assert rows[0]["step"] == "0"
        assert float(rows[0]["discarded_weight"]) == pytest.approx(0.0)

    def test_later_rows_use_prior_discarded_weight(self, tmp_path):
        from intraknot.algorithm.run_xtrg import _write_thermodynamics
        summary = _mock_summary()
        _write_thermodynamics(tmp_path, summary)
        rows = self._read_csv(tmp_path / "thermodynamics.csv")
        assert float(rows[1]["discarded_weight"]) == pytest.approx(
            summary.discarded_weights[0]
        )


# ---------------------------------------------------------------------------
# run() — attempt directory naming
# ---------------------------------------------------------------------------

class TestRunAttemptNaming:
    """run() must create a properly named attempt directory each time it runs."""

    def test_creates_attempt_01_on_first_run(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)
        _run(run_dir)
        assert (run_dir / "main" / "attempts" / "attempt_01").is_dir()

    def test_creates_attempt_02_when_attempt_01_already_exists(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "main" / "attempts" / "attempt_01").mkdir(parents=True)
        _write_config_toml(run_dir)
        _run(run_dir)
        assert (run_dir / "main" / "attempts" / "attempt_02").is_dir()


# ---------------------------------------------------------------------------
# run() — status file transitions
# ---------------------------------------------------------------------------

class TestRunStatus:
    """run() must write the correct `RunState` to the main and attempt status files."""

    def _collect_status_calls(self, run_dir: Path, *, converged: bool):
        """Run with mocked write_status and return all (path, status) calls."""
        from intraknot.algorithm import run_xtrg

        mock_geo = MagicMock()
        mock_geo.L = 8
        calls: list = []

        with (
            patch.object(run_xtrg, "alice") as mock_alice,
            patch.object(run_xtrg, "logging") as ml,
            patch.object(
                run_xtrg, "build_interaction",
                return_value=([], MagicMock(), mock_geo),
            ),
            patch.object(run_xtrg, "build_hamiltonian", return_value=MagicMock()),
            patch.object(run_xtrg, "thermal_mpo", return_value=MagicMock()),
            patch.object(run_xtrg, "xtrg") as mock_xtrg,
            patch.object(
                run_xtrg, "write_status",
                side_effect=lambda p, s: calls.append((p, s)),
            ),
        ):
            mock_alice.__version__ = "0.0.0"
            ml.INFO = 20
            mock_xtrg.Options.from_toml.return_value = _mock_opts()
            mock_xtrg.run.return_value = (
                _mock_summary(converged=converged),
                _mock_artifact(),
            )

            if converged:
                run_xtrg.run(run_dir)
            else:
                with pytest.raises(SystemExit):
                    run_xtrg.run(run_dir)

        return calls

    def test_converged_sets_completed_on_main_status(self, tmp_path):
        from intraknot.status import RunState

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        calls = self._collect_status_calls(run_dir, converged=True)

        main_calls = [(p, s) for p, s in calls if "attempt" not in str(p)]
        assert main_calls[-1][1].state == RunState.COMPLETED

    def test_finished_summary_sets_finished_reason(self, tmp_path):
        """XTRG's success reason is `finished`, not `converged` — XTRG has no
        numerical convergence criterion, only a fixed cooling schedule."""
        from intraknot.status import FailureReason

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        calls = self._collect_status_calls(run_dir, converged=True)

        main_calls = [(p, s) for p, s in calls if "attempt" not in str(p)]
        assert main_calls[-1][1].reason == FailureReason.FINISHED

    def test_not_converged_sets_failed_on_main_status(self, tmp_path):
        from intraknot.status import RunState

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        calls = self._collect_status_calls(run_dir, converged=False)

        main_calls = [(p, s) for p, s in calls if "attempt" not in str(p)]
        assert main_calls[-1][1].state == RunState.FAILED

    def test_unfinished_summary_sets_not_finished_reason(self, tmp_path):
        """The (currently dead) unfinished branch reports `not_finished`,
        not `not_converged`, and remains retryable."""
        from intraknot.status import FailureReason

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        calls = self._collect_status_calls(run_dir, converged=False)

        main_calls = [(p, s) for p, s in calls if "attempt" not in str(p)]
        assert main_calls[-1][1].reason == FailureReason.NOT_FINISHED
        assert main_calls[-1][1].restartable is True

    def test_attempt_status_starts_as_running(self, tmp_path):
        from intraknot.status import RunState

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        calls = self._collect_status_calls(run_dir, converged=True)

        attempt_calls = [(p, s) for p, s in calls if "attempt" in str(p)]
        assert attempt_calls[0][1].state == RunState.RUNNING

    def test_main_status_starts_as_running(self, tmp_path):
        from intraknot.status import RunState

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        calls = self._collect_status_calls(run_dir, converged=True)

        main_calls = [(p, s) for p, s in calls if "attempt" not in str(p)]
        assert main_calls[0][1].state == RunState.RUNNING


# ---------------------------------------------------------------------------
# run() — output files
# ---------------------------------------------------------------------------

class TestRunOutputFiles:
    """run() must write info.json and thermodynamics.csv to the attempt directory."""

    def test_info_json_written(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)
        _run(run_dir)
        assert (run_dir / "main" / "attempts" / "attempt_01" / "info.json").exists()

    def test_thermodynamics_csv_written(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)
        _run(run_dir)
        csv_path = run_dir / "main" / "attempts" / "attempt_01" / "thermodynamics.csv"
        assert csv_path.exists()


# ---------------------------------------------------------------------------
# run() — resumption from a prior attempt's progress.ckpt
# ---------------------------------------------------------------------------

class TestRunResume:
    """run() must resume from the latest progress.ckpt when one exists."""

    def _run_with_mocks(self, run_dir: Path, *, loaded_artifact=None):
        """Call `run_xtrg.run()` with Alice mocked; return the mocks used.

        Parameters
        ----------
        run_dir:
            Run directory containing a pre-written `config.toml`.
        loaded_artifact:
            Value returned by `xtrg.Artifact.load`. Only consulted by the
            runner when a prior `progress.ckpt` is found.

        Returns
        -------
        dict
            Mocks for `xtrg`, `thermal_mpo`, keyed by name.
        """
        from intraknot.algorithm import run_xtrg

        mock_geo = MagicMock()
        mock_geo.L = 8

        with (
            patch.object(run_xtrg, "alice") as mock_alice,
            patch.object(run_xtrg, "logging") as mock_logging,
            patch.object(
                run_xtrg, "build_interaction",
                return_value=([], MagicMock(), mock_geo),
            ),
            patch.object(run_xtrg, "build_hamiltonian", return_value=MagicMock()),
            patch.object(run_xtrg, "thermal_mpo", return_value=MagicMock()) as mock_thermal_mpo,
            patch.object(run_xtrg, "xtrg") as mock_xtrg,
            patch.object(run_xtrg, "write_status"),
        ):
            mock_alice.__version__ = "0.0.0"
            mock_logging.INFO = 20
            mock_xtrg.Options.from_toml.return_value = _mock_opts()
            mock_xtrg.Artifact.load.return_value = loaded_artifact
            mock_xtrg.run.return_value = (_mock_summary(), _mock_artifact())

            run_xtrg.run(run_dir)

        return {"xtrg": mock_xtrg, "thermal_mpo": mock_thermal_mpo}

    def test_no_prior_progress_builds_fresh_state(self, tmp_path):
        """With no prior attempt, the runner builds rho(tau_0) via thermal_mpo."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        mocks = self._run_with_mocks(run_dir)

        mocks["thermal_mpo"].assert_called_once()
        mocks["xtrg"].Artifact.load.assert_not_called()
        # The fresh Artifact is built from thermal_mpo's return value at step 0.
        state = mocks["xtrg"].run.call_args[0][0]
        assert state is mocks["xtrg"].Artifact.return_value

    def test_prior_progress_at_step_zero_resumes_without_thermal_ckpt(self, tmp_path):
        """Resuming from step 0 does not require a thermal.ckpt to exist."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)
        prior_attempt = run_dir / "main" / "attempts" / "attempt_01"
        prior_attempt.mkdir(parents=True)
        (prior_attempt / "progress.ckpt").write_bytes(b"artifact-step-0")

        loaded = MagicMock(step=0, beta=0.001)
        mocks = self._run_with_mocks(run_dir, loaded_artifact=loaded)

        mocks["thermal_mpo"].assert_not_called()
        mocks["xtrg"].Artifact.load.assert_called_once_with(
            prior_attempt / "progress.ckpt"
        )
        state = mocks["xtrg"].run.call_args[0][0]
        assert state is loaded
        # No thermal.ckpt needed or copied at step 0.
        new_attempt = run_dir / "main" / "attempts" / "attempt_02"
        assert not (new_attempt / "thermal.ckpt").exists()

    def test_prior_progress_past_step_zero_copies_thermal_ckpt(self, tmp_path):
        """Resuming past step 0 copies the matching thermal.ckpt forward."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)
        prior_attempt = run_dir / "main" / "attempts" / "attempt_01"
        prior_attempt.mkdir(parents=True)
        (prior_attempt / "progress.ckpt").write_bytes(b"artifact-step-1")
        (prior_attempt / "thermal.ckpt").write_bytes(b"thermal-history")

        loaded = MagicMock(step=1, beta=0.002)
        self._run_with_mocks(run_dir, loaded_artifact=loaded)

        new_attempt = run_dir / "main" / "attempts" / "attempt_02"
        copied = new_attempt / "thermal.ckpt"
        assert copied.exists()
        assert copied.read_bytes() == b"thermal-history"

    def test_prior_progress_past_step_zero_missing_thermal_ckpt_is_invalid(self, tmp_path):
        """A step>0 progress.ckpt without a matching thermal.ckpt cannot resume."""
        from intraknot.status import FailureReason, RunState

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)
        prior_attempt = run_dir / "main" / "attempts" / "attempt_01"
        prior_attempt.mkdir(parents=True)
        (prior_attempt / "progress.ckpt").write_bytes(b"artifact-step-1")
        # thermal.ckpt deliberately absent.

        from intraknot.algorithm import run_xtrg

        mock_geo = MagicMock()
        mock_geo.L = 8
        calls: list = []

        with (
            patch.object(run_xtrg, "alice") as mock_alice,
            patch.object(run_xtrg, "logging") as ml,
            patch.object(
                run_xtrg, "build_interaction",
                return_value=([], MagicMock(), mock_geo),
            ),
            patch.object(run_xtrg, "build_hamiltonian", return_value=MagicMock()),
            patch.object(run_xtrg, "thermal_mpo", return_value=MagicMock()),
            patch.object(run_xtrg, "xtrg") as mock_xtrg,
            patch.object(
                run_xtrg, "write_status",
                side_effect=lambda p, s: calls.append((p, s)),
            ),
        ):
            mock_alice.__version__ = "0.0.0"
            ml.INFO = 20
            mock_xtrg.Options.from_toml.return_value = _mock_opts()
            mock_xtrg.Artifact.load.return_value = MagicMock(step=1, beta=0.002)
            with pytest.raises(SystemExit):
                run_xtrg.run(run_dir)

        mock_xtrg.run.assert_not_called()
        main_calls = [(p, s) for p, s in calls if "attempt" not in str(p)]
        assert main_calls[-1][1].state == RunState.INVALID
        assert main_calls[-1][1].reason == FailureReason.BAD_PARAMETERS
        assert main_calls[-1][1].restartable is False

    def test_current_empty_attempt_dir_is_excluded_from_search(self, tmp_path):
        """The freshly created (empty) attempt dir must not be mistaken for a
        prior attempt to resume from."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)
        # No prior attempts exist; attempt_01 is created fresh by run() itself.

        mocks = self._run_with_mocks(run_dir)

        mocks["thermal_mpo"].assert_called_once()
        mocks["xtrg"].Artifact.load.assert_not_called()


# ---------------------------------------------------------------------------
# run() — exception mapping
# ---------------------------------------------------------------------------

class TestRunExceptionMapping:
    """run() maps Alice/config failures onto IntraKnot status reasons."""

    def _run_with_side_effect(self, run_dir: Path, exc: BaseException):
        from intraknot.algorithm import run_xtrg

        mock_geo = MagicMock()
        mock_geo.L = 8
        calls: list = []

        with (
            patch.object(run_xtrg, "alice") as mock_alice,
            patch.object(run_xtrg, "logging") as ml,
            patch.object(
                run_xtrg, "build_interaction",
                return_value=([], MagicMock(), mock_geo),
            ),
            patch.object(run_xtrg, "build_hamiltonian", return_value=MagicMock()),
            patch.object(run_xtrg, "thermal_mpo", return_value=MagicMock()),
            patch.object(run_xtrg, "xtrg") as mock_xtrg,
            patch.object(
                run_xtrg, "write_status",
                side_effect=lambda p, s: calls.append((p, s)),
            ),
        ):
            mock_alice.__version__ = "0.0.0"
            ml.INFO = 20
            mock_xtrg.Options.from_toml.return_value = _mock_opts()
            mock_xtrg.run.side_effect = exc
            with pytest.raises(SystemExit):
                run_xtrg.run(run_dir)

        return calls

    def test_value_error_sets_invalid(self, tmp_path):
        from intraknot.status import FailureReason, RunState

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)
        calls = self._run_with_side_effect(run_dir, ValueError("bad tau_0"))

        main_calls = [(p, s) for p, s in calls if "attempt" not in str(p)]
        assert main_calls[-1][1].state == RunState.INVALID
        assert main_calls[-1][1].reason == FailureReason.BAD_PARAMETERS
        assert main_calls[-1][1].restartable is False

    def test_runtime_error_sets_linear_algebra(self, tmp_path):
        from intraknot.status import FailureReason, RunState

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)
        calls = self._run_with_side_effect(run_dir, RuntimeError("unphysical trace"))

        main_calls = [(p, s) for p, s in calls if "attempt" not in str(p)]
        assert main_calls[-1][1].state == RunState.FAILED
        assert main_calls[-1][1].reason == FailureReason.LINEAR_ALGEBRA_ERROR
        assert main_calls[-1][1].restartable is False

    def test_memory_error_is_restartable(self, tmp_path):
        from intraknot.status import FailureReason, RunState

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)
        calls = self._run_with_side_effect(run_dir, MemoryError())

        main_calls = [(p, s) for p, s in calls if "attempt" not in str(p)]
        assert main_calls[-1][1].state == RunState.FAILED
        assert main_calls[-1][1].reason == FailureReason.OUT_OF_MEMORY
        assert main_calls[-1][1].restartable is True


# ---------------------------------------------------------------------------
# Plugin forwarding
# ---------------------------------------------------------------------------

class TestPluginForwarding:
    """The `[plugin]` table from `config.toml` must reach `build_interaction`."""

    def test_plugin_section_forwarded(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(
            run_dir,
            extra='[plugin]\nmodel = "plugins/my_model.py:build_model"\n',
        )

        mock_bi = _run(run_dir)

        cfg_passed = mock_bi.call_args[0][0]
        assert "plugin" in cfg_passed
        assert cfg_passed["plugin"].get("model") == "plugins/my_model.py:build_model"

    def test_absent_plugin_section_gives_empty_dict(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        mock_bi = _run(run_dir)

        cfg_passed = mock_bi.call_args[0][0]
        assert "plugin" in cfg_passed
        assert cfg_passed["plugin"] == {}
