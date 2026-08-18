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


"""Tests for src/intraknot/algorithm/run_dmrg.py."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

#: Minimal model config shared by _init_mps tests.
_CFG_MODEL = {
    "geometry": {"lattice": "chain", "lx": 8},
    "model": {
        "category": "bosonic",
        "label": "Heisenberg",
        "symmetry": "U1",
        "spin": 0.5,
        "J": 1.0,
    },
}


def _write_config_toml(run_dir: Path, extra: str = "") -> None:
    """Write a minimal `config.toml` into `run_dir`.

    Parameters
    ----------
    run_dir:
        Run directory that will receive `config.toml`.
    extra:
        Additional TOML text appended verbatim (e.g. a `[plugin]` section).
    """
    (run_dir / "config.toml").write_text(
        '[geometry]\nlattice = "chain"\nlx = 8\n'
        '[model]\ncategory = "bosonic"\nlabel = "Heisenberg"\n'
        'symmetry = "U1"\nspin = 0.5\nJ = 1.0\n'
        '[algorithm]\ninit = "product"\nn_sweeps = 2\nmax_bond = 4\n'
        + extra
    )


def _mock_summary(*, converged: bool = True) -> MagicMock:
    """Return a MagicMock that behaves like a `dmrg.Summary`.

    Parameters
    ----------
    converged:
        Value for `summary.converged`.
    """
    s = MagicMock()
    s.converged = converged
    s.energy = -3.5
    s.n_sweeps = 2
    s.bond_dims = [2, 4, 4, 2]
    s.energies = [-3.0, -3.5]
    s.discarded_weights = [0.01, 0.001]
    return s


def _run(run_dir: Path, *, converged: bool = True) -> MagicMock:
    """Call `run_dmrg.run()` with all external (Alice/Nicole) calls mocked.

    All alice, nicole, and logging side-effects are patched so the test does
    not require a live installation. Returns the mock standing in for
    `build_interaction` so callers can inspect the argument it received.

    Parameters
    ----------
    run_dir:
        Run directory containing a pre-written `config.toml`.
    converged:
        Whether the mock DMRG summary reports convergence. If `False`, `run()`
        will call `sys.exit(1)` at the end — the caller must handle that.

    Returns
    -------
    MagicMock
        The mock standing in for `build_interaction`.
    """
    from intraknot.algorithm import run_dmrg

    mock_geo = MagicMock()
    mock_geo.L = 8

    with (
        patch.object(run_dmrg, "alice"),
        # Suppress FileHandler creation and root-logger mutation; keep
        # logging.INFO as a real int so logger.setLevel() on the module-level
        # real logger doesn't receive a MagicMock.
        patch.object(run_dmrg, "logging") as mock_logging,
        patch.object(run_dmrg, "build_interaction") as mock_bi,
        patch.object(run_dmrg, "build_hamiltonian", return_value=MagicMock()),
        patch.object(run_dmrg, "load_space", return_value=(MagicMock(), {})),
        patch.object(run_dmrg, "init_mps", return_value=MagicMock()),
        patch.object(run_dmrg, "dmrg") as mock_dmrg_mod,
        patch.object(run_dmrg, "write_status"),
    ):
        mock_logging.INFO = 20
        mock_bi.return_value = ([], MagicMock(), mock_geo)
        mock_opts = MagicMock()
        mock_opts.n_sweeps = 2
        mock_dmrg_mod.Options.from_toml.return_value = mock_opts
        mock_dmrg_mod.run.return_value = _mock_summary(converged=converged)

        if converged:
            run_dmrg.run(run_dir)
        else:
            with pytest.raises(SystemExit):
                run_dmrg.run(run_dir)

    return mock_bi


# ---------------------------------------------------------------------------
# _resolve_attempt_dir
# ---------------------------------------------------------------------------

class TestAttemptDirectory:
    """Tests for `_resolve_attempt_dir`."""

    def test_first_attempt_is_01(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _resolve_attempt_dir
        path, name = _resolve_attempt_dir(tmp_path)
        assert name == "attempt_01"
        assert path == tmp_path / "main" / "attempts" / "attempt_01"

    def test_increments_beyond_existing(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _resolve_attempt_dir
        (tmp_path / "main" / "attempts" / "attempt_01").mkdir(parents=True)
        (tmp_path / "main" / "attempts" / "attempt_02").mkdir()
        _, name = _resolve_attempt_dir(tmp_path)
        assert name == "attempt_03"

    def test_creates_attempts_root_if_absent(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _resolve_attempt_dir
        assert not (tmp_path / "main").exists()
        _resolve_attempt_dir(tmp_path)
        assert (tmp_path / "main" / "attempts").is_dir()

    def test_non_attempt_dirs_ignored(self, tmp_path):
        """Directories not matching `attempt_NN` must not affect the index."""
        from intraknot.algorithm.run_dmrg import _resolve_attempt_dir
        root = tmp_path / "main" / "attempts"
        root.mkdir(parents=True)
        (root / "logs").mkdir()
        (root / "attempt_01").mkdir()
        _, name = _resolve_attempt_dir(tmp_path)
        assert name == "attempt_02"


# ---------------------------------------------------------------------------
# _find_latest_checkpoint
# ---------------------------------------------------------------------------

class TestFindLatestCheckpoint:
    """Tests for `_find_latest_checkpoint`."""

    def test_returns_none_when_no_attempts_root(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _find_latest_checkpoint
        assert _find_latest_checkpoint(tmp_path) is None

    def test_returns_none_when_no_ckpt_in_any_attempt(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _find_latest_checkpoint
        (tmp_path / "main" / "attempts" / "attempt_01").mkdir(parents=True)
        assert _find_latest_checkpoint(tmp_path) is None

    def test_returns_ckpt_from_single_attempt(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _find_latest_checkpoint
        a1 = tmp_path / "main" / "attempts" / "attempt_01"
        a1.mkdir(parents=True)
        (a1 / "dmrg.ckpt").write_bytes(b"")
        assert _find_latest_checkpoint(tmp_path) == a1 / "dmrg.ckpt"

    def test_prefers_later_attempt_over_earlier(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _find_latest_checkpoint
        root = tmp_path / "main" / "attempts"
        a1 = root / "attempt_01"
        a1.mkdir(parents=True)
        (a1 / "dmrg.ckpt").write_bytes(b"old")
        a2 = root / "attempt_02"
        a2.mkdir()
        (a2 / "dmrg.ckpt").write_bytes(b"new")
        assert _find_latest_checkpoint(tmp_path) == a2 / "dmrg.ckpt"

    def test_skips_attempt_without_ckpt(self, tmp_path):
        """Returns the most recent attempt that *has* a checkpoint."""
        from intraknot.algorithm.run_dmrg import _find_latest_checkpoint
        root = tmp_path / "main" / "attempts"
        a1 = root / "attempt_01"
        a1.mkdir(parents=True)
        (a1 / "dmrg.ckpt").write_bytes(b"")
        (root / "attempt_02").mkdir()  # no ckpt
        assert _find_latest_checkpoint(tmp_path) == a1 / "dmrg.ckpt"


# ---------------------------------------------------------------------------
# _validate_config
# ---------------------------------------------------------------------------

class TestValidateConfig:
    """Tests for `_validate_config`.

    `[algorithm] engine` decides which runner the submit script invokes, so
    this runner must refuse a config that selects a different engine.
    """

    def test_accepts_dmrg_engine(self):
        from intraknot.algorithm.run_dmrg import _validate_config
        _validate_config({"engine": "dmrg"})

    def test_defaults_to_dmrg(self):
        from intraknot.algorithm.run_dmrg import _validate_config
        _validate_config({})

    def test_rejects_other_engine(self):
        from intraknot.algorithm.run_dmrg import _EngineMismatch, _validate_config
        with pytest.raises(_EngineMismatch, match="engine"):
            _validate_config({"engine": "xtrg"})

    def test_run_marks_engine_mismatch_invalid(self, tmp_path):
        """A mis-dispatched run must not be retried with the same config."""
        from intraknot.algorithm import run_dmrg
        from intraknot.status import FailureReason, RunState

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir, extra='engine = "xtrg"\n')

        mock_geo = MagicMock()
        mock_geo.L = 8
        calls: list = []

        with (
            patch.object(run_dmrg, "alice"),
            patch.object(run_dmrg, "logging") as ml,
            patch.object(run_dmrg, "build_interaction",
                         return_value=([], MagicMock(), mock_geo)),
            patch.object(run_dmrg, "build_hamiltonian", return_value=MagicMock()),
            patch.object(run_dmrg, "dmrg") as mock_dmrg_mod,
            patch.object(run_dmrg, "write_status",
                         side_effect=lambda p, s: calls.append((p, s))),
        ):
            ml.INFO = 20
            with pytest.raises(SystemExit):
                run_dmrg.run(run_dir)

        mock_dmrg_mod.run.assert_not_called()
        main_calls = [(p, s) for p, s in calls if "attempt" not in str(p)]
        assert main_calls[-1][1].state == RunState.INVALID
        assert main_calls[-1][1].reason == FailureReason.BAD_PARAMETERS
        assert main_calls[-1][1].restartable is False


# ---------------------------------------------------------------------------
# _write_observables
# ---------------------------------------------------------------------------

class TestWriteObservables:
    """Tests for `_write_observables`."""

    def _summary(self) -> MagicMock:
        s = MagicMock()
        s.energy = -7.0
        s.converged = True
        s.n_sweeps = 4
        s.bond_dims = [2, 4, 8, 4, 2]
        return s

    def test_creates_info_json(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _write_observables
        _write_observables(tmp_path, self._summary(), L=8)
        assert (tmp_path / "info.json").exists()

    def test_energy_fields(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _write_observables
        _write_observables(tmp_path, self._summary(), L=8)
        data = json.loads((tmp_path / "info.json").read_text())
        assert data["energy"] == pytest.approx(-7.0)
        assert data["energy_per_site"] == pytest.approx(-7.0 / 8)

    def test_scalar_fields(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _write_observables
        _write_observables(tmp_path, self._summary(), L=8)
        data = json.loads((tmp_path / "info.json").read_text())
        assert data["converged"] is True
        assert data["n_sweeps"] == 4
        assert data["max_bond_dim"] == 8
        assert data["bond_dims"] == [2, 4, 8, 4, 2]

    def test_nan_energy_per_site_for_zero_length(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _write_observables
        _write_observables(tmp_path, self._summary(), L=0)
        data = json.loads((tmp_path / "info.json").read_text())
        assert math.isnan(data["energy_per_site"])


# ---------------------------------------------------------------------------
# _write_convergence
# ---------------------------------------------------------------------------

class TestWriteConvergence:
    """Tests for `_write_convergence`."""

    def _summary(self, energies=None, dw=None, converged=True) -> MagicMock:
        s = MagicMock()
        s.energies = energies or [-3.0, -3.4, -3.5]
        s.discarded_weights = dw or [0.01, 0.005, 0.001]
        s.converged = converged
        return s

    def _read_csv(self, path: Path) -> list:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def test_creates_conv_csv(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _write_convergence
        _write_convergence(tmp_path, self._summary())
        assert (tmp_path / "conv.csv").exists()

    def test_header_columns(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _write_convergence
        _write_convergence(tmp_path, self._summary())
        with open(tmp_path / "conv.csv", newline="") as f:
            header = next(csv.reader(f))
        assert header == ["sweep", "energy", "delta_energy", "discarded_weight", "converged"]

    def test_row_count_matches_sweeps(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _write_convergence
        _write_convergence(tmp_path, self._summary())
        rows = self._read_csv(tmp_path / "conv.csv")
        assert len(rows) == 3

    def test_first_row_has_nan_delta(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _write_convergence
        _write_convergence(tmp_path, self._summary())
        rows = self._read_csv(tmp_path / "conv.csv")
        assert rows[0]["sweep"] == "1"
        assert rows[0]["delta_energy"] == "nan"

    def test_delta_energy_computed_correctly(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _write_convergence
        _write_convergence(tmp_path, self._summary(energies=[-3.0, -3.5]))
        rows = self._read_csv(tmp_path / "conv.csv")
        assert float(rows[1]["delta_energy"]) == pytest.approx(0.5)

    def test_converged_flag_set_only_on_last_row(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _write_convergence
        _write_convergence(tmp_path, self._summary())
        rows = self._read_csv(tmp_path / "conv.csv")
        assert rows[0]["converged"] == "False"
        assert rows[1]["converged"] == "False"
        assert rows[2]["converged"] == "True"

    def test_converged_false_on_all_rows_when_not_converged(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _write_convergence
        _write_convergence(tmp_path, self._summary(converged=False))
        rows = self._read_csv(tmp_path / "conv.csv")
        assert all(r["converged"] == "False" for r in rows)


# ---------------------------------------------------------------------------
# _init_mps
# ---------------------------------------------------------------------------

class TestInitMps:
    """Tests for `_init_mps`, covering all four initialisation strategies."""

    def test_product_calls_init_mps_with_bond_dim_1(self, tmp_path):
        from intraknot.algorithm import run_dmrg
        from intraknot.algorithm.run_dmrg import _init_mps

        with (
            patch.object(run_dmrg, "load_space", return_value=(MagicMock(), {})),
            patch.object(run_dmrg, "init_mps") as mock_im,
        ):
            mock_im.return_value = MagicMock()
            _init_mps(_CFG_MODEL, {"init": "product", "max_bond": 32, "seed": 0}, 8, None, tmp_path)

        assert mock_im.call_args.kwargs["bond_dim"] == 1

    def test_random_calls_init_mps_with_max_bond(self, tmp_path):
        from intraknot.algorithm import run_dmrg
        from intraknot.algorithm.run_dmrg import _init_mps

        with (
            patch.object(run_dmrg, "load_space", return_value=(MagicMock(), {})),
            patch.object(run_dmrg, "init_mps") as mock_im,
        ):
            mock_im.return_value = MagicMock()
            _init_mps(_CFG_MODEL, {"init": "random", "max_bond": 64, "seed": 0}, 8, None, tmp_path)

        assert mock_im.call_args.kwargs["bond_dim"] == 64

    def test_product_and_random_return_zero_sweeps_done(self, tmp_path):
        from intraknot.algorithm import run_dmrg
        from intraknot.algorithm.run_dmrg import _init_mps

        for strategy in ("product", "random"):
            with (
                patch.object(run_dmrg, "load_space", return_value=(MagicMock(), {})),
                patch.object(run_dmrg, "init_mps", return_value=MagicMock()),
            ):
                _, sweeps_done = _init_mps(
                    _CFG_MODEL, {"init": strategy, "max_bond": 4, "seed": 0}, 8, None, tmp_path
                )
            assert sweeps_done == 0, f"strategy={strategy!r} should return sweeps_done=0"

    def test_target_qn_list_converted_to_tuple(self, tmp_path):
        from intraknot.algorithm import run_dmrg
        from intraknot.algorithm.run_dmrg import _init_mps

        with (
            patch.object(run_dmrg, "load_space", return_value=(MagicMock(), {})),
            patch.object(run_dmrg, "init_mps") as mock_im,
        ):
            mock_im.return_value = MagicMock()
            _init_mps(
                _CFG_MODEL,
                {"init": "product", "max_bond": 4, "seed": 0, "target_qn": [1, 0]},
                8, None, tmp_path,
            )

        assert mock_im.call_args.kwargs["target_qn"] == (1, 0)

    def test_ckpt_raises_if_default_file_missing(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _init_mps

        with pytest.raises(FileNotFoundError, match="initial.ckpt"):
            _init_mps(_CFG_MODEL, {"init": "ckpt"}, 8, None, tmp_path)

    def test_ckpt_raises_if_explicit_path_missing(self, tmp_path):
        from intraknot.algorithm.run_dmrg import _init_mps

        with pytest.raises(FileNotFoundError):
            _init_mps(
                _CFG_MODEL,
                {"init": "ckpt", "init_ckpt": str(tmp_path / "nonexistent.ckpt")},
                8, None, tmp_path,
            )

    def test_ckpt_loads_state_and_returns_zero_sweeps(self, tmp_path):
        from intraknot.algorithm import run_dmrg
        from intraknot.algorithm.run_dmrg import _init_mps

        ckpt = tmp_path / "initial.ckpt"
        ckpt.write_bytes(b"")

        mock_source = MagicMock()
        mock_source.bond_dims = [2, 2]

        with patch.object(run_dmrg, "dmrg") as mock_dmrg:
            mock_dmrg.Summary.load.return_value = mock_source
            mps, sweeps_done = _init_mps(_CFG_MODEL, {"init": "ckpt"}, 8, None, tmp_path)

        assert mps is mock_source.state
        assert sweeps_done == 0
        mock_source.state.canonical.assert_called_once_with(0)

    def test_ckpt_uses_explicit_init_ckpt_path(self, tmp_path):
        from intraknot.algorithm import run_dmrg
        from intraknot.algorithm.run_dmrg import _init_mps

        explicit = tmp_path / "some" / "state.ckpt"
        explicit.parent.mkdir()
        explicit.write_bytes(b"")

        mock_source = MagicMock()
        mock_source.bond_dims = []

        with patch.object(run_dmrg, "dmrg") as mock_dmrg:
            mock_dmrg.Summary.load.return_value = mock_source
            _init_mps(
                _CFG_MODEL,
                {"init": "ckpt", "init_ckpt": str(explicit)},
                8, None, tmp_path,
            )
            mock_dmrg.Summary.load.assert_called_once_with(explicit)

    def test_resume_returns_state_and_sweeps_done(self, tmp_path):
        from intraknot.algorithm import run_dmrg
        from intraknot.algorithm.run_dmrg import _init_mps

        ckpt = tmp_path / "dmrg.ckpt"
        ckpt.write_bytes(b"")

        mock_prev = MagicMock()
        mock_prev.n_sweeps = 5
        mock_prev.energy = -3.5
        mock_prev.converged = False

        with patch.object(run_dmrg, "dmrg") as mock_dmrg:
            mock_dmrg.Summary.load.return_value = mock_prev
            mps, sweeps_done = _init_mps(_CFG_MODEL, {"init": "resume"}, 8, ckpt, tmp_path)

        assert mps is mock_prev.state
        assert sweeps_done == 5
        mock_prev.state.canonical.assert_called_once_with(0)

    def test_resume_falls_back_to_random_when_no_prior_checkpoint(self, tmp_path):
        from intraknot.algorithm import run_dmrg
        from intraknot.algorithm.run_dmrg import _init_mps

        with (
            patch.object(run_dmrg, "load_space", return_value=(MagicMock(), {})),
            patch.object(run_dmrg, "init_mps") as mock_im,
        ):
            mock_im.return_value = MagicMock()
            _, sweeps_done = _init_mps(
                _CFG_MODEL,
                {"init": "resume", "max_bond": 16, "seed": 0},
                8, None, tmp_path,  # prior_checkpoint=None → fall back
            )

        assert mock_im.called, "init_mps should be called for the random fallback"
        assert sweeps_done == 0
        assert mock_im.call_args.kwargs["bond_dim"] == 16


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
        # Simulate a prior attempt.
        (run_dir / "main" / "attempts" / "attempt_01").mkdir(parents=True)
        _write_config_toml(run_dir)
        _run(run_dir)
        assert (run_dir / "main" / "attempts" / "attempt_02").is_dir()


# ---------------------------------------------------------------------------
# run() — sweep budget
# ---------------------------------------------------------------------------

class TestRunSweepBudget:
    """run() reduces opts.n_sweeps by the number of sweeps already completed."""

    def test_sweep_budget_reduced_on_resume(self, tmp_path):
        """With sweeps_done=3 and an original budget of 10, the runner must set
        opts.n_sweeps to 7 before calling dmrg.run."""
        from intraknot.algorithm import run_dmrg

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        mock_geo = MagicMock()
        mock_geo.L = 8
        mock_opts = MagicMock()
        mock_opts.n_sweeps = 10  # original budget

        with (
            patch.object(run_dmrg, "alice"),
            patch.object(run_dmrg, "logging") as ml,
            patch.object(run_dmrg, "build_interaction", return_value=([], MagicMock(), mock_geo)),
            patch.object(run_dmrg, "build_hamiltonian", return_value=MagicMock()),
            patch.object(run_dmrg, "_init_mps", return_value=(MagicMock(), 3)),
            patch.object(run_dmrg, "dmrg") as mock_dmrg_mod,
            patch.object(run_dmrg, "write_status"),
        ):
            ml.INFO = 20
            mock_dmrg_mod.Options.from_toml.return_value = mock_opts
            mock_dmrg_mod.run.return_value = _mock_summary()
            run_dmrg.run(run_dir)

        assert mock_opts.n_sweeps == 7

    def test_sweep_budget_floored_at_one(self, tmp_path):
        """Remaining budget is at least 1, even if sweeps_done >= n_sweeps."""
        from intraknot.algorithm import run_dmrg

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        mock_geo = MagicMock()
        mock_geo.L = 8
        mock_opts = MagicMock()
        mock_opts.n_sweeps = 2  # budget already exhausted

        with (
            patch.object(run_dmrg, "alice"),
            patch.object(run_dmrg, "logging") as ml,
            patch.object(run_dmrg, "build_interaction", return_value=([], MagicMock(), mock_geo)),
            patch.object(run_dmrg, "build_hamiltonian", return_value=MagicMock()),
            patch.object(run_dmrg, "_init_mps", return_value=(MagicMock(), 5)),
            patch.object(run_dmrg, "dmrg") as mock_dmrg_mod,
            patch.object(run_dmrg, "write_status"),
        ):
            ml.INFO = 20
            mock_dmrg_mod.Options.from_toml.return_value = mock_opts
            mock_dmrg_mod.run.return_value = _mock_summary()
            run_dmrg.run(run_dir)

        assert mock_opts.n_sweeps == 1

    def test_sweep_budget_unchanged_on_fresh_start(self, tmp_path):
        """When sweeps_done=0 (fresh start), opts.n_sweeps is not modified."""
        from intraknot.algorithm import run_dmrg

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        mock_geo = MagicMock()
        mock_geo.L = 8
        mock_opts = MagicMock()
        mock_opts.n_sweeps = 10

        with (
            patch.object(run_dmrg, "alice"),
            patch.object(run_dmrg, "logging") as ml,
            patch.object(run_dmrg, "build_interaction", return_value=([], MagicMock(), mock_geo)),
            patch.object(run_dmrg, "build_hamiltonian", return_value=MagicMock()),
            patch.object(run_dmrg, "_init_mps", return_value=(MagicMock(), 0)),
            patch.object(run_dmrg, "dmrg") as mock_dmrg_mod,
            patch.object(run_dmrg, "write_status"),
        ):
            ml.INFO = 20
            mock_dmrg_mod.Options.from_toml.return_value = mock_opts
            mock_dmrg_mod.run.return_value = _mock_summary()
            run_dmrg.run(run_dir)

        assert mock_opts.n_sweeps == 10


# ---------------------------------------------------------------------------
# run() — status file transitions
# ---------------------------------------------------------------------------

class TestRunStatus:
    """run() must write the correct `RunState` to the main and attempt status files."""

    def _collect_status_calls(self, run_dir: Path, *, converged: bool):
        """Run with mocked write_status and return all (path, status) calls."""
        from intraknot.algorithm import run_dmrg

        mock_geo = MagicMock()
        mock_geo.L = 8
        calls: list = []

        with (
            patch.object(run_dmrg, "alice"),
            patch.object(run_dmrg, "logging") as ml,
            patch.object(run_dmrg, "build_interaction", return_value=([], MagicMock(), mock_geo)),
            patch.object(run_dmrg, "build_hamiltonian", return_value=MagicMock()),
            patch.object(run_dmrg, "load_space", return_value=(MagicMock(), {})),
            patch.object(run_dmrg, "init_mps", return_value=MagicMock()),
            patch.object(run_dmrg, "dmrg") as mock_dmrg_mod,
            patch.object(
                run_dmrg, "write_status",
                side_effect=lambda p, s: calls.append((p, s)),
            ),
        ):
            ml.INFO = 20
            mock_opts = MagicMock()
            mock_opts.n_sweeps = 2
            mock_dmrg_mod.Options.from_toml.return_value = mock_opts
            mock_dmrg_mod.run.return_value = _mock_summary(converged=converged)

            if converged:
                run_dmrg.run(run_dir)
            else:
                with pytest.raises(SystemExit):
                    run_dmrg.run(run_dir)

        return calls

    def test_converged_sets_completed_on_main_status(self, tmp_path):
        from intraknot.status import RunState

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        calls = self._collect_status_calls(run_dir, converged=True)

        # The final write to main/status.json must have state=COMPLETED.
        main_calls = [(p, s) for p, s in calls if "attempt" not in str(p)]
        assert main_calls[-1][1].state == RunState.COMPLETED

    def test_not_converged_sets_failed_on_main_status(self, tmp_path):
        from intraknot.status import RunState

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        calls = self._collect_status_calls(run_dir, converged=False)

        main_calls = [(p, s) for p, s in calls if "attempt" not in str(p)]
        assert main_calls[-1][1].state == RunState.FAILED

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
    """run() must write info.json, conv.csv, and (on convergence) state.ckpt."""

    def test_info_json_written(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)
        _run(run_dir)
        assert (run_dir / "main" / "attempts" / "attempt_01" / "info.json").exists()

    def test_conv_csv_written(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)
        _run(run_dir)
        assert (run_dir / "main" / "attempts" / "attempt_01" / "conv.csv").exists()

    def test_state_ckpt_saved_when_converged(self, tmp_path):
        """On a converged run, summary.save is called to write state.ckpt."""
        from intraknot.algorithm import run_dmrg

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        mock_geo = MagicMock()
        mock_geo.L = 8
        mock_summary = _mock_summary(converged=True)

        with (
            patch.object(run_dmrg, "alice"),
            patch.object(run_dmrg, "logging") as ml,
            patch.object(run_dmrg, "build_interaction", return_value=([], MagicMock(), mock_geo)),
            patch.object(run_dmrg, "build_hamiltonian", return_value=MagicMock()),
            patch.object(run_dmrg, "load_space", return_value=(MagicMock(), {})),
            patch.object(run_dmrg, "init_mps", return_value=MagicMock()),
            patch.object(run_dmrg, "dmrg") as mock_dmrg_mod,
            patch.object(run_dmrg, "write_status"),
        ):
            ml.INFO = 20
            mock_opts = MagicMock()
            mock_opts.n_sweeps = 2
            mock_dmrg_mod.Options.from_toml.return_value = mock_opts
            mock_dmrg_mod.run.return_value = mock_summary
            run_dmrg.run(run_dir)

        mock_summary.save.assert_called_once()
        saved_path = mock_summary.save.call_args[0][0]
        assert saved_path.name == "state.ckpt"

    def test_state_ckpt_skipped_when_not_converged(self, tmp_path):
        """When the run does not converge, summary.save must not be called."""
        from intraknot.algorithm import run_dmrg

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        mock_geo = MagicMock()
        mock_geo.L = 8
        mock_summary = _mock_summary(converged=False)

        with (
            patch.object(run_dmrg, "alice"),
            patch.object(run_dmrg, "logging") as ml,
            patch.object(run_dmrg, "build_interaction", return_value=([], MagicMock(), mock_geo)),
            patch.object(run_dmrg, "build_hamiltonian", return_value=MagicMock()),
            patch.object(run_dmrg, "load_space", return_value=(MagicMock(), {})),
            patch.object(run_dmrg, "init_mps", return_value=MagicMock()),
            patch.object(run_dmrg, "dmrg") as mock_dmrg_mod,
            patch.object(run_dmrg, "write_status"),
        ):
            ml.INFO = 20
            mock_opts = MagicMock()
            mock_opts.n_sweeps = 2
            mock_dmrg_mod.Options.from_toml.return_value = mock_opts
            mock_dmrg_mod.run.return_value = mock_summary
            with pytest.raises(SystemExit):
                run_dmrg.run(run_dir)

        mock_summary.save.assert_not_called()

    def test_not_converged_keeps_dmrg_ckpt(self, tmp_path):
        """A not-converged run must not delete the dmrg.ckpt Alice wrote."""
        from intraknot.algorithm import run_dmrg

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        mock_geo = MagicMock()
        mock_geo.L = 8
        mock_summary = _mock_summary(converged=False)

        def _fake_dmrg_run(mps, mpo, opts):
            # Mimic Alice writing dmrg.ckpt into opts.checkpoint_dir mid-run.
            Path(opts.checkpoint_dir, "dmrg.ckpt").write_bytes(b"fake-checkpoint")
            return mock_summary

        with (
            patch.object(run_dmrg, "alice"),
            patch.object(run_dmrg, "logging") as ml,
            patch.object(run_dmrg, "build_interaction", return_value=([], MagicMock(), mock_geo)),
            patch.object(run_dmrg, "build_hamiltonian", return_value=MagicMock()),
            patch.object(run_dmrg, "load_space", return_value=(MagicMock(), {})),
            patch.object(run_dmrg, "init_mps", return_value=MagicMock()),
            patch.object(run_dmrg, "dmrg") as mock_dmrg_mod,
            patch.object(run_dmrg, "write_status"),
        ):
            ml.INFO = 20
            mock_opts = MagicMock()
            mock_opts.n_sweeps = 2
            mock_dmrg_mod.Options.from_toml.return_value = mock_opts
            mock_dmrg_mod.run.side_effect = _fake_dmrg_run
            with pytest.raises(SystemExit):
                run_dmrg.run(run_dir)

        attempt_dir = run_dir / "main" / "attempts" / "attempt_01"
        assert (attempt_dir / "dmrg.ckpt").exists()
        mock_summary.save.assert_not_called()

    def test_converged_removes_dmrg_ckpt(self, tmp_path):
        """On convergence, the dmrg.ckpt/dmrg_lock.ckpt Alice wrote is removed."""
        from intraknot.algorithm import run_dmrg

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        mock_geo = MagicMock()
        mock_geo.L = 8
        mock_summary = _mock_summary(converged=True)

        def _fake_dmrg_run(mps, mpo, opts):
            # Mimic Alice writing dmrg.ckpt (and its atomic-rename staging
            # file) into opts.checkpoint_dir mid-run.
            Path(opts.checkpoint_dir, "dmrg.ckpt").write_bytes(b"fake-checkpoint")
            Path(opts.checkpoint_dir, "dmrg_lock.ckpt").write_bytes(b"fake-lock")
            return mock_summary

        with (
            patch.object(run_dmrg, "alice"),
            patch.object(run_dmrg, "logging") as ml,
            patch.object(run_dmrg, "build_interaction", return_value=([], MagicMock(), mock_geo)),
            patch.object(run_dmrg, "build_hamiltonian", return_value=MagicMock()),
            patch.object(run_dmrg, "load_space", return_value=(MagicMock(), {})),
            patch.object(run_dmrg, "init_mps", return_value=MagicMock()),
            patch.object(run_dmrg, "dmrg") as mock_dmrg_mod,
            patch.object(run_dmrg, "write_status"),
        ):
            ml.INFO = 20
            mock_opts = MagicMock()
            mock_opts.n_sweeps = 2
            mock_dmrg_mod.Options.from_toml.return_value = mock_opts
            mock_dmrg_mod.run.side_effect = _fake_dmrg_run
            run_dmrg.run(run_dir)

        attempt_dir = run_dir / "main" / "attempts" / "attempt_01"
        assert not (attempt_dir / "dmrg.ckpt").exists()
        assert not (attempt_dir / "dmrg_lock.ckpt").exists()


# ---------------------------------------------------------------------------
# run() — exception mapping
# ---------------------------------------------------------------------------

class TestRunExceptionMapping:
    """run() maps Alice/config failures onto IntraKnot status reasons.

    A deterministic configuration error must not be reported as a restartable
    failure, or `iknot run resume` would relaunch a run that cannot succeed.
    """

    def _run_with_side_effect(self, run_dir: Path, exc: BaseException):
        from intraknot.algorithm import run_dmrg

        mock_geo = MagicMock()
        mock_geo.L = 8
        calls: list = []

        with (
            patch.object(run_dmrg, "alice"),
            patch.object(run_dmrg, "logging") as ml,
            patch.object(run_dmrg, "build_interaction",
                         return_value=([], MagicMock(), mock_geo)),
            patch.object(run_dmrg, "build_hamiltonian", return_value=MagicMock()),
            patch.object(run_dmrg, "load_space", return_value=(MagicMock(), {})),
            patch.object(run_dmrg, "init_mps", return_value=MagicMock()),
            patch.object(run_dmrg, "dmrg") as mock_dmrg_mod,
            patch.object(run_dmrg, "write_status",
                         side_effect=lambda p, s: calls.append((p, s))),
        ):
            ml.INFO = 20
            mock_opts = MagicMock()
            mock_opts.n_sweeps = 2
            mock_dmrg_mod.Options.from_toml.return_value = mock_opts
            mock_dmrg_mod.run.side_effect = exc
            with pytest.raises(SystemExit):
                run_dmrg.run(run_dir)

        return calls

    def test_value_error_sets_invalid(self, tmp_path):
        from intraknot.status import FailureReason, RunState

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)
        calls = self._run_with_side_effect(run_dir, ValueError("bad target_qn"))

        main_calls = [(p, s) for p, s in calls if "attempt" not in str(p)]
        assert main_calls[-1][1].state == RunState.INVALID
        assert main_calls[-1][1].reason == FailureReason.BAD_PARAMETERS
        assert main_calls[-1][1].restartable is False

    def test_missing_checkpoint_is_not_restartable(self, tmp_path):
        """`init = "ckpt"` with an absent file cannot be fixed by retrying."""
        from intraknot.status import FailureReason, RunState

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)
        calls = self._run_with_side_effect(
            run_dir, FileNotFoundError("initial.ckpt")
        )

        main_calls = [(p, s) for p, s in calls if "attempt" not in str(p)]
        assert main_calls[-1][1].state == RunState.INVALID
        assert main_calls[-1][1].reason == FailureReason.BAD_PARAMETERS
        assert main_calls[-1][1].restartable is False

    def test_runtime_error_sets_linear_algebra(self, tmp_path):
        from intraknot.status import FailureReason, RunState

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)
        calls = self._run_with_side_effect(run_dir, RuntimeError("lanczos failed"))

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

    def test_unknown_exception_stays_restartable(self, tmp_path):
        """An unrecognized failure may be transient, so retrying is allowed."""
        from intraknot.status import FailureReason, RunState

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)
        calls = self._run_with_side_effect(run_dir, OSError("node lost"))

        main_calls = [(p, s) for p, s in calls if "attempt" not in str(p)]
        assert main_calls[-1][1].state == RunState.FAILED
        assert main_calls[-1][1].reason == FailureReason.SCHEDULER_FAILURE
        assert main_calls[-1][1].restartable is True


# ---------------------------------------------------------------------------
# Plugin forwarding
# ---------------------------------------------------------------------------

class TestPluginForwarding:
    """The `[plugin]` table from `config.toml` must reach `build_interaction`.

    Before the fix, `cfg_model` was assembled from only `geometry` and `model`
    keys, silently dropping any `[plugin]` section that Alice uses to load
    custom geometry, intrcmap, space, and model callables.
    """

    def test_plugin_section_forwarded(self, tmp_path):
        """A `[plugin]` table in `config.toml` is included in the config dict
        passed to `build_interaction`."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(
            run_dir,
            extra='[plugin]\nmodel = "plugins/my_model.py:build_model"\n',
        )

        mock_bi = _run(run_dir)

        cfg_passed = mock_bi.call_args[0][0]
        assert "plugin" in cfg_passed, (
            "build_interaction was not given the [plugin] section from config.toml"
        )
        assert cfg_passed["plugin"].get("model") == "plugins/my_model.py:build_model"

    def test_absent_plugin_section_gives_empty_dict(self, tmp_path):
        """When `config.toml` has no `[plugin]` table, `build_interaction`
        receives `{'plugin': {}}` so Alice's `config.get('plugin', {})` is
        consistent with the explicit-section case."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(run_dir)

        mock_bi = _run(run_dir)

        cfg_passed = mock_bi.call_args[0][0]
        assert "plugin" in cfg_passed
        assert cfg_passed["plugin"] == {}

    def test_all_plugin_keys_forwarded(self, tmp_path):
        """All four Alice plugin entry-points (geometry, intrcmap, space, model)
        are preserved when forwarded through `cfg_model`."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_config_toml(
            run_dir,
            extra=(
                '[plugin]\n'
                'geometry  = "plugins/geo.py:build_geo"\n'
                'intrcmap  = "plugins/imap.py:build_imap"\n'
                'space     = "plugins/space.py:build_space"\n'
                'model     = "plugins/model.py:build_model"\n'
            ),
        )

        mock_bi = _run(run_dir)

        plugin = mock_bi.call_args[0][0]["plugin"]
        assert plugin["geometry"] == "plugins/geo.py:build_geo"
        assert plugin["intrcmap"] == "plugins/imap.py:build_imap"
        assert plugin["space"]    == "plugins/space.py:build_space"
        assert plugin["model"]    == "plugins/model.py:build_model"
