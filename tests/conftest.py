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


"""Pytest configuration: shared fixtures exposed via conftest.

Helper functions are kept in `helpers.py` to remain importable from test
modules without resorting to `sys.path` manipulation.
"""

import pytest

from helpers import make_machine, make_run_dir  # noqa: F401  (re-exported via conftest)
from intraknot.config import MachineConfig, PathsConfig


@pytest.fixture()
def machine() -> MachineConfig:
    """A `MachineConfig` with the `uv run` Python command."""
    return MachineConfig(paths=PathsConfig(command="uv run"))
