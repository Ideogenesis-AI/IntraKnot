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


"""Algorithm runner scripts for IntraKnot.

This subpackage contains standalone runner scripts that are copied into
campaign and run directories and executed directly by Slurm. Each runner
reads `config.toml` from the run directory, runs the algorithm, and writes
outputs (`state.ckpt`, `info.json`, `conv.csv`, `status.json`).

Available runners
-----------------
run_dmrg
    DMRG ground-state search using Alice.
"""
