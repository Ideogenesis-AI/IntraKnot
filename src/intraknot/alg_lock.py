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


"""Algorithm lock file management for IntraKnot campaigns.

The `algorithm.lock` TOML file records the provenance of every script in a
campaign's `algorithm/` directory. Two kinds of entries are distinguished:

Managed
    Scripts installed from a known source — the `intraknot` package or a
    registered algorithm database. The stored SHA-256 digest lets
    `iknot campaign sync` detect upstream changes and local modifications
    without downloading the file body first.

Custom
    User-authored scripts that IntraKnot never modifies automatically.

The lock file lives at `campaigns/<id>/algorithm/algorithm.lock` so that it
is included in each run's frozen `algorithm/` snapshot automatically.
"""

from __future__ import annotations

import datetime
import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ManagedEntry:
    """Provenance record for a single managed algorithm script.

    Parameters
    ----------
    file:
        Relative path of the script inside `algorithm/`, e.g.
        `"run_dmrg.py"` or `"observables/spin_corr.py"`.
    source:
        Source descriptor in `"<db-name>:<path>"` form, e.g.
        `"intraknot:algorithm/run_dmrg.py"` or
        `"intraknot-database:observables/spin_corr.py"`.
    sha256:
        SHA-256 hex digest of the installed file content.
    installed_at:
        ISO-format date string of when the file was installed or last synced.
    installed_from_version:
        IntraKnot package version at install time. Meaningful only for
        `"intraknot:"`-sourced entries; `None` for database-sourced scripts.
    """

    file: str
    source: str
    sha256: str
    installed_at: str
    installed_from_version: Optional[str] = None


@dataclass
class AlgorithmLock:
    """Contents of an `algorithm.lock` file.

    Parameters
    ----------
    managed:
        Mapping from relative file path to its `ManagedEntry`. The key is
        the same string as `ManagedEntry.file`.
    custom:
        Relative file paths declared as user-owned. IntraKnot never
        overwrites or removes these automatically.
    """

    managed: Dict[str, ManagedEntry] = field(default_factory=dict)
    custom: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "AlgorithmLock":
        """Load `algorithm.lock` from a TOML file.

        Returns an empty lock when the file does not exist, so callers do
        not need to guard against a freshly created campaign that has not
        yet had any managed files installed.

        Parameters
        ----------
        path:
            Path to the `algorithm.lock` file.

        Returns
        -------
        AlgorithmLock
            Parsed lock, or a fresh empty instance when the file is absent.
        """
        if not path.exists():
            return cls()
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)

        managed: Dict[str, ManagedEntry] = {}
        for _toml_key, entry in raw.get("managed", {}).items():
            # The TOML key is a sanitised label; the canonical path is stored
            # inside the entry under "file" so we use that as the dict key.
            filename = entry["file"]
            managed[filename] = ManagedEntry(
                file=filename,
                source=entry["source"],
                sha256=entry["sha256"],
                installed_at=entry.get("installed_at", ""),
                installed_from_version=entry.get("installed_from_version"),
            )

        custom: List[str] = list(raw.get("custom", {}).get("files", []))
        return cls(managed=managed, custom=custom)

    def save(self, path: Path) -> None:
        """Write the lock to `path` in TOML format.

        Parameters
        ----------
        path:
            Destination path, typically `algorithm/algorithm.lock`.
        """
        lines: List[str] = [
            "# algorithm.lock — managed by intraknot, do not edit manually\n",
        ]

        for filename, entry in sorted(self.managed.items()):
            # Derive a safe TOML bare-key by replacing path separators and
            # dots with underscores so the key is a valid TOML identifier.
            toml_key = filename.replace("/", "_").replace(".", "_").replace("-", "_")
            lines.append(f"\n[managed.{toml_key}]\n")
            lines.append(f'file = "{entry.file}"\n')
            lines.append(f'source = "{entry.source}"\n')
            lines.append(f'sha256 = "{entry.sha256}"\n')
            if entry.installed_from_version is not None:
                lines.append(
                    f'installed_from_version = "{entry.installed_from_version}"\n'
                )
            lines.append(f'installed_at = "{entry.installed_at}"\n')

        if self.custom:
            lines.append("\n[custom]\n")
            items_str = ", ".join(f'"{f}"' for f in self.custom)
            lines.append(f"files = [{items_str}]\n")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines))

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_managed(self, entry: ManagedEntry) -> None:
        """Add or replace a managed entry.

        If the file was previously listed as custom it is removed from
        the custom list so the two sets remain disjoint.

        Parameters
        ----------
        entry:
            New `ManagedEntry` to record under its `file` key.
        """
        self.managed[entry.file] = entry
        if entry.file in self.custom:
            self.custom.remove(entry.file)

    def promote_to_custom(self, filename: str) -> None:
        """Promote a managed file to custom (user-owned) status.

        Removes `filename` from `managed` and appends it to `custom`
        unless already present there.

        Parameters
        ----------
        filename:
            Relative path inside `algorithm/` to promote.

        Raises
        ------
        KeyError
            If `filename` is not currently a managed entry.
        """
        if filename not in self.managed:
            raise KeyError(
                f"{filename!r} is not a managed entry in algorithm.lock"
            )
        del self.managed[filename]
        if filename not in self.custom:
            self.custom.append(filename)

    def add_custom(self, filename: str) -> None:
        """Register a file as custom (user-owned).

        Does nothing if `filename` is already in the custom list.

        Parameters
        ----------
        filename:
            Relative path inside `algorithm/` to register.
        """
        if filename not in self.custom:
            self.custom.append(filename)

    def remove(self, filename: str) -> None:
        """Remove a file from the lock entirely.

        Removes `filename` from whichever list it appears in — `managed`
        or `custom`. Raises `KeyError` when the file is not tracked at all.
        The on-disk file is not touched.

        Parameters
        ----------
        filename:
            Relative path inside `algorithm/` to deregister.

        Raises
        ------
        KeyError
            If `filename` is neither a managed nor a custom entry.
        """
        if filename in self.managed:
            del self.managed[filename]
        elif filename in self.custom:
            self.custom.remove(filename)
        else:
            raise KeyError(
                f"{filename!r} is not tracked in algorithm.lock"
            )

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def check_modified(self, alg_dir: Path) -> List[str]:
        """Return managed files whose on-disk SHA-256 differs from the lock.

        Only files that exist on disk are checked; missing files are
        silently skipped (they may have been deleted intentionally).

        Parameters
        ----------
        alg_dir:
            The `algorithm/` directory containing the actual files.

        Returns
        -------
        list[str]
            Relative paths of managed files that have been locally modified
            since the lock was last written.
        """
        modified: List[str] = []
        for filename, entry in self.managed.items():
            file_path = alg_dir / filename
            if not file_path.exists():
                continue
            if _sha256_path(file_path) != entry.sha256:
                modified.append(filename)
        return modified

    def is_managed(self, filename: str) -> bool:
        """Return `True` if `filename` is tracked as a managed entry."""
        return filename in self.managed

    def is_custom(self, filename: str) -> bool:
        """Return `True` if `filename` is tracked as a custom entry."""
        return filename in self.custom


# ---------------------------------------------------------------------------
# SHA-256 helpers (shared with database.py)
# ---------------------------------------------------------------------------

def _sha256_path(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Reads the file in 64 KiB chunks to keep memory usage bounded for large
    checkpoint or data files that might sit next to the scripts.

    Parameters
    ----------
    path:
        File to hash.

    Returns
    -------
    str
        64-character lowercase hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of a byte string.

    Parameters
    ----------
    data:
        Bytes to hash.

    Returns
    -------
    str
        64-character lowercase hex digest.
    """
    return hashlib.sha256(data).hexdigest()


def make_managed_entry(
    file: str,
    source: str,
    sha256: str,
    installed_from_version: Optional[str] = None,
) -> ManagedEntry:
    """Construct a `ManagedEntry` with today's ISO date as `installed_at`.

    Parameters
    ----------
    file:
        Relative path inside `algorithm/`.
    source:
        Source descriptor, e.g. `"intraknot:algorithm/run_dmrg.py"`.
    sha256:
        SHA-256 hex digest of the installed file content.
    installed_from_version:
        IntraKnot package version. Provide only for `"intraknot:"`-sourced
        entries; pass `None` for database-sourced scripts.

    Returns
    -------
    ManagedEntry
        Entry with `installed_at` set to today's date in ISO format.
    """
    return ManagedEntry(
        file=file,
        source=source,
        sha256=sha256,
        installed_at=datetime.date.today().isoformat(),
        installed_from_version=installed_from_version,
    )
