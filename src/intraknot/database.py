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


"""Algorithm database registry for IntraKnot.

A *database* is a collection of reusable algorithm scripts (observables,
model definitions, interaction maps, etc.) hosted in a git repository. Each
database provides a `registry.yaml` at its root that catalogues every
available script, organised by category.

IntraKnot communicates with databases purely over HTTP. It fetches the
`registry.yaml` manifest to discover available scripts, and then fetches
individual script files on demand. No git cloning is performed.

The aggregated manifest for all registered databases is stored in
`configs/registry.yaml` within the project workspace. This file is written
and updated by `iknot database add` and `iknot database update`.

The built-in `intraknot` source is always available and resolves via
`importlib.resources`; it is never listed in `configs/registry.yaml`.
"""

from __future__ import annotations

import datetime
import importlib.resources
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .alg_lock import _sha256_bytes


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The reserved source name for scripts bundled with the intraknot package.
BUILTIN_SOURCE = "intraknot"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScriptEntry:
    """A single script entry in a database category.

    Parameters
    ----------
    name:
        Short identifier for the script (no extension), e.g. `"spin_corr"`.
    file:
        Relative path within the database repository, e.g.
        `"observables/spin_corr.py"`.
    description:
        Human-readable description of what the script does.
    sha256:
        SHA-256 hex digest of the script content as declared in the
        database's `registry.yaml`. Used for efficient staleness detection
        without downloading the file body.
    """

    name: str
    file: str
    description: str
    sha256: str


@dataclass
class CategoryEntry:
    """A named category within a database.

    Parameters
    ----------
    description:
        Human-readable description of the category.
    scripts:
        Scripts belonging to this category.
    """

    description: str
    scripts: List[ScriptEntry] = field(default_factory=list)


@dataclass
class DatabaseEntry:
    """A registered algorithm database.

    Parameters
    ----------
    name:
        Database name as used in source descriptors, e.g.
        `"intraknot-database"`.
    url:
        Human-facing repository URL (GitHub repo page).
    registry_url:
        Direct URL to the raw `registry.yaml` file.
    fetched_at:
        ISO date string of when the manifest was last fetched, or `""`
        when the database has been registered but never fetched.
    description:
        Human-readable description from the manifest.
    categories:
        Script categories loaded from the manifest. Empty when not yet
        fetched.
    """

    name: str
    url: str
    registry_url: str
    fetched_at: str
    description: str = ""
    categories: Dict[str, CategoryEntry] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def _raw_base_url(repo_url: str) -> str:
    """Derive the raw-file base URL from a GitHub repository URL.

    Converts `https://github.com/<owner>/<repo>` to
    `https://raw.githubusercontent.com/<owner>/<repo>/primary`.

    Parameters
    ----------
    repo_url:
        GitHub repository URL. Trailing slashes are stripped automatically.

    Returns
    -------
    str
        Base URL for raw file access under the `main` branch.

    Raises
    ------
    ValueError
        If `repo_url` does not contain `"github.com/"`.
    """
    url = repo_url.rstrip("/")
    if "github.com/" not in url:
        raise ValueError(
            f"Only GitHub repository URLs are currently supported. Got: {repo_url!r}\n"
            "Expected format: https://github.com/<owner>/<repo>"
        )
    raw = url.replace("https://github.com/", "https://raw.githubusercontent.com/", 1)
    return f"{raw}/primary"


def _registry_url(repo_url: str) -> str:
    """Return the raw URL for a database's `registry.yaml`.

    Parameters
    ----------
    repo_url:
        GitHub repository URL.

    Returns
    -------
    str
        URL to `registry.yaml` on the `main` branch.
    """
    return f"{_raw_base_url(repo_url)}/registry.yaml"


def _file_url(repo_url: str, file_path: str) -> str:
    """Return the raw URL for a specific file in a database repo.

    Parameters
    ----------
    repo_url:
        GitHub repository URL.
    file_path:
        Relative path within the repo, e.g. `"observables/spin_corr.py"`.

    Returns
    -------
    str
        Full raw file URL.
    """
    return f"{_raw_base_url(repo_url)}/{file_path.lstrip('/')}"


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _fetch_bytes(url: str) -> bytes:
    """Fetch a URL and return the raw response body.

    Parameters
    ----------
    url:
        HTTP(S) URL to fetch.

    Returns
    -------
    bytes
        Full response body.

    Raises
    ------
    urllib.error.HTTPError
        For HTTP error responses (e.g. 404 when the repo does not yet exist).
    urllib.error.URLError
        For network-level failures.
    """
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Manifest parsing helpers
# ---------------------------------------------------------------------------

def _parse_categories(raw_categories: dict) -> Dict[str, CategoryEntry]:
    """Parse the `categories` block from a raw YAML dict.

    Parameters
    ----------
    raw_categories:
        The value of `data["categories"]` from a parsed `registry.yaml`.

    Returns
    -------
    dict[str, CategoryEntry]
        Parsed category map.
    """
    cats: Dict[str, CategoryEntry] = {}
    for cat_name, cat_data in raw_categories.items():
        if not isinstance(cat_data, dict):
            continue
        scripts: List[ScriptEntry] = []
        for s in cat_data.get("scripts", []):
            scripts.append(ScriptEntry(
                name=s.get("name", ""),
                file=s.get("file", ""),
                description=s.get("description", ""),
                sha256=s.get("sha256", ""),
            ))
        cats[cat_name] = CategoryEntry(
            description=cat_data.get("description", ""),
            scripts=scripts,
        )
    return cats


def _entry_to_yaml_dict(entry: DatabaseEntry) -> dict:
    """Serialise a `DatabaseEntry` to a plain dict suitable for YAML output.

    Parameters
    ----------
    entry:
        Entry to serialise.

    Returns
    -------
    dict
        Serialisable representation.
    """
    d: dict = {
        "url": entry.url,
        "registry_url": entry.registry_url,
        "fetched_at": entry.fetched_at,
    }
    if entry.description:
        d["description"] = entry.description
    if entry.categories:
        cats: dict = {}
        for cat_name, cat in entry.categories.items():
            scripts = []
            for s in cat.scripts:
                scripts.append({
                    "name": s.name,
                    "file": s.file,
                    "description": s.description,
                    "sha256": s.sha256,
                })
            cats[cat_name] = {
                "description": cat.description,
                "scripts": scripts,
            }
        d["categories"] = cats
    return d


# ---------------------------------------------------------------------------
# DatabaseRegistry
# ---------------------------------------------------------------------------

class DatabaseRegistry:
    """Aggregated registry of algorithm databases.

    The registry is persisted as `configs/registry.yaml` in the workspace.
    Each entry stores the database metadata and the full cached manifest from
    the remote `registry.yaml`.

    The built-in `intraknot` source is always available via
    `importlib.resources` and is never stored in `configs/registry.yaml`.
    """

    def __init__(self) -> None:
        self.entries: Dict[str, DatabaseEntry] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, registry_yaml: Path) -> "DatabaseRegistry":
        """Load the unified registry from `configs/registry.yaml`.

        Returns an empty registry when the file does not exist, so callers
        do not need to guard against a freshly initialised project.

        Parameters
        ----------
        registry_yaml:
            Path to the workspace `configs/registry.yaml`.

        Returns
        -------
        DatabaseRegistry
            Populated registry.
        """
        reg = cls()
        if not registry_yaml.exists():
            return reg

        with open(registry_yaml) as fh:
            raw = yaml.safe_load(fh) or {}

        for db_name, db_data in raw.get("databases", {}).items():
            if not isinstance(db_data, dict):
                continue
            url = db_data.get("url", "")
            r_url = db_data.get("registry_url", "")
            if not r_url and url:
                # Derive the registry_url on the fly if absent (e.g. entries
                # written by older versions of the tool).
                try:
                    r_url = _registry_url(url)
                except ValueError:
                    r_url = ""
            cats = _parse_categories(db_data.get("categories", {}))
            reg.entries[db_name] = DatabaseEntry(
                name=db_name,
                url=url,
                registry_url=r_url,
                fetched_at=db_data.get("fetched_at", ""),
                description=db_data.get("description", ""),
                categories=cats,
            )

        return reg

    def save(self, registry_yaml: Path) -> None:
        """Write the unified registry to `configs/registry.yaml`.

        Parameters
        ----------
        registry_yaml:
            Destination path, typically `configs/registry.yaml`.
        """
        header = (
            "# configs/registry.yaml\n"
            "# Unified algorithm database registry.\n"
            "# Managed by iknot database commands — do not edit manually.\n"
            "#\n"
            "# Add a database:  iknot database add <name> <url>\n"
            "# Update all:      iknot database update\n\n"
        )

        databases_dict: dict = {}
        for name, entry in sorted(self.entries.items()):
            databases_dict[name] = _entry_to_yaml_dict(entry)

        body = yaml.dump(
            {"databases": databases_dict},
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

        registry_yaml.parent.mkdir(parents=True, exist_ok=True)
        registry_yaml.write_text(header + body)

    # ------------------------------------------------------------------
    # Database management
    # ------------------------------------------------------------------

    def add(self, name: str, url: str) -> DatabaseEntry:
        """Register a new database and fetch its manifest.

        Derives the `registry_url` from `url`, fetches the remote
        `registry.yaml`, parses it, and adds the entry to the in-memory
        registry. The caller is responsible for calling `save` afterwards.

        Parameters
        ----------
        name:
            Unique name to register the database under (used in source
            descriptors, e.g. `"intraknot-database"`).
        url:
            GitHub repository URL.

        Returns
        -------
        DatabaseEntry
            The newly created and fetched entry.

        Raises
        ------
        ValueError
            If `name` is `"intraknot"` (reserved) or the URL is not a
            supported GitHub URL.
        urllib.error.HTTPError
            If the `registry.yaml` cannot be fetched (e.g. the repo does
            not exist yet).
        """
        if name == BUILTIN_SOURCE:
            raise ValueError(
                f"The name {BUILTIN_SOURCE!r} is reserved for the built-in "
                "intraknot package source and cannot be registered."
            )
        r_url = _registry_url(url)
        entry = self._fetch_entry(name, url, r_url)
        self.entries[name] = entry
        return entry

    def update(self, name: str) -> DatabaseEntry:
        """Re-fetch the manifest for a registered database.

        Parameters
        ----------
        name:
            Name of the database to update.

        Returns
        -------
        DatabaseEntry
            The updated entry.

        Raises
        ------
        KeyError
            If `name` is not registered.
        urllib.error.HTTPError
            If the manifest cannot be fetched.
        """
        if name not in self.entries:
            raise KeyError(
                f"Database {name!r} is not registered. "
                "Add it first with `iknot database add`."
            )
        existing = self.entries[name]
        entry = self._fetch_entry(name, existing.url, existing.registry_url)
        self.entries[name] = entry
        return entry

    def _fetch_entry(self, name: str, url: str, r_url: str) -> DatabaseEntry:
        """Fetch a `registry.yaml` and return a `DatabaseEntry`.

        Parameters
        ----------
        name:
            Database name.
        url:
            Human-facing repo URL.
        r_url:
            Raw URL for `registry.yaml`.

        Returns
        -------
        DatabaseEntry
            Populated entry.
        """
        try:
            data = _fetch_bytes(r_url)
        except urllib.error.HTTPError as exc:
            raise urllib.error.HTTPError(
                exc.url, exc.code, exc.msg, exc.headers, exc.fp
            ) from exc

        raw = yaml.safe_load(data.decode()) or {}
        description = raw.get("description", "")
        cats = _parse_categories(raw.get("categories", {}))

        return DatabaseEntry(
            name=name,
            url=url,
            registry_url=r_url,
            fetched_at=datetime.date.today().isoformat(),
            description=description,
            categories=cats,
        )

    # ------------------------------------------------------------------
    # File resolution
    # ------------------------------------------------------------------

    def resolve_file(self, source: str) -> Tuple[bytes, str]:
        """Fetch the file described by a source descriptor.

        Supports two kinds of source:

        - `"intraknot:<path>"` — reads from the installed package via
          `importlib.resources`, e.g. `"intraknot:algorithm/run_dmrg.py"`.
        - `"<db-name>:<path>"` — fetches the file over HTTP from the
          registered database, e.g.
          `"intraknot-database:observables/spin_corr.py"`.

        Parameters
        ----------
        source:
            Source descriptor in `"<db-name>:<path>"` form.

        Returns
        -------
        tuple[bytes, str]
            `(content, sha256)` — the raw file content and its SHA-256
            hex digest.

        Raises
        ------
        ValueError
            If `source` is malformed or the database is not registered.
        KeyError
            If the database name in `source` is not found in the registry.
        urllib.error.HTTPError
            If the remote file cannot be fetched.
        """
        db_name, file_path = _parse_source(source)

        if db_name == BUILTIN_SOURCE:
            # Read from the installed package using importlib.resources.
            try:
                content = (
                    importlib.resources.files("intraknot")
                    .joinpath(file_path)
                    .read_bytes()
                )
            except (FileNotFoundError, TypeError, AttributeError) as exc:
                raise ValueError(
                    f"Built-in file not found: {file_path!r} "
                    f"(source: {source!r})"
                ) from exc
            return content, _sha256_bytes(content)

        # Database-sourced file — fetch over HTTP.
        if db_name not in self.entries:
            raise KeyError(
                f"Database {db_name!r} is not registered. "
                "Register it first with `iknot database add`."
            )
        entry = self.entries[db_name]
        url = _file_url(entry.url, file_path)
        content = _fetch_bytes(url)
        return content, _sha256_bytes(content)

    def lookup_sha256(self, source: str) -> Optional[str]:
        """Return the SHA-256 for a source file from the cached registry.

        Looks up the `sha256` field stored in the registry manifest without
        downloading the file. Returns `None` when the database has not been
        fetched yet or the script is not listed.

        The built-in `intraknot` source is not cached — `None` is returned
        and the caller should compute the hash via `resolve_file`.

        Parameters
        ----------
        source:
            Source descriptor in `"<db-name>:<path>"` form.

        Returns
        -------
        str or None
            SHA-256 hex digest from the cached manifest, or `None`.
        """
        db_name, file_path = _parse_source(source)

        if db_name == BUILTIN_SOURCE:
            # No manifest cache for built-ins; caller must hash the file directly.
            return None

        if db_name not in self.entries:
            return None

        entry = self.entries[db_name]
        for cat in entry.categories.values():
            for script in cat.scripts:
                if script.file == file_path:
                    return script.sha256 or None
        return None


# ---------------------------------------------------------------------------
# Source descriptor parsing
# ---------------------------------------------------------------------------

def _parse_source(source: str) -> Tuple[str, str]:
    """Parse a source descriptor into `(db_name, file_path)`.

    Parameters
    ----------
    source:
        String of the form `"<db-name>:<path>"`.

    Returns
    -------
    tuple[str, str]
        `(db_name, file_path)`.

    Raises
    ------
    ValueError
        If `source` does not contain exactly one `":"`.
    """
    if ":" not in source:
        raise ValueError(
            f"Invalid source descriptor {source!r}: expected "
            "'<db-name>:<path>', e.g. 'intraknot-database:observables/spin_corr.py'."
        )
    db_name, file_path = source.split(":", 1)
    if not db_name or not file_path:
        raise ValueError(
            f"Invalid source descriptor {source!r}: both the database name "
            "and the file path must be non-empty."
        )
    return db_name, file_path
