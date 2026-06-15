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


"""Tests for src/intraknot/database.py."""

from __future__ import annotations

import datetime
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from intraknot.alg_lock import _sha256_bytes
from intraknot.database import (
    BUILTIN_SOURCE,
    CategoryEntry,
    DatabaseEntry,
    DatabaseRegistry,
    ScriptEntry,
    _file_url,
    _parse_source,
    _raw_base_url,
    _registry_url,
)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

class TestRawBaseUrl:
    def test_converts_github_url(self):
        url = _raw_base_url("https://github.com/Ideogenesis-AI/intraknot-database")
        assert url == "https://raw.githubusercontent.com/Ideogenesis-AI/intraknot-database/primary"

    def test_strips_trailing_slash(self):
        url = _raw_base_url("https://github.com/foo/bar/")
        assert url == "https://raw.githubusercontent.com/foo/bar/primary"

    def test_rejects_non_github_url(self):
        with pytest.raises(ValueError, match="GitHub"):
            _raw_base_url("https://gitlab.com/foo/bar")


class TestRegistryUrl:
    def test_appends_registry_yaml(self):
        url = _registry_url("https://github.com/foo/bar")
        assert url.endswith("/registry.yaml")
        assert "raw.githubusercontent.com" in url


class TestFileUrl:
    def test_constructs_file_url(self):
        url = _file_url("https://github.com/foo/bar", "observables/spin_corr.py")
        assert url == "https://raw.githubusercontent.com/foo/bar/primary/observables/spin_corr.py"

    def test_strips_leading_slash_from_path(self):
        url = _file_url("https://github.com/foo/bar", "/observables/spin_corr.py")
        assert url == "https://raw.githubusercontent.com/foo/bar/primary/observables/spin_corr.py"


# ---------------------------------------------------------------------------
# _parse_source
# ---------------------------------------------------------------------------

class TestParseSource:
    def test_valid_builtin_source(self):
        db, path = _parse_source("intraknot:algorithm/run_dmrg.py")
        assert db == "intraknot"
        assert path == "algorithm/run_dmrg.py"

    def test_valid_database_source(self):
        db, path = _parse_source("intraknot-database:observables/spin_corr.py")
        assert db == "intraknot-database"
        assert path == "observables/spin_corr.py"

    def test_no_colon_raises(self):
        with pytest.raises(ValueError, match="<db-name>:<path>"):
            _parse_source("intraknot_database/spin_corr.py")

    def test_empty_db_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _parse_source(":observables/spin_corr.py")

    def test_empty_path_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _parse_source("intraknot-database:")

    def test_path_with_colons(self):
        # Only the first colon is used as the separator.
        db, path = _parse_source("mydb:some:weird:path.py")
        assert db == "mydb"
        assert path == "some:weird:path.py"


# ---------------------------------------------------------------------------
# DatabaseRegistry — persistence
# ---------------------------------------------------------------------------

_REGISTRY_YAML = """\
databases:
  test-db:
    url: https://github.com/foo/test-db
    registry_url: https://raw.githubusercontent.com/foo/test-db/primary/registry.yaml
    fetched_at: "2026-01-15"
    description: Test database
    categories:
      observables:
        description: Observable scripts
        scripts:
          - name: spin_corr
            file: observables/spin_corr.py
            description: Spin-spin correlation
            sha256: aabbcc
"""


class TestDatabaseRegistryLoad:
    def test_load_missing_returns_empty(self, tmp_path):
        reg = DatabaseRegistry.load(tmp_path / "nonexistent.yaml")
        assert reg.entries == {}

    def test_load_parses_entry(self, tmp_path):
        path = tmp_path / "registry.yaml"
        path.write_text(_REGISTRY_YAML)
        reg = DatabaseRegistry.load(path)
        assert "test-db" in reg.entries
        entry = reg.entries["test-db"]
        assert entry.url == "https://github.com/foo/test-db"
        assert entry.description == "Test database"
        assert entry.fetched_at == "2026-01-15"

    def test_load_parses_categories(self, tmp_path):
        path = tmp_path / "registry.yaml"
        path.write_text(_REGISTRY_YAML)
        reg = DatabaseRegistry.load(path)
        entry = reg.entries["test-db"]
        assert "observables" in entry.categories
        cat = entry.categories["observables"]
        assert cat.description == "Observable scripts"
        assert len(cat.scripts) == 1
        script = cat.scripts[0]
        assert script.name == "spin_corr"
        assert script.file == "observables/spin_corr.py"
        assert script.sha256 == "aabbcc"

    def test_load_derives_registry_url_when_absent(self, tmp_path):
        """Entries without `registry_url` have it derived from `url`."""
        yaml_text = """\
databases:
  old-db:
    url: https://github.com/foo/old-db
    fetched_at: ""
"""
        path = tmp_path / "registry.yaml"
        path.write_text(yaml_text)
        reg = DatabaseRegistry.load(path)
        entry = reg.entries["old-db"]
        assert "raw.githubusercontent.com" in entry.registry_url


class TestDatabaseRegistrySave:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "configs" / "registry.yaml"
        reg = DatabaseRegistry()
        reg.entries["test-db"] = DatabaseEntry(
            name="test-db",
            url="https://github.com/foo/test-db",
            registry_url="https://raw.githubusercontent.com/foo/test-db/primary/registry.yaml",
            fetched_at="2026-01-15",
            description="Test DB",
            categories={
                "observables": CategoryEntry(
                    description="Observables",
                    scripts=[ScriptEntry(
                        name="spin_corr",
                        file="observables/spin_corr.py",
                        description="Spin correlation",
                        sha256="aabb",
                    )],
                )
            },
        )
        reg.save(path)

        loaded = DatabaseRegistry.load(path)
        assert "test-db" in loaded.entries
        entry = loaded.entries["test-db"]
        assert entry.description == "Test DB"
        assert "observables" in entry.categories
        assert entry.categories["observables"].scripts[0].name == "spin_corr"

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "registry.yaml"
        DatabaseRegistry().save(path)
        assert path.exists()

    def test_empty_registry_writes_file(self, tmp_path):
        path = tmp_path / "registry.yaml"
        DatabaseRegistry().save(path)
        content = path.read_text()
        assert "databases:" in content

    def test_header_comment_present(self, tmp_path):
        path = tmp_path / "registry.yaml"
        DatabaseRegistry().save(path)
        text = path.read_text()
        assert "iknot database" in text


# ---------------------------------------------------------------------------
# DatabaseRegistry — add and update (mocked HTTP)
# ---------------------------------------------------------------------------

_MOCK_REGISTRY_YAML = b"""\
description: A mock algorithm database
categories:
  observables:
    description: Observable scripts
    scripts:
      - name: spin_corr
        file: observables/spin_corr.py
        description: Spin-spin correlation function
        sha256: deadbeefdeadbeef
"""


class TestDatabaseRegistryAdd:
    def test_add_populates_entry(self, tmp_path):
        reg = DatabaseRegistry()
        with patch("intraknot.database._fetch_bytes", return_value=_MOCK_REGISTRY_YAML):
            entry = reg.add(
                "test-db", "https://github.com/foo/test-db"
            )
        assert "test-db" in reg.entries
        assert entry.description == "A mock algorithm database"
        assert entry.fetched_at == datetime.date.today().isoformat()

    def test_add_parses_categories(self, tmp_path):
        reg = DatabaseRegistry()
        with patch("intraknot.database._fetch_bytes", return_value=_MOCK_REGISTRY_YAML):
            entry = reg.add("test-db", "https://github.com/foo/test-db")
        assert "observables" in entry.categories
        scripts = entry.categories["observables"].scripts
        assert len(scripts) == 1
        assert scripts[0].file == "observables/spin_corr.py"

    def test_add_rejects_builtin_name(self):
        reg = DatabaseRegistry()
        with pytest.raises(ValueError, match="reserved"):
            reg.add(BUILTIN_SOURCE, "https://github.com/foo/bar")

    def test_add_rejects_non_github_url(self):
        reg = DatabaseRegistry()
        with pytest.raises(ValueError, match="GitHub"):
            reg.add("mydb", "https://gitlab.com/foo/bar")

    def test_add_raises_on_http_error(self):
        reg = DatabaseRegistry()
        http_err = urllib.error.HTTPError(
            url="https://example.com",
            code=404,
            msg="Not Found",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )
        with patch("intraknot.database._fetch_bytes", side_effect=http_err):
            with pytest.raises(urllib.error.HTTPError):
                reg.add("test-db", "https://github.com/foo/test-db")


class TestDatabaseRegistryUpdate:
    def test_update_refreshes_entry(self, tmp_path):
        reg = DatabaseRegistry()
        with patch("intraknot.database._fetch_bytes", return_value=_MOCK_REGISTRY_YAML):
            reg.add("test-db", "https://github.com/foo/test-db")

        updated_yaml = _MOCK_REGISTRY_YAML.replace(
            b"A mock algorithm database", b"Updated description"
        )
        with patch("intraknot.database._fetch_bytes", return_value=updated_yaml):
            entry = reg.update("test-db")

        assert entry.description == "Updated description"
        assert reg.entries["test-db"].description == "Updated description"

    def test_update_raises_for_unknown_db(self):
        reg = DatabaseRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.update("nonexistent")

    def test_update_preserves_url(self, tmp_path):
        reg = DatabaseRegistry()
        with patch("intraknot.database._fetch_bytes", return_value=_MOCK_REGISTRY_YAML):
            reg.add("test-db", "https://github.com/foo/test-db")
        with patch("intraknot.database._fetch_bytes", return_value=_MOCK_REGISTRY_YAML):
            entry = reg.update("test-db")
        assert entry.url == "https://github.com/foo/test-db"


# ---------------------------------------------------------------------------
# DatabaseRegistry — resolve_file
# ---------------------------------------------------------------------------

class TestResolveFile:
    def test_resolves_builtin_source(self):
        """The built-in `intraknot:` source uses importlib.resources."""
        content = b"# run_dmrg stub\n"
        reg = DatabaseRegistry()
        with patch(
            "importlib.resources.files",
        ) as mock_files:
            mock_pkg = mock_files.return_value
            mock_pkg.joinpath.return_value.read_bytes.return_value = content
            data, sha = reg.resolve_file("intraknot:algorithm/run_dmrg.py")

        assert data == content
        assert sha == _sha256_bytes(content)

    def test_resolves_database_source(self):
        content = b"# spin_corr\n"
        reg = DatabaseRegistry()
        reg.entries["test-db"] = DatabaseEntry(
            name="test-db",
            url="https://github.com/foo/test-db",
            registry_url="https://raw.githubusercontent.com/foo/test-db/primary/registry.yaml",
            fetched_at="2026-01-15",
        )
        with patch("intraknot.database._fetch_bytes", return_value=content):
            data, sha = reg.resolve_file("test-db:observables/spin_corr.py")

        assert data == content
        assert sha == _sha256_bytes(content)

    def test_raises_for_unregistered_database(self):
        reg = DatabaseRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.resolve_file("unknown-db:some/file.py")

    def test_raises_for_malformed_source(self):
        reg = DatabaseRegistry()
        with pytest.raises(ValueError, match="<db-name>:<path>"):
            reg.resolve_file("no_colon_here")


# ---------------------------------------------------------------------------
# DatabaseRegistry — lookup_sha256
# ---------------------------------------------------------------------------

class TestLookupSha256:
    def _make_reg_with_script(self) -> DatabaseRegistry:
        reg = DatabaseRegistry()
        reg.entries["test-db"] = DatabaseEntry(
            name="test-db",
            url="https://github.com/foo/test-db",
            registry_url="",
            fetched_at="2026-01-15",
            categories={
                "observables": CategoryEntry(
                    description="Observables",
                    scripts=[ScriptEntry(
                        name="spin_corr",
                        file="observables/spin_corr.py",
                        description="",
                        sha256="cafecafe",
                    )],
                )
            },
        )
        return reg

    def test_returns_sha256_for_known_file(self):
        reg = self._make_reg_with_script()
        sha = reg.lookup_sha256("test-db:observables/spin_corr.py")
        assert sha == "cafecafe"

    def test_returns_none_for_missing_file(self):
        reg = self._make_reg_with_script()
        sha = reg.lookup_sha256("test-db:observables/nonexistent.py")
        assert sha is None

    def test_returns_none_for_unregistered_db(self):
        reg = DatabaseRegistry()
        sha = reg.lookup_sha256("unknown:some/file.py")
        assert sha is None

    def test_returns_none_for_builtin_source(self):
        reg = DatabaseRegistry()
        sha = reg.lookup_sha256("intraknot:algorithm/run_dmrg.py")
        assert sha is None
