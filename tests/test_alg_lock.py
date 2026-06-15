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


"""Tests for src/intraknot/alg_lock.py."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from intraknot.alg_lock import (
    AlgorithmLock,
    ManagedEntry,
    _sha256_bytes,
    _sha256_path,
    make_managed_entry,
)


# ---------------------------------------------------------------------------
# SHA helpers
# ---------------------------------------------------------------------------

class TestSha256Helpers:
    def test_sha256_bytes_known_value(self):
        # echo -n "hello" | sha256sum
        digest = _sha256_bytes(b"hello")
        assert digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_sha256_bytes_empty(self):
        digest = _sha256_bytes(b"")
        assert digest == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_sha256_path(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello")
        assert _sha256_path(f) == _sha256_bytes(b"hello")

    def test_sha256_path_matches_bytes_helper(self, tmp_path):
        content = b"intraknot algorithm content\n" * 100
        f = tmp_path / "script.py"
        f.write_bytes(content)
        assert _sha256_path(f) == _sha256_bytes(content)


# ---------------------------------------------------------------------------
# make_managed_entry
# ---------------------------------------------------------------------------

class TestMakeManagedEntry:
    def test_sets_today_as_installed_at(self):
        entry = make_managed_entry("run_dmrg.py", "intraknot:algorithm/run_dmrg.py", "abc123")
        assert entry.installed_at == datetime.date.today().isoformat()

    def test_fields_preserved(self):
        entry = make_managed_entry(
            file="observables/spin_corr.py",
            source="intraknot-database:observables/spin_corr.py",
            sha256="deadbeef",
            installed_from_version="1.2.3",
        )
        assert entry.file == "observables/spin_corr.py"
        assert entry.source == "intraknot-database:observables/spin_corr.py"
        assert entry.sha256 == "deadbeef"
        assert entry.installed_from_version == "1.2.3"

    def test_no_version_by_default(self):
        entry = make_managed_entry("f.py", "intraknot:algorithm/f.py", "aaa")
        assert entry.installed_from_version is None


# ---------------------------------------------------------------------------
# AlgorithmLock — empty state
# ---------------------------------------------------------------------------

class TestAlgorithmLockEmpty:
    def test_load_missing_file_returns_empty(self, tmp_path):
        lock = AlgorithmLock.load(tmp_path / "nonexistent.lock")
        assert lock.managed == {}
        assert lock.custom == []

    def test_save_and_load_empty(self, tmp_path):
        path = tmp_path / "algorithm" / "algorithm.lock"
        lock = AlgorithmLock()
        lock.save(path)
        loaded = AlgorithmLock.load(path)
        assert loaded.managed == {}
        assert loaded.custom == []

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "algorithm.lock"
        AlgorithmLock().save(path)
        assert path.exists()


# ---------------------------------------------------------------------------
# AlgorithmLock — round-trip serialisation
# ---------------------------------------------------------------------------

class TestAlgorithmLockRoundTrip:
    def _make_entry(self, file: str, source: str, sha: str = "aa" * 32) -> ManagedEntry:
        return make_managed_entry(file, source, sha, installed_from_version="0.1.0")

    def test_single_managed_entry(self, tmp_path):
        path = tmp_path / "algorithm.lock"
        lock = AlgorithmLock()
        lock.add_managed(self._make_entry("run_dmrg.py", "intraknot:algorithm/run_dmrg.py"))
        lock.save(path)

        loaded = AlgorithmLock.load(path)
        assert "run_dmrg.py" in loaded.managed
        entry = loaded.managed["run_dmrg.py"]
        assert entry.source == "intraknot:algorithm/run_dmrg.py"
        assert entry.sha256 == "aa" * 32
        assert entry.installed_from_version == "0.1.0"
        assert entry.installed_at != ""

    def test_subdirectory_file_path(self, tmp_path):
        path = tmp_path / "algorithm.lock"
        lock = AlgorithmLock()
        lock.add_managed(self._make_entry(
            "observables/spin_corr.py",
            "intraknot-database:observables/spin_corr.py",
            sha="bb" * 32,
        ))
        lock.save(path)

        loaded = AlgorithmLock.load(path)
        assert "observables/spin_corr.py" in loaded.managed
        assert loaded.managed["observables/spin_corr.py"].sha256 == "bb" * 32

    def test_custom_files_round_trip(self, tmp_path):
        path = tmp_path / "algorithm.lock"
        lock = AlgorithmLock()
        lock.add_custom("my_obs.py")
        lock.add_custom("post_process/analyse.py")
        lock.save(path)

        loaded = AlgorithmLock.load(path)
        assert "my_obs.py" in loaded.custom
        assert "post_process/analyse.py" in loaded.custom

    def test_mixed_managed_and_custom(self, tmp_path):
        path = tmp_path / "algorithm.lock"
        lock = AlgorithmLock()
        lock.add_managed(self._make_entry("run_dmrg.py", "intraknot:algorithm/run_dmrg.py"))
        lock.add_custom("my_obs.py")
        lock.save(path)

        loaded = AlgorithmLock.load(path)
        assert loaded.is_managed("run_dmrg.py")
        assert loaded.is_custom("my_obs.py")

    def test_no_version_field_not_written(self, tmp_path):
        path = tmp_path / "algorithm.lock"
        lock = AlgorithmLock()
        lock.add_managed(make_managed_entry(
            "run_dmrg.py", "intraknot:algorithm/run_dmrg.py", "aa" * 32
        ))
        lock.save(path)
        text = path.read_text()
        assert "installed_from_version" not in text

    def test_version_field_written_when_set(self, tmp_path):
        path = tmp_path / "algorithm.lock"
        lock = AlgorithmLock()
        lock.add_managed(self._make_entry("run_dmrg.py", "intraknot:algorithm/run_dmrg.py"))
        lock.save(path)
        text = path.read_text()
        assert "installed_from_version" in text
        assert "0.1.0" in text

    def test_multiple_managed_entries(self, tmp_path):
        path = tmp_path / "algorithm.lock"
        lock = AlgorithmLock()
        lock.add_managed(self._make_entry("run_dmrg.py", "intraknot:algorithm/run_dmrg.py"))
        lock.add_managed(self._make_entry(
            "observables/spin_corr.py",
            "intraknot-database:observables/spin_corr.py",
            sha="cc" * 32,
        ))
        lock.save(path)

        loaded = AlgorithmLock.load(path)
        assert len(loaded.managed) == 2
        assert "run_dmrg.py" in loaded.managed
        assert "observables/spin_corr.py" in loaded.managed


# ---------------------------------------------------------------------------
# AlgorithmLock — add_managed
# ---------------------------------------------------------------------------

class TestAddManaged:
    def test_adds_new_entry(self):
        lock = AlgorithmLock()
        entry = make_managed_entry("run_dmrg.py", "intraknot:algorithm/run_dmrg.py", "aa" * 32)
        lock.add_managed(entry)
        assert lock.is_managed("run_dmrg.py")

    def test_replaces_existing_entry(self):
        lock = AlgorithmLock()
        lock.add_managed(make_managed_entry("f.py", "src:f.py", "old" + "a" * 61))
        lock.add_managed(make_managed_entry("f.py", "src:f.py", "new" + "b" * 61))
        assert lock.managed["f.py"].sha256 == "new" + "b" * 61

    def test_removes_from_custom_if_present(self):
        lock = AlgorithmLock()
        lock.add_custom("run_dmrg.py")
        assert lock.is_custom("run_dmrg.py")
        lock.add_managed(make_managed_entry("run_dmrg.py", "src:f.py", "aa" * 32))
        assert not lock.is_custom("run_dmrg.py")
        assert lock.is_managed("run_dmrg.py")


# ---------------------------------------------------------------------------
# AlgorithmLock — promote_to_custom
# ---------------------------------------------------------------------------

class TestPromoteToCustom:
    def test_moves_from_managed_to_custom(self):
        lock = AlgorithmLock()
        lock.add_managed(make_managed_entry("run_dmrg.py", "src:f.py", "aa" * 32))
        lock.promote_to_custom("run_dmrg.py")
        assert not lock.is_managed("run_dmrg.py")
        assert lock.is_custom("run_dmrg.py")

    def test_raises_if_not_managed(self):
        lock = AlgorithmLock()
        with pytest.raises(KeyError, match="not a managed entry"):
            lock.promote_to_custom("nonexistent.py")

    def test_does_not_duplicate_in_custom(self):
        lock = AlgorithmLock()
        lock.add_managed(make_managed_entry("f.py", "src:f.py", "aa" * 32))
        lock.add_custom("f.py")  # pre-add (unusual edge case)
        lock.promote_to_custom("f.py")
        assert lock.custom.count("f.py") == 1


# ---------------------------------------------------------------------------
# AlgorithmLock — add_custom
# ---------------------------------------------------------------------------

class TestAddCustom:
    def test_adds_new_custom(self):
        lock = AlgorithmLock()
        lock.add_custom("my_obs.py")
        assert lock.is_custom("my_obs.py")

    def test_no_op_if_already_present(self):
        lock = AlgorithmLock()
        lock.add_custom("my_obs.py")
        lock.add_custom("my_obs.py")
        assert lock.custom.count("my_obs.py") == 1


# ---------------------------------------------------------------------------
# AlgorithmLock — remove
# ---------------------------------------------------------------------------

class TestRemove:
    def test_removes_managed_entry(self):
        lock = AlgorithmLock()
        lock.add_managed(make_managed_entry("run_dmrg.py", "src:f.py", "aa" * 32))
        lock.remove("run_dmrg.py")
        assert not lock.is_managed("run_dmrg.py")

    def test_removes_custom_entry(self):
        lock = AlgorithmLock()
        lock.add_custom("my_obs.py")
        lock.remove("my_obs.py")
        assert not lock.is_custom("my_obs.py")

    def test_raises_for_untracked_file(self):
        lock = AlgorithmLock()
        with pytest.raises(KeyError, match="not tracked"):
            lock.remove("nonexistent.py")

    def test_removed_entry_absent_from_saved_lock(self, tmp_path):
        path = tmp_path / "algorithm.lock"
        lock = AlgorithmLock()
        lock.add_managed(make_managed_entry("spin_corr.py", "db:spin_corr.py", "aa" * 32))
        lock.add_custom("my_obs.py")
        lock.remove("spin_corr.py")
        lock.save(path)
        loaded = AlgorithmLock.load(path)
        assert not loaded.is_managed("spin_corr.py")
        assert loaded.is_custom("my_obs.py")

    def test_other_entries_unaffected(self):
        lock = AlgorithmLock()
        lock.add_managed(make_managed_entry("a.py", "src:a.py", "aa" * 32))
        lock.add_managed(make_managed_entry("b.py", "src:b.py", "bb" * 32))
        lock.remove("a.py")
        assert not lock.is_managed("a.py")
        assert lock.is_managed("b.py")


# ---------------------------------------------------------------------------
# AlgorithmLock — check_modified
# ---------------------------------------------------------------------------

class TestCheckModified:
    def _make_lock_with_file(self, alg_dir: Path, content: bytes) -> AlgorithmLock:
        """Write a file and build a lock whose sha256 matches the content."""
        (alg_dir / "run_dmrg.py").write_bytes(content)
        sha = _sha256_bytes(content)
        lock = AlgorithmLock()
        lock.add_managed(make_managed_entry("run_dmrg.py", "intraknot:f.py", sha))
        return lock

    def test_unmodified_file_not_reported(self, tmp_path):
        alg_dir = tmp_path / "algorithm"
        alg_dir.mkdir()
        content = b"# script\n"
        lock = self._make_lock_with_file(alg_dir, content)
        assert lock.check_modified(alg_dir) == []

    def test_modified_file_reported(self, tmp_path):
        alg_dir = tmp_path / "algorithm"
        alg_dir.mkdir()
        lock = self._make_lock_with_file(alg_dir, b"# original\n")
        # Modify the file after building the lock.
        (alg_dir / "run_dmrg.py").write_bytes(b"# modified\n")
        modified = lock.check_modified(alg_dir)
        assert "run_dmrg.py" in modified

    def test_missing_file_silently_skipped(self, tmp_path):
        alg_dir = tmp_path / "algorithm"
        alg_dir.mkdir()
        lock = AlgorithmLock()
        lock.add_managed(make_managed_entry("absent.py", "src:f.py", "aa" * 32))
        # No file on disk — should not raise, should not appear in results.
        assert lock.check_modified(alg_dir) == []

    def test_custom_files_ignored(self, tmp_path):
        alg_dir = tmp_path / "algorithm"
        alg_dir.mkdir()
        (alg_dir / "my_obs.py").write_bytes(b"# custom\n")
        lock = AlgorithmLock()
        lock.add_custom("my_obs.py")
        assert lock.check_modified(alg_dir) == []

    def test_subdirectory_file(self, tmp_path):
        alg_dir = tmp_path / "algorithm"
        (alg_dir / "observables").mkdir(parents=True)
        content = b"# obs\n"
        (alg_dir / "observables" / "spin_corr.py").write_bytes(content)
        sha = _sha256_bytes(content)
        lock = AlgorithmLock()
        lock.add_managed(make_managed_entry(
            "observables/spin_corr.py", "db:observables/spin_corr.py", sha
        ))
        assert lock.check_modified(alg_dir) == []

        # Modify the file.
        (alg_dir / "observables" / "spin_corr.py").write_bytes(b"# changed\n")
        assert lock.check_modified(alg_dir) == ["observables/spin_corr.py"]


# ---------------------------------------------------------------------------
# AlgorithmLock — is_managed / is_custom
# ---------------------------------------------------------------------------

class TestIsQueries:
    def test_is_managed_true(self):
        lock = AlgorithmLock()
        lock.add_managed(make_managed_entry("f.py", "src:f.py", "aa" * 32))
        assert lock.is_managed("f.py") is True

    def test_is_managed_false(self):
        assert AlgorithmLock().is_managed("nope.py") is False

    def test_is_custom_true(self):
        lock = AlgorithmLock()
        lock.add_custom("my.py")
        assert lock.is_custom("my.py") is True

    def test_is_custom_false(self):
        assert AlgorithmLock().is_custom("nope.py") is False
