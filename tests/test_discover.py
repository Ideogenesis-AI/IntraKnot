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


"""Tests for src/intraknot/discover.py.

All tests are pure unit tests: none of them call real `sinfo` processes.
The sinfo-parsing functions are exercised through text fixtures that match
the format produced by the actual `sinfo` command.
"""

import textwrap
from unittest.mock import patch

import pytest

from intraknot.config import ClusterDiscovery, FeatureInfo, NodeGroup, PartitionInfo
from intraknot.discover import (
    _build_feature_index,
    _compress_nodelist,
    _count_nodes_in_bracket,
    _fmt_range,
    _group_nodes_for_partition,
    _node_prefix,
    _sinfo_nodes,
    _sinfo_partitions,
    load_discovery,
    save_discovery,
)


# ---------------------------------------------------------------------------
# _node_prefix
# ---------------------------------------------------------------------------

class TestNodePrefix:
    def test_bracket_range(self):
        assert _node_prefix("th-cl-hua[01-29]") == "th-cl-hua"

    def test_bracket_range_with_inner_numbers(self):
        # 'th-cl-rome01n[1-4]': strip brackets → 'th-cl-rome01n',
        # then strip trailing digit sequence → 'th-cl-rome0' ... wait,
        # 'rome01n' ends in 'n' (letter after digit) so strip the 'n' too.
        # Result: 'th-cl-rome01' then strip '01' → 'th-cl-rome'.
        # Actually 'rome01n' → strip digits → 'rome01n' (ends in 'n') →
        # strip single trailing letter after digit → 'rome01' → strip digits
        # Wait, the function only strips once. Let me re-read the logic:
        # step1: strip [1-4] → 'th-cl-rome01n'
        # step2: strip trailing digits → 'th-cl-rome01n' (no trailing digits!)
        # step3: last char 'n' is alpha, penultimate '1' is digit → strip 'n'
        # result: 'th-cl-rome01'
        # That is the expected prefix for this node family.
        assert _node_prefix("th-cl-rome01n[1-4]") == "th-cl-rome01"

    def test_plain_digits(self):
        assert _node_prefix("node42") == "node"

    def test_letter_suffix(self):
        # 'cip-cl-computea': no brackets, no trailing digits, trailing letter
        # after digit? 'a' is alpha, 'e' before it is also alpha → NOT stripped.
        # 'computea' ends in 'a' (alpha), but preceding char is 'e' (alpha),
        # so the strip-single-letter rule does NOT apply.
        assert _node_prefix("cip-cl-computea") == "cip-cl-computea"

    def test_lettered_variant_after_number(self):
        # e.g. 'compute48b': trailing 'b' (alpha) after '8' (digit) → strip 'b',
        # then strip digits '48' → 'compute'.
        # But _node_prefix only strips once, so:
        # step1: no brackets → 'compute48b'
        # step2: strip trailing digits → 'compute48b' (ends in 'b')
        # step3: 'b' is alpha, '8' is digit → strip 'b' → 'compute48'
        # The function does NOT recursively strip digits again.
        assert _node_prefix("compute48b") == "compute48"

    def test_no_suffix(self):
        assert _node_prefix("loginnode") == "loginnode"

    def test_empty_result_falls_back(self):
        # A purely numeric name: '042' → strip digits → '' → fallback 'node'.
        assert _node_prefix("042") == "node"


# ---------------------------------------------------------------------------
# _compress_nodelist
# ---------------------------------------------------------------------------

class TestCompressNodelist:
    def test_single_node(self):
        assert _compress_nodelist(["node01"]) == "node01"

    def test_consecutive_range(self):
        names = [f"node{i:02d}" for i in range(1, 6)]
        assert _compress_nodelist(names) == "node[01-05]"

    def test_non_consecutive(self):
        names = ["node01", "node02", "node03", "node05"]
        assert _compress_nodelist(names) == "node[01-03,05]"

    def test_single_gap(self):
        names = ["node01", "node03"]
        assert _compress_nodelist(names) == "node[01,03]"

    def test_no_numeric_suffix_fallback(self):
        # Names without trailing digits cannot be compressed.
        names = ["alpha", "beta"]
        assert _compress_nodelist(names) == "alpha,beta"

    def test_mixed_prefixes_fallback(self):
        # Different prefixes → fall back to comma-join.
        names = ["th-node01", "met-node01"]
        assert _compress_nodelist(names) == "th-node01,met-node01"

    def test_empty_list(self):
        assert _compress_nodelist([]) == ""

    def test_unpadded_numbers(self):
        names = ["node1", "node2", "node3"]
        assert _compress_nodelist(names) == "node[1-3]"

    def test_order_independence(self):
        # Nodes passed in reverse order should still compress correctly.
        names = ["node05", "node03", "node01", "node02", "node04"]
        assert _compress_nodelist(names) == "node[01-05]"


# ---------------------------------------------------------------------------
# _count_nodes_in_bracket
# ---------------------------------------------------------------------------

class TestCountNodesInBracket:
    def test_single(self):
        assert _count_nodes_in_bracket("node01") == 1

    def test_range(self):
        assert _count_nodes_in_bracket("node[01-05]") == 5

    def test_multi_segment(self):
        assert _count_nodes_in_bracket("node[01-03,07]") == 4

    def test_disjoint_ranges(self):
        assert _count_nodes_in_bracket("node[01-03,10-12]") == 6


# ---------------------------------------------------------------------------
# _group_nodes_for_partition
# ---------------------------------------------------------------------------

class TestGroupNodesForPartition:
    def _make_row(self, node, cpus=32, mem_mb=128000, gres="", features=None):
        return {
            "node": node,
            "partition": "cpu",
            "cpus": cpus,
            "mem_mb": mem_mb,
            "gres": gres,
            "features": features or [],
        }

    def test_uniform_prefix(self):
        rows = [self._make_row(f"node{i:02d}", features=["epyc"]) for i in range(1, 4)]
        groups = _group_nodes_for_partition(rows)
        assert len(groups) == 1
        g = groups[0]
        assert g.nodes == "node[01-03]"
        assert g.cpus == 32
        assert g.features == ["epyc"]

    def test_split_on_different_mem(self):
        rows = (
            [self._make_row(f"node{i:02d}", mem_mb=128000) for i in range(1, 4)]
            + [self._make_row(f"node{i:02d}", mem_mb=256000) for i in range(4, 7)]
        )
        groups = _group_nodes_for_partition(rows)
        # Same prefix 'node' → split by mem → two groups.
        assert len(groups) == 2
        mems = {g.mem_mb for g in groups}
        assert mems == {128000, 256000}

    def test_feature_union_within_group(self):
        # Different features on nodes with same specs → union in the group.
        rows = [
            self._make_row("node01", features=["epyc"]),
            self._make_row("node02", features=["epyc", "avx512"]),
        ]
        groups = _group_nodes_for_partition(rows)
        assert len(groups) == 1
        assert set(groups[0].features) == {"epyc", "avx512"}

    def test_different_prefixes(self):
        rows = (
            [self._make_row(f"alpha{i:02d}") for i in range(1, 3)]
            + [self._make_row(f"beta{i:02d}") for i in range(1, 3)]
        )
        groups = _group_nodes_for_partition(rows)
        prefixes = {g.nodes.split("[")[0] for g in groups}
        assert "alpha" in prefixes
        assert "beta" in prefixes

    def test_empty_input(self):
        assert _group_nodes_for_partition([]) == []


# ---------------------------------------------------------------------------
# _build_feature_index
# ---------------------------------------------------------------------------

class TestBuildFeatureIndex:
    def _make_partition(self, name, groups):
        return PartitionInfo(
            name=name,
            state="up",
            default=False,
            time_limit="7-00:00:00",
            node_groups=groups,
        )

    def test_single_partition(self):
        g = NodeGroup(nodes="node[01-04]", cpus=32, mem_mb=128000, gres="",
                      features=["epyc"])
        p = self._make_partition("cpu", [g])
        idx = _build_feature_index([p])
        assert "epyc" in idx
        assert idx["epyc"].partitions == ["cpu"]
        assert idx["epyc"].node_count == 4

    def test_shared_feature_across_partitions(self):
        g1 = NodeGroup(nodes="node[01-04]", cpus=64, mem_mb=256000, gres="",
                       features=["epyc"])
        g2 = NodeGroup(nodes="node[05-06]", cpus=64, mem_mb=256000, gres="",
                       features=["epyc"])
        p1 = self._make_partition("cluster", [g1])
        p2 = self._make_partition("small", [g2])
        idx = _build_feature_index([p1, p2])
        assert set(idx["epyc"].partitions) == {"cluster", "small"}
        assert idx["epyc"].node_count == 6

    def test_no_features(self):
        g = NodeGroup(nodes="node01", cpus=32, mem_mb=128000, gres="", features=[])
        p = self._make_partition("cpu", [g])
        idx = _build_feature_index([p])
        assert idx == {}

    def test_index_sorted_alphabetically(self):
        g1 = NodeGroup(nodes="node[01-02]", cpus=32, mem_mb=128000, gres="",
                       features=["rome"])
        g2 = NodeGroup(nodes="node[03-04]", cpus=64, mem_mb=256000, gres="",
                       features=["epyc"])
        p = self._make_partition("cpu", [g1, g2])
        idx = _build_feature_index([p])
        assert list(idx.keys()) == sorted(idx.keys())


# ---------------------------------------------------------------------------
# save_discovery / load_discovery round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip:
    def _make_discovery(self):
        g = NodeGroup(
            nodes="th-cl-hua[01-10]",
            cpus=32,
            mem_mb=128000,
            gres="",
            features=["cascade"],
        )
        p = PartitionInfo(
            name="cluster",
            state="up",
            default=True,
            time_limit="21-00:00:00",
            node_groups=[g],
        )
        fi = FeatureInfo(partitions=["cluster"], node_count=10)
        return ClusterDiscovery(
            hostname="login.example.com",
            discovered_at="2026-05-26T01:53:00+00:00",
            partitions=[p],
            features={"cascade": fi},
        )

    def test_roundtrip(self, tmp_path):
        original = self._make_discovery()
        save_discovery(original, tmp_path)
        loaded = load_discovery(tmp_path)
        assert loaded is not None
        assert loaded.hostname == original.hostname
        assert loaded.discovered_at == original.discovered_at
        assert len(loaded.partitions) == 1
        lp = loaded.partitions[0]
        op = original.partitions[0]
        assert lp.name == op.name
        assert lp.state == op.state
        assert lp.default == op.default
        assert lp.time_limit == op.time_limit
        assert len(lp.node_groups) == 1
        lg = lp.node_groups[0]
        og = op.node_groups[0]
        assert lg.nodes == og.nodes
        assert lg.cpus == og.cpus
        assert lg.mem_mb == og.mem_mb
        assert lg.gres == og.gres
        assert lg.features == og.features
        assert loaded.features["cascade"].node_count == 10
        assert loaded.features["cascade"].partitions == ["cluster"]

    def test_cluster_yaml_has_header_comment(self, tmp_path):
        save_discovery(self._make_discovery(), tmp_path)
        text = (tmp_path / "cluster.yaml").read_text()
        assert "Auto-generated" in text
        assert "iknot cluster sync" in text

    def test_load_returns_none_when_absent(self, tmp_path):
        assert load_discovery(tmp_path) is None


# ---------------------------------------------------------------------------
# sinfo text-format parsing
# ---------------------------------------------------------------------------

# Fixture matching the format from a real cluster (--summarize, pipe-delimited).
_SINFO_PARTITION_FIXTURE = textwrap.dedent("""\
    cip|up|2-00:00:00
    cip-ws|up|12:00:00
    inter|up|2-00:00:00
    th-ws|up|3-00:00:00
    small|up|21-00:00:0
    cluster*|up|21-00:00:0
""")

# Fixture matching --format="%n|%P|%c|%m|%G|%f" for a few representative nodes.
_SINFO_NODES_FIXTURE = textwrap.dedent("""\
    th-cl-hua01|cluster|64|256000|gpu:a100:4|cascade,avx512
    th-cl-hua02|cluster|64|256000|gpu:a100:4|cascade,avx512
    th-cl-hua03|cluster|64|128000|(null)|(null)
    th-cl-rome01n1|cluster|128|512000||rome,epyc
    th-cl-rome01n1|small|128|512000||rome,epyc
""")


class TestSinfoPartitionParsing:
    def test_parses_partition_rows(self):
        with patch("intraknot.discover._run_sinfo", return_value=_SINFO_PARTITION_FIXTURE):
            rows = _sinfo_partitions()
        names = [r["name"] for r in rows]
        assert "cip" in names
        assert "cluster" in names

    def test_default_partition_flagged(self):
        with patch("intraknot.discover._run_sinfo", return_value=_SINFO_PARTITION_FIXTURE):
            rows = _sinfo_partitions()
        cluster_row = next(r for r in rows if r["name"] == "cluster")
        assert cluster_row["default"] is True

    def test_non_default_not_flagged(self):
        with patch("intraknot.discover._run_sinfo", return_value=_SINFO_PARTITION_FIXTURE):
            rows = _sinfo_partitions()
        cip_row = next(r for r in rows if r["name"] == "cip")
        assert cip_row["default"] is False

    def test_time_limit_preserved(self):
        with patch("intraknot.discover._run_sinfo", return_value=_SINFO_PARTITION_FIXTURE):
            rows = _sinfo_partitions()
        ws_row = next(r for r in rows if r["name"] == "cip-ws")
        assert ws_row["time_limit"] == "12:00:00"

    def test_no_duplicate_partitions(self):
        # Even if sinfo emits the same partition multiple times (different
        # states), each should appear only once.
        duplicated = _SINFO_PARTITION_FIXTURE + "cluster*|up|21-00:00:0\n"
        with patch("intraknot.discover._run_sinfo", return_value=duplicated):
            rows = _sinfo_partitions()
        cluster_rows = [r for r in rows if r["name"] == "cluster"]
        assert len(cluster_rows) == 1


class TestSinfoNodeParsing:
    def test_parses_node_rows(self):
        with patch("intraknot.discover._run_sinfo", return_value=_SINFO_NODES_FIXTURE):
            rows = _sinfo_nodes()
        node_names = [r["node"] for r in rows]
        assert "th-cl-hua01" in node_names
        assert "th-cl-rome01n1" in node_names

    def test_null_gres_normalised(self):
        with patch("intraknot.discover._run_sinfo", return_value=_SINFO_NODES_FIXTURE):
            rows = _sinfo_nodes()
        hua03 = next(r for r in rows if r["node"] == "th-cl-hua03")
        assert hua03["gres"] == ""

    def test_null_features_normalised(self):
        with patch("intraknot.discover._run_sinfo", return_value=_SINFO_NODES_FIXTURE):
            rows = _sinfo_nodes()
        hua03 = next(r for r in rows if r["node"] == "th-cl-hua03")
        assert hua03["features"] == []

    def test_features_parsed_as_list(self):
        with patch("intraknot.discover._run_sinfo", return_value=_SINFO_NODES_FIXTURE):
            rows = _sinfo_nodes()
        hua01 = next(r for r in rows if r["node"] == "th-cl-hua01")
        assert "cascade" in hua01["features"]
        assert "avx512" in hua01["features"]

    def test_node_in_multiple_partitions(self):
        # th-cl-rome01n1 appears in both 'cluster' and 'small'.
        with patch("intraknot.discover._run_sinfo", return_value=_SINFO_NODES_FIXTURE):
            rows = _sinfo_nodes()
        rome_rows = [r for r in rows if r["node"] == "th-cl-rome01n1"]
        partitions = {r["partition"] for r in rome_rows}
        assert partitions == {"cluster", "small"}

    def test_cpus_parsed_as_int(self):
        with patch("intraknot.discover._run_sinfo", return_value=_SINFO_NODES_FIXTURE):
            rows = _sinfo_nodes()
        hua01 = next(r for r in rows if r["node"] == "th-cl-hua01")
        assert isinstance(hua01["cpus"], int)
        assert hua01["cpus"] == 64
