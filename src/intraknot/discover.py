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


"""Cluster topology discovery via `sinfo`.

This module queries the local Slurm installation using two `sinfo` calls and
builds a `ClusterDiscovery` snapshot that is saved to `configs/cluster.yaml`.

The discovery is deliberately static: it records hardware specs and partition
metadata but not live node states (idle/alloc counts), which go stale quickly
and are not useful for reference documentation.

Grouping strategy
-----------------
Nodes are first bucketed by name prefix (the part of the node name before any
trailing digits or bracket range, e.g. `th-cl-hua` from `th-cl-hua[01-29]`).
Within each prefix bucket, nodes are further split by `(cpus, mem_mb, gres)`.
This ensures that every `NodeGroup` has uniform hardware specs. The `nodes`
field of each group is the node list compressed back to Slurm bracket notation
(e.g. `"th-cl-hua[01-10]"`).
"""

from __future__ import annotations

import re
import socket
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .config import ClusterDiscovery, FeatureInfo, NodeGroup, PartitionInfo


# ---------------------------------------------------------------------------
# sinfo subprocess wrappers
# ---------------------------------------------------------------------------

def _run_sinfo(args: List[str]) -> str:
    """Run `sinfo` with *args* and return stdout as a string.

    Raises `RuntimeError` when `sinfo` is not found or exits non-zero.
    """
    cmd = ["sinfo"] + args
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "`sinfo` not found. Is this machine part of a Slurm cluster?"
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"`sinfo` exited with code {proc.returncode}.\n"
            f"stderr: {proc.stderr.strip()}"
        )
    return proc.stdout


def _sinfo_partitions() -> List[Dict]:
    """Query partition-level info (state, time limit, default flag).

    Returns a list of dicts with keys: `name`, `state`, `default`,
    `time_limit`. Each partition appears once (the `--summarize` flag
    collapses per-state rows).
    """
    # %P includes a trailing '*' on the default partition.
    out = _run_sinfo(["--noheader", "--summarize", "--format=%P|%a|%l"])
    rows: List[Dict] = []
    seen: set = set()
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        raw_name, state, time_limit = parts
        is_default = raw_name.endswith("*")
        name = raw_name.rstrip("*")
        # sinfo --summarize may still emit duplicate rows for drained subsets.
        if name in seen:
            continue
        seen.add(name)
        rows.append({
            "name": name,
            "state": state.strip().lower(),
            "default": is_default,
            "time_limit": time_limit.strip(),
        })
    return rows


def _sinfo_nodes() -> List[Dict]:
    """Query per-node hardware specs.

    Returns a list of dicts with keys: `node`, `partition`, `cpus`,
    `mem_mb`, `gres`, `features`. One dict per (node, partition) pair
    since a node can belong to multiple partitions.
    """
    out = _run_sinfo(["--noheader", "-N", "--format=%n|%P|%c|%m|%G|%f"])
    rows: List[Dict] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 5)
        if len(parts) < 6:
            continue
        node, partition, cpus_str, mem_str, gres, features_str = parts
        partition = partition.strip().rstrip("*")

        # cpus can be reported as "N/A" on heterogeneous nodes; skip those.
        try:
            cpus = int(cpus_str.strip())
        except ValueError:
            continue

        # mem is in MiB; may contain a unit suffix on some Slurm versions.
        mem_str = mem_str.strip()
        try:
            mem_mb = int(re.sub(r"[^0-9]", "", mem_str))
        except ValueError:
            mem_mb = 0

        # Normalise GRES: "(null)" → "".
        gres = gres.strip()
        if gres.lower() in ("(null)", "null", "n/a", ""):
            gres = ""

        # Features: comma-separated list or "(null)".
        features_str = features_str.strip()
        if features_str.lower() in ("(null)", "null", ""):
            features: List[str] = []
        else:
            features = [f.strip() for f in features_str.split(",") if f.strip()]

        rows.append({
            "node": node.strip(),
            "partition": partition,
            "cpus": cpus,
            "mem_mb": mem_mb,
            "gres": gres,
            "features": features,
        })
    return rows


# ---------------------------------------------------------------------------
# Node name prefix extraction
# ---------------------------------------------------------------------------

def _node_prefix(name: str) -> str:
    """Return the name prefix by stripping any trailing bracket range or digits.

    Examples
    --------
    `"th-cl-hua[01-29]"` → `"th-cl-hua"`
    `"th-cl-rome01n[1-4]"` → `"th-cl-rome01n"`
    `"node42"` → `"node"`
    `"cip-cl-computea"` → `"cip-cl-compute"`

    The rule is: strip a trailing `[...]` bracket range if present, then
    strip any remaining trailing alphanumeric suffix that consists only of
    digits or a single trailing letter (common for lettered node variants
    like `computea`, `computeb`).
    """
    # Strip bracket range.
    name = re.sub(r"\[.*?\]$", "", name)
    # Strip trailing digits.
    name = re.sub(r"\d+$", "", name)
    # Strip a single trailing letter that is a variant suffix (e.g. 'a' in
    # 'computea'). Only strip if what remains is non-empty and ends with a
    # non-letter to avoid stripping meaningful parts of hostnames that end
    # in a letter (e.g. 'rome01n' — the 'n' is structural, but we already
    # stripped the numeric part above so 'rome01n' becomes 'rome01n' here
    # after digit-stripping removes nothing since the name ends in 'n').
    # We apply this only when the last char is a lowercase letter and the
    # penultimate char is a digit.
    if len(name) >= 2 and name[-1].isalpha() and name[-2].isdigit():
        name = name[:-1]
    return name if name else "node"


# ---------------------------------------------------------------------------
# Slurm bracket notation compression
# ---------------------------------------------------------------------------

def _compress_nodelist(names: List[str]) -> str:
    """Compress a flat list of node names to Slurm bracket notation.

    Names that share a common prefix followed by a zero-padded (or
    unpadded) integer suffix are grouped into `prefix[range]` form, where
    consecutive runs are expressed as `start-end` and non-consecutive
    entries are comma-separated inside the brackets.

    Single-node lists are returned as-is (no brackets). Lists containing
    names with no numeric suffix are returned joined by commas.

    Parameters
    ----------
    names:
        Flat list of node names, e.g. `["node01", "node02", "node03",
        "node05"]`.

    Returns
    -------
    str
        Compressed form, e.g. `"node[01-03,05]"`.
    """
    if not names:
        return ""
    if len(names) == 1:
        return names[0]

    # Group by (prefix, zero_pad_width).
    # prefix: everything before the trailing integer
    # width: number of digits (for zero-padding); 0 means no padding.
    _SUFFIX_RE = re.compile(r"^(.*?)(\d+)$")

    # Attempt to parse all names under the assumption they share one prefix.
    parsed: List[Tuple[str, int, int]] = []  # (prefix, width, number)
    for n in names:
        m = _SUFFIX_RE.match(n)
        if m:
            prefix, digits = m.group(1), m.group(2)
            parsed.append((prefix, len(digits), int(digits)))
        else:
            # Cannot compress — fall back to comma join.
            return ",".join(names)

    # Check all share the same prefix.
    prefixes = {p for p, _, _ in parsed}
    if len(prefixes) != 1:
        # Mixed prefixes — not compressible as a single group.
        return ",".join(names)

    prefix = prefixes.pop()
    # Use the width of the first entry (assume uniform zero-padding).
    width = parsed[0][1]
    numbers = sorted(n for _, _, n in parsed)

    # Build ranges.
    ranges: List[str] = []
    start = numbers[0]
    end = numbers[0]
    for n in numbers[1:]:
        if n == end + 1:
            end = n
        else:
            ranges.append(_fmt_range(start, end, width))
            start = end = n
    ranges.append(_fmt_range(start, end, width))

    if len(numbers) == 1:
        return f"{prefix}{numbers[0]:0{width}d}" if width else f"{prefix}{numbers[0]}"
    return f"{prefix}[{','.join(ranges)}]"


def _fmt_range(start: int, end: int, width: int) -> str:
    """Format a single numeric range segment for bracket notation."""
    if width:
        s = f"{start:0{width}d}"
        e = f"{end:0{width}d}"
    else:
        s, e = str(start), str(end)
    return s if start == end else f"{s}-{e}"


# ---------------------------------------------------------------------------
# Node grouping
# ---------------------------------------------------------------------------

def _group_nodes_for_partition(
    node_rows: List[Dict],
) -> List[NodeGroup]:
    """Group nodes into hardware-uniform `NodeGroup` objects.

    Algorithm:
    1. Bucket nodes by name prefix.
    2. Within each prefix bucket, sub-group by `(cpus, mem_mb, gres)`.
    3. Compress each sub-group's node list to bracket notation.
    4. Collect the union of feature labels across all nodes in the sub-group.

    Parameters
    ----------
    node_rows:
        Per-node dicts as returned by `_sinfo_nodes()`, pre-filtered for one
        partition.

    Returns
    -------
    list[NodeGroup]
        Hardware-uniform groups, ordered by prefix then by (cpus, mem, gres).
    """
    # Bucket by prefix.
    by_prefix: Dict[str, List[Dict]] = defaultdict(list)
    for row in node_rows:
        by_prefix[_node_prefix(row["node"])].append(row)

    groups: List[NodeGroup] = []
    for prefix in sorted(by_prefix):
        rows = by_prefix[prefix]
        # Sub-group by hardware spec.
        by_spec: Dict[Tuple, List[Dict]] = defaultdict(list)
        for row in rows:
            spec = (row["cpus"], row["mem_mb"], row["gres"])
            by_spec[spec].append(row)
        for spec in sorted(by_spec):
            spec_rows = by_spec[spec]
            cpus, mem_mb, gres = spec
            node_names = [r["node"] for r in spec_rows]
            # Feature union across the sub-group.
            feature_set: List[str] = []
            seen_features: set = set()
            for r in spec_rows:
                for f in r["features"]:
                    if f not in seen_features:
                        seen_features.add(f)
                        feature_set.append(f)
            groups.append(NodeGroup(
                nodes=_compress_nodelist(sorted(node_names)),
                cpus=cpus,
                mem_mb=mem_mb,
                gres=gres,
                features=feature_set,
            ))
    return groups


# ---------------------------------------------------------------------------
# Feature index
# ---------------------------------------------------------------------------

def _build_feature_index(
    partitions: List[PartitionInfo],
) -> Dict[str, FeatureInfo]:
    """Derive the cross-partition feature/constraint index.

    For each feature label that appears on at least one node group, record
    which partitions it is present in and the total node count across those
    partitions.

    Parameters
    ----------
    partitions:
        Discovered partition list.

    Returns
    -------
    dict
        Mapping from feature name to `FeatureInfo`.
    """
    # feature → {partition_name → node_count}
    feature_partitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for part in partitions:
        for group in part.node_groups:
            # Count nodes in this group from the bracket notation.
            n = _count_nodes_in_bracket(group.nodes)
            for feat in group.features:
                feature_partitions[feat][part.name] += n

    index: Dict[str, FeatureInfo] = {}
    for feat in sorted(feature_partitions):
        part_map = feature_partitions[feat]
        index[feat] = FeatureInfo(
            partitions=sorted(part_map.keys()),
            node_count=sum(part_map.values()),
        )
    return index


def _count_nodes_in_bracket(nodelist: str) -> int:
    """Count the number of nodes described by a Slurm bracket notation string.

    Examples: `"node01"` → 1, `"node[01-05]"` → 5,
    `"node[01-03,07]"` → 4.
    """
    m = re.match(r"^[^\[]+\[([^\]]+)\]$", nodelist)
    if not m:
        # Single node or plain name.
        return 1
    total = 0
    for segment in m.group(1).split(","):
        segment = segment.strip()
        if "-" in segment:
            parts = segment.split("-", 1)
            try:
                total += int(parts[1]) - int(parts[0]) + 1
            except ValueError:
                total += 1
        else:
            total += 1
    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_cluster() -> ClusterDiscovery:
    """Run `sinfo` and build a full `ClusterDiscovery` snapshot.

    Queries the Slurm scheduler for partition metadata and per-node hardware
    specs, groups nodes into hardware-uniform `NodeGroup` objects, and derives
    the top-level feature/constraint index.

    Returns
    -------
    ClusterDiscovery
        Populated discovery snapshot ready to be saved with `save_discovery`.

    Raises
    ------
    RuntimeError
        When `sinfo` is not found or exits with a non-zero return code.
    """
    hostname = socket.getfqdn()
    discovered_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    part_rows = _sinfo_partitions()
    node_rows = _sinfo_nodes()

    # Index node rows by partition for efficient lookup.
    nodes_by_partition: Dict[str, List[Dict]] = defaultdict(list)
    for row in node_rows:
        nodes_by_partition[row["partition"]].append(row)

    partitions: List[PartitionInfo] = []
    for pr in part_rows:
        pname = pr["name"]
        node_groups = _group_nodes_for_partition(nodes_by_partition.get(pname, []))
        partitions.append(PartitionInfo(
            name=pname,
            state=pr["state"],
            default=pr["default"],
            time_limit=pr["time_limit"],
            node_groups=node_groups,
        ))

    features = _build_feature_index(partitions)

    return ClusterDiscovery(
        hostname=hostname,
        discovered_at=discovered_at,
        partitions=partitions,
        features=features,
    )


def save_discovery(discovery: ClusterDiscovery, configs_dir: Path) -> None:
    """Serialise *discovery* to `configs_dir/cluster.yaml`.

    The output file is prefixed with a comment warning that it is
    auto-generated. The parent directory is created if absent.

    Parameters
    ----------
    discovery:
        `ClusterDiscovery` snapshot to serialise.
    configs_dir:
        Directory to write `cluster.yaml` into.
    """
    configs_dir.mkdir(parents=True, exist_ok=True)
    path = configs_dir / "cluster.yaml"

    data: Dict = {
        "discovered_at": discovery.discovered_at,
        "hostname": discovery.hostname,
        "partitions": [
            {
                "name": p.name,
                "state": p.state,
                "default": p.default,
                "time_limit": p.time_limit,
                "node_groups": [
                    {
                        "nodes": g.nodes,
                        "cpus": g.cpus,
                        "mem_mb": g.mem_mb,
                        "gres": g.gres,
                        "features": g.features,
                    }
                    for g in p.node_groups
                ],
            }
            for p in discovery.partitions
        ],
        "features": {
            name: {
                "partitions": fi.partitions,
                "node_count": fi.node_count,
            }
            for name, fi in discovery.features.items()
        },
    }

    header = (
        "# Auto-generated by `iknot cluster sync`. Do not edit manually.\n"
        "# Re-run `iknot cluster sync` to refresh after cluster changes.\n"
    )
    path.write_text(header + yaml.dump(data, default_flow_style=False, sort_keys=False))


def load_discovery(configs_dir: Path) -> Optional[ClusterDiscovery]:
    """Load `cluster.yaml` from *configs_dir*, or return `None` if absent.

    This is a thin wrapper around `config.load_cluster_discovery` kept here
    so that callers only need to import from `discover`.

    Parameters
    ----------
    configs_dir:
        Directory containing `cluster.yaml`.

    Returns
    -------
    ClusterDiscovery or None
    """
    from .config import load_cluster_discovery
    return load_cluster_discovery(configs_dir)
