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


"""IntraKnot command-line interface.

Entry point: `iknot` (configured in `[project.scripts]`).

Campaign session
----------------
The active campaign is resolved in this order:

1. `INTRAKNOT_CAMPAIGN` environment variable.
2. `active_campaign` key in `.iknot_state` (TOML file at the project root).
3. `None` — no active campaign.

`iknot campaign activate <id>` writes to `.iknot_state` and prints the
corresponding `export` command so users can optionally source it in their
shell.  `iknot campaign deactivate` clears both.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Optional

import click
import yaml

from .collect import collect_campaign, collect_run
from .config import (
    MachineConfig,
    iter_scan_combinations,
    load_campaign_defaults,
    load_machine_config,
    make_run_id,
    write_data_gitignore,
    write_machines_yaml,
    write_paths_toml,
    write_slurm_toml,
)
from .launch import (
    create_attempt,
    create_campaign,
    create_run,
    delete_run,
    prepare_exec,
    read_runs_by_filter,
    remove_run_from_campaign,
    submit_exec_job,
    submit_job,
    write_exec_slurm_script,
    write_slurm_script,
)
from .resume import find_resumable_runs, is_resumable, resume_campaign, resume_run
from .status import read_status, MainStatus


# ---------------------------------------------------------------------------
# State file helpers
# ---------------------------------------------------------------------------

_STATE_FILE = ".iknot_state"


def _state_file_path() -> Path:
    return Path.cwd() / _STATE_FILE


def _read_state() -> dict:
    p = _state_file_path()
    if not p.exists():
        return {}
    try:
        with open(p, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _write_state(data: dict) -> None:
    p = _state_file_path()
    lines = [f'{k} = "{v}"\n' for k, v in data.items()]
    p.write_text("".join(lines))


def _resolve_active_campaign() -> tuple[Optional[str], str]:
    """Return `(campaign_id, source)` for the currently active campaign.

    The source string is one of `"env"`, `"file"`, or `"none"`.

    Returns `(None, "none")` when no campaign is active.
    """
    env_val = os.environ.get("INTRAKNOT_CAMPAIGN")
    if env_val:
        return env_val, "env"
    state = _read_state()
    if "active_campaign" in state:
        return state["active_campaign"], "file"
    return None, "none"


# ---------------------------------------------------------------------------
# Machine config helper
# ---------------------------------------------------------------------------

def _is_intraknot_source_project(cwd: Path) -> bool:
    """Return True if *cwd* is the IntraKnot source repository.

    Detected by the presence of a `pyproject.toml` at *cwd* whose
    `[project]` table has `name = "intraknot"`.
    """
    toml_path = cwd / "pyproject.toml"
    if not toml_path.exists():
        return False
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        return data.get("project", {}).get("name") == "intraknot"
    except Exception:
        return False


def _load_machine(machine_opt: Optional[str]) -> MachineConfig:
    configs_dir = Path(machine_opt) if machine_opt else Path.cwd() / "configs"
    return load_machine_config(configs_dir)


def _resolve_campaign_dir(
    run_dir: Path,
    campaign_id: Optional[str],
    campaigns_root: str,
) -> Optional[Path]:
    """Resolve the campaign directory for a run.

    Reads from `manifest.yaml` when `campaign_id` is not explicitly given,
    falling back to the active campaign. Returns `None` if unresolvable.
    """
    if campaign_id is None:
        manifest = run_dir / "manifest.yaml"
        if manifest.exists():
            try:
                data = yaml.safe_load(manifest.read_text()) or {}
                campaign_id = data.get("campaign")
            except Exception:
                pass
    if campaign_id is None:
        campaign_id, _ = _resolve_active_campaign()
    if campaign_id is None:
        return None
    return Path(campaigns_root) / campaign_id


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="intraknot")
def main() -> None:
    """IntraKnot — HPC management for tensor-network simulations."""


# ---------------------------------------------------------------------------
# iknot init
# ---------------------------------------------------------------------------

@main.command("init")
@click.option("--campaigns-root", default="campaigns", show_default=True,
              help="Path for the campaigns directory.")
@click.option("--runs-root", default="runs", show_default=True,
              help="Path for the runs directory.")
@click.option("--notebooks-root", default="notebooks", show_default=True,
              help="Path for the notebooks directory.")
def cmd_init(campaigns_root: str, runs_root: str, notebooks_root: str) -> None:
    """Initialise the IntraKnot project skeleton.

    Creates configs/, campaigns/, runs/, and notebooks/. Writes template
    configs/slurm.toml, configs/paths.toml, and configs/machines.yaml.
    Appends .iknot_state to the root .gitignore.

    When run inside the IntraKnot source repository itself, each created
    directory also receives a .gitignore that excludes all its contents from
    git. This guard is intentionally omitted in user projects so that
    IntraKnot does not silently impose git-ignore rules on them.
    """
    cwd = Path.cwd()
    in_dev = _is_intraknot_source_project(cwd)

    # configs/
    configs_dir = cwd / "configs"
    if in_dev:
        write_data_gitignore(configs_dir)
    else:
        configs_dir.mkdir(parents=True, exist_ok=True)
    slurm_path = configs_dir / "slurm.toml"
    paths_path = configs_dir / "paths.toml"
    machines_path = configs_dir / "machines.yaml"
    if not slurm_path.exists():
        write_slurm_toml(slurm_path)
        click.echo(f"  created {slurm_path.relative_to(cwd)}")
    if not paths_path.exists():
        write_paths_toml(paths_path)
        click.echo(f"  created {paths_path.relative_to(cwd)}")
    if not machines_path.exists():
        write_machines_yaml(machines_path)
        click.echo(f"  created {machines_path.relative_to(cwd)}")

    # Data directories.
    for rel in (campaigns_root, runs_root, notebooks_root):
        d = cwd / rel
        if in_dev:
            write_data_gitignore(d)
            click.echo(f"  created {rel}/ with .gitignore")
        else:
            d.mkdir(parents=True, exist_ok=True)
            click.echo(f"  created {rel}/")

    # Root .gitignore — append .iknot_state if not already present.
    root_gitignore = cwd / ".gitignore"
    entry = ".iknot_state\n"
    if root_gitignore.exists():
        existing = root_gitignore.read_text()
        if ".iknot_state" not in existing:
            root_gitignore.write_text(existing.rstrip("\n") + "\n" + entry)
    else:
        root_gitignore.write_text(entry)
    click.echo("  updated .gitignore")

    click.echo("\nDone. Edit configs/slurm.toml and configs/paths.toml before submitting jobs.")


# ---------------------------------------------------------------------------
# iknot campaign
# ---------------------------------------------------------------------------

@main.group("campaign")
def grp_campaign() -> None:
    """Create and manage campaigns."""


@grp_campaign.command("create")
@click.argument("campaign_id")
@click.option("--description", default="", help="Human-readable description.")
@click.option("--algorithm", default="dmrg", show_default=True,
              help="Algorithm runner to copy into the campaign.")
@click.option("--campaigns-root", default="campaigns", show_default=True,
              help="Parent directory for campaign subdirectories.")
@click.option("--machine", "machine_opt", default=None,
              help="Path to configs/ directory. Defaults to ./configs.")
def campaign_create(
    campaign_id: str,
    description: str,
    algorithm: str,
    campaigns_root: str,
    machine_opt: Optional[str],
) -> None:
    """Create a new campaign directory."""
    root = Path(campaigns_root)
    configs_dir = Path(machine_opt) if machine_opt else Path.cwd() / "configs"
    try:
        campaign_dir = create_campaign(
            campaign_id, description, algorithm, root, configs_dir=configs_dir
        )
        click.echo(f"Created campaign: {campaign_dir}")
    except FileExistsError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@grp_campaign.command("activate")
@click.argument("campaign_id")
def campaign_activate(campaign_id: str) -> None:
    """Set CAMPAIGN_ID as the active campaign.

    Writes to .iknot_state and prints the export command for optional
    shell-level sourcing.
    """
    state = _read_state()
    state["active_campaign"] = campaign_id
    _write_state(state)
    click.echo(f"Active campaign set to: {campaign_id}")
    click.echo(f"\nTo also set the environment variable in this shell, run:")
    click.echo(f"  export INTRAKNOT_CAMPAIGN={campaign_id}")


@grp_campaign.command("deactivate")
def campaign_deactivate() -> None:
    """Clear the active campaign."""
    state = _read_state()
    state.pop("active_campaign", None)
    _write_state(state)
    click.echo("Active campaign cleared.")
    click.echo("\nIf you set the environment variable, also run:")
    click.echo("  unset INTRAKNOT_CAMPAIGN")


@grp_campaign.command("status")
def campaign_status() -> None:
    """Show the currently active campaign."""
    campaign_id, source = _resolve_active_campaign()
    if campaign_id:
        source_labels = {
            "env":  "Source (env)    ",
            "file": "Source (file)   ",
        }
        source_values = {
            "env":  "$INTRAKNOT_CAMPAIGN",
            "file": _STATE_FILE,
        }
        label = source_labels.get(source, "Source          ")
        value = source_values.get(source, source)
        click.echo(f"Active campaign : {campaign_id}")
        click.echo(f"{label}: {value}")
    else:
        click.echo("No active campaign.")
        click.echo("Use `iknot campaign activate <id>` to set one.")


# ---------------------------------------------------------------------------
# iknot run
# ---------------------------------------------------------------------------

@main.group("run")
def grp_run() -> None:
    """Create and submit simulation runs."""


@grp_run.command("create")
@click.argument("run_id", required=False, default=None)
@click.option("--config", "config_src", default=None, type=click.Path(),
              help="Path to a per-run config.toml with overrides. "
                   "Falls back to config.toml in the current directory, "
                   "then to the campaign defaults.toml alone. "
                   "Mutually exclusive with --set.")
@click.option("--set", "set_args", multiple=True, metavar="SECTION.KEY=VALUE",
              help="Override a campaign default. Format: section.key=value. "
                   "Comma-separate values for a scan (e.g. algorithm.max_bond=64,128,256). "
                   "Mutually exclusive with RUN_ID and --config. "
                   "Repeatable.")
@click.option("--scan", "scan_id", default=None, metavar="SCAN_ID",
              help="Tag all created runs with this scan identifier. "
                   "Required when any --set value contains multiple entries.")
@click.option("--campaign", "campaign_id", default=None,
              help="Campaign ID. Defaults to the active campaign.")
@click.option("--campaigns-root", default="campaigns", show_default=True)
@click.option("--runs-root", default="runs", show_default=True)
@click.option("--machine", "machine_opt", default=None,
              help="Path to configs/ directory. Defaults to ./configs.")
def run_create(
    run_id: Optional[str],
    config_src: Optional[str],
    set_args: tuple,
    scan_id: Optional[str],
    campaign_id: Optional[str],
    campaigns_root: str,
    runs_root: str,
    machine_opt: Optional[str],
) -> None:
    """Create a new run directory.

    There are two modes of operation:

    \b
    1. Explicit name (legacy):
         iknot run create chi128
         iknot run create chi128 --config overrides.toml

    \b
    2. Auto-named from --set overrides:
         iknot run create --set algorithm.max_bond=128 --set geometry.lx=40
         iknot run create --scan chi_study --set algorithm.max_bond=64,128,256

    In mode 2 the run ID is derived from the campaign name and the overridden
    keys. When any --set value is comma-separated, one run is created per
    value combination (cartesian product) and --scan is required.
    """
    if campaign_id is None:
        campaign_id, _ = _resolve_active_campaign()
    if campaign_id is None:
        click.echo(
            "Error: no campaign specified and no active campaign set.\n"
            "Use --campaign or `iknot campaign activate <id>`.",
            err=True,
        )
        sys.exit(1)

    using_set = bool(set_args)

    # Mutual exclusion checks.
    if run_id is not None and using_set:
        click.echo(
            "Error: RUN_ID and --set are mutually exclusive. "
            "Provide either a positional run ID or --set overrides, not both.",
            err=True,
        )
        sys.exit(1)
    if run_id is not None and config_src is not None and using_set:
        # --config with --set is also excluded (caught above), but keep guard.
        pass
    if not run_id and not using_set:
        click.echo(
            "Error: provide either a RUN_ID or at least one --set override.",
            err=True,
        )
        sys.exit(1)
    if using_set and config_src is not None:
        click.echo(
            "Error: --config and --set are mutually exclusive.",
            err=True,
        )
        sys.exit(1)

    machine = _load_machine(machine_opt)

    if using_set:
        # Auto-naming mode: use --set overrides to build config and run ID.
        campaigns_root_path = Path(campaigns_root)
        campaign_dir = campaigns_root_path / campaign_id
        defaults = load_campaign_defaults(campaign_dir)

        # Check if any --set arg has multiple values (scan mode).
        is_scan = any("," in arg.split("=", 1)[1] for arg in set_args if "=" in arg)
        if is_scan and scan_id is None:
            click.echo(
                "Error: --scan SCAN_ID is required when any --set value "
                "contains multiple comma-separated entries.",
                err=True,
            )
            sys.exit(1)

        try:
            combinations = list(iter_scan_combinations(list(set_args), defaults))
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        any_error = False
        for nested_overrides, flat_overrides in combinations:
            generated_id = make_run_id(campaign_id, flat_overrides)
            try:
                run_dir = create_run(
                    run_id=generated_id,
                    campaign_id=campaign_id,
                    config_src=None,
                    runs_root=Path(runs_root),
                    campaigns_root=campaigns_root_path,
                    machine=machine,
                    scan_id=scan_id or "",
                    overrides=nested_overrides,
                )
                click.echo(f"Created run: {run_dir}")
            except (FileExistsError, FileNotFoundError, ValueError) as e:
                click.echo(f"Error: {e}", err=True)
                any_error = True

        if any_error:
            sys.exit(1)

    else:
        # Explicit-name mode (legacy behaviour).
        if config_src is not None:
            config_path: Optional[Path] = Path(config_src)
            if not config_path.exists():
                click.echo(f"Error: config file not found: {config_path}", err=True)
                sys.exit(1)
        else:
            cwd_default = Path.cwd() / "config.toml"
            config_path = cwd_default if cwd_default.exists() else None

        try:
            run_dir = create_run(
                run_id=run_id,
                campaign_id=campaign_id,
                config_src=config_path,
                runs_root=Path(runs_root),
                campaigns_root=Path(campaigns_root),
                machine=machine,
            )
            click.echo(f"Created run: {run_dir}")
        except (FileExistsError, FileNotFoundError) as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


@grp_run.command("start")
@click.argument("run_id", required=False, default=None)
@click.option("--scan", "scan_id", default=None, metavar="SCAN_ID",
              help="Start all runs belonging to this scan ID. "
                   "Mutually exclusive with RUN_ID.")
@click.option("--status", "status_filter", default=None, metavar="STATUS",
              help="When --scan is given, restrict to runs with this status "
                   "(e.g. 'pending', 'failed').")
@click.option("--campaign", "campaign_id", default=None,
              help="Campaign ID. Required when --scan is used and no campaign "
                   "is active. Defaults to the active campaign.")
@click.option("--campaigns-root", default="campaigns", show_default=True)
@click.option("--runs-root", default="runs", show_default=True)
@click.option("--machine", "machine_opt", default=None,
              help="Path to machine configs/ directory. Defaults to ./configs.")
def run_start(
    run_id: Optional[str],
    scan_id: Optional[str],
    status_filter: Optional[str],
    campaign_id: Optional[str],
    campaigns_root: str,
    runs_root: str,
    machine_opt: Optional[str],
) -> None:
    """Write a Slurm script and execute it directly with bash (no sbatch).

    Equivalent to `submit`, but runs in the foreground on the local machine.
    Useful when Slurm is not available, e.g. on a workstation or during
    interactive testing. `SLURM_JOB_ID` and `SLURM_NODELIST` are stubbed
    automatically so the script runs without a Slurm daemon.

    Provide either a positional RUN_ID to start a single run, or --scan to
    start all runs (optionally filtered by --status) belonging to a scan.
    """
    if run_id is not None and scan_id is not None:
        click.echo("Error: RUN_ID and --scan are mutually exclusive.", err=True)
        sys.exit(1)
    if run_id is None and scan_id is None:
        click.echo(
            "Error: provide either a RUN_ID or --scan SCAN_ID.", err=True
        )
        sys.exit(1)

    machine = _load_machine(machine_opt)

    # Resolve the list of run IDs to start.
    if scan_id is not None:
        if campaign_id is None:
            campaign_id, _ = _resolve_active_campaign()
        if campaign_id is None:
            click.echo(
                "Error: --scan requires a campaign. "
                "Use --campaign or `iknot campaign activate <id>`.",
                err=True,
            )
            sys.exit(1)
        campaign_dir = Path(campaigns_root) / campaign_id
        run_ids = read_runs_by_filter(campaign_dir, scan_id=scan_id, status=status_filter)
        if not run_ids:
            click.echo(
                f"No runs found for scan '{scan_id}'"
                + (f" with status '{status_filter}'" if status_filter else "")
                + "."
            )
            return
    else:
        run_ids = [run_id]

    error_code = 0
    for rid in run_ids:
        run_dir = Path(runs_root) / rid
        if not run_dir.exists():
            click.echo(f"Error: run directory not found: {run_dir}", err=True)
            error_code = 1
            continue
        try:
            script = write_slurm_script(run_dir, machine, rid)
            click.echo(f"Wrote Slurm script: {script}")
            click.echo(f"Starting run: {run_dir}")
            env = os.environ.copy()
            env.setdefault("SLURM_JOB_ID", "local")
            env.setdefault("SLURM_NODELIST", "localhost")
            subprocess.run(["sh", str(script)], check=True, env=env)
        except subprocess.CalledProcessError as e:
            click.echo(f"Script exited with status {e.returncode}.", err=True)
            error_code = e.returncode
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            error_code = 1

    if error_code:
        sys.exit(error_code)


@grp_run.command("submit")
@click.argument("run_id", required=False, default=None)
@click.option("--scan", "scan_id", default=None, metavar="SCAN_ID",
              help="Submit all runs belonging to this scan ID. "
                   "Mutually exclusive with RUN_ID.")
@click.option("--status", "status_filter", default=None, metavar="STATUS",
              help="When --scan is given, restrict to runs with this status "
                   "(e.g. 'pending', 'failed').")
@click.option("--campaign", "campaign_id", default=None,
              help="Campaign ID. Required when --scan is used and no campaign "
                   "is active. Defaults to the active campaign.")
@click.option("--campaigns-root", default="campaigns", show_default=True)
@click.option("--runs-root", default="runs", show_default=True)
@click.option("--machine", "machine_opt", default=None)
def run_submit(
    run_id: Optional[str],
    scan_id: Optional[str],
    status_filter: Optional[str],
    campaign_id: Optional[str],
    campaigns_root: str,
    runs_root: str,
    machine_opt: Optional[str],
) -> None:
    """Write a Slurm script and submit it to the Slurm scheduler.

    Provide either a positional RUN_ID to submit a single run, or --scan to
    submit all runs (optionally filtered by --status) belonging to a scan.
    """
    if run_id is not None and scan_id is not None:
        click.echo("Error: RUN_ID and --scan are mutually exclusive.", err=True)
        sys.exit(1)
    if run_id is None and scan_id is None:
        click.echo(
            "Error: provide either a RUN_ID or --scan SCAN_ID.", err=True
        )
        sys.exit(1)

    machine = _load_machine(machine_opt)

    # Resolve the list of run IDs to submit.
    if scan_id is not None:
        if campaign_id is None:
            campaign_id, _ = _resolve_active_campaign()
        if campaign_id is None:
            click.echo(
                "Error: --scan requires a campaign. "
                "Use --campaign or `iknot campaign activate <id>`.",
                err=True,
            )
            sys.exit(1)
        campaign_dir = Path(campaigns_root) / campaign_id
        run_ids = read_runs_by_filter(campaign_dir, scan_id=scan_id, status=status_filter)
        if not run_ids:
            click.echo(
                f"No runs found for scan '{scan_id}'"
                + (f" with status '{status_filter}'" if status_filter else "")
                + "."
            )
            return
    else:
        run_ids = [run_id]

    any_error = False
    for rid in run_ids:
        run_dir = Path(runs_root) / rid
        if not run_dir.exists():
            click.echo(f"Error: run directory not found: {run_dir}", err=True)
            any_error = True
            continue
        try:
            script = write_slurm_script(run_dir, machine, rid)
            click.echo(f"Wrote Slurm script: {script}")
            job_id = submit_job(run_dir)
            click.echo(f"Submitted job: {job_id}")
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            any_error = True

    if any_error:
        sys.exit(1)


@grp_run.command("exec")
@click.argument("script_name")
@click.argument("run_id", required=False, default=None)
@click.option("--runs-root", default="runs", show_default=True)
@click.option("--campaigns-root", default="campaigns", show_default=True)
@click.option("--campaign", "campaign_id", default=None,
              help="Campaign ID. Resolved from manifest.yaml or active campaign "
                   "when omitted.")
@click.option("--machine", "machine_opt", default=None,
              help="Path to configs/ directory. Defaults to ./configs.")
@click.option("--local", is_flag=True, default=False,
              help="Run directly with bash instead of submitting to Slurm.")
@click.option("--attempt", default=None,
              help="Pin the exec script to a specific attempt "
                   "(e.g. attempt_01). Passed as --attempt to the script.")
def run_exec(
    script_name: str,
    run_id: Optional[str],
    runs_root: str,
    campaigns_root: str,
    campaign_id: Optional[str],
    machine_opt: Optional[str],
    local: bool,
    attempt: Optional[str],
) -> None:
    """Run an exec (follow-up) script in the context of a run.

    SCRIPT_NAME is the filename without the .py extension
    (e.g. `compute_sf`, `entanglement`). The script is located by searching:

    \b
    1. <run_dir>/algorithm/<script_name>.py
    2. <campaign_dir>/algorithm/<script_name>.py
    3. The intraknot package

    A Slurm script is written to exec/<script_name>/submit.slurm using the
    [exec] section of the run's slurm.toml, then submitted via sbatch (or
    run directly with --local).
    """
    machine = _load_machine(machine_opt)

    # Resolve run directory.
    if run_id is None:
        click.echo("Error: RUN_ID is required.", err=True)
        sys.exit(1)
    run_dir = Path(runs_root) / run_id
    if not run_dir.exists():
        click.echo(f"Error: run directory not found: {run_dir}", err=True)
        sys.exit(1)

    # Resolve campaign directory.
    campaign_dir = _resolve_campaign_dir(run_dir, campaign_id, campaigns_root)
    if campaign_dir is None:
        click.echo(
            "Error: could not determine campaign. "
            "Use --campaign or `iknot campaign activate <id>`.",
            err=True,
        )
        sys.exit(1)

    try:
        slot = prepare_exec(run_dir, script_name, campaign_dir)
        script = write_exec_slurm_script(run_dir, script_name, machine)
        click.echo(f"Wrote exec Slurm script: {script}")

        if local:
            click.echo(f"Running exec job locally: {script_name}")
            env = os.environ.copy()
            env.setdefault("SLURM_JOB_ID", "local")
            env.setdefault("SLURM_NODELIST", "localhost")
            cmd = ["sh", str(script)]
            if attempt:
                # Append --attempt to the shell invocation via env var so the
                # script can pick it up; alternatively, scripts read it from
                # IKNOT_ATTEMPT.
                env["IKNOT_ATTEMPT"] = attempt
            subprocess.run(cmd, check=True, env=env)
        else:
            job_id = submit_exec_job(run_dir, script_name)
            click.echo(f"Submitted exec job '{script_name}': {job_id}")
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        click.echo(f"Script exited with status {e.returncode}.", err=True)
        sys.exit(e.returncode)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@grp_run.command("delete")
@click.argument("run_id")
@click.option("--campaign", "campaign_id", default=None,
              help="Campaign ID. Resolved from manifest.yaml or active campaign when omitted.")
@click.option("--campaigns-root", default="campaigns", show_default=True)
@click.option("--runs-root", default="runs", show_default=True)
@click.option("--delete-dir", is_flag=True, default=False,
              help="Also remove the run directory from disk.")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Skip the confirmation prompt when --delete-dir is given.")
def run_delete(
    run_id: str,
    campaign_id: Optional[str],
    campaigns_root: str,
    runs_root: str,
    delete_dir: bool,
    yes: bool,
) -> None:
    """Remove a run from its campaign's runs.csv.

    By default only deregisters the run from the campaign registry
    (runs.csv); the run directory is left on disk. Pass --delete-dir to
    also remove the directory. A confirmation prompt is shown unless --yes
    is supplied.
    """
    run_dir = Path(runs_root) / run_id
    if not run_dir.exists():
        click.echo(f"Error: run directory not found: {run_dir}", err=True)
        sys.exit(1)

    campaign_dir = _resolve_campaign_dir(run_dir, campaign_id, campaigns_root)

    if delete_dir and not yes:
        click.confirm(
            f"Delete run directory '{run_dir}'? This cannot be undone.",
            abort=True,
        )

    try:
        found_in_csv = delete_run(run_dir, campaign_dir=campaign_dir, delete_dir=delete_dir)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if campaign_dir is not None:
        if found_in_csv:
            click.echo(f"Removed {run_id} from {campaign_dir.name}/runs.csv.")
        else:
            click.echo(f"Run {run_id} was not listed in {campaign_dir.name}/runs.csv.")
    else:
        click.echo("No campaign found; runs.csv not updated.")

    if delete_dir:
        click.echo(f"Deleted run directory: {run_dir}")
    else:
        click.echo(f"Run directory preserved: {run_dir}")


# ---------------------------------------------------------------------------
# iknot collect
# ---------------------------------------------------------------------------

@main.group("collect")
def grp_collect() -> None:
    """Collect results and update statuses."""


@grp_collect.command("run")
@click.argument("run_id")
@click.option("--runs-root", default="runs", show_default=True)
def collect_run_cmd(run_id: str, runs_root: str) -> None:
    """Collect results from a single run into its summary/ directory."""
    run_dir = Path(runs_root) / run_id
    if not run_dir.exists():
        click.echo(f"Error: run not found: {run_dir}", err=True)
        sys.exit(1)
    summary = collect_run(run_dir)
    click.echo(f"Run {run_id}: state={summary.get('state')} energy={summary.get('energy')}")


@grp_collect.command("campaign")
@click.option("--id", "campaign_id", default=None,
              help="Campaign ID. Defaults to active campaign.")
@click.option("--campaigns-root", default="campaigns", show_default=True)
@click.option("--runs-root", default="runs", show_default=True)
def collect_campaign_cmd(
    campaign_id: Optional[str],
    campaigns_root: str,
    runs_root: str,
) -> None:
    """Collect results for all runs in a campaign."""
    if campaign_id is None:
        campaign_id, _ = _resolve_active_campaign()
    if campaign_id is None:
        click.echo("Error: no campaign specified.", err=True)
        sys.exit(1)

    campaign_dir = Path(campaigns_root) / campaign_id
    summaries = collect_campaign(campaign_dir, Path(runs_root))
    for s in summaries:
        click.echo(f"  {s.get('run_id')}: {s.get('state')}")
    click.echo(f"\nCollected {len(summaries)} run(s).")


# ---------------------------------------------------------------------------
# iknot resume
# ---------------------------------------------------------------------------

@main.group("resume")
def grp_resume() -> None:
    """Create new attempts for failed or interrupted runs."""


@grp_resume.command("run")
@click.argument("run_id")
@click.option("--runs-root", default="runs", show_default=True)
@click.option("--machine", "machine_opt", default=None)
@click.option("--no-submit", is_flag=True, default=False,
              help="Create attempt directory without submitting to Slurm.")
def resume_run_cmd(
    run_id: str,
    runs_root: str,
    machine_opt: Optional[str],
    no_submit: bool,
) -> None:
    """Resume a single failed run."""
    machine = _load_machine(machine_opt)
    run_dir = Path(runs_root) / run_id
    if not run_dir.exists():
        click.echo(f"Error: run not found: {run_dir}", err=True)
        sys.exit(1)
    try:
        attempt = resume_run(run_dir, machine, submit=not no_submit)
        click.echo(f"Created attempt: {attempt}")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@grp_resume.command("campaign")
@click.option("--id", "campaign_id", default=None)
@click.option("--campaigns-root", default="campaigns", show_default=True)
@click.option("--runs-root", default="runs", show_default=True)
@click.option("--machine", "machine_opt", default=None)
@click.option("--no-submit", is_flag=True, default=False)
def resume_campaign_cmd(
    campaign_id: Optional[str],
    campaigns_root: str,
    runs_root: str,
    machine_opt: Optional[str],
    no_submit: bool,
) -> None:
    """Resume all resumable runs in a campaign."""
    if campaign_id is None:
        campaign_id, _ = _resolve_active_campaign()
    if campaign_id is None:
        click.echo("Error: no campaign specified.", err=True)
        sys.exit(1)

    machine = _load_machine(machine_opt)
    campaign_dir = Path(campaigns_root) / campaign_id
    resumed = resume_campaign(campaign_dir, Path(runs_root), machine, submit=not no_submit)
    if resumed:
        for p in resumed:
            click.echo(f"  resumed: {p}")
        click.echo(f"\nResumed {len(resumed)} run(s).")
    else:
        click.echo("No resumable runs found.")


# ---------------------------------------------------------------------------
# iknot status
# ---------------------------------------------------------------------------

@main.command("status")
@click.argument("run_id")
@click.option("--runs-root", default="runs", show_default=True)
def cmd_status(run_id: str, runs_root: str) -> None:
    """Print the status of a run, including any exec jobs."""
    run_dir = Path(runs_root) / run_id
    status_path = run_dir / "main" / "status.json"
    if not status_path.exists():
        click.echo(f"No status found for run: {run_id}")
        return

    try:
        s = read_status(status_path)
    except (KeyError, ValueError) as e:
        click.echo(f"Error reading status: {e}", err=True)
        sys.exit(1)

    if isinstance(s, MainStatus):
        click.echo(f"Run             : {run_id}")
        click.echo(f"State           : {s.state.value}")
        click.echo(f"Current attempt : {s.current_attempt or '—'}")
        click.echo(f"Reason          : {s.reason.value if s.reason else '—'}")
        click.echo(f"Restartable     : {s.restartable}")

        # Also show summary observables if available.
        obs_path = run_dir / "summary" / "observables.json"
        if obs_path.exists():
            try:
                obs = json.loads(obs_path.read_text())
                energy = obs.get("energy")
                converged = obs.get("converged")
                if energy is not None:
                    click.echo(f"Energy          : {energy:.10f}")
                if converged is not None:
                    click.echo(f"Converged       : {converged}")
            except (json.JSONDecodeError, OSError):
                pass

        # Summarise exec jobs if exec/ directory exists.
        exec_dir = run_dir / "exec"
        if exec_dir.is_dir():
            slots = sorted(p.name for p in exec_dir.iterdir() if p.is_dir())
            if slots:
                click.echo("\nExec jobs:")
                for slot_name in slots:
                    slot_status_path = exec_dir / slot_name / "status.json"
                    if slot_status_path.exists():
                        try:
                            slot_data = json.loads(slot_status_path.read_text())
                            state_val = slot_data.get("state", "unknown")
                        except (json.JSONDecodeError, OSError):
                            state_val = "unreadable"
                    else:
                        state_val = "pending"
                    click.echo(f"  {slot_name:<30} {state_val}")
    else:
        click.echo(json.dumps(s.to_dict(), indent=2))
