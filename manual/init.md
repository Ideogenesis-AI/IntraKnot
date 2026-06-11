# `iknot init`

Initialises the IntraKnot project skeleton in the current directory. Run this command once when starting a new project.

## Common workflow

```bash
cd my_project/
iknot init
```

`iknot init` creates the four standard directories and writes template configuration files that you fill in before submitting any jobs:

```
my_project/
├── configs/
│   ├── slurm.toml      # master Slurm template — edit before first use
│   ├── paths.toml      # filesystem paths for the target cluster
│   ├── tui.toml        # TUI settings (editor, etc.)
│   └── registry.yaml   # algorithm database registry (managed by iknot database)
├── campaigns/          # scientific groupings and run indexes
├── runs/               # simulation case directories
└── notebooks/          # inspection and plotting notebooks
```

After running `iknot init`:

1. Fill in `configs/paths.toml` with the cluster's filesystem layout.
2. Run `iknot cluster sync` to query Slurm and populate `configs/cluster.yaml` with the available partitions, node hardware specs, and valid constraint values.
3. Run `iknot cluster show` to read the discovered values, then open `configs/slurm.toml` and replace every `_init_` placeholder with the correct partition and constraint for your cluster.
4. Optionally run `iknot database update` to fetch the latest manifest from any pre-registered algorithm databases into `configs/registry.yaml`.

See [`iknot cluster`](cluster.md) for details on the discovery workflow.

## Reference

### Synopsis

```
iknot init [OPTIONS]
```

### Options

| Option | Default | Description |
|---|---|---|
| `--campaigns-root PATH` | `campaigns` | Path for the campaigns directory. |
| `--runs-root PATH` | `runs` | Path for the runs directory. |
| `--notebooks-root PATH` | `notebooks` | Path for the notebooks directory. |

### Files created

#### `configs/slurm.toml`

The master Slurm template. Contains three sections:

- `[basic]` — account and email notification settings shared by all jobs.
- `[main]` — resource request for primary (e.g. DMRG ground-state) jobs: partition, constraint, walltime, memory, node and CPU count.
- `[exec]` — resource request for follow-up exec jobs, which are typically lighter than the primary calculation.

This file is copied verbatim to each campaign on `iknot campaign create`, and from there to each run on `iknot run create`. Edit the campaign copy for campaign-wide settings and the run copy for single-run overrides. No automatic merging happens between the copies; the file in the run directory is what gets submitted.

Fields marked `_init_` must be set before any job is submitted.

#### `configs/tui.toml`

Settings for the interactive dashboard (`iknot tui`):

| Key | Default | Description |
|---|---|---|
| `tui.editor` | `vi` | Editor launched when opening log files from the dashboard. The dashboard suspends, runs the editor in the foreground, then resumes. |

#### `configs/paths.toml`

Filesystem path settings for the target cluster:

| Key | Description |
|---|---|
| `project_root` | Root directory for run subdirectories. |
| `scratch_root` | Fast scratch filesystem root used by jobs for temporary data. |
| `scratch_node` | Per-node local scratch, typically `/tmp/$USER`. |
| `command` | Command used to invoke the runner script from Slurm, e.g. `uv run`. |

#### `configs/registry.yaml`

The unified algorithm database registry. Stores metadata for all registered external databases — their URLs, last-fetched timestamps, and the cached manifest of available scripts.

The file is written with a minimal template on `iknot init`. Use `iknot database add`, `iknot database remove`, and `iknot database update` to manage its contents. See [`iknot database`](database.md) for details.

#### `.gitignore`

`iknot init` appends `.iknot_state` to the root `.gitignore`, creating the file if it does not already exist. `.iknot_state` is the local session state file that tracks the active campaign and should not be committed.

#### `manual/` symlink

`iknot init` creates a `manual/` symlink in the project root pointing to the bundled documentation directory inside the installed package. This makes the manual pages accessible at a predictable path (`./manual/<topic>.md`) for both users and agents working in the project, without requiring knowledge of where the package is installed. The symlink is skipped if `manual/` already exists or if the package is running from an editable install.

### Idempotency

`iknot init` is safe to re-run. Existing files are never overwritten; only missing files and directories are created.
