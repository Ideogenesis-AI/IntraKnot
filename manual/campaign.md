# `iknot campaign`

A campaign is a scientific grouping of related runs — typically a parameter study or a set of model variants sharing the same algorithm and resource settings. Every run belongs to exactly one campaign.

## Common workflow

```bash
# 1. Create the campaign directory.
iknot campaign create heisenberg_chi_scan --description "Bond-dimension scan, L=64 Heisenberg chain"

# 2. Edit the campaign's physics parameters and Slurm settings.
#    defaults.toml sets the baseline for every run in this campaign.
$EDITOR campaigns/heisenberg_chi_scan/defaults.toml
$EDITOR campaigns/heisenberg_chi_scan/slurm.toml

# 3. Activate the campaign so that subsequent commands target it by default.
iknot campaign activate heisenberg_chi_scan

# 4. Create and submit runs (see iknot run).
iknot run create chi128
iknot run create chi256

# 5. When done, deactivate to avoid accidentally targeting this campaign later.
iknot campaign deactivate
```

## Reference

### `iknot campaign create <CAMPAIGN_ID>`

Creates a new campaign directory under `campaigns/` and populates it with a standard skeleton.

#### Synopsis

```
iknot campaign create [OPTIONS] CAMPAIGN_ID
```

#### Options


| Option                  | Default     | Description                                           |
| ----------------------- | ----------- | ----------------------------------------------------- |
| `--description TEXT`    | `""`        | Human-readable description stored in `campaign.yaml`. |
| `--algorithm TEXT`      | `dmrg`      | Algorithm runner script to copy into `algorithm/`.    |
| `--campaigns-root PATH` | `campaigns` | Parent directory for campaign subdirectories.         |
| `--machine PATH`        | `./configs` | Path to the `configs/` directory.                     |


#### Files created

```
campaigns/<CAMPAIGN_ID>/
├── campaign.yaml       # metadata: id, description, created timestamp
├── defaults.toml       # baseline physics/algorithm configuration for all runs
├── slurm.toml          # Slurm settings (copied from configs/slurm.toml)
├── runs.csv            # run registry: run_id, scan_id columns
├── notes.md            # free-form notes
├── logs/               # Slurm array job log files
└── algorithm/
    ├── algorithm.lock  # provenance tracking for all scripts in this directory
    └── run_dmrg.py     # algorithm runner script (or the one named by --algorithm)
```

`**defaults.toml**` is the campaign's baseline scientific configuration. It is an Alice-compatible TOML file with up to four sections: `[geometry]`, `[model]`, `[algorithm]`, and `[output]`. When a run is created with `iknot run create`, the campaign's `defaults.toml` is merged with any per-run overrides to produce the run's `config.toml`. Edit `defaults.toml` after creating the campaign and before creating any runs to set the shared parameters for the whole parameter study.

`**slurm.toml**` is a verbatim copy of `configs/slurm.toml` at the time the campaign is created. Edit it here to set campaign-wide Slurm settings (e.g. partition, walltime suited to the expected bond dimension).

---

### `iknot campaign activate <CAMPAIGN_ID>`

Sets `CAMPAIGN_ID` as the active campaign, so that `iknot run` and `iknot resume` commands that do not specify `--campaign` automatically target it.

#### Synopsis

```
iknot campaign activate CAMPAIGN_ID
```

Activation is recorded in `.iknot_state` (TOML file at the project root), which persists across shell sessions and is automatically excluded from git.

The command also prints instructions for setting the `INTRAKNOT_CAMPAIGN` environment variable in the current shell:

```
Active campaign set to: heisenberg_chi_scan

To also set the environment variable in this shell, run:
  export INTRAKNOT_CAMPAIGN=heisenberg_chi_scan
```

Setting the environment variable is optional. It takes precedence over `.iknot_state` when both are present, which is useful when running multiple terminal sessions that target different campaigns simultaneously.

---

### `iknot campaign deactivate`

Clears the active campaign from `.iknot_state`.

#### Synopsis

```
iknot campaign deactivate
```

If you also set the environment variable, clear it manually:

```bash
unset INTRAKNOT_CAMPAIGN
```

---

### `iknot campaign install <SOURCE>`

Fetch a script from a registered algorithm database and add it to the campaign's `algorithm/` directory as a managed entry.

#### Synopsis

```
iknot campaign install [OPTIONS] SOURCE
```

`SOURCE` is a source descriptor of the form `<db-name>:<path>`, e.g.:

```
intraknot-database:observables/spin_corr.py
```

#### Options

| Option | Default | Description |
| --- | --- | --- |
| `--as FILENAME` | (same as source path) | Destination filename inside `algorithm/`. Relative subdirectory structure from the source path is preserved by default. |
| `--campaign TEXT` | active campaign | Campaign to install into. |
| `--campaigns-root PATH` | `campaigns` | Parent directory for campaign subdirectories. |
| `--machine PATH` | `./configs` | Path to the `configs/` directory (for `registry.yaml`). |

#### What it does

1. Resolves `SOURCE` — either via `importlib.resources` for `intraknot:` sources or via an HTTP fetch for registered database sources.
2. Writes the file to `algorithm/<dest>`, creating subdirectories as needed.
3. Records the file as a **managed entry** in `algorithm.lock` with its SHA-256 digest and source descriptor.

Once installed, `iknot campaign sync` will track the file and notify you of upstream changes.

---

### `iknot campaign sync [FILE ...]`

Synchronise managed algorithm scripts from their sources.

#### Synopsis

```
iknot campaign sync [OPTIONS] [FILE ...]
```

Without `FILE` arguments, all managed entries are checked. When one or more `FILE` arguments are given, only those entries are processed.

#### Options

| Option | Default | Description |
| --- | --- | --- |
| `--force` | off | Overwrite locally modified files instead of blocking. |
| `--yes`, `-y` | off | Accept all updates without prompting. |
| `--campaign TEXT` | active campaign | Campaign to sync. |
| `--campaigns-root PATH` | `campaigns` | Parent directory for campaign subdirectories. |
| `--machine PATH` | `./configs` | Path to the `configs/` directory (for `registry.yaml`). |

#### Conflict policy

For each target file:

1. If the on-disk SHA-256 differs from the lock record, the file is **locally modified**. Without `--force`, sync is blocked for that file with a message explaining how to proceed (promote with `iknot campaign override`, or discard local changes with `--force`).
2. If the file is unmodified, the upstream SHA-256 is checked against the cached registry manifest. When it differs, the updated file is downloaded and you are prompted to confirm (skipped with `--yes`).

Files not listed in `algorithm.lock` are never touched.

---

### `iknot campaign override <FILENAME>`

Promote a managed script to **custom** (user-owned) status.

#### Synopsis

```
iknot campaign override [OPTIONS] FILENAME
```

#### Options

| Option | Default | Description |
| --- | --- | --- |
| `--campaign TEXT` | active campaign | Campaign to modify. |
| `--campaigns-root PATH` | `campaigns` | Parent directory for campaign subdirectories. |

After promotion:

- The file is moved from `[managed]` to `[custom]` in `algorithm.lock`.
- `iknot campaign sync` will never touch the file automatically again.
- The on-disk file content is **not** modified.

Use this when you want to fork a managed script and apply local modifications without IntraKnot trying to overwrite them on the next sync.

---

### `iknot campaign add-script <FILENAME>`

Register an existing user-authored script as **custom** in `algorithm.lock`.

#### Synopsis

```
iknot campaign add-script [OPTIONS] FILENAME
```

`FILENAME` is a path relative to the campaign's `algorithm/` directory, e.g. `"my_observable.py"` or `"post_process/analyse.py"`. The file must already exist on disk.

#### Options

| Option | Default | Description |
| --- | --- | --- |
| `--campaign TEXT` | active campaign | Campaign to modify. |
| `--campaigns-root PATH` | `campaigns` | Parent directory for campaign subdirectories. |

The file is added to the `[custom]` list in `algorithm.lock` so that sync commands know to leave it alone. Unlike `install`, this command does not download anything — it merely records pre-existing files that you authored yourself.

---

## Algorithm lock (`algorithm.lock`)

Every campaign's `algorithm/` directory contains an `algorithm.lock` TOML file that tracks the provenance of each script:

```
campaigns/<CAMPAIGN_ID>/
└── algorithm/
    ├── algorithm.lock      # provenance tracking file
    ├── run_dmrg.py         # managed entry (installed from intraknot package)
    └── my_obs.py           # custom entry (user-authored)
```

The lock records two kinds of entries:

**Managed** scripts
: Installed from a known source (the `intraknot` package or a registered database). The stored SHA-256 digest lets `iknot campaign sync` detect upstream changes and local modifications without re-downloading the file.

**Custom** scripts
: User-authored scripts that IntraKnot never modifies or removes automatically.

The lock file is included in each run's frozen `algorithm/` snapshot, ensuring full reproducibility — every run carries a record of exactly which version of each script it used.

---

### `iknot campaign status`

Shows which campaign is currently active and where the setting was resolved from.

#### Synopsis

```
iknot campaign status
```

#### Output examples

When the campaign was set via `iknot campaign activate`:

```
Active campaign : heisenberg_chi_scan
Source (file)   : .iknot_state
```

When the campaign was set via the environment variable:

```
Active campaign : heisenberg_chi_scan
Source (env)    : $INTRAKNOT_CAMPAIGN
```

If no campaign is active:

```
No active campaign.
Use `iknot campaign activate <id>` to set one.
```

---

### `iknot campaign resume`

Iterates over every run listed in the campaign's `runs.csv` and resumes each one that is currently resumable (state is `failed` and `restartable` is `true`). Each resumed run gets a fresh Slurm script and, unless `--no-submit` is given, is resubmitted to the scheduler.

The scientific configuration (`config.toml`) of each run is never modified.

#### Synopsis

```
iknot campaign resume [OPTIONS]
```

#### Options

| Option | Default | Description |
|---|---|---|
| `--campaign TEXT` | active campaign | Campaign ID. |
| `--campaigns-root PATH` | `campaigns` | Parent directory for campaigns. |
| `--runs-root PATH` | `runs` | Parent directory for runs. |
| `--machine PATH` | `./configs` | Path to the `configs/` directory. |
| `--no-submit` | off | Create attempt directories without submitting to Slurm. |

#### Output example

```
  resumed: runs/chi256/main/attempts/attempt_02
  resumed: runs/chi512/main/attempts/attempt_03

Resumed 2 run(s).
```

If no resumable runs are found:

```
No resumable runs found.
```

#### Restartability

A run is resumable when **both** of the following are true in `main/status.json`:

- `state` is `failed`
- `restartable` is `true`

The `restartable` flag is set by the algorithm runner script. It is `true` only for failure modes where resubmitting the same configuration is safe:

| `reason` | Retryable | Notes |
|---|---|---|
| `timeout` | yes | Job ran out of walltime; extend via `slurm.toml` if needed. |
| `out_of_memory` | yes | Job was killed by the OOM killer; increase `mem` in `slurm.toml` if needed. |
| `scheduler_failure` | yes | Slurm node failure or preemption; retry unchanged. |
| `checkpoint_missing` | yes | Expected checkpoint not found; retry from scratch or earlier checkpoint. |
| `not_converged` | yes | Sweep loop finished without meeting the convergence threshold; resume to continue sweeping from the last checkpoint. |
| `max_sweeps_reached` | no | Sweep budget exhausted; increase `max_sweeps` in `config.toml`. |
| `bad_parameters` | no | Configuration error; fix `config.toml` and create a new run instead. |
| `checkpoint_incompatible` | no | Saved checkpoint is incompatible with the current code version. |
| `nan_detected` | no | Numerical instability; requires parameter changes. |
| `linear_algebra_error` | no | Low-level linear algebra failure; requires investigation. |

---

## Active-campaign resolution order

Commands that accept a `--campaign` option resolve the campaign as follows when the option is omitted:

1. `INTRAKNOT_CAMPAIGN` environment variable.
2. `active_campaign` key in `.iknot_state` at the project root.
3. `None` — the command errors with a message asking you to specify a campaign.
