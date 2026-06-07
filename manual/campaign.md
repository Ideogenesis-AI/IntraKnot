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

## Active-campaign resolution order

Commands that accept a `--campaign` option resolve the campaign as follows when the option is omitted:

1. `INTRAKNOT_CAMPAIGN` environment variable.
2. `active_campaign` key in `.iknot_state` at the project root.
3. `None` — the command errors with a message asking you to specify a campaign.

