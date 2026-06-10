# `iknot run`

A run is one simulation case, usually one parameter point. Each run has its own directory with a frozen configuration, a Slurm script, and a log of all execution attempts. Runs are registered in one or more campaigns' `runs.csv` files and carry a unique UUID that is independent of any campaign.

## Common workflows

### Single run

```bash
iknot campaign activate heisenberg_chi_scan

# Create a run with an explicit name, then edit its parameters if needed.
iknot run create chi128
$EDITOR runs/chi128/config.toml   # optional: fine-tune physics config
$EDITOR runs/chi128/slurm.toml    # optional: adjust walltime / memory

# Submit to Slurm, or run locally without Slurm.
iknot run submit chi128
iknot run start  chi128           # local alternative (workstation / CI)

# After the job completes, run a follow-up analysis script.
iknot run exec compute_sf chi128
```

### Parameter scan

```bash
iknot campaign activate heisenberg_chi_scan

# Create one run per value of max_bond (cartesian product across --set keys).
# All runs are tagged with scan_id=chi_study.
iknot run create --scan chi_study --set algorithm.max_bond=64,128,256,512
# → runs/dmrg_heisenberg_chain_len=20_max_bond=64_a3f7b291/
# → runs/dmrg_heisenberg_chain_len=20_max_bond=128_c91d4e02/
# → runs/dmrg_heisenberg_chain_len=20_max_bond=256_7fb83a10/
# → runs/dmrg_heisenberg_chain_len=20_max_bond=512_e204dc58/

# Submit all runs in the scan at once.
iknot run submit --scan chi_study

# Resubmit only the failed ones (status is read live from each run's main/status.json).
iknot run submit --scan chi_study --status failed
```

### Auto-named single run

```bash
# Omit RUN_ID and use --set; the run ID is derived from the config and overridden keys.
iknot run create --set algorithm.max_bond=128 --set geometry.lx=40
# → runs/dmrg_heisenberg_chain_len=40_max_bond=128_a3f7b291/

iknot run submit dmrg_heisenberg_chain_len=40_max_bond=128_a3f7b291
```

## Reference

### `iknot run create [RUN_ID]`

Creates a new run directory and registers the run in the campaign's `runs.csv`.

There are two modes of operation:

**Explicit-name mode**: provide a positional `RUN_ID`. Optionally supply `--config` to override specific parameters. A UUID is generated and stored in `manifest.yaml` even when the name is user-supplied.

**Auto-name mode**: omit `RUN_ID` and use `--set section.key=value` instead. The run ID is derived from the merged configuration and the explicitly overridden keys. When a `--set` value contains comma-separated entries (e.g. `algorithm.max_bond=64,128,256`), one run is created per combination (cartesian product across all multi-valued keys) and `--scan` is required.

Run ID format:

```
<engine>_<model_label>_<lattice_descriptor>[_<key1>=<val1>_<key2>=<val2>]_<uuid8>
```

- `engine` — algorithm engine, lower-case (e.g. `dmrg`)
- `model_label` — model label from config, lower-case (e.g. `heisenberg`)
- `lattice_descriptor` — geometry-aware size descriptor:
  - 1-D: `<lattice>_len=<lx>` (e.g. `chain_len=20`)
  - 2-D: `<lattice>_cell=<lx>x<ly>` (e.g. `square_cell=8x8`, `kagome_cell=6x6`)
- Override pairs are sorted alphabetically; floats use `g`-format.
- `uuid8` — first 8 hex characters of a UUID4 generated at creation time.

#### Synopsis

```
iknot run create [OPTIONS] [RUN_ID]
```

#### Options

| Option | Default | Description |
|---|---|---|
| `--config PATH` | CWD `config.toml` → none | Per-run TOML override file. Mutually exclusive with `--set`. |
| `--set SECTION.KEY=VALUE` | — | Override a campaign default. Repeatable. Comma-separate values for a scan. Mutually exclusive with `RUN_ID` and `--config`. |
| `--scan SCAN_ID` | — | Tag all created runs with this scan identifier. Required when any `--set` value contains multiple entries. |
| `--campaign TEXT` | active campaign | Campaign ID. |
| `--campaigns-root PATH` | `campaigns` | Parent directory for campaigns. |
| `--runs-root PATH` | `runs` | Parent directory for runs. |
| `--machine PATH` | `./configs` | Path to the `configs/` directory. |

#### Files created

```
runs/<RUN_ID>/
├── manifest.yaml           # run identity: id, uuid, algorithm, created timestamp
├── config.toml             # scientific configuration (merged from campaign defaults)
├── slurm.toml              # Slurm settings (copied from campaign's slurm.toml)
├── initial.ckpt            # symlink → campaigns/<id>/initial.ckpt (only when init="ckpt")
├── algorithm/
│   └── run_dmrg.py         # algorithm runner script (copied from campaign)
├── main/
│   ├── status.json         # initial state: {"state": "pending", ...}
│   ├── attempts/           # one subdirectory per execution attempt
│   └── logs/               # Slurm output and error logs
└── summary/                # optional; may contain observables snapshots
```

**`config.toml`** is the run's scientific configuration. It is produced by merging the campaign's `defaults.toml` with any per-run overrides: run-level keys take precedence over campaign defaults at every section (`[geometry]`, `[model]`, `[algorithm]`, `[output]`). Edit `config.toml` after the run is created to set parameter values that differ from the campaign baseline — for example, to vary the bond dimension `chi` across the parameter study.

#### MPS initialisation strategies

The `[algorithm]` section's `init` key controls how the initial MPS is constructed at the start of the first DMRG attempt:

| Value | Description |
|---|---|
| `"random"` | Random MPS with `bond_dim = max_bond` (default). |
| `"product"` | Deterministic product state (`bond_dim = 1`). Recommended for 2-site or CBE DMRG. |
| `"resume"` | Loads `summary.state` from the most recent `dmrg.ckpt` in `main/attempts/`. Reduces the remaining sweep budget so the total sweep count is consistent. |
| `"ckpt"` | Loads **only** the MPS state from a checkpoint file. No sweep count or convergence history is carried over — DMRG starts fresh from this state. |

##### `target_qn` — right-boundary quantum number

For `init = "product"` and `init = "random"`, the optional `target_qn` key pins the right-boundary charge of the initial MPS to a specific quantum number sector. Alice's `init_mps` will raise a `ValueError` if the requested sector is unreachable for the given chain length and physical space.

When `target_qn` is omitted, Alice defaults to the vacuum charge (total charge zero), which is correct for even-length chains at half-filling / zero magnetization. For **odd-length chains** (e.g. a 3×3 Kagome lattice with L = 27 sites), the zero-charge sector does not exist and `target_qn` must be set explicitly.

```toml
[algorithm]
init       = "random"

# Spin-½ U1 — odd L, target Sz = −½  (Alice charge convention: 2Sz = −1)
target_qn  = -1

# Band U1,U1 — odd L, target N=13 electrons, Sz = 0
# target_qn = [-1, 0]
```

The value must match Alice's internal charge convention for the active symmetry group:

| Symmetry | Type | Meaning |
|---|---|---|
| `U1` (Spin, Ferm) | integer | `2 × Sz` or `2N − L` |
| `SU2` | integer | total spin `S` in units of ½ |
| `U1,U1` / `Z2,U1` | array of 2 integers | `[charge, spin]` components |
| `U1,SU2` / `Z2,SU2` | array of 2 integers | `[charge, S_in_half_units]` |

`target_qn` is ignored for `init = "ckpt"` and `init = "resume"`.

When `init = "ckpt"`, the checkpoint file is resolved as follows:

1. If `init_ckpt` is set (an absolute or relative path), that file is used directly.
2. Otherwise, the runner looks for `initial.ckpt` in the run root directory (same directory as `config.toml`).

When `iknot run create` is called and `init = "ckpt"` is active without an explicit `init_ckpt`, a relative symlink `initial.ckpt → ../../campaigns/<id>/initial.ckpt` is automatically created in the run directory so that all runs in the campaign share a single initial state placed at the campaign root.

```toml
[algorithm]
init         = "ckpt"

# Optional: override the default initial.ckpt location.
# init_ckpt  = "/path/to/some/other/run/state.ckpt"
```

**`slurm.toml`** is a copy of the campaign's `slurm.toml`. Edit it here for single-run Slurm overrides (e.g. a longer walltime for a larger bond dimension) without affecting other runs in the campaign.

---

### `iknot run submit [RUN_ID]`

Writes `main/submit.slurm` from the run's `slurm.toml` and submits it to the Slurm scheduler with `sbatch`. The assigned job ID is recorded in `main/job_id.txt`.

Provide either a positional `RUN_ID` to submit a single run, or `--scan` to submit all runs belonging to a scan (optionally filtered by `--status`).

#### Synopsis

```
iknot run submit [OPTIONS] [RUN_ID]
```

#### Options

| Option | Default | Description |
|---|---|---|
| `--scan SCAN_ID` | — | Submit all runs with this scan ID. Mutually exclusive with `RUN_ID`. |
| `--status STATUS` | — | When `--scan` is given, restrict to runs with this status (e.g. `pending`, `failed`). |
| `--campaign TEXT` | active campaign | Campaign ID. Required when `--scan` is used and no campaign is active. |
| `--campaigns-root PATH` | `campaigns` | Parent directory for campaigns. |
| `--runs-root PATH` | `runs` | Parent directory for runs. |
| `--machine PATH` | `./configs` | Path to the `configs/` directory. |

---

### `iknot run start [RUN_ID]`

Writes `main/submit.slurm` and executes it directly with `sh` on the local machine, without requiring a Slurm daemon. `SLURM_JOB_ID` is set to `local` and `SLURM_NODELIST` to `localhost` so the runner script works without modification.

Use this command on workstations or in CI environments where Slurm is not available.

Provide either a positional `RUN_ID` to start a single run, or `--scan` to start all runs belonging to a scan (optionally filtered by `--status`).

#### Synopsis

```
iknot run start [OPTIONS] [RUN_ID]
```

#### Options

| Option | Default | Description |
|---|---|---|
| `--scan SCAN_ID` | — | Start all runs with this scan ID. Mutually exclusive with `RUN_ID`. |
| `--status STATUS` | — | When `--scan` is given, restrict to runs with this status (e.g. `pending`, `failed`). |
| `--campaign TEXT` | active campaign | Campaign ID. Required when `--scan` is used and no campaign is active. |
| `--campaigns-root PATH` | `campaigns` | Parent directory for campaigns. |
| `--runs-root PATH` | `runs` | Parent directory for runs. |
| `--machine PATH` | `./configs` | Path to the `configs/` directory. |

---

### `iknot run exec <SCRIPT_NAME> <RUN_ID>`

Runs a follow-up (exec) script against a completed run. Exec jobs are bespoke Python scripts for post-processing, measurement, or analysis that depend on the run's output (e.g. computing a structure factor or entanglement spectrum from the converged ground state).

#### Synopsis

```
iknot run exec [OPTIONS] SCRIPT_NAME RUN_ID
```

#### Options

| Option | Default | Description |
|---|---|---|
| `--runs-root PATH` | `runs` | Parent directory for runs. |
| `--campaigns-root PATH` | `campaigns` | Parent directory for campaigns. |
| `--campaign TEXT` | active campaign | Campaign ID. |
| `--machine PATH` | `./configs` | Path to the `configs/` directory. |
| `--local` | off | Run with `sh` instead of submitting to Slurm. |
| `--attempt TEXT` | none | Pin the exec script to a specific attempt (e.g. `attempt_01`). Only effective with `--local`; passed to the script as the `IKNOT_ATTEMPT` environment variable. |

#### Script search path

`SCRIPT_NAME` is the script filename without the `.py` extension (e.g. `compute_sf`). The script is located by searching in order:

1. `runs/<RUN_ID>/algorithm/<SCRIPT_NAME>.py` — a run-specific override already promoted to the run directory.
2. `campaigns/<CAMPAIGN_ID>/algorithm/<SCRIPT_NAME>.py` — the campaign's shared script.
3. The `intraknot` package — bundled scripts distributed with IntraKnot.

The script is copied into `runs/<RUN_ID>/algorithm/` and a Slurm script is written to `runs/<RUN_ID>/exec/<SCRIPT_NAME>/submit.slurm` using the `[exec]` section of the run's `slurm.toml`. The exec job is then submitted via `sbatch`, or run locally with `--local`.

#### Output layout

```
runs/<RUN_ID>/exec/<SCRIPT_NAME>/
├── submit.slurm
├── logs/               # Slurm output and error logs
└── status.json         # written by the script on completion
```

---

### `iknot run delete <RUN_ID>`

Deregisters a run from a campaign's `runs.csv`. By default the run directory is preserved on disk.

#### Synopsis

```
iknot run delete [OPTIONS] RUN_ID
```

#### Options

| Option | Default | Description |
|---|---|---|
| `--campaign TEXT` | active campaign | Campaign ID. Used only without `--delete-dir`. |
| `--campaigns-root PATH` | `campaigns` | Parent directory for campaigns. |
| `--runs-root PATH` | `runs` | Parent directory for runs. |
| `--delete-dir` | off | Remove the run from **every** campaign's `runs.csv`, then delete the run directory from disk. |
| `--yes` / `-y` | off | Skip the confirmation prompt when `--delete-dir` is given. |

Without `--delete-dir`, only the single active (or specified) campaign's `runs.csv` registration is removed. The directory and all its data remain intact, and the run can be re-registered by editing `runs.csv` directly.

With `--delete-dir`, every campaign under `--campaigns-root` is scanned and the run is removed from each one before the directory is deleted. This is the correct way to fully destroy a run that belongs to multiple campaigns.
