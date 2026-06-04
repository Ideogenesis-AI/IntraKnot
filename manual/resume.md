# `iknot resume`

Creates new execution attempts for runs that have failed in a retryable way (e.g. due to a scheduler timeout or out-of-memory error) and optionally resubmits them to Slurm. The scientific configuration — `config.toml` — is never modified; only the execution infrastructure is recreated.

## Common workflow

```bash
# After iknot collect campaign has updated run statuses:

# Resume all retryable failures in the active campaign in one command.
iknot resume campaign

# Or inspect which runs would be resumed before doing so.
iknot resume campaign --no-submit   # creates attempt directories, does not sbatch

# Or target a single run.
iknot resume run chi256
```

## Reference

### Restartability

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

### Attempt directory layout

Each resumption creates the next numbered attempt directory and updates the `main/current` symlink:

```
runs/<RUN_ID>/main/
├── status.json
├── current -> attempts/attempt_02   # updated symlink
├── job_id.txt                       # updated with new Slurm job ID
├── submit.slurm                     # regenerated from slurm.toml
├── logs/
└── attempts/
    ├── attempt_01/                  # previous (failed) attempt
    └── attempt_02/                  # newly created attempt
```

---

### `iknot resume run <RUN_ID>`

Creates the next `attempt_NN` directory under `main/attempts/`, updates the `main/current` symlink, regenerates `main/submit.slurm`, and submits the job to Slurm (unless `--no-submit` is given).

#### Synopsis

```
iknot resume run [OPTIONS] RUN_ID
```

#### Options

| Option | Default | Description |
|---|---|---|
| `--runs-root PATH` | `runs` | Parent directory for runs. |
| `--machine PATH` | `./configs` | Path to the `configs/` directory. |
| `--no-submit` | off | Create the attempt directory and write the Slurm script without calling `sbatch`. |

If the run is not resumable (wrong state or `restartable = false`), the command exits with an error and no files are modified.

---

### `iknot resume campaign`

Iterates over all runs listed in the campaign's `runs.csv` and calls `resume run` for each that is currently resumable.

#### Synopsis

```
iknot resume campaign [OPTIONS]
```

#### Options

| Option | Default | Description |
|---|---|---|
| `--id TEXT` | active campaign | Campaign ID. |
| `--campaigns-root PATH` | `campaigns` | Parent directory for campaigns. |
| `--runs-root PATH` | `runs` | Parent directory for runs. |
| `--machine PATH` | `./configs` | Path to the `configs/` directory. |
| `--no-submit` | off | Create attempt directories without submitting. |

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
