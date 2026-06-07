# `iknot status`

Prints a formatted status summary for a single run: execution state, current attempt, failure details, physical observables, and a table of all exec jobs.

## Common workflow

```bash
# Check a run immediately after submission.
iknot status chi128

iknot status chi128
```

`iknot status` reads directly from the run directory and requires no active campaign. It is the fastest way to inspect a single run without opening any files manually.

## Reference

### Synopsis

```
iknot status [OPTIONS] RUN_ID
```

### Options

| Option | Default | Description |
|---|---|---|
| `--runs-root PATH` | `runs` | Parent directory for runs. |

### Output

#### Example — completed run

```
Run             : chi128
State           : completed
Current attempt : attempt_01
Reason          : converged
Restartable     : False
Hostname        : node42
Energy          : -0.4431471805
Converged       : True
```

#### Example — failed run

```
Run             : chi512
State           : failed
Current attempt : attempt_02
Reason          : timeout
Restartable     : True
Hostname        : node17
```

#### Example — run with exec jobs

```
Run             : chi128
State           : completed
Current attempt : attempt_01
Reason          : converged
Restartable     : False
Hostname        : node42
Energy          : -0.4431471805
Converged       : True

Exec jobs:
  compute_sf                     completed
  entanglement                   pending
```

### Fields

| Field | Source | Description |
|---|---|---|
| `State` | `main/status.json` | Overall execution state of the run. |
| `Current attempt` | `main/status.json` | Name of the most recent attempt directory (e.g. `attempt_02`). `—` if no attempt has been created yet. |
| `Reason` | `main/status.json` | Reason code from the latest attempt. `—` while pending or running. |
| `Restartable` | `main/status.json` | Whether `iknot resume run` can create a new attempt. |
| `Hostname` | `main/status.json` | Hostname of the node where the most recent attempt ran. `—` until the runner starts. |
| `Energy` | current attempt's `info.json` | Ground-state energy from the latest attempt. Shown only when available. |
| `Converged` | current attempt's `info.json` | Convergence flag from the runner. Shown only when present. |
| Exec jobs table | `exec/*/status.json` | One row per exec slot found under `exec/`. State is read from `exec/<name>/status.json`; shows `pending` if the file does not exist yet. |

### Run states

| State | Meaning |
|---|---|
| `pending` | Run created; no attempt submitted yet. |
| `running` | A Slurm job is currently executing. |
| `completed` | The primary calculation finished successfully. |
| `failed` | The latest attempt ended in a failure; see `Reason`. |
| `cancelled` | The job was cancelled by the user or the scheduler. |
| `invalid` | Configuration or setup error detected before execution. |
| `skipped` | Run was intentionally skipped (set manually). |

### Failure reasons

| Reason | Retryable | Description |
|---|---|---|
| `converged` | — | Success; DMRG converged to the ground state. |
| `not_converged` | yes | Sweep loop finished without meeting the convergence threshold; resume to continue sweeping from the last checkpoint. |
| `max_sweeps_reached` | no | Sweep budget exhausted before convergence. |
| `timeout` | yes | Job exceeded the requested walltime. |
| `out_of_memory` | yes | Job was killed by the out-of-memory handler. |
| `nan_detected` | no | Numerical instability during the sweep. |
| `bad_parameters` | no | Configuration values rejected by the runner. |
| `checkpoint_missing` | yes | Expected restart checkpoint was not found. |
| `checkpoint_incompatible` | no | Saved checkpoint is incompatible with the current code. |
| `linear_algebra_error` | no | Low-level linear algebra routine failed. |
| `scheduler_failure` | yes | Slurm node failure or preemption. |

Retryable reasons (marked yes) allow `iknot resume run` to create a new attempt with the same `config.toml`. Non-retryable reasons require changes to `config.toml` or the code before resubmitting.
