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

#### Example — completed DMRG run

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

#### Example — completed XTRG run

```
Run             : chi128_xtrg
State           : completed
Current attempt : attempt_01
Reason          : finished
Restartable     : False
Hostname        : node42
Finished        : True
```

XTRG's `info.json` has no `energy` or `converged` key (it has no ground state or convergence criterion); `Finished` reflects `alice.algorithm.xtrg.Summary.finished` — the fixed cooling schedule ran to completion.

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
| `Energy` | current attempt's `info.json` | Ground-state energy from the latest attempt (DMRG only). Shown only when available. |
| `Converged` | current attempt's `info.json` | DMRG's numerical convergence flag (`info.json` key `converged`). Shown only when present. |
| `Finished` | current attempt's `info.json` | XTRG's schedule-completion flag (`info.json` key `finished`, mirroring Alice's `Summary.finished`). XTRG has no convergence criterion, so this does not imply numerical convergence. Shown only when present; mutually exclusive with `Converged`. |
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
| `converged` | — | Success (DMRG only). DMRG converged to the ground state within its convergence tolerance. |
| `not_converged` | yes | Failure (DMRG only). Sweep loop finished without meeting the convergence threshold; resume to continue sweeping from the last checkpoint. |
| `finished` | — | Success (XTRG only). XTRG completed its fixed cooling schedule (`n_steps` doublings). Does not imply numerical convergence — XTRG has no convergence criterion. |
| `not_finished` | yes | Failure (XTRG only, currently unreachable). Reserved for a schedule that ends without completing all `n_steps`; resume to continue cooling from the last checkpoint. |
| `max_sweeps_reached` | no | DMRG-specific: sweep budget exhausted before convergence. Does not apply to XTRG. |
| `timeout` | yes | Job exceeded the requested walltime. Resuming continues from the last checkpoint for both DMRG (`main/dmrg.ckpt` or `main/artifacts/state.ckpt`) and XTRG (`main/xtrg.ckpt` + `main/thermal.ckpt`). |
| `out_of_memory` | yes | Job was killed by the out-of-memory handler. Resuming continues from the last checkpoint, as for `timeout`. |
| `nan_detected` | no | Numerical instability during the sweep (DMRG) or cooling step (XTRG). |
| `bad_parameters` | no | Configuration values rejected by the runner. |
| `checkpoint_missing` | yes | Expected restart checkpoint was not found. |
| `checkpoint_incompatible` | no | Saved checkpoint is incompatible with the current code. |
| `linear_algebra_error` | no | Low-level linear algebra routine failed. |
| `scheduler_failure` | yes | Slurm node failure or preemption. Resuming continues from the last checkpoint, as for `timeout`. |

Retryable reasons (marked yes) allow `iknot resume run` to create a new attempt with the same `config.toml`. Non-retryable reasons require changes to `config.toml` or the code before resubmitting.
