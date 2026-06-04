# `iknot collect`

Harvests results from completed runs and writes them into each run's `summary/` directory. Also updates the `status` column in the campaign's `runs.csv` so that the campaign-level overview is always current.

## Common workflow

```bash
# Collect results for all runs in the active campaign at once.
iknot collect campaign

# Or collect a single run.
iknot collect run chi128
```

Run `iknot collect campaign` after a batch of jobs finishes to synchronise `runs.csv` and populate `summary/` before inspecting results in a notebook or running `iknot status`.

## Reference

### `iknot collect run <RUN_ID>`

Reads `main/status.json` and the current attempt's `info.json`, then writes `summary/status.json` and `summary/info.json`. Prints the run state and energy to stdout.

#### Synopsis

```
iknot collect run [OPTIONS] RUN_ID
```

#### Options

| Option | Default | Description |
|---|---|---|
| `--runs-root PATH` | `runs` | Parent directory for runs. |

#### Output

```
Run chi128: state=completed energy=-0.4431471805
```

The files written under `summary/`:

| File | Contents |
|---|---|
| `summary/status.json` | Subset of `main/status.json` (run_id, state, current_attempt, reason, restartable). |
| `summary/info.json` | Copy of the current attempt's `info.json` (energy, converged, and any other fields written by the runner). |

---

### `iknot collect campaign`

Calls `collect run` for every run listed in the campaign's `runs.csv`, then updates the `status` column in `runs.csv` in-place with the collected state.

#### Synopsis

```
iknot collect campaign [OPTIONS]
```

#### Options

| Option | Default | Description |
|---|---|---|
| `--id TEXT` | active campaign | Campaign ID. |
| `--campaigns-root PATH` | `campaigns` | Parent directory for campaigns. |
| `--runs-root PATH` | `runs` | Parent directory for runs. |

#### Output example

```
  chi128: completed
  chi256: completed
  chi512: running

Collected 3 run(s).
```

After running `iknot collect campaign`, the `status` column of `runs.csv` reflects the current state of every run, making it easy to inspect the campaign summary at a glance or load it into a notebook.
