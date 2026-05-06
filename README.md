# Intraknot

Intraknot is a lightweight Python tool for managing tensor-network simulations on HPC clusters. It is the orchestration layer around an existing tensor-network package: it creates simulation directories, records parameters, submits Slurm jobs, tracks retries, collects results, and connects related runs through campaigns.

Intraknot is built on top of [Nicole](https://github.com/Ideogenesis-AI/Nicole) and [Alice](https://github.com/Ideogenesis-AI/Alice), and integrates PyTorch for machine-learning-assisted many-body physics.

## Design philosophy

The design is intentionally compact. Intraknot does not try to be a general data-management framework. It supports the practical needs of tensor-network HPC work — campaigns, Slurm arrays, restartability, failure classification, post-processing, and revisitability — with the minimum structure necessary.

The key rule is: **same scientific definition → new attempt; changed scientific definition → new run**.

## Core concepts

```
campaign  a scientific group or parameter study
run       one simulation case, usually one parameter point
main      the primary calculation of a run
task      optional sub-calculation inside a run
attempt   one execution try of main or a task
```

Most calculations follow the simple path `campaign → run → main → attempt`. The optional `tasks/` layer is available when a run has multiple independently tracked sub-calculations (e.g., disorder samples or measurement chunks), but should not be used otherwise.

## Repository layout

```
intraknot/
├── src/
│   └── intraknot/
│       ├── __init__.py
│       ├── cli.py           # command-line interface
│       ├── config.py        # machine and run configuration
│       ├── launch.py        # directory creation and job submission
│       ├── collect.py       # result and status collection
│       ├── retry.py         # new-attempt creation for failed runs
│       ├── status.py        # status model definitions
│       └── templates/
│           └── submit_array.slurm
├── configs/                 # machine, path, and scheduler settings
├── campaigns/               # scientific groupings and run indexes
├── runs/                    # actual simulation cases
├── notebooks/               # inspection, comparison, and plotting
└── scripts/                 # small shell wrappers around the CLI
```

### Configuration

`configs/` holds machine and execution settings — scratch paths, Slurm account, partition, default walltime, Python executable. It does not hold scientific or model configurations.

```yaml
# configs/machine.yaml (example)
machine: cluster_a
paths:
  run_root: /scratch/user/intraknot/runs
slurm:
  account: my_account
  partition: normal
  default_time: "04:00:00"
```

### Campaigns

A campaign records which runs belong together and why.

```
campaigns/heisenberg_dmrg_chi_scan/
├── campaign.yaml       # scientific purpose
├── runs.csv            # parameter table and per-run status
├── submit_array.slurm  # optional Slurm array script
└── notes.md
```

`runs.csv` maps array indices to run directories and tracks status:

```csv
array_id,run_id,L,chi,g,status
1,heis_L64_chi064_g1.0,64,64,1.0,completed
2,heis_L64_chi128_g1.0,64,128,1.0,completed
3,heis_L64_chi256_g1.0,64,256,1.0,failed
```

### Runs

A run is one simulation case, typically one parameter point.

```
runs/heis_L64_chi128_g1.0/
├── manifest.yaml        # run identity, campaign, code version, machine
├── config.yaml          # exact scientific configuration (source of truth)
├── submit/              # Slurm script and recorded job ID
├── logs/                # scheduler-level logs
├── main/
│   ├── status.json
│   ├── attempts/
│   │   └── attempt_01/
│   │       ├── log.txt
│   │       ├── checkpoint.h5
│   │       ├── state.h5
│   │       ├── observables.json
│   │       └── convergence.csv
│   └── current -> attempts/attempt_01
└── summary/
    ├── observables.json
    └── status.json
```

`config.yaml` is the source of truth for the scientific configuration of a run and must not be silently modified after the run is created.

### Status model

Valid states: `pending`, `running`, `completed`, `failed`, `invalid`, `skipped`, `cancelled`.

The distinction between `failed` and `invalid` is intentional:

```
failed   execution failed; may be retryable with a new attempt
invalid  parameters or inputs are wrong; do not retry unchanged
```

Common tensor-network-specific failure reasons: `timeout`, `out_of_memory`, `nan_detected`, `not_converged`, `max_sweeps_reached`, `bad_parameters`, `checkpoint_missing`, `scheduler_failure`.

### Restart policy

| Situation | Action |
|---|---|
| Timeout, OOM, node failure, preemption | New attempt in the same `main/` |
| Changed Hamiltonian, lattice size, bond dimension, algorithm | New run in the same campaign |
| Bug fix that changes scientific results | New run |
| Wrong `config.yaml` | Mark run as `invalid`; create corrected run |

## Installation

```bash
uv sync
```

## Development

Run the test suite:

```bash
UV_NO_SYNC=1 uv run pytest
```

Run the linter:

```bash
UV_NO_SYNC=1 uv run ruff check src/ tests/
UV_NO_SYNC=1 uv run mypy src/
```

## License

GNU General Public License v3 or later. See [LICENSE](LICENSE).
