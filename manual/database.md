# `iknot database`

An algorithm database is a GitHub repository that provides a curated collection of reusable algorithm scripts — observables, model definitions, colormaps, post-processing utilities, and similar components. Scripts are catalogued in a `registry.yaml` manifest at the repository root.

IntraKnot communicates with databases purely over HTTP. It fetches the `registry.yaml` manifest to discover available scripts, then fetches individual files on demand. No git cloning is performed.

The aggregated manifest for all registered databases is stored in `configs/registry.yaml`. This file is created by `iknot init` and updated by `iknot database add` and `iknot database update`.

## Common workflow

```bash
# 1. Register the community database.
iknot database add intraknot-database https://github.com/Ideogenesis-AI/IntraKnot-Database

# 2. Browse available scripts.
iknot database list

# 3. Install a script into the active campaign.
iknot campaign install intraknot-database:observables/spin_corr.py

# 4. Keep your local manifest up to date when new scripts are published.
iknot database update
```

## Reference

### `iknot database add <NAME> <URL>`

Register a new algorithm database and fetch its manifest.

#### Synopsis

```
iknot database add [OPTIONS] NAME URL
```

`NAME` is the short identifier used in source descriptors (e.g. `"intraknot-database"`).
`URL` is the GitHub repository URL (e.g. `https://github.com/Ideogenesis-AI/IntraKnot-Database`).

The remote `registry.yaml` is fetched immediately. If the repository does not yet exist the command fails with a clear error message. The database is not registered until a successful fetch.

#### Options

| Option | Default | Description |
| --- | --- | --- |
| `--machine PATH` | `./configs` | Path to the `configs/` directory. |

#### What it does

1. Derives the raw `registry.yaml` URL from the GitHub repository URL.
2. Fetches and parses the remote manifest.
3. Appends the database entry to `configs/registry.yaml`.

#### Notes

- The name `intraknot` is reserved for the built-in package source and cannot be registered.
- Only GitHub repository URLs are currently supported.

---

### `iknot database remove <NAME>`

Remove a registered database from the registry.

#### Synopsis

```
iknot database remove [OPTIONS] NAME
```

#### Options

| Option | Default | Description |
| --- | --- | --- |
| `--machine PATH` | `./configs` | Path to the `configs/` directory. |

This command removes the entry from `configs/registry.yaml`. It does **not** affect any scripts already installed in campaigns — those are tracked independently in each campaign's `algorithm.lock`.

---

### `iknot database list`

List all registered databases and their contents.

#### Synopsis

```
iknot database list [OPTIONS]
```

#### Options

| Option | Default | Description |
| --- | --- | --- |
| `--machine PATH` | `./configs` | Path to the `configs/` directory. |

#### Output example

```
intraknot-database
  URL         : https://github.com/Ideogenesis-AI/IntraKnot-Database
  Last fetched: 2026-06-11
  Description : Community algorithm scripts for IntraKnot
  observables (3 scripts)
    observables/spin_corr.py      Spin-spin correlation function
    observables/entanglement.py   Entanglement entropy profile
    observables/current.py        Bond current observable
```

---

### `iknot database update [NAME]`

Re-fetch the registry manifest for one or all databases.

#### Synopsis

```
iknot database update [OPTIONS] [NAME]
```

When `NAME` is given, only that database is refreshed. When omitted, all registered databases are updated.

#### Options

| Option | Default | Description |
| --- | --- | --- |
| `--machine PATH` | `./configs` | Path to the `configs/` directory. |

Run this command after the remote database adds new scripts, or to ensure your local `configs/registry.yaml` reflects the latest available content.

---

## The `registry.yaml` manifest format

Each database must provide a `registry.yaml` at its repository root with the following structure:

```yaml
description: Human-readable description of the database

categories:
  observables:
    description: Observable measurement scripts
    scripts:
      - name: spin_corr
        file: observables/spin_corr.py
        description: Spin-spin correlation function ⟨SᵢSⱼ⟩
        sha256: <hex digest of the current file content>

  models:
    description: Custom Hamiltonian definitions
    scripts:
      - name: bose_hubbard
        file: models/bose_hubbard.py
        description: Bose-Hubbard model with interaction and hopping terms
        sha256: <hex digest>
```

Scripts in different categories are accessed via source descriptors of the form `<db-name>:<file>`, for example:

```
intraknot-database:observables/spin_corr.py
intraknot-database:models/bose_hubbard.py
```

The `sha256` field in `registry.yaml` is used by `iknot campaign sync` to detect upstream changes without downloading the file body, enabling efficient staleness checks.

---

## Source descriptors

Scripts from any database (including the built-in `intraknot` package) are identified by a source descriptor of the form `<db-name>:<path>`:

| Source descriptor | Resolved from |
| --- | --- |
| `intraknot:algorithm/run_dmrg.py` | Built-in IntraKnot package (via `importlib.resources`) |
| `intraknot-database:observables/spin_corr.py` | Remote HTTP fetch from the registered database |
| `my-local-db:utils/helper.py` | Remote HTTP fetch from a custom registered database |

Source descriptors are recorded in each campaign's `algorithm.lock` file, so the origin of every managed script is always traceable.
