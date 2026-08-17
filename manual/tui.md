# `iknot tui`

Launches an interactive terminal dashboard for monitoring a campaign's runs without leaving the shell. The dashboard reads data directly from the run directories on disk and requires no background process.

## Common workflow

```bash
# Activate a campaign first, then launch the dashboard.
iknot campaign activate heisenberg_dmrg_chi_scan
iknot tui
```

The dashboard opens on the active campaign (resolved from `INTRAKNOT_CAMPAIGN` or `.iknot_state`). Use `c` to switch campaigns without restarting.

## Layout

```
╔═ heisenberg_dmrg_chi_scan — 6 run(s) ══════════════════════════ page 1/2 ═╗
║  run_id                         scan         state     iters  bond        ║
║  dmrg_heis_lx64_chi128_a3f7b2…  chi_study    ✓ done    100    128         ║
║  dmrg_heis_lx64_chi256_c91d4e…  chi_study    ✓ done    120    256         ║
║  …                                                                        ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  ┌── state: done · reason: converged · attempt: 01 · restartable: no ──┐  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
║  ┌─ Geometry ──────┐  ┌─ Model ─────────────────┐  ┌─ Algorithm ───────┐  ║
║  │ lattice = chain │  │ category  = bosonic     │  │ engine   = dmrg   │  ║
║  │ lx      = 64    │  │ label     = Heisenberg  │  │ max_bond = 128    │  ║
║  │ bcx     = OBC   │  │ symmetry  = U1          │  │ n_sweeps = 20     │  ║
║  │ n2x     = true  │  │ spin      = 0.5         │  │ e_tol    = 1e-8   │  ║
║  └─────────────────┘  │ J         = 1.0         │  └───────────────────┘  ║
║                       └─────────────────────────┘                         ║
╚═══════════════════════════════════════════════════════════════════════════╝
  c Campaign  r Refresh  l iknot.log  a alice.log  ← Prev page  → Next page  q Quit
```

The status box border is color-coded by run state: green for `completed`, yellow for `running`, red for `failed`, grey for `pending` or `cancelled`.

## Reference

### Synopsis

```
iknot tui [OPTIONS]
```

### Options

| Option | Default | Description |
|---|---|---|
| `--campaigns-root PATH` | `campaigns` | Directory containing campaign subdirectories. |
| `--runs-root PATH` | `runs` | Directory containing run subdirectories. |
| `--configs-dir PATH` | `configs` | Directory containing `tui.toml` and other machine configs. |

### Key bindings

| Key | Action |
|---|---|
| `c` | Open campaign selector overlay. Arrow keys navigate; Enter switches. |
| `r` | Refresh all run data from disk (re-reads `runs.csv`, `status.json`, `info.json`). |
| `v` | Open the file viewer overlay for the selected run (see [View overlay](#view-overlay)). |
| `l` | Open `iknot.log` for the selected run in the configured editor (shortcut; also available via `v`). |
| `a` | Open `alice.log` for the selected run in the configured editor (shortcut; also available via `v`). |
| `← →` (or `[ ]`) | Page through the run list (6 runs per page). |
| `↑ ↓` | Move the row cursor within the current page. |
| `q` | Quit. |

The `←` / `→` arrow keys and `[` / `]` both perform page navigation. The footer shows `←` / `→` labels for the corresponding bindings; clicking them in a mouse-enabled terminal also works.

### View overlay

Pressing `v` opens a file picker listing all viewable files for the selected run. Select a file with Enter to open it in the configured editor. Files that do not yet exist on disk are shown but cannot be opened (a warning notification appears instead).

Files are listed in the following order:

| File | Location |
|---|---|
| `info.json` | `main/current/info.json` — DMRG observables written by Alice at the end of a sweep. |
| `iknot.log` | `main/current/iknot.log` — combined `INFO`-level log from IntraKnot and Alice. |
| `alice.log` | `main/current/alice.log` — `DEBUG`-level Alice output (verbose, timestamped). |
| `status.json` | `main/status.json` — run state machine status. |
| `slurm .out` | `main/logs/slurm-<job>.out` — Slurm stdout (most recent job). |
| `slurm .err` | `main/logs/slurm-<job>.err` — Slurm stderr (most recent job). |

### Run table columns

| Column | Source | Notes |
|---|---|---|
| `run_id` | `runs.csv` | Truncated if long. |
| `scan` | `runs.csv` | Scan identifier; `—` if not part of a scan. |
| `state` | `main/status.json` | Color-coded with a state icon: ○ pending, ● running, ✓ done, ✗ failed. |
| `iters` | `main/current/info.json` | Number of optimization/cooling steps completed (`n_sweeps` for DMRG, `n_steps` for XTRG). |
| `bond` | `main/current/info.json` | Peak bond dimension reached. |

### Detail pane

The detail pane updates whenever the row cursor moves and shows three sections for the highlighted run:

- **Status box**: `state`, `reason`, `current_attempt`, `restartable`. The box border is color-coded to match the run state.
- **Geometry** box: all keys from the `[geometry]` section of `config.toml`, aligned by `=`.
- **Model** box: all keys from the `[model]` section of `config.toml`, aligned by `=`.
- **Algorithm** box: all keys from the `[algorithm]` section of `config.toml`, aligned by `=`.

All parameters are shown in the order they appear in `config.toml`; no keys are dropped regardless of the model type.

### Log viewer

`l` opens `iknot.log` and `a` opens `alice.log` for the selected run. Both files are read from the current attempt directory (`main/current/`, a symlink to the most recent attempt).

- `iknot.log` — `INFO`-level messages from the IntraKnot runner and Alice (combined).
- `alice.log` — `DEBUG`-level output from the Alice library (timestamped, more verbose).

The TUI suspends cleanly, passes the terminal to the editor, and resumes when the editor exits. If the log file does not exist (for example, the run has never started), a notification is shown and the editor is not launched.

### Configuration

The editor is read from `configs/tui.toml`:

```toml
[tui]
editor = "vi"  # alternatives: "hx", "nano", "vim", "nvim", etc.
```

Falls back to `vi` if the file or key is absent. `iknot init` creates `configs/tui.toml` with `editor = "vi"` as the default; edit this file to set your preferred editor before using the log viewer.
