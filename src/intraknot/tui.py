# Copyright (C) 2026 Changkai Zhang.
#
# This file is part of IntraKnot.
#
# IntraKnot is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# IntraKnot is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with IntraKnot. If not, see <https://www.gnu.org/licenses/>.


"""Terminal dashboard for IntraKnot campaign and run monitoring.

Layout (three bands)
--------------------

    ┌─ campaign — N run(s) ─────────────────── page P/T ← → ─┐
    │  run_id                scan   state    iters  bond     │
    │  dmrg_heis_lx64…       sweep  ✓ done   100    128      │
    │  …                                                     │
    ├─ Detail: run_id ───────────────────────────────────────┤
    │  state: X  ·  reason: X  ·  attempt: XX  ·  restart: X │
    │  geo+model:  lattice=chain  lx=64  ·  J=1.0  spin=0.5  │
    │  algorithm:  engine=dmrg  max_bond=128  n_sweeps=100   │
    └────────────────────────────────────────────────────────┘
    [c] campaign  [r] refresh  [l] log  [←→] page  [q] quit

Key bindings
------------
- `c` — open campaign selector overlay.
- `r` — refresh all run data from disk.
- `l` — suspend TUI, open alice.log for the selected run in the configured
  editor, then resume.
- `y` — copy the selected run's run_id to the clipboard (via OSC 52; requires
  a terminal that supports it, e.g. iTerm2, Kitty, WezTerm, Windows Terminal).
- `← →` — page through the run list (6 rows per page).
- `↑ ↓` — move the row cursor (DataTable handles these natively).
- `q` — quit.

Editor configuration
--------------------
The editor used by `l` is read from `[tui] editor` in `configs/tui.toml`;
falls back to `"vi"` if the key or file is absent.
"""

from __future__ import annotations

import csv
import json
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import DataTable, Footer, Header, Label, ListItem, ListView, Static

from .config import resolve_active_campaign


def _read_json(path: Path) -> Optional[dict]:
    """Read a JSON file, returning `None` if the file is absent or invalid."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _current_attempt_dir(run_dir: Path) -> Optional[Path]:
    """Resolve `main/current` to an absolute attempt directory path.

    Reads the symlink target or falls back to `main/current.txt`.
    """
    main_dir = run_dir / "main"
    current = main_dir / "current"

    if current.is_symlink():
        target = (main_dir / current.readlink()).resolve()
        return target if target.exists() else None

    txt = main_dir / "current.txt"
    if txt.exists():
        name = txt.read_text().strip()
        candidate = main_dir / "attempts" / name
        return candidate if candidate.exists() else None

    return None


ROWS_PER_PAGE = 6

# Unicode icon per run state.
_STATE_ICONS: dict[str, str] = {
    "pending": "○",
    "running": "●",
    "completed": "✓",
    "failed": "✗",
    "invalid": "!",
    "skipped": "—",
    "cancelled": "⊘",
}

# Rich style per run state (applied to the state cell in the DataTable).
_STATE_STYLES: dict[str, str] = {
    "pending": "dim",
    "running": "bold yellow",
    "completed": "bold green",
    "failed": "bold red",
    "invalid": "bold red",
    "skipped": "dim",
    "cancelled": "dim",
}

# Color used for the status box border and inline Rich markup in the detail pane.
_STATE_COLORS: dict[str, str] = {
    "pending": "grey",
    "running": "yellow",
    "completed": "green",
    "failed": "red",
    "invalid": "red",
    "skipped": "grey",
    "cancelled": "grey",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RunRow:
    """All display-relevant data for one run."""

    run_id: str
    scan_id: str
    state: Optional[str] = None
    reason: Optional[str] = None
    current_attempt: Optional[str] = None
    restartable: bool = False
    nodename: Optional[str] = None
    iterations: Optional[int] = None
    max_bond_dim: Optional[int] = None
    geo_str: str = ""
    model_str: str = ""
    algo_str: str = ""
    iknot_log_path: Optional[Path] = None
    alice_log_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _fmt_section(data: dict) -> str:
    """Format a config section as left-aligned key = value lines.

    Shows all keys present in `data` in their natural (TOML insertion) order.
    Returns `"(no data)"` when the dict is empty.
    """
    if not data:
        return "(no data)"
    max_key = max(len(k) for k in data)
    return "\n".join(f"{k:<{max_key}} = {v}" for k, v in data.items())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run_rows(
    campaign_id: str,
    campaigns_root: Path,
    runs_root: Path,
) -> List[RunRow]:
    """Load all run data for a campaign into `RunRow` objects.

    Parameters
    ----------
    campaign_id:
        Campaign directory name.
    campaigns_root:
        Parent directory of campaign directories.
    runs_root:
        Parent directory of run directories.

    Returns
    -------
    list[RunRow]
        One entry per run registered in `runs.csv`, in file order.
    """
    campaign_dir = campaigns_root / campaign_id
    runs_csv = campaign_dir / "runs.csv"
    if not runs_csv.exists():
        return []

    with open(runs_csv, newline="") as f:
        csv_rows = list(csv.DictReader(f))

    rows: List[RunRow] = []
    for row in csv_rows:
        run_id = row.get("run_id", "").strip()
        if not run_id:
            continue

        scan_id = row.get("scan_id", "").strip()
        run_dir = runs_root / run_id

        # Main status (state, reason, current_attempt, restartable).
        status_data = _read_json(run_dir / "main" / "status.json") or {}
        state = status_data.get("state")
        reason = status_data.get("reason")
        current_attempt = status_data.get("current_attempt")
        restartable = bool(status_data.get("restartable", False))
        nodename: Optional[str] = status_data.get("nodename")

        # Resolve current attempt directory once; reused for both observables
        # and the log path to avoid the overhead of a second filesystem traversal.
        attempt_dir = _current_attempt_dir(run_dir)

        # Observables from the current attempt.
        info_data = {}
        if attempt_dir:
            info_data = _read_json(attempt_dir / "info.json") or {}

        # "iterations" generalizes DMRG's n_sweeps and XTRG's n_steps: both
        # are a plain count of the optimization/cooling steps completed by
        # the current attempt, just under a different name per engine.
        iterations: Optional[int] = info_data.get("n_sweeps")
        if iterations is None:
            iterations = info_data.get("n_steps")
        max_bond_dim: Optional[int] = info_data.get("max_bond_dim")

        # Scientific config (for detail pane).
        config_data: dict = {}
        config_path = run_dir / "config.toml"
        if config_path.exists():
            try:
                with open(config_path, "rb") as f:
                    config_data = tomllib.load(f)
            except Exception:
                pass

        # Log paths: iknot.log and alice.log inside the current attempt directory.
        iknot_log_path: Optional[Path] = None
        alice_log_path: Optional[Path] = None
        if attempt_dir:
            p = attempt_dir / "iknot.log"
            if p.exists():
                iknot_log_path = p
            p = attempt_dir / "alice.log"
            if p.exists():
                alice_log_path = p

        rows.append(RunRow(
            run_id=run_id,
            scan_id=scan_id,
            state=state,
            reason=reason,
            current_attempt=current_attempt,
            restartable=restartable,
            nodename=nodename,
            iterations=iterations,
            max_bond_dim=max_bond_dim,
            geo_str=_fmt_section(config_data.get("geometry", {})),
            model_str=_fmt_section(config_data.get("model", {})),
            algo_str=_fmt_section(config_data.get("algorithm", {})),
            iknot_log_path=iknot_log_path,
            alice_log_path=alice_log_path,
        ))

    return rows


def list_campaigns(campaigns_root: Path) -> List[str]:
    """Return sorted campaign IDs that have a `campaign.yaml` file."""
    if not campaigns_root.exists():
        return []
    return sorted(
        d.name
        for d in campaigns_root.iterdir()
        if d.is_dir() and (d / "campaign.yaml").exists()
    )


def _read_tui_config(configs_dir: Path) -> dict:
    """Read `configs/tui.toml`; return empty dict if absent or invalid."""
    path = configs_dir / "tui.toml"
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _resolve_active_campaign() -> Optional[str]:
    """Resolve the active campaign; delegates to `config.resolve_active_campaign`."""
    return resolve_active_campaign()


# ---------------------------------------------------------------------------
# Campaign selector overlay
# ---------------------------------------------------------------------------

class CampaignModal(ModalScreen):
    """Campaign selector overlay.

    Opens a list of all available campaigns. Press Enter or click a row to
    select; press Escape to cancel without changing the active campaign.
    """

    CSS = """
    CampaignModal {
        align: center middle;
    }
    #modal-box {
        width: 52;
        height: auto;
        max-height: 24;
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }
    #modal-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    ListView {
        height: auto;
        max-height: 18;
    }
    """

    def __init__(self, campaigns: List[str], current: Optional[str] = None) -> None:
        super().__init__()
        self._campaigns = campaigns
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label("Select Campaign", id="modal-title")
            yield ListView(
                *[ListItem(Label(c), id=f"camp-{c}") for c in self._campaigns]
            )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        if item_id.startswith("camp-"):
            self.dismiss(item_id[5:])

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)


# ---------------------------------------------------------------------------
# View-file selector overlay
# ---------------------------------------------------------------------------

class ViewModal(ModalScreen):
    """File viewer selector overlay.

    Displays a list of viewable files for the selected run. Files that do not
    exist on disk are shown dimmed with a `(not found)` suffix; selecting one
    dismisses the modal with `None` so the caller can notify the user.
    """

    CSS = """
    ViewModal {
        align: center middle;
    }
    #view-box {
        width: 52;
        height: auto;
        max-height: 24;
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }
    #view-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    ListView {
        height: auto;
        max-height: 18;
    }
    """

    def __init__(self, files: List[Tuple[str, Optional[Path]]]) -> None:
        super().__init__()
        # Each entry is (display label, resolved path or None if absent).
        self._files = files

    def compose(self) -> ComposeResult:
        with Vertical(id="view-box"):
            yield Label("View file", id="view-title")
            items = []
            for i, (label, path) in enumerate(self._files):
                if path is not None:
                    items.append(ListItem(Label(label), id=f"view-{i}"))
                else:
                    items.append(
                        ListItem(
                            Label(f"[dim]{label}  (not found)[/dim]"),
                            id=f"view-{i}",
                        )
                    )
            yield ListView(*items)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        if item_id.startswith("view-"):
            idx = int(item_id[5:])
            _, path = self._files[idx]
            self.dismiss(path)

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)
# ---------------------------------------------------------------------------

class RunDetail(Widget):
    """Detail pane with a status line and three bordered config boxes.

    Layout::

        ┌─ Detail: <run_id> ─────────────────────────────────────────┐
        │  state: X  ·  reason: X  ·  attempt: XX  ·  restartable: X │
        │ ┌─ Geometry ──┐ ┌─ Model ──────────┐ ┌─ Algorithm ───────┐ │
        │ │ lattice = … │ │ category = …     │ │ engine       = …  │ │
        │ │ lx      = … │ │ label    = …     │ │ max_bond     = …  │ │
        │ │ …           │ │ …                │ │ …                 │ │
        │ └─────────────┘ └──────────────────┘ └───────────────────┘ │
        └─────────────────────────────────────────────────────────────┘
    """

    DEFAULT_CSS = """
    RunDetail {
        height: auto;
        border: round $secondary;
        border-title-align: left;
        padding: 0;
    }
    #detail-status {
        height: 3;
        padding: 0 2;
        text-align: center;
        border: heavy $panel;
    }
    #detail-cols {
        height: auto;
    }
    .detail-box {
        width: 1fr;
        height: auto;
        min-height: 3;
        border: round $panel;
        padding: 0 1;
        border-title-align: left;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="detail-status")
        with Horizontal(id="detail-cols"):
            yield Static("", id="detail-geo", classes="detail-box")
            yield Static("", id="detail-model", classes="detail-box")
            yield Static("", id="detail-algo", classes="detail-box")

    def on_mount(self) -> None:
        self.border_title = "Detail"
        self.query_one("#detail-geo").border_title = "Geometry"
        self.query_one("#detail-model").border_title = "Model"
        self.query_one("#detail-algo").border_title = "Algorithm"

    def show_run(self, run: Optional[RunRow]) -> None:
        """Re-render the pane for a run, or show a placeholder.

        Parameters
        ----------
        run:
            Run to display, or `None` to show an empty placeholder.
        """
        status_widget = self.query_one("#detail-status", Static)
        geo_widget = self.query_one("#detail-geo", Static)
        model_widget = self.query_one("#detail-model", Static)
        algo_widget = self.query_one("#detail-algo", Static)

        if run is None:
            self.border_title = "Detail"
            status_widget.update("[dim]No run selected.[/dim]")
            status_widget.styles.border = ("heavy", "grey")
            geo_widget.update("")
            model_widget.update("")
            algo_widget.update("")
            return

        self.border_title = f"Detail: {run.run_id}"

        effective_state = run.state or "pending"
        color = _STATE_COLORS.get(effective_state, "grey")
        icon = _STATE_ICONS.get(effective_state, "?")

        reason_str = run.reason or "—"
        attempt_str = run.current_attempt or "—"
        restart_str = "yes" if run.restartable else "no"
        node_str = run.nodename or "—"

        # Colored markup: labels are dim, state/reason values are colored.
        status_markup = (
            f"[dim]state:[/dim] [{color} bold]{icon} {effective_state}[/]"
            f"  ·  [dim]node:[/dim] {node_str}"
            f"  ·  [dim]attempt:[/dim] [bold]{attempt_str}[/]"
            f"  ·  [dim]restartable:[/dim] {restart_str}"
        )
        status_widget.update(status_markup)
        # Border color tracks the run state for an immediate visual signal.
        status_widget.styles.border = ("heavy", color)

        geo_widget.update(run.geo_str)
        model_widget.update(run.model_str)
        algo_widget.update(run.algo_str)



# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


class DashboardApp(App):
    """IntraKnot campaign run dashboard.

    Parameters
    ----------
    campaigns_root:
        Directory containing campaign subdirectories.
        Defaults to `./campaigns` relative to the working directory.
    runs_root:
        Directory containing run subdirectories.
        Defaults to `./runs`.
    configs_dir:
        Directory containing `tui.toml` and other machine configs.
        Defaults to `./configs`.
    """

    TITLE = "iknot"

    BINDINGS = [
        Binding("c", "campaign", "Campaign"),
        Binding("r", "refresh", "Refresh"),
        Binding("v", "view_file", "View"),
        # l and a remain as direct shortcuts but are hidden from the footer;
        # the v overlay covers the same files and more. y is likewise hidden
        # to keep the footer compact.
        Binding("l", "open_iknot_log", show=False),
        Binding("a", "open_alice_log", show=False),
        Binding("y", "copy_run_id", show=False),
        # [ and ] are the clickable footer entries; key_display makes them
        # render as ← → so the UI stays intuitive.  The hidden priority
        # bindings on the actual arrow keys let users press ← → on the
        # keyboard without those events being consumed by DataTable first.
        Binding("bracketleft", "prev_page", "Prev page", key_display="←"),
        Binding("bracketright", "next_page", "Next page", key_display="→"),
        Binding("left", "prev_page", show=False, priority=True),
        Binding("right", "next_page", show=False, priority=True),
        Binding("q", "quit", "Quit"),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }
    DataTable {
        height: 10;
        border: round $primary;
        border-title-align: left;
        border-subtitle-align: right;
    }
    """

    def __init__(
        self,
        campaigns_root: Optional[Path] = None,
        runs_root: Optional[Path] = None,
        configs_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        cwd = Path.cwd()
        self._campaigns_root = campaigns_root or cwd / "campaigns"
        self._runs_root = runs_root or cwd / "runs"
        self._configs_dir = configs_dir or cwd / "configs"

        tui_cfg = _read_tui_config(self._configs_dir)
        self._editor: str = tui_cfg.get("tui", {}).get("editor", "vi")

        self._campaign_id: Optional[str] = _resolve_active_campaign()
        self._all_runs: List[RunRow] = []
        self._page: int = 0
        # Row cursor within the current page (0-based).
        self._cursor: int = 0

    # -----------------------------------------------------------------------
    # Compose / mount
    # -----------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="run-table", zebra_stripes=True, cursor_type="row")
        yield RunDetail(id="run-detail")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#run-table", DataTable)
        table.add_column("run_id", width=38)
        table.add_column("scan", width=16)
        table.add_column("state", width=13)
        table.add_column("iters", width=6)
        table.add_column("bond", width=6)
        self._reload()

    # -----------------------------------------------------------------------
    # Data loading and rendering
    # -----------------------------------------------------------------------

    def _reload(self) -> None:
        """Re-read all run data from disk and re-render the current page."""
        if self._campaign_id:
            self._all_runs = load_run_rows(
                self._campaign_id, self._campaigns_root, self._runs_root
            )
        else:
            self._all_runs = []

        ts = datetime.now().strftime("%H:%M:%S")
        campaign_label = self._campaign_id or "(no active campaign)"
        self.title = f"iknot · {campaign_label}"
        self.sub_title = f"last refresh: {ts}"
        self._render_page()

    def _render_page(self) -> None:
        """Clear and repopulate the DataTable with the current page's rows."""
        table = self.query_one("#run-table", DataTable)
        table.clear()

        n_total = len(self._all_runs)
        n_pages = max(1, (n_total + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)
        self._page = max(0, min(self._page, n_pages - 1))

        start = self._page * ROWS_PER_PAGE
        page_runs = self._all_runs[start : start + ROWS_PER_PAGE]

        for run in page_runs:
            effective_state = run.state or "pending"
            icon = _STATE_ICONS.get(effective_state, "?")
            style = _STATE_STYLES.get(effective_state, "")
            state_cell = Text(f"{icon} {effective_state}", style=style)

            iters_cell = (
                str(run.iterations) if run.iterations is not None else "—"
            )
            bond_cell = str(run.max_bond_dim) if run.max_bond_dim is not None else "—"

            rid = run.run_id
            if len(rid) > 36:
                rid = rid[:35] + "…"

            table.add_row(
                rid,
                run.scan_id or "—",
                state_cell,
                iters_cell,
                bond_cell,
                key=run.run_id,
            )

        campaign_label = self._campaign_id or "no campaign"
        table.border_title = f"{campaign_label}  —  {n_total} run(s)"
        table.border_subtitle = f"page {self._page + 1}/{n_pages}  ← →"

        # Reset cursor to the first row of the new page.
        self._cursor = 0
        if page_runs:
            table.move_cursor(row=0)
        self._update_detail()

    def _update_detail(self) -> None:
        """Refresh the detail pane from the current cursor position."""
        start = self._page * ROWS_PER_PAGE
        idx = start + self._cursor
        run = self._all_runs[idx] if 0 <= idx < len(self._all_runs) else None
        self.query_one(RunDetail).show_run(run)

    def _selected_run(self) -> Optional[RunRow]:
        """Return the currently highlighted run, or None."""
        start = self._page * ROWS_PER_PAGE
        idx = start + self._cursor
        return self._all_runs[idx] if 0 <= idx < len(self._all_runs) else None

    # -----------------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------------

    def action_refresh(self) -> None:
        """Reload run data from disk."""
        self._reload()

    def action_prev_page(self) -> None:
        """Go to the previous page."""
        if self._page > 0:
            self._page -= 1
            self._render_page()

    def action_next_page(self) -> None:
        """Go to the next page."""
        n_pages = max(1, (len(self._all_runs) + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)
        if self._page < n_pages - 1:
            self._page += 1
            self._render_page()

    def action_campaign(self) -> None:
        """Open the campaign selector overlay."""
        campaigns = list_campaigns(self._campaigns_root)
        self.push_screen(
            CampaignModal(campaigns, current=self._campaign_id),
            self._on_campaign_selected,
        )

    def _on_campaign_selected(self, campaign_id: Optional[str]) -> None:
        """Handle a campaign selection from the modal."""
        if campaign_id and campaign_id != self._campaign_id:
            self._campaign_id = campaign_id
            self._page = 0
            self._cursor = 0
            self._reload()

    def action_open_iknot_log(self) -> None:
        """Suspend the TUI, open iknot.log in the configured editor, then resume."""
        self._open_log_file("iknot_log_path", "iknot.log")

    def action_open_alice_log(self) -> None:
        """Suspend the TUI, open alice.log in the configured editor, then resume."""
        self._open_log_file("alice_log_path", "alice.log")

    def _open_log_file(self, attr: str, filename: str) -> None:
        """Shared logic for opening a log file in the configured editor."""
        run = self._selected_run()
        if run is None:
            self.notify("No run selected.", severity="warning")
            return
        log_path: Optional[Path] = getattr(run, attr)
        if log_path is None:
            self.notify(
                f"No {filename} found for {run.run_id}.", severity="warning"
            )
            return
        # suspend() hands the terminal back to the shell for the duration of
        # the editor subprocess and resumes the TUI when the editor exits.
        with self.suspend():
            subprocess.run([self._editor, str(log_path)], check=False)

    def action_copy_run_id(self) -> None:
        """Copy the selected run's run_id to the clipboard via OSC 52.

        OSC 52 is relayed by the terminal emulator, so this only reaches the
        clipboard when the terminal in use supports it (most do, notably
        excluding macOS Terminal.app). This is the mechanism that also works
        when `iknot tui` is run over SSH, since the escape sequence is
        forwarded to the local terminal on the client side.
        """
        run = self._selected_run()
        if run is None:
            self.notify("No run selected.", severity="warning")
            return
        self.copy_to_clipboard(run.run_id)
        self.notify(f"Copied {run.run_id}")

    def action_view_file(self) -> None:
        """Open the file viewer overlay for the selected run."""
        run = self._selected_run()
        if run is None:
            self.notify("No run selected.", severity="warning")
            return
        self.push_screen(
            ViewModal(self._build_view_file_list(run)), self._on_view_file_selected
        )

    def _build_view_file_list(self, run: RunRow) -> List[Tuple[str, Optional[Path]]]:
        """Build the ordered list of (label, path) pairs for the view overlay.

        A `None` path means the file does not currently exist on disk.
        For Slurm output/error files (whose names embed the job ID), the most
        recently modified match from `main/logs/` is used.

        Parameters
        ----------
        run:
            Currently selected run.

        Returns
        -------
        List[Tuple[str, Optional[Path]]]
            Display label paired with the resolved path (or `None`).
        """
        run_dir = self._runs_root / run.run_id
        files: List[Tuple[str, Optional[Path]]] = []

        # info.json from the current attempt directory (first in the list).
        attempt_dir = _current_attempt_dir(run_dir)
        if attempt_dir is not None:
            info_path: Optional[Path] = attempt_dir / "info.json"
            files.append(("info.json", info_path if info_path.exists() else None))
        else:
            files.append(("info.json", None))

        # Algorithm logs from the current attempt directory.
        files.append(("iknot.log", run.iknot_log_path))
        files.append(("alice.log", run.alice_log_path))

        # Main status file.
        status_path = run_dir / "main" / "status.json"
        files.append(("status.json", status_path if status_path.exists() else None))

        # Slurm stdout and stderr — glob for the most recently modified file of
        # each kind since the job ID is embedded in the filename at submit time.
        logs_dir = run_dir / "main" / "logs"
        for suffix in ("out", "err"):
            if logs_dir.exists():
                matches = sorted(
                    logs_dir.glob(f"slurm-*.{suffix}"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            else:
                matches = []
            files.append((f"slurm .{suffix}", matches[0] if matches else None))

        return files

    def _on_view_file_selected(self, path: Optional[Path]) -> None:
        """Open the file chosen in the view overlay, or notify if absent.

        Parameters
        ----------
        path:
            Resolved file path returned by `ViewModal`, or `None` when the
            user cancelled or selected a file that does not exist.
        """
        if path is None:
            return
        if not path.exists():
            self.notify(f"{path.name} not found.", severity="warning")
            return
        with self.suspend():
            subprocess.run([self._editor, str(path)], check=False)

    @on(DataTable.RowHighlighted, "#run-table")
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Sync internal cursor and update detail pane on DataTable navigation."""
        table = self.query_one("#run-table", DataTable)
        self._cursor = table.cursor_row
        self._update_detail()
