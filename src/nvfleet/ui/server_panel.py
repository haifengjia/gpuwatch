"""
Single-server GPU panel widget.

Renders GPU utilization bars, memory usage, temperatures, power draw,
ENC/DEC activity, and per-GPU expandable process lists.
Keyboard: up/down select a GPU row, Enter/Space expand/collapse its
process list (own processes green with full cmdline; others dim),
Esc clears the selection everywhere.
"""

from __future__ import annotations

import time

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import events
from textual.widgets import Static

from ..models import GPUInfo, HostMetrics, ServerSnapshot

from .gpu_bar import (
    _LEVEL_COLORS,
    _format_mem,
    freq_bar,
    level_from_values,
    media_str,
    mem_labeled_bar,
    memory_bar,
    percent_labeled_bar,
    power_bar,
    temp_bar,
    utilization_bar,
    watts_bar,
)


def _truncate(text: str, max_len: int = 70) -> str:
    """Truncate a string if too long, appending '…'."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


class ServerPanel(Static):
    """A panel displaying one server's GPU status."""

    can_focus = True

    def __init__(self, host: str, label: str) -> None:
        super().__init__("")
        self._host = host
        self._label = label
        self._snapshot: ServerSnapshot | None = None
        self.name_width: int = 10  # set by Dashboard, updated dynamically
        self._selected: int | None = None  # selected GPU index (None = none)
        self._expanded: set[int] = set()  # expanded GPU indexes

    @property
    def host(self) -> str:
        return self._host

    def clear_selection(self) -> None:
        """Reset to the initial state: nothing selected, nothing expanded."""
        if self._selected is None and not self._expanded:
            return
        self._selected = None
        self._expanded.clear()
        self.refresh(layout=True)

    def update_snapshot(self, snapshot: ServerSnapshot) -> None:
        """Update with a new snapshot and re-render."""
        self._snapshot = snapshot
        if snapshot.gpus:
            if self._selected is not None:
                self._selected = min(self._selected, len(snapshot.gpus) - 1)
            self._expanded = {g for g in self._expanded if g < len(snapshot.gpus)}
        self.refresh(layout=True)

    def update_name_width(self, width: int) -> None:
        """Compatibility hook used by Dashboard."""
        self.name_width = width
        self.refresh(layout=True)

    # ── keyboard: select / expand / collapse ----------------------

    def on_key(self, event: events.Key) -> None:
        if self._snapshot is None or not self._snapshot.gpus:
            return
        gpu_count = len(self._snapshot.gpus)
        if event.key in ("down", "up"):
            if self._selected is None:
                self._selected = 0
            else:
                step = 1 if event.key == "down" else -1
                self._selected = max(0, min(gpu_count - 1, self._selected + step))
            self.refresh(layout=True)
            event.prevent_default()
            event.stop()
        elif event.key == "escape":
            self.clear_selection()
            event.prevent_default()
            event.stop()
        elif event.key in ("enter", "space"):
            if self._selected is None:
                self._selected = 0
                self.refresh(layout=True)
                event.prevent_default()
                event.stop()
                return
            gpu = self._snapshot.gpus[self._selected]
            idx = gpu.index
            if idx in self._expanded:
                self._expanded.discard(idx)
            else:
                self._expanded.add(idx)
            self.refresh(layout=True)
            event.prevent_default()
            event.stop()

    # ── rendering ------------------------------------------------

    def render(self) -> Panel:
        if self._snapshot is None:
            return Panel(
                Text("Waiting for first poll...", style="dim"),
                title=self._label,
                border_style="bright_black",
            )

        snap = self._snapshot
        border_style = "#ffa500" if self._selected is not None else "bright_black"
        return Panel(
            self._build_content(snap),
            title=self._build_title(snap),
            border_style=border_style,
        )

    def _build_title(self, snap: ServerSnapshot) -> Text:
        """Build the panel title line: 'two4090     OK  42ms  12:31:04'."""
        title = Text()
        title.append(snap.label, style="bold cyan")

        status_colors = {
            "ok": "green",
            "connecting": "yellow",
            "timeout": "red",
            "stale": "yellow",
            "error": "red",
            "auth_error": "red",
            "no_python": "red",
            "down": "red",
        }
        color = status_colors.get(snap.status, "red")
        status_labels = {
            "ok": "OK",
            "connecting": "CONNECTING",
            "timeout": "TIMEOUT",
            "stale": "STALE",
            "error": "ERROR",
            "auth_error": "AUTH ERR",
            "no_python": "NO PYTHON",
            "down": "DOWN",
        }
        label = status_labels.get(snap.status, snap.status.upper())
        title.append(f"  {label}", style=f"bold {color}")

        if snap.latency_ms is not None:
            title.append(f"  {snap.latency_ms:.0f}ms", style="bright_black")

        if snap.updated_at:
            ts = time.strftime("%H:%M:%S", time.localtime(snap.updated_at))
            title.append(f"  {ts}", style="bright_black")

        return title

    def _build_content(self, snap: ServerSnapshot) -> Table:
        """Build a Rich Table of GPU rows + expandable process details."""
        if snap.status == "connecting":
            t = Table(show_header=False, expand=True, box=None)
            t.add_row(Text("Connecting...", style="yellow"))
            return t

        gpu_table = self._build_full(snap)

        if snap.error:
            wrapper = Table(show_header=False, expand=True, box=None)
            err_text = Text("Error: ", style="bold red")
            err_text.append(snap.error, style="red")
            wrapper.add_row(err_text)
            if snap.gpus:
                wrapper.add_row(Text(""))
                wrapper.add_row(Text("Showing last known data:", style="dim"))
            wrapper.add_row(Text(""))
            wrapper.add_row(gpu_table)
            return wrapper

        return gpu_table

    def _build_full(self, snap: ServerSnapshot) -> Table:
        """Host row + GPU metric rows + per-GPU process detail (expandable)."""
        wrapper = Table(show_header=False, expand=True, box=None, padding=(0, 1))
        wrapper.add_column("body", justify="left")

        host_row = self._build_host_row(snap.host_metrics)
        if host_row is not None:
            wrapper.add_row(host_row)
            wrapper.add_row(Text(""))
        wrapper.add_row(self._gpu_grid(snap))

        for gpu in snap.gpus:
            if gpu.index in self._expanded:
                wrapper.add_row(Text(""))
                wrapper.add_row(self._build_proc_table(gpu))
            else:
                summary = self._build_summary(gpu)
                if summary is not None:
                    wrapper.add_row(Text(""))
                    wrapper.add_row(summary)

        return wrapper

    def _gpu_row_color(self, gpu: GPUInfo) -> str:
        """Short-circuit worst-case color over temp / util / memory."""
        sev = 0
        for value in (gpu.temperature_c, gpu.utilization_gpu, gpu.memory_percent):
            if value is None or value <= 0:
                continue
            sev = max(sev, level_from_values(value))
        return _LEVEL_COLORS[sev]

    def _marker(self, gpu_index: int, selected: bool) -> tuple[str, str]:
        """Marker cell: ▣ outline for the selected row, ▾ expanded, ▸ idle."""
        if gpu_index in self._expanded:
            return "▾", "cyan"
        if selected:
            return "▣", "bold white"
        return "▸", "cyan"

    def _gpu_grid(self, snap: ServerSnapshot) -> Table:
        """GPU summary grid with selection marker and ENC/DEC columns."""
        grid = Table(show_header=False, expand=True, box=None, padding=0)
        grid.add_column("sel", width=1, justify="left")
        grid.add_column("gpu", width=5, justify="left")
        grid.add_column("name", width=self.name_width, justify="left")
        grid.add_column("util", width=15, justify="left")
        grid.add_column("mem", width=36, justify="left")
        grid.add_column("temp", width=9, justify="left")
        grid.add_column("power", width=11, justify="left")
        grid.add_column("clock", width=13, justify="left")
        grid.add_column("enc", width=13, justify="left")
        grid.add_column("dec", width=13, justify="left")

        for gpu in snap.gpus:
            marker, marker_style = self._marker(
                gpu.index, selected=(gpu.index == self._selected)
            )
            row_color = self._gpu_row_color(gpu)
            row = [
                Text(marker, style=marker_style),
                Text(f"GPU {gpu.index}", style=f"bold {row_color}"),
                Text(gpu.name, style="white"),
                utilization_bar(gpu.utilization_gpu, width=11),
                memory_bar(gpu.memory_used_mb, gpu.memory_total_mb, width=18),
                temp_bar(gpu.temperature_c),
                power_bar(gpu.power_watts, gpu.power_limit_watts),
                freq_bar(gpu.graphics_clock_mhz, gpu.graphics_clock_max_mhz or None, width=6),
                media_str("ENC", gpu.encoder_util, gpu.encoder_sessions),
                media_str("DEC", gpu.decoder_util, gpu.decoder_sessions),
            ]
            grid.add_row(*row)

        return grid

    def _build_proc_table(self, gpu: GPUInfo) -> Table:
        """Full process list for one expanded GPU (real table columns)."""
        table = Table(
            show_header=True,
            header_style="bold underline",
            box=None,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("PID", width=7, justify="left")
        table.add_column("USER", width=10, justify="left")
        table.add_column("MEM", justify="right")
        table.add_column("SM", justify="right")
        table.add_column("MEM%", justify="right")
        table.add_column("ENC", justify="right")
        table.add_column("DEC", justify="right")
        table.add_column("COMMAND")

        def pct_text(value: int | None) -> Text:
            if value is None:
                return Text("-", style="bright_black")
            return Text(f"{value}%", style="green")

        procs = sorted(gpu.processes, key=lambda p: -p.gpu_memory_mb)
        for p in procs:
            cmd = _truncate(p.display_name(), 80)
            table.add_row(
                Text(f"{p.pid}", style="dim"),
                Text(p.user or "?", style="dim"),
                Text(_format_mem(p.gpu_memory_mb), style="dim"),
                pct_text(p.sm_percent),
                pct_text(p.mem_percent),
                pct_text(p.enc_percent),
                pct_text(p.dec_percent),
                Text(cmd, style="green" if p.is_own else "dim"),
            )

        if gpu.other_users:
            table.add_row(Text(""))
            for ou in gpu.other_users:
                mem_str = _format_mem(ou.total_memory_mb)
                table.add_row(Text(
                    f"{ou.user}: {ou.process_count} proc, {mem_str}",
                    style="dim",
                ))

        return table

    def _build_summary(self, gpu: GPUInfo) -> Text | None:
        """One-line process summary for a collapsed GPU."""
        own = [p for p in gpu.processes if p.is_own]
        other = gpu.other_users
        if not own and not other:
            return None
        parts: list[str] = []
        if own:
            total = sum(p.gpu_memory_mb for p in own)
            parts.append(f"{len(own)} own proc, {_format_mem(total)}")
        for ou in other:
            parts.append(
                f"{ou.user}: {ou.process_count} proc, {_format_mem(ou.total_memory_mb)}"
            )
        return Text(f"  GPU {gpu.index}  " + " | ".join(parts), style="dim")

    def _build_host_row(self, host: HostMetrics | None) -> Text | None:
        """One-line host summary above the GPU grid."""
        if host is None:
            return None
        if (
            host.cpu_percent is None
            and host.memory_used_mb is None
            and host.temp_c is None
        ):
            return None

        line = Text()
        if host.cpu_percent is not None:
            line.append_text(percent_labeled_bar("CPU", host.cpu_percent, width=6))
        if host.temp_c is not None:
            line.append(Text("   "))
            line.append_text(temp_bar(host.temp_c, width=5))
        if host.cpu_power_watts is not None:
            line.append(Text("   "))
            line.append_text(watts_bar(host.cpu_power_watts, host.cpu_power_max_watts, width=6))
        if host.cpu_freq_mhz is not None:
            line.append(Text("   "))
            line.append_text(freq_bar(host.cpu_freq_mhz, host.cpu_freq_max_mhz, width=6))
        if host.memory_used_mb is not None and host.memory_total_mb:
            line.append(Text("   "))
            line.append_text(mem_labeled_bar("RAM", host.memory_used_mb, host.memory_total_mb, width=10))
        if host.swap_used_mb is not None and host.swap_total_mb:
            line.append(Text("   "))
            line.append_text(mem_labeled_bar("SWAP", host.swap_used_mb, host.swap_total_mb, width=8))
        if host.driver_version:
            line.append(Text("   "))
            line.append(Text("NV" + host.driver_version, style="dim"))
        if host.cuda_versions:
            line.append(Text("   "))
            cuda_parts = [f"{v}*" if d else v for v, d in host.cuda_versions]
            line.append(Text("CUDA " + ", ".join(cuda_parts), style="dim"))
        return line
