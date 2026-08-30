"""
Server selector sidebar with checkboxes.

Shows a list of discovered GPU servers with [x]/[ ] toggles.
Space key toggles monitoring on/off.
"""

from __future__ import annotations

from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Label, Static

from ..models import DiskInfo
from .gpu_bar import _bar_text, level_style


class ServerItem(Static, can_focus=True):
    """A single server row with checkbox and label.

    Messages bubble naturally to parent widgets via Textual's DOM,
    so the App can handle ServerItem.Toggled directly.
    """

    class Toggled(Message):
        """Emitted when a server is toggled on/off. Bubbles up."""

        def __init__(self, host: str, enabled: bool) -> None:
            super().__init__()
            self.host = host
            self.enabled = enabled

    def __init__(self, host: str, label: str, enabled: bool = False) -> None:
        super().__init__()
        self.host = host
        self.server_label = label
        self._enabled = enabled
        self._status: str = ""
        self._ip: str = ""
        self._user: str = ""
        self._disks: list[DiskInfo] = []
        # Remote machines start expanded (like in the screenshot);
        # the local machine stays collapsed.
        self._disks_open: bool = host != "local"
        self._cpu_desc: str = ""

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_status(self, status: str) -> None:
        """Show a small status indicator after the label."""
        self._status = status
        self.refresh()

    def set_ssh(self, ip: str, user: str) -> None:
        """SSH connection info: IP and user, one line each."""
        self._ip = ip
        self._user = user
        self.refresh()

    def set_disks(self, disks: list[DiskInfo]) -> None:
        self._disks = disks
        # Always relayout: expanded content changes the row height, and
        # refresh() without layout=True clips new lines (auto height).
        self.refresh(layout=True)

    def set_cpu(self, desc: str) -> None:
        self._cpu_desc = desc
        self.refresh()

    def toggle(self) -> None:
        """Toggle monitoring state."""
        self._enabled = not self._enabled
        self.refresh()
        self.post_message(self.Toggled(self.host, self._enabled))

    def render(self) -> Text:
        """Render with rich Text styles — identical palette to the right
        hand hardware rows (level_style, 4 tiers)."""
        check = "◉" if self._enabled else "○"
        check_style = Style(bold=True, color="green") if self._enabled else Style(color="bright_black")
        cursor = "▸" if self.has_focus else " "
        cursor_style = Style(color="cyan")
        t = Text()
        t.append(cursor, style=cursor_style)
        t.append(" ", style=cursor_style)
        t.append(check, style=check_style)
        t.append(f" {self.server_label}", style=None)
        if self._status:
            t.append(f" {self._status}", style=None)
        if self._ip or self._user:
            t.append(f"\n    {self._user or '?'}@{self._ip or '?'}", style=Style(color="bright_black"))
        if self._disks_open and self._disks:
            t.append_text(self._disks_text())
        return t

    # ── CPU + disk usage (visible while this row is focused AND expanded) --

    def _disks_text(self) -> Text:
        """CPU line + per-disk blocks fitted to the actual sidebar width."""
        out = Text()
        width = self.size.width if self.size and self.size.width else 50
        pad = width - 10  # indent(4) + padding slack
        bar_w = max(10, width - 8)

        dim = Style(color="bright_black")
        if self._cpu_desc:
            out.append(f"\n    {self._cpu_desc}", style=dim)

        for d in self._disks:
            label = d.name + (f" ({d.kind})" if d.kind else "")
            total_g = d.total_mb / 1024
            used_g = d.used_mb / 1024
            pct = d.percent
            style = level_style(pct)
            line = Text()
            line.append(f"    {label}", style=style)
            line.append(" " * max(1, pad - len(label) - len(f"{used_g:.1f}G/{total_g:.1f}G") - 5))
            line.append(f"{used_g:.1f}G/{total_g:.1f}G", style=style)
            line.append(" " * max(1, 5 - len(f"{pct:3.0f}%")))
            line.append(f"{pct:3.0f}%", style=style)
            out.append("\n")
            out.append_text(line)
            out.append("\n    ")
            out.append_text(_bar_text(pct, bar_w, style))
        return out

    def on_key(self, event: events.Key) -> None:
        """Space toggles the checkbox; Enter expands/collapses disks."""
        if event.key == "space":
            self.toggle()
            event.prevent_default()
            event.stop()
        elif event.key == "enter":
            self._disks_open = not self._disks_open
            self.refresh(layout=True)
            event.prevent_default()
            event.stop()


class ServerSelector(VerticalScroll):
    """Sidebar listing all discovered servers with toggles.

    Scrollable (mouse wheel + drag bar) once expanded items grow tall.
    Handles up/down arrow keys for navigating between ServerItem children.
    Messages from ServerItem bubble through here to the App automatically.
    """

    DEFAULT_CSS = """
    ServerSelector {
        width: 50;
        height: 1fr;
        border: solid $primary-background;
        padding: 1 0;
        scrollbar-size: 1 1;
        scrollbar-color: $primary 20%;
        scrollbar-color-active: $accent;
    }

    ServerItem {
        height: auto;
        padding: 0 1;
    }
    ServerItem:focus {
        background: $boost;
    }
    """

    def __init__(self, servers: list[tuple[str, str, bool]]) -> None:
        super().__init__()
        self._items: dict[str, ServerItem] = {}
        self._server_list = servers
        self._ordered: list[str] = []  # host order for navigation

    def compose(self) -> ComposeResult:
        yield Label("  [bold]Servers[/]")
        for host, label, enabled in self._server_list:
            item = ServerItem(host, label, enabled=enabled)
            self._items[host] = item
            self._ordered.append(host)
            yield item

    def get_item(self, host: str) -> ServerItem | None:
        return self._items.get(host)

    def update_status(self, host: str, status: str) -> None:
        """Update the status hint for a server item."""
        item = self._items.get(host)
        if item is not None:
            item.set_status(status)

    def update_ssh(self, host: str, ip: str, user: str) -> None:
        """Update the SSH connection info lines of a server item."""
        item = self._items.get(host)
        if item is not None:
            item.set_ssh(ip, user)

    def update_disks(self, host: str, disks: list) -> None:
        """Update the disk usage list of a server item."""
        item = self._items.get(host)
        if item is not None:
            item.set_disks(disks)

    def update_cpu(self, host: str, desc: str) -> None:
        """Update the CPU description line of a server item."""
        item = self._items.get(host)
        if item is not None:
            item.set_cpu(desc)

    def on_key(self, event: events.Key) -> None:
        """Arrow keys navigate between server items."""
        if event.key not in ("up", "down"):
            return

        focused = self.screen.focused
        current_host = None
        if isinstance(focused, ServerItem):
            current_host = focused.host

        # Find current index (or -1 if nothing focused)
        try:
            idx = self._ordered.index(current_host) if current_host else -1
        except ValueError:
            idx = -1

        if event.key == "up":
            idx = max(0, idx - 1) if idx >= 0 else 0
        else:  # down
            idx = min(len(self._ordered) - 1, idx + 1)

        if 0 <= idx < len(self._ordered):
            target = self._items.get(self._ordered[idx])
            if target is not None:
                target.focus()

        event.prevent_default()
        event.stop()
