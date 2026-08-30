"""
Server selector sidebar with checkboxes.

Shows a list of discovered GPU servers with [x]/[ ] toggles.
Space key toggles monitoring on/off.
"""

from __future__ import annotations

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Label, Static

from ..models import DiskInfo
from .gpu_bar import _LEVEL_COLORS, level_from_values


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
        if self.has_focus:
            self.refresh(layout=True)
        else:
            self.refresh()

    def toggle(self) -> None:
        """Toggle monitoring state."""
        self._enabled = not self._enabled
        self.refresh()
        self.post_message(self.Toggled(self.host, self._enabled))

    def render(self) -> str:
        check = "[bold green]◉[/]" if self._enabled else "[dim]○[/]"
        has_focus = "[cyan]▸[/]" if self.has_focus else " "
        status = f" {self._status}" if self._status else ""
        head = f"{has_focus} {check} {self.server_label}{status}"
        summary = ""
        if self._ip:
            summary += f"\n    [dim]{self._ip}[/]"
        if self._user:
            summary += f"\n    [dim]{self._user}[/]"
        if self.has_focus and self._disks:
            summary += self._render_disks()
        return head + summary

    # ── disk usage (only while this row is focused/selected) -------

    _BAR_WIDTH = 44

    def _render_disks(self) -> str:
        out = ""
        for d in self._disks:
            label = d.name + (f" ({d.kind})" if d.kind else "")
            total_g = d.total_mb / 1024
            used_g = d.used_mb / 1024
            pct = d.percent
            outer = int(round(pct / 100.0 * self._BAR_WIDTH))
            outer = min(outer, self._BAR_WIDTH)
            color = _LEVEL_COLORS[level_from_values(pct)]
            bar = (
                f"    [{color}]{'█' * outer}[/]"
                f"[bright_black]{'░' * (self._BAR_WIDTH - outer)}[/]"
            )
            label_line = (
                f"    [white]{label:<18}[/]"
                f"[dim]{used_g:7.1f} GiB / {total_g:7.1f} GiB        [/]"
                f"[green]{pct:3.0f}%[/]"
            )
            out += "\n" + label_line
            out += "\n" + bar
        return out

    def on_key(self, event: events.Key) -> None:
        """Space toggles the checkbox."""
        if event.key == "space":
            self.toggle()
            event.prevent_default()
            event.stop()


class ServerSelector(Vertical):
    """Sidebar listing all discovered servers with toggles.

    Handles up/down arrow keys for navigating between ServerItem children.
    Messages from ServerItem bubble through here to the App automatically.
    """

    DEFAULT_CSS = """
    ServerSelector {
        width: 50;
        height: 1fr;
        border: solid $primary-background;
        padding: 1 0;
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
