"""
Rich bar rendering for GPU utilization and memory.

Produces colored progress bar strings suitable for Rich renderables.
Uses manual Unicode block characters for reliable rendering across Rich versions.
"""

from __future__ import annotations

from rich.style import Style
from rich.text import Text


def level_style(percent: float) -> Style:
    """Shared 4-tier scale (temperature palette):
    blue < 50, green < 70, yellow < 85, red >= 85."""
    if percent < 50:
        return Style(color="blue")
    elif percent < 70:
        return Style(color="green")
    elif percent < 85:
        return Style(color="yellow")
    else:
        return Style(color="red")


def level_name(percent: float) -> str:
    """Level color name, matching level_style."""
    if percent < 50:
        return "blue"
    elif percent < 70:
        return "green"
    elif percent < 85:
        return "yellow"
    else:
        return "red"


def _bar_style(percent: float) -> Style:
    """Backwards-compatible alias of level_style."""
    return level_style(percent)


_LEVEL_COLORS = ("blue", "green", "yellow", "red")


def level_from_values(*values: float | int | None) -> int:
    """Short-circuit (worst-case) severity 0-3 across metrics:
    returns the most urgent tier among all supplied values."""
    sev = 0
    for v in values:
        if v is None:
            continue
        name = level_name(float(v))
        if name in _LEVEL_COLORS:
            sev = max(sev, _LEVEL_COLORS.index(name))
    return sev


_EMPTY_STYLE = Style(color="bright_black")


def percent_labeled_bar(
    label: str, percent: float, width: int = 6, color: str | None = None
) -> Text:
    """Generic labeled bar: 'CPU ▓▓▓░░░ 38%'."""
    style = Style(color=color) if color else level_style(percent)
    result = Text(f"{label} ", style=style)
    result.append_text(_bar_text(percent, width, style))
    result.append(f" {percent:3.0f}%", style=style)
    return result


def mem_labeled_bar(
    label: str, used_mb: int, total_mb: int, width: int = 10, color: str | None = None
) -> Text:
    """Labeled memory bar: 'RAM ▓░░ 21.18GiB / 31.12GiB'."""
    if total_mb <= 0:
        return Text(f"{label} ─ N/A", style=_EMPTY_STYLE)
    pct = (used_mb / total_mb) * 100.0
    style = Style(color=color) if color else level_style(pct)
    result = Text(f"{label} ", style=style)
    result.append_text(_bar_text(pct, width, style))
    result.append(f" {_format_mem(used_mb)} / {_format_mem(total_mb)}", style=style)
    return result


def _bar_text(percent: float, width: int, fill_style: Style) -> Text:
    """Bar with bright filled cells and dim empty cells, like nvitop."""
    pct = max(0.0, min(float(percent), 100.0))
    filled = int(round(pct / 100.0 * width))
    filled = min(filled, width)
    result = Text()
    if filled:
        result.append("█" * filled, style=fill_style)
    if width - filled:
        result.append("░" * (width - filled), style=_EMPTY_STYLE)
    return result


def _render_bar(percent: float, width: int) -> str:
    """Build a bar string with █ (filled) and ░ (empty) characters."""
    pct = max(0.0, min(percent, 100.0))
    filled = int(round(pct / 100.0 * width))
    filled = min(filled, width)
    empty = width - filled
    return "█" * filled + "░" * empty


def utilization_bar(percent: int, width: int = 10, color: str | None = None) -> Text:
    """Render a utilization bar like: '████████░░ 72%'."""
    pct = max(0, min(percent, 100))
    style = Style(color=color) if color else level_style(pct)
    result = _bar_text(pct, width, style)
    result.append(f" {pct:3d}%", style=style)
    return result


def _format_mem(mb: int) -> str:
    """Format memory value in GiB (0.00-999.00), both used and totals."""
    return f"{mb / 1024:.2f}GiB"


def memory_bar(used_mb: int, total_mb: int, width: int = 16, color: str | None = None) -> Text:
    """Render a memory bar like: '████████░░░░ 21.24GiB / 23.99GiB'."""
    if total_mb <= 0:
        return Text("─" * width + " N/A")
    pct = (used_mb / total_mb) * 100.0
    style = Style(color=color) if color else level_style(pct)
    result = _bar_text(pct, width, style)
    used_str = _format_mem(used_mb)
    total_str = _format_mem(total_mb)
    result.append(f" {used_str} / {total_str}", style=style)
    return result


def temp_bar(celsius: int, width: int = 5, color: str | None = None) -> Text:
    """Temperature bar with 100°C ceiling (4-tier palette)."""
    if celsius <= 0:
        return Text("░" * width + " N/A", style=_EMPTY_STYLE)
    style = Style(color=color) if color else level_style(celsius)
    result = _bar_text(celsius, width, style)
    result.append(f" {celsius}°C", style=style)
    return result


def power_bar(watts: float, limit_watts: float, width: int = 6, color: str | None = None) -> Text:
    """Power bar scaled to the GPU's own power limit (4-tier palette)."""
    if limit_watts <= 0:
        result = Text("░" * width, style=_EMPTY_STYLE)
        result.append(f" {watts:.0f}W", style=_EMPTY_STYLE)
        return result
    ratio = (watts / limit_watts) * 100
    style = Style(color=color) if color else level_style(ratio)
    result = _bar_text(ratio, width, style)
    result.append(f" {watts:.0f}W", style=style)
    return result


def freq_bar(current: int, maximum: int | None = None, width: int = 6) -> Text:
    """Frequency bar like '████░░ 2100MHz'. Max is queried at runtime."""
    if not current:
        return Text("░" * width + "    -", style=_EMPTY_STYLE)
    pct = (current / maximum) * 100 if maximum else 0.0
    style = level_style(pct)
    result = _bar_text(pct, width, style)
    result.append(f" {current}MHz", style=style)
    return result


def watts_bar(watts: float | None, max_watts: float | None = None, width: int = 6) -> Text:
    """Power bar like '████░░ 87W'. No label, matches the hardware row."""
    if watts is None:
        return Text("░" * width + "    -", style=_EMPTY_STYLE)
    pct = (watts / max_watts) * 100 if max_watts else 0.0
    style = level_style(pct)
    result = _bar_text(pct, width, style)
    result.append(f" {watts:.0f}W", style=style)
    return result


def media_str(label: str, util: int, sessions: int = 0, bar_width: int = 4) -> Text:
    """Render ENC/DEC cell like: 'ENC ███░ 75% x2'.

    Colored (cyan for encoder, magenta for decoder) only when active;
    dimmed otherwise. Sessions appended when > 0.
    """
    if util <= 0 and sessions <= 0:
        style = Style(color="bright_black")
        result = Text(f"{label} ", style=style)
        result.append_text(Text("░" * bar_width, style=_EMPTY_STYLE))
        result.append(" 0%", style=style)
        return result

    style = level_style(util)
    pct = max(0, min(util, 100))
    result = Text(f"{label} ", style=style)
    result.append_text(_bar_text(pct, bar_width, style))
    result.append(f" {pct:2d}%", style=style)
    if sessions:
        result.append(f" x{sessions}", style=style)
    return result



