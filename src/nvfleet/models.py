"""
Data models for GPU monitoring.

All data is ephemeral — received as JSON from remote probes,
parsed in memory, rendered to TUI. Nothing persists to disk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class GPUProcess:
    """A process running on a specific GPU."""

    pid: int
    name: str
    gpu_memory_mb: int
    user: str | None = None
    cmdline: str | None = None  # full command line (own processes only)
    sm_percent: int | None = None  # SM utilization 0-100, None if unreported
    mem_percent: int | None = None  # memory bandwidth %
    enc_percent: int | None = None  # encoder %
    dec_percent: int | None = None  # decoder %
    own: bool = False  # owned by the local viewer

    @classmethod
    def from_probe(cls, data: dict[str, Any], own_user: str | None = None) -> GPUProcess:
        user = data.get("user")
        return cls(
            pid=data["pid"],
            name=data.get("name", "?"),
            gpu_memory_mb=data["gpu_memory_mb"],
            user=user,
            cmdline=data.get("cmdline"),
            sm_percent=data.get("sm_percent"),
            mem_percent=data.get("mem_percent"),
            enc_percent=data.get("enc_percent"),
            dec_percent=data.get("dec_percent"),
            own=own_user is not None and user == own_user,
        )

    @property
    def is_own(self) -> bool:
        return self.own

    def display_name(self) -> str:
        """Full command line for own processes, else short name."""
        if self.cmdline:
            return self.cmdline
        return self.name


@dataclass
class OtherUserMemory:
    """Aggregated memory usage by another user on a GPU."""

    user: str
    process_count: int
    total_memory_mb: int

    @classmethod
    def from_probe(cls, data: dict[str, Any]) -> OtherUserMemory:
        return cls(
            user=data["user"],
            process_count=data["process_count"],
            total_memory_mb=data["total_memory_mb"],
        )


@dataclass
class GPUInfo:
    """Snapshot of a single GPU's state."""

    index: int
    uuid: str
    name: str
    utilization_gpu: int  # 0–100
    utilization_mem: int  # 0–100
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    temperature_c: int
    power_watts: float
    power_limit_watts: float
    encoder_util: int = 0  # NVENC utilization % (device-wide)
    encoder_sessions: int = 0
    encoder_avg_fps: int = 0
    decoder_util: int = 0  # NVDEC utilization % (device-wide)
    decoder_sessions: int = 0
    graphics_clock_mhz: int = 0
    graphics_clock_max_mhz: int = 0
    processes: list[GPUProcess] = field(default_factory=list)
    other_users: list[OtherUserMemory] = field(default_factory=list)

    @property
    def memory_percent(self) -> float:
        """Memory usage as percentage."""
        if self.memory_total_mb == 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100

    @property
    def has_media_activity(self) -> bool:
        """True if any ENC/DEC metric is non-zero (used to show media column)."""
        return (
            self.encoder_util > 0
            or self.decoder_util > 0
            or self.encoder_sessions > 0
            or self.decoder_sessions > 0
        )

    @classmethod
    def from_probe(cls, data: dict[str, Any], own_user: str | None = None) -> GPUInfo:
        processes = [
            GPUProcess.from_probe(p, own_user=own_user) for p in data.get("processes", [])
        ]
        other_users = [
            OtherUserMemory.from_probe(o) for o in data.get("other_users", [])
        ]
        return cls(
            index=data["index"],
            uuid=data.get("uuid", "unknown"),
            name=data.get("name", "unknown"),
            utilization_gpu=data.get("utilization_gpu", 0),
            utilization_mem=data.get("utilization_mem", 0),
            memory_total_mb=data.get("memory_total_mb", 0),
            memory_used_mb=data.get("memory_used_mb", 0),
            memory_free_mb=data.get("memory_free_mb", 0),
            temperature_c=data.get("temperature_c", 0),
            power_watts=data.get("power_watts", 0.0),
            power_limit_watts=data.get("power_limit_watts", 0.0),
            encoder_util=data.get("encoder_util", 0),
            encoder_sessions=data.get("encoder_sessions", 0),
            encoder_avg_fps=data.get("encoder_avg_fps", 0),
            decoder_util=data.get("decoder_util", 0),
            decoder_sessions=data.get("decoder_sessions", 0),
            graphics_clock_mhz=data.get("graphics_clock_mhz", 0),
            graphics_clock_max_mhz=data.get("graphics_clock_max_mhz", 0),
            processes=processes,
            other_users=other_users,
        )


@dataclass
class DiskInfo:
    """One real block-device mount with usage."""

    mount_point: str
    used_mb: int
    total_mb: int
    kind: str | None = None  # 'hdd' | 'ssd' | None

    @property
    def name(self) -> str:
        """Short display name: 'Root' for '/', basename capitalized else."""
        if self.mount_point == "/":
            return "Root"
        base = self.mount_point.rstrip("/").rsplit("/", 1)[-1]
        return base.capitalize() if base else self.mount_point

    @property
    def percent(self) -> float:
        if self.total_mb <= 0:
            return 0.0
        return (self.used_mb / self.total_mb) * 100

    @classmethod
    def from_probe(cls, data: dict[str, Any]) -> DiskInfo:
        return cls(
            mount_point=data["mount_point"],
            used_mb=data.get("used_mb", 0),
            total_mb=data.get("total_mb", 0),
            kind=data.get("kind"),
        )


@dataclass
class HostMetrics:
    """Host-level (non-GPU) metrics: CPU, temperature, RAM, swap."""

    cpu_percent: float | None = None
    cpu_freq_mhz: int | None = None
    cpu_freq_max_mhz: int | None = None
    cpu_power_watts: float | None = None
    cpu_power_max_watts: float | None = None
    memory_used_mb: int | None = None
    memory_total_mb: int | None = None
    swap_used_mb: int | None = None
    swap_total_mb: int | None = None
    temp_c: int | None = None
    driver_version: str | None = None
    cuda_versions: list[tuple[str, bool]] = field(default_factory=list)
    disks: list[DiskInfo] = field(default_factory=list)

    @property
    def memory_percent(self) -> float | None:
        if self.memory_used_mb is None or not self.memory_total_mb:
            return None
        return (self.memory_used_mb / self.memory_total_mb) * 100

    @classmethod
    def from_probe(cls, data: dict[str, Any] | None) -> HostMetrics | None:
        if not data:
            return None
        return cls(
            cpu_percent=data.get("cpu_percent"),
            cpu_freq_mhz=data.get("cpu_freq_mhz"),
            cpu_freq_max_mhz=data.get("cpu_freq_max_mhz"),
            cpu_power_watts=data.get("cpu_power_watts"),
            cpu_power_max_watts=data.get("cpu_power_max_watts"),
            memory_used_mb=data.get("memory_used_mb"),
            memory_total_mb=data.get("memory_total_mb"),
            swap_used_mb=data.get("swap_used_mb"),
            swap_total_mb=data.get("swap_total_mb"),
            temp_c=data.get("temp_c"),
            driver_version=data.get("driver_version"),
            cuda_versions=[
                (str(v), bool(d))
                for v, d in (data.get("cuda_versions") or [])
            ],
            disks=[DiskInfo.from_probe(d) for d in (data.get("disks") or [])],
        )


def gpu_short_name(name: str) -> str:
    """'NVIDIA GeForce RTX 3090' → 'RTX 3090'; 'NVIDIA A100...' → 'A100...'."""
    for prefix in ("NVIDIA GeForce ", "NVIDIA "):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


@dataclass
class HostHardware:
    """Static hardware summary of one machine (CPU / RAM / GPU list)."""

    cpu_cores: int | None = None
    cpu_model: str | None = None
    memory_total_mb: int | None = None

    @classmethod
    def from_probe(cls, data: dict[str, Any] | None) -> HostHardware | None:
        if not data:
            return None
        return cls(
            cpu_cores=data.get("cpu_cores"),
            cpu_model=data.get("cpu_model"),
            memory_total_mb=data.get("memory_total_mb"),
        )

    def cpu_brief(self) -> str:
        """Compact label like 'Intel Core i7-11700F (16c)' or 'AMD EPYC 7K62 (96c)'."""
        import re

        model = self.cpu_model or ""
        # '11th Gen Intel...' -> drop the generation prefix
        model = re.sub(r"^\s*[\d]+(st|nd|rd|th)?\s+Gen\s+", "", model, flags=re.I)
        # drop ' @ 2.50GHz' / trailing clock info
        model = re.sub(r"\s*@\s*[\d.]+\s*GHz.*$", "", model, flags=re.I)
        # de-corporate the tm/reg marks
        model = model.replace("(R)", "").replace("(TM)", "")
        model = re.sub(r"\s+", " ", model).strip()
        # trailing '-Core' / 'Processor' / 'CPU' words
        model = re.sub(
            r"\s+[\d]+-Core\s*$|\s+Processor\s*$|\s+Core\s*$|\s+CPU\s*$",
            "",
            model,
            flags=re.I,
        ).strip()

        parts = [model] if model else []
        if self.cpu_cores:
            parts.append(f"({self.cpu_cores}c)")
        return " ".join(parts)

    def gpu_desc(self, gpus: list[GPUInfo]) -> str:
        """'RTX 3090 ×2 | 5 proc running'."""
        from collections import Counter
        parts: list[str] = []
        counters = Counter(
            gpu_short_name(g.name)
            for g in gpus
            if g.name and g.name != "unknown"
        )
        for name, cnt in counters.items():
            parts.append(f"{name} ×{cnt}" if cnt > 1 else name)
        n_procs = sum(len(g.processes) for g in gpus)
        parts.append(f"{n_procs} proc running")
        return " | ".join(parts)


ServerStatus = Literal[
    "ok", "connecting", "timeout", "error", "stale", "auth_error", "no_python", "down"
]


@dataclass
class ServerSnapshot:
    """Snapshot of one server's GPU state at a point in time."""

    host: str
    label: str
    status: ServerStatus
    gpus: list[GPUInfo]
    error: str | None = None
    updated_at: float = 0.0
    latency_ms: float | None = None
    own_user: str | None = None
    host_metrics: HostMetrics | None = None
    hardware: HostHardware | None = None

    @classmethod
    def from_probe(
        cls,
        host: str,
        label: str,
        data: dict[str, Any],
        latency_ms: float,
        own_user: str | None = None,
    ) -> ServerSnapshot:
        """Build a snapshot from successful probe output."""
        gpus = [
            GPUInfo.from_probe(g, own_user=own_user) for g in data.get("gpus", [])
        ]
        return cls(
            host=host,
            label=label,
            status="ok",
            gpus=gpus,
            updated_at=time.time(),
            latency_ms=latency_ms,
            own_user=own_user,
            host_metrics=HostMetrics.from_probe(data.get("host")),
            hardware=HostHardware.from_probe(data.get("host")),
        )

    @classmethod
    def error_snapshot(
        cls,
        host: str,
        label: str,
        status: ServerStatus,
        error: str,
        previous: ServerSnapshot | None = None,
    ) -> ServerSnapshot:
        """Build a snapshot representing an error state, preserving old GPU data if available."""
        return cls(
            host=host,
            label=label,
            status=status,
            gpus=previous.gpus if previous else [],
            error=error,
            updated_at=time.time(),
            latency_ms=previous.latency_ms if previous else None,
        )


@dataclass
class ServerConfig:
    """Configuration for a monitored server."""

    host: str  # SSH alias (or "local")
    label: str  # display name
    enabled: bool = False  # whether polling is active
    ssh_user: str | None = None  # SSH login user (for process highlighting)
    transport: Literal["local", "ssh"] = "ssh"  # probe execution path
    hostname: str | None = None  # HostName for ping RTT (None = use host)
