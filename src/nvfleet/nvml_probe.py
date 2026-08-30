#!/usr/bin/env python3
"""
NVML GPU probe -- runs on remote server via SSH stdin, outputs JSON to stdout.

Zero dependencies: uses only Python stdlib ctypes + libnvidia-ml.so (part of
NVIDIA driver). No pip install required on the remote side.

Usage (on remote server):
    python3 nvml_probe.py          # runs locally
    ssh host 'python3 -' < nvml_probe.py   # sent over SSH stdin

Output: JSON object to stdout, diagnostics to stderr.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import sys
import time
from typing import Any


# ---------------------------------------------------------------------------
# NVML constants
# ---------------------------------------------------------------------------

NVML_TEMPERATURE_GPU = 0
NVML_SUCCESS = 0

# Error codes are resolved dynamically from nvmlErrorString at runtime:
# the numeric values differ between driver generations (older: 4 =
# INSUFFICIENT_SIZE / 7 = NO_PERMISSION; newer (580+): 7 =
# INSUFFICIENT_SIZE / 4 = NO_PERMISSION). Never hardcode them.

_NVML_ERRORS: dict[str, int] = {}


def resolve_nvml_error_codes(lib) -> None:
    """Fill _NVML_ERRORS from nvmlErrorString so we survive enum churn."""
    fn = lib.nvmlErrorString
    fn.restype = ctypes.c_char_p
    names = {
        "Success": "SUCCESS",
        "Insufficient Size": "INSUFFICIENT_SIZE",
        "Not Found": "NOT_FOUND",
        "Insufficient Permissions": "NO_PERMISSION",
        "Not Supported": "NOT_SUPPORTED",
    }
    for code in range(0, 64):
        s = fn(code)
        if not s:
            continue
        name = names.get(s.decode(errors="replace"))
        if name and name not in _NVML_ERRORS:
            _NVML_ERRORS[name] = code
    _NVML_ERRORS.setdefault("SUCCESS", 0)
    # Fallbacks for the unlikely case the driver has no string table:
    # default to the classic enum and also accept the swapped variant later.
    _NVML_ERRORS.setdefault("INSUFFICIENT_SIZE", 4)
    _NVML_ERRORS.setdefault("NO_PERMISSION", 7)


def nvml_error(name: str, *alt_codes: int) -> int:
    """Return the runtime error code for *name, or fall back to alt_codes."""
    code = _NVML_ERRORS.get(name)
    if code is not None:
        return code
    return alt_codes[0] if alt_codes else -1

NVML_DEVICE_NAME_BUFFER_SIZE = 96
NVML_DEVICE_UUID_BUFFER_SIZE = 96

# ---------------------------------------------------------------------------
# C struct definitions
# ---------------------------------------------------------------------------


class NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class NvmlUtilization(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]


class NvmlProcessInfo(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
    ]


class NvmlProcessInfoV3(ctypes.Structure):
    """Wider process info struct used by newer drivers (info streams include
    GPU/MIG instance ids). Reading only the first two fields is safe."""

    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


class NvmlProcessUtilizationSample(ctypes.Structure):
    """Per-process utilization sample returned by nvmlDeviceGetProcessUtilization.

    timeStamp is in microseconds since Unix epoch (CPU timestamp).
    All utilizations are in percent (0-100 for a whole device).
    """

    _fields_ = [
        ("pid", ctypes.c_uint),
        ("time_stamp", ctypes.c_ulonglong),
        ("sm_util", ctypes.c_uint),
        ("mem_util", ctypes.c_uint),
        ("enc_util", ctypes.c_uint),
        ("dec_util", ctypes.c_uint),
    ]


# ---------------------------------------------------------------------------
# NVML function signatures
# ---------------------------------------------------------------------------


def _setup_nvml(lib) -> None:
    """Declare argtypes/restype for all NVML functions used."""
    # Init / shutdown
    lib.nvmlInit.restype = ctypes.c_int
    lib.nvmlShutdown.restype = ctypes.c_int

    # Device count
    lib.nvmlDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_uint)]
    lib.nvmlDeviceGetCount.restype = ctypes.c_int

    # Device handle
    lib.nvmlDeviceGetHandleByIndex.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    lib.nvmlDeviceGetHandleByIndex.restype = ctypes.c_int

    # Device name
    lib.nvmlDeviceGetName.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    lib.nvmlDeviceGetName.restype = ctypes.c_int

    # Device UUID
    lib.nvmlDeviceGetUUID.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    lib.nvmlDeviceGetUUID.restype = ctypes.c_int

    # Memory info
    lib.nvmlDeviceGetMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NvmlMemory),
    ]
    lib.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int

    # Utilization rates
    lib.nvmlDeviceGetUtilizationRates.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NvmlUtilization),
    ]
    lib.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int

    # Temperature
    lib.nvmlDeviceGetTemperature.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint),
    ]
    lib.nvmlDeviceGetTemperature.restype = ctypes.c_int

    # Power usage (milliwatts)
    lib.nvmlDeviceGetPowerUsage.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
    ]
    lib.nvmlDeviceGetPowerUsage.restype = ctypes.c_int

    # Power management limit (milliwatts)
    lib.nvmlDeviceGetPowerManagementLimit.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
    ]
    lib.nvmlDeviceGetPowerManagementLimit.restype = ctypes.c_int

    # Compute / graphics running processes (count-first ABI on NVML 580+;
    # buffer is sized for the widest struct variant we know of).
    for func_name in (
        "nvmlDeviceGetComputeRunningProcesses",
        "nvmlDeviceGetGraphicsRunningProcesses",
    ):
        if hasattr(lib, func_name):
            func = getattr(lib, func_name)
            func.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(NvmlProcessInfoV3),
            ]
            func.restype = ctypes.c_int


# ---------------------------------------------------------------------------
# Process info helpers (pure Python, /proc filesystem)
# ---------------------------------------------------------------------------


def _proc_exists(pid: int) -> bool:
    """True when /proc/<pid> exists (same namespace as the probe)."""
    try:
        return os.path.isdir(f"/proc/{pid}")
    except OSError:
        return False


def _read_proc_comm(pid: int) -> str | None:
    """Read process name from /proc/<pid>/comm."""
    try:
        path = f"/proc/{pid}/comm"
        with open(path, "r") as f:
            return f.read().strip()
    except (OSError, PermissionError):
        return None


def _read_proc_cmdline(pid: int) -> str | None:
    """Read full command line from /proc/<pid>/cmdline.

    Arguments are separated by null bytes; we replace them with spaces.
    Returns None on failure (process exited, permission denied, etc.).
    """
    try:
        path = f"/proc/{pid}/cmdline"
        with open(path, "rb") as f:
            raw = f.read()
        if not raw:
            return None
        # Replace null bytes with spaces
        return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except (OSError, PermissionError):
        return None


def _read_proc_uid(pid: int) -> int | None:
    """Read UID (owner) of /proc/<pid>/status. Returns None on failure."""
    try:
        path = f"/proc/{pid}/status"
        with open(path, "r") as f:
            for line in f:
                if line.startswith("Uid:"):
                    # "Uid:\t1000\t1000\t1000\t1000"
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except (OSError, PermissionError):
        pass
    return None


def _uid_to_name(uid: int) -> str | None:
    """Convert numeric UID to username."""
    try:
        import pwd

        return pwd.getpwuid(uid).pw_name
    except (ImportError, KeyError):
        return str(uid)


def _run_nvsmi_processes() -> dict[str, list[dict[str, Any]]]:
    """Fallback: run nvidia-smi to get GPU process info.

    NVML process queries may return NVML_ERROR_NO_PERMISSION when the
    current user cannot read other users' process details. nvidia-smi
    handles this via driver-level access, so we use it as a fallback.

    Returns: dict mapping GPU UUID -> list of {pid, name, used_memory_mb}
    """
    import subprocess

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return {}

    result: dict[str, list[dict[str, Any]]] = {}
    for line in output.decode("utf-8", errors="replace").strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",", 3)]
        if len(parts) < 4:
            continue
        gpu_uuid, pid_str, proc_name, mem_str = parts
        try:
            pid = int(pid_str)
            mem_mb = int(mem_str)
        except ValueError:
            continue
        if gpu_uuid not in result:
            result[gpu_uuid] = []
        result[gpu_uuid].append(
            {"pid": pid, "name": proc_name, "used_memory_mb": mem_mb}
        )
    return result


def _try_nvml_v2_memory(lib, handle) -> tuple[int, int] | None:
    """Try NVML v2 memory info to get reserved field. Returns (used_mb, free_mb) or None."""
    try:
        func = lib.nvmlDeviceGetMemoryInfo_v2
    except AttributeError:
        return None

    class NvmlMemoryV2(ctypes.Structure):
        _fields_ = [
            ("version", ctypes.c_uint),
            ("_pad", ctypes.c_uint),
            ("total", ctypes.c_ulonglong),
            ("reserved", ctypes.c_ulonglong),  # must precede free/used per NVML ABI
            ("free", ctypes.c_ulonglong),
            ("used", ctypes.c_ulonglong),
        ]

    func.argtypes = [ctypes.c_void_p, ctypes.POINTER(NvmlMemoryV2)]
    func.restype = ctypes.c_int
    m = NvmlMemoryV2()
    # NVML versioned structs: version = sizeof(struct) | (major << 24)
    m.version = ctypes.sizeof(NvmlMemoryV2) | (2 << 24)
    rc = func(handle, ctypes.byref(m))
    if rc == NVML_SUCCESS:
        total_mb = int(m.total // (1024 * 1024))
        free_mb = int(m.free // (1024 * 1024))
        used_mb = int((m.total - m.free - m.reserved) // (1024 * 1024))
        return (used_mb, free_mb)
    return None


def _calibrate_reserved(lib, handle, count) -> dict[int, int]:
    """One-time: run nvidia-smi to get per-GPU reserved memory offsets.
    Returns {gpu_index: reserved_mb} so subsequent polls can compute
    user-visible used = NVML_total - NVML_free - reserved.
    """
    import subprocess
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=3,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return {}

    offsets: dict[int, int] = {}
    smi_mem: dict[int, int] = {}
    for line in output.decode("utf-8", errors="replace").strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                smi_mem[int(parts[0])] = int(parts[1])
            except ValueError:
                pass

    # For each GPU, compute reserved = NVML(total-free) - nvidia_smi(used)
    for i in range(count.value):
        if i not in smi_mem:
            continue
        test_handle = ctypes.c_void_p()
        rc = lib.nvmlDeviceGetHandleByIndex(i, ctypes.byref(test_handle))
        if rc != NVML_SUCCESS:
            continue
        mem = NvmlMemory()
        rc = lib.nvmlDeviceGetMemoryInfo(test_handle, ctypes.byref(mem))
        if rc != NVML_SUCCESS:
            continue
        nvml_used_mb = int((mem.total - mem.free) // (1024 * 1024))
        reserved = nvml_used_mb - smi_mem[i]
        if reserved > 0:
            offsets[i] = reserved

    return offsets


def _gpu_process_utilizations(lib, handle) -> dict[int, tuple[int, int, int, int]]:
    """Fetch per-process utilization samples via nvmlDeviceGetProcessUtilization.

    Returns {pid: (sm%, mem%, enc%, dec%)}. The driver keeps a ring buffer of
    samples stamped in microseconds since epoch; requesting ''all samples after
    now-1s'' yields the freshest sample per process (same approach as nvitop),
    so this stays stateless across probe invocations.

    Sample collection is scoped per cgroup/namespace on modern drivers, so
    other processes may be missing — returns whatever the driver reports.
    """
    fn = getattr(lib, "nvmlDeviceGetProcessUtilization", None)
    if fn is None:
        return {}
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NvmlProcessUtilizationSample),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.c_ulonglong,
    ]
    fn.restype = ctypes.c_int

    # µs since epoch, one second back to ensure the ring buffer has samples.
    last_seen = time.time_ns() // 1000 - 1_000_000

    count = ctypes.c_uint(0)
    try:
        rc = fn(handle, None, ctypes.byref(count), last_seen)
    except Exception:
        return {}
    if rc != nvml_error("INSUFFICIENT_SIZE", 4, 7) or not count.value:
        return {}

    buf = (NvmlProcessUtilizationSample * count.value)()
    try:
        rc = fn(handle, buf, ctypes.byref(count), last_seen)
    except Exception:
        return {}
    if rc != NVML_SUCCESS:
        return {}

    result: dict[int, tuple[int, int, int, int]] = {}
    for i in range(count.value):
        s = buf[i]
        result[s.pid] = (s.sm_util, s.mem_util, s.enc_util, s.dec_util)
    return result


def _get_gfx_clock(lib, handle) -> tuple[int | None, int | None]:
    """(current_graphics_mhz, max_graphics_mhz) via NVML (not hardcoded)."""
    fn = getattr(lib, "nvmlDeviceGetClockInfo", None)
    fn_max = getattr(lib, "nvmlDeviceGetMaxClockInfo", None)
    if fn is None:
        return None, None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
    fn.restype = ctypes.c_int
    if fn_max is not None:
        fn_max.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
        fn_max.restype = ctypes.c_int

    cur = ctypes.c_uint(0)
    try:
        rc = fn(handle, 1, ctypes.byref(cur))  # NVML_CLOCK_GRAPHICS
    except Exception:
        return None, None
    cur_mhz = cur.value if rc == NVML_SUCCESS else None

    max_mhz = None
    if fn_max is not None:
        m = ctypes.c_uint(0)
        try:
            rc = fn_max(handle, 1, ctypes.byref(m))
        except Exception:
            return cur_mhz, None
        max_mhz = m.value if rc == NVML_SUCCESS else None
    return cur_mhz, max_mhz


def _get_enc_util(lib, handle) -> int | None:
    """Encoder utilization percent, or None if unsupported/failed."""
    fn = getattr(lib, "nvmlDeviceGetEncoderUtilization", None)
    if fn is None:
        return None
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
    fn.restype = ctypes.c_int
    util = ctypes.c_uint(0)
    period = ctypes.c_uint(0)
    try:
        rc = fn(handle, ctypes.byref(util), ctypes.byref(period))
    except Exception:
        return None
    return util.value if rc == NVML_SUCCESS else None


def _get_dec_util(lib, handle) -> int | None:
    """Decoder utilization percent, or None if unsupported/failed."""
    fn = getattr(lib, "nvmlDeviceGetDecoderUtilization", None)
    if fn is None:
        return None
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
    fn.restype = ctypes.c_int
    util = ctypes.c_uint(0)
    period = ctypes.c_uint(0)
    try:
        rc = fn(handle, ctypes.byref(util), ctypes.byref(period))
    except Exception:
        return None
    return util.value if rc == NVML_SUCCESS else None


def _get_enc_stats(lib, handle) -> tuple[int, int, int] | None:
    """Encoder (sessionCount, averageFps, averageLatencyMs), or None."""
    fn = getattr(lib, "nvmlDeviceGetEncoderStats", None)
    if fn is None:
        return None
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_ulonglong),
    ]
    fn.restype = ctypes.c_int
    sessions = ctypes.c_ulonglong(0)
    fps = ctypes.c_ulonglong(0)
    latency = ctypes.c_ulonglong(0)
    try:
        rc = fn(handle, ctypes.byref(sessions), ctypes.byref(fps), ctypes.byref(latency))
    except Exception:
        return None
    if rc != NVML_SUCCESS:
        return None
    return (sessions.value, fps.value, latency.value)


def _get_dec_stats(lib, handle) -> tuple[int, int, int] | None:
    """Decoder (sessionCount, averageFps, averageLatencyMs), or None.

    Deprecated in newer NVML — return None if the symbol is unavailable.
    """
    fn = getattr(lib, "nvmlDeviceGetDecoderStats", None)
    if fn is None:
        return None
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_ulonglong),
    ]
    fn.restype = ctypes.c_int
    sessions = ctypes.c_ulonglong(0)
    fps = ctypes.c_ulonglong(0)
    latency = ctypes.c_ulonglong(0)
    try:
        rc = fn(handle, ctypes.byref(sessions), ctypes.byref(fps), ctypes.byref(latency))
    except Exception:
        return None
    if rc != NVML_SUCCESS:
        return None
    return (sessions.value, fps.value, latency.value)


def _nvml_proc_list(
    lib, handle, func_name: str
) -> tuple[list[tuple[int, int]], int]:
    """Fetch (pid, used_memory_bytes) tuples for one NVML proc-list API.

    Returns (pids, rc). Count-first ABI: first call with NULL info buffer
    returns NVML_ERROR_INSUFFICIENT_SIZE and the element count.
    """
    func = getattr(lib, func_name, None)
    if func is None:
        return [], -1

    count = ctypes.c_uint(0)
    try:
        rc = func(handle, ctypes.byref(count), None)
    except Exception:
        return [], -1
    if rc == NVML_SUCCESS:
        return [], rc  # no processes
    if rc != nvml_error("INSUFFICIENT_SIZE", 4, 7):
        return [], rc
    if count.value == 0:
        return [], rc

    buf = (NvmlProcessInfoV3 * (count.value + 8))()
    try:
        rc2 = func(handle, ctypes.byref(count), buf)
    except Exception:
        return [], -1
    if rc2 != NVML_SUCCESS:
        return [], rc2

    result = []
    for j in range(count.value):
        pi = buf[j]
        mem = pi.usedGpuMemory
        if mem >= (1 << 63):
            mem = 0  # NVML_VALUE_NOT_AVAILABLE
        # Older NvmlProcessInfo reading of a newer struct variant yields
        # garbage pids with 0 memory on some drivers — drop those silently
        # (a real GPU process practically always holds some memory).
        if pi.pid > (1 << 31) or mem == 0:
            continue
        result.append((pi.pid, mem))
    return result, rc2


def _gpu_processes(
    lib,
    handle,
    gpu_uuid: str,
    own_user: str | None,
    nvsmi_cache: list[dict[str, list[dict[str, Any]]] | None],
    util_map: dict[int, tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    """Collect ALL running processes for one GPU.

    NVML answers both compute and graphics contexts (FFmpeg NVENC/NVDEC,
    video decode processes, CUDA compute, etc.), merged by pid. Falls back
    to nvidia-smi (lazily fetched once and cached via nvsmi_cache) when
    NVML reports NO_PERMISSION.

    Every process is returned with pid/name/memory/user. cmdline is only
    populated for processes owned by ``own_user`` (/proc permissions).
    Utilization comes from ``util_map`` when the driver reports it.
    """
    raw_procs: list[dict[str, Any]] = []
    use_nvsmi = False

    for func_name in (
        "nvmlDeviceGetComputeRunningProcesses",
        "nvmlDeviceGetGraphicsRunningProcesses",
    ):
        try:
            pids, rc = _nvml_proc_list(lib, handle, func_name)
        except Exception:
            pids, rc = [], -1
        if rc == nvml_error("NO_PERMISSION", 7, 4):
            use_nvsmi = True
            continue
        if rc != NVML_SUCCESS and rc != nvml_error("INSUFFICIENT_SIZE", 4, 7):
            continue
        for pid, mem_bytes in pids:
            if not _proc_exists(pid):
                # Phantom rows from struct-layout slack on some drivers:
                # the pid does not exist in /proc, so it cannot be real.
                continue
            name = _read_proc_comm(pid)
            uid = _read_proc_uid(pid)
            username = _uid_to_name(uid) if uid is not None else None
            raw_procs.append(
                {
                    "pid": pid,
                    "gpu_memory_mb": int(mem_bytes // (1024 * 1024)),
                    "name": name or "?",
                    "user": username,
                }
            )
    # Deduplicate by pid (a process may appear in both lists):
    seen: set[int] = set()
    unique_procs: list[dict[str, Any]] = []
    for rp in raw_procs:
        if rp["pid"] not in seen:
            seen.add(rp["pid"])
            unique_procs.append(rp)
    raw_procs = unique_procs

    # Fallback to nvidia-smi data, fetched lazily on first NO_PERMISSION
    # and cached for subsequent GPUs in this probe cycle.
    if use_nvsmi:
        if nvsmi_cache[0] is None:
            nvsmi_cache[0] = _run_nvsmi_processes()
        for pi in nvsmi_cache[0].get(gpu_uuid, []):
            # Resolve user from /proc for own/other classification
            uid = _read_proc_uid(pi["pid"])
            username = _uid_to_name(uid) if uid is not None else None
            # nvidia-smi process_name is the full command line; prefer the
            # short /proc/<pid>/comm name when readable.
            name = _read_proc_comm(pi["pid"]) or pi["name"]
            raw_procs.append(
                {
                    "pid": pi["pid"],
                    "gpu_memory_mb": pi["used_memory_mb"],
                    "name": name,
                    "user": username,
                }
            )

    # Attach per-process utilization (driver-permission permitting) and
    # full cmdline for our own processes only.
    processes: list[dict[str, Any]] = []
    for rp in raw_procs:
        pid = rp["pid"]
        cmdline = None
        if own_user and rp.get("user") == own_user:
            cmdline = _read_proc_cmdline(pid)
        sm, mem, enc, dec = util_map.get(pid, (None, None, None, None))
        processes.append(
            {
                "pid": pid,
                "gpu_memory_mb": rp["gpu_memory_mb"],
                "name": rp["name"],
                "user": rp.get("user"),
                "cmdline": cmdline,
                "sm_percent": sm,
                "mem_percent": mem,
                "enc_percent": enc,
                "dec_percent": dec,
            }
        )
    return processes


# ---------------------------------------------------------------------------
# Main probe logic
# ---------------------------------------------------------------------------


def _read_cpu_stat() -> tuple[int, int] | None:
    """(total, idle) jiffies from the first /proc/stat line."""
    try:
        with open("/proc/stat", "r") as f:
            parts = f.readline().split()[1:]
        times = [int(p) for p in parts[:8]]
        return sum(times), times[3] + times[4]  # total, idle
    except (OSError, ValueError):
        return None


def _cpu_pct_from(s1: tuple[int, int] | None) -> float | None:
    """CPU% over the interval [s1, now]. Stateless: the interval is the
    probe's own runtime, no extra sleeping required."""
    s2 = _read_cpu_stat()
    if s1 is None or s2 is None:
        return None
    dt_total = s2[0] - s1[0]
    dt_idle = s2[1] - s1[1]
    if dt_total <= 0:
        return None
    return round((dt_total - dt_idle) / dt_total * 100, 1)


def _meminfo() -> tuple[int, int, int, int] | None:
    """(ram_used_mb, ram_total_mb, swap_used_mb, swap_total_mb) from /proc/meminfo."""
    try:
        total = avail = stotal = sfree = None
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1])  # kB
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1])
                elif line.startswith("SwapTotal:"):
                    stotal = int(line.split()[1])
                elif line.startswith("SwapFree:"):
                    sfree = int(line.split()[1])
                if total is not None and avail is not None and stotal is not None and sfree is not None:
                    break
        if total is None or avail is None:
            return None
        used_kb = total - avail
        if stotal is not None and sfree is not None:
            return (
                used_kb // 1024,
                total // 1024,
                (stotal - sfree) // 1024,
                stotal // 1024,
            )
        return (used_kb // 1024, total // 1024, 0, 0)
    except (OSError, ValueError):
        return None


def _thermal_zone_temp() -> int | None:
    """CPU temp from /sys/class/thermal (first tier)."""
    try:
        import os
        zones = []
        for name in os.listdir("/sys/class/thermal"):
            if not name.startswith("thermal_zone"):
                continue
            base = f"/sys/class/thermal/{name}"
            try:
                with open(f"{base}/type") as f:
                    zone_type = f.read().strip()
                with open(f"{base}/temp") as f:
                    temp = int(f.read().strip()) // 1000
            except (OSError, ValueError):
                continue
            if 10 <= temp <= 115:
                zones.append((zone_type, temp))
        if not zones:
            return None
        # Prefer package/CPU zones, otherwise the hottest plausible zone.
        for pref in ("x86_pkg", "cpu", "coretemp"):
            best = [t for z, t in zones if pref in z.lower()]
            if best:
                return max(best)
        return max(zones, key=lambda z: z[1])[1]
    except OSError:
        return None


def _hwmon_cpu_temp() -> int | None:
    """CPU temp from /sys/class/hwmon (k10temp on AMD, coretemp on Intel)."""
    try:
        import os
        cpu_temps: list[int] = []
        all_temps: list[int] = []
        for hwdir in os.listdir("/sys/class/hwmon"):
            base = f"/sys/class/hwmon/{hwdir}"
            try:
                with open(f"{base}/name") as f:
                    name = f.read().strip().lower()
                temps: list[int] = []
                for fname in os.listdir(base):
                    if fname.startswith("temp") and fname.endswith("_input"):
                        try:
                            with open(f"{base}/{fname}") as f:
                                t = int(f.read().strip())
                            if t <= 0:
                                continue
                            temps.append(t // 1000)
                        except (OSError, ValueError):
                            continue
                if temps:
                    all_temps.extend(temps)
                    if "k10temp" in name or "coretemp" in name:
                        cpu_temps.extend(temps)
            except OSError:
                continue
        sane_cpu = [t for t in cpu_temps if 10 <= t <= 115]
        if sane_cpu:
            return max(sane_cpu)
        sane_all = [t for t in all_temps if 10 <= t <= 115]
        return max(sane_all) if sane_all else None
    except OSError:
        return None


def _sensors_temp() -> int | None:
    """CPU temp via the lm-sensors CLI (last tier)."""
    import re
    import subprocess
    try:
        out = subprocess.check_output(
            ["sensors"], stderr=subprocess.DEVNULL, timeout=3
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    text = out.decode("utf-8", errors="replace")

    def pick(pattern: str) -> float | None:
        for m in re.finditer(pattern, text):
            try:
                v = float(m.group(1))
                if 10 <= v <= 115:
                    return v
            except ValueError:
                continue
        return None

    for pattern in (
        r"Tctl:\s*\+(?:\s*)(\d+(?:\.\d+)?)",
        r"Package id [0-9]+:\s*\+(\d+(?:\.\d+)?)",
        r"Core [0-9]+:\s*\+(\d+(?:\.\d+)?)",
    ):
        v = pick(pattern)
        if v is not None:
            return int(v)
    vals = [v for v in re.findall(r"\+(\d+(?:\.\d+)?)°C", text)]
    sane = [float(v) for v in vals if 10 <= float(v) <= 115]
    return int(max(sane)) if sane else None


def _cpu_temperature() -> int | None:
    """Best-effort CPU temp (°C): thermal zones → hwmon → sensors."""
    for source in (_thermal_zone_temp, _hwmon_cpu_temp, _sensors_temp):
        t = source()
        if t is not None:
            return t
    return None


def _cpu_freq() -> tuple[int | None, int | None]:
    """(current_mhz, max_mhz) for the CPU.

    Max comes from cpufreq sysfs when the kernel exposes it; otherwise
    (most servers disable cpufreq) it falls back to the highest core
    frequency currently observed — never hardcoded.
    """
    cur: list[int] = []
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("cpu MHz"):
                    try:
                        cur.append(round(float(line.split(":")[1].strip())))
                    except ValueError:
                        pass
    except OSError:
        pass

    max_mhz = None
    for path in (
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq",
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
    ):
        try:
            with open(path, "r") as f:
                max_mhz = int(f.read().strip()) // 1000  # kHz -> MHz
            break
        except (OSError, ValueError):
            continue
    if max_mhz is None and cur:
        max_mhz = max(cur)

    return (round(sum(cur) / len(cur)) if cur else None), max_mhz


def _rapl_start() -> tuple[float, int] | None:
    """Sample CPU package energy (µJ) for power calculation.

    Returns (monotonic_time, energy_uj).
    """
    try:
        import os
        for zone in os.listdir("/sys/class/powercap"):
            base = f"/sys/class/powercap/{zone}"
            energy = os.path.join(base, "energy_uj")
            if os.path.exists(energy):
                try:
                    with open(energy, "r") as f:
                        e = int(f.read().strip())
                    return (time.monotonic(), e)
                except (OSError, ValueError):
                    continue
    except OSError:
        pass
    return None


def _rapl_max_power_watts() -> float | None:
    """Package power limit from powercap, converted to watts."""
    try:
        import os
        for zone in os.listdir("/sys/class/powercap"):
            base = f"/sys/class/powercap/{zone}"
            for name in ("constraint_0_max_power_uw", "constraint_1_max_power_uw"):
                path = os.path.join(base, name)
                if os.path.exists(path):
                    try:
                        with open(path, "r") as f:
                            return int(f.read().strip()) / 1e6
                    except (OSError, ValueError):
                        continue
    except OSError:
        pass
    return None


def _cpu_power(rapl_start: tuple[float, int] | None) -> float | None:
    """CPU package power in watts from a delta energy reading."""
    try:
        import os
        if rapl_start is None:
            return None
        for zone in os.listdir("/sys/class/powercap"):
            energy = f"/sys/class/powercap/{zone}/energy_uj"
            if os.path.exists(energy):
                try:
                    with open(energy, "r") as f:
                        e2 = int(f.read().strip())
                except (OSError, ValueError):
                    continue
                t2 = time.monotonic()
                dt = t2 - rapl_start[0]
                energy_u = e2 - rapl_start[1]
                if dt <= 0:
                    return None
                return max(round(energy_u / dt / 1e6, 1), 0.0)
    except OSError:
        pass
    return None


def _driver_version(lib) -> str | None:
    """NVIDIA driver version via NVML (no privileges needed)."""
    fn = getattr(lib, "nvmlSystemGetDriverVersion", None)
    if fn is None:
        return None
    fn.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(80)
    try:
        rc = fn(buf, ctypes.c_uint(len(buf)))
    except Exception:
        return None
    if rc != NVML_SUCCESS:
        return None
    return buf.value.decode("utf-8", errors="replace")


def _cuda_versions() -> list[str]:
    """All installed CUDA toolkit versions, discovered from nvcc binaries.

    Scans /usr/local/cuda*, standard /usr/local/cuda, and nvcc on PATH —
    no sudo required. Returns a de-duplicated version list.
    """
    import glob
    import re
    import subprocess
    import os

    candidates: list[str] = []
    seen_paths: set[str] = set()

    def add(path: str) -> None:
        if path and path not in seen_paths and os.path.exists(path):
            seen_paths.add(path)
            candidates.append(path)

    add("/usr/local/cuda/bin/nvcc")
    for p in sorted(glob.glob("/usr/local/cuda-*/bin/nvcc")):
        add(p)
    try:
        import shutil
        add(shutil.which("nvcc") or "")
    except OSError:
        pass

    versions: list[str] = []
    for nvcc in candidates:
        try:
            out = subprocess.check_output(
                [nvcc, "--version"],
                stderr=subprocess.STDOUT,
                timeout=3,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            continue
        text = out.decode("utf-8", errors="replace")
        m = re.search(r"release\s+([0-9.]+),", text)
        if m and m.group(1) not in versions:
            versions.append(m.group(1))
    return versions


def _cpu_info() -> tuple[int | None, str | None]:
    """(logical_cores, model_name) from /proc/cpuinfo.

    Careful: Intel cpuinfo also has `model : 167` (a numeric field) —
    must NOT be mistaken for the human-readable model name.
    """
    try:
        with open("/proc/cpuinfo", "r") as f:
            lines = f.readlines()
    except OSError:
        return None, None
    cores = 0
    model = None
    for line in lines:
        if line.startswith("processor") and ":" in line:
            cores += 1
        elif model is None and line.startswith("model name") and ":" in line:
            model = line.split(":", 1)[1].strip()
    if model is None:
        # ARM and some architectures: Hardware / model keys.
        for line in lines:
            if not (line.startswith("Hardware") or line.startswith("model")) or ":" not in line:
                continue
            val = line.split(":", 1)[1].strip()
            if val and not val.isdigit():
                model = val
                break
    return (cores or None), model


def probe(
    own_user: str | None = None,
    reserved_offsets: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Collect GPU + host information via NVML and return as a dict.

    Args:
        own_user: Highlight processes for this user.
        reserved_offsets: Cached {gpu_index: reserved_mb} from prior
            calibration. If None, NVML v2 is tried first, then
            nvidia-smi is used for one-time calibration.
    """
    t_start = time.monotonic()

    # CPU% and CPU package power are sampled over the probe's own
    # runtime (start here, read again just before returning).
    stat_start = _read_cpu_stat()
    rapl_start = _rapl_start()
    meminfo = _meminfo()
    temp_c = _cpu_temperature()
    cpu_cores, cpu_model = _cpu_info()
    cpu_freq_mhz, cpu_freq_max_mhz = _cpu_freq()

    # Find and load libnvidia-ml
    lib_path = ctypes.util.find_library("nvidia-ml")
    if lib_path is None:
        # Try common locations directly
        for candidate in (
            "libnvidia-ml.so.1",
            "libnvidia-ml.so",
            "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
            "/usr/lib64/libnvidia-ml.so.1",
            "/usr/lib/libnvidia-ml.so.1",
        ):
            try:
                lib = ctypes.CDLL(candidate)
                lib_path = candidate
                break
            except OSError:
                continue
    else:
        lib = ctypes.CDLL(lib_path)

    if lib_path is None:
        return {
            "ok": False,
            "error": "Cannot find libnvidia-ml.so -- NVIDIA driver not installed?",
            "elapsed_ms": (time.monotonic() - t_start) * 1000,
        }

    resolve_nvml_error_codes(lib)
    _setup_nvml(lib)

    # Initialize NVML
    rc = lib.nvmlInit()
    if rc != NVML_SUCCESS:
        return {
            "ok": False,
            "error": f"nvmlInit failed with code {rc}",
            "elapsed_ms": (time.monotonic() - t_start) * 1000,
        }

    try:
        # Get GPU count
        count = ctypes.c_uint(0)
        rc = lib.nvmlDeviceGetCount(ctypes.byref(count))
        if rc != NVML_SUCCESS:
            return {
                "ok": False,
                "error": f"nvmlDeviceGetCount failed with code {rc}",
                "elapsed_ms": (time.monotonic() - t_start) * 1000,
            }

        gpus: list[dict[str, Any]] = []
        handle = ctypes.c_void_p()

        # Lazy cache for nvidia-smi process fallback.
        nvsmi_cache: list[dict[str, list[dict[str, Any]]] | None] = [None]

        # Memory offset calibration. Try NVML v2 first. If unavailable
        # and no cached offsets, calibrate once via nvidia-smi.
        if reserved_offsets is None:
            # First poll: try v2, fall back to one-time calibration
            if count.value > 0:
                test_h = ctypes.c_void_p()
                rc0 = lib.nvmlDeviceGetHandleByIndex(0, ctypes.byref(test_h))
                if rc0 == NVML_SUCCESS:
                    v2 = _try_nvml_v2_memory(lib, test_h)
                    if v2 is not None:
                        reserved_offsets = {}  # v2 available, no offsets needed
                    else:
                        reserved_offsets = _calibrate_reserved(lib, test_h, count)
                else:
                    reserved_offsets = {}
            else:
                reserved_offsets = {}
        # Cached empty dict means "already tried, fall back to raw NVML".
        # Distinguish from "not yet calibrated" by the fact that it's never
        # None on subsequent polls (collector always sends the cached value).

        for i in range(count.value):
            gpu: dict[str, Any] = {"index": i}

            # Get device handle
            rc = lib.nvmlDeviceGetHandleByIndex(i, ctypes.byref(handle))
            if rc != NVML_SUCCESS:
                gpu["error"] = f"get handle failed: {rc}"
                gpus.append(gpu)
                continue

            # Name
            try:
                name_buf = ctypes.create_string_buffer(NVML_DEVICE_NAME_BUFFER_SIZE)
                lib.nvmlDeviceGetName(handle, name_buf, NVML_DEVICE_NAME_BUFFER_SIZE)
                gpu["name"] = name_buf.value.decode("utf-8", errors="replace")
            except Exception:
                gpu["name"] = "unknown"

            # UUID
            try:
                uuid_buf = ctypes.create_string_buffer(NVML_DEVICE_UUID_BUFFER_SIZE)
                lib.nvmlDeviceGetUUID(handle, uuid_buf, NVML_DEVICE_UUID_BUFFER_SIZE)
                gpu["uuid"] = uuid_buf.value.decode("utf-8", errors="replace")
            except Exception:
                gpu["uuid"] = "unknown"

            # Memory — try v2 first (fast, no subprocess), then cached offsets
            try:
                mem = NvmlMemory()
                lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
                total_mb = int(mem.total // (1024 * 1024))
                free_mb = int(mem.free // (1024 * 1024))

                # Try NVML v2 for user-visible used (= total - free - reserved)
                v2 = _try_nvml_v2_memory(lib, handle)
                if v2 is not None:
                    used_mb, free_mb_adj = v2
                    free_mb = free_mb_adj
                elif i in reserved_offsets:
                    used_mb = total_mb - free_mb - reserved_offsets[i]
                else:
                    # No calibration available — raw NVML value
                    used_mb = total_mb - free_mb

                gpu["memory_total_mb"] = total_mb
                gpu["memory_used_mb"] = max(used_mb, 0)
                gpu["memory_free_mb"] = total_mb - max(used_mb, 0)
            except Exception:
                gpu["memory_total_mb"] = 0
                gpu["memory_used_mb"] = 0
                gpu["memory_free_mb"] = 0

            # Utilization
            try:
                util = NvmlUtilization()
                lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util))
                gpu["utilization_gpu"] = util.gpu
                gpu["utilization_mem"] = util.memory
            except Exception:
                gpu["utilization_gpu"] = 0
                gpu["utilization_mem"] = 0

            # Temperature
            try:
                temp = ctypes.c_uint(0)
                lib.nvmlDeviceGetTemperature(
                    handle, NVML_TEMPERATURE_GPU, ctypes.byref(temp)
                )
                gpu["temperature_c"] = temp.value
            except Exception:
                gpu["temperature_c"] = 0

            # Power usage (NVML returns milliwatts)
            try:
                power = ctypes.c_uint(0)
                lib.nvmlDeviceGetPowerUsage(handle, ctypes.byref(power))
                gpu["power_watts"] = round(power.value / 1000.0, 1)
            except Exception:
                gpu["power_watts"] = 0.0

            # Power limit
            try:
                power_limit = ctypes.c_uint(0)
                lib.nvmlDeviceGetPowerManagementLimit(
                    handle, ctypes.byref(power_limit)
                )
                gpu["power_limit_watts"] = round(power_limit.value / 1000.0, 1)
            except Exception:
                gpu["power_limit_watts"] = 0.0

            # GPU graphics clock (current / max from NVML)
            gfx_cur_mhz, gfx_max_mhz = _get_gfx_clock(lib, handle)
            gpu["graphics_clock_mhz"] = gfx_cur_mhz if gfx_cur_mhz is not None else 0
            gpu["graphics_clock_max_mhz"] = gfx_max_mhz if gfx_max_mhz is not None else 0

            # ENC / DEC utilization + encoder session stats
            enc_util = _get_enc_util(lib, handle)
            dec_util = _get_dec_util(lib, handle)
            gpu["encoder_util"] = enc_util if enc_util is not None else 0
            gpu["decoder_util"] = dec_util if dec_util is not None else 0
            enc_stats = _get_enc_stats(lib, handle)
            if enc_stats:
                gpu["encoder_sessions"] = enc_stats[0]
                gpu["encoder_avg_fps"] = enc_stats[1]
            else:
                gpu["encoder_sessions"] = 0
                gpu["encoder_avg_fps"] = 0
            dec_stats = _get_dec_stats(lib, handle)
            gpu["decoder_sessions"] = dec_stats[0] if dec_stats else 0

            # Per-process utilization samples (fresh from driver ring buffer)
            util_map = _gpu_process_utilizations(lib, handle)

            # All running processes — own and others, aggregated by the UI
            processes = _gpu_processes(
                lib, handle,
                gpu_uuid=gpu.get("uuid", ""),
                own_user=own_user,
                nvsmi_cache=nvsmi_cache,
                util_map=util_map,
            )
            gpu["processes"] = processes
            gpus.append(gpu)

        elapsed = (time.monotonic() - t_start) * 1000

        return {
            "ok": True,
            "gpus": gpus,
            "host": {
                "cpu_percent": _cpu_pct_from(stat_start),
                "memory_used_mb": meminfo[0] if meminfo else None,
                "memory_total_mb": meminfo[1] if meminfo else None,
                "swap_used_mb": meminfo[2] if meminfo else None,
                "swap_total_mb": meminfo[3] if meminfo else None,
                "temp_c": temp_c,
                "cpu_cores": cpu_cores,
                "cpu_model": cpu_model,
                "cpu_freq_mhz": cpu_freq_mhz,
                "cpu_freq_max_mhz": cpu_freq_max_mhz,
                "cpu_power_watts": _cpu_power(rapl_start),
                "cpu_power_max_watts": _rapl_max_power_watts(),
                "driver_version": _driver_version(lib),
                "cuda_versions": _cuda_versions(),
            },
            "elapsed_ms": round(elapsed, 1),
            "reserved_offsets": reserved_offsets if reserved_offsets else {},
        }

    finally:
        lib.nvmlShutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--own-user", default=None, help="Highlight processes for this user")
    parser.add_argument("--reserved-offsets", default=None,
                        help="JSON: {gpu_index: reserved_mb} from prior calibration")
    args = parser.parse_args()

    offsets = None
    if args.reserved_offsets:
        try:
            # Keys arrive as strings from JSON; convert to int
            raw = json.loads(args.reserved_offsets)
            offsets = {int(k): v for k, v in raw.items()}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    result = probe(own_user=args.own_user, reserved_offsets=offsets)
    json.dump(result, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
