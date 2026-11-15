"""Run one step as a subprocess: its own working directory and session (process group), a scrubbed environment,
and CPU-time, memory and wall-clock limits. In local mode this isolates credentials and working files; it is not a
security boundary (PLAN.md section 6). In the Docker image steps also run as a separate unprivileged user
(`Limits.user`), so step code cannot read the control plane's database, key or other projects' uploads."""
from __future__ import annotations

import os
import pwd
import signal
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

LAUNCHER = Path(__file__).with_name("launch.py")
BASE_ENV = {
    "PATH": "/usr/bin:/bin", "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8", "TZ": "UTC",
    # one thread removes thread-count variability; a different R/BLAS build can still change the last bits
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
}
GRACE_S = 5


class StepCancelled(Exception):
    pass


@dataclass(frozen=True)
class Limits:
    cpu_s: int = 1800
    memory_mb: int = 8192
    wall_s: int = 3600
    user: str | None = None             # run as this user; the control plane needs CAP_SETUID, CAP_SETGID, CAP_CHOWN


@dataclass(frozen=True)
class Outcome:
    returncode: int | None
    duration_s: float
    timed_out: bool
    cancelled: bool
    log: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not (self.timed_out or self.cancelled)

    def describe(self, lines: int = 30) -> str:
        if self.cancelled:
            head = "cancelled"
        elif self.timed_out:
            head = f"stopped after the {self.duration_s:.0f}s wall-clock limit"
        elif self.returncode is not None and self.returncode < 0:
            head = f"killed by {signal.Signals(-self.returncode).name}"
        else:
            head = f"exited with status {self.returncode}"
        tail = self.log.read_text(errors="replace").splitlines()[-lines:] if lines > 0 else []
        return "\n".join([head, *tail])


class Process:
    """`start()` then `wait()`; `cancel()` may be called from another thread at any time."""

    def __init__(self, argv: list[str], cwd: Path, env: dict[str, str] | None = None, limits: Limits = Limits(),
                 log: Path | None = None):
        self.argv, self.cwd, self.limits = [str(a) for a in argv], cwd, limits
        self.log = log or cwd / "step.log"
        tmp = cwd / "tmp"
        self.env = BASE_ENV | {"HOME": str(cwd), "TMPDIR": str(tmp)} | (env or {})
        self._popen: subprocess.Popen | None = None
        self._cancelled = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> Process:
        (self.cwd / "tmp").mkdir(exist_ok=True)
        with self._lock:
            if self._cancelled.is_set():
                return self
            with open(self.log, "wb") as log:
                as_user = _hand_over(self.cwd, self.limits.user) if self.limits.user else {}
                self._popen = subprocess.Popen(
                    [sys.executable, "-I", LAUNCHER, str(self.limits.cpu_s), str(self.limits.memory_mb), "--",
                     *self.argv],
                    cwd=self.cwd, env=self.env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, close_fds=True, **as_user)
        self._t0 = time.monotonic()
        return self

    def wait(self) -> Outcome:
        p = self._popen
        if p is None:
            self.log.touch()
            return Outcome(None, 0.0, False, True, self.log)
        timed_out = False
        try:
            p.wait(timeout=self.limits.wall_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill()
            p.wait()
        self._killpg(signal.SIGKILL)                   # anything the step left behind in its group
        if self.limits.user:                           # frozen outputs must not stay writable by later steps
            _chown_tree(self.cwd, os.getuid(), os.getgid())
        return Outcome(p.returncode, time.monotonic() - self._t0, timed_out, self._cancelled.is_set(), self.log)

    def cancel(self) -> None:
        with self._lock:
            self._cancelled.set()
            if self._popen is not None and self._popen.poll() is None:
                threading.Thread(target=self._kill, daemon=True).start()

    def _kill(self) -> None:
        self._killpg(signal.SIGTERM)
        try:
            self._popen.wait(timeout=GRACE_S)
        except subprocess.TimeoutExpired:
            self._killpg(signal.SIGKILL)

    def _killpg(self, sig: int) -> None:
        try:
            os.killpg(self._popen.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def _hand_over(cwd: Path, user: str) -> dict:
    """Give the step user its working directory, and its ancestors traverse-only access (no listing)."""
    pw = pwd.getpwnam(user)
    _chown_tree(cwd, pw.pw_uid, pw.pw_gid)
    for d in cwd.resolve().parents:
        mode = d.stat().st_mode
        if not mode & stat.S_IXOTH:
            d.chmod(stat.S_IMODE(mode) | stat.S_IXOTH)
    return {"user": pw.pw_uid, "group": pw.pw_gid, "extra_groups": []}


def _chown_tree(top: Path, uid: int, gid: int) -> None:
    for root, dirs, files in os.walk(top):
        for name in [root, *(os.path.join(root, n) for n in dirs + files)]:
            os.lchown(name, uid, gid)


def run(argv: list[str], cwd: Path, env: dict[str, str] | None = None, limits: Limits = Limits()) -> Outcome:
    return Process(argv, cwd, env, limits).start().wait()
