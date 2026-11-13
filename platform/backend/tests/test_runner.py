import os
import sys
import threading
import time

import pytest

from controlplane.runner import Limits, Process, run

PY = [sys.executable, "-I", "-c"]


def test_environment_is_scrubbed(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    out = run(PY + ["import os, json; print(json.dumps(dict(os.environ)))"], tmp_path)
    assert out.ok, out.describe()
    env = eval(out.log.read_text())
    assert "OPENAI_API_KEY" not in env and env["HOME"] == str(tmp_path) and env["TZ"] == "UTC"
    assert set(env) <= {"PATH", "LANG", "LC_ALL", "TZ", "HOME", "TMPDIR", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "__CF_USER_TEXT_ENCODING"}


def test_step_runs_in_its_own_process_group(tmp_path):
    out = run(PY + ["import os; print(os.getpgrp() == os.getpid(), os.getsid(0) == os.getpid())"], tmp_path)
    assert out.log.read_text().split() == ["True", "True"]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


SPAWNS_CHILD = ("import subprocess, sys, time; c = subprocess.Popen(['sleep', '60']); "
                "print(c.pid, flush=True); time.sleep(60)")


def test_wall_clock_limit_kills_the_whole_group(tmp_path):
    out = run(PY + [SPAWNS_CHILD], tmp_path, limits=Limits(wall_s=1))
    assert out.timed_out and not out.ok and "wall-clock" in out.describe()
    child = int(out.log.read_text().split()[0])
    time.sleep(0.2)
    assert not _alive(child)


def test_cpu_limit(tmp_path):
    out = run(PY + ["while True: pass"], tmp_path, limits=Limits(cpu_s=1, wall_s=30))
    assert not out.ok and not out.timed_out and out.returncode < 0


def test_cancel_from_another_thread(tmp_path):
    p = Process(PY + [SPAWNS_CHILD], tmp_path).start()
    threading.Timer(0.5, p.cancel).start()
    out = p.wait()
    assert out.cancelled and not out.ok and out.duration_s < 10
    assert not _alive(int(out.log.read_text().split()[0]))


def test_cancel_before_start_never_spawns(tmp_path):
    p = Process(PY + ["open('ran', 'w')"], tmp_path)
    p.cancel()
    assert p.start().wait().cancelled and not (tmp_path / "ran").exists()


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS does not enforce address-space limits")
def test_memory_limit(tmp_path):
    out = run(PY + ["x = bytearray(2 << 30)"], tmp_path, limits=Limits(memory_mb=512))
    assert not out.ok and "MemoryError" in out.log.read_text()
