import os
import subprocess
import sys
import threading
import time
from pathlib import Path

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


def test_step_runs_in_its_own_session_under_the_launcher(tmp_path):
    out = run(PY + ["import os; print(os.getpgrp() == os.getppid(), os.getsid(0) == os.getppid(), "
                    "os.getppid() != os.getpid())"], tmp_path)
    assert out.log.read_text().split() == ["True", "True", "True"]


def test_exit_status_and_signals_pass_through_the_launcher(tmp_path):
    assert run(PY + ["raise SystemExit(3)"], tmp_path).returncode == 3
    out = run(PY + ["import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"], tmp_path)
    assert out.returncode == -11 and "killed by SIGSEGV" in out.describe()
    assert run(["/nonexistent/R"], tmp_path).returncode == 127


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


def test_a_dead_control_plane_takes_its_steps_with_it(tmp_path):
    """The control plane is killed mid-step (SIGKILL: no cleanup runs). The step and anything it spawned die too,
    instead of running on behind the job that restart recovery marks interrupted."""
    plane = (f"import sys; sys.path.insert(0, {str(Path(__file__).parents[1])!r}); from pathlib import Path; "
             "from controlplane.runner import Process; "
             f"p = Process({PY + [SPAWNS_CHILD]!r}, Path({str(tmp_path)!r})).start(); "
             "print(p._popen.pid, flush=True); p.wait()")
    cp = subprocess.Popen([sys.executable, "-c", plane], stdout=subprocess.PIPE, text=True)
    leader = int(cp.stdout.readline())
    for _ in range(100):
        if (tmp_path / "step.log").read_text().strip():
            break
        time.sleep(0.05)
    grandchild = int((tmp_path / "step.log").read_text().split()[0])
    assert _alive(leader) and _alive(grandchild)
    cp.kill()
    cp.wait()
    deadline = time.monotonic() + 5
    while (_alive(leader) or _alive(grandchild)) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(leader) and not _alive(grandchild)


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


@pytest.mark.skipif(os.geteuid() != 0, reason="running steps as another user needs root (the Docker image)")
def test_a_step_user_gets_only_its_working_directory(tmp_path):
    secret = tmp_path / "controlplane.db"
    secret.write_text("key")
    secret.chmod(0o600)
    tmp_path.chmod(0o700)                                             # the control plane runs with umask 077
    work = tmp_path / "projects" / "p" / "runs" / "j"
    work.mkdir(parents=True)
    out = run(PY + ["import os, pathlib; pathlib.Path('made').write_text('x'); print(os.getuid()); "
                    f"print(os.access({str(secret)!r}, os.R_OK), os.access({str(tmp_path)!r}, os.R_OK))"],
              work, limits=Limits(user="nobody"))
    uid, can_read_secret, can_list = out.log.read_text().split()
    assert out.ok and int(uid) != 0 and (can_read_secret, can_list) == ("False", "False")
    assert (work / "made").stat().st_uid == os.getuid()                # handed back to the control plane
