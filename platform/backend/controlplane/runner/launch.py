"""Apply resource limits, run the step, and supervise it: launch.py <cpu_s> <memory_mb> -- <argv...>

Run with `python -I` as the first process of the step's own session (process group). The step runs as a child; this
process stays as the group leader and kills the whole group if the control plane that started it dies, so a killed
or crashed control plane leaves no step running behind the job it marks interrupted on restart. It exits with the
step's status, re-raising the signal that killed the step so the caller sees the same return code. Doing this in a
separate process also keeps `preexec_fn`, which is unsafe in a threaded server, out of the control plane."""
import os
import resource
import signal
import sys
import time

POLL_S = 0.2

cpu, memory_mb, sep, *argv = sys.argv[1:]
assert sep == "--" and argv
parent = os.getppid()
signal.signal(signal.SIGTERM, signal.SIG_IGN)      # a cancel signals the whole group; report the step's own status
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    resource.setrlimit(resource.RLIMIT_CPU, (int(cpu), int(cpu) + 5))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (int(memory_mb) << 20,) * 2)
    except (ValueError, OSError):       # macOS does not support an address-space limit; the wall clock still applies
        pass
    try:
        os.execv(argv[0], argv)
    except OSError as e:
        print(f"cannot run {argv[0]}: {e}", file=sys.stderr)
        os._exit(127)

while True:
    done, status = os.waitpid(pid, os.WNOHANG)
    if done:
        break
    if os.getppid() != parent:          # the control plane is gone
        os.killpg(0, signal.SIGKILL)
    time.sleep(POLL_S)

if os.WIFSIGNALED(status):
    sig = os.WTERMSIG(status)
    if sig != signal.SIGKILL:
        signal.signal(sig, signal.SIG_DFL)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.kill(os.getpid(), sig)
sys.exit(os.waitstatus_to_exitcode(status))
