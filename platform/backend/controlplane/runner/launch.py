"""Apply resource limits, then exec the step: launch.py <cpu_s> <memory_mb> -- <argv...>

Run with `python -I` as the first process of the step's own session. Doing this in a separate process keeps
`preexec_fn`, which is unsafe in a threaded server, out of the control plane."""
import os
import resource
import sys

cpu, memory_mb, sep, *argv = sys.argv[1:]
assert sep == "--" and argv
resource.setrlimit(resource.RLIMIT_CPU, (int(cpu), int(cpu) + 5))
try:
    resource.setrlimit(resource.RLIMIT_AS, (int(memory_mb) << 20,) * 2)
except (ValueError, OSError):           # macOS does not support an address-space limit; the wall clock still applies
    pass
os.execv(argv[0], argv)
