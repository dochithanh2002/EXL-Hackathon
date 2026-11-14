import socket
import subprocess
import time
import urllib.request

import pytest
from cryptography.fernet import Fernet

from controlplane import support
from controlplane.agentservice import AgentService
from controlplane.endpoints import Endpoints
from controlplane.environments import ENVS_DIR, EnvironmentNotReady, Environments
from controlplane.jobs import Jobs
from controlplane.settings import Settings
from controlplane.tracking import Tracker
from controlplane.workflow import Store

MLFLOW = ENVS_DIR / "mlflow" / ".venv" / "bin" / "mlflow"


@pytest.fixture(scope="session")
def envs():
    e = Environments()
    try:
        e.r()
    except EnvironmentNotReady as err:
        pytest.skip(str(err))
    return e


@pytest.fixture(scope="session")
def mlflow_uri(tmp_path_factory):
    """A throwaway MLflow server, the same build `dev.sh` runs."""
    if not MLFLOW.exists():
        pytest.skip("envs/mlflow is not built; run platform/bootstrap.sh")
    d = tmp_path_factory.mktemp("mlflow")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [MLFLOW, "server", "--host", "127.0.0.1", "--port", str(port), "--workers", "1",
         "--backend-store-uri", f"sqlite:///{d / 'mlflow.db'}", "--artifacts-destination", str(d / "artifacts")],
        stdout=open(d / "server.log", "wb"), stderr=subprocess.STDOUT, env={"PATH": "/usr/bin:/bin",
                                                                              "MLFLOW_DISABLE_AGENT_HINT": "1"})
    uri = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(f"{uri}/health", timeout=1)
            break
        except OSError:
            if proc.poll() is not None:
                pytest.fail((d / "server.log").read_text())
            time.sleep(0.25)
    else:
        proc.kill()
        pytest.fail("MLflow server did not start")
    yield uri
    proc.terminate()
    proc.wait(10)


@pytest.fixture
def tracker(mlflow_uri):
    return Tracker(mlflow_uri)


def make_jobs(tmp_path, envs, tracker, agents_url="http://127.0.0.1:9", url="http://127.0.0.1:9") -> Jobs:
    settings = Settings(data_dir=tmp_path / "var", agents_url=agents_url, url=url)
    store = Store(tmp_path / "db.sqlite")
    return Jobs(store, settings, envs, tracker, Endpoints(tmp_path / "db.sqlite", Fernet(Fernet.generate_key())),
                AgentService(agents_url), support.load())
