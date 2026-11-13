import hashlib
import stat

import pytest
from fastapi.testclient import TestClient

from controlplane.api.app import create_app
from controlplane.settings import Settings
from controlplane.workflow import Store

CSV, R = b"y,x\n1,2\n0,3\n", b"fit <- glm(y ~ x, binomial, df)\n"


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, retry_limit=5))) as c:
        yield c


def upload(client, dataset=("d.csv", CSV), r_code=(("m.R", R),), name="M4"):
    files = [("dataset", dataset)] + [("r_code", f) for f in r_code]
    return client.post("/api/projects", data={"name": name}, files=files)


def test_create_project_hashes_inputs_and_starts_alignment(client, tmp_path):
    r = upload(client)
    assert r.status_code == 201, r.text
    pid, state = r.json()["id"], r.json()["state"]
    assert (state["step"], state["status"], state["retry_limit"]) == (2, "working", 5)
    assert {f["name"]: f["sha256"] for f in state["files"]} == {
        "d.csv": hashlib.sha256(CSV).hexdigest(), "m.R": hashlib.sha256(R).hexdigest()}
    saved = tmp_path / "projects" / pid / "inputs" / "d.csv"
    assert saved.read_bytes() == CSV and not saved.stat().st_mode & stat.S_IWUSR
    assert [p["id"] for p in client.get("/api/projects").json()] == [pid]
    assert client.get(f"/api/projects/{pid}/transitions").json()[0]["event"]["type"] == "files_uploaded"


@pytest.mark.parametrize("dataset,r_code", [
    (("../evil.csv", CSV), (("m.R", R),)),
    (("d.csv", CSV), ((".hidden.R", R),)),
    (("d.txt", CSV), (("m.R", R),)),
    (("d.csv", CSV), (("m.R", R), ("m.R", R))),
])
def test_bad_uploads_are_refused_before_anything_is_stored(client, tmp_path, dataset, r_code):
    assert upload(client, dataset, r_code).status_code == 422
    assert client.get("/api/projects").json() == []


def test_missing_project_and_schema_endpoints(client):
    assert client.get("/api/projects/nope").status_code == 404
    assert client.get("/api/schemas/analyst").json()["title"] == "ModelSpec"
    assert client.get("/api/schemas/other").status_code == 404


def test_startup_recovers_interrupted_jobs(tmp_path):
    settings = Settings(data_dir=tmp_path)
    with TestClient(create_app(settings)) as c:
        pid = upload(c).json()["id"]
    jid = Store(settings.db_path).start_job(pid, "analyst")
    with TestClient(create_app(settings)) as c:
        body = c.get(f"/api/projects/{pid}").json()
    assert body["state"]["reason"] == "interrupted" and body["running_job"] is None
    assert jid
