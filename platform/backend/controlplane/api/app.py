from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import support
from ..agentservice import AgentService
from ..analyst import Analyst, ToolError
from ..endpoints import EndpointIn, Endpoints, InvalidSetting, RoleModel, load_cipher
from ..environments import EnvironmentNotReady, Environments
from ..jobs import Jobs
from ..runner.parity import _coefs, score
from ..schemas import ROLES, canonical_hash, sha256_file
from ..settings import Settings
from ..tracking import Mirror, Tracker
from ..workflow import GateError, NotFound, Status, Store
from ..workflow.machine import (Bindings, CandidateApproved, CandidateRejected, FilesUploaded, InputFile,
                                ResumeRepair, RetryStep, SpecConfirmed)

LOG_TAIL = 1 << 20
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$")
EXTENSIONS = {"dataset": {".csv"}, "data": {".csv"}, "r_code": {".r", ".rmd", ".qmd"}, "document": {".md", ".txt", ".pdf", ".docx"}}


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class Confirmation(BaseModel):
    parity_risk_accepted: bool = False


class ModelChoice(BaseModel):
    model: str


class Resume(BaseModel):
    additional_attempts: int = Field(default=0, ge=0, le=20)


class Rejection(BaseModel):
    bindings: Bindings
    comment: str = Field(min_length=1, max_length=20_000)


class TrackingRetry(BaseModel):
    id: int | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        st = app.state.store = Store(settings.db_path)
        app.state.endpoints = Endpoints(settings.db_path, load_cipher(settings.data_dir))
        app.state.agents = AgentService(settings.agents_url)
        mirror = app.state.mirror = Mirror(st, Tracker(settings.mlflow_uri), settings.data_dir,
                                           settings.mlflow_max_file_mb)
        app.state.jobs = Jobs(st, settings, Environments(), app.state.endpoints, app.state.agents, support.load(),
                              mirror)
        app.state.jobs.track_interrupted(st.recover())
        mirror.start()
        if settings.autostart:
            for p in st.projects():
                if p["state"].status == Status.WORKING and st.running_job(p["id"]) is None:
                    app.state.jobs.advance(p["id"])
        yield
        mirror.stop()

    app = FastAPI(title="controlplane", lifespan=lifespan)

    def store(request: Request) -> Store:
        return request.app.state.store

    def jobs(request: Request) -> Jobs:
        return request.app.state.jobs

    def endpoints(request: Request) -> Endpoints:
        return request.app.state.endpoints

    def advance(request: Request, pid: str) -> None:
        if settings.autostart:
            jobs(request).advance(pid)

    def analyst(request: Request, pid: str) -> Analyst:
        s = store(request).state(pid)
        if not s.files:
            raise HTTPException(409, "no files uploaded")
        return Analyst(s.files, settings.project_dir(pid) / "inputs", settings.project_dir(pid) / "cache",
                       support.load(), Environments(), settings.limits)

    @app.exception_handler(GateError)
    async def _gate(_, e: GateError):
        return JSONResponse(status_code=409, content={"detail": str(e)})

    @app.exception_handler(NotFound)
    async def _missing(_, e: NotFound):
        return JSONResponse(status_code=404, content={"detail": f"not found: {e}"})

    @app.exception_handler(InvalidSetting)
    async def _invalid(_, e: InvalidSetting):
        return JSONResponse(status_code=422, content={"detail": str(e)})

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/environment")
    def environment():
        envs = Environments()
        status = {}
        for name, check in (("r", envs.r), ("python", envs.py)):
            try:
                check()
                status[name] = "ready"
            except EnvironmentNotReady as e:
                status[name] = str(e)
        return {"version": envs.version(), "lockfiles": {n: sha256_file(p) for n, p in envs.lockfiles.items()},
                **status}

    @app.get("/api/schemas/{role}")
    def schema(role: str):
        if role not in ROLES:
            raise HTTPException(404, f"unknown role {role}")
        return ROLES[role].model_json_schema()

    @app.get("/api/projects")
    def list_projects(request: Request):
        return store(request).projects()

    @app.post("/api/projects", status_code=201)
    async def create_project(
        request: Request,
        name: Annotated[str, Form(min_length=1, max_length=200)],
        dataset: Annotated[UploadFile, File()],
        r_code: Annotated[list[UploadFile], File()],
        data: Annotated[list[UploadFile], File()] = [],
        documents: Annotated[list[UploadFile], File()] = [],
    ):
        """`dataset` is the CSV the model's rows come from; `data` are other CSVs the R code reads."""
        uploads = ([("dataset", dataset)] + [("data", f) for f in data] + [("r_code", f) for f in r_code]
                   + [("document", f) for f in documents])
        for role, f in uploads:
            _check_name(role, f.filename)
        if len({f.filename for _, f in uploads}) != len(uploads):
            raise HTTPException(422, "file names must be unique")
        st = store(request)
        pid = st.create_project(name, settings.retry_limit)
        inputs = settings.project_dir(pid) / "inputs"
        inputs.mkdir(parents=True)
        try:
            files = [InputFile(role=role, name=f.filename, sha256=await _save(f, inputs / f.filename))
                     for role, f in uploads]
            st.apply(pid, FilesUploaded(files=tuple(files)))
        except BaseException:
            shutil.rmtree(inputs, ignore_errors=True)
            raise
        state = st.state(pid)
        advance(request, pid)
        return {"id": pid, "state": state}

    @app.get("/api/projects/{pid}")
    def get_project(request: Request, pid: str):
        st = store(request)
        return {"id": pid, "name": st.name(pid), "state": st.state(pid), "running_job": st.running_job(pid),
                "decisions": st.decisions(pid)}

    @app.get("/api/projects/{pid}/mlflow")
    def mlflow(request: Request, pid: str):
        """Where the browser finds the project's MLflow experiment; run pages are `{url}/#/experiments/{id}/runs/{run}`."""
        store(request).state(pid)
        try:
            exp = request.app.state.mirror.tracker.experiment(pid)
        except Exception as e:
            raise HTTPException(503, f"MLflow is not reachable: {e}")
        return {"url": (settings.mlflow_public_url or settings.mlflow_uri).rstrip("/"), "experiment_id": exp}

    @app.get("/api/projects/{pid}/tracking")
    def tracking(request: Request, pid: str):
        """The project's runs queued for MLflow and how mirroring them went."""
        return store(request).tracking(pid)

    @app.post("/api/projects/{pid}/tracking/retry")
    def retry_tracking(request: Request, pid: str, r: TrackingRetry = Body(default_factory=TrackingRetry)):
        """Mirror the failed rows again, or only row `id`."""
        try:
            ids = store(request).retry_tracking(pid, r.id)
        except ValueError as e:
            raise HTTPException(409, str(e))
        request.app.state.mirror.wake()
        return {"queued": ids}

    def with_run(request: Request, pid: str, evidence: dict) -> dict:
        """Evidence with the MLflow run mirroring it, or null while it is not mirrored."""
        runs = store(request).mirrored(pid).get(evidence["id"])
        body = evidence["body"] | {"mlflow_run_id": runs[-1] if runs else evidence["body"].get("mlflow_run_id")}
        return evidence | {"body": body}

    @app.get("/api/projects/{pid}/jobs")
    def project_jobs(request: Request, pid: str):
        runs = store(request).mirrored(pid)
        return [j | {"mlflow_run_ids": runs.get(j["id"], [])} for j in store(request).jobs(pid)]

    @app.get("/api/projects/{pid}/jobs/{jid}/log", response_class=PlainTextResponse)
    def job_log(request: Request, pid: str, jid: str):
        """The step log of a job that ran a process (R baseline, candidate), its last megabyte if longer."""
        store(request).job(pid, jid)
        path = settings.project_dir(pid) / "runs" / jid / "step.log"
        if not path.is_file():
            raise HTTPException(404, "this job has no step log")
        with path.open("rb") as f:
            f.seek(max(0, path.stat().st_size - LOG_TAIL))
            return f.read().decode("utf-8", errors="replace")

    @app.get("/api/projects/{pid}/transitions")
    def transitions(request: Request, pid: str):
        st = store(request)
        st.state(pid)
        return st.transitions(pid)

    @app.get("/api/projects/{pid}/files/{name}", response_class=PlainTextResponse)
    def read_file(request: Request, pid: str, name: str):
        try:
            return analyst(request, pid).read_file(name)
        except ToolError as e:
            raise HTTPException(404, str(e))

    @app.get("/api/projects/{pid}/profile")
    def dataset_profile(request: Request, pid: str, name: str | None = None):
        """The profile of the dataset, or of the data file `name`."""
        try:
            return analyst(request, pid).profile(name)
        except ToolError as e:
            raise HTTPException(422, str(e))

    # ------------------------------------------------------------ step 2
    @app.get("/api/projects/{pid}/chat")
    def chat(request: Request, pid: str):
        return store(request).chat(pid)

    @app.post("/api/projects/{pid}/chat", status_code=202)
    def post_chat(request: Request, pid: str, message: Message):
        store(request).ds_message(pid, message.text)
        advance(request, pid)
        return {"state": store(request).state(pid)}

    @app.get("/api/projects/{pid}/specs")
    def specs(request: Request, pid: str):
        store(request).state(pid)
        return store(request).specs(pid)

    @app.get("/api/projects/{pid}/specs/{version}")
    def spec(request: Request, pid: str, version: int):
        body = store(request).spec(pid, version)
        return {"version": version, "spec": body, "support": support.load().support(body),
                "baseline_key": canonical_hash(body.baseline_key())}

    @app.post("/api/projects/{pid}/specs/{version}/confirm")
    def confirm(request: Request, pid: str, version: int, c: Confirmation = Body(default_factory=Confirmation)):
        body = store(request).spec(pid, version)
        sup = support.load().support(body)
        state = store(request).apply(pid, SpecConfirmed(
            version=version, tier=sup.tier, proven=sup.proven, baseline_key=canonical_hash(body.baseline_key()),
            parity_risk_accepted=c.parity_risk_accepted))
        advance(request, pid)
        return {"state": state}

    # ------------------------------------------------------------ step 4
    @app.get("/api/projects/{pid}/evidence")
    def evidence_list(request: Request, pid: str, kind: str | None = None):
        return store(request).evidence_list(pid, kind)

    @app.get("/api/projects/{pid}/evidence/{eid}")
    def evidence(request: Request, pid: str, eid: str):
        return with_run(request, pid, store(request).evidence(pid, eid))

    @app.get("/api/projects/{pid}/evidence/{eid}/coefficients")
    def coefficients(request: Request, pid: str, eid: str):
        """Every term of a baseline (R only) or a validated candidate (R and Python, with relative differences).
        Non-finite values are null."""
        e = store(request).evidence(pid, eid)
        if e["kind"] == "baseline":
            r_dir, py_dir = settings.data_dir / e["body"]["dir"], None
        elif e["kind"] == "validation":
            r_dir = settings.data_dir / store(request).evidence(pid, e["body"]["baseline_id"])["body"]["dir"]
            py_dir = settings.project_dir(pid) / "runs" / e["body"]["job_id"] / "output"
        else:
            raise HTTPException(404, f"{eid} has no coefficients")
        rc = _coefs(r_dir / "coefficients.csv")
        pc = _coefs(py_dir / "coefficients.csv") if py_dir else {}
        tol = store(request).spec(pid, e["body"]["spec_version"]).tolerances if py_dir else None
        rows = []
        for term in list(rc) + [t for t in pc if t not in rc]:
            row = {"term": term}
            if term in rc:
                row |= {"r_estimate": _finite(rc[term][0]), "r_std_error": _finite(rc[term][1])}
            if term in pc:
                row |= {"python_estimate": _finite(pc[term][0]), "python_std_error": _finite(pc[term][1])}
            if term in rc and term in pc:
                # |python - r| / (atol + rtol |r|): at most 1 is within tolerance; null when infinite
                row |= {"estimate_ratio": _finite(score(rc[term][0], pc[term][0], tol.coefficients.atol,
                                                        tol.coefficients.rtol)),
                        "std_error_ratio": _finite(score(rc[term][1], pc[term][1], tol.std_errors.atol,
                                                         tol.std_errors.rtol))}
            rows.append(row)
        return {"python": bool(pc), "terms": rows}

    @app.get("/api/projects/{pid}/evidence/{eid}/files/{name}")
    def captured(request: Request, pid: str, eid: str, name: str):
        """A file the R capture harness exported for a baseline (coefficients, fit statistics, sessionInfo, ...)."""
        e = store(request).evidence(pid, eid)
        if e["kind"] != "baseline" or name not in e["body"]["files"]:
            raise HTTPException(404, f"{eid} has no file {name}")
        return FileResponse(settings.data_dir / e["body"]["dir"] / name, filename=name)

    @app.get("/api/projects/{pid}/evidence/{eid}/plots/{name}")
    def validation_plot(request: Request, pid: str, eid: str, name: str):
        """A plot of a validation, from its work directory."""
        e = store(request).evidence(pid, eid)
        if e["kind"] != "validation" or name not in {p["name"] for p in e["body"].get("plots", [])}:
            raise HTTPException(404, f"{eid} has no plot {name}")
        return FileResponse(jobs(request).run_dir(pid, e["body"]) / "plots" / name, media_type="image/svg+xml")

    @app.post("/api/projects/{pid}/approve")
    def approve(request: Request, pid: str, bindings: Bindings):
        """The DS approves the candidate and validation they reviewed; `bindings` must name exactly those."""
        state = jobs(request).decide(pid, CandidateApproved(bindings=bindings))
        advance(request, pid)
        return {"state": state}

    @app.post("/api/projects/{pid}/reject")
    def reject(request: Request, pid: str, r: Rejection):
        state = jobs(request).decide(pid, CandidateRejected(bindings=r.bindings, comment=r.comment))
        advance(request, pid)
        return {"state": state}

    # ------------------------------------------------------------ step 5
    @app.post("/api/projects/{pid}/resume")
    def resume(request: Request, pid: str, r: Resume = Body(default_factory=Resume)):
        """Resume the repair after the retry limit (raising it by `additional_attempts`), or after a diagnosis that
        asked for a spec change, keeping the current spec. To change the spec instead, write to the Analyst."""
        state = store(request).apply(pid, ResumeRepair(additional_attempts=r.additional_attempts))
        advance(request, pid)
        return {"state": state}

    # ------------------------------------------------------------ steps 6 and 7
    @app.get("/api/projects/{pid}/report")
    def report(request: Request, pid: str):
        rid = store(request).state(pid).report_id
        if rid is None:
            raise HTTPException(404, "the report is not built")
        return with_run(request, pid, store(request).evidence(pid, rid))

    @app.get("/api/projects/{pid}/report/files/{path:path}")
    def report_file(request: Request, pid: str, path: str):
        """A plot the report embeds, from the report's directory."""
        rid = store(request).state(pid).report_id
        body = store(request).evidence(pid, rid)["body"] if rid else {}
        if path not in {f["path"] for f in body.get("figures", [])}:          # only listed figures, never a traversal
            raise HTTPException(404, f"the report has no file {path}")
        return FileResponse(settings.data_dir / body["dir"] / path, media_type="image/svg+xml")

    def bundle(request: Request, pid: str) -> dict:
        bid = store(request).state(pid).bundle_id
        if bid is None:
            raise HTTPException(404, "the deliverables are not built")
        return store(request).evidence(pid, bid)

    @app.get("/api/projects/{pid}/deliverables")
    def deliverables(request: Request, pid: str):
        return with_run(request, pid, bundle(request, pid))["body"]

    @app.get("/api/projects/{pid}/deliverables/bundle.zip")
    def deliverables_archive(request: Request, pid: str):
        b = bundle(request, pid)["body"]
        return FileResponse(settings.data_dir / b["archive"], media_type="application/zip",
                            filename=f"{pid}-deliverables.zip")

    @app.get("/api/projects/{pid}/deliverables/files/{path:path}")
    def deliverable(request: Request, pid: str, path: str):
        b = bundle(request, pid)["body"]
        if path != "manifest.json" and path not in b["manifest"]["files"]:     # only bundle paths, never a traversal
            raise HTTPException(404, f"no deliverable {path}")
        return FileResponse(settings.data_dir / b["dir"] / path, filename=Path(path).name)

    # ------------------------------------------------------------ jobs
    @app.post("/api/projects/{pid}/retry")
    def retry(request: Request, pid: str):
        state = store(request).apply(pid, RetryStep())
        advance(request, pid)
        return {"state": state}

    @app.post("/api/projects/{pid}/cancel", status_code=202)
    def cancel(request: Request, pid: str):
        job = store(request).running_job(pid)
        if job is None:
            raise HTTPException(409, "no job is running")
        jobs(request).cancel(job["id"])
        return {"job_id": job["id"]}

    @app.get("/api/jobs/{jid}/events")
    def job_events(request: Request, jid: str):
        stream = jobs(request).stream(jid)
        if stream is None:
            raise HTTPException(404, "the job is not running")

        def events():
            pos = 0
            while True:
                batch, done = stream.read(pos, 15)
                if done:
                    return
                pos += len(batch)
                yield "".join(f"data: {json.dumps(e)}\n\n" for e in batch) or ": keepalive\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/internal/tools/{name}")
    def tool(request: Request, name: str, args: Annotated[dict, Body()], authorization: str = Header("")):
        try:
            return jobs(request).tool(authorization.removeprefix("Bearer "), name, args)
        except PermissionError as e:
            raise HTTPException(401, str(e))

    # ------------------------------------------------------------ settings
    @app.get("/api/settings/endpoints")
    def list_endpoints(request: Request):
        return endpoints(request).list()

    @app.post("/api/settings/endpoints", status_code=201)
    def create_endpoint(request: Request, e: EndpointIn):
        return endpoints(request).create(e)

    @app.put("/api/settings/endpoints/{eid}")
    def update_endpoint(request: Request, eid: str, e: EndpointIn):
        return endpoints(request).update(eid, e)

    @app.delete("/api/settings/endpoints/{eid}", status_code=204)
    def delete_endpoint(request: Request, eid: str):
        endpoints(request).delete(eid)

    @app.post("/api/settings/endpoints/{eid}/test")
    def test_endpoint(request: Request, eid: str, choice: ModelChoice):
        ep = endpoints(request).get(eid)
        if choice.model not in ep.models:
            raise HTTPException(422, f"{ep.name} does not list model {choice.model}")
        return request.app.state.agents.test(endpoints(request).model_ref(eid, choice.model))

    @app.get("/api/settings/roles")
    def roles(request: Request):
        assigned = endpoints(request).roles()
        return {role: assigned.get(role) for role in ROLES}

    @app.put("/api/settings/roles/{role}")
    def assign_role(request: Request, role: str, m: RoleModel):
        endpoints(request).assign(role, m)
        return m

    if (settings.static_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="ui")
    return app


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


def _check_name(role: str, name: str | None) -> None:
    if not name or not SAFE_NAME.match(name) or name in (".", ".."):
        raise HTTPException(422, f"invalid file name {name!r}")
    if Path(name).suffix.lower() not in EXTENSIONS[role]:
        raise HTTPException(422, f"{name}: {role} must be one of {sorted(EXTENSIONS[role])}")


async def _save(upload: UploadFile, dest: Path) -> str:
    """Stream the upload to disk, hash it on the way, and make it read-only."""
    h = hashlib.sha256()
    with dest.open("xb") as out:
        while chunk := await upload.read(1 << 20):
            h.update(chunk)
            out.write(chunk)
    dest.chmod(0o444)
    return h.hexdigest()
