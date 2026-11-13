from __future__ import annotations

import hashlib
import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ..schemas import ROLES
from ..settings import Settings
from ..workflow import GateError, NotFound, Store
from ..workflow.machine import FilesUploaded, InputFile

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$")
EXTENSIONS = {"dataset": {".csv"}, "r_code": {".r", ".rmd", ".qmd"}, "document": {".md", ".txt", ".pdf", ".docx"}}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        app.state.store = Store(settings.db_path)
        app.state.store.recover()
        yield

    app = FastAPI(title="controlplane", lifespan=lifespan)

    def store(request: Request) -> Store:
        return request.app.state.store

    @app.exception_handler(GateError)
    async def _gate(_, e: GateError):
        return JSONResponse(status_code=409, content={"detail": str(e)})

    @app.exception_handler(NotFound)
    async def _missing(_, e: NotFound):
        return JSONResponse(status_code=404, content={"detail": f"not found: {e}"})

    @app.get("/api/health")
    def health():
        return {"ok": True}

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
        documents: Annotated[list[UploadFile], File()] = [],
    ):
        uploads = [("dataset", dataset)] + [("r_code", f) for f in r_code] + [("document", f) for f in documents]
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
        return {"id": pid, "state": st.state(pid)}

    @app.get("/api/projects/{pid}")
    def get_project(request: Request, pid: str):
        st = store(request)
        return {"id": pid, "state": st.state(pid), "running_job": st.running_job(pid),
                "decisions": st.decisions(pid)}

    @app.get("/api/projects/{pid}/transitions")
    def transitions(request: Request, pid: str):
        st = store(request)
        st.state(pid)
        return st.transitions(pid)

    return app


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
