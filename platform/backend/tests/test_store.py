import sqlite3

import pytest

from controlplane import support
from controlplane.schemas import canonical_hash
from controlplane.workflow import Reason, Status, Step, Store
from controlplane.workflow.machine import (
    BaselineSucceeded, Bindings, CandidateApproved, CandidateValidated, FilesUploaded, GateError, InputFile,
    SpecConfirmed, SpecDrafted,
)
from tests.test_schemas import spec

H = "a" * 64
FILES = (InputFile(role="dataset", name="d.csv", sha256=H), InputFile(role="r_code", name="m.R", sha256=H))


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "db.sqlite")


def at_baseline(store):
    pid = store.create_project("m4")
    store.apply(pid, FilesUploaded(files=FILES))
    v = store.add_spec(pid, spec(), support.load())
    s = store.state(pid)
    assert (s.drafted_tier, s.drafted_proven) == ("exact", False)      # stats::glm has no parity fixture yet
    store.apply(pid, SpecConfirmed(version=v, tier="exact", proven=False, baseline_key="1" * 64,
                                   parity_risk_accepted=True))
    return pid


def test_transitions_are_recorded_and_gate_errors_change_nothing(store):
    pid = at_baseline(store)
    log = store.transitions(pid)
    assert [t["event"]["type"] for t in log] == ["files_uploaded", "spec_drafted", "spec_confirmed"]
    assert (log[-1]["to_step"], log[-1]["to_status"]) == (Step.BASELINE, "working")
    before = store.state(pid)
    with pytest.raises(GateError):
        store.apply(pid, BaselineSucceeded(spec_version=9, baseline_id="x"))
    assert store.state(pid) == before and len(store.transitions(pid)) == 3
    assert store.spec(pid, 1) == spec()


def test_job_result_applies_and_stale_result_is_discarded(store):
    pid = at_baseline(store)
    jid = store.start_job(pid, "r_baseline")
    with pytest.raises(sqlite3.IntegrityError):
        store.start_job(pid, "r_baseline")                     # one running job per project
    store.finish_job(jid, BaselineSucceeded(spec_version=1, baseline_id="b1"))
    assert store.state(pid).step == Step.TRANSLATION and store.running_job(pid) is None

    jid = store.start_job(pid, "translate")
    store.apply(pid, SpecDrafted(version=2, tier="exact", proven=True))
    store.apply(pid, SpecConfirmed(version=2, tier="exact", proven=True, baseline_key="2" * 64))
    b = Bindings(spec_version=1, baseline_id="b1", candidate_hash="c" * 64, validation_id="v1")
    store.finish_job(jid, CandidateValidated(bindings=b, passed=True))
    assert store.state(pid).step == Step.BASELINE
    with store._tx() as db:
        job = store._job(db, jid)
    assert job["status"] == "failed" and "discarded" in job["detail"]


def test_recover_marks_running_jobs_interrupted(store, tmp_path):
    pid = at_baseline(store)
    jid = store.start_job(pid, "r_baseline")
    restarted = Store(tmp_path / "db.sqlite")
    assert restarted.recover() == [jid]
    s = restarted.state(pid)
    assert (s.step, s.status, s.reason) == (Step.BASELINE, Status.AWAITING_DS, Reason.INTERRUPTED)
    assert restarted.running_job(pid) is None


def test_cancel_job(store):
    pid = at_baseline(store)
    store.cancel_job(store.start_job(pid, "r_baseline"))
    assert store.state(pid).reason == Reason.CANCELLED


def test_decisions_are_void_after_spec_change(store):
    pid = at_baseline(store)
    store.apply(pid, BaselineSucceeded(spec_version=1, baseline_id="b1"))
    b = Bindings(spec_version=1, baseline_id="b1", candidate_hash="c" * 64, validation_id="v1")
    store.apply(pid, CandidateValidated(bindings=b, passed=True))
    store.apply(pid, CandidateApproved(bindings=b))
    assert [d["void"] for d in store.decisions(pid)] == [False]
    store.apply(pid, SpecDrafted(version=2, tier="exact", proven=True))
    store.apply(pid, SpecConfirmed(version=2, tier="exact", proven=True, baseline_key="1" * 64))
    assert [d["void"] for d in store.decisions(pid)] == [True]


def test_evidence_roundtrip(store):
    pid = store.create_project("p")
    eid, h = store.record(pid, "validation", {"pass": True, "checks": []})
    got = store.evidence(pid, eid)
    assert got["hash"] == h and got["body"] == {"pass": True, "checks": []}


def confirm(store, pid, spec):
    v = store.add_spec(pid, spec, support.load())
    s = store.state(pid)
    return store.apply(pid, SpecConfirmed(version=v, tier=s.drafted_tier, proven=s.drafted_proven,
                                          baseline_key=canonical_hash(spec.baseline_key()), parity_risk_accepted=True))


def test_changing_the_seed_restarts_the_baseline(store):
    pid = store.create_project("m4")
    store.apply(pid, FilesUploaded(files=FILES))
    confirm(store, pid, spec())
    store.apply(pid, BaselineSucceeded(spec_version=1, baseline_id="b1"))
    split = spec().split.model_dump()
    s = confirm(store, pid, spec(tolerances={"coef_rel": 1e-4}))
    assert (s.step, s.baseline_id) == (Step.TRANSLATION, "b1")           # tolerances only: baseline reused
    s = confirm(store, pid, spec(tolerances={"coef_rel": 1e-4}, split=split | {"seed": 2}))
    assert (s.step, s.baseline_id) == (Step.BASELINE, None)              # new seed: baseline rerun
