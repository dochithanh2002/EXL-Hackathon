"""The fixed seven-step workflow as a pure function: `apply(state, event) -> state`.

Agents and jobs never move a project; they produce results that the control plane turns into events here, and a
gate that does not hold raises `GateError`. Every job result carries the spec version (and, where relevant, the
baseline and candidate) it was produced for, so a result that arrives after the spec changed is refused.
"""
from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.agents import Tier

SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Step(IntEnum):
    INTAKE = 1
    ALIGNMENT = 2
    BASELINE = 3
    TRANSLATION = 4
    REPAIR = 5
    REPORT = 6
    DELIVERABLES = 7


class Status(StrEnum):
    WORKING = "working"          # an agent or static step is expected to act
    AWAITING_DS = "awaiting_ds"  # the data scientist must act; `reason` says why
    DONE = "done"


class Reason(StrEnum):
    SPEC_DRAFTED = "spec_drafted"
    ANALYST_REPLIED = "analyst_replied"
    UNSUPPORTED = "unsupported"
    CANDIDATE_PASSED = "candidate_passed"
    RETRY_LIMIT = "retry_limit"
    SPEC_CHANGE = "spec_change"
    STEP_FAILED = "step_failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


RETRYABLE = {Reason.STEP_FAILED, Reason.INTERRUPTED, Reason.CANCELLED}


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Bindings(Frozen):
    """What a DS approval is bound to. If any of these changes, the approval is void."""
    spec_version: int
    baseline_id: str
    candidate_hash: SHA256
    validation_id: str


class InputFile(Frozen):
    role: Literal["dataset", "r_code", "document"]
    name: str
    sha256: SHA256


class ProjectState(Frozen):
    step: Step = Step.INTAKE
    status: Status = Status.AWAITING_DS
    reason: Reason | None = None
    detail: str = ""
    files: tuple[InputFile, ...] = ()
    retry_limit: int = 3
    drafted_version: int | None = None
    drafted_tier: Tier | None = None
    drafted_proven: bool | None = None
    spec_version: int | None = None
    spec_tier: Tier | None = None
    spec_proven: bool | None = None
    baseline_key: str | None = None
    baseline_id: str | None = None
    candidate_hash: str | None = None
    validation_id: str | None = None
    validation_passed: bool | None = None
    repairs: int = 0
    ds_comment: str | None = None
    diagnosis_id: str | None = None
    approval: Bindings | None = None
    report_id: str | None = None
    bundle_id: str | None = None

    def bindings(self) -> Bindings | None:
        if None in (self.spec_version, self.baseline_id, self.candidate_hash, self.validation_id):
            return None
        return Bindings(spec_version=self.spec_version, baseline_id=self.baseline_id,
                        candidate_hash=self.candidate_hash, validation_id=self.validation_id)

    def approval_valid(self) -> bool:
        return self.approval is not None and self.approval == self.bindings()


# ---------------------------------------------------------------- events
class FilesUploaded(Frozen):
    type: Literal["files_uploaded"] = "files_uploaded"
    files: tuple[InputFile, ...]


class SpecDrafted(Frozen):
    """`tier` and `proven` come from the support matrix, not from the Analyst."""
    type: Literal["spec_drafted"] = "spec_drafted"
    version: int = Field(ge=1)
    tier: Tier
    proven: bool


class DsMessaged(Frozen):
    """The DS wrote to the Analyst; the Analyst takes the turn. After a failed or cancelled baseline this returns the
    project to step 2, where a corrected spec version can be drafted and confirmed."""
    type: Literal["ds_messaged"] = "ds_messaged"


class AnalystReplied(Frozen):
    """The Analyst ended its turn without a new draft."""
    type: Literal["analyst_replied"] = "analyst_replied"


class SpecConfirmed(Frozen):
    type: Literal["spec_confirmed"] = "spec_confirmed"
    version: int = Field(ge=1)
    tier: Tier
    proven: bool
    baseline_key: SHA256
    parity_risk_accepted: bool = False


class BaselineSucceeded(Frozen):
    type: Literal["baseline_succeeded"] = "baseline_succeeded"
    spec_version: int
    baseline_id: str


class CandidateValidated(Frozen):
    type: Literal["candidate_validated"] = "candidate_validated"
    bindings: Bindings
    passed: bool


class CandidateApproved(Frozen):
    type: Literal["candidate_approved"] = "candidate_approved"
    bindings: Bindings


class CandidateRejected(Frozen):
    type: Literal["candidate_rejected"] = "candidate_rejected"
    bindings: Bindings
    comment: str = Field(min_length=1)


class DiagnosisSubmitted(Frozen):
    type: Literal["diagnosis_submitted"] = "diagnosis_submitted"
    spec_version: int
    candidate_hash: SHA256
    diagnosis_id: str
    spec_change_needed: bool


class ResumeRepair(Frozen):
    type: Literal["resume_repair"] = "resume_repair"
    additional_attempts: int = Field(default=0, ge=0)


class ReportBuilt(Frozen):
    type: Literal["report_built"] = "report_built"
    spec_version: int
    report_id: str


class DeliverablesBuilt(Frozen):
    type: Literal["deliverables_built"] = "deliverables_built"
    spec_version: int
    bundle_id: str


class StepFailed(Frozen):
    type: Literal["step_failed"] = "step_failed"
    step: Step
    spec_version: int | None
    detail: str
    interrupted: bool = False


class Cancelled(Frozen):
    type: Literal["cancelled"] = "cancelled"
    step: Step


class RetryStep(Frozen):
    type: Literal["retry_step"] = "retry_step"


Event = Annotated[
    FilesUploaded | DsMessaged | AnalystReplied | SpecDrafted | SpecConfirmed | BaselineSucceeded
    | CandidateValidated | CandidateApproved | CandidateRejected | DiagnosisSubmitted | ResumeRepair | ReportBuilt
    | DeliverablesBuilt | StepFailed | Cancelled | RetryStep,
    Field(discriminator="type"),
]


class GateError(Exception):
    pass


# ---------------------------------------------------------------- transitions
def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise GateError(msg)


def _at(s: ProjectState, step: Step, status: Status) -> None:
    _require(s.step == step and s.status == status,
             f"requires step {step.value} {step.name.lower()} ({status.value}); project is at step "
             f"{s.step.value} {s.step.name.lower()} ({s.status.value})")


def _spec(s: ProjectState, version: int | None) -> None:
    _require(version == s.spec_version, f"result is for spec v{version}; current spec is v{s.spec_version}")


def _move(s: ProjectState, step: Step, status: Status = Status.WORKING, reason: Reason | None = None,
          detail: str = "", **changes) -> ProjectState:
    return s.model_copy(update={"step": step, "status": status, "reason": reason, "detail": detail, **changes})


def _to_repair(s: ProjectState, **changes) -> ProjectState:
    if s.repairs >= s.retry_limit:
        return _move(s, Step.REPAIR, Status.AWAITING_DS, Reason.RETRY_LIMIT,
                     f"{s.repairs} of {s.retry_limit} repair attempts used", **changes)
    return _move(s, Step.REPAIR, **changes)


def apply(s: ProjectState, e: Event) -> ProjectState:
    match e:
        case FilesUploaded(files=files):
            _at(s, Step.INTAKE, Status.AWAITING_DS)
            roles = [f.role for f in files]
            _require(roles.count("dataset") == 1, "exactly one dataset is required")
            _require("r_code" in roles, "at least one R code file is required")
            _require(len({f.name for f in files}) == len(files), "file names must be unique")
            return _move(s, Step.ALIGNMENT, files=files)

        case DsMessaged():
            if (s.step, s.status) != (Step.BASELINE, Status.AWAITING_DS):
                _at(s, Step.ALIGNMENT, Status.AWAITING_DS)
            return _move(s, Step.ALIGNMENT)

        case AnalystReplied():
            _at(s, Step.ALIGNMENT, Status.WORKING)
            return _move(s, Step.ALIGNMENT, Status.AWAITING_DS, Reason.ANALYST_REPLIED)

        case SpecDrafted(version=v, tier=tier, proven=proven):
            _require(s.step != Step.INTAKE, "upload files first")
            _require(v > max(s.drafted_version or 0, s.spec_version or 0), f"spec v{v} is not newer than the last")
            s = s.model_copy(update={"drafted_version": v, "drafted_tier": tier, "drafted_proven": proven})
            if s.step != Step.ALIGNMENT:
                return s
            if tier == "unsupported":
                return _move(s, Step.ALIGNMENT, Status.AWAITING_DS, Reason.UNSUPPORTED,
                             "the model has no statsmodels counterpart")
            return _move(s, Step.ALIGNMENT, Status.AWAITING_DS, Reason.SPEC_DRAFTED)

        case SpecConfirmed(version=v, tier=tier, proven=proven, baseline_key=key, parity_risk_accepted=accepted):
            _require(s.step != Step.INTAKE, "upload files first")
            _require((s.step, s.status) != (Step.ALIGNMENT, Status.WORKING), "the Analyst is still working")
            _require(v == s.drafted_version, f"spec v{v} is not the latest draft (v{s.drafted_version})")
            _require(v > (s.spec_version or 0), f"spec v{v} is already confirmed")
            _require(tier == s.drafted_tier, f"tier '{tier}' does not match the draft's tier '{s.drafted_tier}'")
            _require(proven == s.drafted_proven, "proven flag does not match the draft")
            _require(tier != "unsupported", "an unsupported model cannot be confirmed")
            _require(tier != "approximate" or accepted, "an approximate-tier model needs explicit DS acceptance")
            _require(proven or accepted, "an exact-tier pair without a passing fixture needs explicit DS acceptance")
            reset = dict(spec_version=v, spec_tier=tier, spec_proven=proven, candidate_hash=None,
                         validation_id=None, validation_passed=None, repairs=0, ds_comment=None, diagnosis_id=None,
                         approval=None, report_id=None, bundle_id=None)
            if s.baseline_id is not None and key == s.baseline_key:
                return _move(s, Step.TRANSLATION, **reset)
            return _move(s, Step.BASELINE, baseline_key=key, baseline_id=None, **reset)

        case BaselineSucceeded(spec_version=v, baseline_id=bid):
            _at(s, Step.BASELINE, Status.WORKING)
            _spec(s, v)
            return _move(s, Step.TRANSLATION, baseline_id=bid)

        case CandidateValidated(bindings=b, passed=passed):
            _at(s, Step.TRANSLATION, Status.WORKING)
            _spec(s, b.spec_version)
            _require(b.baseline_id == s.baseline_id, f"result is for baseline {b.baseline_id}, not {s.baseline_id}")
            changes = dict(candidate_hash=b.candidate_hash, validation_id=b.validation_id, validation_passed=passed)
            if passed:
                return _move(s, Step.TRANSLATION, Status.AWAITING_DS, Reason.CANDIDATE_PASSED, **changes)
            return _to_repair(s, **changes)

        case CandidateApproved(bindings=b):
            _at(s, Step.TRANSLATION, Status.AWAITING_DS)
            _require(s.validation_passed is True, "only a passing validation can be approved")
            _require(b == s.bindings(), "approval does not match the current spec, baseline, candidate and validation")
            return _move(s, Step.REPORT, approval=b)

        case CandidateRejected(bindings=b, comment=comment):
            _at(s, Step.TRANSLATION, Status.AWAITING_DS)
            _require(b == s.bindings(), "rejection does not match the current spec, baseline, candidate and validation")
            return _to_repair(s, ds_comment=comment)

        case DiagnosisSubmitted(spec_version=v, candidate_hash=h, diagnosis_id=did, spec_change_needed=change):
            _at(s, Step.REPAIR, Status.WORKING)
            _spec(s, v)
            _require(h == s.candidate_hash, "diagnosis is for a different candidate")
            if change:
                return _move(s, Step.REPAIR, Status.AWAITING_DS, Reason.SPEC_CHANGE,
                             "the diagnosis says the fix needs a new spec version", diagnosis_id=did)
            return _move(s, Step.TRANSLATION, diagnosis_id=did, repairs=s.repairs + 1, ds_comment=None,
                         candidate_hash=None, validation_id=None, validation_passed=None)

        case ResumeRepair(additional_attempts=n):
            _at(s, Step.REPAIR, Status.AWAITING_DS)
            s = s.model_copy(update={"retry_limit": s.retry_limit + n})
            if s.reason == Reason.RETRY_LIMIT:
                _require(s.repairs < s.retry_limit, "raise the retry limit to resume")
                return _move(s, Step.REPAIR)
            _require(s.reason == Reason.SPEC_CHANGE, f"nothing to resume ({s.reason})")
            return _move(s, Step.TRANSLATION, repairs=s.repairs + 1, ds_comment=None,
                         candidate_hash=None, validation_id=None, validation_passed=None)

        case ReportBuilt(spec_version=v, report_id=rid):
            _at(s, Step.REPORT, Status.WORKING)
            _spec(s, v)
            _require(s.approval_valid(), "the approval is void")
            return _move(s, Step.DELIVERABLES, report_id=rid)

        case DeliverablesBuilt(spec_version=v, bundle_id=bid):
            _at(s, Step.DELIVERABLES, Status.WORKING)
            _spec(s, v)
            _require(s.approval_valid(), "the approval is void")
            return _move(s, Step.DELIVERABLES, Status.DONE, bundle_id=bid)

        case StepFailed(step=step, spec_version=v, detail=detail, interrupted=interrupted):
            _at(s, step, Status.WORKING)
            _spec(s, v)
            return _move(s, step, Status.AWAITING_DS, Reason.INTERRUPTED if interrupted else Reason.STEP_FAILED,
                         detail)

        case Cancelled(step=step):
            _at(s, step, Status.WORKING)
            return _move(s, step, Status.AWAITING_DS, Reason.CANCELLED)

        case RetryStep():
            _require(s.status == Status.AWAITING_DS and s.reason in RETRYABLE, "nothing to retry")
            return _move(s, s.step)

    raise GateError(f"unknown event {e!r}")
