import pytest

from controlplane.workflow.machine import (
    AnalystReplied, BaselineSucceeded, Bindings, CandidateApproved, CandidateRejected, CandidateValidated, Cancelled,
    DeliverablesBuilt, DiagnosisSubmitted, DsMessaged, FilesUploaded, GateError, InputFile, ProjectState, Reason,
    ReportBuilt, ResumeRepair, RetryStep, SpecConfirmed, SpecDrafted, Status, Step, StepFailed, apply,
)

H = "a" * 64
KEY1, KEY2 = "1" * 64, "2" * 64


def cand(n: int) -> str:
    return f"{n:064x}"


FILES = (InputFile(role="dataset", name="d.csv", sha256=H), InputFile(role="r_code", name="m.R", sha256=H))


def run(s, *events):
    for e in events:
        s = apply(s, e)
    return s


def confirmed(version=1, key=KEY1, tier="exact", proven=True, **kw):
    return (SpecDrafted(version=version, tier=tier, proven=proven),
            SpecConfirmed(version=version, tier=tier, proven=proven, baseline_key=key, **kw))


def at_translation():
    return run(ProjectState(), FilesUploaded(files=FILES), *confirmed(),
               BaselineSucceeded(spec_version=1, baseline_id="b1"))


def validated(s, n=1, passed=True):
    b = Bindings(spec_version=s.spec_version, baseline_id=s.baseline_id, candidate_hash=cand(n), validation_id=f"v{n}")
    return apply(s, CandidateValidated(bindings=b, passed=passed)), b


def where(s):
    return s.step, s.status, s.reason


def test_happy_path_to_deliverables():
    s, b = validated(at_translation())
    assert where(s) == (Step.TRANSLATION, Status.AWAITING_DS, Reason.CANDIDATE_PASSED)
    s = run(s, CandidateApproved(bindings=b), ReportBuilt(spec_version=1, report_id="r1"),
            DeliverablesBuilt(spec_version=1, bundle_id="z1"))
    assert where(s) == (Step.DELIVERABLES, Status.DONE, None)
    assert s.approval_valid()


def test_intake_requires_one_dataset_and_r_code():
    with pytest.raises(GateError, match="dataset"):
        apply(ProjectState(), FilesUploaded(files=FILES[1:]))
    with pytest.raises(GateError, match="R code"):
        apply(ProjectState(), FilesUploaded(files=FILES[:1]))
    with pytest.raises(GateError, match="upload"):
        apply(ProjectState(), SpecDrafted(version=1, tier="exact", proven=True))


def test_unsupported_model_stops_at_alignment():
    s = run(ProjectState(), FilesUploaded(files=FILES), SpecDrafted(version=1, tier="unsupported", proven=False))
    assert where(s) == (Step.ALIGNMENT, Status.AWAITING_DS, Reason.UNSUPPORTED)
    with pytest.raises(GateError, match="unsupported"):
        apply(s, SpecConfirmed(version=1, tier="unsupported", proven=False, baseline_key=KEY1))


def test_approximate_tier_needs_ds_acceptance():
    s = run(ProjectState(), FilesUploaded(files=FILES), SpecDrafted(version=1, tier="approximate", proven=False))
    with pytest.raises(GateError, match="approximate"):
        apply(s, SpecConfirmed(version=1, tier="approximate", proven=False, baseline_key=KEY1))
    s = apply(s, SpecConfirmed(version=1, tier="approximate", proven=False, baseline_key=KEY1,
                               parity_risk_accepted=True))
    assert s.step == Step.BASELINE


@pytest.mark.parametrize("drafted", ["unsupported", "approximate"])
def test_confirmation_cannot_upgrade_the_drafted_tier(drafted):
    s = run(ProjectState(), FilesUploaded(files=FILES), SpecDrafted(version=1, tier=drafted, proven=False))
    with pytest.raises(GateError, match="does not match the draft's tier"):
        apply(s, SpecConfirmed(version=1, tier="exact", proven=False, baseline_key=KEY1, parity_risk_accepted=True))


def test_reconfirming_a_version_cannot_reset_the_retry_limit():
    s = at_translation()
    for n in range(1, 4):
        s, _ = validated(s, n, passed=False)
        s = apply(s, DiagnosisSubmitted(spec_version=1, candidate_hash=cand(n), diagnosis_id=f"d{n}",
                                        spec_change_needed=False))
    s, _ = validated(s, 4, passed=False)
    assert s.reason == Reason.RETRY_LIMIT
    with pytest.raises(GateError, match="already confirmed"):
        apply(s, SpecConfirmed(version=1, tier="exact", proven=True, baseline_key=KEY1))


def test_unproven_exact_pair_needs_ds_acceptance():
    s = run(ProjectState(), FilesUploaded(files=FILES), SpecDrafted(version=1, tier="exact", proven=False))
    with pytest.raises(GateError, match="does not match the draft"):
        apply(s, SpecConfirmed(version=1, tier="exact", proven=True, baseline_key=KEY1))
    with pytest.raises(GateError, match="without a passing fixture"):
        apply(s, SpecConfirmed(version=1, tier="exact", proven=False, baseline_key=KEY1))
    s = apply(s, SpecConfirmed(version=1, tier="exact", proven=False, baseline_key=KEY1, parity_risk_accepted=True))
    assert (s.step, s.spec_tier, s.spec_proven) == (Step.BASELINE, "exact", False)


def test_only_the_latest_draft_can_be_confirmed():
    s = run(ProjectState(), FilesUploaded(files=FILES), SpecDrafted(version=1, tier="exact", proven=True),
            SpecDrafted(version=2, tier="exact", proven=True))
    with pytest.raises(GateError, match="latest draft"):
        apply(s, SpecConfirmed(version=1, tier="exact", proven=True, baseline_key=KEY1))
    with pytest.raises(GateError, match="newer"):
        apply(s, SpecDrafted(version=2, tier="exact", proven=True))


def test_stale_results_are_refused():
    s = run(ProjectState(), FilesUploaded(files=FILES), *confirmed(1, KEY1))
    s = run(s, *confirmed(2, KEY2))            # spec revised while the v1 baseline was running
    with pytest.raises(GateError, match="spec v1"):
        apply(s, BaselineSucceeded(spec_version=1, baseline_id="b-old"))
    s = apply(s, BaselineSucceeded(spec_version=2, baseline_id="b2"))
    bad = Bindings(spec_version=2, baseline_id="b-old", candidate_hash=cand(1), validation_id="v1")
    with pytest.raises(GateError, match="baseline"):
        apply(s, CandidateValidated(bindings=bad, passed=True))


def test_approval_must_match_current_bindings():
    s, b = validated(at_translation())
    with pytest.raises(GateError, match="does not match"):
        apply(s, CandidateApproved(bindings=b.model_copy(update={"validation_id": "other"})))


def test_failing_validation_cannot_be_approved_and_goes_to_repair():
    s, b = validated(at_translation(), passed=False)
    assert where(s) == (Step.REPAIR, Status.WORKING, None)
    with pytest.raises(GateError):
        apply(s, CandidateApproved(bindings=b))


def test_rejection_carries_comment_to_repair_and_back():
    s, b = validated(at_translation())
    s = apply(s, CandidateRejected(bindings=b, comment="reference level of Area is wrong"))
    assert where(s) == (Step.REPAIR, Status.WORKING, None) and s.ds_comment.startswith("reference")
    s = apply(s, DiagnosisSubmitted(spec_version=1, candidate_hash=cand(1), diagnosis_id="d1",
                                    spec_change_needed=False))
    assert where(s) == (Step.TRANSLATION, Status.WORKING, None)
    assert (s.repairs, s.diagnosis_id, s.ds_comment, s.candidate_hash) == (1, "d1", None, None)


def test_retry_limit_escalates_and_can_be_raised():
    s = at_translation()
    for n in range(1, 4):
        s, _ = validated(s, n, passed=False)
        s = apply(s, DiagnosisSubmitted(spec_version=1, candidate_hash=cand(n), diagnosis_id=f"d{n}",
                                        spec_change_needed=False))
    assert s.repairs == 3
    s, _ = validated(s, 4, passed=False)
    assert where(s) == (Step.REPAIR, Status.AWAITING_DS, Reason.RETRY_LIMIT)
    with pytest.raises(GateError, match="raise the retry limit"):
        apply(s, ResumeRepair())
    s = apply(s, ResumeRepair(additional_attempts=1))
    assert where(s) == (Step.REPAIR, Status.WORKING, None) and s.retry_limit == 4


def test_diagnosis_needing_spec_change_routes_to_ds():
    s, _ = validated(at_translation(), passed=False)
    s = apply(s, DiagnosisSubmitted(spec_version=1, candidate_hash=cand(1), diagnosis_id="d1",
                                    spec_change_needed=True))
    assert where(s) == (Step.REPAIR, Status.AWAITING_DS, Reason.SPEC_CHANGE)
    # the DS can dismiss it and let the repair go ahead ...
    assert where(apply(s, ResumeRepair())) == (Step.TRANSLATION, Status.WORKING, None)
    # ... or revise the spec, which restarts translation on the same baseline
    s = run(s, *confirmed(2, KEY1))
    assert where(s) == (Step.TRANSLATION, Status.WORKING, None) and s.baseline_id == "b1"


def test_from_a_stopped_repair_the_ds_can_return_to_the_analyst():
    s, _ = validated(at_translation(), passed=False)
    with pytest.raises(GateError):
        apply(s, DsMessaged())                                  # not while the Diagnostician works
    s = apply(s, DiagnosisSubmitted(spec_version=1, candidate_hash=cand(1), diagnosis_id="d1",
                                    spec_change_needed=True))
    s = apply(s, DsMessaged())
    assert where(s) == (Step.ALIGNMENT, Status.WORKING, None) and s.spec_version == 1
    s = run(s, AnalystReplied(), *confirmed(2, KEY1))
    assert where(s) == (Step.TRANSLATION, Status.WORKING, None) and (s.repairs, s.diagnosis_id) == (0, None)


def test_spec_change_after_approval_voids_it():
    s, b = validated(at_translation())
    s = apply(s, CandidateApproved(bindings=b))
    assert s.approval_valid()
    # tolerances only: baseline key unchanged, restart at step 4 with the baseline kept
    s4 = run(s, *confirmed(2, KEY1))
    assert (s4.step, s4.baseline_id, s4.approval, s4.repairs) == (Step.TRANSLATION, "b1", None, 0)
    assert not s4.approval_valid()
    # baseline slice changed: restart at step 3
    s3 = run(s, *confirmed(2, KEY2))
    assert (s3.step, s3.baseline_id) == (Step.BASELINE, None)
    with pytest.raises(GateError):
        apply(s3, ReportBuilt(spec_version=2, report_id="r"))


def test_failure_cancel_and_retry():
    s = run(ProjectState(), FilesUploaded(files=FILES), *confirmed())
    f = apply(s, StepFailed(step=Step.BASELINE, spec_version=1, detail="Rscript exited 1"))
    assert where(f) == (Step.BASELINE, Status.AWAITING_DS, Reason.STEP_FAILED) and f.detail == "Rscript exited 1"
    assert where(apply(f, RetryStep())) == (Step.BASELINE, Status.WORKING, None)
    c = apply(s, Cancelled(step=Step.BASELINE))
    assert c.reason == Reason.CANCELLED
    with pytest.raises(GateError):
        apply(s, Cancelled(step=Step.TRANSLATION))
    with pytest.raises(GateError, match="nothing to retry"):
        apply(s, RetryStep())


def test_chat_turns_alternate_between_the_ds_and_the_analyst():
    s = run(ProjectState(), FilesUploaded(files=FILES))
    with pytest.raises(GateError):
        apply(s, DsMessaged())                                     # the Analyst has the opening turn
    with pytest.raises(GateError, match="still working"):
        apply(run(s, SpecDrafted(version=1, tier="exact", proven=True), DsMessaged()),
              SpecConfirmed(version=1, tier="exact", proven=True, baseline_key=KEY1))
    s = run(s, AnalystReplied())
    assert where(s) == (Step.ALIGNMENT, Status.AWAITING_DS, Reason.ANALYST_REPLIED)
    with pytest.raises(GateError):
        apply(s, AnalystReplied())
    s = run(s, DsMessaged(), SpecDrafted(version=1, tier="exact", proven=True))
    assert where(s) == (Step.ALIGNMENT, Status.AWAITING_DS, Reason.SPEC_DRAFTED)
    s = run(s, DsMessaged(), AnalystReplied())                     # a question answered; the draft stands
    assert where(run(s, SpecConfirmed(version=1, tier="exact", proven=True, baseline_key=KEY1))) == (
        Step.BASELINE, Status.WORKING, None)
    with pytest.raises(GateError):
        apply(at_translation(), DsMessaged())


def test_a_failed_baseline_returns_to_the_analyst_for_a_corrected_spec():
    s = run(ProjectState(), FilesUploaded(files=FILES), *confirmed())
    with pytest.raises(GateError):
        apply(s, DsMessaged())                                     # not while the baseline runs
    s = run(s, StepFailed(step=Step.BASELINE, spec_version=1, detail="R error"), DsMessaged())
    assert where(s) == (Step.ALIGNMENT, Status.WORKING, None) and s.spec_version == 1
    s = run(s, *confirmed(version=2, key=KEY1))                    # same baseline key, but no baseline exists
    assert where(s) == (Step.BASELINE, Status.WORKING, None) and s.spec_version == 2
