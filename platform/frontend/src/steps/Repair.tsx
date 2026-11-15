import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { ActionBar, ErrorText, Fields, Muted, Panel, RunLink } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { p, post } from "@/lib/api";
import { useAttempts } from "@/lib/attempts";
import type { StepProps } from "@/pages/Project";

const CATEGORY = {
  migration_defect: "migration defect", specification: "specification mismatch",
  not_reproducible: "not reproducible in Python", model_fitness: "model fitness concern",
};

export function Repair({ project, rev }: StepProps) {
  const s = project.state;
  const { attempts } = useAttempts(project, rev);
  const decisions = new Map(project.decisions.map((d) => [d.bindings.validation_id, d]));
  const latest = [...attempts].reverse().find((a) => a.diagnosis)?.diagnosis;
  return (
    <>
      <Panel title={`Discrepancy analysis · ${s.repairs} of ${s.retry_limit} repair attempts used`}>
        {attempts.length ? (
          <Table>
            <THead>
              <TR><TH>Attempt</TH><TH>Parity</TH><TH>Failed checks</TH><TH>DS</TH><TH>Root cause</TH></TR>
            </THead>
            <TBody>
              {attempts.map((a) => {
                const v = a.validation;
                const d = v && decisions.get(v.id);
                return (
                  <TR key={a.candidate.id} className="align-top">
                    <TD>{a.candidate.body.attempt}</TD>
                    <TD>{v ? <Badge variant={v.body.passed ? "pass" : "fail"}>{v.body.passed ? "passed" : "failed"}</Badge> : <Badge variant="secondary">did not run</Badge>}</TD>
                    <TD className="max-w-64 text-xs">{a.diagnosis?.body.failed_checks.join("; ")}</TD>
                    <TD className="max-w-48 text-xs">{d ? <><Badge variant={d.decision === "approved" ? "pass" : "warn"}>{d.decision === "approved" ? "accepted" : d.decision}</Badge> {d.comment}</> : null}</TD>
                    <TD className="max-w-96 text-xs">{a.diagnosis?.body.root_cause}</TD>
                  </TR>
                );
              })}
            </TBody>
          </Table>
        ) : <Muted>No attempts under the current model specification.</Muted>}
      </Panel>
      {latest && (
        <Panel title="Latest diagnosis" actions={<RunLink pid={project.id} runId={latest.body.mlflow_run_id} evidenceId={latest.id} />}>
          <div className="flex flex-col gap-4 text-sm">
            <Fields rows={[
              ["Finding", latest.body.category ? <Badge variant={latest.body.category === "migration_defect" ? "secondary" : "warn"}>{CATEGORY[latest.body.category]}</Badge> : null],
              ["Root cause", latest.body.root_cause],
              ["Limitation", latest.body.limitation],
              ["Specification change", latest.body.spec_change_needed ? <Badge variant="warn">needed</Badge> : "no"],
              ["Proposed change", latest.body.proposed_spec_change],
              ["DS comment", latest.body.ds_comment],
            ]} />
            <div>
              <h4 className="mb-1 font-medium">Evidence</h4>
              <ul className="list-disc pl-5">{latest.body.evidence.map((x) => <li key={x}>{x}</li>)}</ul>
            </div>
            {(latest.body.assessment_findings ?? []).length > 0 && (
              <div>
                <h4 className="mb-1 font-medium">Model validation</h4>
                <ul className="list-disc pl-5">{latest.body.assessment_findings!.map((x) => <li key={x}>{x}</li>)}</ul>
              </div>
            )}
            {latest.body.repair_instructions.length > 0 && (
              <div>
                <h4 className="mb-1 font-medium">Repair instructions</h4>
                <ol className="list-decimal pl-5">{latest.body.repair_instructions.map((x) => <li key={x}>{x}</li>)}</ol>
              </div>
            )}
          </div>
        </Panel>
      )}
      {s.step === 5 && s.status === "awaiting_ds" && <Resume project={project} rev={rev} />}
    </>
  );
}

function Resume({ project }: StepProps) {
  const qc = useQueryClient();
  const s = project.state;
  const [more, setMore] = useState(1);
  const resume = useMutation({
    mutationFn: (n: number) => post(p(project.id, "/resume"), { additional_attempts: n }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["project", project.id] }),
  });
  if (s.reason !== "retry_limit" && s.reason !== "spec_change") return null;
  return (
    <ActionBar tone="warn">
      <div className="flex items-center gap-3">
        <span className="shrink-0 font-medium">{s.reason === "retry_limit" ? "Repair limit reached" : "The Diagnostician proposes a specification change"}</span>
        <span className="text-muted-foreground">Write to the Analyst for a new model specification, or</span>
        <div className="ml-auto flex items-center gap-2">
          {s.reason === "retry_limit" ? (
            <>
              <Input type="number" min={1} max={20} value={more} onChange={(e) => setMore(Number(e.target.value))} className="h-9 w-16" />
              <Button disabled={resume.isPending || more < 1} onClick={() => resume.mutate(more)}>Allow more attempts</Button>
            </>
          ) : (
            <Button disabled={resume.isPending} onClick={() => resume.mutate(0)}>Repair under the current specification</Button>
          )}
        </div>
      </div>
      <ErrorText error={resume.error} />
    </ActionBar>
  );
}
