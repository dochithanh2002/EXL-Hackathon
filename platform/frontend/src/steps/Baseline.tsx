import { useQuery } from "@tanstack/react-query";
import { Code, Details, ErrorText, Fields, Mono, Muted, Panel, RunLink } from "@/components/common";
import { Coefficients } from "@/components/validation";
import { type BaselineBody, type Evidence, get, type Job, p } from "@/lib/api";
import { num, short, when } from "@/lib/format";
import type { StepProps } from "@/pages/Project";

export function Baseline({ project, rev }: StepProps) {
  const s = project.state;
  const b = useQuery({
    queryKey: ["evidence", project.id, s.baseline_id],
    queryFn: () => get<Evidence<BaselineBody>>(p(project.id, `/evidence/${s.baseline_id}`)),
    enabled: !!s.baseline_id,
    staleTime: Infinity,
  });
  const failed = s.step === 3 && s.status === "awaiting_ds";
  if (!s.baseline_id) return failed ? <BaselineLog project={project} rev={rev} /> : s.status === "working" ? null : <Muted>R reference results are computed once a model specification is confirmed.</Muted>;
  if (!b.data) return <ErrorText error={b.error} />;
  const body = b.data.body;
  const fit = Object.entries(body.fit).filter(([, v]) => typeof v === "number" || typeof v === "string");
  const files = (name: string) => p(project.id, `/evidence/${b.data.id}/files/${name}`);
  return (
    <>
      <Panel title={`R Reference Results · Model Specification v${body.spec_version}`} actions={<RunLink pid={project.id} runId={body.mlflow_run_id} evidenceId={b.data.id} />}>
        <div className="flex flex-col gap-4">
          <div className="grid gap-6 md:grid-cols-2">
            <Fields rows={[
              ["Captured", when(b.data.created_at)],
              ["R", String(body.meta.r_version ?? "—")],
              ["Statistical model", [(body.meta.class as string[] | undefined)?.[0], body.meta.family, body.meta.link].filter(Boolean).join(" · ")],
              ["Converged", body.meta.converged === null || body.meta.converged === undefined ? "not reported" : body.meta.converged ? "yes" : <span className="font-medium text-destructive">no</span>],
              ["Fitted rows", num(body.meta.n_fit)],
              ["Files", (
                <span className="flex flex-wrap gap-x-3">
                  {Object.keys(body.files).map((f) => <a key={f} className="underline-offset-4 hover:underline" href={files(f)}>{f}</a>)}
                </span>
              )],
            ]} />
            <Fields rows={fit.map(([k, v]) => [k.replaceAll("_", " "), <span className="font-mono text-xs">{num(v)}</span>])} />
          </div>
          <Details rows={[
            ["Id", b.data.id],
            ["Hash", <Mono>{short(body.hash)}</Mono>],
            ["Environment", <Mono>{short(body.environment_version)}</Mono>],
          ]} />
        </div>
      </Panel>
      <Coefficients pid={project.id} evidenceId={b.data.id} />
    </>
  );
}

function BaselineLog({ project, rev }: StepProps) {
  const jobs = useQuery({ queryKey: ["jobs", project.id, rev], queryFn: () => get<Job[]>(p(project.id, "/jobs")) });
  const last = jobs.data?.filter((j) => j.kind === "r_baseline").at(-1);
  const log = useQuery({
    queryKey: ["log", project.id, last?.id],
    queryFn: () => get<string>(p(project.id, `/jobs/${last!.id}/log`)),
    enabled: !!last,
    retry: false,
  });
  if (!log.data) return null;
  return (
    <Panel title="Step log">
      <Code code={log.data} />
    </Panel>
  );
}
