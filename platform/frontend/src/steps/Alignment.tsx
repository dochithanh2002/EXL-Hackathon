import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Code, Diff, ErrorText, Fields, Mono, Muted, Panel } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/input";
import type * as React from "react";
import { get, type ModelSpec, p, post, type SpecRow, type SpecVersion, type Tier, type Tolerance } from "@/lib/api";
import { num } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { StepProps } from "@/pages/Project";

export function TierBadge({ tier, proven }: { tier: Tier; proven: boolean }) {
  const variant = tier === "exact" ? (proven ? "pass" : "secondary") : tier === "approximate" ? "warn" : "fail";
  const label = tier === "unsupported" ? "no faithful Python implementation" : `${tier} parity`;
  return <Badge variant={variant}>{label}{tier !== "unsupported" && !proven ? " · no passing fixture yet" : ""}</Badge>;
}

const tol = (t: Tolerance) => `atol ${num(t.atol)}, rtol ${num(t.rtol)}`;

function List({ title, items, tone }: { title: string; items: React.ReactNode[]; tone?: string }) {
  if (!items.length) return null;
  return (
    <div>
      <h4 className={cn("mb-2 text-sm font-medium", tone)}>{title}</h4>
      <ul className="flex list-disc flex-col gap-1 pl-5 text-sm">{items.map((x, i) => <li key={i}>{x}</li>)}</ul>
    </div>
  );
}

export function Alignment({ project, rev }: StepProps) {
  const s = project.state;
  const specs = useQuery({ queryKey: ["specs", project.id, rev], queryFn: () => get<SpecRow[]>(p(project.id, "/specs")) });
  const latest = specs.data?.at(-1)?.version;
  const [picked, setPicked] = useState<number>();
  const version = picked ?? latest;
  if (specs.data && !specs.data.length) return <Muted>The Analyst is reading the R code and the dataset profile; its draft model specification appears here.</Muted>;
  if (!version) return <ErrorText error={specs.error} />;
  return (
    <>
      {specs.data && specs.data.length > 1 && (
        <div className="flex items-center gap-2 text-sm">
          <Select className="w-44" value={version} onChange={(e) => setPicked(Number(e.target.value))}>
            {specs.data.map((v) => (
              <option key={v.version} value={v.version}>
                v{v.version}{v.version === s.spec_version ? " · confirmed" : ""}{v.version === latest ? " · latest" : ""}
              </option>
            ))}
          </Select>
        </div>
      )}
      <Spec project={project} rev={rev} version={version} />
    </>
  );
}

function Spec({ project, version }: StepProps & { version: number }) {
  const qc = useQueryClient();
  const s = project.state;
  const sv = useQuery({ queryKey: ["spec", project.id, version], queryFn: () => get<SpecVersion>(p(project.id, `/specs/${version}`)), staleTime: Infinity });
  const rFiles = s.files.filter((f) => f.role === "r_code");
  const [file, setFile] = useState(rFiles[0]?.name);
  const original = useQuery({ queryKey: ["file", project.id, file], queryFn: () => get<string>(p(project.id, `/files/${encodeURIComponent(file!)}`)), enabled: !!file, staleTime: Infinity });
  const [view, setView] = useState<"diff" | "slice">("diff");
  const [accepted, setAccepted] = useState(false);
  const confirm = useMutation({
    mutationFn: () => post(p(project.id, `/specs/${version}/confirm`), { parity_risk_accepted: accepted }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["project", project.id] }),
  });
  if (!sv.data) return <ErrorText error={sv.error} />;
  const { spec, support } = sv.data;
  const risk = support.tier === "approximate" || !support.proven;
  const canConfirm = version === s.drafted_version && version > (s.spec_version ?? 0) && support.tier !== "unsupported"
    && !(s.step === 2 && s.status === "working") && s.step !== 1;
  return (
    <>
      <Panel title={<span className="flex items-center gap-2">Model Specification v{version} <TierBadge tier={support.tier} proven={support.proven} /></span>}>
        <div className="flex flex-col gap-5">
          <p className="text-sm">{spec.target_model}</p>
          {spec.limitation && (
            <div className="rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2 text-sm">{spec.limitation}</div>
          )}
          <Fields rows={[
            ["Statistical model", <Mono>{spec.r_function}</Mono>],
            ["Python implementation", <Mono>{spec.python_implementation}</Mono>],
            ["Family", spec.family ? `${spec.family}${spec.link ? ` (${spec.link})` : ""}` : null],
            ["Formula", <Mono>{spec.formula}</Mono>],
            ["Split", <>{spec.split.method}{spec.split.seed !== null ? `, seed ${spec.split.seed}` : ""}</>],
            ["R packages", spec.r_packages.join(", ") || null],
          ]} />
          <Semantics spec={spec} />
          {spec.preprocessing.length > 0 && (
            <div>
              <h4 className="mb-2 text-sm font-medium">Preprocessing</h4>
              <ol className="flex list-decimal flex-col gap-1 pl-5 text-sm">
                {spec.preprocessing.map((x) => (
                  <li key={x.order}>{x.description} <span className="text-muted-foreground">{x.columns.join(", ")}</span></li>
                ))}
              </ol>
            </div>
          )}
          <List title="Not translated" tone="text-destructive" items={spec.unsupported_items} />
          <List title="Substitutions" items={spec.substitutions.map((x) => <>{x.original} → {x.proposed} <span className="text-muted-foreground">{x.rationale}</span></>)} />
        </div>
      </Panel>

      <Assessment spec={spec} />

      <Panel
        title="baseline.R"
        actions={
          <>
            {view === "diff" && rFiles.length > 1 && (
              <Select className="h-8 w-48" value={file} onChange={(e) => setFile(e.target.value)}>
                {rFiles.map((f) => <option key={f.name}>{f.name}</option>)}
              </Select>
            )}
            <div className="flex rounded-md border p-0.5">
              {(["diff", "slice"] as const).map((v) => (
                <button key={v} className={cn("rounded px-2 py-0.5 text-xs", view === v && "bg-muted font-medium")} onClick={() => setView(v)}>
                  {v === "diff" ? `against ${file}` : "slice"}
                </button>
              ))}
            </div>
          </>
        }
      >
        {view === "slice" || !original.data ? <Code code={spec.baseline_r} /> : <Diff before={original.data} after={spec.baseline_r} />}
      </Panel>

      {canConfirm && (
        <div className="flex items-center justify-end gap-4">
          {risk && (
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={accepted} onChange={(e) => setAccepted(e.target.checked)} />
              {support.tier === "approximate" ? "I accept approximate parity for this model" : support.known ? "I accept that this pair has no passing fixture yet" : "I accept that this R → Python pair is untested"}
            </label>
          )}
          <ErrorText error={confirm.error} />
          <Button disabled={(risk && !accepted) || confirm.isPending} onClick={() => confirm.mutate()}>Confirm Model Specification v{version}</Button>
        </div>
      )}
    </>
  );
}

function Semantics({ spec }: { spec: ModelSpec }) {
  const pr = spec.prediction;
  const t = spec.tolerances;
  return (
    <div>
      <h4 className="mb-2 text-sm font-medium">Statistical semantics</h4>
      <Fields rows={[
        ["Response", <><Mono>{spec.response}</Mono> {spec.response_type.replace("_", " ")}</>],
        ["Event", spec.event && <>{spec.event.meaning} <span className="text-muted-foreground">coded <Mono>{spec.event.coding}</Mono>; otherwise {spec.event.complement}</span></>],
        ["Grouped binomial", spec.grouped_binomial && <>{spec.grouped_binomial.form.replaceAll("_", " ")}: trials <Mono>{spec.grouped_binomial.trials}</Mono>{spec.grouped_binomial.successes && <>, successes <Mono>{spec.grouped_binomial.successes}</Mono></>}</>],
        ["Weights", spec.weights && <><Mono>{spec.weights.expression}</Mono> {spec.weights.type} weights, normalization {spec.weights.normalization.replaceAll("_", " ")}</>],
        ["Offset", spec.offset && <Mono>{spec.offset}</Mono>],
        ["Prediction", <>{pr.units} <span className="text-muted-foreground">{pr.kind.replace("_", " ")}, {pr.quantity.replace("_", " ")}{pr.exposure ? ` per ${pr.exposure}` : ""}</span></>],
        ["Prediction keys", pr.keys.length ? pr.keys.join(", ") : null],
        ["Aligned on", pr.alignment_keys.join(" × ")],
        ["R prediction", <Mono>{pr.r_predict}</Mono>],
        ["Evaluation metrics", spec.metrics.join(", ") || null],
        ["Optional metrics", Object.entries(spec.optional_metrics).map(([s, ms]) => `${s}: ${ms.join(", ")}`).join("; ") || "None"],
        ["R convergence check", spec.convergence.r_check || spec.convergence.mode],
        ["Convergence rationale", spec.convergence.rationale],
        ["Tolerances", (
          <span className="flex flex-col">
            {(["coefficients", "std_errors", "predictions", "fit_statistics", "metrics"] as const).map((k) => (
              <span key={k}>{k.replace("_", " ")}: {tol(t[k])}</span>
            ))}
            {Object.entries(t.per_metric).map(([k, v]) => <span key={k}>{k}: {tol(v)}</span>)}
          </span>
        )],
      ]} />
    </div>
  );
}

function Assessment({ spec }: { spec: ModelSpec }) {
  const a = spec.assessment;
  const criteria = Object.entries(a.criteria).filter(([, v]) => v !== null);
  return (
    <Panel title="Financial model assessment">
      <div className="flex flex-col gap-5">
        <Fields rows={[
          ["Business use", a.business_use],
          ["Intended population", a.intended_population],
          ["Sample", a.sample_population],
          ["Evaluation sets", (
            <span className="flex flex-col">
              {a.evaluation_sets.map((e) => <span key={e.set}><Mono>{e.set}</Mono> {e.role.replace("_", " ")} <span className="text-muted-foreground">{e.description}</span></span>)}
            </span>
          )],
          ["Period column", a.period_column && <Mono>{a.period_column}</Mono>],
          ["Decision", a.decision && `${a.decision.action} at or above ${num(a.decision.threshold)}`],
          ["Acceptance criteria", criteria.length ? criteria.map(([k, v]) => `${k.replaceAll("_", " ")} ${num(v)}`).join(" · ") : null],
        ]} />
        <List title="Still to provide" tone="text-[oklch(0.5_0.12_75)]" items={a.missing_information} />
      </div>
    </Panel>
  );
}
