import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Code, Diff, ErrorText, Fields, Mono, Muted, Panel } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/input";
import { get, p, post, type SpecRow, type SpecVersion, type Tier } from "@/lib/api";
import { num } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { StepProps } from "@/pages/Project";

export function TierBadge({ tier, proven }: { tier: Tier; proven: boolean }) {
  const variant = tier === "exact" ? (proven ? "pass" : "secondary") : tier === "approximate" ? "warn" : "fail";
  return <Badge variant={variant}>{tier}{tier === "exact" && !proven ? " · no passing fixture yet" : ""}</Badge>;
}

export function Alignment({ project, rev }: StepProps) {
  const s = project.state;
  const specs = useQuery({ queryKey: ["specs", project.id, rev], queryFn: () => get<SpecRow[]>(p(project.id, "/specs")) });
  const latest = specs.data?.at(-1)?.version;
  const [picked, setPicked] = useState<number>();
  const version = picked ?? latest;
  if (specs.data && !specs.data.length) return <Muted>The Analyst is reading the R code and the dataset profile; its draft spec appears here.</Muted>;
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
      <Panel title={<span className="flex items-center gap-2">Spec v{version} <TierBadge tier={support.tier} proven={support.proven} /></span>}>
        <div className="flex flex-col gap-5">
          <p className="text-sm">{spec.target_model}</p>
          <Fields rows={[
            ["R", <Mono>{spec.r_function}</Mono>],
            ["statsmodels", <Mono>{spec.statsmodels_counterpart}</Mono>],
            ["Family", spec.family ? `${spec.family}${spec.link ? ` (${spec.link})` : ""}` : null],
            ["Formula", <Mono>{spec.formula}</Mono>],
            ["Response", <Mono>{spec.response}</Mono>],
            ["Offset", spec.offset && <Mono>{spec.offset}</Mono>],
            ["Weights", spec.weights && <Mono>{spec.weights}</Mono>],
            ["Split", <>{spec.split.method}{spec.split.seed !== null ? `, seed ${spec.split.seed}` : ""}</>],
            ["KPIs", spec.kpis.join(", ") || null],
            ["Tolerances", Object.entries(spec.tolerances).map(([k, v]) => `${k} ${num(v)}`).join(" · ")],
            ["R packages", spec.r_packages.join(", ") || null],
          ]} />
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
          {spec.unsupported_items.length > 0 && (
            <div>
              <h4 className="mb-2 text-sm font-medium text-destructive">Unsupported</h4>
              <ul className="list-disc pl-5 text-sm">{spec.unsupported_items.map((u) => <li key={u}>{u}</li>)}</ul>
            </div>
          )}
          {spec.substitutions.length > 0 && (
            <div>
              <h4 className="mb-2 text-sm font-medium">Substitutions</h4>
              <ul className="flex flex-col gap-1 text-sm">
                {spec.substitutions.map((x) => (
                  <li key={x.original}>{x.original} → {x.proposed} <span className="text-muted-foreground">{x.rationale}</span></li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </Panel>

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
              {support.tier === "approximate" ? "I accept approximate parity for this model" : "I accept that this pair has no passing fixture yet"}
            </label>
          )}
          <ErrorText error={confirm.error} />
          <Button disabled={(risk && !accepted) || confirm.isPending} onClick={() => confirm.mutate()}>Confirm spec v{version}</Button>
        </div>
      )}
    </>
  );
}
