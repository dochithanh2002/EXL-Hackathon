import { useQuery } from "@tanstack/react-query";
import type * as React from "react";
import { ErrorText, Panel, RunLink } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { type AssessmentAnswer, type Check, type CoefficientRow, type Evidence, get, p, type ValidationBody } from "@/lib/api";
import { num } from "@/lib/format";
import { cn } from "@/lib/utils";

/** For a ratio |Δ| / (atol + rtol·|R|): pass within a tenth of the tolerance, near up to it, fail beyond it. */
export function closeness(ratio: number | null | undefined): "pass" | "near" | "fail" {
  if (ratio === null || ratio === undefined || Number.isNaN(ratio)) return "fail";
  return ratio > 1 ? "fail" : ratio > 0.1 ? "near" : "pass";
}

const TONE = { pass: "text-pass", near: "text-warn-foreground", fail: "text-destructive font-medium" };
const STATUS_BADGE = { passed: "pass", failed: "fail", unavailable: "secondary" } as const;

function expected(c: Check): string {
  if (c.kind === "count") return "0";
  if (c.kind === "flag") return "yes";
  if (c.kind === "minimum") return `≥ ${num(c.tolerance)}${c.total !== null ? ` of ${num(c.total)}` : ""}`;
  return `${num(c.atol)} + ${num(c.rtol)}·|R|`;
}

function observed(c: Check): string {
  if (c.status === "unavailable") return "—";
  if (c.kind === "flag") return c.value ? "yes" : "no";
  if (c.kind === "tolerance") return num(c.max_abs_diff);
  return num(c.value);
}

export function Checks({ v, actions }: { v: ValidationBody; actions?: React.ReactNode }) {
  const required = v.checks.filter((c) => c.required);
  const passed = required.filter((c) => c.passed).length;
  const unavailable = v.checks.filter((c) => c.status === "unavailable").length;
  return (
    <Panel
      title={
        <span className="flex items-center gap-2">
          Parity checks
          <Badge variant={v.passed ? "pass" : "fail"}>{passed} of {required.length} passed</Badge>
          {unavailable > 0 && <Badge variant="secondary">{unavailable} unavailable</Badge>}
        </span>
      }
      actions={actions}
    >
      {v.error && (
        <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm">
          <p className="font-medium">The Python code failed at {String(v.error.stage)}: {String(v.error.type)}</p>
          <pre className="mt-2 overflow-x-auto text-xs whitespace-pre-wrap">{String(v.error.message ?? "")}</pre>
        </div>
      )}
      {v.checks.length > 0 && (
        <Table>
          <THead>
            <TR>
              <TH>Check</TH><TH className="text-right">|Python − R|</TH><TH className="text-right">Allowed</TH>
              <TH className="text-right">Ratio</TH><TH />
            </TR>
          </THead>
          <TBody>
            {v.checks.map((c) => {
              const tone = c.status !== "passed" ? "fail" : c.kind === "tolerance" ? closeness(c.value) : "pass";
              return (
                <TR key={c.name} className={cn(!c.required && "text-muted-foreground")}>
                  <TD title={c.reason || undefined}>
                    {c.name}
                    {c.reason && <span className="ml-2 text-xs text-muted-foreground">{c.reason}</span>}
                  </TD>
                  <TD className={cn("text-right font-mono text-xs tabular-nums", c.status !== "unavailable" && TONE[tone])}>{observed(c)}</TD>
                  <TD className="text-right font-mono text-xs tabular-nums text-muted-foreground">{expected(c)}</TD>
                  <TD className="text-right font-mono text-xs tabular-nums">{c.kind === "tolerance" && c.status !== "unavailable" ? num(c.value) : ""}</TD>
                  <TD className="w-24 text-right"><Badge variant={STATUS_BADGE[c.status]}>{c.status}</Badge></TD>
                </TR>
              );
            })}
          </TBody>
        </Table>
      )}
      {v.notes.length > 0 && (
        <ul className="mt-3 list-disc pl-5 text-xs text-muted-foreground">{v.notes.map((n) => <li key={n}>{n}</li>)}</ul>
      )}
    </Panel>
  );
}

export function EvaluationMetrics({ v }: { v: ValidationBody }) {
  const rows = Object.entries(v.metrics ?? {}).flatMap(([set, ks]) =>
    Object.entries(ks).filter(([, x]) => x.r !== null || x.python !== null).map(([k, x]) => ({ set, k, ...x })));
  if (!rows.length) return null;
  return (
    <Panel title="Evaluation metrics">
      <Table>
        <THead>
          <TR><TH>Metric</TH><TH>Set</TH><TH className="text-right">R</TH><TH className="text-right">Python</TH></TR>
        </THead>
        <TBody>
          {rows.map((r) => (
            <TR key={`${r.set}-${r.k}`}>
              <TD>{r.k.replaceAll("_", " ")}</TD>
              <TD>{r.set}</TD>
              <TD className="text-right font-mono text-xs tabular-nums">{num(r.r)}</TD>
              <TD className="text-right font-mono text-xs tabular-nums">{num(r.python)}</TD>
            </TR>
          ))}
        </TBody>
      </Table>
    </Panel>
  );
}

const QUESTIONS = [
  ["sample_suitability", "Sample suitability"], ["stability", "Stability"], ["calibration", "Calibration"],
  ["decision_performance", "Decision performance"],
] as const;
const ASSESSMENT_BADGE = { assessed: "pass", "partially assessed": "warn", "not assessed": "secondary" } as const;

/** Headline numbers per question; the full summaries are in the validation record and the report. */
function headline(q: string, r: Record<string, any>): [string, unknown][] {
  if (q === "sample_suitability") {
    const sets = Object.entries(r.sets ?? {}) as [string, any][];
    return [
      ...sets.map(([s, e]): [string, unknown] => [s, e.event_rate !== undefined ? `${num(e.rows)} rows, event rate ${num(e.event_rate)}` : `${num(e.rows)} rows`]),
      [`per parameter (${r.per_parameter_basis ?? ""})`, r.per_parameter],
    ];
  }
  if (q === "stability") return [["largest PSI", r.max_psi], ...Object.entries(r.psi_vs_development ?? {}).map(([s, v]): [string, unknown] => [`PSI ${s} vs development`, v])];
  if (q === "calibration") {
    return Object.entries(r.sets ?? {}).flatMap(([s, e]: [string, any]) =>
      "coverage" in e ? [[`${s} coverage gap`, e.max_deviation] as [string, unknown]]
        : Object.entries(e).map(([k, c]: [string, any]): [string, unknown] => [`${s}${k === "value" ? "" : ` ${k}`} observed / expected`, c.observed_expected]));
  }
  if (q === "decision_performance") {
    return Object.entries(r.discrimination ?? {}).map(([s, d]: [string, any]): [string, unknown] => [`${s} AUC · KS`, `${num(d.auc)} · ${num(d.ks)}`]);
  }
  return [];
}

export function FinancialAssessment({ v }: { v: ValidationBody }) {
  const a = v.assessment?.python ?? v.assessment?.r;
  if (!a) return null;
  return (
    <Panel title="Financial model assessment">
      <div className="grid gap-4 md:grid-cols-2">
        {QUESTIONS.map(([q, title]) => {
          const ans: AssessmentAnswer | undefined = a[q];
          if (!ans) return null;
          return (
            <div key={q} className="flex flex-col gap-2 rounded-md border p-3 text-sm">
              <div className="flex items-center gap-2">
                <span className="font-medium">{title}</span>
                <Badge variant={ASSESSMENT_BADGE[ans.status]}>{ans.status[0].toUpperCase() + ans.status.slice(1)}</Badge>
                {ans.meets !== null && <Badge variant={ans.meets ? "pass" : "fail"}>{ans.criterion} {ans.meets ? "met" : "not met"}</Badge>}
              </div>
              {headline(q, ans.results).length > 0 && (
                <dl className="grid grid-cols-[1fr_max-content] gap-x-4 gap-y-0.5 text-xs">
                  {headline(q, ans.results).map(([k, x]) => (
                    <div key={k} className="contents">
                      <dt className="text-muted-foreground">{k}</dt>
                      <dd className="text-right font-mono tabular-nums">{num(x)}</dd>
                    </div>
                  ))}
                </dl>
              )}
              {ans.needs.length > 0 && <p className="text-xs text-muted-foreground">Needs {ans.needs.join("; ")}.</p>}
            </div>
          );
        })}
      </div>
    </Panel>
  );
}

const CHECKS = [
  ["baseline_comparison", "Baseline comparison"], ["calibration", "Calibration"],
  ["subgroup_performance", "Subgroup performance"], ["stability", "Stability"], ["sensitivity", "Sensitivity"],
  ["assumptions", "Model assumptions"],
] as const;
const CHECK_BADGE = { executed: "pass", "partially executed": "warn", "not executed": "fail", "not planned": "secondary" } as const;

/** The validation plan's checks as the platform ran them, with their plots; business fitness, apart from parity. */
export function ModelValidation({ pid, v }: { pid: string; v: Evidence<ValidationBody> }) {
  const ev = v.body.evaluation;
  if (!ev?.checks) return null;
  const plots = (v.body.plots ?? []).filter((x) => x.section.startsWith("validation."));
  return (
    <Panel title="Model validation" actions={<RunLink pid={pid} runId={v.body.mlflow_run_id} />}>
      <div className="flex flex-col gap-5">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {CHECKS.map(([k, title]) => {
            const c = ev.checks[k];
            if (!c) return null;
            return (
              <div key={k} className="flex flex-col gap-1.5 rounded-md border p-3 text-sm">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{title}</span>
                  <Badge variant={CHECK_BADGE[c.status]}>{c.status[0].toUpperCase() + c.status.slice(1)}</Badge>
                </div>
                <p className="text-xs text-muted-foreground">{c.rationale}</p>
                {c.needs.length > 0 && <p className="text-xs">Needs {c.needs.join("; ")}.</p>}
              </div>
            );
          })}
        </div>
        {plots.length > 0 && (
          <div className="grid gap-4 xl:grid-cols-2">
            {plots.map((x) => (
              <figure key={x.name} className="min-w-0">
                <img src={p(pid, `/evidence/${v.id}/plots/${x.name}`)} alt={x.title} className="w-full rounded-md border" loading="lazy" />
              </figure>
            ))}
          </div>
        )}
        {ev.missing_evidence.length > 0 && (
          <div className="rounded-md border border-warn/50 bg-warn/5 px-3 py-2 text-sm">
            <p className="mb-1 font-medium">Missing evidence</p>
            <ul className="list-disc pl-5 text-xs">{ev.missing_evidence.map((m) => <li key={m}>{m}</li>)}</ul>
          </div>
        )}
      </div>
    </Panel>
  );
}

export function Coefficients({ pid, evidenceId }: { pid: string; evidenceId: string }) {
  const q = useQuery({
    queryKey: ["coefficients", pid, evidenceId],
    queryFn: () => get<{ python: boolean; terms: CoefficientRow[] }>(p(pid, `/evidence/${evidenceId}/coefficients`)),
    staleTime: Infinity,
  });
  if (!q.data) return <ErrorText error={q.error} />;
  if (!q.data.terms.length) return null;
  const py = q.data.python;
  const ratio = (row: CoefficientRow, key: "estimate_ratio" | "std_error_ratio") => {
    const both = row.r_estimate !== undefined && row.python_estimate !== undefined;
    const value = row[key];
    return <TD className={cn("text-right font-mono text-xs tabular-nums", TONE[both ? closeness(value ?? Infinity) : "fail"])}>{!both ? "missing" : value === null ? "∞" : num(value)}</TD>;
  };
  return (
    <Panel title="Coefficients">
      <Table>
        <THead>
          <TR>
            <TH>Term</TH>
            <TH className="text-right">R estimate</TH>
            {py && <TH className="text-right">Python estimate</TH>}
            {py && <TH className="text-right">Ratio</TH>}
            <TH className="text-right">R std error</TH>
            {py && <TH className="text-right">Python std error</TH>}
            {py && <TH className="text-right">Ratio</TH>}
          </TR>
        </THead>
        <TBody>
          {q.data.terms.map((t) => (
            <TR key={t.term}>
              <TD className="font-mono text-xs">{t.term}</TD>
              <TD className="text-right font-mono text-xs tabular-nums">{num(t.r_estimate)}</TD>
              {py && <TD className="text-right font-mono text-xs tabular-nums">{num(t.python_estimate)}</TD>}
              {py && ratio(t, "estimate_ratio")}
              <TD className="text-right font-mono text-xs tabular-nums">{num(t.r_std_error)}</TD>
              {py && <TD className="text-right font-mono text-xs tabular-nums">{num(t.python_std_error)}</TD>}
              {py && ratio(t, "std_error_ratio")}
            </TR>
          ))}
        </TBody>
      </Table>
    </Panel>
  );
}
