import { useQuery } from "@tanstack/react-query";
import type * as React from "react";
import { ErrorText, Panel } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { type CoefficientRow, get, p, type ValidationBody } from "@/lib/api";
import { num } from "@/lib/format";
import { cn } from "@/lib/utils";

/** pass: within a tenth of the tolerance; near: within it; fail: outside it, or not computed. */
export function closeness(value: number | null | undefined, tolerance: number, exact = false): "pass" | "near" | "fail" {
  if (value === null || value === undefined || Number.isNaN(value)) return "fail";
  if (exact) return value === tolerance ? "pass" : "fail";
  if (value > tolerance) return "fail";
  return value > tolerance / 10 ? "near" : "pass";
}

const TONE = { pass: "text-pass", near: "text-[oklch(0.55_0.13_75)]", fail: "text-destructive font-medium" };

export function Checks({ v, actions }: { v: ValidationBody; actions?: React.ReactNode }) {
  const passed = v.checks.filter((c) => c.passed).length;
  return (
    <Panel title={<span className="flex items-center gap-2">Validation <Badge variant={v.passed ? "pass" : "fail"}>{passed} of {v.checks.length} checks passed</Badge></span>} actions={actions}>
      {v.error && (
        <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm">
          <p className="font-medium">The candidate failed at {String(v.error.stage)}: {String(v.error.type)}</p>
          <pre className="mt-2 overflow-x-auto text-xs whitespace-pre-wrap">{String(v.error.message ?? "")}</pre>
        </div>
      )}
      {v.checks.length > 0 && (
        <Table>
          <THead>
            <TR><TH>Check</TH><TH className="text-right">Value</TH><TH className="text-right">Tolerance</TH><TH /></TR>
          </THead>
          <TBody>
            {v.checks.map((c) => {
              const exact = c.tolerance === 0;
              const tone = c.passed ? closeness(c.value, c.tolerance, exact) : "fail";
              return (
                <TR key={c.name}>
                  <TD>{c.name}</TD>
                  <TD className={cn("text-right font-mono text-xs tabular-nums", TONE[tone])}>{num(c.value)}</TD>
                  <TD className="text-right font-mono text-xs tabular-nums text-muted-foreground">{num(c.tolerance)}</TD>
                  <TD className="w-16 text-right"><Badge variant={c.passed ? "pass" : "fail"}>{c.passed ? "pass" : "fail"}</Badge></TD>
                </TR>
              );
            })}
          </TBody>
        </Table>
      )}
      {v.notes.length > 0 && (
        <ul className="mt-3 list-disc pl-5 text-xs text-muted-foreground">{v.notes.map((n) => <li key={n}>{n}</li>)}</ul>
      )}
      <Kpis v={v} />
    </Panel>
  );
}

function Kpis({ v }: { v: ValidationBody }) {
  const rows = Object.entries(v.kpis).flatMap(([set, ks]) =>
    Object.entries(ks).filter(([, x]) => x.r !== null || x.python !== null).map(([k, x]) => ({ set, k, ...x })));
  if (!rows.length) return null;
  return (
    <Table className="mt-4">
      <THead>
        <TR><TH>KPI</TH><TH>Set</TH><TH className="text-right">R</TH><TH className="text-right">Python</TH></TR>
      </THead>
      <TBody>
        {rows.map((r) => (
          <TR key={`${r.set}-${r.k}`}>
            <TD>{r.k}</TD>
            <TD>{r.set}</TD>
            <TD className="text-right font-mono text-xs tabular-nums">{num(r.r)}</TD>
            <TD className="text-right font-mono text-xs tabular-nums">{num(r.python)}</TD>
          </TR>
        ))}
      </TBody>
    </Table>
  );
}

export function Coefficients({ pid, evidenceId, tolerances }: { pid: string; evidenceId: string; tolerances?: Record<string, number> }) {
  const q = useQuery({
    queryKey: ["coefficients", pid, evidenceId],
    queryFn: () => get<{ python: boolean; terms: CoefficientRow[] }>(p(pid, `/evidence/${evidenceId}/coefficients`)),
    staleTime: Infinity,
  });
  if (!q.data) return <ErrorText error={q.error} />;
  const py = q.data.python;
  const rel = (row: CoefficientRow, key: "estimate_rel" | "std_error_rel", tol?: number) => {
    const both = row.r_estimate !== undefined && row.python_estimate !== undefined;
    const value = row[key];
    const tone = !both ? "fail" : tol === undefined ? "pass" : closeness(value ?? null, tol);
    return <TD className={cn("text-right font-mono text-xs tabular-nums", TONE[tone])}>{!both ? "missing" : value === null ? "∞" : num(value)}</TD>;
  };
  return (
    <Panel title="Coefficients">
      <Table>
        <THead>
          <TR>
            <TH>Term</TH>
            <TH className="text-right">R estimate</TH>
            {py && <TH className="text-right">Python estimate</TH>}
            {py && <TH className="text-right">Rel diff</TH>}
            <TH className="text-right">R std error</TH>
            {py && <TH className="text-right">Python std error</TH>}
            {py && <TH className="text-right">Rel diff</TH>}
          </TR>
        </THead>
        <TBody>
          {q.data.terms.map((t) => (
            <TR key={t.term}>
              <TD className="font-mono text-xs">{t.term}</TD>
              <TD className="text-right font-mono text-xs tabular-nums">{num(t.r_estimate)}</TD>
              {py && <TD className="text-right font-mono text-xs tabular-nums">{num(t.python_estimate)}</TD>}
              {py && rel(t, "estimate_rel", tolerances?.coef_rel)}
              <TD className="text-right font-mono text-xs tabular-nums">{num(t.r_std_error)}</TD>
              {py && <TD className="text-right font-mono text-xs tabular-nums">{num(t.python_std_error)}</TD>}
              {py && rel(t, "std_error_rel", tolerances?.se_rel)}
            </TR>
          ))}
        </TBody>
      </Table>
    </Panel>
  );
}
