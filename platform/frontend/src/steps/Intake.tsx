import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { ErrorText, Mono, Panel } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { get, p, type Profile } from "@/lib/api";
import { num } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { StepProps } from "@/pages/Project";

const ROLE = { dataset: "dataset", data: "other data", r_code: "R code", document: "document" };

export function Intake({ project }: StepProps) {
  const csvs = project.state.files.filter((f) => f.role === "dataset" || f.role === "data");
  const [name, setName] = useState(csvs[0]?.name);
  const profile = useQuery({
    queryKey: ["profile", project.id, name],
    queryFn: () => get<Profile>(p(project.id, `/profile?name=${encodeURIComponent(name!)}`)),
    enabled: !!name,
    staleTime: Infinity,
  });
  return (
    <>
      <Panel title="Files">
        <Table>
          <THead>
            <TR><TH>Name</TH><TH>Role</TH><TH>SHA-256</TH></TR>
          </THead>
          <TBody>
            {project.state.files.map((f) => (
              <TR key={f.name}>
                <TD className="font-medium">{f.name}</TD>
                <TD>{ROLE[f.role]}</TD>
                <TD><Mono>{f.sha256}</Mono></TD>
              </TR>
            ))}
          </TBody>
        </Table>
      </Panel>
      <Panel
        title={profile.data ? `${profile.data.file} · ${num(profile.data.rows)} rows` : name}
        actions={csvs.length > 1 && (
          <div className="flex rounded-md border p-0.5">
            {csvs.map((f) => (
              <button key={f.name} className={cn("rounded px-2 py-0.5 text-xs", f.name === name && "bg-muted font-medium")} onClick={() => setName(f.name)}>
                {f.name}
              </button>
            ))}
          </div>
        )}
      >
        <ErrorText error={profile.error} />
        {profile.isPending && <p className="text-sm text-muted-foreground">Profiling…</p>}
        {profile.data && (
          <Table>
            <THead>
              <TR><TH>Column</TH><TH>R class</TH><TH className="text-right">Missing</TH><TH className="text-right">Distinct</TH><TH className="text-right">Min</TH><TH className="text-right">Mean</TH><TH className="text-right">Max</TH><TH>Levels</TH></TR>
            </THead>
            <TBody>
              {profile.data.columns.map((c) => (
                <TR key={c.name}>
                  <TD className="font-mono text-xs">{c.name}</TD>
                  <TD>{c.class}</TD>
                  <TD className="text-right tabular-nums">{c.missing ? <Badge variant="warn">{num(c.missing)}</Badge> : 0}</TD>
                  <TD className="text-right tabular-nums">{num(c.distinct)}</TD>
                  <TD className="text-right tabular-nums">{num(c.min)}</TD>
                  <TD className="text-right tabular-nums">{num(c.mean)}</TD>
                  <TD className="text-right tabular-nums">{num(c.max)}</TD>
                  <TD className="max-w-72 truncate text-xs text-muted-foreground">{c.levels?.map((l) => `${l.value} (${num(l.n)})`).join(", ")}</TD>
                </TR>
              ))}
            </TBody>
          </Table>
        )}
      </Panel>
    </>
  );
}
