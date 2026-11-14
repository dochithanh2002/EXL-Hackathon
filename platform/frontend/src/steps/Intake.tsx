import { useQuery } from "@tanstack/react-query";
import { ErrorText, Mono, Panel } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { get, p, type Profile } from "@/lib/api";
import { num } from "@/lib/format";
import type { StepProps } from "@/pages/Project";

export function Intake({ project }: StepProps) {
  const profile = useQuery({ queryKey: ["profile", project.id], queryFn: () => get<Profile>(p(project.id, "/profile")), staleTime: Infinity });
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
                <TD>{f.role.replace("_", " ")}</TD>
                <TD><Mono>{f.sha256}</Mono></TD>
              </TR>
            ))}
          </TBody>
        </Table>
      </Panel>
      <Panel title={profile.data ? `${profile.data.file} · ${num(profile.data.rows)} rows` : "Dataset"}>
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
