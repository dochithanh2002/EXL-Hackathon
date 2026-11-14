import { useQuery } from "@tanstack/react-query";
import type * as React from "react";
import { Download } from "lucide-react";
import { Details, ErrorText, Mono, Muted, Panel, Prose, RunLink } from "@/components/common";
import { LinkButton } from "@/components/ui/button";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { type Bundle, get, p } from "@/lib/api";
import { short } from "@/lib/format";
import type { StepProps } from "@/pages/Project";

const ORDER = ["model.py", "verify.py", "README.md", "report.md", "validation.json", "spec.json", "baseline.R"];

export function Deliverables({ project }: StepProps) {
  const s = project.state;
  const bundle = useQuery({
    queryKey: ["bundle", project.id, s.bundle_id],
    queryFn: () => get<Bundle>(p(project.id, "/deliverables")),
    enabled: !!s.bundle_id,
    staleTime: Infinity,
  });
  const readme = useQuery({
    queryKey: ["readme", project.id, s.bundle_id],
    queryFn: () => get<string>(p(project.id, "/deliverables/files/README.md")),
    enabled: !!s.bundle_id,
    staleTime: Infinity,
  });
  if (!s.bundle_id) return s.status === "working" ? null : <Muted>The package is built after the report.</Muted>;
  if (!bundle.data) return <ErrorText error={bundle.error} />;
  const b = bundle.data;
  const files = Object.entries(b.manifest.files).sort(([a], [c]) => rank(a) - rank(c) || a.localeCompare(c));
  return (
    <>
      <Panel
        title="Code & reproducibility package"
        actions={
          <>
            <RunLink pid={project.id} runId={b.mlflow_run_id} evidenceId={s.bundle_id} />
            <LinkButton size="sm" href={p(project.id, "/deliverables/bundle.zip")}><Download />Package (.zip)</LinkButton>
          </>
        }
      >
        <div className="flex flex-col gap-4">
          <Table>
            <THead>
              <TR><TH>File</TH></TR>
            </THead>
            <TBody>
              {files.map(([name]) => (
                <TR key={name}>
                  <TD><a className="font-mono text-xs underline-offset-4 hover:underline" href={p(project.id, `/deliverables/files/${name}`)}>{name}</a></TD>
                </TR>
              ))}
            </TBody>
          </Table>
          <Details rows={[
            ["Package SHA-256", <Mono>{b.archive_sha256}</Mono>],
            ["Environment", <Mono>{short(b.manifest.environment_version)}</Mono>],
            ["Specification hash", <Mono>{short(b.manifest.spec_hash)}</Mono>],
            ...files.map(([name, sha]): [string, React.ReactNode] => [name, <Mono>{short(sha, 16)}</Mono>]),
          ]} />
        </div>
      </Panel>
      {readme.data && (
        <Panel>
          <Prose markdown={readme.data} />
        </Panel>
      )}
    </>
  );
}

function rank(name: string): number {
  const i = ORDER.indexOf(name);
  return i < 0 ? ORDER.length + (name.includes("/") ? 1 : 0) : i;
}
