import { useQuery } from "@tanstack/react-query";
import { Download } from "lucide-react";
import { ErrorText, Muted, Panel, Prose, RunLink } from "@/components/common";
import { LinkButton } from "@/components/ui/button";
import { type Evidence, get, p } from "@/lib/api";
import type { StepProps } from "@/pages/Project";

export function Report({ project }: StepProps) {
  const s = project.state;
  const report = useQuery({
    queryKey: ["report", project.id, s.report_id],
    queryFn: () => get<Evidence<{ markdown: string; mlflow_run_id: string; model: string }>>(p(project.id, "/report")),
    enabled: !!s.report_id,
    staleTime: Infinity,
  });
  if (!s.report_id) return s.status === "working" ? null : <Muted>The report is written after the migration is accepted.</Muted>;
  if (!report.data) return <ErrorText error={report.error} />;
  const md = report.data.body.markdown;
  const download = `data:text/markdown;charset=utf-8,${encodeURIComponent(md)}`;
  return (
    <Panel
      actions={
        <>
          <RunLink pid={project.id} runId={report.data.body.mlflow_run_id} />
          <LinkButton size="sm" variant="outline" href={download} download="report.md"><Download />report.md</LinkButton>
        </>
      }
    >
      <Prose markdown={md} />
    </Panel>
  );
}
