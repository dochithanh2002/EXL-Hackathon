import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { get, p, type Project } from "./api";

export function useProject(pid: string) {
  return useQuery({
    queryKey: ["project", pid],
    queryFn: () => get<Project>(p(pid)),
    refetchInterval: (q) => (q.state.data && (q.state.data.state.status === "working" || q.state.data.running_job) ? 1000 : 5000),
  });
}

/** Changes whenever the project moves or a job starts or ends; other project queries key on it to refetch then. */
export function rev(project: Project | undefined): string {
  if (!project) return "";
  const s = project.state;
  return [s.step, s.status, s.reason, s.spec_version, s.drafted_version, s.validation_id, s.repairs, project.running_job?.id].join(":");
}

export function useMlflow(pid: string, enabled: boolean) {
  return useQuery({
    queryKey: ["mlflow", pid],
    queryFn: () => get<{ url: string; experiment_id: string }>(p(pid, "/mlflow")),
    staleTime: Infinity,
    retry: false,
    enabled,
    refetchInterval: (q) => q.state.status === "error" ? 5000 : false,
  });
}

type TrackingRow = { evidence_id: string | null; status: string; mlflow_run_id: string | null };

export function useTrackingRun(pid: string, evidenceId?: string) {
  const find = (rows?: TrackingRow[]) => rows?.findLast((row) =>
    row.evidence_id === evidenceId && row.status === "done" && row.mlflow_run_id)?.mlflow_run_id;
  return useQuery({
    queryKey: ["tracking", pid],
    queryFn: () => get<TrackingRow[]>(p(pid, "/tracking")),
    enabled: Boolean(evidenceId),
    select: find,
    refetchInterval: (q) => find(q.state.data) ? false : 5000,
  }).data;
}

export type ActivityItem =
  | { kind: "text"; id: number; text: string }
  | { kind: "tool"; id: string; name: string; done: boolean; error: boolean };

/** The live events of a running agent job, in the order they happened: reply text and tool calls interleaved. */
export function useJobEvents(jid: string | undefined): ActivityItem[] {
  const [items, setItems] = useState<ActivityItem[]>([]);
  useEffect(() => {
    setItems([]);
    if (!jid) return;
    const es = new EventSource(`/api/jobs/${jid}/events`);
    es.onmessage = (m) => {
      const e = JSON.parse(m.data);
      setItems((prev) => {
        if (e.type === "text") {
          const last = prev[prev.length - 1];
          if (last?.kind === "text") return [...prev.slice(0, -1), { ...last, text: last.text + e.delta }];
          return [...prev, { kind: "text", id: prev.length, text: e.delta }];
        }
        if (e.type === "tool_start") return [...prev, { kind: "tool", id: e.id, name: e.name, done: false, error: false }];
        if (e.type === "tool_end")
          return prev.map((t) => (t.kind === "tool" && t.id === e.id ? { ...t, done: true, error: !!e.is_error } : t));
        return prev;
      });
    };
    es.onerror = () => es.close();                // the job ended (404 or closed stream); polling takes over
    return () => es.close();
  }, [jid]);
  return items;
}
