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

export function useMlflow(pid: string) {
  return useQuery({
    queryKey: ["mlflow", pid],
    queryFn: () => get<{ url: string; experiment_id: string }>(p(pid, "/mlflow")),
    staleTime: Infinity,
    retry: false,
  });
}

export interface Activity {
  text: string;
  tools: { id: string; name: string; done: boolean; error: boolean }[];
}

/** The live events of a running agent job: streamed reply text and tool calls. */
export function useJobEvents(jid: string | undefined): Activity {
  const [a, setA] = useState<Activity>({ text: "", tools: [] });
  useEffect(() => {
    setA({ text: "", tools: [] });
    if (!jid) return;
    const es = new EventSource(`/api/jobs/${jid}/events`);
    es.onmessage = (m) => {
      const e = JSON.parse(m.data);
      setA((prev) => {
        if (e.type === "text") return { ...prev, text: prev.text + e.delta };
        if (e.type === "tool_start") return { ...prev, tools: [...prev.tools, { id: e.id, name: e.name, done: false, error: false }] };
        if (e.type === "tool_end")
          return { ...prev, tools: prev.tools.map((t) => (t.id === e.id ? { ...t, done: true, error: !!e.is_error } : t)) };
        return prev;
      });
    };
    es.onerror = () => es.close();                // the job ended (404 or closed stream); polling takes over
    return () => es.close();
  }, [jid]);
  return a;
}
