import { useQueries, useQuery } from "@tanstack/react-query";
import { type CandidateBody, type DiagnosisBody, type Evidence, type EvidenceRow, get, p, type Project, type ValidationBody } from "./api";

export interface Attempt {
  candidate: Evidence<CandidateBody>;
  validation?: Evidence<ValidationBody>;
  diagnosis?: Evidence<DiagnosisBody>;
}

/** Every translation attempt under the project's current spec and baseline, oldest first. */
export function useAttempts(project: Project, rev: string): { attempts: Attempt[]; loading: boolean } {
  const s = project.state;
  const list = useQuery({ queryKey: ["evidence-list", project.id, rev], queryFn: () => get<EvidenceRow[]>(p(project.id, "/evidence")) });
  const rows = (list.data ?? []).filter((e) => ["candidate", "validation", "diagnosis"].includes(e.kind));
  const bodies = useQueries({
    queries: rows.map((e) => ({
      queryKey: ["evidence", project.id, e.id],
      queryFn: () => get<Evidence>(p(project.id, `/evidence/${e.id}`)),
      staleTime: Infinity,
    })),
  });
  const loaded = bodies.map((b) => b.data).filter((b): b is Evidence => !!b)
    .filter((e) => e.body.spec_version === s.spec_version && e.body.baseline_id === s.baseline_id);
  const attempts = new Map<string, Attempt>();
  for (const e of loaded) if (e.kind === "candidate") attempts.set(e.id, { candidate: e as Evidence<CandidateBody> });
  for (const e of loaded) {
    const a = attempts.get(e.body.candidate_id);
    if (a && e.kind === "validation") a.validation = e as Evidence<ValidationBody>;
    if (a && e.kind === "diagnosis") a.diagnosis = e as Evidence<DiagnosisBody>;
  }
  return { attempts: [...attempts.values()], loading: list.isPending || bodies.some((b) => b.isPending) };
}
