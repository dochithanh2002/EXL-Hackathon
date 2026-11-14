// Types mirror the control plane's Pydantic models (controlplane/workflow/machine.py, schemas/agents.py).

export type Step = 1 | 2 | 3 | 4 | 5 | 6 | 7;
export type Status = "working" | "awaiting_ds" | "done";
export type Reason =
  | "spec_drafted" | "analyst_replied" | "unsupported" | "candidate_passed" | "retry_limit" | "spec_change"
  | "step_failed" | "interrupted" | "cancelled";
export type Tier = "exact" | "approximate" | "unsupported";

export const STEPS: Record<Step, string> = {
  1: "Intake", 2: "Alignment", 3: "R baseline", 4: "Translation", 5: "Repair", 6: "Report", 7: "Deliverables",
};
export const RETRYABLE: Reason[] = ["step_failed", "interrupted", "cancelled"];
export const ROLES = ["analyst", "translator", "diagnostician", "reporter"] as const;
export type Role = (typeof ROLES)[number];

export interface Bindings { spec_version: number; baseline_id: string; candidate_hash: string; validation_id: string }
export interface InputFile { role: "dataset" | "r_code" | "document"; name: string; sha256: string }

export interface ProjectState {
  step: Step; status: Status; reason: Reason | null; detail: string; files: InputFile[]; retry_limit: number;
  drafted_version: number | null; drafted_tier: Tier | null; drafted_proven: boolean | null;
  spec_version: number | null; spec_tier: Tier | null; spec_proven: boolean | null;
  baseline_key: string | null; baseline_id: string | null; candidate_hash: string | null;
  validation_id: string | null; validation_passed: boolean | null; repairs: number; ds_comment: string | null;
  diagnosis_id: string | null; approval: Bindings | null; report_id: string | null; bundle_id: string | null;
}

export interface Job {
  id: string; project_id: string; step: Step; kind: string; spec_version: number | null;
  status: "running" | "succeeded" | "failed" | "cancelled" | "interrupted"; started_at: string;
  ended_at: string | null; detail: string;
}
export interface Decision { at: string; decision: "approved" | "rejected"; comment: string | null; bindings: Bindings; void: boolean }
export interface ProjectSummary { id: string; name: string; created_at: string; state: ProjectState }
export interface Project { id: string; name: string; state: ProjectState; running_job: Job | null; decisions: Decision[] }

export interface ChatMessage { seq: number; at: string; author: "ds" | "analyst" | "platform"; text: string; job_id: string | null; spec_version: number | null }

export interface ModelSpec {
  target_model: string; r_function: string; statsmodels_counterpart: string; support_tier: Tier;
  family: string | null; link: string | null; formula: string; response: string; offset: string | null;
  weights: string | null;
  preprocessing: { order: number; description: string; r_code: string; columns: string[] }[];
  split: { method: string; r_code: string; seed: number | null };
  kpis: string[]; tolerances: Record<string, number>; unsupported_items: string[];
  substitutions: { original: string; proposed: string; rationale: string }[];
  r_packages: string[]; baseline_r: string; baseline_objects: { model: string; train: string; test: string | null };
}
export interface SpecVersion { version: number; spec: ModelSpec; support: { tier: Tier; proven: boolean }; baseline_key: string }
export interface SpecRow { version: number; hash: string; created_at: string }

export interface Evidence<T = Record<string, any>> { id: string; kind: string; hash: string; created_at: string; body: T }
export interface EvidenceRow { id: string; kind: string; hash: string; created_at: string }

export interface Check { name: string; value: number | null; tolerance: number; passed: boolean }
export interface ValidationBody {
  passed: boolean; stage: "runtime" | "parity"; error: Record<string, any> | null; checks: Check[];
  kpis: Record<string, Record<string, { r: number | null; python: number | null }>>;
  differences: Record<string, any>; notes: string[]; candidate_id: string; candidate_hash: string; job_id: string;
  spec_version: number; baseline_id: string; mlflow_run_id: string;
}
export interface CandidateBody {
  attempt: number; diagnosis_id: string | null; hash: string; module: string; notes: string[]; changes: string;
  reply: string; model: string; mlflow_run_id: string; spec_version: number; baseline_id: string; job_id: string;
}
export interface DiagnosisBody {
  root_cause: string; evidence: string[]; repair_instructions: string[]; spec_change_needed: boolean;
  proposed_spec_change: string | null; ds_comment: string | null; failed_checks: string[]; reply: string;
  candidate_id: string; validation_id: string; spec_version: number; baseline_id: string; mlflow_run_id: string;
}
export interface BaselineBody {
  spec_version: number; job_id: string; hash: string; files: Record<string, string>; environment_version: string;
  fit: Record<string, unknown>; meta: Record<string, unknown>; mlflow_run_id: string;
}
export interface CoefficientRow {
  term: string; r_estimate?: number | null; r_std_error?: number | null; python_estimate?: number | null;
  python_std_error?: number | null; estimate_rel?: number | null; std_error_rel?: number | null;
}
export interface Bundle {
  dir: string; archive: string; archive_sha256: string; report_id: string; job_id: string; mlflow_run_id: string;
  manifest: { files: Record<string, string>; environment_version: string; spec_hash: string; dataset: Record<string, string> };
}

export interface Endpoint { id: string; name: string; base_url: string; api: string; models: string[]; has_key: boolean }
export interface RoleModel { endpoint_id: string; model: string; thinking: string }
export interface Environment { version: string; lockfiles: Record<string, string>; r: string; python: string }

export interface ProfileColumn {
  name: string; r_name: string; class: string; missing: number; distinct: number; non_finite?: number;
  min?: number; max?: number; mean?: number; sd?: number | null; levels?: { value: string | number; n: number }[];
}
export interface Profile { file: string; rows: number; columns: ProfileColumn[] }

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers = { "content-type": "application/json" };
  }
  const r = await fetch(path, init);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const j = await r.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(r.status, detail);
  }
  if (r.status === 204) return undefined as T;
  return (r.headers.get("content-type")?.includes("json") ? r.json() : r.text()) as Promise<T>;
}

export const get = <T>(path: string) => request<T>("GET", path);
export const post = <T>(path: string, body?: unknown) => request<T>("POST", path, body);
export const put = <T>(path: string, body?: unknown) => request<T>("PUT", path, body);
export const del = <T>(path: string) => request<T>("DELETE", path);

export const p = (pid: string, rest = "") => `/api/projects/${pid}${rest}`;
