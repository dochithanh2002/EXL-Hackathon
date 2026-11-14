// Types mirror the control plane's Pydantic models (controlplane/workflow/machine.py, schemas/agents.py).

export type Step = 1 | 2 | 3 | 4 | 5 | 6 | 7;
export type Status = "working" | "awaiting_ds" | "done";
export type Reason =
  | "spec_drafted" | "analyst_replied" | "unsupported" | "candidate_passed" | "retry_limit" | "spec_change"
  | "step_failed" | "interrupted" | "cancelled";
export type Tier = "exact" | "approximate" | "unsupported";

export const STEPS: Record<Step, string> = {
  1: "Data & Code", 2: "Model Specification", 3: "R Reference Results", 4: "Python Migration & Comparison",
  5: "Discrepancy Analysis", 6: "Migration Validation Report", 7: "Code & Reproducibility Package",
};
/** Step 5 runs only when parity checks fail or the DS rejects the migration, then returns to step 4. */
export const CONDITIONAL_STEP: Step = 5;
export const RETRYABLE: Reason[] = ["step_failed", "interrupted", "cancelled"];
export const ROLES = ["analyst", "translator", "diagnostician", "reporter"] as const;
export type Role = (typeof ROLES)[number];

export interface Bindings { spec_version: number; baseline_id: string; candidate_hash: string; validation_id: string }
export interface InputFile { role: "dataset" | "data" | "r_code" | "document"; name: string; sha256: string }

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

export interface Tolerance { atol: number; rtol: number }
export interface ModelSpec {
  target_model: string; r_function: string; python_implementation: string; support_tier: Tier; limitation: string | null;
  family: string | null; link: string | null; formula: string; response: string;
  response_type: string;
  event: { meaning: string; complement: string; coding: string } | null;
  grouped_binomial: { form: string; successes: string | null; trials: string } | null;
  offset: string | null;
  weights: { expression: string; type: string; normalization: string } | null;
  prediction: { kind: string; quantity: string; units: string; exposure: string | null; keys: string[]; alignment_keys: string[]; r_predict: string };
  preprocessing: { order: number; description: string; r_code: string; columns: string[] }[];
  split: { method: string; r_code: string; seed: number | null };
  assessment: {
    business_use: string; intended_population: string; sample_population: string;
    evaluation_sets: { set: string; role: string; description: string }[]; period_column: string | null;
    decision: { threshold: number; action: string } | null; criteria: Record<string, number | null>;
    missing_information: string[];
  };
  metrics: string[];
  optional_metrics: Record<string, string[]>;
  convergence: { mode: string; r_check: string | null; rationale: string | null };
  tolerances: { coefficients: Tolerance; std_errors: Tolerance; predictions: Tolerance; fit_statistics: Tolerance; metrics: Tolerance; per_metric: Record<string, Tolerance> };
  unsupported_items: string[];
  substitutions: { original: string; proposed: string; rationale: string }[];
  r_packages: string[]; baseline_r: string; baseline_objects: { model: string; train: string; test: string | null };
}
export interface SpecVersion { version: number; spec: ModelSpec; support: { tier: Tier; proven: boolean; known: boolean }; baseline_key: string }
export interface SpecRow { version: number; hash: string; created_at: string }

export interface Evidence<T = Record<string, any>> { id: string; kind: string; hash: string; created_at: string; body: T }
export interface EvidenceRow { id: string; kind: string; hash: string; created_at: string }

export interface Check {
  name: string; quantity: string; kind: "tolerance" | "count" | "flag" | "minimum"; value: number | null; tolerance: number;
  status: "passed" | "failed" | "unavailable"; passed: boolean; required: boolean; reason: string;
  atol: number | null; rtol: number | null; max_abs_diff: number | null; compared: number | null;
  missing_both: number | null; missing_one_side: number | null; nonfinite_mismatch: number | null; total: number | null;
}
export type AssessmentStatus = "assessed" | "partially assessed" | "not assessed";
export interface AssessmentAnswer { status: AssessmentStatus; needs: string[]; criterion: string | null; meets: boolean | null; results: Record<string, any> }
export interface ValidationBody {
  passed: boolean; stage: "runtime" | "parity"; error: Record<string, any> | null; checks: Check[];
  metrics: Record<string, Record<string, { r: number | null; python: number | null }>>;
  differences: Record<string, any>; notes: string[]; candidate_id: string; candidate_hash: string; job_id: string;
  assessment: { r?: Record<string, AssessmentAnswer>; python?: Record<string, AssessmentAnswer> };
  design: Record<string, any>;
  evaluation?: Evaluation; plots?: PlotRef[]; mlflow_runs?: Record<string, string>;
  spec_version: number; baseline_id: string; mlflow_run_id: string;
}
export type CheckStatus = "executed" | "partially executed" | "not executed" | "not planned";
export interface EvaluationCheck { status: CheckStatus; rationale: string; needs: string[]; notes: string[]; results: Record<string, any> }
export interface Evaluation { checks: Record<string, EvaluationCheck>; missing_evidence: string[]; primary_metric: string; notes: string[] }
export interface PlotRef { name: string; title: string; section: string }
export interface CandidateBody {
  attempt: number; diagnosis_id: string | null; hash: string; module: string; notes: string[]; limitations?: string[]; changes: string;
  reply: string; model: string; mlflow_run_id: string; spec_version: number; baseline_id: string; job_id: string;
}
export interface DiagnosisBody {
  category?: "migration_defect" | "specification" | "not_reproducible" | "model_fitness";
  root_cause: string; evidence: string[]; repair_instructions: string[]; spec_change_needed: boolean;
  assessment_findings?: string[]; limitation?: string | null; proposed_spec_change: string | null; ds_comment: string | null; failed_checks: string[]; reply: string;
  candidate_id: string; validation_id: string; spec_version: number; baseline_id: string; mlflow_run_id: string;
}
export interface BaselineBody {
  spec_version: number; job_id: string; hash: string; files: Record<string, string>; environment_version: string;
  fit: Record<string, unknown>; meta: Record<string, unknown>; mlflow_run_id: string;
}
export interface CoefficientRow {
  term: string; r_estimate?: number | null; r_std_error?: number | null; python_estimate?: number | null;
  python_std_error?: number | null; estimate_ratio?: number | null; std_error_ratio?: number | null;
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
