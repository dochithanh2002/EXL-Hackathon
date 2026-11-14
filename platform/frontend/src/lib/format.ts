import type { ProjectState } from "./api";

export function num(x: unknown): string {
  if (x === null || x === undefined) return "—";
  if (typeof x !== "number") return String(x);
  if (Number.isInteger(x) && Math.abs(x) < 1e15) return x.toLocaleString("en-US");
  if (x === 0) return "0";
  const a = Math.abs(x);
  return a >= 1e-3 && a < 1e6 ? x.toPrecision(6).replace(/\.?0+$/, "") : x.toExponential(3);
}

export const short = (h: string | null | undefined, n = 12) => (h ? h.slice(0, n) : "—");

export function when(iso: string | null | undefined): string {
  return iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—";
}

const REASONS: Record<string, string> = {
  spec_drafted: "spec drafted",
  analyst_replied: "Analyst replied",
  unsupported: "unsupported model",
  candidate_passed: "awaiting approval",
  retry_limit: "retry limit reached",
  spec_change: "spec change proposed",
  step_failed: "failed",
  interrupted: "interrupted",
  cancelled: "cancelled",
};

export function statusText(s: ProjectState): string {
  if (s.status === "done") return "done";
  if (s.status === "working") return "working";
  return s.reason ? REASONS[s.reason] : "awaiting you";
}

export function statusVariant(s: ProjectState): "pass" | "fail" | "warn" | "secondary" {
  if (s.status === "done" || s.reason === "candidate_passed") return "pass";
  if (s.reason === "step_failed" || s.reason === "unsupported" || s.reason === "interrupted") return "fail";
  if (s.status === "awaiting_ds") return "warn";
  return "secondary";
}
