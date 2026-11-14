import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Circle, CircleDot, LoaderCircle, Minus, Repeat, Wrench } from "lucide-react";
import type * as React from "react";
import { useEffect, useRef, useState } from "react";
import { ErrorText, RunLink } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/input";
import { type ChatMessage, CONDITIONAL_STEP, get, p, post, type Project, RETRYABLE, type Step, STEPS } from "@/lib/api";
import { statusText, statusVariant } from "@/lib/format";
import { rev, useJobEvents, useProject } from "@/lib/hooks";
import { href } from "@/lib/route";
import { cn } from "@/lib/utils";
import { Alignment } from "@/steps/Alignment";
import { Baseline } from "@/steps/Baseline";
import { Deliverables } from "@/steps/Deliverables";
import { Intake } from "@/steps/Intake";
import { Repair } from "@/steps/Repair";
import { Report } from "@/steps/Report";
import { Translation } from "@/steps/Translation";

const AGENT_JOBS: Record<string, string> = { analyst: "Analyst", translation: "Translator", diagnosis: "Diagnostician", report: "Reporter" };
const JOB_LABELS: Record<string, string> = { r_baseline: "Computing the R reference results", deliverables: "Building the code & reproducibility package" };

export interface StepProps {
  project: Project;
  rev: string;
}

export function ProjectPage({ pid, step }: { pid: string; step?: number }) {
  const project = useProject(pid);
  if (project.error) return <div className="p-6"><ErrorText error={project.error} /></div>;
  if (!project.data) return null;
  const pr = project.data;
  const s = pr.state;
  const shown = (step && step >= 1 && step <= s.step ? step : s.step) as Step;
  const r = rev(pr);
  const chat = shown === 2 || (shown === 5 && s.status === "awaiting_ds") || (shown === 3 && s.step === 3 && s.status === "awaiting_ds");
  const views: Record<Step, React.ReactNode> = {
    1: <Intake project={pr} rev={r} />,
    2: <Alignment project={pr} rev={r} />,
    3: <Baseline project={pr} rev={r} />,
    4: <Translation project={pr} rev={r} />,
    5: <Repair project={pr} rev={r} />,
    6: <Report project={pr} rev={r} />,
    7: <Deliverables project={pr} rev={r} />,
  };
  return (
    <div className="flex min-h-0 flex-1">
      <Stepper project={pr} shown={shown} />
      <div className="flex min-w-0 flex-1 flex-col">
        <Header project={pr} />
        <div className="flex min-h-0 flex-1">
          <div className="min-w-0 flex-1 overflow-y-auto p-6">
            <div className="mx-auto flex max-w-6xl flex-col gap-5">
              {s.step === shown && <JobBar project={pr} />}
              {views[shown]}
            </div>
          </div>
          {chat && <Chat project={pr} rev={r} />}
        </div>
      </div>
    </div>
  );
}

function Stepper({ project, shown }: { project: Project; shown: Step }) {
  const s = project.state;
  // Discrepancy Analysis is a loop off step 4: entered only when parity checks fail or the DS rejects the migration
  const looped = s.step === CONDITIONAL_STEP || s.repairs > 0 || s.diagnosis_id !== null;
  return (
    <nav className="w-60 shrink-0 border-r p-3">
      <ol className="flex flex-col gap-0.5">
        {(Object.keys(STEPS).map(Number) as Step[]).map((n) => {
          const loop = n === CONDITIONAL_STEP;
          const skipped = loop && !looped && s.step > n;
          const reached = n <= s.step && !skipped;
          const done = !skipped && (n < s.step || (n === s.step && s.status === "done"));
          const Icon = skipped ? Minus : done ? Check : n === s.step ? (s.status === "working" ? LoaderCircle : CircleDot) : loop ? Repeat : Circle;
          const item = (
            <span
              className={cn("flex items-center gap-2 rounded-md px-2 py-1.5 text-sm", loop && "ml-4 border-l border-dashed pl-3", shown === n && "bg-muted font-medium", (!reached || skipped) && "text-muted-foreground/60")}
              title={loop ? "Runs only when parity checks fail or the migration is rejected, then returns to step 4" : undefined}
            >
              <Icon className={cn("size-4 shrink-0", n === s.step && s.status === "working" && "animate-spin", done && "text-pass")} />
              {loop ? STEPS[n] : `${n < CONDITIONAL_STEP ? n : n - 1}. ${STEPS[n]}`}
            </span>
          );
          return <li key={n}>{reached ? <a href={href.project(project.id, n)}>{item}</a> : item}</li>;
        })}
      </ol>
    </nav>
  );
}

function Header({ project }: { project: Project }) {
  const qc = useQueryClient();
  const s = project.state;
  const retry = useMutation({ mutationFn: () => post(p(project.id, "/retry")), onSuccess: () => qc.invalidateQueries({ queryKey: ["project", project.id] }) });
  return (
    <div className="flex h-14 shrink-0 items-center gap-3 border-b px-6">
      <h1 className="font-semibold">{project.name}</h1>
      <Badge variant={statusVariant(s)}>{STEPS[s.step]} · {statusText(s)}</Badge>
      {s.spec_version && <Badge variant="outline">Model Specification v{s.spec_version}{s.spec_tier ? ` · ${s.spec_tier} parity` : ""}</Badge>}
      <div className="ml-auto flex items-center gap-3">
        <RunLink pid={project.id}>MLflow experiment</RunLink>
        {s.status === "awaiting_ds" && s.reason && RETRYABLE.includes(s.reason) && (
          <Button size="sm" onClick={() => retry.mutate()} disabled={retry.isPending}>Retry step</Button>
        )}
      </div>
      <ErrorText error={retry.error} />
    </div>
  );
}

function JobBar({ project }: { project: Project }) {
  const qc = useQueryClient();
  const job = project.running_job;
  const agent = job ? AGENT_JOBS[job.kind] : undefined;
  const activity = useJobEvents(agent ? job?.id : undefined);
  const cancel = useMutation({ mutationFn: () => post(p(project.id, "/cancel")), onSuccess: () => qc.invalidateQueries({ queryKey: ["project", project.id] }) });
  const s = project.state;
  if (!job) {
    if (s.status === "awaiting_ds" && s.detail && (s.reason === "step_failed" || s.reason === "interrupted" || s.reason === "cancelled" || s.reason === "retry_limit" || s.reason === "spec_change" || s.reason === "unsupported")) {
      return (
        <div className={cn("rounded-md border px-4 py-3 text-sm whitespace-pre-wrap", s.reason === "step_failed" || s.reason === "unsupported" ? "border-destructive/40 bg-destructive/5" : "bg-muted/50")}>
          {s.detail}
        </div>
      );
    }
    return null;
  }
  return (
    <div className="rounded-md border bg-muted/40 px-4 py-3">
      <div className="flex items-center gap-3 text-sm">
        <LoaderCircle className="size-4 animate-spin" />
        <span className="font-medium">{agent ? `${agent} is working` : JOB_LABELS[job.kind] ?? job.kind}</span>
        {activity.tools.map((t) => (
          <Badge key={t.id} variant={t.error ? "fail" : t.done ? "secondary" : "outline"}>
            <Wrench className="size-3" />
            {t.name}
          </Badge>
        ))}
        <Button size="sm" variant="outline" className="ml-auto" disabled={cancel.isPending} onClick={() => cancel.mutate()}>Cancel</Button>
      </div>
      {agent && agent !== "Analyst" && activity.text && <p className="mt-2 text-sm whitespace-pre-wrap text-muted-foreground">{activity.text}</p>}
      <ErrorText error={cancel.error} />
    </div>
  );
}

function Chat({ project, rev }: StepProps) {
  const qc = useQueryClient();
  const s = project.state;
  const messages = useQuery({ queryKey: ["chat", project.id, rev], queryFn: () => get<ChatMessage[]>(p(project.id, "/chat")) });
  const job = project.running_job;
  const live = useJobEvents(job?.kind === "analyst" ? job.id : undefined);
  const [text, setText] = useState("");
  const send = useMutation({
    mutationFn: () => post(p(project.id, "/chat"), { text }),
    onSuccess: () => {
      setText("");
      qc.invalidateQueries({ queryKey: ["project", project.id] });
    },
  });
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [messages.data?.length, live.text]);
  const open = s.status === "awaiting_ds" && [2, 3, 5].includes(s.step);
  return (
    <aside className="flex w-[26rem] shrink-0 flex-col border-l">
      <div className="flex h-10 shrink-0 items-center border-b px-4 text-sm font-medium">Analyst</div>
      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {messages.data?.map((m) => <Bubble key={m.seq} author={m.author} text={m.text} />)}
        {job?.kind === "analyst" && <Bubble author="analyst" text={live.text || "…"} pending />}
        <div ref={end} />
      </div>
      <form
        className="flex flex-col gap-2 border-t p-3"
        onSubmit={(e) => {
          e.preventDefault();
          if (text.trim()) send.mutate();
        }}
      >
        <Textarea
          value={text}
          rows={3}
          disabled={!open}
          placeholder={open ? "Write to the Analyst" : job?.kind === "analyst" ? "The Analyst is replying…" : ""}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) e.currentTarget.form?.requestSubmit();
          }}
        />
        <div className="flex items-center justify-between">
          <ErrorText error={send.error} />
          <Button size="sm" type="submit" className="ml-auto" disabled={!open || !text.trim() || send.isPending}>Send</Button>
        </div>
      </form>
    </aside>
  );
}

function Bubble({ author, text, pending }: { author: ChatMessage["author"]; text: string; pending?: boolean }) {
  if (author === "platform") return <div className="rounded-md border border-dashed px-3 py-2 text-xs whitespace-pre-wrap text-muted-foreground">{text}</div>;
  return (
    <div className={cn("flex", author === "ds" && "justify-end")}>
      <div className={cn("max-w-[90%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap", author === "ds" ? "bg-primary text-primary-foreground" : "bg-muted", pending && "opacity-70")}>
        {text}
      </div>
    </div>
  );
}
