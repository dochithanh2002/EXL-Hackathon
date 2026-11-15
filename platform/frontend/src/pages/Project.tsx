import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, LoaderCircle, Minus, Repeat, X } from "lucide-react";
import type * as React from "react";
import { useEffect, useRef, useState } from "react";
import { ErrorText, Prose, RunLink } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/input";
import { type ChatMessage, CONDITIONAL_STEP, get, p, post, type Project, RETRYABLE, type Step, STEPS } from "@/lib/api";
import { statusText, statusVariant, TITLE } from "@/lib/format";
import { type ActivityItem, rev, useJobEvents, useProject } from "@/lib/hooks";
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
  const name = project.data?.name;
  const waiting = project.data?.state.status === "awaiting_ds";
  // The tab says when a project left running in the background is waiting on the DS.
  useEffect(() => {
    if (name) document.title = `${waiting ? "● " : ""}${name} · ${TITLE}`;
    return () => void (document.title = TITLE);
  }, [name, waiting]);
  if (project.error) return <div className="p-6"><ErrorText error={project.error} /></div>;
  if (!project.data) return null;
  const pr = project.data;
  const s = pr.state;
  const shown = (step && step >= 1 && step <= s.step ? step : s.step) as Step;
  const r = rev(pr);
  const agent = s.step === shown && !!pr.running_job && pr.running_job.kind in AGENT_JOBS;
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
        <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
          <div className="min-h-0 min-w-0 flex-1 overflow-y-auto p-4 lg:p-6">
            <div className="mx-auto flex max-w-6xl flex-col gap-5">
              {s.step === shown && <JobBar project={pr} />}
              {views[shown]}
            </div>
          </div>
          {(chat || agent) && <Chat project={pr} rev={r} history={chat} />}
        </div>
      </div>
    </div>
  );
}

function Stepper({ project, shown }: { project: Project; shown: Step }) {
  const s = project.state;
  // Discrepancy Analysis is a loop off step 4: entered only when parity checks fail or the DS rejects the migration
  const looped = s.step === CONDITIONAL_STEP || s.repairs > 0 || s.diagnosis_id !== null;
  const steps = Object.keys(STEPS).map(Number) as Step[];
  return (
    <nav className="w-14 shrink-0 overflow-y-auto border-r bg-card p-2 lg:w-64 lg:p-3">
      <ol className="flex flex-col">
        {steps.map((n, i) => {
          const loop = n === CONDITIONAL_STEP;
          const skipped = loop && !looped && s.step > n;
          const reached = n <= s.step && !skipped;
          const done = !skipped && (n < s.step || (n === s.step && s.status === "done"));
          const current = n === s.step && !done;
          const tone = statusVariant(s);
          const item = (
            <span
              className={cn("flex items-start gap-3 rounded-md px-2 py-2 text-sm leading-6", reached && "hover:bg-muted/60", shown === n && "bg-muted font-medium", !reached && "text-muted-foreground/70")}
              title={loop ? "Runs only when parity checks fail or the migration is rejected, then returns to step 4" : STEPS[n]}
            >
              <span
                className={cn(
                  "relative z-10 grid size-6 shrink-0 place-items-center rounded-full border bg-card text-xs font-medium tabular-nums [&_svg]:size-3.5",
                  loop && !reached && "border-dashed",
                  done && "border-pass/40 bg-[color-mix(in_oklch,var(--pass)_12%,var(--card))] text-pass",
                  current && tone === "secondary" && "border-primary text-primary",
                  current && tone === "warn" && "border-warn bg-warn text-white",
                  current && tone === "pass" && "border-pass bg-pass text-white",
                  current && tone === "fail" && "border-destructive bg-destructive text-white",
                )}
              >
                {skipped ? <Minus /> : done ? <Check /> : current && s.status === "working" ? <LoaderCircle className="animate-spin" /> : loop ? <Repeat /> : n < CONDITIONAL_STEP ? n : n - 1}
              </span>
              <span className="hidden min-w-0 lg:block">
                {STEPS[n]}
                {current && <span className={cn("block text-xs leading-4 font-normal", tone === "fail" ? "text-destructive" : tone === "warn" ? "text-warn-foreground" : tone === "pass" ? "text-pass" : "text-muted-foreground")}>{statusText(s)}</span>}
              </span>
            </span>
          );
          return (
            <li key={n} className="relative">
              {i < steps.length - 1 && <span className={cn("absolute top-8 -bottom-2 left-5 w-px -translate-x-1/2", done ? "bg-pass/40" : "bg-border")} />}
              {reached ? <a href={href.project(project.id, n)} aria-current={shown === n ? "step" : undefined}>{item}</a> : item}
            </li>
          );
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
    <div className="flex h-14 shrink-0 items-center gap-3 border-b bg-card px-6">
      <h1 className="font-semibold">{project.name}</h1>
      <a href={href.project(project.id)}><Badge variant={statusVariant(s)}>{STEPS[s.step]} · {statusText(s)}</Badge></a>
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
  const cancel = useMutation({ mutationFn: () => post(p(project.id, "/cancel")), onSuccess: () => qc.invalidateQueries({ queryKey: ["project", project.id] }) });
  const s = project.state;
  if (!job) {
    if (s.status === "awaiting_ds" && s.detail && (s.reason === "step_failed" || s.reason === "interrupted" || s.reason === "cancelled" || s.reason === "retry_limit" || s.reason === "spec_change" || s.reason === "unsupported")) {
      return (
        <div className={cn("rounded-lg border bg-card px-4 py-3 text-sm whitespace-pre-wrap shadow-xs", (s.reason === "step_failed" || s.reason === "unsupported") && "border-destructive/40 bg-[color-mix(in_oklch,var(--destructive)_5%,var(--card))]")}>
          {s.detail}
        </div>
      );
    }
    return null;
  }
  return (
    <div className="rounded-lg border border-primary/30 bg-card px-4 py-3 shadow-xs">
      <div className="flex items-center gap-3 text-sm">
        <LoaderCircle className="size-4 animate-spin text-primary" />
        <span className="font-medium">{agent ? `${agent} is working` : JOB_LABELS[job.kind] ?? job.kind}</span>
        <Button size="sm" variant="outline" className="ml-auto" disabled={cancel.isPending} onClick={() => cancel.mutate()}>Cancel</Button>
      </div>
      <ErrorText error={cancel.error} />
    </div>
  );
}

function Chat({ project, rev, history }: StepProps & { history: boolean }) {
  const qc = useQueryClient();
  const s = project.state;
  const messages = useQuery({ queryKey: ["chat", project.id, rev], queryFn: () => get<ChatMessage[]>(p(project.id, "/chat")), enabled: history });
  const job = project.running_job;
  const agent = job ? AGENT_JOBS[job.kind] : undefined;
  const live = useJobEvents(agent ? job?.id : undefined);
  const [text, setText] = useState("");
  const send = useMutation({
    mutationFn: () => post(p(project.id, "/chat"), { text }),
    onSuccess: () => {
      setText("");
      qc.invalidateQueries({ queryKey: ["project", project.id] });
    },
  });
  const last = useRef<HTMLDivElement>(null);
  const end = useRef<HTMLDivElement>(null);
  // A finished message is read from its top; a streaming one is followed at its tail.
  useEffect(() => {
    last.current?.scrollIntoView({ block: "start" });
  }, [messages.data?.length]);
  useEffect(() => {
    if (agent) end.current?.scrollIntoView({ block: "end" });
  }, [agent, live]);
  const shownMessages = history ? messages.data ?? [] : [];
  const open = s.status === "awaiting_ds" && [2, 3, 5].includes(s.step);
  return (
    <aside className="flex h-72 shrink-0 flex-col border-t bg-card lg:h-auto lg:w-[22rem] lg:border-t-0 lg:border-l 2xl:w-[26rem]">
      <div className="flex h-10 shrink-0 items-center border-b px-4 text-sm font-medium">{agent ?? "Analyst"}</div>
      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {shownMessages.map((m, i) => (
          <div key={m.seq} ref={i === shownMessages.length - 1 ? last : undefined}><Bubble author={m.author} text={m.text} /></div>
        ))}
        {agent && (
          <>
            {live.map((i) => (i.kind === "text" ? <Bubble key={`t${i.id}`} author="analyst" text={i.text} /> : <ToolRow key={i.id} tool={i} />))}
            {!live.length && <Bubble author="analyst" text="…" pending />}
          </>
        )}
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
          placeholder={open ? "Write to the Analyst" : agent ? `The ${agent} is working…` : ""}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) e.currentTarget.form?.requestSubmit();
          }}
        />
        <div className="flex items-center justify-between">
          <ErrorText error={send.error} />
          <Button size="sm" type="submit" className="ml-auto" disabled={!open || !text.trim() || send.isPending}>Send<kbd className="font-sans text-xs opacity-70">⌘↵</kbd></Button>
        </div>
      </form>
    </aside>
  );
}

function ToolRow({ tool: t }: { tool: Extract<ActivityItem, { kind: "tool" }> }) {
  const Icon = t.error ? X : t.done ? Check : LoaderCircle;
  return (
    <div className="flex items-center gap-2 px-1 text-xs text-muted-foreground">
      <Icon className={cn("size-3 shrink-0", t.error ? "text-destructive" : t.done ? "text-pass" : "animate-spin")} />
      <span className="truncate font-mono">{t.name}</span>
    </div>
  );
}

function Bubble({ author, text, pending }: { author: ChatMessage["author"]; text: string; pending?: boolean }) {
  if (author === "platform") return <div className="rounded-md border border-dashed px-3 py-2 text-xs whitespace-pre-wrap text-muted-foreground">{text}</div>;
  return (
    <div className={cn("flex", author === "ds" && "justify-end")}>
      <div className={cn("min-w-0 rounded-lg px-3 py-2 text-sm", author === "ds" ? "max-w-[90%] bg-primary whitespace-pre-wrap text-primary-foreground" : "w-full bg-muted", pending && "opacity-70")}>
        {author === "ds" ? text : <Prose markdown={text} />}
      </div>
    </div>
  );
}
