import { diffLines } from "diff";
import { Check, ChevronDown, Copy, ExternalLink } from "lucide-react";
import type * as React from "react";
import { useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useMlflow, useTrackingRun } from "@/lib/hooks";
import { cn } from "@/lib/utils";

export function Panel({ title, actions, children, className }: { title?: React.ReactNode; actions?: React.ReactNode; children: React.ReactNode; className?: string }) {
  const [open, setOpen] = useState(true);
  return (
    <Card className={className}>
      {(title || actions) && (
        <CardHeader className={cn(!open && "pb-4")}>
          <CardTitle>
            <button className="flex items-center gap-2 text-left" aria-expanded={open} onClick={() => setOpen(!open)}>
              <ChevronDown className={cn("size-4 shrink-0 text-muted-foreground transition-transform", !open && "-rotate-90")} />
              {title}
            </button>
          </CardTitle>
          {open && actions && <div className="flex items-center gap-2">{actions}</div>}
        </CardHeader>
      )}
      {open && <CardContent>{children}</CardContent>}
    </Card>
  );
}

export function Fields({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[max-content_1fr] gap-x-6 gap-y-1.5 text-sm">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-muted-foreground">{k}</dt>
          <dd className="min-w-0 break-words">{v ?? "—"}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Mono({ children, className }: { children: React.ReactNode; className?: string }) {
  return <code className={cn("rounded bg-muted px-1 py-0.5 font-mono text-[0.85em]", className)}>{children}</code>;
}

export function Code({ code, className }: { code: string; className?: string }) {
  const lines = code.replace(/\n$/, "").split("\n");
  const [copied, setCopied] = useState(false);
  const copy = () => navigator.clipboard.writeText(code).then(() => {
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  });
  return (
    <div className="group relative">
      <button
        aria-label="Copy"
        onClick={copy}
        className="absolute top-2 right-2 z-10 grid size-7 place-items-center rounded-md border bg-card text-muted-foreground opacity-0 shadow-xs transition-opacity group-hover:opacity-100 hover:text-foreground focus-visible:opacity-100"
      >
        {copied ? <Check className="size-3.5 text-pass" /> : <Copy className="size-3.5" />}
      </button>
      <pre className={cn("max-h-[36rem] overflow-auto rounded-md border bg-muted/50 py-2 font-mono text-xs leading-5", className)}>
        {lines.map((l, i) => (
          <div key={i} className="flex">
            <span className="w-10 shrink-0 select-none pr-3 text-right text-muted-foreground/70">{i + 1}</span>
            <span className="whitespace-pre pr-4">{l || " "}</span>
          </div>
        ))}
      </pre>
    </div>
  );
}

/** The decision a step is waiting on, kept in view at the bottom of the step while its evidence scrolls. */
export function ActionBar({ tone, children }: { tone?: "pass" | "warn"; children: React.ReactNode }) {
  return (
    <div className={cn("sticky -bottom-6 z-10 flex flex-col gap-3 rounded-lg border bg-card px-4 py-3 text-sm shadow-lg", tone === "pass" && "border-pass/50", tone === "warn" && "border-warn/60")}>
      {children}
    </div>
  );
}

export function Diff({ before, after }: { before: string; after: string }) {
  const parts = diffLines(before, after);
  return (
    <pre className="max-h-[36rem] overflow-auto rounded-md border bg-muted/50 py-2 font-mono text-xs leading-5">
      {parts.flatMap((part, i) =>
        part.value.replace(/\n$/, "").split("\n").map((l, j) => (
          <div
            key={`${i}-${j}`}
            className={cn("whitespace-pre px-3", part.added && "bg-pass/15", part.removed && "bg-destructive/10 text-muted-foreground")}
          >
            {part.added ? "+ " : part.removed ? "- " : "  "}
            {l}
          </div>
        )),
      )}
    </pre>
  );
}

/** Markdown; relative image paths resolve against `assets`, the URL the document's files are served from. */
export function Prose({ markdown, assets }: { markdown: string; assets?: string }) {
  return (
    <div
      className={cn(
        "max-w-none text-sm leading-6 break-words [&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
        "[&_h1]:mb-4 [&_h1]:text-xl [&_h1]:font-semibold [&_h2]:mt-8 [&_h2]:mb-3 [&_h2]:text-base [&_h2]:font-semibold",
        "[&_p]:my-3 [&_ul]:my-3 [&_ul]:list-disc [&_ul]:pl-6 [&_ol]:my-3 [&_ol]:list-decimal [&_ol]:pl-6 [&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:font-mono [&_code]:text-[0.85em]",
        "[&_pre]:overflow-x-auto [&_pre]:rounded-md [&_pre]:bg-muted [&_pre]:p-3 [&_pre_code]:bg-transparent [&_pre_code]:p-0",
        "[&_h3]:mt-6 [&_h3]:mb-2 [&_h3]:font-semibold [&_h4]:mt-4 [&_h4]:mb-2 [&_h4]:text-xs [&_h4]:font-medium [&_h4]:text-muted-foreground [&_table]:my-3 [&_table]:block [&_table]:w-full [&_table]:overflow-x-auto [&_table]:text-xs [&_th]:border-b [&_th]:px-2 [&_th]:py-1.5 [&_th]:text-left [&_th]:font-medium [&_th]:text-muted-foreground [&_td]:border-b [&_td]:px-2 [&_td]:py-1",
      )}
    >
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          img: ({ src, alt }) => (
            <img src={assets && typeof src === "string" && !/^[a-z]+:|^\//i.test(src) ? `${assets}/${src}` : src} alt={alt} className="my-3 w-full max-w-3xl rounded-md border" loading="lazy" />
          ),
          a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer" className="underline underline-offset-2 hover:text-foreground">{children}</a>,
        }}
      >
        {markdown}
      </Markdown>
    </div>
  );
}

/** The project's experiment or a run, whose ID may become available after the evidence was loaded. */
export function RunLink({ pid, runId, evidenceId, children = "MLflow" }: {
  pid: string; runId?: string | null; evidenceId?: string; children?: React.ReactNode;
}) {
  const tracked = useTrackingRun(pid, runId ? undefined : evidenceId);
  const resolved = runId ?? tracked;
  const ready = Boolean(resolved) || (runId === undefined && !evidenceId);
  const m = useMlflow(pid, ready);
  if (!m.data || !ready) return null;
  const url = `${m.data.url}/#/experiments/${m.data.experiment_id}${resolved ? `/runs/${resolved}` : ""}`;
  return (
    <a href={url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
      {children}
      <ExternalLink className="size-3" />
    </a>
  );
}

/** Technical ids and hashes, folded away under the content they identify. */
export function Details({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <details className="text-sm">
      <summary className="cursor-pointer text-xs text-muted-foreground select-none hover:text-foreground">Details</summary>
      <div className="mt-2"><Fields rows={rows} /></div>
    </details>
  );
}

export function ErrorText({ error }: { error: unknown }) {
  if (!error) return null;
  return <p className="text-sm text-destructive">{error instanceof Error ? error.message : String(error)}</p>;
}

export function Muted({ children }: { children: React.ReactNode }) {
  return <p className="text-sm text-muted-foreground">{children}</p>;
}
