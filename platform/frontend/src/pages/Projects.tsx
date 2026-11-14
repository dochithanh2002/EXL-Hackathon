import { useMutation, useQuery } from "@tanstack/react-query";
import { Plus } from "lucide-react";
import { useState } from "react";
import { ErrorText, Panel } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input, Label } from "@/components/ui/input";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { get, post, type ProjectState, type ProjectSummary, STEPS } from "@/lib/api";
import { progress, statusText, statusVariant, when } from "@/lib/format";
import { href, navigate } from "@/lib/route";
import { cn } from "@/lib/utils";

const BAR = { pass: "bg-pass", warn: "bg-warn", fail: "bg-destructive", secondary: "bg-primary" };

function Progress({ state }: { state: ProjectState }) {
  const { done, total } = progress(state);
  return (
    <span className="flex w-28 gap-0.5" role="progressbar" aria-valuenow={done} aria-valuemax={total}>
      {Array.from({ length: total }, (_, i) => (
        <span key={i} className={cn("h-1.5 flex-1 rounded-full bg-border", i < done && "bg-pass", i === done && BAR[statusVariant(state)], i === done && state.status === "working" && "animate-pulse")} />
      ))}
    </span>
  );
}

export function Projects() {
  const projects = useQuery({ queryKey: ["projects"], queryFn: () => get<ProjectSummary[]>("/api/projects"), refetchInterval: 5000 });
  const [opened, setOpened] = useState(false);
  const empty = projects.data?.length === 0;
  const creating = opened || empty;
  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">Projects</h1>
        {!creating && <Button onClick={() => setOpened(true)}><Plus />New project</Button>}
      </div>
      {creating && <NewProject onCancel={empty ? undefined : () => setOpened(false)} />}
      <ErrorText error={projects.error} />
      {projects.data && projects.data.length > 0 && (
        <Card className="px-3 py-1">
          <Table>
            <THead>
              <TR className="hover:bg-transparent">
                <TH>Name</TH>
                <TH>Step</TH>
                <TH>Status</TH>
                <TH>Created</TH>
              </TR>
            </THead>
            <TBody>
              {projects.data.map((p) => (
                <TR key={p.id} className="cursor-pointer" onClick={() => navigate(href.project(p.id))}>
                  <TD className="py-3 font-medium"><a href={href.project(p.id)}>{p.name}</a></TD>
                  <TD><span className="flex items-center gap-3"><Progress state={p.state} />{STEPS[p.state.step]}</span></TD>
                  <TD><Badge variant={statusVariant(p.state)}>{statusText(p.state)}</Badge></TD>
                  <TD className="text-muted-foreground">{when(p.created_at)}</TD>
                </TR>
              ))}
            </TBody>
          </Table>
        </Card>
      )}
    </div>
  );
}

function NewProject({ onCancel }: { onCancel?: () => void }) {
  const [csvs, setCsvs] = useState<File[]>([]);
  const [dataset, setDataset] = useState(0);
  const create = useMutation({
    mutationFn: (form: FormData) => post<{ id: string }>("/api/projects", form),
    onSuccess: (r) => navigate(href.project(r.id)),
  });
  return (
    <Panel title="New project">
      <form
        className="grid gap-4 sm:grid-cols-2"
        onSubmit={(e) => {
          e.preventDefault();
          const form = new FormData(e.currentTarget);
          form.delete("csv");
          form.delete("dataset_choice");
          csvs.forEach((f, i) => form.append(i === dataset ? "dataset" : "data", f));
          // an empty file input still sends a nameless part
          const documents = form.getAll("documents").filter((f) => f instanceof File && f.name);
          form.delete("documents");
          documents.forEach((f) => form.append("documents", f));
          create.mutate(form);
        }}
      >
        <Label className="sm:col-span-2">
          Name
          <Input name="name" required maxLength={200} />
        </Label>
        <div className="flex flex-col gap-1.5">
          <Label>
            Data (.csv)
            <Input
              name="csv"
              type="file"
              accept=".csv"
              multiple
              required
              onChange={(e) => {
                const files = Array.from(e.currentTarget.files ?? []);
                setCsvs(files);
                // the model's dataset is usually the largest file
                setDataset(files.reduce((best, f, i) => (f.size > files[best].size ? i : best), 0));
              }}
            />
          </Label>
          {csvs.length > 1 && (
            <fieldset className="flex flex-col gap-1 text-sm">
              <legend className="mb-1 font-medium">Model dataset</legend>
              {csvs.map((f, i) => (
                <label key={f.name} className="flex items-center gap-2">
                  <input type="radio" name="dataset_choice" checked={i === dataset} onChange={() => setDataset(i)} />
                  {f.name}
                </label>
              ))}
            </fieldset>
          )}
        </div>
        <Label>
          R code
          <Input name="r_code" type="file" accept=".r,.R,.rmd,.Rmd,.qmd" multiple required />
        </Label>
        <Label>
          <span>Documents <span className="font-normal text-muted-foreground">optional</span></span>
          <Input name="documents" type="file" accept=".md,.txt,.pdf,.docx" multiple />
        </Label>
        <div className="flex items-end justify-end gap-2">
          {onCancel && <Button type="button" variant="ghost" onClick={onCancel}>Cancel</Button>}
          <Button type="submit" disabled={create.isPending}>{create.isPending ? "Uploading…" : "Create"}</Button>
        </div>
        <ErrorText error={create.error} />
      </form>
    </Panel>
  );
}
