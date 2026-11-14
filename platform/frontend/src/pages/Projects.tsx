import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { ErrorText, Panel } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { get, post, type ProjectSummary, STEPS } from "@/lib/api";
import { statusText, statusVariant, when } from "@/lib/format";
import { href, navigate } from "@/lib/route";

export function Projects() {
  const projects = useQuery({ queryKey: ["projects"], queryFn: () => get<ProjectSummary[]>("/api/projects"), refetchInterval: 5000 });
  const [creating, setCreating] = useState(false);
  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">Projects</h1>
        {!creating && <Button onClick={() => setCreating(true)}>New project</Button>}
      </div>
      {creating && <NewProject onCancel={() => setCreating(false)} />}
      <ErrorText error={projects.error} />
      {projects.data && projects.data.length > 0 && (
        <Table>
          <THead>
            <TR>
              <TH>Name</TH>
              <TH>Step</TH>
              <TH>Status</TH>
              <TH>Created</TH>
            </TR>
          </THead>
          <TBody>
            {projects.data.map((p) => (
              <TR key={p.id} className="cursor-pointer hover:bg-muted/50" onClick={() => navigate(href.project(p.id))}>
                <TD className="font-medium">{p.name}</TD>
                <TD>{STEPS[p.state.step]}</TD>
                <TD><Badge variant={statusVariant(p.state)}>{statusText(p.state)}</Badge></TD>
                <TD className="text-muted-foreground">{when(p.created_at)}</TD>
              </TR>
            ))}
          </TBody>
        </Table>
      )}
    </div>
  );
}

function NewProject({ onCancel }: { onCancel: () => void }) {
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
          Documents
          <Input name="documents" type="file" accept=".md,.txt,.pdf,.docx" multiple />
        </Label>
        <div className="flex items-end justify-end gap-2">
          <Button type="button" variant="ghost" onClick={onCancel}>Cancel</Button>
          <Button type="submit" disabled={create.isPending}>{create.isPending ? "Uploading…" : "Create"}</Button>
        </div>
        <ErrorText error={create.error} />
      </form>
    </Panel>
  );
}
