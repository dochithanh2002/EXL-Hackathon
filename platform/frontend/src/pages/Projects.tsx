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
                <TD>{p.state.step}. {STEPS[p.state.step]}</TD>
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
          for (const field of ["data", "documents"]) {     // an empty file input still sends a nameless part
            const files = form.getAll(field).filter((f) => f instanceof File && f.name);
            form.delete(field);
            files.forEach((f) => form.append(field, f));
          }
          create.mutate(form);
        }}
      >
        <Label className="sm:col-span-2">
          Name
          <Input name="name" required maxLength={200} />
        </Label>
        <Label>
          Model dataset (.csv)
          <Input name="dataset" type="file" accept=".csv" required />
        </Label>
        <Label>
          Other data (.csv)
          <Input name="data" type="file" accept=".csv" multiple />
        </Label>
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
