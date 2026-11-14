import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type * as React from "react";
import { useState } from "react";
import { ErrorText, Fields, Mono, Panel } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input, Label, Select } from "@/components/ui/input";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { del, type Endpoint, type Environment, get, post, put, type Role, type RoleModel, ROLES } from "@/lib/api";
import { short } from "@/lib/format";

const APIS = ["openai-responses", "openai-completions", "anthropic-messages", "google-generative-ai"];
const THINKING = ["off", "minimal", "low", "medium", "high", "xhigh"];

export function Settings() {
  const endpoints = useQuery({ queryKey: ["endpoints"], queryFn: () => get<Endpoint[]>("/api/settings/endpoints") });
  const [editing, setEditing] = useState<Endpoint | "new" | null>(null);
  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-6 p-6">
      <h1 className="text-lg font-semibold">Settings</h1>
      <Panel title="Model endpoints" actions={editing === null && <Button size="sm" onClick={() => setEditing("new")}>Add endpoint</Button>}>
        <div className="flex flex-col gap-4">
          {editing !== null && <EndpointForm endpoint={editing === "new" ? null : editing} onDone={() => setEditing(null)} />}
          <ErrorText error={endpoints.error} />
          {endpoints.data?.length ? (
            <Table>
              <THead>
                <TR>
                  <TH>Name</TH>
                  <TH>API</TH>
                  <TH>Base URL</TH>
                  <TH>Models</TH>
                  <TH />
                </TR>
              </THead>
              <TBody>
                {endpoints.data.map((e) => (
                  <EndpointRow key={e.id} endpoint={e} onEdit={() => setEditing(e)} />
                ))}
              </TBody>
            </Table>
          ) : null}
        </div>
      </Panel>
      {endpoints.data && <Roles endpoints={endpoints.data} />}
      <EnvironmentPanel />
    </div>
  );
}

function EndpointRow({ endpoint: e, onEdit }: { endpoint: Endpoint; onEdit: () => void }) {
  const qc = useQueryClient();
  const [model, setModel] = useState(e.models[0]);
  const test = useMutation({ mutationFn: () => post<{ ok: boolean; text?: string; error?: string }>(`/api/settings/endpoints/${e.id}/test`, { model }) });
  const remove = useMutation({
    mutationFn: () => del(`/api/settings/endpoints/${e.id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["endpoints"] }).then(() => qc.invalidateQueries({ queryKey: ["roles"] })),
  });
  return (
    <TR>
      <TD className="font-medium">{e.name}</TD>
      <TD>{e.api}</TD>
      <TD className="max-w-60 truncate">{e.base_url}</TD>
      <TD>{e.models.join(", ")}</TD>
      <TD>
        <div className="flex items-center justify-end gap-2">
          {test.data && <Badge variant={test.data.ok ? "pass" : "fail"} title={test.data.error ?? test.data.text}>{test.data.ok ? "connected" : "failed"}</Badge>}
          {test.error && <Badge variant="fail" title={String(test.error)}>failed</Badge>}
          {e.models.length > 1 && (
            <Select className="h-8 w-36" value={model} onChange={(ev) => setModel(ev.target.value)}>
              {e.models.map((m) => <option key={m}>{m}</option>)}
            </Select>
          )}
          <Button size="sm" variant="outline" disabled={test.isPending} onClick={() => test.mutate()}>
            {test.isPending ? "Testing…" : "Test"}
          </Button>
          <Button size="sm" variant="ghost" onClick={onEdit}>Edit</Button>
          <Button size="sm" variant="ghost" onClick={() => confirm(`Delete ${e.name}? Roles using it are unassigned.`) && remove.mutate()}>Delete</Button>
        </div>
      </TD>
    </TR>
  );
}

function EndpointForm({ endpoint, onDone }: { endpoint: Endpoint | null; onDone: () => void }) {
  const qc = useQueryClient();
  const save = useMutation({
    mutationFn: (body: object) => (endpoint ? put(`/api/settings/endpoints/${endpoint.id}`, body) : post("/api/settings/endpoints", body)),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["endpoints"] }).then(onDone),
  });
  return (
    <form
      className="grid gap-3 rounded-md border p-4 sm:grid-cols-2"
      onSubmit={(ev) => {
        ev.preventDefault();
        const f = new FormData(ev.currentTarget);
        const key = String(f.get("api_key") ?? "");
        save.mutate({
          name: f.get("name"), base_url: f.get("base_url"), api: f.get("api"),
          models: String(f.get("models")).split(",").map((m) => m.trim()).filter(Boolean),
          ...(key ? { api_key: key } : {}),
        });
      }}
    >
      <Label>
        Name
        <Input name="name" required defaultValue={endpoint?.name} />
      </Label>
      <Label>
        API
        <Select name="api" defaultValue={endpoint?.api ?? APIS[0]}>
          {APIS.map((a) => <option key={a}>{a}</option>)}
        </Select>
      </Label>
      <Label>
        Base URL
        <Input name="base_url" type="url" required defaultValue={endpoint?.base_url} placeholder="https://…/v1" />
      </Label>
      <Label>
        Models
        <Input name="models" required defaultValue={endpoint?.models.join(", ")} placeholder="comma separated" />
      </Label>
      <Label>
        API key
        <Input name="api_key" type="password" autoComplete="off" required={!endpoint} placeholder={endpoint ? "unchanged" : ""} />
      </Label>
      <div className="flex items-end justify-end gap-2">
        <Button type="button" variant="ghost" onClick={onDone}>Cancel</Button>
        <Button type="submit" disabled={save.isPending}>Save</Button>
      </div>
      <ErrorText error={save.error} />
    </form>
  );
}

function Roles({ endpoints }: { endpoints: Endpoint[] }) {
  const roles = useQuery({ queryKey: ["roles"], queryFn: () => get<Record<Role, RoleModel | null>>("/api/settings/roles") });
  return (
    <Panel title="Agent models">
      <ErrorText error={roles.error} />
      {roles.data && (
        <Table>
          <THead>
            <TR>
              <TH>Role</TH>
              <TH>Endpoint</TH>
              <TH>Model</TH>
              <TH>Reasoning</TH>
              <TH />
            </TR>
          </THead>
          <TBody>
            {ROLES.map((r) => <RoleRow key={r} role={r} current={roles.data[r]} endpoints={endpoints} />)}
          </TBody>
        </Table>
      )}
    </Panel>
  );
}

function RoleRow({ role, current, endpoints }: { role: Role; current: RoleModel | null; endpoints: Endpoint[] }) {
  const qc = useQueryClient();
  const [v, setV] = useState<RoleModel>(current ?? { endpoint_id: "", model: "", thinking: "medium" });
  const ep = endpoints.find((e) => e.id === v.endpoint_id);
  const dirty = JSON.stringify(v) !== JSON.stringify(current);
  const save = useMutation({
    mutationFn: () => put(`/api/settings/roles/${role}`, v),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["roles"] }),
  });
  return (
    <TR>
      <TD className="font-medium capitalize">{role}</TD>
      <TD>
        <Select value={v.endpoint_id} onChange={(e) => {
          const next = endpoints.find((x) => x.id === e.target.value);
          setV({ ...v, endpoint_id: e.target.value, model: next?.models[0] ?? "" });
        }}>
          <option value="" disabled>—</option>
          {endpoints.map((e) => <option key={e.id} value={e.id}>{e.name}</option>)}
        </Select>
      </TD>
      <TD>
        <Select value={v.model} disabled={!ep} onChange={(e) => setV({ ...v, model: e.target.value })}>
          {ep?.models.map((m) => <option key={m}>{m}</option>)}
        </Select>
      </TD>
      <TD>
        <Select value={v.thinking} onChange={(e) => setV({ ...v, thinking: e.target.value })}>
          {THINKING.map((t) => <option key={t}>{t}</option>)}
        </Select>
      </TD>
      <TD className="text-right">
        {save.error ? <Badge variant="fail" title={String(save.error)}>error</Badge> : null}
        <Button size="sm" variant={dirty ? "default" : "outline"} disabled={!dirty || !v.endpoint_id || !v.model || save.isPending} onClick={() => save.mutate()}>
          Save
        </Button>
      </TD>
    </TR>
  );
}

function EnvironmentPanel() {
  const env = useQuery({ queryKey: ["environment"], queryFn: () => get<Environment>("/api/environment") });
  const state = (s: string) => (s === "ready" ? <Badge variant="pass">ready</Badge> : <span className="text-destructive">{s}</span>);
  return (
    <Panel title="Environments">
      <ErrorText error={env.error} />
      {env.data && (
        <Fields rows={[
          ["Version", <Mono>{short(env.data.version)}</Mono>],
          ["R", state(env.data.r)],
          ["Python", state(env.data.python)],
          ...Object.entries(env.data.lockfiles).map(([n, h]) => [n, <Mono>{short(h)}</Mono>] as [string, React.ReactNode]),
        ]} />
      )}
    </Panel>
  );
}
