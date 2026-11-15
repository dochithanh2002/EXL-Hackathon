/** Agent service: runs one role with the given model, history and prompt, streams its events as NDJSON, and returns
 * the transcript. It holds no workflow state and stores nothing: keys arrive per request and live in memory only
 * for the run; every tool calls back into the control plane. */
import { mkdtempSync, rmSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type Api, InMemoryCredentialStore, type Message } from "@earendil-works/pi-ai";
import {
  type AgentSession,
  createAgentSession,
  createExtensionRuntime,
  ModelRuntime,
  type ResourceLoader,
  SessionManager,
  SettingsManager,
  type ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { ROLES, type ToolSpec } from "./roles/index.ts";

type Thinking = "off" | "minimal" | "low" | "medium" | "high" | "xhigh";

interface ModelRef {
  api: Api;
  base_url: string;
  api_key: string;
  model: string;
  thinking: Thinking;
  context_window?: number;
  max_tokens?: number;
}

interface RunRequest {
  run_id: string;
  role: string;
  model: ModelRef;
  submit_schema: Record<string, unknown>;
  callback: { url: string; token: string };
  history: Message[];
  prompt: string;
}

type Emit = (event: Record<string, unknown>) => void;

const PROVIDER = "endpoint";
const runs = new Map<string, AgentSession>();

async function modelFor(ref: ModelRef) {
  const runtime = await ModelRuntime.create({
    credentials: new InMemoryCredentialStore(),
    modelsPath: null,
    refreshOnCreate: false,
  });
  runtime.registerProvider(PROVIDER, {
    baseUrl: ref.base_url,
    api: ref.api,
    models: [{
      id: ref.model,
      name: ref.model,
      reasoning: ref.thinking !== "off",
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: ref.context_window ?? 128_000,
      maxTokens: ref.max_tokens ?? 16_384,
    }],
  });
  // Never through registerProvider's apiKey, which expands `$VAR` and runs `!command` values.
  await runtime.setRuntimeApiKey(PROVIDER, ref.api_key);
  const model = runtime.getModel(PROVIDER, ref.model);
  if (!model) throw new Error(`model ${ref.model} could not be registered`);
  return { runtime, model };
}

function callbackTool(spec: ToolSpec, callback: RunRequest["callback"]): ToolDefinition {
  return {
    name: spec.name,
    label: spec.name,
    description: spec.description,
    parameters: spec.parameters as ToolDefinition["parameters"],
    executionMode: "sequential",
    async execute(_id, params, signal) {
      const r = await fetch(`${callback.url}/${spec.name}`, {
        method: "POST",
        headers: { authorization: `Bearer ${callback.token}`, "content-type": "application/json" },
        body: JSON.stringify(params),
        signal,
      });
      const body = (await r.json().catch(() => ({}))) as { text?: string; is_error?: boolean };
      if (!r.ok || body.is_error) throw new Error(body.text ?? `control plane returned HTTP ${r.status}`);
      return { content: [{ type: "text", text: body.text ?? "" }], details: {} };
    },
  };
}

function loader(prompt: string): ResourceLoader {
  return {
    getExtensions: () => ({ extensions: [], errors: [], runtime: createExtensionRuntime() }),
    getSkills: () => ({ skills: [], diagnostics: [] }),
    getPrompts: () => ({ prompts: [], diagnostics: [] }),
    getThemes: () => ({ themes: [], diagnostics: [] }),
    getAgentsFiles: () => ({ agentsFiles: [] }),
    getSystemPrompt: () => prompt,
    getSystemPromptSource: () => undefined,
    getAppendSystemPrompt: () => [],
    getAppendSystemPromptSources: () => [],
    extendResources: () => {},
    reload: async () => {},
  };
}

/** One session in an empty scratch directory, with no discovery, no built-in tools and nothing persisted. */
async function session(ref: ModelRef, prompt: string, tools: ToolDefinition[], history: Message[]) {
  const dir = mkdtempSync(join(tmpdir(), "agent-"));
  const { runtime, model } = await modelFor(ref);
  const sessionManager = SessionManager.inMemory(dir);
  for (const m of history) sessionManager.appendMessage(m);
  const { session } = await createAgentSession({
    cwd: dir,
    agentDir: dir,
    model,
    thinkingLevel: ref.thinking,
    modelRuntime: runtime,
    resourceLoader: loader(prompt),
    tools: tools.map((t) => t.name),
    customTools: tools,
    sessionManager,
    settingsManager: SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: true, maxRetries: 2 } }),
  });
  return { session, cleanup: () => { session.dispose(); rmSync(dir, { recursive: true, force: true }); } };
}

/** Prompt and report how the run ended; Pi reports provider failures on the last message, not by throwing. */
async function promptAndSettle(s: AgentSession, prompt: string) {
  await s.prompt(prompt);
  const last = s.messages.at(-1) as { role?: string; stopReason?: string; errorMessage?: string } | undefined;
  if (last?.role === "assistant" && last.stopReason === "aborted") return { type: "aborted" };
  if (last?.role === "assistant" && last.stopReason === "error") {
    return { type: "error", message: last.errorMessage ?? "the model request failed" };
  }
  return { type: "done", text: s.getLastAssistantText() ?? "", messages: s.messages };
}

async function run(req: RunRequest, emit: Emit, closed: AbortSignal) {
  const role = ROLES[req.role];
  if (!role) throw new Error(`unknown role ${req.role}`);
  const tools = role.tools(req.submit_schema).map((t) => callbackTool(t, req.callback));
  const { session: s, cleanup } = await session(req.model, role.prompt, tools, req.history);
  runs.set(req.run_id, s);
  const abort = () => void s.abort();
  closed.addEventListener("abort", abort);
  emit({ type: "started" });
  s.subscribe((e) => {
    if (e.type === "message_update" && e.assistantMessageEvent.type === "text_delta") {
      emit({ type: "text", delta: e.assistantMessageEvent.delta });
    } else if (e.type === "tool_execution_start") {
      emit({ type: "tool_start", id: e.toolCallId, name: e.toolName });
    } else if (e.type === "tool_execution_end") {
      emit({ type: "tool_end", id: e.toolCallId, name: e.toolName, is_error: e.isError });
    }
  });
  try {
    emit(await promptAndSettle(s, req.prompt));
  } finally {
    closed.removeEventListener("abort", abort);
    runs.delete(req.run_id);
    cleanup();
  }
}

async function test(ref: ModelRef) {
  const { session: s, cleanup } = await session(ref, "Reply with the single word OK.", [], []);
  try {
    const r = await promptAndSettle(s, "Reply with the single word OK.");
    return r.type === "done" ? { ok: true, text: r.text } : { ok: false, error: r.message ?? r.type };
  } finally {
    cleanup();
  }
}

async function body<T>(req: IncomingMessage): Promise<T> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  return JSON.parse(Buffer.concat(chunks).toString("utf8")) as T;
}

function json(res: ServerResponse, status: number, value: unknown) {
  res.writeHead(status, { "content-type": "application/json" }).end(JSON.stringify(value));
}

const server = createServer(async (req, res) => {
  try {
    const abortMatch = req.url?.match(/^\/runs\/([\w-]+)\/abort$/);
    if (req.method === "GET" && req.url === "/health") return json(res, 200, { ok: true });
    if (req.method === "POST" && req.url === "/test") return json(res, 200, await test((await body<{ model: ModelRef }>(req)).model));
    if (req.method === "POST" && abortMatch) {
      const s = runs.get(abortMatch[1]);
      if (!s) return json(res, 404, { error: "no such run" });
      await s.abort();
      return json(res, 200, { ok: true });
    }
    if (req.method === "POST" && req.url === "/runs") {
      const r = await body<RunRequest>(req);
      res.writeHead(200, { "content-type": "application/x-ndjson" });
      const closed = new AbortController();
      res.on("close", () => closed.abort());
      const emit: Emit = (e) => void res.write(`${JSON.stringify(e)}\n`);
      try {
        await run(r, emit, closed.signal);
      } catch (e) {
        emit({ type: "error", message: e instanceof Error ? e.message : String(e) });
      }
      return res.end();
    }
    json(res, 404, { error: "not found" });
  } catch (e) {
    if (!res.headersSent) json(res, 500, { error: e instanceof Error ? e.message : String(e) });
    else res.end();
  }
});

const port = Number(process.env.AGENTS_PORT ?? 8100);
server.listen(port, process.env.AGENTS_HOST ?? "127.0.0.1", () => console.log(`agent service on :${port}`));
