import { analyst } from "./analyst.ts";
import { diagnostician } from "./diagnostician.ts";
import { translator } from "./translator.ts";

export interface ToolSpec {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

/** A role is a system prompt and its tools. Every tool runs in the control plane; the submit tool's parameters are
 * the role's output schema, which the control plane sends with each run. */
export interface Role {
  prompt: string;
  tools: (submitSchema: Record<string, unknown>) => ToolSpec[];
}

export const ROLES: Record<string, Role> = { analyst, translator, diagnostician };
