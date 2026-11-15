import { useSyncExternalStore } from "react";

// Hash routes, so the control plane can serve the build as static files: #/, #/projects/<id>, #/settings
export type Route = { page: "projects" } | { page: "project"; id: string; step?: number } | { page: "settings" };

function parse(): Route {
  const [, a, b, c] = window.location.hash.replace(/^#/, "").split("/");
  if (a === "projects" && b) return { page: "project", id: b, step: c ? Number(c) : undefined };
  if (a === "settings") return { page: "settings" };
  return { page: "projects" };
}

let current = parse();
const listeners = new Set<() => void>();
window.addEventListener("hashchange", () => {
  current = parse();
  listeners.forEach((l) => l());
});

export function useRoute(): Route {
  return useSyncExternalStore((l) => (listeners.add(l), () => listeners.delete(l)), () => current);
}

export const href = {
  projects: "#/",
  project: (id: string, step?: number) => `#/projects/${id}${step ? `/${step}` : ""}`,
  settings: "#/settings",
};

export function navigate(to: string) {
  window.location.hash = to.replace(/^#/, "");
}
