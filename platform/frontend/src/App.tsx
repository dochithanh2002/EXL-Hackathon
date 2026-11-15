import { ProjectPage } from "@/pages/Project";
import { Projects } from "@/pages/Projects";
import { Settings } from "@/pages/Settings";
import { TITLE } from "@/lib/format";
import { href, useRoute } from "@/lib/route";
import { cn } from "@/lib/utils";

export function App() {
  const route = useRoute();
  const nav = (to: string, label: string, active: boolean) => (
    <a
      href={to}
      aria-current={active ? "page" : undefined}
      className={cn("flex h-full items-center border-b-2 border-transparent text-sm text-muted-foreground hover:text-foreground", active && "border-primary font-medium text-foreground")}
    >
      {label}
    </a>
  );
  return (
    <div className="flex h-dvh flex-col">
      <header className="flex h-12 shrink-0 items-center gap-6 border-b bg-card px-6">
        <a href={href.projects} className="mr-2 flex items-center gap-2 font-semibold tracking-tight">
          <svg viewBox="0 0 24 24" className="size-6" aria-hidden>
            <rect width="24" height="24" rx="6" className="fill-primary" />
            <path d="M6.5 12h10m-4-4.5 4.5 4.5-4.5 4.5" className="fill-none stroke-primary-foreground" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          {TITLE}
        </a>
        {nav(href.projects, "Projects", route.page !== "settings")}
        {nav(href.settings, "Settings", route.page === "settings")}
      </header>
      <main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
        {route.page === "projects" && <Projects />}
        {route.page === "settings" && <Settings />}
        {route.page === "project" && <ProjectPage key={route.id} pid={route.id} step={route.step} />}
      </main>
    </div>
  );
}
