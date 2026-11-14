import { ProjectPage } from "@/pages/Project";
import { Projects } from "@/pages/Projects";
import { Settings } from "@/pages/Settings";
import { href, useRoute } from "@/lib/route";
import { cn } from "@/lib/utils";

export function App() {
  const route = useRoute();
  const nav = (to: string, label: string, active: boolean) => (
    <a href={to} className={cn("text-sm text-muted-foreground hover:text-foreground", active && "text-foreground font-medium")}>
      {label}
    </a>
  );
  return (
    <div className="flex min-h-screen flex-col">
      <header className="flex h-12 shrink-0 items-center gap-6 border-b px-6">
        <a href={href.projects} className="font-semibold">R → Python</a>
        {nav(href.projects, "Projects", route.page !== "settings")}
        {nav(href.settings, "Settings", route.page === "settings")}
      </header>
      <main className="flex min-h-0 flex-1 flex-col">
        {route.page === "projects" && <Projects />}
        {route.page === "settings" && <Settings />}
        {route.page === "project" && <ProjectPage key={route.id} pid={route.id} step={route.step} />}
      </main>
    </div>
  );
}
