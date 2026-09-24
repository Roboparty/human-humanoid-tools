import type { CSSProperties } from "react";

import { cn } from "@/lib/utils";
import { localize, type WorkspaceLocale } from "@/localization";

import { navigationGroups, type ViewId } from "../navigation";

interface SidebarProps {
  activeView: ViewId;
  locale: WorkspaceLocale;
  hidden?: boolean;
  onSelect(view: ViewId): void;
}

type IconStyle = CSSProperties & { "--sidebar-icon": string };

export function Sidebar({ activeView, locale, hidden = false, onSelect }: SidebarProps) {
  return (
    <aside
      id="sidebar"
      hidden={hidden}
      className="col-start-1 row-start-2 row-span-2 flex min-h-0 min-w-0 flex-col overflow-hidden border-r border-border-subtle bg-surface max-[780px]:row-span-3"
      aria-label="Workspace navigation"
    >
      <nav className="min-h-0 min-w-0 flex-1 overflow-y-auto px-3 py-3.5 max-[900px]:px-2">
        <div data-tutorial="workspace-navigation" className="flex flex-col gap-3.5">
          {navigationGroups.map((group) => (
            <section
              key={group.label}
              className="flex flex-col gap-[3px]"
              aria-label={localize(locale, group.label, group.zhLabel)}
            >
              {group.items.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={cn(
                    "flex w-full cursor-pointer items-center gap-3 rounded-md border-0 bg-transparent px-[13px] py-2.5 text-left text-sm font-medium tracking-normal text-muted-foreground hover:bg-accent hover:text-accent-foreground focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-ring max-[900px]:justify-center max-[900px]:gap-0 max-[900px]:px-0",
                    activeView === item.id &&
                      "bg-accent text-accent-foreground",
                  )}
                  aria-current={activeView === item.id ? "page" : undefined}
                  title={localize(locale, item.label, item.zhLabel)}
                  onClick={() => onSelect(item.id)}
                >
                  <span
                    className="sidebar-icon size-5 shrink-0 bg-current [mask:var(--sidebar-icon)_center/18px_18px_no-repeat] [-webkit-mask:var(--sidebar-icon)_center/18px_18px_no-repeat]"
                    style={
                      {
                        "--sidebar-icon": `url(${item.icon})`,
                      } as IconStyle
                    }
                    aria-hidden="true"
                  />
                  <span className="min-w-0 truncate max-[900px]:sr-only">
                    {localize(locale, item.label, item.zhLabel)}
                  </span>
                </button>
              ))}
            </section>
          ))}
        </div>
      </nav>
      <footer className="flex shrink-0 justify-start px-[25px] py-3 max-[600px]:px-3">
        <span className="block h-[41px] w-[120px] overflow-hidden max-[900px]:h-[30px] max-[900px]:w-[30px]">
          <img
            className="block h-auto w-[120px] max-w-none max-[900px]:h-[30px] max-[900px]:w-auto [html[data-theme=dark]_&]:brightness-0 [html[data-theme=dark]_&]:invert"
            src="/roboparty.svg"
            alt="ROBOPARTY"
          />
        </span>
      </footer>
    </aside>
  );
}
