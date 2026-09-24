import { useEffect, useRef, useState } from "react";

import {
  createApplicationMenus,
  type ApplicationCommandContext,
  type ApplicationMenuId,
} from "@/appCommands";
import { cn } from "@/lib/utils";

interface OpenMenu {
  readonly id: ApplicationMenuId;
  readonly interaction: "hover" | "click";
}

/** The menubar renders application commands; App retains all business state. */
export function Navbar(props: ApplicationCommandContext) {
  const menus = createApplicationMenus(props);
  const [openMenu, setOpenMenu] = useState<OpenMenu | null>(null);
  const root = useRef<HTMLElement>(null);

  useEffect(() => {
    const closeOutside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpenMenu(null);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpenMenu(null);
    };

    document.addEventListener("pointerdown", closeOutside);
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOutside);
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, []);

  return (
    <header
      id="topbar"
      className="col-span-full row-start-1 z-[200] flex min-w-0 items-center border-b border-border-subtle bg-surface py-0 pr-3 pl-[25px] max-[600px]:pl-3"
    >
      <div
        className="flex basis-[201px] shrink-0 items-center gap-2 text-lg font-bold tracking-normal text-foreground max-[600px]:basis-[39px]"
        aria-label="HHTOOLS"
      >
        <img className="size-6 object-contain" src="/hhtools-robot.svg" alt="" />
        <span className="max-[600px]:sr-only">HHTOOLS</span>
      </div>

      <nav
        ref={root}
        className="flex min-w-0 self-stretch items-stretch gap-2 max-[600px]:overflow-x-auto"
        aria-label="Application menu"
        role="menubar"
        onMouseLeave={() =>
          setOpenMenu((current) =>
            current?.interaction === "hover" ? null : current,
          )
        }
      >
        {menus.map((menu) => (
          <div key={menu.id} className="relative flex items-center">
            <button
              type="button"
              data-menu-trigger={menu.id}
              className={cn(
                "h-7 cursor-pointer whitespace-nowrap rounded-sm border-0 bg-transparent px-2.5 text-xs font-medium tracking-normal text-muted-foreground hover:bg-accent hover:text-accent-foreground focus-visible:bg-accent focus-visible:text-accent-foreground focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-ring",
                openMenu?.id === menu.id && "bg-accent text-accent-foreground",
              )}
              role="menuitem"
              aria-haspopup="menu"
              aria-expanded={openMenu?.id === menu.id}
              onMouseEnter={() =>
                setOpenMenu((current) => ({
                  id: menu.id,
                  interaction: current?.interaction ?? "hover",
                }))
              }
              onFocus={() =>
                setOpenMenu((current) =>
                  current?.id === menu.id
                    ? current
                    : { id: menu.id, interaction: "hover" },
                )
              }
              onClick={() =>
                setOpenMenu((current) =>
                  current?.id === menu.id && current.interaction === "click"
                    ? null
                    : { id: menu.id, interaction: "click" },
                )
              }
            >
              {menu.label}
            </button>
            {openMenu?.id === menu.id && (
              <div
                className="absolute top-full left-0 z-[120] min-w-[286px] max-w-[340px] pt-[3px] max-[600px]:fixed max-[600px]:top-10 max-[600px]:right-2 max-[600px]:left-2 max-[600px]:min-w-0 max-[600px]:max-w-none max-[600px]:pt-0"
              >
                <div
                  className="rounded-lg border border-border-subtle bg-surface p-[5px] shadow-[0_8px_24px_rgba(2,18,46,0.1)]"
                  role="menu"
                  aria-label={menu.label}
                >
                  {menu.commands.map((command) => (
                    <div key={command.id}>
                      {command.dividerBefore && (
                        <div
                          className="mx-1.5 my-1 h-px bg-border-subtle"
                          role="separator"
                        />
                      )}
                      <button
                        type="button"
                        className="flex min-h-9 w-full cursor-pointer items-center justify-between gap-3 rounded-sm border-0 bg-transparent px-2 py-1.5 text-left text-foreground hover:bg-accent focus-visible:bg-accent focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-ring disabled:cursor-default disabled:text-muted-foreground disabled:opacity-70 disabled:hover:bg-transparent"
                        role="menuitem"
                        disabled={command.enabled === false}
                        title={command.disabledReason ?? command.detail}
                        onClick={() => {
                          command.run();
                          setOpenMenu(null);
                        }}
                      >
                        <span className="min-w-0 truncate text-[13px] font-semibold">
                          {command.label}
                        </span>
                        {command.disabledReason ? (
                          <small className="shrink-0 text-[9px] text-muted-foreground">
                            {command.disabledReason}
                          </small>
                        ) : null}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </nav>
      <div className="flex-1" />
    </header>
  );
}
