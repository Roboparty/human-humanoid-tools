import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { createPortal } from "react-dom";

import { Button } from "@/components/ui/button";
import {
  TUTORIAL_STEPS,
  type TutorialPlacement,
  type TutorialStep,
} from "@/features/tutorial/model";
import { localize, type WorkspaceLocale } from "@/localization";
import type { ViewId } from "@/navigation";

export type TutorialCloseReason = "completed" | "skipped";

export interface TutorialOverlayProps {
  readonly open: boolean;
  readonly locale: WorkspaceLocale;
  readonly onNavigate: (view: ViewId) => void;
  readonly onClose: (reason: TutorialCloseReason) => void;
  readonly onStepChange?: (step: TutorialStep) => void;
}

interface RectStyle extends CSSProperties {
  readonly left: number;
  readonly top: number;
  readonly width: number;
  readonly height: number;
}

interface PointStyle extends CSSProperties {
  readonly left: number;
  readonly top: number;
}

const VIEWPORT_MARGIN = 12;
const ANCHOR_PADDING = 8;
const POPOVER_GAP = 14;

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), Math.max(minimum, maximum));
}

function placedPopover(
  anchor: DOMRect,
  placement: TutorialPlacement,
  popoverWidth: number,
  popoverHeight: number,
): PointStyle {
  let left: number;
  let top: number;

  if (placement === "left") {
    left = anchor.left - popoverWidth - POPOVER_GAP;
    top = anchor.top + (anchor.height - popoverHeight) / 2;
  } else if (placement === "right") {
    left = anchor.right + POPOVER_GAP;
    top = anchor.top + (anchor.height - popoverHeight) / 2;
  } else if (placement === "top") {
    left = anchor.left + (anchor.width - popoverWidth) / 2;
    top = anchor.top - popoverHeight - POPOVER_GAP;
  } else {
    left = anchor.left + (anchor.width - popoverWidth) / 2;
    top = anchor.bottom + POPOVER_GAP;
  }

  return {
    left: clamp(
      left,
      VIEWPORT_MARGIN,
      window.innerWidth - popoverWidth - VIEWPORT_MARGIN,
    ),
    top: clamp(
      top,
      VIEWPORT_MARGIN,
      window.innerHeight - popoverHeight - VIEWPORT_MARGIN,
    ),
  };
}

/** React-owned spotlight tour; application state changes stay behind callbacks. */
export function TutorialOverlay({
  open,
  locale,
  onNavigate,
  onClose,
  onStepChange,
}: TutorialOverlayProps) {
  const [index, setIndex] = useState(0);
  const [highlightStyle, setHighlightStyle] = useState<RectStyle | null>(null);
  const [popoverStyle, setPopoverStyle] = useState<PointStyle | null>(null);
  const [positionedIndex, setPositionedIndex] = useState<number | null>(null);
  const popover = useRef<HTMLDivElement>(null);
  const returnFocus = useRef<HTMLElement | null>(null);
  const titleId = useId();
  const step = TUTORIAL_STEPS[index] ?? TUTORIAL_STEPS[0];

  const position = useCallback(
    (revealAnchor = false) => {
      if (!open) return;
      const anchor = document.querySelector<HTMLElement>(step.anchor);
      if (revealAnchor) {
        anchor?.scrollIntoView({ block: "nearest", inline: "nearest" });
      }

      const popoverRect = popover.current?.getBoundingClientRect();
      const popoverWidth = popoverRect?.width || 320;
      const popoverHeight = popoverRect?.height || 180;
      const anchorRect = anchor?.getBoundingClientRect();
      if (!anchorRect || anchorRect.width <= 0 || anchorRect.height <= 0) {
        setHighlightStyle(null);
        setPopoverStyle({
          left: Math.max(VIEWPORT_MARGIN, (window.innerWidth - popoverWidth) / 2),
          top: Math.max(VIEWPORT_MARGIN, (window.innerHeight - popoverHeight) / 2),
        });
        setPositionedIndex(index);
        return;
      }

      const left = Math.max(0, anchorRect.left - ANCHOR_PADDING);
      const top = Math.max(0, anchorRect.top - ANCHOR_PADDING);
      const right = Math.min(window.innerWidth, anchorRect.right + ANCHOR_PADDING);
      const bottom = Math.min(window.innerHeight, anchorRect.bottom + ANCHOR_PADDING);
      setHighlightStyle({
        left,
        top,
        width: right - left,
        height: bottom - top,
      });
      setPopoverStyle(
        placedPopover(
          anchorRect,
          step.placement,
          popoverWidth,
          popoverHeight,
        ),
      );
      setPositionedIndex(index);
    },
    [index, open, step],
  );

  useEffect(() => {
    if (!open) return;
    onNavigate(step.view);
    onStepChange?.(step);
  }, [onNavigate, onStepChange, open, step]);

  useLayoutEffect(() => {
    if (!open) {
      setHighlightStyle(null);
      setPopoverStyle(null);
      setPositionedIndex(null);
      setIndex(0);
      return;
    }

    setHighlightStyle(null);
    setPositionedIndex(null);
    let secondFrame = 0;
    const firstFrame = window.requestAnimationFrame(() => {
      secondFrame = window.requestAnimationFrame(() => position(true));
    });
    return () => {
      window.cancelAnimationFrame(firstFrame);
      if (secondFrame) window.cancelAnimationFrame(secondFrame);
    };
  }, [index, open, position]);

  useEffect(() => {
    if (!open) return;
    let repositionFrame = 0;
    const reposition = () => {
      window.cancelAnimationFrame(repositionFrame);
      repositionFrame = window.requestAnimationFrame(() => position(false));
    };
    const resizeObserver = new ResizeObserver(reposition);
    const observeAnchor = () => {
      resizeObserver.disconnect();
      const anchor = document.querySelector<HTMLElement>(step.anchor);
      if (anchor) resizeObserver.observe(anchor);
    };
    const application = document.getElementById("app");
    const mutationObserver = new MutationObserver(() => {
      observeAnchor();
      reposition();
    });
    observeAnchor();
    if (application) {
      mutationObserver.observe(application, {
        attributes: true,
        attributeFilter: ["class", "hidden", "style"],
        childList: true,
        subtree: true,
      });
    }
    window.addEventListener("resize", reposition);
    window.addEventListener("scroll", reposition, true);
    return () => {
      window.cancelAnimationFrame(repositionFrame);
      mutationObserver.disconnect();
      resizeObserver.disconnect();
      window.removeEventListener("resize", reposition);
      window.removeEventListener("scroll", reposition, true);
    };
  }, [open, position, step.anchor]);

  const close = useCallback(
    (reason: TutorialCloseReason) => {
      setIndex(0);
      onClose(reason);
    },
    [onClose],
  );

  useEffect(() => {
    if (!open) return;
    returnFocus.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const previousOverflow = document.body.style.overflow;
    const application = document.getElementById("app");
    const applicationWasInert = application?.inert ?? false;
    document.body.style.overflow = "hidden";
    if (application) application.inert = true;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        close("skipped");
        return;
      }
      if (event.key !== "Tab" || !popover.current) return;
      const controls = Array.from(
        popover.current.querySelectorAll<HTMLElement>(
          'button:not(:disabled), [href], [tabindex]:not([tabindex="-1"])',
        ),
      );
      const first = controls[0];
      const last = controls.at(-1);
      if (!first || !last) return;
      const active = document.activeElement;
      if (
        event.shiftKey &&
        (active === first || active === popover.current || !popover.current.contains(active))
      ) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = previousOverflow;
      if (application) application.inert = applicationWasInert;
      const previous = returnFocus.current;
      const focusTarget =
        previous && previous !== document.body && previous.isConnected
          ? previous
          : document.querySelector<HTMLElement>('[data-menu-trigger="help"]');
      focusTarget?.focus();
      returnFocus.current = null;
    };
  }, [close, open]);

  useEffect(() => {
    if (open && positionedIndex === index) popover.current?.focus();
  }, [index, open, positionedIndex]);

  if (!open) return null;

  const last = index === TUTORIAL_STEPS.length - 1;
  const text = (english: string, chinese: string) =>
    localize(locale, english, chinese);

  return createPortal(
    <div
      className="pointer-events-auto fixed inset-0 z-[500]"
      data-tutorial-overlay
      data-tutorial-step={step.id}
    >
      {highlightStyle ? (
        <div
          className="pointer-events-none fixed rounded-lg border-2 border-primary opacity-100 shadow-[0_0_0_9999px_rgba(15,23,42,0.52)] transition-opacity duration-[180ms]"
          style={highlightStyle}
          aria-hidden="true"
        />
      ) : (
        <div className="pointer-events-none fixed inset-0 bg-black/50" aria-hidden="true" />
      )}
      <div
        ref={popover}
        className={`fixed grid max-h-[calc(100vh-24px)] w-[min(320px,calc(100vw-24px))] gap-3 overflow-y-auto rounded-lg border border-border-subtle bg-surface px-4 pt-3.5 pb-4 text-foreground shadow-[0_18px_50px_rgba(0,0,0,0.22)] outline-none transition-opacity duration-[180ms] ${positionedIndex === index ? "visible opacity-100" : "invisible pointer-events-none opacity-0"}`}
        style={popoverStyle ?? { left: VIEWPORT_MARGIN, top: VIEWPORT_MARGIN }}
        role="dialog"
        aria-modal="true"
        aria-hidden={positionedIndex !== index}
        aria-labelledby={titleId}
        tabIndex={-1}
      >
        <header className="flex items-center justify-between gap-2">
          <span className="text-[11px] font-bold text-muted-foreground">
            {index + 1} / {TUTORIAL_STEPS.length}
          </span>
          <Button size="sm" variant="ghost" onClick={() => close("skipped")}>
            {text("Skip tutorial", "跳过教程")}
          </Button>
        </header>
        <div className="grid gap-2">
          <h2 id={titleId} className="text-[15px] leading-snug font-bold text-foreground">
            {localize(locale, step.title.en, step.title.zh)}
          </h2>
          <p className="whitespace-pre-line text-[13px] leading-relaxed text-muted-foreground">
            {localize(locale, step.body.en, step.body.zh)}
          </p>
        </div>
        <Button
          className="w-full"
          size="sm"
          variant="primary"
          onClick={() => {
            if (last) close("completed");
            else setIndex((current) => current + 1);
          }}
        >
          {last ? text("Finish", "完成") : text("Next", "下一步")}
        </Button>
      </div>
    </div>,
    document.body,
  );
}
