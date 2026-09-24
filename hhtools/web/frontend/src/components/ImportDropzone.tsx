import { useState, type CSSProperties, type DragEvent, type ReactNode } from "react";

import type { UploadFile } from "@/lib/api";
import { collectDroppedFiles } from "@/lib/dropFiles";
import { cn } from "@/lib/utils";

type IconStyle = CSSProperties & { "--dropzone-icon": string };

interface ImportDropzoneProps {
  label: string;
  icon: string;
  title: string;
  hint?: string;
  children: ReactNode;
  className?: string;
  disabled?: boolean;
  onFiles?: (files: readonly UploadFile[]) => void | Promise<void>;
}

export function ImportDropzone({
  label,
  icon,
  title,
  hint,
  children,
  className,
  disabled = false,
  onFiles,
}: ImportDropzoneProps) {
  const [dragging, setDragging] = useState(false);

  function acceptDrag(event: DragEvent<HTMLDivElement>): void {
    if (!onFiles || disabled) return;
    event.preventDefault();
    setDragging(true);
  }

  function receiveDrop(event: DragEvent<HTMLDivElement>): void {
    if (!onFiles || disabled) return;
    event.preventDefault();
    event.stopPropagation();
    setDragging(false);
    void collectDroppedFiles(event.dataTransfer).then((files) => {
      if (files.length) return onFiles(files);
      return undefined;
    });
  }

  return (
    <div
      className={cn(
        "flex min-h-[134px] flex-col items-center justify-center gap-2 rounded-lg border-[1.5px] border-dashed border-border bg-[#f4f8ff] px-9 py-[22px] text-center text-muted-foreground transition-colors data-[dragging=true]:border-primary data-[dragging=true]:!bg-accent [html[data-theme=dark]_&]:bg-background",
        className,
      )}
      data-dragging={dragging}
      role="group"
      aria-label={label}
      onDragEnter={acceptDrag}
      onDragOver={acceptDrag}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          setDragging(false);
        }
      }}
      onDrop={receiveDrop}
    >
      <span
        className="size-7 shrink-0 bg-current opacity-[.58] [mask:var(--dropzone-icon)_center/contain_no-repeat] [-webkit-mask:var(--dropzone-icon)_center/contain_no-repeat]"
        style={{ "--dropzone-icon": `url(${icon})` } as IconStyle}
        aria-hidden="true"
      />
      <p className="min-h-[18px] text-[13px] leading-[1.4] font-semibold text-foreground">
        {title}
      </p>
      {hint && <p className="text-[11px] leading-[1.4]">{hint}</p>}
      <div className="mt-0.5 grid w-full grid-flow-col auto-cols-fr gap-2">
        {children}
      </div>
    </div>
  );
}
