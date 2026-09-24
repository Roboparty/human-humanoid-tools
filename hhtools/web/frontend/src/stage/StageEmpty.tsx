import { useLocaleText } from "@/LocaleProvider";

/** Initial Stage copy from the legacy renderer, shown until a payload is loaded. */
export function StageEmpty({ visible = true }: { visible?: boolean }) {
  const text = useLocaleText();
  return (
    <div
      className="pointer-events-none absolute inset-0 z-10 grid place-items-center text-center"
      hidden={!visible}
      aria-hidden={!visible}
    >
      <div>
        <span
          className="mx-auto mb-[18px] block size-[54px] bg-foreground opacity-[.18] [mask:url(/icons/motion/film.svg)_center/contain_no-repeat] [-webkit-mask:url(/icons/motion/film.svg)_center/contain_no-repeat]"
          aria-hidden="true"
        />
        <p className="mb-1.5 text-[19px] leading-tight font-semibold text-foreground">
          {text("Drop a motion here to preview", "把动作拖到这里预览")}
        </p>
        <p className="max-w-[360px] text-[13px] leading-normal text-muted-foreground">
          {text(
            "Supports BVH / GLB / NPZ and common motion datasets.",
            "支持 BVH / GLB / NPZ 与常见动作数据集。",
          )}
        </p>
      </div>
    </div>
  );
}
