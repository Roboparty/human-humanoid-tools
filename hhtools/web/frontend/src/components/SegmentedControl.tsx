import {
  ToggleGroup,
  ToggleGroupItem,
} from "@/components/ui/toggle-group";

interface Segment {
  id: string;
  label: string;
}

export function SegmentedControl<const TItems extends readonly Segment[]>({
  label,
  items,
  value,
  onValueChange,
}: {
  label: string;
  items: TItems;
  value: TItems[number]["id"];
  onValueChange(value: TItems[number]["id"]): void;
}) {
  return (
    <ToggleGroup
      type="single"
      value={value}
      onValueChange={(next) => {
        const selected = items.find((item) => item.id === next);
        if (selected) onValueChange(selected.id);
      }}
      aria-label={label}
      className="grid w-full grid-cols-3 gap-1.5"
    >
      {items.map((item) => (
        <ToggleGroupItem
          key={item.id}
          value={item.id}
          className="h-8 min-h-8 min-w-0 w-full rounded-md border border-border-subtle bg-surface px-1.5 py-1.5 text-[11px] leading-none font-semibold text-muted-foreground hover:border-border hover:bg-background hover:text-foreground data-[state=on]:border-primary data-[state=on]:bg-accent data-[state=on]:text-accent-foreground data-[state=on]:hover:bg-accent"
        >
          {item.label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}
