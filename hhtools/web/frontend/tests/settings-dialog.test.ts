import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(
  new URL("../src/components/ApplicationDialogs.tsx", import.meta.url),
  "utf8",
);

test("settings footer keeps refresh lightweight on the left and save on the right", () => {
  assert.doesNotMatch(source, /Reset layout|重置布局|onResetLayout/);

  const footerStart = source.indexOf('<footer className="flex items-center justify-between');
  const footerEnd = source.indexOf("</footer>", footerStart);
  assert.notEqual(footerStart, -1);
  assert.notEqual(footerEnd, -1);

  const footer = source.slice(footerStart, footerEnd);
  assert.ok(footer.indexOf("<RefreshButton") < footer.indexOf("<Button"));
  assert.match(footer, /<RefreshButton\s+variant="ghost"/);
});

test("about credits the named authors and project contributors", () => {
  assert.match(
    source,
    /Jagger Shen, Nora Sun and hhtools contributors/,
  );
  assert.match(source, /Jagger Shen、Nora Sun 与 hhtools 贡献者/);
  assert.doesNotMatch(source, /jaggerShen/);
});
