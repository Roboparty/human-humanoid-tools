/** hhtools project page — i18n, nav, video placeholder detection */

const STORAGE_KEY = "hhtools.pages.lang";
const DEFAULT_LANG = "en";
const BIBTEX = `@software{human_humanoid_tools2026,
  title        = {human-humanoid-tools (hhtools): humanoid motion retargeting and dataset analysis},
  author       = {jaggerShen and hhtools contributors},
  year         = {2026},
  url          = {https://github.com/Roboparty/human-humanoid-tools},
  license      = {Apache-2.0}
}`;

const QUICKSTART = `git clone https://github.com/Roboparty/human-humanoid-tools.git
cd human-humanoid-tools
curl -LsSf https://astral.sh/uv/install.sh | sh   # if needed
uv sync --extra all
uv run hhtools web`;

const BULLET_MAP = {
  "bullets-fast-retarget": "highlights.fastRetarget.bullets",
  "bullets-any-motion": "highlights.anyMotion.bullets",
  "bullets-any-urdf": "highlights.anyUrdf.bullets",
  "bullets-batch-retarget": "highlights.batchRetarget.bullets",
  "bullets-r2r": "highlights.r2r.bullets",
  "bullets-dataset-viz": "highlights.datasetViz.bullets",
};

let strings = {};
let lang = DEFAULT_LANG;

function getNested(obj, path) {
  return path.split(".").reduce((o, k) => (o != null ? o[k] : undefined), obj);
}

function applyI18n() {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.title = strings.meta?.title ?? "hhtools";

  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const val = getNested(strings, el.dataset.i18n);
    if (val == null) return;
    if (el.dataset.i18nHtml === "true") {
      el.innerHTML = val;
    } else {
      el.textContent = val;
    }
  });

  document.querySelectorAll(".lang-toggle button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.lang === lang);
  });

  renderWorkflow();
  renderDatasets();
  renderBullets();
}

function renderBullets() {
  Object.entries(BULLET_MAP).forEach(([id, path]) => {
    const ul = document.getElementById(id);
    const items = getNested(strings, path);
    if (ul && items) {
      ul.innerHTML = items.map((t) => `<li>${t}</li>`).join("");
    }
  });
}

function renderWorkflow() {
  const grid = document.getElementById("workflow-grid");
  if (!grid || !strings.workflow?.steps) return;
  grid.innerHTML = strings.workflow.steps
    .map(
      (s) => `
    <div class="workflow-card">
      <div class="step-num">${s.num}</div>
      <h4>${s.title}</h4>
      <p>${s.desc}</p>
    </div>`
    )
    .join("");
}

function renderDatasets() {
  const tbody = document.getElementById("datasets-body");
  if (!tbody || !strings.datasets?.rows) return;
  tbody.innerHTML = strings.datasets.rows
    .map(
      (r) => `
    <tr>
      <td><span class="mode-tag">${r.mode}</span></td>
      <td>${r.dataset}</td>
      <td><a href="${r.paperUrl}" target="_blank" rel="noopener">${r.paper}</a></td>
      <td><a href="${r.downloadUrl}" target="_blank" rel="noopener">${r.download}</a></td>
    </tr>`
    )
    .join("");
}

async function loadLang(next) {
  lang = next;
  localStorage.setItem(STORAGE_KEY, lang);
  try {
    const res = await fetch(`assets/i18n/${lang}.json`);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    strings = await res.json();
  } catch (err) {
    console.warn("[hhtools pages] i18n fetch failed:", err);
    strings = {
      meta: { title: document.title },
      video: {
        placeholder: "Video coming soon",
        placeholderHint: "Place MP4 under docs/assets/videos/",
      },
      workflow: { steps: [] },
      datasets: { rows: [] },
    };
    showServeHint();
  }
  applyI18n();
  await checkVideoSlots();
}

function showServeHint() {
  if (document.getElementById("serve-hint")) return;
  const bar = document.createElement("div");
  bar.id = "serve-hint";
  bar.setAttribute("role", "status");
  bar.style.cssText =
    "position:fixed;bottom:0;left:0;right:0;z-index:9999;padding:12px 16px;" +
    "background:#fff3cd;color:#664d03;border-top:1px solid #ffc107;" +
    "font:14px/1.4 system-ui,sans-serif;text-align:center;";
  bar.innerHTML =
    "页面需通过本地 HTTP 服务打开（不能直接双击 HTML）。" +
    "在仓库里运行：<code style=\"background:rgba(0,0,0,.08);padding:2px 6px;border-radius:4px\">" +
    "cd docs && python3 -m http.server 8080</code> " +
    "然后访问 <a href=\"http://127.0.0.1:8080\">http://127.0.0.1:8080</a>";
  document.body.appendChild(bar);
}

async function checkVideoSlots() {
  const placeholderText = strings.video?.placeholder ?? "Video coming soon";
  const hintText = strings.video?.placeholderHint ?? "";

  await Promise.all(
    [...document.querySelectorAll(".video-slot")].map(
      (slot) =>
        new Promise((resolve) => {
          const src = slot.dataset.video;
          if (!src) {
            resolve();
            return;
          }

          const ph = slot.querySelector(".video-placeholder");
          if (ph) {
            ph.querySelector(".ph-title").textContent = placeholderText;
            ph.querySelector(".ph-hint").textContent = hintText || src;
          }

          const video = slot.querySelector("video");
          if (!video) {
            slot.classList.add("is-missing");
            resolve();
            return;
          }

          const mark = (exists) => {
            slot.classList.toggle("is-missing", !exists);
            resolve();
          };

          video.addEventListener("loadedmetadata", () => mark(true), { once: true });
          video.addEventListener("error", () => mark(false), { once: true });
          video.load();
        })
    )
  );
}

function initNav() {
  document.querySelectorAll('.topbar-nav a[href^="#"]').forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const id = link.getAttribute("href").slice(1);
      document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
    });
  });
}

function initLangToggle() {
  document.querySelectorAll(".lang-toggle button").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.dataset.lang !== lang) loadLang(btn.dataset.lang);
    });
  });
}

function initStaticBlocks() {
  const pre = document.getElementById("quickstart-code");
  if (pre) pre.textContent = QUICKSTART;

  const bib = document.getElementById("bibtex-code");
  if (bib) bib.textContent = BIBTEX;
}

document.addEventListener("DOMContentLoaded", () => {
  if (location.protocol === "file:") showServeHint();
  initStaticBlocks();
  initNav();
  initLangToggle();
  const saved = localStorage.getItem(STORAGE_KEY);
  loadLang(saved === "zh" || saved === "en" ? saved : DEFAULT_LANG);
});
