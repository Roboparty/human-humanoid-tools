// Dataset Visualization & Analysis panel.

const bridge = () => window.__hhApp || {};

const DV_USER_ROOT_KEY = "hh.dvUserSourceRoot";

const state = {
  clips: [],
  summary: null,
  catalog: null,
  dataKind: "unknown",
  activeTags: new Set(),
  tagMode: "or",
  selected: new Set(),
  subsetIds: new Set(),
  viewDim: "num:complexity",
  histBrush: null,
  catBrush: null,
  analyzeSource: "",
  analyzeSourceRoot: "",
  uploadSummary: null,
  embeddingName: "handcrafted",
  previewRobot: "",
  scatterView: { scale: 1, panX: 0, panY: 0, dragging: false, dragMoved: false, lastX: 0, lastY: 0 },
  histLayout: null,
  histDrag: { active: false, startBin: -1 },
  subsetTimer: null,
  hoverClipId: null,
  hoverBin: -1,
};

const $ = (id) => document.getElementById(id);
const QUALITY_TAGS = ["quality_ok", "quality_warn", "quality_bad"];
const DYN_TAGS = ["static", "low_dynamic", "mid_dynamic", "high_dynamic", "burst"];
const CAT_DIMS = ["cluster_id", "folder_label", "quality_band", "dynamics_band", "source_kind"];
const CATEGORICAL = ["#6366f1", "#14b8a6", "#f472b6", "#fb923c", "#38bdf8", "#a78bfa", "#34d399", "#fbbf24", "#f87171", "#64748b"];

/** Logseq-style graph palette — soft, distinct hues per cluster. */
const GRAPH_PALETTE = [
  "#5b7cfa", "#2dd4bf", "#e879f9", "#fb7185", "#fbbf24",
  "#38bdf8", "#a3e635", "#c084fc", "#f97316", "#94a3b8",
];

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return {
    r: parseInt(h.slice(0, 2), 16),
    g: parseInt(h.slice(2, 4), 16),
    b: parseInt(h.slice(4, 6), 16),
  };
}

function rgba(hex, a) {
  const { r, g, b } = hexToRgb(hex);
  return `rgba(${r},${g},${b},${a})`;
}

function clusterColor(clusterId, colorMap) {
  const k = String(clusterId ?? "?");
  if (!colorMap.has(k)) colorMap.set(k, GRAPH_PALETTE[colorMap.size % GRAPH_PALETTE.length]);
  return colorMap.get(k);
}

function roundRect(ctx, x, y, w, h, r) {
  if (ctx.roundRect) { ctx.roundRect(x, y, w, h, r); return; }
  ctx.rect(x, y, w, h);
}

// ------------------------------------------------------------------ catalog
async function loadCatalog() {
  if (state.catalog) return state.catalog;
  try { state.catalog = await bridge().API.get("/api/dataset/catalog"); }
  catch { state.catalog = {}; }
  applyCatalogTexts();
  return state.catalog;
}

function applyCatalogTexts() {
  const c = state.catalog || {};
  const grid = $("dv-format-grid");
  if (grid) {
    grid.innerHTML =
      `<div class="dv-format-item"><b>人体</b><span>拖入含 BVH / NPZ / PKL / NPY / GLB 的文件夹即可（AMASS、ACCAD、CMU、LAFAN、OMOMO、parc_ms、holosoma 等；支持 mimic / intermimic / meshmimic 多级目录）</span></div>`
      + `<div class="dv-format-item"><b>机器人</b><span>拖入 retarget 导出的轨迹 CSV / PKL / NPZ 文件夹（含 <code>*_export</code>、terrain / object 侧车）</span></div>`
      + `<div class="dv-format-warn">请勿混合拖入人体与机器人数据；一次分析只支持一种。</div>`;
  }
}

function detectDataKind() {
  const clips = okClips();
  const hasHuman = clips.some((c) => c.source_kind !== "robot");
  const hasRobot = clips.some((c) => c.source_kind === "robot");
  if (hasHuman && hasRobot) return "mixed";
  if (hasRobot) return "robot";
  if (hasHuman) return "human";
  return "unknown";
}

function updateKindBadge() {
  if (okClips().length) state.dataKind = detectDataKind();
  const badge = $("dv-kind-badge");
  if (!badge) return;
  const map = { human: "人体", robot: "机器人", mixed: "混合 ⚠", unknown: "—" };
  badge.textContent = map[state.dataKind] || "—";
  badge.className = "dv-card-badge" + (state.dataKind === "mixed" ? " warn" : "");
  const humanBtn = $("dv-human-basket");
  const robotBtn = $("dv-export-robot");
  const humanOk = state.dataKind === "human";
  const robotOk = state.dataKind === "robot";
  if (humanBtn) {
    humanBtn.hidden = false;
    humanBtn.disabled = !humanOk;
    humanBtn.title = humanOk ? "" : "当前为机器人数据，无法加入人体批量篮子";
  }
  if (robotBtn) {
    robotBtn.hidden = false;
    robotBtn.disabled = !robotOk;
    robotBtn.title = robotOk
      ? "导出选中机器人 clip；可勾选是否打包轨迹文件"
      : "当前为人体数据，无法导出机器人轨迹";
  }
  const robotOpts = $("dv-robot-export-opts");
  if (robotOpts) robotOpts.hidden = !robotOk;
  const userRootWrap = $("dv-user-root-wrap");
  const needsRoot = !!state.analyzeSource
    || okClips().some((c) => looksLikeTempSourcePath(c.source_path));
  if (userRootWrap) userRootWrap.hidden = !needsRoot;
  syncRobotExportLabel();
  void refreshRobotPreviewUI();
}

function syncRobotExportLabel() {
  const btn = $("dv-export-robot");
  const pack = $("dv-robot-export-files")?.checked !== false;
  if (!btn || btn.disabled) return;
  btn.textContent = pack ? "导出机器人数据 (ZIP)" : "导出机器人清单 (JSON)";
}

// ------------------------------------------------------------------ clip helpers
function okClips() {
  return state.clips.filter((c) => !c.error && c.metrics && Object.keys(c.metrics).length);
}

function clipMatchesTags(clip) {
  if (!state.activeTags.size) return true;
  const tags = new Set(clip.tags || []);
  if (state.tagMode === "and") {
    for (const t of state.activeTags) if (!tags.has(t)) return false;
    return true;
  }
  for (const t of state.activeTags) if (tags.has(t)) return true;
  return false;
}

function clipCategory(clip, dim) {
  switch (dim) {
    case "cluster_id": return String(clip.cluster_id ?? "?");
    case "folder_label": return clip.folder_label || "?";
    case "source_kind": return clip.source_kind || "?";
    case "quality_band":
      for (const t of QUALITY_TAGS) if ((clip.tags || []).includes(t)) return t;
      return "—";
    case "dynamics_band":
      for (const t of DYN_TAGS) if ((clip.tags || []).includes(t)) return t;
      return "—";
    default: return "?";
  }
}

function parseViewDim() {
  const [kind, key] = (state.viewDim || "num:complexity").split(":");
  return { kind, key };
}

function clipInBrush(clip) {
  const { kind, key } = parseViewDim();
  if (kind === "cat") {
    if (!state.catBrush?.size) return true;
    return state.catBrush.has(clipCategory(clip, key));
  }
  if (!state.histBrush) return true;
  const v = clip.metrics?.[key];
  if (v == null || !isFinite(v)) return false;
  return v >= state.histBrush.lo && v <= state.histBrush.hi;
}

function tagFilteredClips() { return okClips().filter(clipMatchesTags); }
function visibleClips() { return tagFilteredClips().filter(clipInBrush); }

function exportTargetIds() {
  return [...new Set([...state.subsetIds, ...manualSelectedIds()])];
}

function manualSelectedIds() {
  return [...state.selected].filter((id) => !state.subsetIds.has(id));
}

function isManualSelectable(id) {
  return !state.subsetIds.has(id);
}

function pruneManualSelection() {
  for (const id of state.subsetIds) state.selected.delete(id);
}

function entryFromClip(clip) {
  const sp = clip.source_path || "";
  const seq = sp.split("/").pop();
  return {
    dataset: clip.dataset,
    folder_label: clip.folder_label,
    sequence_id: seq,
    source_path: sp,
    stem: (seq || "").replace(/\.[^.]+$/, ""),
  };
}

function inferDefaultRobot() {
  const counts = new Map();
  for (const c of okClips()) {
    if (c.source_kind !== "robot") continue;
    const p = String(c.metrics?.robot_preset || "").trim();
    if (p) counts.set(p, (counts.get(p) || 0) + 1);
  }
  if (!counts.size) return state.previewRobot || "";
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
}

async function refreshRobotPreviewUI() {
  const box = $("dv-robot-preview");
  if (!box) return;
  const show = state.dataKind === "robot";
  box.hidden = !show;
  if (!show) return;
  const inferred = inferDefaultRobot();
  const hint = $("dv-robot-hint");
  if (hint) {
    hint.textContent = inferred
      ? `已从 CSV 推断：${inferred} · 点击散点/列表 ▶ 用 mesh 播放`
      : "未检测到 robot meta，请手动选择机器人 preset";
  }
  const pick = state.previewRobot || inferred;
  const val = await bridge().populateDvRobotSelect?.(pick);
  if (val) state.previewRobot = val;
}

async function previewClip(clip) {
  const entry = entryFromClip(clip);
  if (clip.source_kind === "robot" || clip.dataset === "robot") {
    await bridge().previewRobotClip?.(entry, state.previewRobot || undefined);
  } else {
    await bridge().loadLibraryEntry?.(entry);
  }
}

// ------------------------------------------------------------------ subset FPS
function rankNormalize(values) {
  const n = values.length;
  if (n <= 1) return values.map(() => 0);
  const order = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
  const ranks = new Array(n);
  order.forEach((idx, r) => { ranks[idx] = r; });
  return ranks.map((r) => r / (n - 1));
}

function globalWeightedFps(embeddings, complexity, k, alpha) {
  const n = embeddings.length;
  if (!n || k <= 0) return [];
  k = Math.min(k, n);
  const cHat = rankNormalize(complexity);
  let anchor = 0;
  for (let i = 1; i < n; i++) if (cHat[i] > cHat[anchor]) anchor = i;
  const selected = [anchor];
  let dist = embeddings.map((e) => {
    let s = 0;
    for (let j = 0; j < e.length; j++) { const d = e[j] - embeddings[anchor][j]; s += d * d; }
    return Math.sqrt(s);
  });
  dist[anchor] = -Infinity;
  while (selected.length < k) {
    const finite = dist.filter((d) => isFinite(d));
    const dMax = finite.length ? Math.max(...finite) : 0;
    let best = -1, bestScore = -Infinity;
    for (let i = 0; i < n; i++) {
      if (selected.includes(i)) continue;
      const score = alpha * (dMax > 1e-12 ? dist[i] / dMax : 0) + (1 - alpha) * cHat[i];
      if (score > bestScore) { bestScore = score; best = i; }
    }
    if (best < 0) break;
    selected.push(best);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < embeddings[i].length; j++) {
        const d = embeddings[i][j] - embeddings[best][j]; s += d * d;
      }
      dist[i] = Math.min(dist[i], Math.sqrt(s));
    }
    for (const s of selected) dist[s] = -Infinity;
  }
  return selected;
}

function recomputeSubset() {
  const flt = visibleClips().filter((c) => c.embedding);
  if (!flt.length) { state.subsetIds = new Set(); return; }
  const ratio = parseInt($("dv-subset-ratio").value, 10) / 100;
  const alpha = parseInt($("dv-subset-alpha").value, 10) / 100;
  const k = Math.max(1, Math.round(flt.length * ratio));
  const idx = globalWeightedFps(
    flt.map((c) => c.embedding),
    flt.map((c) => Number(c.metrics?.complexity || 0)),
    k, alpha,
  );
  state.subsetIds = new Set(idx.map((i) => flt[i].clip_id));
  pruneManualSelection();
}

function scheduleSubset() {
  clearTimeout(state.subsetTimer);
  state.subsetTimer = setTimeout(() => {
    recomputeSubset();
    renderScatter();
    renderSelbar();
    renderOverview();
  }, 60);
}

// ------------------------------------------------------------------ upload / analyze
function walkEntry(entry, out, prefix = "") {
  return new Promise((resolve) => {
    if (entry.isFile) {
      entry.file((f) => { f._relpath = prefix + f.name; out.push(f); resolve(); });
    } else if (entry.isDirectory) {
      entry.createReader().readEntries(async (entries) => {
        await Promise.all(entries.map((e) => walkEntry(e, out, prefix + entry.name + "/")));
        resolve();
      });
    } else resolve();
  });
}

function guessUploadKind(files) {
  let csv = 0, motion = 0, robotCsv = 0;
  for (const f of files) {
    const n = (f._relpath || f.name || "").toLowerCase();
    const base = n.split("/").pop();
    if (base.startsWith("object_") && base.endsWith(".csv")) continue;
    if (n.endsWith(".csv")) {
      csv++;
      if (/root_x|dof_/.test(f.name || "")) robotCsv++;
    } else if (/\.(bvh|npz|pkl|npy|glb|pt)$/.test(n)) motion++;
  }
  if (csv && motion) return "mixed";
  if (csv) return "robot";
  return "human";
}

function resolveUploadKind(info, fallback = "unknown") {
  const r = info?.robot_count || 0;
  const h = info?.human_count || 0;
  if (r && !h) return "robot";
  if (h && !r) return "human";
  if (r && h) return "mixed";
  return fallback;
}

function renderUploadBasket(info) {
  const n = info?.clip_count || 0;
  const kind = resolveUploadKind(info, state.dataKind);
  const basket = $("dv-upload-basket");
  const dropzone = $("dv-dropzone");
  const dropLabel = $("dv-drop-label");

  if (!n) {
    if (basket) basket.hidden = true;
    if (dropzone) dropzone.classList.remove("ok", "busy");
    if (dropLabel) dropLabel.innerHTML = "拖入<b>人体</b>或<b>机器人</b>数据集文件夹";
    $("dv-source-display").textContent = "未指定目录";
    return;
  }

  state.uploadSummary = info;
  state.dataKind = kind;
  if ($("dv-kind-badge")) {
    $("dv-kind-badge").textContent = kind === "robot" ? "机器人" : kind === "human" ? "人体" : "混合 ⚠";
    $("dv-kind-badge").className = "dv-card-badge" + (kind === "mixed" ? " warn" : "");
  }
  $("dv-source-display").textContent =
    `${kind === "robot" ? "机器人" : "人体"} · 共 ${n} clip（可继续拖入追加）`;

  if (dropzone) {
    dropzone.classList.remove("busy", "err");
    dropzone.classList.add("ok");
  }
  if ($("dv-drop-icon")) $("dv-drop-icon").textContent = "✓";
  if (dropLabel) {
    dropLabel.innerHTML = `✓ 已加载 <b>${n}</b> 个 clip · 继续拖入可追加到同一批次`;
  }

  if (basket) {
    basket.hidden = false;
    $("dv-basket-summary").textContent =
      `${kind === "robot" ? "机器人" : "人体"} · ${n} clip · ${Object.keys(info.folders || {}).length} 组`;
    const list = $("dv-basket-list");
    if (list) {
      list.innerHTML = "";
      const clips = info.clips || [];
      const byFolder = new Map();
      for (const c of clips) {
        const f = c.folder_label || "—";
        if (!byFolder.has(f)) byFolder.set(f, []);
        byFolder.get(f).push(c.clip_id.split("/").pop());
      }
      for (const [folder, names] of [...byFolder.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
        const row = document.createElement("li");
        row.className = "dv-basket-item";
        const folderEl = document.createElement("span");
        folderEl.className = "dv-basket-folder";
        folderEl.textContent = folder;
        const metaEl = document.createElement("span");
        metaEl.className = "dv-basket-meta";
        metaEl.textContent = `${names.length} clip`;
        const namesEl = document.createElement("span");
        namesEl.className = "dv-basket-names";
        namesEl.textContent = `${names.slice(0, 3).join(" · ")}${names.length > 3 ? " …" : ""}`;
        const rmBtn = document.createElement("button");
        rmBtn.type = "button";
        rmBtn.className = "dv-basket-remove btn-link";
        rmBtn.title = "移除此文件夹";
        rmBtn.textContent = "×";
        rmBtn.onclick = (ev) => {
          ev.stopPropagation();
          removeBasketFolder(folder);
        };
        row.append(folderEl, metaEl, namesEl, rmBtn);
        list.appendChild(row);
      }
    }
  }
  updateKindBadge();
}

function clearUploadBasket() {
  state.analyzeSource = "";
  state.uploadSummary = null;
  state.dataKind = "unknown";
  $("dv-source").value = "";
  $("dv-upload-basket").hidden = true;
  $("dv-dropzone")?.classList.remove("ok", "busy", "err");
  if ($("dv-drop-icon")) $("dv-drop-icon").textContent = "📁";
  $("dv-drop-label").innerHTML = "拖入<b>人体</b>或<b>机器人</b>数据集文件夹";
  $("dv-source-display").textContent = "未指定目录";
  $("dv-status").textContent = "";
  updateKindBadge();
}

async function removeBasketFolder(folderLabel) {
  const { API, toast } = bridge();
  if (!state.analyzeSource) {
    toast("当前无上传批次", true);
    return;
  }
  try {
    const info = await API.post("/api/dataset/upload/remove", {
      source: state.analyzeSource,
      folder_label: folderLabel,
    });
    if (!info.clip_count) {
      clearUploadBasket();
      if (state.clips.length) {
        state.clips = [];
        state.summary = null;
        state.selected.clear();
        state.subsetIds.clear();
        if ($("dv-results")) $("dv-results").hidden = true;
      }
      toast(`已移除「${folderLabel}」，批次已空`);
      return;
    }
    state.analyzeSource = info.source || state.analyzeSource;
    $("dv-source").value = state.analyzeSource;
    renderUploadBasket(info);
    if (state.clips.length) {
      state.clips = state.clips.filter((c) => c.folder_label !== folderLabel);
      for (const id of [...state.selected]) {
        if (!state.clips.some((c) => c.clip_id === id)) state.selected.delete(id);
      }
      for (const id of [...state.subsetIds]) {
        if (!state.clips.some((c) => c.clip_id === id)) state.subsetIds.delete(id);
      }
      recomputeSubset();
      renderAll();
    }
    toast(`已移除「${folderLabel}」`);
  } catch (e) {
    toast(e.message, true);
  }
}

async function ingestDroppedFiles(files) {
  const { uploadFilesXHR, toast } = bridge();
  if (!files?.length) return;
  const kind = guessUploadKind(files);
  if (kind === "mixed") {
    toast("请勿同时拖入人体动作与机器人 CSV，请分开分析", true);
    return;
  }
  if (state.analyzeSource && state.dataKind !== "unknown" && state.dataKind !== kind) {
    toast("与当前批次类型不同，请先点「清空批次」再拖入", true);
    return;
  }

  const dropzone = $("dv-dropzone");
  dropzone?.classList.remove("ok", "err");
  dropzone?.classList.add("busy");
  $("dv-status").textContent = `上传 ${files.length} 个文件…`;
  const appendTo = state.analyzeSource || null;
  syncUserRootField();
  const userRoot = getUserSourceRoot();
  try {
    const info = await uploadFilesXHR("/api/dataset/upload", files, {
      appendTo,
      userSourceRoot: userRoot || undefined,
    });
    state.analyzeSource = info.source || "";
    $("dv-source").value = state.analyzeSource;
    if (info.user_source_root) setUserSourceRoot(info.user_source_root);
    const n = info.clip_count || 0;
    if (!n) {
      dropzone?.classList.remove("busy");
      dropzone?.classList.add("err");
      toast("未识别到可分析 clip：人体请拖入含 BVH/NPZ/PKL 等的文件夹；机器人请拖入轨迹 CSV/PKL/NPZ 文件夹", true);
      $("dv-source-display").textContent = appendTo ? "追加后仍无 clip" : "未识别到 clip";
      $("dv-status").textContent = "";
      return;
    }
    renderUploadBasket(info);
    $("dv-status").textContent = appendTo
      ? `追加成功 · 当前共 ${n} clip`
      : `加载成功 · ${n} clip`;
    toast(appendTo ? `已追加，当前共 ${n} 个 clip` : `已加载 ${n} 个 clip`);
  } catch (e) {
    dropzone?.classList.remove("busy");
    dropzone?.classList.add("err");
    toast(e.message, true);
    $("dv-status").textContent = "上传失败";
  }
}

async function pickFolder() {
  const inp = document.createElement("input");
  inp.type = "file"; inp.multiple = true; inp.webkitdirectory = true; inp.style.display = "none";
  inp.onchange = () => {
    const files = Array.from(inp.files || []);
    for (const f of files) f._relpath = f.webkitRelativePath || f.name;
    document.body.removeChild(inp);
    ingestDroppedFiles(files);
  };
  document.body.appendChild(inp);
  inp.click();
}

async function runAnalysis() {
  const { API, toast } = bridge();
  await loadCatalog();
  const source = state.analyzeSource || $("dv-source").value.trim();
  const prog = $("dv-progress");
  prog.style.display = "block";
  prog.querySelector(".bar").style.width = "4%";
  $("dv-analyze").disabled = true;
  $("dv-status").textContent = "分析中…";
  try {
    const body = { embedding: $("dv-embedding").value, force: $("dv-force").checked };
    if (source) body.source = source;
    const { job_id } = await API.post("/api/dataset/analyze", body);
    let result = null;
    while (true) {
      const j = await API.get(`/api/job/${job_id}`);
      prog.querySelector(".bar").style.width = `${Math.round((j.progress || 0) * 100)}%`;
      $("dv-status").textContent = j.message || "分析中…";
      if (j.status === "done") { result = j.result; break; }
      if (j.status === "error") throw new Error(j.error || "失败");
      await new Promise((r) => setTimeout(r, 400));
    }
    state.clips = result.clips || [];
    state.summary = result.summary || null;
    state.analyzeSourceRoot = result.meta?.source_root || state.analyzeSource || "";
    state.embeddingName = result.meta?.embedding || $("dv-embedding")?.value || "handcrafted";
    state.activeTags.clear();
    state.selected.clear();
    state.histBrush = null;
    state.catBrush = null;
    resetScatterView(false);
    $("dv-results").hidden = false;
    const s = result.summary || {};
    $("dv-status").textContent = `完成 · ${s.num_ok || 0} clip`;
    updateKindBadge();
    await refreshRobotPreviewUI();
    if (state.dataKind === "mixed") {
      toast("检测到人体与机器人混合数据，建议分开目录分析", true);
      $("dv-status").textContent += " · ⚠ 混合数据";
    }
    buildViewDimOptions();
    recomputeSubset();
    renderAll();
  } catch (e) {
    toast(e.message, true);
    $("dv-status").textContent = "失败：" + e.message;
  } finally {
    $("dv-analyze").disabled = false;
    setTimeout(() => { prog.style.display = "none"; }, 500);
  }
}

// ------------------------------------------------------------------ info panels
function renderTagInfo() {
  const box = $("dv-tag-info");
  if (!state.activeTags.size) { box.hidden = true; return; }
  box.hidden = false;
  const tags = state.catalog?.tags || {};
  box.innerHTML = [...state.activeTags].map((t) => {
    const info = tags[t] || {};
    return `<div class="dv-info-card"><b>${info.title || t}</b>`
      + (info.desc ? `<p>${info.desc}</p>` : "")
      + (info.formula ? `<code>${info.formula}</code>` : "") + `</div>`;
  }).join("");
}

function renderMetricInfo() {
  const { kind, key } = parseViewDim();
  const c = state.catalog || {};
  const info = kind === "cat" ? (c.categories?.[key] || {}) : (c.metrics?.[key] || {});
  const parts = [];
  if (info.desc) {
    parts.push(`<span>${info.title || key}${info.unit ? ` (${info.unit})` : ""} — ${info.desc}</span>`);
  } else {
    parts.push(`<span>${info.title || key}</span>`);
  }
  if (key === "cluster_id" && kind === "cat") {
    const cl = c.clustering || {};
    const emb = state.embeddingName === "pae" ? "档B PAE" : "档A 手工特征";
    parts.push(`<div class="dv-info-detail"><b>聚类输入（${emb}）</b> ${cl.handcrafted_inputs || info.detail || ""}</div>`);
    if (cl.algorithm) parts.push(`<div class="dv-info-detail">${cl.algorithm}</div>`);
  } else if (info.detail) {
    parts.push(`<div class="dv-info-detail">${info.detail}</div>`);
  }
  if (info.formula) parts.push(`<code class="dv-info-formula">${info.formula}</code>`);
  $("dv-metric-info").innerHTML = parts.join("");
}

function brushBinRange(lay) {
  if (!lay || lay.kind !== "num" || !state.histBrush) return null;
  const { edges, nbins } = lay;
  let i0 = nbins, i1 = -1;
  for (let i = 0; i < nbins; i++) {
    if (edges[i + 1] >= state.histBrush.lo && edges[i] <= state.histBrush.hi) {
      i0 = Math.min(i0, i); i1 = Math.max(i1, i);
    }
  }
  return i1 >= i0 ? { i0, i1 } : null;
}

function formatBrushRange() {
  if (state.histBrush) {
    const lo = state.histBrush.lo;
    const hi = state.histBrush.hi;
    const fmt = (v) => (Math.abs(v) >= 10 || Number.isInteger(v) ? v.toFixed(2) : v.toFixed(3));
    return `${fmt(lo)} – ${fmt(hi)}`;
  }
  if (state.catBrush?.size) return [...state.catBrush].join(", ");
  return "";
}

function buildViewDimOptions() {
  const sel = $("dv-view-dim");
  sel.innerHTML = "";
  const ogN = document.createElement("optgroup");
  ogN.label = "数值指标";
  for (const k of state.summary?.numeric_keys || []) {
    const o = document.createElement("option");
    o.value = `num:${k}`;
    o.textContent = state.catalog?.metrics?.[k]?.title || k;
    ogN.appendChild(o);
  }
  sel.appendChild(ogN);
  const ogC = document.createElement("optgroup");
  ogC.label = "类别";
  for (const k of CAT_DIMS) {
    const o = document.createElement("option");
    o.value = `cat:${k}`;
    o.textContent = state.catalog?.categories?.[k]?.title || k;
    ogC.appendChild(o);
  }
  sel.appendChild(ogC);
  if ([...sel.options].some((o) => o.value === state.viewDim)) sel.value = state.viewDim;
  else if (sel.options.length) state.viewDim = sel.value = sel.options[0].value;
}

// ------------------------------------------------------------------ histogram
function canvasXY(ev, canvas) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (ev.clientX - rect.left) * (canvas.width / rect.width),
    y: (ev.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function binAtX(x, lay) {
  if (!lay || x < lay.pad.l || x > lay.pad.l + lay.plotW) return -1;
  const i = Math.floor((x - lay.pad.l) / lay.bw);
  if (lay.kind === "cat") return i >= 0 && i < lay.keys.length ? i : -1;
  return i >= 0 && i < lay.nbins ? i : -1;
}

function applyBinBrush(lay, i0, i1) {
  const a = Math.min(i0, i1), b = Math.max(i0, i1);
  if (lay.kind === "cat") {
    state.catBrush = new Set(lay.keys.slice(a, b + 1));
    state.histBrush = null;
  } else {
    state.histBrush = { lo: lay.edges[a], hi: lay.edges[b + 1] };
    state.catBrush = null;
  }
  recomputeSubset();
  renderAll();
}

function renderDistribution() {
  const { kind, key } = parseViewDim();
  state.histLayout = null;
  if (kind === "cat") renderCategoryBars(key);
  else renderNumericHistogram(key);
  renderMetricInfo();
}

function renderNumericHistogram(metric) {
  const canvas = $("dv-hist-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const hist = state.summary?.histograms?.[metric];
  const info = state.catalog?.metrics?.[metric] || {};
  if (!hist) return;

  const { edges, counts: allCounts, min, max, mean, median } = hist;
  const nbins = edges.length - 1;
  const tagCounts = new Array(nbins).fill(0);
  for (const c of tagFilteredClips()) {
    const v = c.metrics?.[metric];
    if (v == null || !isFinite(v)) continue;
    let b = Math.floor(((v - min) / (max - min)) * nbins);
    b = Math.max(0, Math.min(nbins - 1, b));
    tagCounts[b]++;
  }

  const pad = { l: 52, r: 16, t: 20, b: 44 };
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const maxC = Math.max(1, ...allCounts, ...tagCounts);
  const gap = 3;
  const bw = plotW / nbins;
  state.histLayout = { pad, plotW, plotH, bw, nbins, edges, metric, kind: "num", min, max };
  const brushRange = brushBinRange(state.histLayout);

  ctx.strokeStyle = "#e0e0e5"; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + plotH);
  ctx.lineTo(pad.l + plotW, pad.t + plotH); ctx.stroke();

  ctx.fillStyle = "#8e8e93"; ctx.font = "11px -apple-system,sans-serif";
  ctx.textAlign = "right";
  ctx.fillText("clip 数", pad.l - 6, pad.t + 4);
  ctx.fillText(String(maxC), pad.l - 6, pad.t + 12);
  ctx.textAlign = "left";

  for (let i = 0; i < nbins; i++) {
    const x = pad.l + i * bw + gap / 2;
    const w = Math.max(2, bw - gap);
    const hAll = (allCounts[i] / maxC) * plotH;
    const hTag = (tagCounts[i] / maxC) * plotH;
    const yBase = pad.t + plotH;
    const inBrush = brushRange && i >= brushRange.i0 && i <= brushRange.i1;

    ctx.fillStyle = inBrush ? "#dce8f8" : "#eef2f7";
    ctx.beginPath(); roundRect(ctx, x, yBase - hAll, w, hAll, 3); ctx.fill();

    const isHover = state.hoverBin === i;
    if (hTag > 0) {
      ctx.fillStyle = inBrush ? (isHover ? "#004999" : "#005bb5") : (isHover ? "#0066cc" : "#0a84ff");
      ctx.beginPath(); roundRect(ctx, x, yBase - hTag, w, hTag, 3); ctx.fill();
    }
    if (inBrush) {
      ctx.strokeStyle = "#0a84ff"; ctx.lineWidth = 1.5;
      ctx.beginPath(); roundRect(ctx, x, yBase - Math.max(hAll, hTag, 2), w, Math.max(hAll, hTag, 2), 3); ctx.stroke();
    }
  }

  if (brushRange) {
    const x0 = pad.l + brushRange.i0 * bw;
    const x1 = pad.l + (brushRange.i1 + 1) * bw;
    ctx.fillStyle = "rgba(10,132,255,0.06)";
    ctx.fillRect(x0, pad.t, x1 - x0, plotH);
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = "#0a84ff";
    ctx.lineWidth = 2;
    ctx.strokeRect(x0 + 1, pad.t + 1, x1 - x0 - 2, plotH - 2);
    ctx.setLineDash([]);
    ctx.fillStyle = "#0a84ff";
    ctx.font = "bold 10px -apple-system,sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("刷选范围", (x0 + x1) / 2, pad.t + 12);
    ctx.textAlign = "left";
  }

  ctx.fillStyle = "#6e6e73"; ctx.font = "10px -apple-system,sans-serif";
  ctx.textAlign = "center";
  const tickStep = Math.max(1, Math.floor(nbins / 5));
  for (let i = 0; i < nbins; i += tickStep) {
    ctx.fillText(edges[i].toFixed(1), pad.l + i * bw + bw / 2, H - 22);
  }
  ctx.fillText(max.toFixed(1), pad.l + plotW, H - 22);
  ctx.textAlign = "left";
  ctx.fillStyle = "#1d1d1f"; ctx.font = "12px -apple-system,sans-serif";
  ctx.fillText(info.title || metric, pad.l, 14);

  $("dv-hist-stats").textContent = brushRange
    ? `刷选 ${formatBrushRange()} · μ ${mean} · med ${median}`
    : `μ ${mean} · med ${median}`;
  $("dv-hist-axis-hint").textContent = brushRange
    ? "虚线框 = 刷选范围（联动散点）；深蓝柱 = 框内 bin"
    : "拖拽柱形刷选 · 浅灰=全库 · 蓝色=Stage I 后";
}

function renderCategoryBars(dim) {
  const canvas = $("dv-hist-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const info = state.catalog?.categories?.[dim] || {};
  const counts = {};
  for (const c of tagFilteredClips()) {
    const k = clipCategory(c, dim);
    counts[k] = (counts[k] || 0) + 1;
  }
  const keys = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
  if (!keys.length) return;

  const pad = { l: 52, r: 16, t: 20, b: 52 };
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const maxC = Math.max(1, ...Object.values(counts));
  const gap = 6;
  const bw = plotW / keys.length;
  state.histLayout = { pad, plotW, plotH, bw, keys, counts, dim, kind: "cat" };

  ctx.strokeStyle = "#e0e0e5";
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + plotH);
  ctx.lineTo(pad.l + plotW, pad.t + plotH); ctx.stroke();

  keys.forEach((k, i) => {
    const x = pad.l + i * bw + gap / 2;
    const w = Math.max(4, bw - gap);
    const h = (counts[k] / maxC) * plotH;
    const sel = state.catBrush?.has(k);
    ctx.fillStyle = sel ? "#005bb5" : (state.hoverBin === i ? "#7eb8f7" : "#c9def8");
    ctx.beginPath(); roundRect(ctx, x, pad.t + plotH - h, w, h, 4); ctx.fill();
    if (sel) {
      ctx.strokeStyle = "#0a84ff"; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
      ctx.strokeRect(x - 1, pad.t + plotH - h - 1, w + 2, h + 2);
      ctx.setLineDash([]);
    }
    ctx.fillStyle = "#444"; ctx.font = "10px -apple-system,sans-serif";
    ctx.textAlign = "center";
    const label = k.length > 8 ? k.slice(0, 7) + "…" : k;
    ctx.fillText(label, x + w / 2, pad.t + plotH + 14);
    ctx.fillText(String(counts[k]), x + w / 2, pad.t + plotH + 28);
  });
  ctx.textAlign = "left";
  ctx.fillStyle = "#1d1d1f"; ctx.font = "12px -apple-system,sans-serif";
  ctx.fillText(info.title || dim, pad.l, 14);
  $("dv-hist-stats").textContent = state.catBrush?.size
    ? `刷选 ${formatBrushRange()}`
    : "";
  $("dv-hist-axis-hint").textContent = state.catBrush?.size
    ? "虚线框 = 已选类别（联动散点）"
    : "点击或拖拽柱形刷选散点";
}

// ------------------------------------------------------------------ scatter
let scatterPts = [];
let scatterBounds = null;

function resetScatterView(render = true) {
  state.scatterView = { scale: 1, panX: 0, panY: 0, dragging: false, dragMoved: false, lastX: 0, lastY: 0 };
  if (render) renderScatter();
}

function worldToScreen(x, y, W, H) {
  const b = scatterBounds;
  const sv = state.scatterView;
  const pad = 36;
  const sx = (W - 2 * pad) / Math.max(1e-6, b.maxX - b.minX);
  const sy = (H - 2 * pad) / Math.max(1e-6, b.maxY - b.minY);
  const cx = W / 2, cy = H / 2;
  let px = pad + (x - b.minX) * sx;
  let py = H - (pad + (y - b.minY) * sy);
  px = (px - cx) * sv.scale + cx + sv.panX;
  py = (py - cy) * sv.scale + cy + sv.panY;
  return { px, py };
}

function drawScatterNode(ctx, px, py, color, opts = {}) {
  const { hover, selected, subset, dimmed } = opts;
  const baseR = hover ? 7.5 : 6;
  ctx.save();
  ctx.globalAlpha = dimmed ? (hover ? 0.55 : 0.14) : 1;

  if (hover) {
    ctx.shadowBlur = 18;
    ctx.shadowColor = rgba(color, 0.55);
  } else if (subset && !dimmed) {
    ctx.shadowBlur = 10;
    ctx.shadowColor = rgba("#ff9f0a", 0.45);
  }

  // soft halo
  ctx.beginPath();
  ctx.arc(px, py, baseR + 3, 0, Math.PI * 2);
  ctx.fillStyle = rgba(color, hover ? 0.28 : 0.16);
  ctx.fill();

  // radial fill — lighter center like graph nodes
  const grad = ctx.createRadialGradient(px - baseR * 0.25, py - baseR * 0.3, 0, px, py, baseR);
  grad.addColorStop(0, rgba(color, 0.95));
  grad.addColorStop(0.55, color);
  grad.addColorStop(1, rgba(color, 0.82));
  ctx.beginPath();
  ctx.arc(px, py, baseR, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.shadowBlur = 0;
  ctx.strokeStyle = hover ? "#fff" : rgba("#fff", 0.88);
  ctx.lineWidth = hover ? 2.5 : 1.8;
  ctx.stroke();

  if (subset && !dimmed) {
    ctx.strokeStyle = "#ff9f0a";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px, py, baseR + 5, 0, Math.PI * 2);
    ctx.stroke();
  }
  if (selected) {
    ctx.strokeStyle = "#1d1d1f";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.arc(px, py, baseR + 1.5, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function updateScatterTooltip(clip, px, py, dimmed = false) {
  const tip = $("dv-scatter-tip");
  if (!tip) return;
  if (!clip) {
    tip.hidden = true;
    return;
  }
  tip.hidden = false;
  const m = clip.metrics || {};
  const tags = [];
  if (state.subsetIds.has(clip.clip_id)) tags.push("推荐");
  if (state.selected.has(clip.clip_id) && isManualSelectable(clip.clip_id)) tags.push("手动");
  if (dimmed) tags.push("刷选外");
  const suffix = tags.length ? ` · ${tags.join("/")}` : "";
  tip.textContent = `${clip.clip_id} · C ${m.complexity ?? "—"}${suffix}`;
  tip.style.left = `${px}px`;
  tip.style.top = `${py + 14}px`;
  tip.style.transform = "translate(-50%, 0)";
}

function hitScatterPoint(x, y, maxD2 = 576) {
  let best = null, bestD = maxD2;
  for (const p of scatterPts) {
    const d = (p.px - x) ** 2 + (p.py - y) ** 2;
    if (d < bestD) { bestD = d; best = p; }
  }
  return best;
}

function renderScatter() {
  const canvas = $("dv-scatter-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  scatterPts = [];
  const all = okClips().filter((c) => c.scatter);
  if (!all.length) {
    ctx.fillStyle = "#999"; ctx.fillText("无散点", 20, 30);
    updateScatterTooltip(null);
    return;
  }

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const c of all) {
    minX = Math.min(minX, c.scatter[0]); maxX = Math.max(maxX, c.scatter[0]);
    minY = Math.min(minY, c.scatter[1]); maxY = Math.max(maxY, c.scatter[1]);
  }
  scatterBounds = { minX, maxX, minY, maxY };
  const visIds = new Set(visibleClips().map((c) => c.clip_id));
  const colorMap = new Map();
  let hoverPt = null;

  for (const c of all) {
    const { px, py } = worldToScreen(c.scatter[0], c.scatter[1], W, H);
    scatterPts.push({ clip: c, px, py });
    const inVis = visIds.has(c.clip_id);
    const color = clusterColor(c.cluster_id, colorMap);
    const isHover = state.hoverClipId === c.clip_id;
    if (isHover) hoverPt = { clip: c, px, py };
    drawScatterNode(ctx, px, py, color, {
      hover: isHover,
      selected: state.selected.has(c.clip_id) && isManualSelectable(c.clip_id),
      subset: state.subsetIds.has(c.clip_id),
      dimmed: !inVis,
    });
  }

  updateScatterTooltip(hoverPt?.clip, hoverPt?.px, hoverPt?.py, hoverPt && !visIds.has(hoverPt.clip.clip_id));

  const leg = $("dv-legend");
  const clusters = [...colorMap.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  leg.innerHTML =
    `<span class="dv-legend-item"><i style="background:#ff9f0a"></i>推荐子集</span>`
    + `<span class="dv-legend-item"><i style="background:#1d1d1f"></i>手动补选</span>`
    + `<span class="dv-legend-item hint">淡色=刷选外</span>`
    + clusters.slice(0, 6).map(([k, col]) =>
      `<span class="dv-legend-item"><i style="background:${col}"></i>簇 ${k}</span>`).join("")
    + (clusters.length > 6 ? `<span class="dv-legend-item hint">+${clusters.length - 6}</span>` : "");
}

function renderClipList() {
  const vis = visibleClips();
  $("dv-list-count").textContent = `${vis.length}`;
  const box = $("dv-clip-list");
  box.innerHTML = "";
  for (const c of vis.slice(0, 60)) {
    const row = document.createElement("div");
    row.className = "dv-clip-row"
      + (state.selected.has(c.clip_id) && isManualSelectable(c.clip_id) ? " sel" : "")
      + (state.subsetIds.has(c.clip_id) ? " subset" : "");
    const m = c.metrics || {};
    row.innerHTML =
      `<span class="dv-cr-id">${c.clip_id}</span>`
      + `<span class="dv-cr-meta">${(m.s_phy ?? "—")} · ${(m.complexity ?? "—")}</span>`
      + `<button type="button" class="dv-cr-play">▶</button>`;
    row.onclick = (ev) => {
      if (ev.target.closest(".dv-cr-play")) {
        showDetail(c);
        previewClip(c);
      } else toggleSelect(c.clip_id);
    };
    box.appendChild(row);
  }
}

function renderOverview() {
  const s = state.summary;
  if (!s) return;
  $("dv-overview").innerHTML =
    `<div class="dv-stat-pill"><b>${s.num_ok}</b><span>clip</span></div>`
    + `<div class="dv-stat-pill"><b>${tagFilteredClips().length}</b><span>Stage I</span></div>`
    + `<div class="dv-stat-pill"><b>${visibleClips().length}</b><span>刷选</span></div>`
    + `<div class="dv-stat-pill accent"><b>${state.subsetIds.size}</b><span>推荐</span></div>`;
}

function renderChips() {
  const counts = state.summary?.tag_counts || {};
  const box = $("dv-chips");
  box.innerHTML = "";
  const groups = [
    ["质量", QUALITY_TAGS.filter((t) => counts[t])],
    ["动态", DYN_TAGS.filter((t) => counts[t])],
    ["其他", Object.keys(counts).filter((t) => !QUALITY_TAGS.includes(t) && !DYN_TAGS.includes(t))],
  ];
  for (const [label, tags] of groups) {
    if (!tags.length) continue;
    const hdr = document.createElement("div");
    hdr.className = "dv-chip-group-label";
    hdr.textContent = label;
    box.appendChild(hdr);
    for (const tag of tags) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "dv-chip" + (state.activeTags.has(tag) ? " on" : "");
      chip.innerHTML = `${tag}<span class="dv-chip-n">${counts[tag]}</span>`;
      chip.onclick = () => {
        state.activeTags.has(tag) ? state.activeTags.delete(tag) : state.activeTags.add(tag);
        recomputeSubset(); renderAll();
      };
      box.appendChild(chip);
    }
  }
  renderTagInfo();
}

function renderSelbar() {
  const sub = state.subsetIds.size;
  const manual = manualSelectedIds().length;
  $("dv-selbar").textContent =
    `推荐 ${sub} 个` + (manual ? ` · 手动补选 ${manual} 个（不含推荐）` : "");
}

function renderAll() {
  renderOverview();
  renderChips();
  renderDistribution();
  renderScatter();
  renderClipList();
  renderSelbar();
  updateKindBadge();
}

function showDetail(clip) {
  const m = clip.metrics || {};
  $("dv-clip-detail").textContent =
    `${clip.clip_id} · S_phy ${m.s_phy ?? "—"} · C(x) ${m.complexity ?? "—"}`;
}

function toggleSelect(id) {
  if (!isManualSelectable(id)) {
    bridge().toast?.("该 clip 已在推荐子集中，无需手动补选", false);
    return;
  }
  state.selected.has(id) ? state.selected.delete(id) : state.selected.add(id);
  renderScatter(); renderClipList(); renderSelbar();
}

// ------------------------------------------------------------------ hist interaction
function setupHistInteraction() {
  const canvas = $("dv-hist-canvas");

  canvas.addEventListener("mousemove", (ev) => {
    const lay = state.histLayout;
    if (!lay) return;
    const { x } = canvasXY(ev, canvas);
    const bin = binAtX(x, lay);
    if (bin !== state.hoverBin) {
      state.hoverBin = bin;
      renderDistribution();
    }
    if (state.histDrag.active && bin >= 0) {
      applyBinBrush(lay, state.histDrag.startBin, bin);
    }
  });

  canvas.addEventListener("mousedown", (ev) => {
    const lay = state.histLayout;
    if (!lay) return;
    const bin = binAtX(canvasXY(ev, canvas).x, lay);
    if (bin < 0) return;
    state.histDrag = { active: true, startBin: bin };
    if (!ev.shiftKey) applyBinBrush(lay, bin, bin);
  });

  window.addEventListener("mouseup", () => {
    state.histDrag.active = false;
  });

  canvas.addEventListener("mouseleave", () => {
    if (state.hoverBin >= 0) { state.hoverBin = -1; renderDistribution(); }
  });
}

function setupScatterNav() {
  const canvas = $("dv-scatter-canvas");
  canvas.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    state.scatterView.scale = Math.max(0.25, Math.min(10,
      state.scatterView.scale * (ev.deltaY > 0 ? 0.9 : 1.1)));
    renderScatter();
  }, { passive: false });

  canvas.addEventListener("mousemove", (ev) => {
    if (state.scatterView.dragging) return;
    const { x, y } = canvasXY(ev, canvas);
    const hit = hitScatterPoint(x, y);
    const id = hit ? hit.clip.clip_id : null;
    if (id !== state.hoverClipId) {
      state.hoverClipId = id;
      renderScatter();
    }
    canvas.style.cursor = id ? "pointer" : "grab";
  });

  canvas.addEventListener("mouseleave", () => {
    if (state.hoverClipId) {
      state.hoverClipId = null;
      renderScatter();
    }
    canvas.style.cursor = "grab";
  });

  canvas.addEventListener("mousedown", (ev) => {
    if (ev.button !== 0) return;
    state.scatterView.dragging = true;
    state.scatterView.dragMoved = false;
    state.scatterView.lastX = ev.clientX;
    state.scatterView.lastY = ev.clientY;
    canvas.style.cursor = "grabbing";
  });

  window.addEventListener("mousemove", (ev) => {
    if (!state.scatterView.dragging) return;
    const dx = ev.clientX - state.scatterView.lastX;
    const dy = ev.clientY - state.scatterView.lastY;
    if (Math.abs(dx) + Math.abs(dy) > 4) state.scatterView.dragMoved = true;
    state.scatterView.panX += dx;
    state.scatterView.panY += dy;
    state.scatterView.lastX = ev.clientX;
    state.scatterView.lastY = ev.clientY;
    renderScatter();
  });

  window.addEventListener("mouseup", () => {
    if (state.scatterView.dragging) {
      state.scatterView.dragging = false;
      canvas.style.cursor = state.hoverClipId ? "pointer" : "grab";
    }
  });

  canvas.addEventListener("click", (ev) => {
    if (state.scatterView.dragMoved) return;
    const { x, y } = canvasXY(ev, canvas);
    const hit = hitScatterPoint(x, y);
    if (!hit) return;
    const id = hit.clip.clip_id;
    showDetail(hit.clip);
    previewClip(hit.clip);
    if (!isManualSelectable(id)) return;
    if (ev.shiftKey) {
      toggleSelect(id);
      return;
    }
    if (!state.selected.has(id)) {
      state.selected.clear();
      state.selected.add(id);
    } else if (state.selected.size === 1) {
      state.selected.delete(id);
    } else {
      state.selected.clear();
      state.selected.add(id);
    }
    renderScatter();
    renderClipList();
    renderSelbar();
  });
}

function getUserSourceRoot() {
  return ($("dv-user-source-root")?.value || localStorage.getItem(DV_USER_ROOT_KEY) || "").trim();
}

function setUserSourceRoot(v) {
  const val = String(v || "").trim();
  const inp = $("dv-user-source-root");
  if (inp) inp.value = val;
  if (val) localStorage.setItem(DV_USER_ROOT_KEY, val);
  else localStorage.removeItem(DV_USER_ROOT_KEY);
}

function syncUserRootField() {
  const inp = $("dv-user-source-root");
  if (inp && !inp.value.trim()) {
    const saved = localStorage.getItem(DV_USER_ROOT_KEY);
    if (saved) inp.value = saved;
  }
}

function exportManifestPayload(ids, extra = {}) {
  return {
    clips: state.clips,
    ids,
    analyze_source: state.analyzeSourceRoot || state.analyzeSource || "",
    user_source_root: getUserSourceRoot(),
    format: "json",
    ...extra,
  };
}

function looksLikeTempSourcePath(p) {
  return /\/tmp\/|hhtools_web_up/i.test(String(p || ""));
}

function needsUserSourceRoot(ids) {
  return state.clips.some((c) => ids.includes(c.clip_id) && looksLikeTempSourcePath(c.source_path));
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.setTimeout(() => URL.revokeObjectURL(url), 2000);
}

async function exportManifest(ids, filename) {
  const { toast } = bridge();
  if (!ids.length) { toast("没有可导出的 clip", true); return false; }
  const needsRoot = needsUserSourceRoot(ids);
  const userRoot = getUserSourceRoot();
  if (needsRoot && !userRoot) {
    toast("请先填写「本地数据目录」（如 /home/motions），manifest 才能写入真实路径", true);
    $("dv-user-source-root")?.focus();
    return false;
  }
  const payload = exportManifestPayload(ids);
  const r = await fetch("/api/dataset/export_manifest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error("导出失败");
  const blob = await r.blob();
  downloadBlob(blob, filename);
  return true;
}

function addHumanToBasket() {
  if ($("dv-human-basket")?.disabled) return;
  const ids = new Set(exportTargetIds());
  const clips = okClips().filter((c) => ids.has(c.clip_id) && c.source_kind !== "robot");
  if (!clips.length) { bridge().toast?.("没有可加入的人体 clip", true); return; }
  bridge().addToBasket?.(clips.map(entryFromClip));
}

async function exportRobotData() {
  if ($("dv-export-robot")?.disabled) return;
  const ids = exportTargetIds().filter((id) => {
    const c = state.clips.find((x) => x.clip_id === id);
    return c?.source_kind === "robot";
  });
  if (!ids.length) {
    bridge().toast?.("没有可导出的机器人 clip", true);
    return;
  }
  const packFiles = $("dv-robot-export-files")?.checked !== false;
  const { toast } = bridge();
  $("dv-export-robot").disabled = true;
  try {
    if (!packFiles) {
      const ok = await exportManifest(ids, "robot_subset_manifest.json");
      if (ok) toast(`已导出 ${ids.length} 条机器人 clip 清单 (JSON)`);
      return;
    }
    const r = await fetch("/api/dataset/export_robot_zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(exportManifestPayload(ids)),
    });
    if (!r.ok) {
      let msg = "打包失败";
      try {
        const j = await r.json();
        msg = j.detail || msg;
      } catch {
        msg = (await r.text()) || msg;
      }
      throw new Error(msg);
    }
    const blob = await r.blob();
    const cd = r.headers.get("Content-Disposition") || "";
    const m = cd.match(/filename="?([^";]+)"?/);
    const filename = m ? m[1] : "robot_subset_export.zip";
    downloadBlob(blob, filename);
    toast(`已打包 ${ids.length} 个机器人 clip 文件夹`);
  } catch (e) {
    toast(e.message, true);
  } finally {
    updateKindBadge();
  }
}

function setupDropzone() {
  const el = $("dv-dropzone");
  if (!el) return;
  ["dragenter", "dragover"].forEach((ev) =>
    el.addEventListener(ev, (e) => { e.preventDefault(); el.classList.add("hover"); }));
  ["dragleave", "drop"].forEach((ev) =>
    el.addEventListener(ev, (e) => { e.preventDefault(); el.classList.remove("hover"); }));
  el.addEventListener("drop", async (e) => {
    const files = [];
    const walks = [];
    for (const it of e.dataTransfer.items) {
      const entry = it.webkitGetAsEntry?.();
      if (entry) walks.push(walkEntry(entry, files));
      else if (it.getAsFile) files.push(it.getAsFile());
    }
    await Promise.all(walks);
    if (files.length) ingestDroppedFiles(files);
  });
}

function bind() {
  loadCatalog();
  syncUserRootField();
  updateKindBadge();
  setupDropzone();
  setupHistInteraction();
  setupScatterNav();
  $("dv-pick-folder")?.addEventListener("click", pickFolder);
  $("dv-clear-upload")?.addEventListener("click", clearUploadBasket);
  $("dv-analyze")?.addEventListener("click", runAnalysis);
  $("dv-clear-tags")?.addEventListener("click", () => {
    state.activeTags.clear(); recomputeSubset(); renderAll();
  });
  $("dv-clear-brush")?.addEventListener("click", () => {
    state.histBrush = null; state.catBrush = null; recomputeSubset(); renderAll();
  });
  document.querySelectorAll('input[name="dv-tagmode"]').forEach((r) => {
    r.addEventListener("change", () => {
      state.tagMode = document.querySelector('input[name="dv-tagmode"]:checked').value;
      recomputeSubset(); renderAll();
    });
  });
  $("dv-view-dim")?.addEventListener("change", (e) => {
    state.viewDim = e.target.value;
    state.histBrush = null; state.catBrush = null;
    recomputeSubset(); renderAll();
  });
  $("dv-scatter-reset")?.addEventListener("click", () => resetScatterView());
  $("dv-subset-ratio")?.addEventListener("input", (e) => {
    $("dv-subset-pct").textContent = e.target.value + "%";
    scheduleSubset();
  });
  $("dv-subset-alpha")?.addEventListener("input", (e) => {
    $("dv-subset-alpha-val").textContent = (parseInt(e.target.value, 10) / 100).toFixed(2);
    scheduleSubset();
  });
  $("dv-human-basket")?.addEventListener("click", addHumanToBasket);
  $("dv-export-robot")?.addEventListener("click", exportRobotData);
  $("dv-robot-export-files")?.addEventListener("change", syncRobotExportLabel);
  $("dv-user-source-root")?.addEventListener("change", (e) => {
    setUserSourceRoot(e.target.value);
  });
  $("dv-user-source-root")?.addEventListener("input", (e) => {
    setUserSourceRoot(e.target.value);
  });
  $("dv-export-json")?.addEventListener("click", async () => {
    try {
      await exportManifest(exportTargetIds(), "dataset_manifest.json");
    } catch (e) { bridge().toast?.(e.message, true); }
  });
  $("dv-clear-sel")?.addEventListener("click", () => {
    state.selected.clear(); renderScatter(); renderClipList(); renderSelbar();
  });
  $("dv-robot-select")?.addEventListener("change", (e) => {
    state.previewRobot = e.target.value;
  });
  document.querySelector('.nav-item[data-panel="dataset-viz"]')?.addEventListener("click", () => {
    const root = bridge().getLibrarySourceRoot?.();
    if (root && $("dv-drop-hint")) $("dv-drop-hint").textContent = `留空 = ${root}`;
  });
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bind);
else bind();
