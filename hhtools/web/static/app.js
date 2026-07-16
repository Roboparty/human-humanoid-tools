// hhtools web — three.js front-end.
// All heavy compute happens on the FastAPI backend; this file renders + drives UX.


/** Parse a positive FPS from a number input, or ``null`` to mean “use default”. */
function parseOptionalFps(el) {
  if (!el) return null;
  const v = parseFloat(el.value);
  return v > 0 && Number.isFinite(v) ? v : null;
}

function footClampAntiPenetrationEnabled() {
  return !!document.getElementById("rt-foot-clamp-anti-penetration")?.checked;
}

function wireSyncedCheckboxes(ids) {
  const els = ids.map((id) => document.getElementById(id)).filter(Boolean);
  if (els.length < 2) return;
  let syncing = false;
  for (const el of els) {
    el.addEventListener("change", () => {
      if (syncing) return;
      syncing = true;
      const { checked } = el;
      for (const other of els) {
        if (other !== el) other.checked = checked;
      }
      syncing = false;
    });
  }
}

wireSyncedCheckboxes([
  "rt-foot-clamp-anti-penetration",
  "batch-foot-clamp-anti-penetration",
]);

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Playback timeline when long clips are downsampled for the browser payload. */
function effectivePlaybackDuration(payload) {
  if (payload == null) return 1;
  if (payload.playback_duration != null && Number.isFinite(payload.playback_duration)) {
    return Math.max(0.1, payload.playback_duration);
  }
  const nPlay = payload.playback_frames
    ?? payload.positions?.length
    ?? payload.frames?.length
    ?? payload.num_frames_total;
  const nTotal = payload.num_frames_total ?? nPlay;
  const fps = payload.framerate || payload.sample_rate || 30;
  // Always span the FULL clip duration — downsampled frames are interpolated
  // across it, so never shorten the timeline to the downsampled frame count
  // (that made long, heavily-downsampled clips play several times too fast).
  const d = payload.duration;
  if (d != null && d > 0) return Math.max(0.1, d);
  return Math.max(0.1, (nTotal - 1) / fps);
}

function isPlaybackPreview(payload) {
  if (!payload) return false;
  const nPlay = payload.playback_frames
    ?? payload.positions?.length
    ?? payload.frames?.length
    ?? 0;
  const nTotal = payload.num_frames_total ?? nPlay;
  return nTotal > nPlay && nPlay > 0;
}

/**
 * Downsampled clips spread sparse keys across the full timeline; linear blend
 * between keys that are far apart in the source cuts corners (LAFAN 折返 → 滑步).
 */
function resolvePlaybackFrame(frameIndices, fi, max) {
  const f0 = Math.min(max, Math.floor(fi));
  const t = fi - f0;
  if (t <= 1e-5 || f0 >= max) return { ia: f0, ib: f0, t: 0 };
  const ib = f0 + 1;
  const gap = frameIndices && frameIndices.length > ib
    ? frameIndices[ib] - frameIndices[f0]
    : 1;
  if (gap > 1) {
    const pick = t >= 0.5 ? ib : f0;
    return { ia: pick, ib: pick, t: 0 };
  }
  return { ia: f0, ib, t };
}

function updateRetargetFpsPlaceholder() {
  const inp = document.getElementById("rt-retarget-fps");
  if (!inp) return;
  const src = state.motion?.framerate;
  inp.placeholder = src ? `留空 = 原始 ${src.toFixed(0)} fps` : "留空 = 动作原始帧率";
}

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { initTutorial } from "./tutorial.js";

// ----------------------------------------------------------------- API helpers
// FastAPI's `detail` can be a string OR (for 422 validation errors) an array of
// objects.  Flatten whatever we get into a human-readable string so the UI never
// shows the useless "[object Object]".
async function httpError(r) {
  let detail;
  try {
    detail = (await r.json()).detail;
  } catch {
    detail = null;
  }
  let msg;
  if (typeof detail === "string") msg = detail;
  else if (Array.isArray(detail)) msg = detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  else if (detail && typeof detail === "object") msg = detail.msg || JSON.stringify(detail);
  return new Error(msg || `${r.status} ${r.statusText}`);
}

const API = {
  async get(url) {
    const r = await fetch(url);
    if (!r.ok) throw await httpError(r);
    return r.json();
  },
  async post(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw await httpError(r);
    return r.json();
  },
  async upload(url, files, { profile, name } = {}) {
    const fd = new FormData();
    for (const f of files) fd.append("files", f, f._relpath || f.name);
    const qs = [];
    if (profile) qs.push(`profile=${encodeURIComponent(profile)}`);
    if (name) qs.push(`name=${encodeURIComponent(name)}`);
    const u = qs.length ? `${url}?${qs.join("&")}` : url;
    const r = await fetch(u, { method: "POST", body: fd });
    if (!r.ok) throw await httpError(r);
    return r.json();
  },
  async delete(url) {
    const r = await fetch(url, { method: "DELETE" });
    if (!r.ok) throw await httpError(r);
    return r.json();
  },
};

/** Trigger a file save into the browser's default download folder. */
async function triggerBrowserDownload(url, filename) {
  const r = await fetch(url);
  if (!r.ok) throw await httpError(r);
  const blob = await r.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename || "download";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

// Stop the browser from navigating to / downloading a file when a drop misses
// a dropzone (the default behaviour the user hit).
["dragover", "drop"].forEach((ev) =>
  window.addEventListener(ev, (e) => { e.preventDefault(); }, false)
);

const TOAST_MS = 3200;
const TOAST_ERR_EXTRA_MS = 5000;

function toast(msg, isErr = false) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = isErr ? "show err" : "show";
  clearTimeout(t._timer);
  const hideMs = isErr ? TOAST_MS + TOAST_ERR_EXTRA_MS : TOAST_MS;
  t._timer = setTimeout(() => (t.className = isErr ? "err" : ""), hideMs);
}

// ----------------------------------------------------------------- loading bar
function fmtBytes(n) {
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
}

function showLoading(label) {
  const o = document.getElementById("load-overlay");
  if (!o) return;
  document.getElementById("load-label").textContent = label || "加载中…";
  document.getElementById("load-sub").textContent = "";
  document.getElementById("load-bar").style.width = "0%";
  o.classList.remove("hidden");
  o.classList.add("indet"); // server still computing → animated sweep
}

/** ``frac`` in [0,1] for a determinate bar, or ``null`` for indeterminate. */
function setLoadingProgress(frac, sub) {
  const o = document.getElementById("load-overlay");
  if (!o) return;
  const bar = document.getElementById("load-bar");
  if (frac == null) {
    o.classList.add("indet");
  } else {
    o.classList.remove("indet");
    bar.style.width = `${Math.max(2, Math.min(100, frac * 100)).toFixed(0)}%`;
  }
  if (sub != null) document.getElementById("load-sub").textContent = sub;
}

function hideLoading() {
  const o = document.getElementById("load-overlay");
  if (!o) return;
  o.classList.add("hidden");
  o.classList.remove("indet");
}

// Read a (large) JSON response as a stream so the load bar reflects real
// download progress.  The server computes FK / bakes the SMPL mesh before the
// first byte, so `onProgress(null, …)` (indeterminate) covers that wait, then
// the determinate bar tracks the payload transfer — the part that actually
// scales with clip length.
async function readJsonStream(r, onProgress) {
  const total = Number(r.headers.get("Content-Length") || 0);
  if (!r.body || !total) {
    if (onProgress) onProgress(null, 0, 0);
    return r.json();
  }
  const reader = r.body.getReader();
  let received = 0;
  const chunks = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (onProgress) onProgress(received / total, received, total);
  }
  const all = new Uint8Array(received);
  let pos = 0;
  for (const c of chunks) { all.set(c, pos); pos += c.length; }
  return JSON.parse(new TextDecoder("utf-8").decode(all));
}

async function postJsonWithProgress(url, body, onProgress) {
  if (onProgress) onProgress(null, 0, 0);
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw await httpError(r);
  return readJsonStream(r, onProgress);
}

async function uploadWithProgress(url, files, { profile } = {}, onProgress) {
  const fd = new FormData();
  for (const f of files) fd.append("files", f, f._relpath || f.name);
  const qs = [];
  if (profile) qs.push(`profile=${encodeURIComponent(profile)}`);
  const u = qs.length ? `${url}?${qs.join("&")}` : url;
  if (onProgress) onProgress(null, 0, 0);
  const r = await fetch(u, { method: "POST", body: fd });
  if (!r.ok) throw await httpError(r);
  return readJsonStream(r, onProgress);
}

/** Upload files with real byte progress, then return the JSON body (``{job_id}``). */
function uploadFilesXHR(url, files, { profile, appendTo, libraryFolderLabel } = {}, onUploadProgress) {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    for (const f of files) fd.append("files", f, f._relpath || f.name);
    const qs = new URLSearchParams();
    if (profile) qs.set("profile", profile);
    if (appendTo) qs.set("append_to", appendTo);
    if (libraryFolderLabel) qs.set("library_folder_label", libraryFolderLabel);
    const q = qs.toString() ? `?${qs.toString()}` : "";
    const xhr = new XMLHttpRequest();
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onUploadProgress) {
        onUploadProgress(e.loaded / e.total, e.loaded, e.total);
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText)); }
        catch (err) { reject(err); }
        return;
      }
      reject(new Error(xhr.responseText || `upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("upload failed"));
    xhr.open("POST", url + q);
    xhr.send(fd);
  });
}

function formatJobProgress(job, prefix = "") {
  const pct = Math.round(Math.max(0, Math.min(100, (job.progress || 0) * 100)));
  const msg = job.message || "处理中…";
  return `${prefix}${msg} (${pct}%)`;
}

async function waitMotionJob(jobId, onProgress, { uploadFrac = 0 } = {}) {
  while (true) {
    const j = await API.get(`/api/job/${jobId}`);
    if (onProgress) {
      const frac = uploadFrac + (j.progress || 0) * (1 - uploadFrac);
      onProgress(frac, formatJobProgress(j));
    }
    if (j.status === "done") {
      if (!j.result) throw new Error(j.error || "motion load failed");
      return j.result;
    }
    if (j.status === "error") throw new Error(j.error || "motion load failed");
    await new Promise((r) => setTimeout(r, 350));
  }
}

// ----------------------------------------------------------------- 3D scene
const canvas = document.getElementById("three-canvas");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 200);
camera.position.set(2.6, 1.9, 3.2);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.target.set(0, 0.9, 0);
orbit.enableDamping = true;
orbit.dampingFactor = 0.08;
orbit.zoomSpeed = 0.028;
orbit.zoomToCursor = true;
orbit.screenSpacePanning = true;
// OrbitControls uses pow(0.95, zoomSpeed*deltaY) — one wheel notch (±100) jumps ~5×
// at default speeds.  Use linear dolly steps for continuous zoom instead.
orbit.enableZoom = false;
const _smoothZoomOffset = new THREE.Vector3();
function smoothOrbitWheel(event) {
  if (!orbit.enabled) return;
  let delta = event.deltaY;
  if (event.deltaMode === 1) delta *= 16;
  else if (event.deltaMode === 2) delta *= 400;
  const step = THREE.MathUtils.clamp(-delta / 120, -2.5, 2.5);
  const scale = Math.pow(0.968, step);
  _smoothZoomOffset.copy(camera.position).sub(orbit.target);
  const dist = _smoothZoomOffset.length();
  if (dist < 1e-6) return;
  const next = THREE.MathUtils.clamp(dist * scale, orbit.minDistance, orbit.maxDistance);
  _smoothZoomOffset.setLength(next);
  camera.position.copy(orbit.target).add(_smoothZoomOffset);
  orbit.update();
  _orbitManualUntil = performance.now() + 2800;
  event.preventDefault();
}
renderer.domElement.addEventListener("wheel", smoothOrbitWheel, { passive: false });

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 1.35));
const key = new THREE.DirectionalLight(0xffffff, 1.5);
key.position.set(3, 6, 4);
scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.85);
fill.position.set(-3, 4, -2);
scene.add(fill);

// World group: hhtools is Z-up; rotate so Z maps to three.js Y (up).
const world = new THREE.Group();
world.rotation.x = -Math.PI / 2;
scene.add(world);

// Spatial axes in the motion frame (X=red, Y=green, Z=blue in hhtools Z-up).
const axes = new THREE.AxesHelper(1.2);
world.add(axes);

// Environment (terrain + interaction objects) lives in its own group so it
// stays visible regardless of which figure (skeleton / mesh / robot) is shown.
const env = new THREE.Group();
world.add(env);
const scaledEnvGroup = new THREE.Group();
world.add(scaledEnvGroup);

// Triangulated heightfield mesh (matches Viser TerrainHeightfieldRenderer).
function buildTerrainMesh(t) {
  if (!t?.vertices?.length || !t?.faces?.length) return null;
  const pos = new Float32Array(t.vertices.length * 3);
  for (let i = 0; i < t.vertices.length; i++) {
    pos[i * 3] = t.vertices[i][0];
    pos[i * 3 + 1] = t.vertices[i][1];
    pos[i * 3 + 2] = t.vertices[i][2];
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setIndex(t.faces.flat());
  geo.computeVertexNormals();
  return new THREE.Mesh(
    geo,
    // flatShading keeps stair risers looking like sharp steps instead of
    // smooth-shaded ramps; the user reported stairs rendering as slopes.
    new THREE.MeshStandardMaterial({
      color: 0x9a9aa0, roughness: 0.95, side: THREE.DoubleSide, flatShading: true,
    })
  );
}

// Ground grid (in three.js Y-up space, so add outside world).
const grid = new THREE.GridHelper(20, 40, 0x99a0ab, 0xd2d6dd);
grid.material.opacity = 0.35;
grid.material.transparent = true;
scene.add(grid);

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (w === 0 || h === 0) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
new ResizeObserver(resize).observe(document.getElementById("stage"));

// ----------------------------------------------------------------- render loop
const clock = new THREE.Clock();
const _camFocus = new THREE.Vector3();
const _defaultCamTarget = new THREE.Vector3(0, 0.9, 0);
const _defaultCamOffset = new THREE.Vector3(2.6, 1.0, 3.2);
const _viewFocusBox = new THREE.Box3();
const _viewFocusTmp = new THREE.Box3();
let _orbitManualUntil = 0;
orbit.addEventListener("start", () => { _orbitManualUntil = performance.now() + 2800; });
orbit.addEventListener("end", () => { _orbitManualUntil = performance.now() + 2800; });

function getViewFocus(out = new THREE.Vector3()) {
  const candidates = [
    robot.links?.length ? robot.group : null,
    scaledSkel.joints ? scaledSkel.group : null,
    skel.joints ? skel.group : null,
    mesh.ready ? mesh.group : null,
    env.children.length ? env : null,
    scaledEnvGroup.children.length ? scaledEnvGroup : null,
  ].filter(Boolean);
  let has = false;
  for (const g of candidates) {
    _viewFocusTmp.setFromObject(g);
    if (_viewFocusTmp.isEmpty()) continue;
    if (!has) {
      _viewFocusBox.copy(_viewFocusTmp);
      has = true;
    } else {
      _viewFocusBox.union(_viewFocusTmp);
    }
    if (g === robot.group) break;
  }
  if (!has) {
    out.copy(_defaultCamTarget);
    return out;
  }
  _viewFocusBox.getCenter(out);
  return out;
}

function resetDefaultView() {
  focusRobotView({ resetOffset: true });
}

function calibRobotGroup() {
  return r2r.calibrating ? r2rTgt.group : robot.group;
}

/** Frame robot (+ reference skeleton during calibration) with sane orbit limits. */
function focusRobotView({ resetOffset = false } = {}) {
  const focusGroups = [calibRobotGroup()];
  if ((state.calibrationMode || r2r.calibrating) && refSkel.group.visible) {
    focusGroups.push(refSkel.group);
  }
  let has = false;
  for (const g of focusGroups) {
    if (!g?.visible) continue;
    _viewFocusTmp.setFromObject(g);
    if (_viewFocusTmp.isEmpty()) continue;
    if (!has) {
      _viewFocusBox.copy(_viewFocusTmp);
      has = true;
    } else {
      _viewFocusBox.union(_viewFocusTmp);
    }
  }
  if (!has) {
    getViewFocus(_camFocus);
    orbit.target.copy(_camFocus);
    if (resetOffset) camera.position.copy(_camFocus).add(_defaultCamOffset);
    orbit.update();
    _orbitManualUntil = performance.now() + 2800;
    return;
  }
  _viewFocusBox.getCenter(_camFocus);
  orbit.target.copy(_camFocus);
  if (resetOffset) {
    const size = _viewFocusBox.getSize(new THREE.Vector3());
    const span = Math.max(0.55, size.length());
    const dist = Math.max(1.35, span * 0.9);
    camera.position.copy(_camFocus).add(
      new THREE.Vector3(dist * 0.58, dist * 0.44, dist * 0.68),
    );
  }
  orbit.update();
  _orbitManualUntil = performance.now() + 2800;
}

/** Orbit distance limits scaled to the visible robot (calibration zoom range). */
function calibOrbitDistanceLimits() {
  let has = false;
  for (const g of [calibRobotGroup(), refSkel.group.visible ? refSkel.group : null]) {
    if (!g) continue;
    _viewFocusTmp.setFromObject(g);
    if (_viewFocusTmp.isEmpty()) continue;
    if (!has) {
      _viewFocusBox.copy(_viewFocusTmp);
      has = true;
    } else {
      _viewFocusBox.union(_viewFocusTmp);
    }
  }
  const span = has ? Math.max(0.75, _viewFocusBox.getSize(new THREE.Vector3()).length()) : 1.6;
  return {
    minDistance: Math.max(0.28, span * 0.12),
    maxDistance: Math.max(span * 6, 18),
  };
}

function applyCalibOrbitLimits({ snapCamera = false } = {}) {
  const lim = calibOrbitDistanceLimits();
  orbit.minDistance = lim.minDistance;
  orbit.maxDistance = lim.maxDistance;
  if (!snapCamera) return;
  const dist = camera.position.distanceTo(orbit.target);
  if (dist < lim.minDistance || dist > lim.maxDistance) {
    const dir = camera.position.clone().sub(orbit.target);
    if (dir.lengthSq() < 1e-8) dir.set(0.58, 0.44, 0.68);
    dir.normalize().multiplyScalar(Math.min(lim.maxDistance, Math.max(lim.minDistance, dist)));
    camera.position.copy(orbit.target).add(dir);
    orbit.update();
  }
}

function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  player.update(dt);
  // Follow the retargeted robot in world space; pause while the user orbits.
  // On loop wrap the robot teleports (start ≠ end).  Always hard-snap the
  // camera with that wrap — even during the post-orbit "manual" grace period —
  // otherwise the robot flies across the viewport while the camera stays put.
  //
  // Target and camera must translate by the *same* delta.  Lerping only
  // ``orbit.target`` leaves the eye behind; at walk speed the lag settles
  // near the hard-snap threshold (~0.5 m) and the view stutter-snaps every
  // few frames.
  const loopSnap = player._justLooped;
  player._justLooped = false;
  if (
    !state.calibrationMode &&
    robot.group.visible && robot.trajectory &&
    (loopSnap || performance.now() > _orbitManualUntil)
  ) {
    robot.group.getWorldPosition(_camFocus);
    const dx = _camFocus.x - orbit.target.x;
    const dy = _camFocus.y - orbit.target.y;
    const dz = _camFocus.z - orbit.target.z;
    const jumpSq = dx * dx + dy * dy + dz * dz;
    const a = (loopSnap || jumpSq > 0.25) ? 1 : Math.min(1, dt * 12);
    const ox = dx * a;
    const oy = dy * a;
    const oz = dz * a;
    orbit.target.x += ox;
    orbit.target.y += oy;
    orbit.target.z += oz;
    camera.position.x += ox;
    camera.position.y += oy;
    camera.position.z += oz;
  }
  if ((state.calibrationMode || r2r.calibrating) && calibManip.active && !calibManip._hudCardDrag) {
    calibManip._positionTags();
  }
  orbit.update();
  renderer.render(scene, camera);
}
resize();
// NOTE: the render loop is started at the very bottom of this module, after
// `player` is defined — calling animate() here would hit the const TDZ.

// =================================================================  SKELETON
class SkeletonView {
  constructor() {
    this.group = new THREE.Group();
    world.add(this.group);
    this.joints = null; // (F, J, 3)
    this.parents = null;
    this.spheres = [];
    this.lineGeom = null;
    this.lines = null;
    this.frameIndices = null;
    this.color = 0x0a84ff;
  }
  clear() {
    while (this.group.children.length) this.group.remove(this.group.children[0]);
    this.spheres = [];
    this.joints = null;
  }
  load(motion, color = 0x0a84ff) {
    this.clear();
    this.color = color;
    this.joints = motion.positions; // (F, J, 3)
    this.parents = motion.parent_indices;
    this.exclude = new Set(motion.exclude_joint_indices || []);
    this.frameIndices = motion.frame_indices;
    this.clipDuration = effectivePlaybackDuration(motion);
    const J = this.parents.length;
    const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.1 });
    const sphereGeo = new THREE.SphereGeometry(0.028, 12, 12);
    for (let j = 0; j < J; j++) {
      const s = new THREE.Mesh(sphereGeo, mat);
      if (this.exclude.has(j)) s.visible = false;
      this.group.add(s);
      this.spheres.push(s);
    }
    let segCount = 0;
    for (let j = 0; j < J; j++) {
      const p = this.parents[j];
      if (p < 0 || this.exclude.has(j) || this.exclude.has(p)) continue;
      segCount++;
    }
    const positions = new Float32Array(segCount * 2 * 3);
    this.lineGeom = new THREE.BufferGeometry();
    this.lineGeom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    this.lines = new THREE.LineSegments(
      this.lineGeom,
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 })
    );
    this.group.add(this.lines);
    this.setFrame(0);
  }
  get numFrames() {
    return this.joints ? this.joints.length : 0;
  }
  setFrame(f) {
    this.setFrameFrac(f);
  }
  setFrameFrac(fi) {
    if (!this.joints) return;
    const max = this.joints.length - 1;
    const { ia, ib, t } = resolvePlaybackFrame(this.frameIndices, fi, max);
    const fr = this.joints[ia];
    if (!fr) return;
    const blend = t > 1e-5 && ia !== ib;
    const nxt = blend ? this.joints[ib] : null;
    for (let j = 0; j < this.spheres.length; j++) {
      if (blend) {
        this.spheres[j].position.set(
          fr[j][0] + (nxt[j][0] - fr[j][0]) * t,
          fr[j][1] + (nxt[j][1] - fr[j][1]) * t,
          fr[j][2] + (nxt[j][2] - fr[j][2]) * t,
        );
      } else {
        this.spheres[j].position.set(fr[j][0], fr[j][1], fr[j][2]);
      }
    }
    const arr = this.lineGeom.attributes.position.array;
    let k = 0;
    for (let j = 0; j < this.parents.length; j++) {
      const p = this.parents[j];
      if (p < 0 || this.exclude.has(j) || this.exclude.has(p)) continue;
      if (blend) {
        arr[k++] = fr[j][0] + (nxt[j][0] - fr[j][0]) * t;
        arr[k++] = fr[j][1] + (nxt[j][1] - fr[j][1]) * t;
        arr[k++] = fr[j][2] + (nxt[j][2] - fr[j][2]) * t;
        arr[k++] = fr[p][0] + (nxt[p][0] - fr[p][0]) * t;
        arr[k++] = fr[p][1] + (nxt[p][1] - fr[p][1]) * t;
        arr[k++] = fr[p][2] + (nxt[p][2] - fr[p][2]) * t;
      } else {
        arr[k++] = fr[j][0]; arr[k++] = fr[j][1]; arr[k++] = fr[j][2];
        arr[k++] = fr[p][0]; arr[k++] = fr[p][1]; arr[k++] = fr[p][2];
      }
    }
    this.lineGeom.attributes.position.needsUpdate = true;
  }
}

// Blue reference T-pose shown only during calibration (Viser ReferenceSkeletonRenderer).
class ReferenceSkeletonView {
  constructor() {
    this.group = new THREE.Group();
    this.group.visible = false;
    world.add(this.group);
    this.spheres = [];
    this.parents = null;
    this.exclude = new Set();
    this.lineGeom = null;
    this.lines = null;
  }
  clear() {
    while (this.group.children.length) this.group.remove(this.group.children[0]);
    this.spheres = [];
    this.parents = null;
    this.exclude = new Set();
    this.lineGeom = null;
    this.lines = null;
  }
  load(ref) {
    this.clear();
    if (!ref?.positions?.length) return;
    const color = ref.color != null ? ref.color : 0x5eb3ff;
    const fr = ref.positions[0];
    this.parents = ref.parent_indices;
    this.exclude = new Set(ref.exclude_joint_indices || []);
    const J = this.parents.length;
    const mat = new THREE.MeshStandardMaterial({
      color, roughness: 0.4, metalness: 0.05, emissive: 0x1a3a66, emissiveIntensity: 0.35,
    });
    const sphereGeo = new THREE.SphereGeometry(0.032, 12, 12);
    for (let j = 0; j < J; j++) {
      const s = new THREE.Mesh(sphereGeo, mat);
      if (this.exclude.has(j)) s.visible = false;
      this.group.add(s);
      this.spheres.push(s);
    }
    let segCount = 0;
    for (let j = 0; j < J; j++) {
      const p = this.parents[j];
      if (p < 0 || this.exclude.has(j) || this.exclude.has(p)) continue;
      segCount++;
    }
    const positions = new Float32Array(segCount * 2 * 3);
    this.lineGeom = new THREE.BufferGeometry();
    this.lineGeom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    this.lines = new THREE.LineSegments(
      this.lineGeom,
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 })
    );
    this.group.add(this.lines);
    for (let j = 0; j < J; j++) {
      if (this.exclude.has(j)) continue;
      this.spheres[j].position.set(fr[j][0], fr[j][1], fr[j][2]);
    }
    const arr = this.lineGeom.attributes.position.array;
    let k = 0;
    for (let j = 0; j < J; j++) {
      const p = this.parents[j];
      if (p < 0 || this.exclude.has(j) || this.exclude.has(p)) continue;
      arr[k++] = fr[j][0]; arr[k++] = fr[j][1]; arr[k++] = fr[j][2];
      arr[k++] = fr[p][0]; arr[k++] = fr[p][1]; arr[k++] = fr[p][2];
    }
    this.lineGeom.attributes.position.needsUpdate = true;
    this.group.visible = true;
  }
}

// =================================================================  ENVIRONMENT (terrain + interaction objects)
// Owns the static terrain mesh AND the per-frame object props.  Crucially this
// is a *separate* view from the skeleton: in Viser the objects follow the clip
// even when the stick figure is hidden, so object animation must NOT be tied to
// SkeletonView visibility (the previous bug: hiding the skeleton froze props).
class EnvView {
  constructor() {
    this.group = env; // reuse the existing env group (child of world)
    this.objectMeshes = [];
    this.objectTraj = [];
    this.joints = null; // present-but-unused marker so the player animates us
  }
  clear() {
    while (this.group.children.length) this.group.remove(this.group.children[0]);
    this.objectMeshes = [];
    this.objectTraj = [];
    this.joints = null;
  }
  load(motion) {
    this.clear();
    this.clipDuration = effectivePlaybackDuration(motion);
    if (motion.terrain) {
      const m = buildTerrainMesh(motion.terrain);
      if (m) this.group.add(m);
    }
    (motion.objects || []).forEach((o, i) => this._buildObject(o, i, motion.token));
    // Mark as animatable so the shared player drives setFrame each tick.
    this.joints = this.objectTraj.length ? this.objectTraj : null;
    this.setFrame(0);
  }
  _buildObject(o, i, token) {
    const c = o.color ? (o.color[0] << 16) | (o.color[1] << 8) | o.color[2] : 0xff9f0a;
    const box = new THREE.Mesh(
      new THREE.BoxGeometry(o.extents[0], o.extents[1], o.extents[2]),
      new THREE.MeshStandardMaterial({
        color: c, transparent: true, opacity: o.opacity ?? 0.55, roughness: 0.6,
      })
    );
    this.group.add(box);
    this.objectMeshes.push(box);
    this.objectTraj.push(o);
    if (o.has_mesh && token) {
      const loader = new GLTFLoader();
      loader.load(
        `/api/object_glb?token=${token}&index=${i}`,
        (gltf) => {
          const real = gltf.scene;
          // GLB from /api/object_glb is already centred + scaled on the server.
          box.geometry.dispose();
          box.visible = false;
          this.group.add(real);
          this.objectMeshes[i] = real;
        },
        undefined,
        () => {} // keep box on failure
      );
    }
  }
  get numFrames() {
    return this.objectTraj.length && this.objectTraj[0].positions
      ? this.objectTraj[0].positions.length : 0;
  }
  setFrame(f) {
    for (let i = 0; i < this.objectMeshes.length; i++) {
      const o = this.objectTraj[i];
      if (!o || !o.positions[f]) continue;
      const m = this.objectMeshes[i];
      m.position.set(o.positions[f][0], o.positions[f][1], o.positions[f][2]);
      const q = o.quaternions[f];
      m.quaternion.set(q[0], q[1], q[2], q[3]); // backend sends xyzw
    }
  }
}

// Scaled terrain + props in the robot retarget frame (teal tint, co-located with robot).
class ScaledEnvView {
  constructor(group = scaledEnvGroup) {
    this.group = group;
    this.objectMeshes = [];
    this.objectTraj = [];
    this.joints = null;
    this.motionToken = null;
    this._objectGlbUrl = null;
    this.clipDuration = 1;
    this.group.visible = false;
  }
  clear() {
    while (this.group.children.length) this.group.remove(this.group.children[0]);
    this.objectMeshes = [];
    this.objectTraj = [];
    this.joints = null;
  }
  load(scene, motionToken, opts = {}) {
    this.clear();
    if (!scene) return;
    this.motionToken = motionToken;
    this._objectGlbUrl = opts.objectGlbUrl || null;
    this.clipDuration = Math.max(0.1, opts.duration ?? state.motion?.duration ?? 1);
    if (scene.terrain) {
      const m = buildTerrainMesh(scene.terrain);
      if (m) {
        m.material = new THREE.MeshStandardMaterial({
          color: 0x5c7a9e, roughness: 0.9, side: THREE.DoubleSide, flatShading: true,
          transparent: true, opacity: 0.92,
        });
        this.group.add(m);
      }
    }
    (scene.objects || []).forEach((o, i) => this._buildObject(o, i));
    this.joints = this.objectTraj.length ? this.objectTraj : null;
    this.setFrame(0);
  }
  _buildObject(o, i) {
    const c = o.color ? (o.color[0] << 16) | (o.color[1] << 8) | o.color[2] : 0x6a9fd4;
    const box = new THREE.Mesh(
      new THREE.BoxGeometry(o.extents[0], o.extents[1], o.extents[2]),
      new THREE.MeshStandardMaterial({
        color: c, transparent: true, opacity: o.opacity ?? 0.7, roughness: 0.55,
      })
    );
    this.group.add(box);
    this.objectMeshes.push(box);
    this.objectTraj.push(o);
    const srcIdx = o.source_index ?? i;
    const glbUrl = this._objectGlbUrl
      ? this._objectGlbUrl(o, srcIdx)
      : (this.motionToken
        ? `/api/object_glb?token=${this.motionToken}&index=${srcIdx}${
          o.scale != null && Number.isFinite(o.scale)
            ? `&scale=${encodeURIComponent(o.scale)}` : ""
        }`
        : null);
    if (o.has_mesh && glbUrl) {
      const loader = new GLTFLoader();
      loader.load(
        glbUrl,
        (gltf) => {
          const real = gltf.scene;
          box.geometry.dispose();
          box.visible = false;
          this.group.add(real);
          this.objectMeshes[i] = real;
        },
        undefined,
        () => {}
      );
    }
  }
  get numFrames() {
    return this.objectTraj.length && this.objectTraj[0].positions
      ? this.objectTraj[0].positions.length : 0;
  }
  setFrame(f) {
    this.setFrameFrac(f);
  }
  setFrameFrac(fi) {
    if (!this.objectTraj.length) return;
    const max = this.numFrames - 1;
    const { ia, ib, t } = resolvePlaybackFrame(null, fi, max);
    for (let i = 0; i < this.objectMeshes.length; i++) {
      const o = this.objectTraj[i];
      if (!o?.positions?.length) continue;
      const fr = o.positions[ia];
      if (!fr) continue;
      const m = this.objectMeshes[i];
      const blend = t > 1e-5 && ia !== ib && o.positions[ib];
      if (blend) {
        const nxt = o.positions[ib];
        m.position.set(
          fr[0] + (nxt[0] - fr[0]) * t,
          fr[1] + (nxt[1] - fr[1]) * t,
          fr[2] + (nxt[2] - fr[2]) * t,
        );
        const qa = o.quaternions[ia];
        const qb = o.quaternions[ib];
        m.quaternion.set(qa[0], qa[1], qa[2], qa[3]);
        _robotRootQuatB.set(qb[0], qb[1], qb[2], qb[3]);
        m.quaternion.slerp(_robotRootQuatB, t);
      } else {
        m.position.set(fr[0], fr[1], fr[2]);
        const q = o.quaternions[ia];
        m.quaternion.set(q[0], q[1], q[2], q[3]);
      }
    }
  }
}

// =================================================================  BODY MESH
// Per-bone tube + joint-bead "pseudo body" mesh, rebuilt from the same joint
// positions as the skeleton — works for ANY format (no SMPL weights needed).
// Mirrors hhtools.viewer.renderers.capsule_mesh.
const _SEG = 6; // fewer tube segments → smoother LAFAN / long clips
function _unitCylinder(segments) {
  const verts = [];
  for (let r = 0; r < 2; r++)
    for (let i = 0; i < segments; i++) {
      const a = (i / segments) * Math.PI * 2;
      verts.push([Math.cos(a), Math.sin(a), r]); // bottom ring z=0, top ring z=1
    }
  const faces = [];
  for (let i = 0; i < segments; i++) {
    const j = (i + 1) % segments;
    faces.push([i, j, i + segments], [j, j + segments, i + segments]);
  }
  return { verts, faces };
}
function _unitIcosphere() {
  const t = (1 + Math.sqrt(5)) / 2;
  let verts = [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
    [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ].map((v) => { const n = Math.hypot(...v); return [v[0] / n, v[1] / n, v[2] / n]; });
  const faces = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];
  return { verts, faces };
}
class CapsuleMeshView {
  constructor() {
    this.group = new THREE.Group();
    this.group.visible = false;
    world.add(this.group);
    this.heavy = false;
    this.joints = null;
    this.frameIndices = null;
    this.mesh = null;
    this.boneRadius = 0.035;
    this.jointRadius = 0.05;
    this.cyl = _unitCylinder(_SEG);
    this.sph = _unitIcosphere();
  }
  clear() {
    if (this.mesh) { this.group.remove(this.mesh); this.mesh.geometry.dispose(); this.mesh = null; }
    this.joints = null;
    this.frameIndices = null;
  }
  load(motion) {
    this.clear();
    this.joints = motion.positions;
    this.frameIndices = motion.frame_indices;
    this.clipDuration = effectivePlaybackDuration(motion);
    const parents = motion.parent_indices;
    const exclude = new Set(motion.exclude_joint_indices || []);
    this.edges = [];
    for (let j = 0; j < parents.length; j++) {
      const p = parents[j];
      if (p < 0 || exclude.has(j) || exclude.has(p)) continue;
      this.edges.push([p, j]);
    }
    this.visibleJoints = [];
    for (let j = 0; j < parents.length; j++) {
      if (!exclude.has(j)) this.visibleJoints.push(j);
    }
    this.numJoints = this.visibleJoints.length;
    // build index buffer once
    const vpb = this.cyl.verts.length; // verts per bone
    const vpj = this.sph.verts.length; // verts per joint
    const totalBoneV = this.edges.length * vpb;
    const idx = [];
    this.edges.forEach((_, e) => this.cyl.faces.forEach((f) =>
      idx.push(f[0] + e * vpb, f[1] + e * vpb, f[2] + e * vpb)));
    for (let j = 0; j < this.numJoints; j++)
      this.sph.faces.forEach((f) => idx.push(
        f[0] + totalBoneV + j * vpj, f[1] + totalBoneV + j * vpj, f[2] + totalBoneV + j * vpj));
    const nVerts = totalBoneV + this.numJoints * vpj;
    this.positions = new Float32Array(nVerts * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    geo.setIndex(idx);
    this.mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      color: 0xf7a470, roughness: 0.6, metalness: 0.05,
      side: THREE.DoubleSide, flatShading: true,
    }));
    this.group.add(this.mesh);
    this.setFrame(0);
  }
  get numFrames() { return this.joints ? this.joints.length : 0; }
  setFrame(f) {
    this.setFrameFrac(f);
  }
  setFrameFrac(fi) {
    if (!this.mesh || !this.joints) return;
    const max = this.joints.length - 1;
    const { ia, ib, t } = resolvePlaybackFrame(this.frameIndices, fi, max);
    const fr = this.joints[ia];
    if (!fr) return;
    const blend = t > 1e-5 && ia !== ib;
    const nxt = blend ? this.joints[ib] : null;
    const pos = this.positions;
    let o = 0;
    const r = this.boneRadius;
    for (const [pi, ci] of this.edges) {
      let sx = fr[pi][0], sy = fr[pi][1], sz = fr[pi][2];
      let ex = fr[ci][0], ey = fr[ci][1], ez = fr[ci][2];
      if (blend) {
        sx += (nxt[pi][0] - sx) * t;
        sy += (nxt[pi][1] - sy) * t;
        sz += (nxt[pi][2] - sz) * t;
        ex += (nxt[ci][0] - ex) * t;
        ey += (nxt[ci][1] - ey) * t;
        ez += (nxt[ci][2] - ez) * t;
      }
      const s = [sx, sy, sz], e = [ex, ey, ez];
      let dx = e[0] - s[0], dy = e[1] - s[1], dz = e[2] - s[2];
      let len = Math.hypot(dx, dy, dz) || 1e-6;
      dx /= len; dy /= len; dz /= len;
      // orthonormal basis
      let rx, ry, rz;
      if (Math.abs(dx) < 0.9) { rx = 1; ry = 0; rz = 0; } else { rx = 0; ry = 1; rz = 0; }
      let ax = dy * rz - dz * ry, ay = dz * rx - dx * rz, az = dx * ry - dy * rx;
      let an = Math.hypot(ax, ay, az) || 1; ax /= an; ay /= an; az /= an;
      const ux = dy * az - dz * ay, uy = dz * ax - dx * az, uz = dx * ay - dy * ax;
      for (const v of this.cyl.verts) {
        pos[o++] = s[0] + ax * (v[0] * r) + ux * (v[1] * r) + dx * (v[2] * len);
        pos[o++] = s[1] + ay * (v[0] * r) + uy * (v[1] * r) + dy * (v[2] * len);
        pos[o++] = s[2] + az * (v[0] * r) + uz * (v[1] * r) + dz * (v[2] * len);
      }
    }
    const jr = this.jointRadius;
    for (const j of this.visibleJoints) {
      let cx = fr[j][0], cy = fr[j][1], cz = fr[j][2];
      if (blend) {
        cx += (nxt[j][0] - cx) * t;
        cy += (nxt[j][1] - cy) * t;
        cz += (nxt[j][2] - cz) * t;
      }
      for (const v of this.sph.verts) {
        pos[o++] = cx + v[0] * jr; pos[o++] = cy + v[1] * jr; pos[o++] = cz + v[2] * jr;
      }
    }
    this.mesh.geometry.attributes.position.needsUpdate = true;
  }
}

// =================================================================  SCALED SKELETON (pre-IK, robot-calibrated)
class ScaledSkeletonView {
  constructor() {
    this.group = new THREE.Group();
    this.group.visible = false;
    world.add(this.group);
    this.joints = null;
    this.parents = null;
    this.frameIndices = null;
    this.spheres = [];
    this.lineGeom = null;
    this.lines = null;
    this.color = 0xffb020;
  }
  clear() {
    while (this.group.children.length) this.group.remove(this.group.children[0]);
    this.spheres = [];
    this.joints = null;
    this.frameIndices = null;
  }
  load(motion) {
    this.clear();
    this.joints = motion.positions;
    this.parents = motion.parent_indices;
    this.frameIndices = motion.frame_indices;
    this.clipDuration = effectivePlaybackDuration(motion);
    const J = this.parents.length;
    const mat = new THREE.MeshStandardMaterial({
      color: this.color, roughness: 0.45, metalness: 0.15, emissive: 0x442200,
    });
    const sphereGeo = new THREE.SphereGeometry(0.026, 10, 10);
    for (let j = 0; j < J; j++) {
      const s = new THREE.Mesh(sphereGeo, mat);
      this.group.add(s);
      this.spheres.push(s);
    }
    const segCount = this.parents.filter((p) => p >= 0).length;
    const positions = new Float32Array(segCount * 2 * 3);
    this.lineGeom = new THREE.BufferGeometry();
    this.lineGeom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    this.lines = new THREE.LineSegments(
      this.lineGeom,
      new THREE.LineBasicMaterial({ color: this.color, transparent: true, opacity: 0.85 })
    );
    this.group.add(this.lines);
    this.setFrame(0);
  }
  get numFrames() { return this.joints ? this.joints.length : 0; }
  setFrame(f) {
    this.setFrameFrac(f);
  }
  setFrameFrac(fi) {
    if (!this.joints) return;
    const max = this.joints.length - 1;
    const { ia, ib, t } = resolvePlaybackFrame(this.frameIndices, fi, max);
    const fr = this.joints[ia];
    if (!fr) return;
    const blend = t > 1e-5 && ia !== ib;
    const nxt = blend ? this.joints[ib] : null;
    for (let j = 0; j < this.spheres.length; j++) {
      if (blend) {
        this.spheres[j].position.set(
          fr[j][0] + (nxt[j][0] - fr[j][0]) * t,
          fr[j][1] + (nxt[j][1] - fr[j][1]) * t,
          fr[j][2] + (nxt[j][2] - fr[j][2]) * t,
        );
      } else {
        this.spheres[j].position.set(fr[j][0], fr[j][1], fr[j][2]);
      }
    }
    const arr = this.lineGeom.attributes.position.array;
    let k = 0;
    for (let j = 0; j < this.parents.length; j++) {
      const p = this.parents[j];
      if (p < 0) continue;
      if (blend) {
        arr[k++] = fr[j][0] + (nxt[j][0] - fr[j][0]) * t;
        arr[k++] = fr[j][1] + (nxt[j][1] - fr[j][1]) * t;
        arr[k++] = fr[j][2] + (nxt[j][2] - fr[j][2]) * t;
        arr[k++] = fr[p][0] + (nxt[p][0] - fr[p][0]) * t;
        arr[k++] = fr[p][1] + (nxt[p][1] - fr[p][1]) * t;
        arr[k++] = fr[p][2] + (nxt[p][2] - fr[p][2]) * t;
      } else {
        arr[k++] = fr[j][0]; arr[k++] = fr[j][1]; arr[k++] = fr[j][2];
        arr[k++] = fr[p][0]; arr[k++] = fr[p][1]; arr[k++] = fr[p][2];
      }
    }
    this.lineGeom.attributes.position.needsUpdate = true;
  }
}

// =================================================================  SKINNED BODY (SMPL / baked)
class BakedMeshView {
  constructor() {
    this.group = new THREE.Group();
    this.group.visible = false;
    world.add(this.group);
    this.heavy = true;
    this.mesh = null;
    this.verts = null; // Float32Array (F * V * 3)
    this.numVerts = 0;
    this.ready = false;
  }
  clear() {
    if (this.mesh) {
      this.group.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh = null;
    }
    this.verts = null;
    this.ready = false;
  }
  async load(bodyMesh) {
    this.clear();
    if (!bodyMesh?.available) return;
    try {
      const bin = Uint8Array.from(atob(bodyMesh.vertices_gz_b64), (c) => c.charCodeAt(0));
      const ds = new DecompressionStream("gzip");
      const buf = await new Response(new Blob([bin]).stream().pipeThrough(ds)).arrayBuffer();
      this.verts = new Float32Array(buf);
      this.numVerts = bodyMesh.num_verts;
      const numFrames = bodyMesh.num_frames;
      const expected = numFrames * this.numVerts * 3;
      if (this.verts.length !== expected) {
        console.warn("baked mesh vertex buffer size mismatch", this.verts.length, expected);
        return;
      }
      this.clipDuration = null; // driven by skeleton timeline
      const idx = bodyMesh.triangles.flat();
      const geo = new THREE.BufferGeometry();
      geo.setAttribute(
        "position",
        new THREE.BufferAttribute(this.verts.slice(0, this.numVerts * 3), 3)
      );
      geo.setIndex(idx);
      geo.computeVertexNormals();
      this.mesh = new THREE.Mesh(
        geo,
        new THREE.MeshStandardMaterial({
          color: 0xb4c8dc, roughness: 0.55, metalness: 0.05,
          side: THREE.DoubleSide, flatShading: true,
        })
      );
      this.group.add(this.mesh);
      this.ready = true;
      this.setFrame(0);
    } catch (e) {
      console.warn("baked mesh decode failed", e);
      this.ready = false;
    }
  }
  get numFrames() {
    return this.ready && this.numVerts ? this.verts.length / (this.numVerts * 3) : 0;
  }
  setFrame(f) {
    this.setFrameFrac(f);
  }
  setFrameFrac(fi) {
    if (!this.ready || !this.mesh) return;
    const max = this.numFrames - 1;
    const f0 = Math.min(max, Math.floor(fi));
    const off0 = f0 * this.numVerts * 3;
    const attr = this.mesh.geometry.attributes.position;
    const t = fi - f0;
    if (t <= 1e-5 || f0 >= max) {
      attr.array.set(this.verts.subarray(off0, off0 + this.numVerts * 3));
    } else {
      const off1 = (f0 + 1) * this.numVerts * 3;
      const dst = attr.array;
      const a = this.verts;
      const n = this.numVerts * 3;
      for (let i = 0; i < n; i++) {
        dst[i] = a[off0 + i] + (a[off1 + i] - a[off0 + i]) * t;
      }
    }
    attr.needsUpdate = true;
  }
}

// =================================================================  ROBOT
const _robotLinkDelta = new THREE.Matrix4();
const _robotMeshMat = new THREE.Matrix4();
const _robotLinkMat = new THREE.Matrix4();
const _robotRootQuatB = new THREE.Quaternion();
const _robotMatB = new THREE.Matrix4();
const _robotPosA = new THREE.Vector3();
const _robotPosB = new THREE.Vector3();
const _robotQuatA = new THREE.Quaternion();
const _robotQuatB2 = new THREE.Quaternion();
const _robotScaleA = new THREE.Vector3();
const _robotScaleB = new THREE.Vector3();
class RobotView {
  constructor() {
    this.group = new THREE.Group();
    world.add(this.group);
    this.linkMeshes = {}; // link -> [{mesh, bakedWorld}]
    this.meshToLink = {}; // GLB node / mesh basename -> URDF link
    this.zeroInv = {}; // link -> THREE.Matrix4 inverse of zero transform
    this.links = [];
    this.trajectory = null;
    this.frameIndices = null;
    this.groundOffset = 0;
    this.group.visible = false;
  }
  clear() {
    while (this.group.children.length) this.group.remove(this.group.children[0]);
    this.linkMeshes = {};
    this.meshToLink = {};
    this.zeroInv = {};
    this.trajectory = null;
  }
  setVisible(v) {
    this.group.visible = v;
  }
  // No trajectory yet: drop the robot on the ground at its zero/T-pose.
  applyStatic() {
    this.group.position.set(0, 0, this.groundOffset);
    this.group.quaternion.identity();
    for (const link of this.links) {
      const entry = this.linkMeshes[link];
      if (!entry) continue;
      for (const { mesh, baked } of entry) mesh.matrix.copy(baked);
    }
  }
  async load(robot) {
    this.clear();
    this.links = robot.links;
    this.meshToLink = robot.mesh_to_link || {};
    this.zero = robot.link_transforms_zero;
    this.groundOffset = robot.ground_offset_z || 0;
    for (const link of this.links) {
      const m = mat4(this.zero[link]);
      this.zeroInv[link] = m.clone().invert();
    }
    if (!robot.glb_base64) {
      // fall back to link-frame skeleton
      this._buildLinkSkeleton();
      this.applyStatic();
      return;
    }
    const bytes = Uint8Array.from(atob(robot.glb_base64), (c) => c.charCodeAt(0));
    const loader = new GLTFLoader();
    await new Promise((resolve) => {
      loader.parse(bytes.buffer, "", (gltf) => {
        gltf.scene.updateMatrixWorld(true);
        const meshes = [];
        gltf.scene.traverse((n) => { if (n.isMesh) meshes.push(n); });
        for (const mesh of meshes) {
          const link = this._linkForNode(mesh);
          if (!link) continue;
          mesh.userData.hhtoolsLink = link;
          const baked = mesh.matrixWorld.clone();
          mesh.matrixAutoUpdate = false;
          // trimesh→GLB exports frequently omit vertex normals; without them
          // any lit material renders pure black.  Compute them once here.
          const g = mesh.geometry;
          if (g && !g.getAttribute("normal")) {
            g.computeVertexNormals();
          }
          applyRobotMaterial(mesh);
          this.group.add(mesh);
          mesh.matrix.copy(baked);
          mesh.updateMatrixWorld(true);
          (this.linkMeshes[link] ||= []).push({ mesh, baked });
        }
        this.group.updateMatrixWorld(true);
        resolve();
      }, () => { this._buildLinkSkeleton(); resolve(); });
    });
    this.applyStatic();
  }
  _normPickKey(s) {
    return (s || "").toLowerCase().replace(/[^a-z0-9]/g, "");
  }
  _meshBasename(name) {
    const base = String(name || "").split(/[/\\]/).pop();
    return base.replace(/\.[^.]+$/, "");
  }
  _linkForNode(node) {
    const names = this.links;
    let cur = node;
    while (cur) {
      const tagged = cur.userData?.hhtoolsLink;
      if (tagged) return tagged;
      const raw = cur.name || "";
      if (this.meshToLink[raw]) return this.meshToLink[raw];
      const base = this._meshBasename(raw);
      if (this.meshToLink[base]) return this.meshToLink[base];
      const cn = this._normPickKey(raw);
      for (const l of names) {
        if (this._normPickKey(l) === cn && cn) return l;
      }
      const sn = this._normPickKey(base);
      if (sn) {
        for (const l of names) {
          const ln = this._normPickKey(l);
          const lc = ln.endsWith("link") ? ln.slice(0, -4) : ln;
          if (lc === sn || ln === sn) return l;
        }
      }
      cur = cur.parent;
    }
    return null;
  }
  _buildLinkSkeleton() {
    const geo = new THREE.SphereGeometry(0.02, 8, 8);
    const matl = new THREE.MeshStandardMaterial({ color: 0xb8bdc6 });
    for (const link of this.links) {
      const s = new THREE.Mesh(geo, matl);
      s.matrixAutoUpdate = false;
      s.matrix.copy(mat4(this.zero[link]));
      this.group.add(s);
      (this.linkMeshes[link] ||= []).push({ mesh: s, baked: mat4(this.zero[link]) });
    }
  }
  setTrajectory(traj) {
    this.trajectory = traj;
    this.frameIndices = traj.frame_indices;
    this.clipDuration = effectivePlaybackDuration(traj);
    // IK root + mesh_z_lift (align mesh sole to yellow overlay foot when present).
    this.setFrame(0);
  }
  get numFrames() {
    return this.trajectory ? this.trajectory.frames.length : 0;
  }
  setFrame(f) {
    this.setFrameFrac(f);
  }
  setFrameFrac(fi) {
    if (!this.trajectory) return;
    const max = this.trajectory.frames.length - 1;
    const { ia, ib, t } = resolvePlaybackFrame(this.frameIndices, fi, max);
    const frame = this.trajectory.frames[ia];
    if (!frame) return;
    const nxtFrame = t > 1e-5 && ia !== ib ? this.trajectory.frames[ib] : null;
    const root = frame.root;
    const liftA = frame.mesh_z_lift || 0;
    const liftB = nxtFrame?.mesh_z_lift ?? liftA;
    const meshLift = liftA + (liftB - liftA) * t;
    if (root) {
      if (t > 1e-5 && ia !== ib) {
        const nxt = this.trajectory.frames[ib]?.root;
        if (nxt) {
          this.group.position.set(
            root[0] + (nxt[0] - root[0]) * t,
            root[1] + (nxt[1] - root[1]) * t,
            root[2] + (nxt[2] - root[2]) * t + meshLift,
          );
          this.group.quaternion.set(root[3], root[4], root[5], root[6]);
          _robotRootQuatB.set(nxt[3], nxt[4], nxt[5], nxt[6]);
          this.group.quaternion.slerp(_robotRootQuatB, t);
        } else {
          this.group.position.set(root[0], root[1], root[2] + meshLift);
          this.group.quaternion.set(root[3], root[4], root[5], root[6]);
        }
      } else {
        this.group.position.set(root[0], root[1], root[2] + meshLift);
        this.group.quaternion.set(root[3], root[4], root[5], root[6]);
      }
    }
    this._applyLinkTransforms(frame.links, nxtFrame ? nxtFrame.links : null, t);
  }
  /** Pose link meshes from FK (calibration preview) or trajectory frame. */
  _applyLinkTransforms(linkTransforms, nextTransforms = null, t = 0) {
    const lerp = nextTransforms != null && t > 1e-5;
    for (const link of this.links) {
      const entry = this.linkMeshes[link];
      if (!entry || !linkTransforms[link]) continue;
      mat4Into(linkTransforms[link], _robotLinkMat);
      if (lerp && nextTransforms[link]) {
        mat4Into(nextTransforms[link], _robotMatB);
        _robotLinkMat.decompose(_robotPosA, _robotQuatA, _robotScaleA);
        _robotMatB.decompose(_robotPosB, _robotQuatB2, _robotScaleB);
        _robotPosA.lerp(_robotPosB, t);
        _robotQuatA.slerp(_robotQuatB2, t);
        _robotLinkMat.compose(_robotPosA, _robotQuatA, _robotScaleA);
      }
      _robotLinkDelta.copy(_robotLinkMat).multiply(this.zeroInv[link]);
      for (const { mesh, baked } of entry) {
        _robotMeshMat.copy(_robotLinkDelta).multiply(baked);
        mesh.matrix.copy(_robotMeshMat);
      }
    }
    this.group.updateMatrixWorld(true);
  }
  /** Static calibration pose on the ground (no floating-base trajectory yet). */
  applyCalibPose(linkTransforms, groundZ) {
    const z = groundZ != null && Number.isFinite(groundZ) ? groundZ : this.groundOffset;
    this.group.position.set(0, 0, z);
    this.group.quaternion.identity();
    this._applyLinkTransforms(linkTransforms);
  }

  /** Calibration pick/hover: tint link meshes (hover = soft blue, selected = accent). */
  setCalibHighlights({ hover = null, selected = null } = {}) {
    const BASE = { color: 0xc8ccd4, emissive: 0x6b7280, emissiveIntensity: 0.55 };
    const HOVER = { color: 0xd6e4ff, emissive: 0x3b82f6, emissiveIntensity: 0.92 };
    const SELECT = { color: 0xbfdbfe, emissive: 0x1d4ed8, emissiveIntensity: 1.15 };
    for (const [link, entries] of Object.entries(this.linkMeshes)) {
      let pal = BASE;
      if (selected && link === selected) pal = SELECT;
      else if (hover && link === hover) pal = HOVER;
      for (const { mesh } of entries) {
        const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        for (const m of mats) {
          if (!m?.isMeshStandardMaterial) continue;
          m.color.setHex(pal.color);
          m.emissive.setHex(pal.emissive);
          m.emissiveIntensity = pal.emissiveIntensity;
        }
      }
    }
  }
}

function applyRobotMaterial(mesh) {
  // Light brushed-metal look. A bright emissive floor guarantees the robot is
  // clearly visible even if a mesh still ends up without usable normals.
  const make = () => new THREE.MeshStandardMaterial({
    color: 0xc8ccd4,
    emissive: 0x6b7280,
    emissiveIntensity: 0.55,
    roughness: 0.6,
    metalness: 0.15,
    side: THREE.DoubleSide,
    vertexColors: false,
  });
  if (Array.isArray(mesh.material)) {
    mesh.material = mesh.material.map(() => make());
  } else {
    mesh.material = make();
  }
}

function mat4Into(flat, out) {
  // backend sends row-major flattened 4x4; three.js wants column-major.
  return out.set(
    flat[0], flat[1], flat[2], flat[3],
    flat[4], flat[5], flat[6], flat[7],
    flat[8], flat[9], flat[10], flat[11],
    flat[12], flat[13], flat[14], flat[15]
  );
}

function mat4(flat) {
  return mat4Into(flat, new THREE.Matrix4());
}

// =================================================================  PLAYER
const skel = new SkeletonView();
const refSkel = new ReferenceSkeletonView();
const mesh = new CapsuleMeshView();
const skin = new BakedMeshView();
const scaledSkel = new ScaledSkeletonView();
const envView = new EnvView();
const scaledEnv = new ScaledEnvView();
const robot = new RobotView();
const ALL_VIEWS = [skel, mesh, skin, scaledSkel, envView, scaledEnv, robot];

function bodyUsesSkin() {
  return skin.ready;
}
function setBodyVisible(on) {
  const btn = document.getElementById("tg-mesh");
  btn.classList.toggle("on", on);
  if (on && bodyUsesSkin()) {
    skin.group.visible = true;
    mesh.group.visible = false;
  } else {
    skin.group.visible = false;
    mesh.group.visible = on;
  }
  player.refreshFrame();
}
function bodyIsVisible() {
  return skin.group.visible || mesh.group.visible;
}

// All animatable views share ONE timeline (fraction of clip duration) so the
// human skeleton / body-mesh and the retargeted robot stay frame-synced and
// can be shown together.
const player = {
  playing: false,
  loop: true,
  t: 0,
  duration: 0,
  active: false,
  speed: 1, // playback rate multiplier (0.1×–4×), independent of the timeline
  // Set by update() on a loop wrap; consumed by animate() to hard-snap the camera.
  _justLooped: false,
  ready(duration) {
    this.duration = Math.max(0.1, duration || 1);
    this.t = 0;
    this.active = true;
    this._justLooped = false;
    revealStage();
  },
  _heavyTick: 0,
  _applyFrac(frac, { force = false } = {}) {
    if (r2r.calibrating || (state.calibrationMode && !r2r.active)) return;
    if (!force) this._heavyTick = (this._heavyTick + 1) % 2;
    const robotReady = Boolean(robot.trajectory && robot.trajectory.frames?.length);
    for (const v of ALL_VIEWS) {
      if (v.numFrames <= 0) continue;
      // Yellow overlay / scaled env only track playback once retarget has produced
      // a robot trajectory; otherwise they animate against a frozen robot pose.
      if ((v === scaledSkel || v === scaledEnv) && !robotReady) continue;
      // env views animate even when "invisible" to the HUD — except scaledEnv
      // which follows its toggle.
      if (!v.group.visible) continue;
      // Heavy views (robot mesh / baked body) update every other frame while
      // playing — but NEVER skip on a forced seek / loop wrap, or the robot
      // stays at the last frame for one tick while the timeline is already
      // back at the start (looks like a global teleport).
      if (!force && this.playing && v.heavy && this._heavyTick === 1) continue;
      const fi = frac * (v.numFrames - 1);
      if (v.setFrameFrac) v.setFrameFrac(fi);
      else v.setFrame(Math.min(v.numFrames - 1, Math.floor(fi)));
    }
  },
  update(dt) {
    if (!this.playing || !this.active) return;
    // Cap dt so a backgrounded tab cannot leap many seconds and land mid-clip
    // after a modulo wrap (reads as a random global jump on the 2nd play).
    const step = Math.min(Math.max(0, dt), 0.1) * this.speed;
    this.t += step;
    let looped = false;
    if (this.t >= this.duration) {
      if (this.loop) {
        // Exact restart at t=0 — do NOT use ``t % duration``.  Overshoot
        // remainder lands mid-first-frame (or much later after a large dt),
        // which looks like the robot teleporting to a wrong global pose
        // when the clip wraps for the second playthrough.
        this.t = 0;
        looped = true;
        this._justLooped = true;
      } else {
        this.t = this.duration;
        this.setPlaying(false);
      }
    }
    const frac = this.duration > 0 ? this.t / this.duration : 0;
    this._applyFrac(frac, { force: looped });
    this._syncScrub(frac);
  },
  setPlaying(p) {
    this.playing = p && this.active;
    document.getElementById("play-btn").textContent = this.playing ? "❚❚" : "▶";
  },
  seek(frac) {
    if (!this.active) return;
    const f = Math.min(1, Math.max(0, Number(frac) || 0));
    this.t = f * this.duration;
    this._justLooped = false;
    this._applyFrac(f, { force: true });
    this._syncScrub(f);
  },
  setSpeed(mult) {
    const m = Math.min(4, Math.max(0.1, Number(mult) || 1));
    this.speed = m;
    const slider = document.getElementById("speed-slider");
    if (slider && Math.abs(parseFloat(slider.value) - m) > 1e-6) slider.value = m;
    const label = document.getElementById("speed-label");
    if (label) label.textContent = `${m.toFixed(1)}×`;
  },
  // Re-pose whatever is currently visible at the current cursor (after a toggle).
  refreshFrame() {
    if (this.active) this._applyFrac(this.t / this.duration, { force: true });
  },
  _syncScrub(frac) {
    document.getElementById("scrubber").value = frac * 100;
    const src = state.motion || state.robotTrajectory;
    let label = `${this.t.toFixed(2)} / ${this.duration.toFixed(2)} s`;
    if (isPlaybackPreview(src) && src.duration > this.duration + 0.5) {
      label += `（预览，原片 ${src.duration.toFixed(1)} s）`;
    }
    document.getElementById("time-label").textContent = label;
  },
};

function revealStage() {
  document.getElementById("playbar").classList.remove("hidden");
  document.getElementById("view-reset-btn")?.classList.remove("hidden");
  document.getElementById("view-hud").classList.remove("hidden");
  document.getElementById("stage-empty").style.display = "none";
}

document.getElementById("view-reset-btn")?.addEventListener("click", resetDefaultView);

document.getElementById("play-btn").onclick = () => player.setPlaying(!player.playing);
document.getElementById("loop-btn").onclick = (e) => {
  player.loop = !player.loop;
  e.target.style.opacity = player.loop ? 1 : 0.4;
};
document.getElementById("scrubber").oninput = (e) => player.seek(e.target.value / 100);
document.getElementById("speed-slider").oninput = (e) => player.setSpeed(e.target.value);
// Double-click the speed control to snap back to 1× (real-time).
document.querySelector(".speed-ctrl")?.addEventListener("dblclick", () => player.setSpeed(1));

// ----------------------------------------------------------------- view toggles
function motionHasEnvironment(payload) {
  if (!payload) return false;
  if (payload.has_terrain || payload.terrain) return true;
  if (Array.isArray(payload.objects) && payload.objects.length > 0) return true;
  const meta = payload.meta;
  if (meta && typeof meta === "object") {
    if (meta.terrain_mesh) return true;
    if (Number(meta.num_objects) > 0) return true;
  }
  return false;
}

function syncEnvToggleButton() {
  const btn = document.getElementById("tg-env");
  if (!btn) return;
  const available = motionHasEnvironment(state.motion);
  btn.disabled = !available;
  if (!available) {
    btn.classList.remove("on");
    return;
  }
  btn.classList.toggle("on", envView.group.visible);
}

function setViewVisible(view, btnId, on) {
  if (state.calibrationMode) {
    const blocked = new Set(["tg-skeleton", "tg-scaled", "tg-scaled-env", "tg-env"]);
    if (blocked.has(btnId) && on) return;
  }
  view.group.visible = on;
  document.getElementById(btnId).classList.toggle("on", on);
  if (btnId === "tg-env") syncEnvToggleButton();
  player.refreshFrame();
}
document.getElementById("tg-skeleton").onclick = () =>
  setViewVisible(skel, "tg-skeleton", !skel.group.visible);
document.getElementById("tg-mesh").onclick = () => setBodyVisible(!bodyIsVisible());
document.getElementById("tg-env").onclick = (e) => {
  if (e.currentTarget.disabled) return;
  setViewVisible(envView, "tg-env", !envView.group.visible);
};
document.getElementById("tg-scaled").onclick = (e) => {
  if (e.currentTarget.disabled) return;
  setViewVisible(scaledSkel, "tg-scaled", !scaledSkel.group.visible);
};
document.getElementById("tg-scaled-env").onclick = (e) => {
  if (e.currentTarget.disabled) return;
  setViewVisible(scaledEnv, "tg-scaled-env", !scaledEnv.group.visible);
};
document.getElementById("tg-robot").onclick = (e) => {
  if (e.currentTarget.disabled) return;
  setViewVisible(robot, "tg-robot", !robot.group.visible);
};

async function refreshScaledPreview() {
  const btnSkel = document.getElementById("tg-scaled");
  const btnEnv = document.getElementById("tg-scaled-env");
  if (!state.motion || !state.robot || !state.calibration) {
    scaledSkel.clear();
    scaledEnv.clear();
    btnSkel.disabled = true;
    btnEnv.disabled = true;
    setViewVisible(scaledSkel, "tg-scaled", false);
    setViewVisible(scaledEnv, "tg-scaled-env", false);
    return;
  }
  try {
    const data = await API.post("/api/scaled_preview", {
      robot: state.robot.name,
      motion_token: state.motion.token,
      reference: state.reference,
    });
    const preview = data.preview ?? data;
    scaledSkel.load(preview);
    btnSkel.disabled = false;
    if (data.scaled_scene) {
      scaledEnv.load(data.scaled_scene, state.motion.token);
      btnEnv.disabled = false;
    } else {
      scaledEnv.clear();
      btnEnv.disabled = true;
    }
    // Preload yellow overlay data but keep it hidden until a retarget completes
    // (or the user explicitly toggles it on).  Playing motion against a frozen
    // calibration / zero robot makes the overlay look collapsed inside the mesh.
    if (!state.robotTrajectory) {
      setViewVisible(scaledSkel, "tg-scaled", false);
      setViewVisible(scaledEnv, "tg-scaled-env", false);
    }
    if (player.active) player.refreshFrame();
  } catch (e) {
    scaledSkel.clear();
    scaledEnv.clear();
    btnSkel.disabled = true;
    btnEnv.disabled = true;
    console.warn("scaled preview", e.message);
  }
}

// =================================================================  STATE
const state = {
  motion: null, // serialized payload incl token
  libraryEntry: null, // resource-library row for batch basket
  robot: null, // serialized robot
  reference: null,
  calibration: false,
  calibrationMode: false,
  calibNeedsCameraFocus: false,
  calibOrbitSaved: null,
  calibLimits: null,
  calibRestore: null,
  exportToken: null,
  calibQ: {},
  calibSliderRows: {},
  calibBaselineQ: null,
  calibHasSaved: false,
};

const REFERENCE_LABELS = {
  smpl: "SMPL",
  smplx: "SMPL-X",
  gvhmr: "GVHMR",
  soma_bvh: "SOMA BVH",
  lafan_bvh: "LAFAN / Mixamo BVH",
  mocap_bvh: "MOCAP BVH (Spine3 chest)",
  xsens_mocap: "Xsens mocap BVH",
  glb: "GLB / GLTF",
};

/** Mirror server ``_DATASET_TO_REFERENCE`` for basket rows without ``reference``. */
const DATASET_TO_REFERENCE = {
  amass: "smpl",
  motion_x: "smplx",
  phuma: "smpl",
  lafan: "lafan_bvh",
  mocap: "mocap_bvh",
  soma: "soma_bvh",
  xsens_mocap: "xsens_mocap",
  gvhmr: "gvhmr",
  omomo: "smplx",
  meshmimic_holosoma: "smplx",
  glb: "glb",
  unified_npz: "smpl",
  parc_ms: "smpl",
};

function entryReference(e, fallback = "smpl") {
  return (e?.reference || "").trim()
    || DATASET_TO_REFERENCE[e?.dataset]
    || fallback;
}

function referenceLabel(ref) {
  return REFERENCE_LABELS[ref] || ref || "—";
}

/** Human-readable adapter / dataset id (basket ``dataset`` field). */
const DATASET_LABELS = {
  soma: "SOMA BVH",
  lafan: "LAFAN / Mixamo BVH",
  mocap: "MOCAP BVH (Spine3 chest)",
  xsens_mocap: "Xsens mocap BVH",
  amass: "AMASS (SMPL 参数)",
  motion_x: "Motion-X (SMPL-X)",
  phuma: "PHUMA (SMPL)",
  gvhmr: "GVHMR (SMPL-H)",
  omomo: "OMOMO (SMPL-X)",
  glb: "GLB 骨骼",
  parc_ms: "parc_ms / meshmimic",
  meshmimic_holosoma: "holosoma NPY",
  unified_npz: "hhtools NPZ",
  unknown: "未识别",
};

/**
 * What each calibration reference means for retarget (not the same as SMPL weights).
 * ``reference`` = which saved calibration YAML + reference T-pose to use.
 */
const REFERENCE_HELP = {
  soma_bvh: {
    input: "SOMA 统一比例骨架 .bvh（关节名如 Hips、LeftUpLeg；来自 SOMA / soma-retargeter）",
    calib: "标定参考「SOMA BVH」— 对齐<b>蓝色 SOMA 标准骨架</b>与机器人",
    file: "retarget_calibration_soma_bvh.yaml",
  },
  lafan_bvh: {
    input: "LAFAN / Mixamo 风格 .bvh（如 Hips、LeftLeg）",
    calib: "标定参考「LAFAN / Mixamo BVH」— 对齐<b>蓝色 LAFAN 参考骨架</b>",
    file: "retarget_calibration_lafan_bvh.yaml",
  },
  mocap_bvh: {
    input: "四节脊柱 MOCAP .bvh（Hips、Spine3 挂肩、LeftToeBase）",
    calib: "标定参考「MOCAP BVH」— 对齐<b>蓝色 MOCAP 参考骨架</b>（Spine3 = chest）",
    file: "retarget_calibration_mocap_bvh.yaml",
  },
  xsens_mocap: {
    input: "Xsens MVN / 生物力学 .bvh（如 Hips、LeftHip、LeftKnee、Chest）",
    calib: "标定参考「Xsens mocap BVH」— 对齐<b>蓝色 Xsens 参考骨架</b>",
    file: "retarget_calibration_xsens_mocap.yaml",
  },
  smpl: {
    input: "AMASS / SMPL 参数 .npz（poses + trans，需 SMPL 体模）",
    calib: "标定参考「SMPL」— 对齐<b>蓝色 SMPL T-pose 参考骨架</b>",
    file: "retarget_calibration_smpl.yaml",
  },
  smplx: {
    input: "SMPL-X 参数或 OMOMO / Motion-X 等",
    calib: "标定参考「SMPL-X」— 对齐<b>蓝色 SMPL-X 参考骨架</b>",
    file: "retarget_calibration_smplx.yaml",
  },
  gvhmr: {
    input: "GVHMR / HMR4D 输出的 .pt 或 SMPL-H 轨迹",
    calib: "标定参考「GVHMR」— 对齐<b>蓝色 GVHMR 参考骨架</b>",
    file: "retarget_calibration_gvhmr.yaml",
  },
  glb: {
    input: "带骨骼的 .glb / .gltf",
    calib: "标定参考「GLB / GLTF」— 对齐<b>蓝色 GLB 第 0 帧参考骨架</b>",
    file: "retarget_calibration_glb.yaml",
  },
};

function datasetLabel(ds) {
  return DATASET_LABELS[ds] || ds || "未识别";
}

let referenceCatalog = [];

async function loadReferenceCatalog() {
  try {
    const { references } = await API.get("/api/calibration/references");
    referenceCatalog = references?.length ? references : Object.keys(REFERENCE_LABELS);
  } catch {
    referenceCatalog = Object.keys(REFERENCE_LABELS);
  }
  populateRefSelect();
}

function populateRefSelect() {
  const sel = document.getElementById("rt-ref-select");
  if (!sel) return;
  const prev = state.reference || sel.value;
  sel.innerHTML = "";
  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = "—";
  sel.appendChild(blank);
  for (const ref of referenceCatalog) {
    const opt = document.createElement("option");
    opt.value = ref;
    opt.textContent = REFERENCE_LABELS[ref] || ref;
    sel.appendChild(opt);
  }
  if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
  syncRefSelect();
}

function syncRefSelect() {
  const sel = document.getElementById("rt-ref-select");
  if (!sel) return;
  if (state.reference && [...sel.options].some((o) => o.value === state.reference)) {
    sel.value = state.reference;
  } else if (!state.reference) {
    sel.value = "";
  }
  sel.disabled = !state.robot;
  const hint = document.getElementById("rt-ref-hint");
  if (!hint) return;
  if (state.motion?.dataset) {
    const ref = state.reference || "—";
    hint.textContent = `自动识别数据集: ${state.motion.dataset} → 建议参考 ${ref}`;
    hint.style.display = "block";
  } else {
    hint.textContent = "";
    hint.style.display = "none";
  }
}

async function onReferenceChange(newRef) {
  if (!newRef || newRef === state.reference) return;
  const wasCalibrating = state.calibrationMode;
  const savedQ = wasCalibrating ? { ...state.calibQ } : null;
  if (wasCalibrating) await exitCalibrationMode();
  state.reference = newRef;
  syncRefSelect();
  if (wasCalibrating && state.robot && state.motion) {
    await enterCalibrationMode(savedQ);
  } else {
    await refreshRetargetPanel();
  }
}

function updatePills() {
  document.getElementById("motion-pill").textContent = state.motion
    ? `🎞 ${state.motion.name}` : "未加载动作";
  document.getElementById("robot-pill").textContent = state.robot
    ? `🤖 ${state.robot.display_name}` : "未加载机器人";
}

// =================================================================  NAV
function switchInspectorPanel(panelId) {
  if (!panelId) return;
  const btn = document.querySelector(`.nav-item[data-panel="${panelId}"]`);
  const panel = document.querySelector(`#inspector .panel[data-panel="${panelId}"]`);
  if (!btn || !panel) return;
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
  document.querySelectorAll("#inspector .panel").forEach((p) => p.classList.remove("active"));
  btn.classList.add("active");
  panel.classList.add("active");
}

document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.onclick = () => switchInspectorPanel(btn.dataset.panel);
});

/** After a robot is loaded, jump to the panel that matches the current workflow. */
async function routeAfterRobotLoad() {
  if (!state.motion) {
    switchInspectorPanel("motion");
    await refreshRetargetPanel();
    return;
  }
  switchInspectorPanel("robot");
  await refreshRetargetPanel();
}

// =================================================================  MOTION
async function loadMotionPayload(payload) {
  state.motion = payload;
  state.libraryEntry = payload.library_entry || null;
  state.reference = payload.suggested_reference;
  syncRefSelect();
  state.exportToken = null;
  // In calibration mode only the robot + blue reference T-pose should be visible.
  if (state.calibrationMode) {
    state.robotTrajectory = null;
    robot.trajectory = null;
    scaledSkel.clear();
    scaledEnv.clear();
    player.setPlaying(false);
    await refreshRetargetPanel();
    _applyCalibSceneLayout();
    toast(`已加载 ${payload.name}（标定模式）`);
    updatePills();
    return;
  }
  skel.load(payload, 0x0a84ff);
  mesh.load(payload);
  envView.load(payload);
  const hasEnv = motionHasEnvironment(payload);
  if (hasEnv) {
    setViewVisible(envView, "tg-env", true);
  } else {
    envView.clear();
    envView.group.visible = false;
    syncEnvToggleButton();
  }
  await skin.load(payload.body_mesh);
  // Terrain/objects clips default to the interaction-mesh backend (matches Viser
  // "Auto"); pure skeletal clips stay on Newton IK.
  if (payload.suggested_backend) {
    const rb = document.getElementById("rt-backend");
    const bb = document.getElementById("batch-backend");
    if (rb) rb.value = payload.suggested_backend;
    if (bb) bb.value = payload.suggested_backend;
  }
  // A fresh motion invalidates any previous retarget result.
  state.robotTrajectory = null;
  robot.trajectory = null;
  if (state.robot) robot.applyStatic();
  // parc_ms / skeletal-only: default skeleton lines (capsules collapse when FK rest is wrong).
  const isParcMs =
    payload.meta?.dataset === "parc_ms" ||
    payload.meta?.source_format === "parc_ms_pkl";
  const hasSkin = Boolean(payload.body_mesh?.available);
  const showSkeleton = isParcMs || !hasSkin;
  setViewVisible(skel, "tg-skeleton", showSkeleton);
  setBodyVisible(!showSkeleton || hasSkin);
  setViewVisible(robot, "tg-robot", false);
  player.ready(effectivePlaybackDuration(payload));
  player.setPlaying(true);
  // meta card
  document.getElementById("motion-meta-card").style.display = "block";
  document.getElementById("motion-name").textContent = payload.name;
  const previewNote = isPlaybackPreview(payload)
    ? `（预览 ${payload.playback_frames ?? payload.positions.length} 帧 / ${effectivePlaybackDuration(payload).toFixed(1)} s）`
    : "";
  document.getElementById("motion-meta").innerHTML = `
    <div class="meta-row"><span class="k">格式</span><span class="v">${payload.source_format}</span></div>
    <div class="meta-row"><span class="k">帧数</span><span class="v">${payload.num_frames_total}</span></div>
    <div class="meta-row"><span class="k">帧率</span><span class="v">${payload.framerate.toFixed(1)}</span></div>
    <div class="meta-row"><span class="k">时长</span><span class="v">${payload.duration.toFixed(2)} s${previewNote}</span></div>
    <div class="meta-row"><span class="k">骨骼</span><span class="v">${payload.bone_names.length}</span></div>
    ${payload.objects.length ? `<div class="meta-row"><span class="k">交互物体</span><span class="v">${payload.objects.length}</span></div>` : ""}
    ${payload.has_terrain ? `<div class="meta-row"><span class="k">地形</span><span class="v">有</span></div>` : ""}
    <div class="meta-row"><span class="k">身体 mesh</span><span class="v">${
      payload.body_mesh?.available ? "SMPL/皮肤" : payload.body_mesh?.reason || "管状近似"
    }</span></div>`;
  updatePills();
  updateRetargetFpsPlaceholder();
  if (state.robot) switchInspectorPanel("robot");
  await refreshRetargetPanel();
  toast(`已加载 ${payload.name}`);
}

function datasetSceneGlbUrl(token, o) {
  const mesh = o.mesh_file || "";
  if (!token || !mesh) return null;
  return `/api/dataset/scene_glb?token=${encodeURIComponent(token)}&mesh=${encodeURIComponent(mesh)}`;
}

async function loadRobotExportPreview(result) {
  if (state.calibrationMode) {
    toast("标定模式下无法预览机器人轨迹", true);
    return;
  }

  state.motion = null;
  state.libraryEntry = null;
  state.exportToken = null;
  skel.clear();
  mesh.clear();
  skin.clear();
  envView.clear();
  envView.group.visible = false;

  const robotName = result.robot;
  if (!state.robot || state.robot.name !== robotName) {
    const robotData = await API.post("/api/robot/select", { name: robotName });
    state.robot = robotData;
    await robot.load(robotData);
  }

  state.robotTrajectory = result.trajectory;
  robot.setTrajectory(result.trajectory);

  scaledSkel.clear();
  scaledSkel.group.visible = false;
  const clipDur = Math.max(0.1, (result.num_frames - 1) / (result.framerate || 30));
  if (result.scaled_scene) {
    scaledEnv.load(result.scaled_scene, result.preview_token, {
      duration: clipDur,
      objectGlbUrl: (o) => datasetSceneGlbUrl(result.preview_token, o),
    });
    document.getElementById("tg-scaled-env").disabled = false;
    setViewVisible(scaledEnv, "tg-scaled-env", true);
  } else {
    scaledEnv.clear();
    scaledEnv.group.visible = false;
    syncEnvToggleButton();
  }

  setViewVisible(skel, "tg-skeleton", false);
  setBodyVisible(false);
  setViewVisible(mesh, "tg-mesh", false);
  setViewVisible(scaledSkel, "tg-scaled", false);
  document.getElementById("tg-scaled").disabled = true;
  document.getElementById("tg-robot").disabled = false;
  setViewVisible(robot, "tg-robot", true);

  document.getElementById("motion-meta-card").style.display = "none";
  player.ready(robot.clipDuration || clipDur);
  player.setPlaying(true);
  robot.group.getWorldPosition(_camFocus);
  orbit.target.copy(_camFocus);
  _orbitManualUntil = 0;
  revealStage();
  updatePills();
  toast(`机器人 mesh 播放：${result.name}`);
}

async function previewRobotClip(entry, robotName) {
  const label = entry.stem || entry.sequence_id || "";
  showLoading(`加载机器人轨迹 ${label}`.trim());
  try {
    const body = { source_path: entry.source_path };
    if (robotName) body.robot = robotName;
    const { job_id } = await API.post("/api/dataset/preview_robot", body);
    const result = await waitMotionJob(job_id, (frac, sub) => {
      setLoadingProgress(frac, sub);
    });
    setLoadingProgress(1, "构建机器人场景…");
    await loadRobotExportPreview(result);
    return result;
  } catch (e) {
    toast(e.message, true);
    throw e;
  } finally {
    hideLoading();
  }
}

async function populateDvRobotSelect(preferred) {
  const sel = document.getElementById("dv-robot-select");
  if (!sel) return preferred || "";
  const data = await API.get("/api/robots");
  const prev = preferred || sel.value;
  sel.innerHTML = "";
  for (const r of data.robots || []) {
    if (!r.has_urdf) continue;
    const opt = document.createElement("option");
    opt.value = r.name;
    opt.textContent = r.display_name || r.name;
    sel.appendChild(opt);
  }
  if (prev && [...sel.options].some((o) => o.value === prev)) {
    sel.value = prev;
  } else if (sel.options.length) {
    sel.selectedIndex = 0;
  }
  return sel.value;
}

async function loadLibraryEntry(entry) {
  const label = entry.stem || entry.sequence_id || "";
  showLoading(`加载动作中… ${label}`.trim());
  try {
    const { job_id } = await API.post("/api/motion/load_library", entry);
    const payload = await waitMotionJob(job_id, (frac, sub) => {
      setLoadingProgress(frac, sub);
    });
    setLoadingProgress(1, "构建场景…");
    await loadMotionPayload(payload);
  } catch (e) {
    toast(e.message, true);
  } finally {
    hideLoading();
  }
}

// library navigator
let libMotionsRoot = "";

async function linkLibraryPath() {
  const hint = libMotionsRoot
    ? `链接到资源库目录（${libMotionsRoot}）`
    : "链接到资源库（~/.config/hhtools/motions）";
  const path = window.prompt(hint, "");
  if (!path?.trim()) return;
  try {
    const data = await API.post("/api/library/link", { path: path.trim() });
    if (data.motions_library_root) libMotionsRoot = data.motions_library_root;
    updateMotionsLibraryHint();
    await refreshLibrary();
    const sel = document.getElementById("lib-folder");
    if (sel && data.folder_label) sel.value = data.folder_label;
    renderLibrary();
    toast(`已链接：${data.folder_label}（${data.clip_count} clip）`);
  } catch (e) {
    toast(e.message, true);
  }
}

function updateMotionsLibraryHint() {
  const el = document.getElementById("lib-motions-hint");
  if (!el) return;
  if (!libMotionsRoot) {
    el.textContent = "";
    return;
  }
  el.innerHTML =
    `拖入数据集会自动软链接到 <code>${libMotionsRoot}</code>；`
    + "建议将常用数据集中放到该目录。";
}

// library navigator
let libEntries = [];
let libSourceRoot = "";
async function refreshLibrary() {
  const list = document.getElementById("lib-list");
  try {
    const data = await API.get("/api/library");
    libEntries = data.entries || [];
    libSourceRoot = data.source_root || "";
    if (data.motions_library_root) libMotionsRoot = data.motions_library_root;
    updateMotionsLibraryHint();
    // populate folder dropdown
    const sel = document.getElementById("lib-folder");
    sel.innerHTML = `<option value="">全部目录 (${(data.folders || []).length})</option>`;
    for (const f of data.folders || []) {
      const o = document.createElement("option");
      o.value = f; o.textContent = f;
      sel.appendChild(o);
    }
    renderLibrary();
  } catch (e) {
    document.getElementById("lib-count").textContent = "加载失败";
    list.innerHTML = `<div class="hint" style="padding:12px">无法读取资源库：${e.message}</div>`;
  }
}
function renderLibrary() {
  const query = document.getElementById("lib-search").value || "";
  const folder = document.getElementById("lib-folder").value || "";
  const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
  const list = document.getElementById("lib-list");
  list.innerHTML = "";
  const filtered = libEntries.filter((e) => {
    if (folder && e.folder_label !== folder) return false;
    const hay = (e.folder_label + " " + e.stem).toLowerCase();
    return tokens.every((t) => hay.includes(t));
  });
  document.getElementById("lib-count").textContent =
    libEntries.length ? `${filtered.length} / ${libEntries.length} clip` : "";

  if (!libEntries.length) {
    list.innerHTML = `<div class="hint" style="padding:12px">在 <b>${libSourceRoot || "assets/motions"}</b> 未找到可识别的 clip。<br>直接拖入文件夹，会自动软链接到 <code>${libMotionsRoot || "~/.config/hhtools/motions"}</code>。</div>`;
    return;
  }
  if (!filtered.length) {
    list.innerHTML = `<div class="hint" style="padding:12px">没有匹配「${query}${folder ? " @" + folder : ""}」的结果</div>`;
    return;
  }
  for (const e of filtered.slice(0, 300)) {
    const row = document.createElement("div");
    row.className = "lib-row";
    row.innerHTML = `<span class="lr-folder">${e.folder_label}</span>
      <span class="lr-stem">${e.stem}</span>
      <button class="lr-add" title="加入篮子">＋</button>`;
    row.onclick = () => loadLibraryEntry(e);
    row.querySelector(".lr-add").onclick = (ev) => { ev.stopPropagation(); addToBasket([e]); };
    list.appendChild(row);
  }
  if (filtered.length > 300) {
    const more = document.createElement("div");
    more.className = "hint";
    more.style.padding = "8px 10px";
    more.textContent = `… 还有 ${filtered.length - 300} 条，继续输入以缩小范围`;
    list.appendChild(more);
  }
}
document.getElementById("lib-search").oninput = () => renderLibrary();
document.getElementById("lib-folder").onchange = () => renderLibrary();

// drag-drop helpers (folder-aware)
function readAllDirectoryEntries(reader) {
  return new Promise((resolve, reject) => {
    const entries = [];
    const readBatch = () => {
      reader.readEntries((batch) => {
        if (!batch.length) {
          resolve(entries);
          return;
        }
        entries.push(...batch);
        readBatch();
      }, reject);
    };
    readBatch();
  });
}

function walkEntry(entry, out, prefix = "") {
  return new Promise((resolve, reject) => {
    if (entry.isFile) {
      entry.file((f) => {
        f._relpath = prefix + f.name;
        out.push(f);
        resolve();
      }, reject);
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      readAllDirectoryEntries(reader)
        .then((entries) => Promise.all(
          entries.map((e) => walkEntry(e, out, `${prefix}${entry.name}/`)),
        ))
        .then(() => resolve())
        .catch(reject);
    } else {
      resolve();
    }
  });
}

async function collectDroppedFiles(dataTransfer) {
  const files = [];
  // Prefer the entry API: it recurses into dropped folders AND distinguishes a
  // real file from a *directory*.  A dropped folder shows up in
  // ``dataTransfer.files`` as a single zero-byte, type-less File whose body
  // cannot be read — appending it to FormData makes the upload ``fetch`` reject
  // with "Failed to fetch".  ``webkitGetAsEntry`` must be called synchronously
  // while the drop event's items are still alive, so capture every entry first,
  // then walk them.
  const items = dataTransfer?.items;
  if (items?.length) {
    const entries = [];
    const looseFiles = [];
    for (const it of items) {
      const entry = it.webkitGetAsEntry?.();
      if (entry) entries.push(entry);
      else {
        const f = it.getAsFile?.();
        if (f) looseFiles.push(f);
      }
    }
    if (entries.length) await Promise.all(entries.map((e) => walkEntry(e, files)));
    for (const f of looseFiles) {
      f._relpath = f._relpath || f.webkitRelativePath || f.name;
      files.push(f);
    }
    if (files.length) return files;
  }
  // Fallback for browsers without the entry API: a flat file list only. Best-
  // effort skip of a dropped folder, which surfaces here as a zero-byte,
  // type-less, extension-less File that would break the upload fetch.
  if (dataTransfer?.files?.length) {
    for (const f of dataTransfer.files) {
      if (!f) continue;
      const looksLikeDir = f.size === 0 && !f.type && !/\.[^/.]+$/.test(f.name || "");
      if (looksLikeDir) continue;
      f._relpath = f._relpath || f.webkitRelativePath || f.name;
      files.push(f);
    }
  }
  return files;
}

function setupDropzone(el, onFiles) {
  ["dragenter", "dragover"].forEach((ev) =>
    el.addEventListener(ev, (e) => { e.preventDefault(); el.classList.add("hover"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    el.addEventListener(ev, (e) => { e.preventDefault(); el.classList.remove("hover"); })
  );
  el.addEventListener("drop", (e) => {
    e.stopPropagation();
    el.classList.remove("hover");
    void collectDroppedFiles(e.dataTransfer).then((files) => {
      if (files.length) onFiles(files);
    });
  });
}
// Hidden <input> based file / folder picker (for environments where native
// drag-drop is awkward). Folder picker preserves relative paths via
// webkitRelativePath so mesh subdirs + sidecars survive.
function pickFiles({ folder = false } = {}) {
  return new Promise((resolve) => {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.multiple = true;
    if (folder) inp.webkitdirectory = true;
    inp.style.display = "none";
    inp.onchange = () => {
      const files = Array.from(inp.files || []);
      for (const f of files) f._relpath = f.webkitRelativePath || f.name;
      document.body.removeChild(inp);
      resolve(files);
    };
    document.body.appendChild(inp);
    inp.click();
  });
}

function inferLibraryFolderLabel(files) {
  if (!files?.length) return undefined;
  const rels = files.map((f) => f._relpath || f.name);
  if (rels.some((r) => r.includes("/"))) return rels[0].split("/")[0];
  return undefined;
}

async function ingestMotionFiles(files, profile = "mimic") {
  if (!files || !files.length) return;
  const libraryFolderLabel = inferLibraryFolderLabel(files);
  showLoading(`链接并解析中… (${files.length} 个文件)`);
  try {
    const uploadResp = await uploadFilesXHR(
      "/api/motion/upload",
      files,
      { profile, libraryFolderLabel },
      () => {},
    );
    const { job_id, linked, folder_label, materialize_mode } = uploadResp;
    const payload = await waitMotionJob(job_id, (frac, sub) => {
      setLoadingProgress(frac, sub);
    }, { uploadFrac: 0 });
    setLoadingProgress(1, "构建场景…");
    await loadMotionPayload(payload);
    if (linked || folder_label || payload.linked_folder) {
      await refreshLibrary();
      const label = folder_label || payload.linked_folder;
      if (label) {
        const sel = document.getElementById("lib-folder");
        if (sel) sel.value = label;
        renderLibrary();
      }
    }
    const modeHint = materialize_mode === "symlink" ? "软链接" : "已复制";
    if (payload.library_entry) {
      addToBasket([payload.library_entry]);
      toast(`已${modeHint}并加载：${payload.name}（资源库 · ${folder_label || payload.linked_folder}）`);
    } else if (linked || payload.linked_folder) {
      toast(`已${modeHint}到资源库：${payload.linked_folder || folder_label}，已加载首条 clip`);
    }
  } catch (e) {
    toast(e.message, true);
  } finally {
    hideLoading();
  }
}

function initMotionImportZones() {
  for (const el of document.querySelectorAll(".motion-import-grid [data-profile]")) {
    const profile = el.dataset.profile || "mimic";
    setupDropzone(el, (files) => ingestMotionFiles(files, profile));
  }
  document.querySelectorAll("[data-pick]").forEach((btn) => {
    btn.onclick = async () => {
      const profile = btn.dataset.pick || "mimic";
      const folder = btn.dataset.folder === "1";
      await ingestMotionFiles(await pickFiles({ folder }), profile);
    };
  });
}
initMotionImportZones();
setupDropzone(document.getElementById("stage"), (files) => ingestMotionFiles(files, "mimic"));

document.getElementById("add-to-basket").onclick = () => {
  if (state.libraryEntry) {
    addToBasket([state.libraryEntry]);
    return;
  }
  toast("请从资源库加载动作后再加入篮子，或使用资源库列表行的 ＋", true);
};

// =================================================================  ROBOT
let _robotPanelLockDepth = 0;

function setRobotPanelLocked(locked) {
  if (locked) _robotPanelLockDepth++;
  else _robotPanelLockDepth = Math.max(0, _robotPanelLockDepth - 1);
  const busy = _robotPanelLockDepth > 0;
  state.robotPanelLocked = busy;

  const sel = document.getElementById("robot-select");
  if (sel) sel.disabled = busy;
  for (const id of ["robot-load-btn", "robot-pick-urdf", "robot-pick-mesh-folder"]) {
    const el = document.getElementById(id);
    if (el) el.disabled = busy;
  }
  const delBtn = document.getElementById("robot-delete-btn");
  if (delBtn && busy) delBtn.disabled = true;
  if (!busy) updateRobotDeleteBtn();
  for (const id of ["robot-drop-urdf", "robot-drop-mesh"]) {
    document.getElementById(id)?.classList.toggle("disabled", busy);
  }
}

async function applyRobot(robotData) {
  if (state.robotPanelLocked) {
    toast("Retarget 进行中，请等待完成后再切换机器人", true);
    return;
  }
  state.robot = robotData;
  await robot.load(robotData);
  document.getElementById("robot-meta-card").style.display = "block";
  document.getElementById("robot-name").textContent = robotData.display_name;
  document.getElementById("robot-meta").innerHTML = `
    <div class="meta-row"><span class="k">链接 links</span><span class="v">${robotData.links.length}</span></div>
    <div class="meta-row"><span class="k">自由度 DOF</span><span class="v">${robotData.num_dof}</span></div>
    <div class="meta-row"><span class="k">ik_map 槽位</span><span class="v">${Object.keys(robotData.ik_map).length}</span></div>`;
  document.getElementById("batch-robot").textContent = robotData.display_name;
  void syncBatchRefHint();
  renderBasket();
  updatePills();
  const tgRobot = document.getElementById("tg-robot");
  tgRobot.disabled = false;
  setViewVisible(robot, "tg-robot", true);
  revealStage();
  // Await so state.calibration is fresh; refreshRetargetPanel itself loads the
  // scaled skeleton/scene when a calibration already exists (no retarget needed).
  if (state.calibrationMode) {
    switchInspectorPanel("robot");
    await enterCalibrationMode(state.calibQ);
    toast(`机器人已加载（标定姿态）：${robotData.display_name}`);
    return;
  }
  await routeAfterRobotLoad();
  toast(
    state.motion
      ? `机器人已加载：${robotData.display_name}`
      : `机器人已加载：${robotData.display_name} — 请先加载动作`,
  );
}

function updateRobotDeleteBtn() {
  const sel = document.getElementById("robot-select");
  const btn = document.getElementById("robot-delete-btn");
  if (!sel || !btn) return;
  const opt = sel.selectedOptions[0];
  const deletable = opt?.dataset.deletable === "1";
  btn.style.display = deletable ? "" : "none";
  btn.dataset.deletable = deletable ? "1" : "0";
  btn.disabled = state.robotPanelLocked || !deletable;
}

async function refreshRobotList() {
  try {
    const data = await API.get("/api/robots");
    const sel = document.getElementById("robot-select");
    const hint = document.getElementById("robot-library-hint");
    const prev = sel.value;
    sel.innerHTML = "";
    for (const r of data.robots) {
      const opt = document.createElement("option");
      opt.value = r.name;
      opt.dataset.deletable = r.deletable ? "1" : "0";
      const tag = r.deletable ? " · 用户库" : "";
      opt.textContent = `${r.display_name} (${r.num_dof} DOF)${tag}${r.has_urdf ? "" : " — 无URDF"}`;
      opt.disabled = !r.has_urdf;
      sel.appendChild(opt);
    }
    if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
    if (hint && data.library_dir) {
      hint.innerHTML = `通过 UI 注册的机器人保存在 <code>${data.library_dir}</code>，重启 <code>hhtools web</code> 后仍可用。`;
    }
    updateRobotDeleteBtn();
  } catch (e) { /* ignore */ }
}
document.getElementById("robot-select")?.addEventListener("change", updateRobotDeleteBtn);
document.getElementById("robot-load-btn").onclick = async () => {
  if (state.robotPanelLocked) {
    toast("Retarget 进行中，请等待完成后再切换机器人", true);
    return;
  }
  const name = document.getElementById("robot-select").value;
  if (!name) return;
  toast("加载机器人…");
  try { await applyRobot(await API.post("/api/robot/select", { name })); }
  catch (e) { toast(e.message, true); }
};
document.getElementById("robot-delete-btn").onclick = async () => {
  if (state.robotPanelLocked) {
    toast("Retarget 进行中，请等待完成后再操作", true);
    return;
  }
  const sel = document.getElementById("robot-select");
  const name = sel?.value;
  if (!name) return;
  const label = sel.selectedOptions[0]?.textContent || name;
  if (!confirm(`确定从资源库删除「${label}」？\n将永久删除对应目录，不可恢复。`)) return;
  toast("删除机器人…");
  try {
    await API.delete(`/api/robot/${encodeURIComponent(name)}`);
    if (state.robot?.name === name) {
      state.robot = null;
      robot.group.visible = false;
      document.getElementById("robot-meta-card").style.display = "none";
      document.getElementById("robot-pill").textContent = "未加载机器人";
      document.getElementById("batch-robot").textContent = "未加载";
      renderBasket();
      refreshRetargetPanel();
    }
    await refreshRobotList();
    toast(`已从资源库删除：${name}`);
  } catch (e) { toast(e.message, true); }
};

const robotImport = { urdf: null, meshes: [] };

function isUrdfFile(f) {
  return (f._relpath || f.name).toLowerCase().endsWith(".urdf");
}
function isMeshFile(f) {
  const p = (f._relpath || f.name).toLowerCase();
  return /\.(stl|obj|dae|ply|glb|gltf)$/i.test(p);
}
function updateRobotImportStatus() {
  const el = document.getElementById("robot-import-status");
  if (!el) return;
  const parts = [];
  if (robotImport.urdf) parts.push(`URDF：${robotImport.urdf.name || "robot.urdf"}`);
  if (robotImport.meshes.length) parts.push(`Mesh：${robotImport.meshes.length} 个文件`);
  if (robotImport.urdf && !robotImport.meshes.length) {
    parts.push("请接着拖入 meshes/ 文件夹完成注册");
  }
  el.textContent = parts.length ? parts.join(" · ") : "尚未选择 URDF。";
}

async function tryUploadRobot() {
  if (state.robotPanelLocked) {
    toast("Retarget 进行中，请等待完成后再切换机器人", true);
    return;
  }
  if (!robotImport.urdf) {
    toast("请先放入 .urdf 文件", true);
    return;
  }
  // The backend wipes the upload dir on every call, so URDF + meshes MUST be
  // sent together.  ``name`` is passed as a query param so the temp dir matches
  // the URDF stem (the registered preset name still comes from the URDF's
  // ``<robot name>`` during scaffolding).
  const files = [robotImport.urdf, ...robotImport.meshes];
  const name = (robotImport.urdf.name || "robot")
    .replace(/\.urdf$/i, "")
    .replace(/[^a-z0-9_]/gi, "_")
    .toLowerCase();
  toast(`上传机器人… (${files.length} 个文件)`);
  try {
    const robotData = await API.upload("/api/robot/upload", files, { name });
    await applyRobot(robotData);
    // The clip is now a registered preset (name derived from the URDF) — show
    // it in the "已注册机器人" list and select it.
    await refreshRobotList();
    const sel = document.getElementById("robot-select");
    if (sel && robotData.name) sel.value = robotData.name;
    robotImport.urdf = null;
    robotImport.meshes = [];
    updateRobotImportStatus();
    toast(`机器人已注册：${robotData.display_name || robotData.name}`);
  } catch (e) { toast(e.message, true); }
}

function ingestRobotUrdf(files) {
  if (state.robotPanelLocked) {
    toast("Retarget 进行中，请等待完成后再切换机器人", true);
    return;
  }
  if (!files?.length) return;
  const urdf = files.find(isUrdfFile);
  if (!urdf) { toast("此区域需要 .urdf 文件", true); return; }
  robotImport.urdf = urdf;
  const extra = files.filter((f) => f !== urdf && (isMeshFile(f) || !isUrdfFile(f)));
  if (extra.length) robotImport.meshes = [...robotImport.meshes, ...extra];
  updateRobotImportStatus();
  // Only upload now when the same drop already carried the meshes (a whole
  // robot folder).  A bare .urdf drop must WAIT for step 2 (the meshes/ folder)
  // — uploading immediately used to register a mesh-less robot and reset the
  // stored URDF, so the subsequent meshes drop hit "请先放入 .urdf 文件".
  if (robotImport.meshes.length) {
    tryUploadRobot();
  } else {
    toast("已读取 URDF，请接着拖入 meshes/ 文件夹完成注册");
  }
}

function ingestRobotMesh(files) {
  if (state.robotPanelLocked) {
    toast("Retarget 进行中，请等待完成后再切换机器人", true);
    return;
  }
  if (!files?.length) return;
  const meshes = files.filter((f) => !isUrdfFile(f));
  if (!meshes.length) { toast("未找到 mesh 文件", true); return; }
  if (!robotImport.urdf) {
    toast("请先在「1 · URDF 文件」区域放入 .urdf，再拖入 meshes/", true);
    return;
  }
  robotImport.meshes = meshes;
  updateRobotImportStatus();
  tryUploadRobot();
}

setupDropzone(document.getElementById("robot-drop-urdf"), ingestRobotUrdf);
setupDropzone(document.getElementById("robot-drop-mesh"), ingestRobotMesh);
document.getElementById("robot-pick-urdf").onclick = async () =>
  ingestRobotUrdf(await pickFiles());
document.getElementById("robot-pick-mesh-folder").onclick = async () =>
  ingestRobotMesh(await pickFiles({ folder: true }));

// =================================================================  CALIBRATION 3D MANIPULATOR
const _hhtoolsWorld = new THREE.Vector3();
const _hhtoolsAxis = new THREE.Vector3();
const _projScratch = new THREE.Vector3();
const _dragPlane = new THREE.Plane();
const _arcRef = new THREE.Vector3();
const _arcCross = new THREE.Vector3();

/** Map hhtools Z-up coordinates to three.js world (inside the rotated ``world`` group). */
function hhtoolsToWorldVec3(x, y, z, out = _hhtoolsWorld) {
  out.set(x, y, z);
  return out.applyMatrix4(world.matrixWorld);
}

/** Point on a rotation arc: pivot + R·(cos θ·ref + sin θ·(axis×ref)). */
function arcPointWorld(pivot, axis, ref, angle, radius) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  _arcCross.crossVectors(axis, ref);
  return pivot.clone()
    .add(ref.clone().multiplyScalar(c * radius))
    .add(_arcCross.multiplyScalar(s * radius));
}

class CalibManipulator {
  constructor({ canvasEl, hudEl, stageEl }) {
    this.canvas = canvasEl;
    this.hud = hudEl;
    this.stage = stageEl;
    this.active = false;
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.jointMeta = {};
    this.linkToJoint = {};
    this.jointToLink = {};
    this.jointWorld = {};
    this.selected = null;
    this.hoveredLink = null;
    this.hoveredJoint = null;
    this.dragging = false;
    this._dragRef = null;
    this._dragStartQ = 0;
    this._tags = new Map();
    this._limitGroup = null;
    this._pickScreen = null;
    this._pickAnchor = null;
    this._hudPinned = null;
    this._hudCardDrag = null;
    this._onDown = (e) => this._pointerDown(e);
    this._onMove = (e) => this._pointerMove(e);
    this._onUp = () => this._pointerUp();
  }

  _defaultCtx() {
    return {
      robotView: robot,
      getQ: () => state.calibQ,
      getSliderRows: () => state.calibSliderRows,
      jointChange: (name, val, opts) => setCalibJointValue(name, val, opts),
      previewFk: (opts) => previewCalibPose(opts),
    };
  }

  start(limitsList, ctx = null) {
    this.active = true;
    this._ctx = ctx || this._defaultCtx();
    this.jointMeta = {};
    this.linkToJoint = {};
    this.jointToLink = {};
    for (const L of limitsList || []) {
      if (!L.name || L.type === "fixed") continue;
      const lo = L.lower != null ? L.lower : -Math.PI;
      const hi = L.upper != null ? L.upper : Math.PI;
      this.jointMeta[L.name] = {
        child_link: L.child_link,
        lower: lo,
        upper: hi,
        type: L.type || "revolute",
      };
      if (L.child_link) {
        this.linkToJoint[L.child_link] = L.name;
        this.jointToLink[L.name] = L.child_link;
      }
    }
    this.hud.classList.remove("hidden");
    this.hud.setAttribute("aria-hidden", "false");
    this.stage.classList.add("calib-pickable");
    this._initLimitGizmo();
    this._buildTags();
    this.canvas.addEventListener("pointerdown", this._onDown);
    window.addEventListener("pointermove", this._onMove);
    window.addEventListener("pointerup", this._onUp);
    window.addEventListener("pointercancel", this._onUp);
  }

  stop() {
    this.active = false;
    this.selected = null;
    this.hoveredLink = null;
    this.hoveredJoint = null;
    this.dragging = false;
    this._pickScreen = null;
    this._pickAnchor = null;
    this._hudPinned = null;
    this._hudCardDrag = null;
    this.hud.innerHTML = "";
    this.hud.classList.add("hidden");
    this.hud.setAttribute("aria-hidden", "true");
    this.stage.classList.remove("calib-pickable", "calib-dragging", "calib-hover-joint");
    this._tags.clear();
    this._disposeLimitGizmo();
    (this._ctx?.robotView || robot).setCalibHighlights({});
    this._ctx = null;
    document.getElementById("calib-hover-hint")?.classList.remove("show");
    this.canvas.removeEventListener("pointerdown", this._onDown);
    window.removeEventListener("pointermove", this._onMove);
    window.removeEventListener("pointerup", this._onUp);
    window.removeEventListener("pointercancel", this._onUp);
    orbit.enabled = true;
  }

  _initLimitGizmo() {
    this._disposeLimitGizmo();
    const g = new THREE.Group();
    const arcMat = new THREE.LineBasicMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.85 });
    const loMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const hiMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const curMat = new THREE.MeshBasicMaterial({ color: 0x2563eb });
    const needleMat = new THREE.LineBasicMaterial({ color: 0x2563eb, linewidth: 2 });
    const tickGeo = new THREE.SphereGeometry(0.012, 10, 10);
    this._limitGroup = {
      group: g,
      arc: new THREE.Line(new THREE.BufferGeometry(), arcMat),
      loTick: new THREE.Mesh(tickGeo, loMat),
      hiTick: new THREE.Mesh(tickGeo.clone(), hiMat),
      curTick: new THREE.Mesh(tickGeo.clone(), curMat),
      needle: new THREE.Line(new THREE.BufferGeometry(), needleMat),
    };
    g.add(this._limitGroup.arc, this._limitGroup.loTick, this._limitGroup.hiTick,
      this._limitGroup.curTick, this._limitGroup.needle);
    g.visible = false;
    world.add(g);
  }

  _disposeLimitGizmo() {
    if (!this._limitGroup) return;
    world.remove(this._limitGroup.group);
    this._limitGroup.arc.geometry.dispose();
    this._limitGroup.needle.geometry.dispose();
    this._limitGroup.loTick.geometry.dispose();
    this._limitGroup.hiTick.geometry.dispose();
    this._limitGroup.curTick.geometry.dispose();
    this._limitGroup = null;
  }

  _buildTags() {
    this.hud.innerHTML = "";
    this._tags.clear();
    for (const name of Object.keys(this.jointMeta)) {
      const meta = this.jointMeta[name];
      const card = document.createElement("div");
      card.className = "calib-hud-card";
      card.dataset.joint = name;

      const head = document.createElement("div");
      head.className = "calib-hud-head calib-hud-drag-handle";
      head.title = "拖动标题栏移动控件";
      const grip = document.createElement("span");
      grip.className = "calib-hud-grip";
      grip.setAttribute("aria-hidden", "true");
      grip.textContent = "⋮⋮";
      const nameEl = document.createElement("span");
      nameEl.className = "joint-name";
      nameEl.textContent = name;
      nameEl.title = name;
      const unit = document.createElement("span");
      unit.className = "joint-unit";
      unit.textContent = "rad";
      head.append(grip, nameEl, unit);

      const limitRow = document.createElement("div");
      limitRow.className = "calib-limit-row";
      const loEl = document.createElement("span");
      loEl.className = "limit-end limit-lo";
      loEl.textContent = meta.lower.toFixed(2);
      const track = document.createElement("div");
      track.className = "limit-track";
      const fill = document.createElement("div");
      fill.className = "limit-fill";
      const thumb = document.createElement("div");
      thumb.className = "limit-thumb";
      track.appendChild(fill);
      track.appendChild(thumb);
      const hiEl = document.createElement("span");
      hiEl.className = "limit-end limit-hi";
      hiEl.textContent = meta.upper.toFixed(2);
      limitRow.append(loEl, track, hiEl);

      const input = document.createElement("input");
      input.type = "number";
      input.className = "calib-angle-input";
      input.step = "0.001";
      input.value = "0.000";
      input.min = String(meta.lower);
      input.max = String(meta.upper);
      input.addEventListener("input", () => {
        this._ctx.jointChange(name, input.value, { from: "hud-input", live: true });
      });
      input.addEventListener("change", () => {
        this._ctx.jointChange(name, input.value, { from: "hud-input" });
      });
      input.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") input.blur();
        ev.stopPropagation();
      });
      input.addEventListener("pointerdown", (ev) => ev.stopPropagation());

      card.append(head, limitRow, input);
      this.hud.appendChild(card);
      this._bindHudCardDrag(card, head);
      this._bindHudTrackDrag(name, track, thumb, meta);
      this._tags.set(name, { el: card, input, nameEl, loEl, hiEl, track, thumb, fill });
    }
  }

  _hudLayout() {
    const canvasRect = this.canvas.getBoundingClientRect();
    const hudRect = this.hud.getBoundingClientRect();
    return {
      ox: canvasRect.left - hudRect.left,
      oy: canvasRect.top - hudRect.top,
      w: canvasRect.width,
      h: canvasRect.height,
      cardW: 180,
      cardH: 112,
      pad: 14,
    };
  }

  _applyHudPin(el, x, y, layout = this._hudLayout()) {
    const { w, h, cardW, cardH, pad } = layout;
    const clamped = this._clampHudCard(x, y, w, h, cardW, cardH, pad);
    el.classList.remove("screen-docked", "screen-pick");
    el.classList.add("user-pinned", "visible");
    el.style.left = `${clamped.x}px`;
    el.style.top = `${clamped.y}px`;
    return clamped;
  }

  _bindHudCardDrag(card, head) {
    const onDown = (e) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      e.preventDefault();
      const layout = this._hudLayout();
      const hudRect = this.hud.getBoundingClientRect();
      const cardRect = card.getBoundingClientRect();
      card.classList.add("user-pinned", "is-dragging");
      const anchorX = cardRect.left - hudRect.left + cardRect.width * 0.5;
      const anchorY = cardRect.top - hudRect.top + cardRect.height * 0.5;
      const start = { px: e.clientX, py: e.clientY, ax: anchorX, ay: anchorY };
      this._hudCardDrag = true;
      orbit.enabled = false;
      try { head.setPointerCapture(e.pointerId); } catch { /* ignore */ }
      const onMove = (ev) => {
        const x = start.ax + (ev.clientX - start.px);
        const y = start.ay + (ev.clientY - start.py);
        this._hudPinned = { x, y };
        this._applyHudPin(card, x, y, layout);
      };
      const onUp = (ev) => {
        this._hudCardDrag = false;
        card.classList.remove("is-dragging");
        orbit.enabled = true;
        try { head.releasePointerCapture(ev.pointerId); } catch { /* ignore */ }
        head.removeEventListener("pointermove", onMove);
        head.removeEventListener("pointerup", onUp);
        head.removeEventListener("pointercancel", onUp);
      };
      head.addEventListener("pointermove", onMove);
      head.addEventListener("pointerup", onUp);
      head.addEventListener("pointercancel", onUp);
    };
    head.addEventListener("pointerdown", onDown);
  }

  _bindHudTrackDrag(name, track, thumb, meta) {
    const tag = () => this._tags.get(name);
    const paintThumb = (clientX) => {
      const rect = track.getBoundingClientRect();
      const t = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      const pct = `${(t * 100).toFixed(2)}%`;
      const row = tag();
      if (row) {
        row.thumb.style.left = pct;
        row.fill.style.width = pct;
      }
      return meta.lower + t * (meta.upper - meta.lower);
    };
    const move = (clientX) => {
      const val = paintThumb(clientX);
      this._ctx.jointChange(name, val, { from: "hud-track", live: true });
    };
    const onDown = (e) => {
      e.stopPropagation();
      e.preventDefault();
      this._hudTrackDrag = name;
      this.setSelected(name);
      const row = tag();
      row?.el.classList.add("track-dragging");
      this.stage.classList.add("calib-dragging");
      orbit.enabled = false;
      try { track.setPointerCapture(e.pointerId); } catch { /* ignore */ }
      move(e.clientX);
      const onMove = (ev) => { if (this._hudTrackDrag === name) move(ev.clientX); };
      const onUp = (ev) => {
        if (this._hudTrackDrag !== name) return;
        this._hudTrackDrag = null;
        row?.el.classList.remove("track-dragging");
        this.stage.classList.remove("calib-dragging");
        orbit.enabled = true;
        this._ctx.previewFk({ flush: true });
        try { track.releasePointerCapture(ev.pointerId); } catch { /* ignore */ }
        track.removeEventListener("pointermove", onMove);
        track.removeEventListener("pointerup", onUp);
        track.removeEventListener("pointercancel", onUp);
      };
      track.addEventListener("pointermove", onMove);
      track.addEventListener("pointerup", onUp);
      track.addEventListener("pointercancel", onUp);
    };
    track.addEventListener("pointerdown", onDown);
    thumb.addEventListener("pointerdown", onDown);
  }

  setSelected(jointName, { scrollPanel = false } = {}) {
    if (!this.active) return;
    this.selected = jointName;
    for (const [j, { el }] of this._tags) {
      el.classList.toggle("visible", j === jointName);
    }
    const sliderRows = this._ctx.getSliderRows();
    for (const [j, rowRec] of Object.entries(sliderRows)) {
      rowRec.row?.classList.toggle("selected", j === jointName);
    }
    this._syncHighlights();
    this._updateLimitGizmo();
    if (scrollPanel && jointName && sliderRows[jointName]?.row) {
      sliderRows[jointName].row.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }

  _syncHighlights() {
    const selLink = this.selected ? this.jointToLink[this.selected] : null;
    const hovLink = this.hoveredLink;
    this._ctx.robotView.setCalibHighlights({ hover: hovLink, selected: selLink });
    this.stage.classList.toggle("calib-hover-joint", !!(this.hoveredJoint && !this.dragging));
  }

  updateHudValue(jointName, value, { live = false, syncInput = true } = {}) {
    const tag = this._tags.get(jointName);
    if (!tag) return;
    const x = parseFloat(value);
    if (!Number.isFinite(x)) return;
    const meta = this.jointMeta[jointName];
    if (syncInput) tag.input.value = live ? x.toFixed(4) : x.toFixed(3);
    if (meta) {
      const span = meta.upper - meta.lower;
      const t = span > 1e-9 ? (x - meta.lower) / span : 0.5;
      const pct = `${Math.min(100, Math.max(0, t * 100)).toFixed(1)}%`;
      tag.thumb.style.left = pct;
      tag.fill.style.width = pct;
      const atLo = Math.abs(x - meta.lower) < 0.008;
      const atHi = Math.abs(x - meta.upper) < 0.008;
      tag.el.classList.toggle("at-limit-lo", atLo);
      tag.el.classList.toggle("at-limit-hi", atHi);
    }
    if (jointName === this.selected) this._updateLimitGizmo();
  }

  updateJointWorld(jointWorld) {
    this.jointWorld = jointWorld || {};
    this._positionTags();
    if (this.selected) this._updateLimitGizmo();
  }

  _perpRef(axis, pivot) {
    const camDir = camera.position.clone().sub(pivot).normalize();
    _arcRef.crossVectors(axis, camDir);
    if (_arcRef.lengthSq() < 1e-8) _arcRef.crossVectors(axis, new THREE.Vector3(0, 1, 0));
    return _arcRef.normalize();
  }

  _updateLimitGizmo() {
    if (!this._limitGroup || !this.selected) {
      if (this._limitGroup) this._limitGroup.group.visible = false;
      return;
    }
    const joint = this.selected;
    const meta = this.jointMeta[joint];
    const jw = this.jointWorld[joint];
    if (!meta || !jw?.pivot || !jw?.axis || meta.type === "prismatic") {
      this._limitGroup.group.visible = false;
      return;
    }
    const q = this._ctx.getQ()[joint] ?? 0;
    const pivot = hhtoolsToWorldVec3(jw.pivot[0], jw.pivot[1], jw.pivot[2], new THREE.Vector3());
    const axis = hhtoolsToWorldVec3(jw.axis[0], jw.axis[1], jw.axis[2], _hhtoolsAxis).normalize();
    const ref = this._perpRef(axis, pivot);
    const R = 0.11;

    const steps = 36;
    const arcPts = [];
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const ang = meta.lower + (meta.upper - meta.lower) * t;
      arcPts.push(arcPointWorld(pivot, axis, ref, ang, R));
    }
    this._limitGroup.arc.geometry.setFromPoints(arcPts);

    const loP = arcPointWorld(pivot, axis, ref, meta.lower, R);
    const hiP = arcPointWorld(pivot, axis, ref, meta.upper, R);
    const curP = arcPointWorld(pivot, axis, ref, q, R);
    this._limitGroup.loTick.position.copy(loP);
    this._limitGroup.hiTick.position.copy(hiP);
    this._limitGroup.curTick.position.copy(curP);
    this._limitGroup.needle.geometry.setFromPoints([pivot, curP]);

    this._limitGroup.group.visible = true;
  }

  _clampHudCard(sx, sy, w, h, cardW, cardH, pad) {
    return {
      x: Math.min(w - pad - cardW * 0.5, Math.max(pad + cardW * 0.5, sx)),
      y: Math.min(h - pad, Math.max(pad + cardH, sy)),
    };
  }

  _projectToHud(worldPoint, w, h, ox, oy, out = new THREE.Vector3()) {
    out.copy(worldPoint).project(camera);
    return {
      x: (out.x * 0.5 + 0.5) * w + ox,
      y: (-out.y * 0.5 + 0.5) * h + oy,
      inFront: out.z >= -1 && out.z <= 1,
    };
  }

  _positionTags() {
    if (!this.active || this._hudCardDrag) return;
    const layout = this._hudLayout();
    const { ox, oy, w, h } = layout;
    const _proj = new THREE.Vector3();
    for (const [name, { el }] of this._tags) {
      if (!this.selected || name !== this.selected) {
        el.classList.remove("visible", "screen-docked", "screen-pick", "user-pinned", "is-dragging");
        continue;
      }
      const jw = this.jointWorld[name];
      if (!jw?.pivot) continue;

      if (this._hudPinned) {
        this._applyHudPin(el, this._hudPinned.x, this._hudPinned.y, layout);
        continue;
      }

      let sx = w * 0.72 + ox;
      let sy = h * 0.38 + oy;
      let mode = "screen-docked";

      const anchor = this._pickAnchor;
      if (anchor) {
        const hit = this._projectToHud(anchor, w, h, ox, oy, _proj);
        if (hit.inFront) {
          sx = hit.x;
          sy = hit.y - 18;
          mode = "screen-pick";
        }
      } else {
        const pivot = hhtoolsToWorldVec3(jw.pivot[0], jw.pivot[1], jw.pivot[2], _proj);
        const hit = this._projectToHud(pivot, w, h, ox, oy, _proj);
        if (hit.inFront) {
          sx = hit.x;
          sy = hit.y - 18;
          mode = "anchored";
        }
      }

      const clamped = this._clampHudCard(sx, sy, w, h, layout.cardW, layout.cardH, layout.pad);
      el.classList.remove("user-pinned");
      el.classList.toggle("screen-docked", mode === "screen-docked");
      el.classList.toggle("screen-pick", mode === "screen-pick");
      el.style.left = `${clamped.x}px`;
      el.style.top = `${clamped.y}px`;
      el.classList.add("visible");
    }
  }

  _pointerNdc(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  }

  _pickMeshes(clientX, clientY) {
    this._pointerNdc(clientX, clientY);
    this.raycaster.setFromCamera(this.pointer, camera);
    const meshes = [];
    this._ctx.robotView.group.traverse((n) => { if (n.isMesh && n.visible) meshes.push(n); });
    return this.raycaster.intersectObjects(meshes, false);
  }

  _pickLink(clientX, clientY) {
    const hits = this._pickMeshes(clientX, clientY);
    if (!hits.length) return null;
    return this._ctx.robotView._linkForNode(hits[0].object);
  }

  _jointForLink(link) {
    if (!link) return null;
    return this.linkToJoint[link] || null;
  }

  _updateHover(clientX, clientY) {
    const link = this._pickLink(clientX, clientY);
    const joint = this._jointForLink(link);
    this.hoveredLink = link;
    this.hoveredJoint = joint;
    this._syncHighlights();
    if (joint && joint !== this.selected) {
      const hint = document.getElementById("calib-hover-hint");
      if (hint) {
        hint.textContent = joint;
        hint.classList.add("show");
      }
    } else {
      document.getElementById("calib-hover-hint")?.classList.remove("show");
    }
  }

  _pointerDown(e) {
    if (!this.active || e.button !== 0) return;
    if (e.target.closest(".calib-hud-card")) return;
    const hits = this._pickMeshes(e.clientX, e.clientY);
    const joint = this._jointForLink(
      hits.length ? this._ctx.robotView._linkForNode(hits[0].object) : null,
    );
    if (!joint) {
      this.selected = null;
      for (const { el } of this._tags.values()) el.classList.remove("visible");
      for (const rowRec of Object.values(this._ctx.getSliderRows())) {
        rowRec.row?.classList.remove("selected");
      }
      this._updateLimitGizmo();
      this._syncHighlights();
      orbit.enabled = true;
      return;
    }
    e.preventDefault();
    this._pickScreen = { x: e.clientX, y: e.clientY };
    this._pickAnchor = hits[0].point.clone();
    this._hudPinned = null;
    this.setSelected(joint, { scrollPanel: true });
    const meta = this.jointMeta[joint];
    if (!meta || meta.type === "prismatic") {
      orbit.enabled = false;
      return;
    }
    this.dragging = true;
    this._dragRef = null;
    this._dragStartQ = this._ctx.getQ()[joint] ?? 0;
    this.stage.classList.add("calib-dragging");
    orbit.enabled = false;
    try { this.canvas.setPointerCapture(e.pointerId); } catch { /* ignore */ }
  }

  _pointerMove(e) {
    if (!this.active) return;
    if (this.dragging && this.selected) {
      this._applyDrag(e.clientX, e.clientY);
    } else {
      this._updateHover(e.clientX, e.clientY);
    }
    this._positionTags();
  }

  _pointerUp() {
    if (!this.dragging) return;
    this.dragging = false;
    this._dragRef = null;
    this.stage.classList.remove("calib-dragging");
    orbit.enabled = true;
    this._ctx.previewFk({ flush: true });
  }

  _applyDrag(clientX, clientY) {
    const joint = this.selected;
    const jw = this.jointWorld[joint];
    const meta = this.jointMeta[joint];
    if (!jw?.pivot || !jw?.axis || !meta) return;

    const pivot = hhtoolsToWorldVec3(jw.pivot[0], jw.pivot[1], jw.pivot[2], new THREE.Vector3());
    const axis = hhtoolsToWorldVec3(jw.axis[0], jw.axis[1], jw.axis[2], _hhtoolsAxis).normalize();

    this._pointerNdc(clientX, clientY);
    this.raycaster.setFromCamera(this.pointer, camera);
    _dragPlane.setFromNormalAndCoplanarPoint(axis, pivot);
    if (!this.raycaster.ray.intersectPlane(_dragPlane, _projScratch)) return;

    const vec = _projScratch.clone().sub(pivot);
    const len = vec.length();
    if (len < 1e-6) return;
    vec.divideScalar(len);

    if (!this._dragRef) {
      this._dragRef = vec.clone();
      return;
    }

    const cross = new THREE.Vector3().crossVectors(this._dragRef, vec);
    const sinA = axis.dot(cross);
    const cosA = this._dragRef.dot(vec);
    const delta = Math.atan2(sinA, cosA);
    const newQ = Math.min(meta.upper, Math.max(meta.lower, this._dragStartQ + delta));
    this._ctx.jointChange(joint, newQ, { from: "drag", live: true });
  }
}

const calibManip = new CalibManipulator({
  canvasEl: document.getElementById("three-canvas"),
  hudEl: document.getElementById("calib-hud"),
  stageEl: document.getElementById("stage"),
});

// =================================================================  RETARGET / CALIBRATION
function setCalChip(text, cls) {
  document.getElementById("rt-cal").innerHTML =
    `<span class="status-chip ${cls}"><span class="dot"></span>${text}</span>`;
}

function _snapshotVis() {
  const playbar = document.getElementById("playbar");
  return {
    skel: skel.group.visible,
    body: bodyIsVisible(),
    scaled: scaledSkel.group.visible,
    scaledEnv: scaledEnv.group.visible,
    env: envView.group.visible,
    robot: robot.group.visible,
    playing: player.playing,
    t: player.t,
    playbar: playbar ? !playbar.classList.contains("hidden") : false,
  };
}

function _setPlaybarVisible(on) {
  document.getElementById("playbar")?.classList.toggle("hidden", !on);
}

function _setCalibViewTogglesDisabled(disabled) {
  for (const id of ["tg-skeleton", "tg-mesh", "tg-env", "tg-scaled", "tg-scaled-env"]) {
    const btn = document.getElementById(id);
    if (btn) btn.disabled = disabled;
  }
}

function _restoreViewToggleButtons() {
  const skBtn = document.getElementById("tg-skeleton");
  const meshBtn = document.getElementById("tg-mesh");
  if (skBtn) skBtn.disabled = false;
  if (meshBtn) meshBtn.disabled = false;
  syncEnvToggleButton();
  const scaledReady = !!(state.motion && state.robot && state.calibration);
  const ss = document.getElementById("tg-scaled");
  const se = document.getElementById("tg-scaled-env");
  if (ss) ss.disabled = !scaledReady;
  if (se) se.disabled = !scaledReady;
}

function updateCalibBanner(reference) {
  const el = document.getElementById("calib-banner");
  if (!el) return;
  el.innerHTML =
    `<span class="dot"></span>` +
    `<span>标定模式 · 请将灰色机器人对齐到<b>蓝色参考骨架</b>` +
    ` · 点击关节拖动或右栏滑块调整，完成后保存</span>`;
}

function updateR2rCalibBanner() {
  const el = document.getElementById("calib-banner");
  if (!el) return;
  const src = r2r.sourcePayload?.display_name || r2r.sourceName || "源机器人";
  const tgt = r2r.targetPayload?.display_name || r2r.targetName || "目标机器人";
  el.innerHTML =
    `<span class="dot"></span>` +
    `<span>R2R 标定 · 将<b>${tgt}</b>对齐到<b>蓝色 ${src} 参考姿态</b>` +
    ` · 点击关节拖动或右侧滑块调整，完成后保存</span>`;
}

function _applyCalibSceneLayout() {
  state.robotTrajectory = null;
  robot.trajectory = null;
  scaledSkel.clear();
  scaledEnv.clear();
  setViewVisible(skel, "tg-skeleton", false);
  setBodyVisible(false);
  setViewVisible(envView, "tg-env", false);
  setViewVisible(scaledSkel, "tg-scaled", false);
  setViewVisible(scaledEnv, "tg-scaled-env", false);
  setViewVisible(robot, "tg-robot", true);
  robot.applyStatic();
  refSkel.group.visible = true;
  player.setPlaying(false);
  _setPlaybarVisible(false);
  _setCalibViewTogglesDisabled(true);
}

function _restoreVis(snap) {
  if (!snap) return;
  refSkel.clear();
  refSkel.group.visible = false;
  setViewVisible(skel, "tg-skeleton", snap.skel);
  setBodyVisible(snap.body);
  setViewVisible(envView, "tg-env", snap.env);
  setViewVisible(scaledSkel, "tg-scaled", snap.scaled);
  setViewVisible(scaledEnv, "tg-scaled-env", snap.scaledEnv);
  setViewVisible(robot, "tg-robot", snap.robot);
  _setPlaybarVisible(snap.playbar);
  _restoreViewToggleButtons();
  player.t = snap.t;
  player.setPlaying(snap.playing);
  player.refreshFrame();
}

async function enterCalibrationMode(initialQ = null) {
  if (!state.robot || !state.reference) return;
  const calCard = document.getElementById("calib-card");
  calCard.style.display = "block";
  document.getElementById("retarget-btn").disabled = true;
  setCalChip("标定中…", "warn");

  if (!state.calibrationMode) {
    state.calibRestore = _snapshotVis();
  }
  state.calibrationMode = true;
  state.calibNeedsCameraFocus = true;
  state.calibOrbitSaved = {
    minDistance: orbit.minDistance,
    maxDistance: orbit.maxDistance,
    zoomSpeed: orbit.zoomSpeed,
  };
  orbit.zoomSpeed = 0.022;
  applyCalibOrbitLimits();
  updateCalibBanner(state.reference);
  document.getElementById("calib-banner")?.classList.remove("hidden");
  _applyCalibSceneLayout();
  toast("已进入标定模式：请对齐蓝色参考骨架");
  if (player.active) player.seek(0);

  let session;
  try {
    session = await API.post("/api/calibration/session", {
      robot: state.robot.name,
      reference: state.reference,
      motion_token: state.motion?.token || null,
    });
  } catch (e) {
    state.calibrationMode = false;
    state.calibNeedsCameraFocus = false;
    if (state.calibOrbitSaved) {
      orbit.minDistance = state.calibOrbitSaved.minDistance;
      orbit.maxDistance = state.calibOrbitSaved.maxDistance;
      orbit.zoomSpeed = state.calibOrbitSaved.zoomSpeed ?? orbit.zoomSpeed;
      state.calibOrbitSaved = null;
    }
    document.getElementById("calib-banner")?.classList.add("hidden");
    const snap = state.calibRestore;
    state.calibRestore = null;
    _restoreVis(snap);
    toast(e.message, true);
    return;
  }

  state.calibLimits = session.joint_limits || [];
  robot.groundOffset = session.ground_offset_z ?? robot.groundOffset;
  refSkel.load(session.reference);
  if (session.reference_name) updateCalibBanner(session.reference_name);
  _applyCalibSceneLayout();

  const q = initialQ && typeof initialQ === "object"
    ? initialQ
    : (session.joint_q || {});
  state.calibBaselineQ = { ...q };
  state.calibHasSaved = !!session.has_saved_calibration;
  updateCalibRestoreButton();
  calibManip.start(state.calibLimits);
  await buildCalibSliders(q, state.calibLimits);
  calCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function updateCalibRestoreButton() {
  const btn = document.getElementById("calib-restore");
  if (!btn) return;
  btn.disabled = !state.calibHasSaved;
  btn.title = state.calibHasSaved
    ? "恢复到上次保存的标定值"
    : "尚无已保存标定（保存后可重置）";
}

async function exitCalibrationMode() {
  state.calibrationMode = false;
  state.calibNeedsCameraFocus = false;
  if (state.calibOrbitSaved) {
    orbit.minDistance = state.calibOrbitSaved.minDistance;
    orbit.maxDistance = state.calibOrbitSaved.maxDistance;
    orbit.zoomSpeed = state.calibOrbitSaved.zoomSpeed ?? orbit.zoomSpeed;
    state.calibOrbitSaved = null;
  }
  calibManip.stop();
  state.calibSliderRows = {};
  document.getElementById("calib-banner")?.classList.add("hidden");
  state.calibLimits = null;
  state.calibBaselineQ = null;
  state.calibHasSaved = false;
  const snap = state.calibRestore;
  state.calibRestore = null;
  _restoreVis(snap);
  if (robot.trajectory) {
    robot.setFrame(0);
  } else {
    robot.applyStatic();
  }
}

function setCalibJointValue(jointName, value, { from, live = false } = {}) {
  const limByName = {};
  for (const L of state.calibLimits || []) limByName[L.name] = L;
  const lim = limByName[jointName] || {};
  let lo = lim.lower != null ? lim.lower : -Math.PI;
  let hi = lim.upper != null ? lim.upper : Math.PI;
  if (hi <= lo) { lo = -Math.PI; hi = Math.PI; }
  let x = parseFloat(value);
  if (!Number.isFinite(x)) return;
  x = Math.min(hi, Math.max(lo, x));
  state.calibQ[jointName] = x;

  const row = state.calibSliderRows[jointName];
  const prec = live ? 4 : 3;
  if (row) {
    if (from === "slider") {
      row.range.value = String(x);
      row.num.value = x.toFixed(prec);
    } else if (from === "number") {
      row.range.value = String(x);
      if (!live) row.num.value = x.toFixed(prec);
    } else if (from !== "hud-input") {
      row.range.value = String(x);
      row.num.value = x.toFixed(prec);
    }
  }
  if (from === "hud-input") {
    calibManip.updateHudValue(jointName, x, { live, syncInput: false });
  } else {
    calibManip.updateHudValue(jointName, x, { live });
  }
  if (from === "slider" || from === "number") calibManip.setSelected(jointName);
  previewCalibPose({ live });
}

async function buildCalibSliders(initialQ, limitsList) {
  const box = document.getElementById("calib-sliders");
  box.innerHTML = "";
  state.calibQ = {};
  state.calibSliderRows = {};
  if (!state.robot) return;

  const limByName = {};
  for (const L of limitsList || []) limByName[L.name] = L;

  const q = initialQ && typeof initialQ === "object" ? initialQ : {};
  const joints = (limitsList || []).map((L) => L.name)
    .filter(Boolean)
    .concat(state.robot.actuated_joints.filter((j) => !limByName[j]));

  const seen = new Set();
  for (const j of joints) {
    if (seen.has(j)) continue;
    seen.add(j);
    const lim = limByName[j] || {};
    let lo = lim.lower != null ? lim.lower : -Math.PI;
    let hi = lim.upper != null ? lim.upper : Math.PI;
    if (hi <= lo) { lo = -Math.PI; hi = Math.PI; }
    let v = q[j] != null ? parseFloat(q[j]) : 0;
    v = Math.min(hi, Math.max(lo, v));
    state.calibQ[j] = v;

    const row = document.createElement("div");
    row.className = "slider-row";
    row.innerHTML = `<label title="${j}">${j}</label>
      <input type="range" min="${lo}" max="${hi}" step="0.001" value="${v}" />
      <input type="number" class="calib-num" min="${lo}" max="${hi}" step="0.001" value="${v.toFixed(3)}" />`;
    const range = row.querySelector('input[type="range"]');
    const num = row.querySelector(".calib-num");

    state.calibSliderRows[j] = { row, range, num, lo, hi };
    calibManip.updateHudValue(j, v);

    range.oninput = () => setCalibJointValue(j, range.value, { from: "slider", live: true });
    num.oninput = () => setCalibJointValue(j, num.value, { from: "number", live: true });
    num.onchange = () => setCalibJointValue(j, num.value, { from: "number" });
    num.onkeydown = (ev) => {
      if (ev.key === "Enter") { setCalibJointValue(j, num.value, { from: "number" }); num.blur(); }
    };
    row.onclick = () => {
      calibManip._pickScreen = null;
      calibManip._pickAnchor = null;
      calibManip._hudPinned = null;
      calibManip.setSelected(j);
    };
    box.appendChild(row);
  }
  previewCalibPose();
}

let calibFkRaf = 0;
let calibFkInFlight = false;
let calibFkQueued = false;

function previewCalibPose({ live = false, flush = false } = {}) {
  if (!state.robot || !state.calibrationMode) return;
  if (flush) {
    if (calibFkRaf) cancelAnimationFrame(calibFkRaf);
    calibFkRaf = 0;
    _runCalibFk();
    return;
  }
  if (calibFkRaf) return;
  calibFkRaf = requestAnimationFrame(() => {
    calibFkRaf = 0;
    _runCalibFk();
  });
}

async function _runCalibFk() {
  if (calibFkInFlight) {
    calibFkQueued = true;
    return;
  }
  calibFkInFlight = true;
  calibFkQueued = false;
  try {
    const data = await API.post("/api/robot/fk_preview", {
      robot: state.robot.name,
      joint_q: state.calibQ,
    });
    robot.applyCalibPose(data.link_transforms, data.ground_offset_z);
    if (calibManip.active) {
      calibManip.updateJointWorld(data.joint_world);
    }
    if (state.calibrationMode && state.calibNeedsCameraFocus) {
      state.calibNeedsCameraFocus = false;
      applyCalibOrbitLimits({ snapCamera: true });
      focusRobotView({ resetOffset: true });
    }
  } catch (e) {
    console.warn("calib FK preview", e.message);
  } finally {
    calibFkInFlight = false;
    if (calibFkQueued) previewCalibPose();
  }
}

async function refreshRetargetPanel() {
  document.getElementById("rt-motion").textContent = state.motion ? state.motion.name : "未加载";
  document.getElementById("rt-robot").textContent = state.robot ? state.robot.display_name : "未加载";
  syncRefSelect();
  if (state.calibrationMode) return;
  const calCard = document.getElementById("calib-card");
  const btn = document.getElementById("retarget-btn");
  const recal = document.getElementById("recalib-btn");
  recal.disabled = !(state.robot && state.reference);
  if (!state.robot || !state.reference) {
    setCalChip("—", "");
    calCard.style.display = "none";
    btn.disabled = true;
    return;
  }
  try {
    const st = await API.get(
      `/api/calibration/status?robot=${encodeURIComponent(state.robot.name)}&reference=${encodeURIComponent(state.reference)}`
    );
    state.calibration = st.calibrated;
    if (st.calibrated) {
      setCalChip(st.bundled && !st.path ? "内置缩放参数" : "已标定", "ok");
      calCard.style.display = "none";
      btn.disabled = !state.motion;
      if (state.motion) await refreshScaledPreview();
    } else {
      setCalChip("未标定 — 请先标定", "warn");
      btn.disabled = true;
      if (state.motion) {
        await enterCalibrationMode(st.joint_q || null);
      } else {
        calCard.style.display = "none";
      }
    }
  } catch (e) {
    setCalChip("未标定", "warn");
    btn.disabled = true;
    if (state.motion) {
      await enterCalibrationMode(null);
    } else {
      calCard.style.display = "none";
    }
  }
}

document.getElementById("rt-ref-select")?.addEventListener("change", (ev) => {
  const val = ev.target.value;
  if (!val) return;
  onReferenceChange(val);
});

document.getElementById("recalib-btn").onclick = async () => {
  if (!state.robot || !state.reference) return;
  let jq = null;
  try {
    const st = await API.get(
      `/api/calibration/status?robot=${encodeURIComponent(state.robot.name)}&reference=${encodeURIComponent(state.reference)}`
    );
    jq = st.joint_q || null;
  } catch { /* session seeds from yaml */ }
  await enterCalibrationMode(jq);
};

document.getElementById("calib-zero").onclick = async () => {
  const zeros = {};
  for (const j of Object.keys(state.calibQ)) zeros[j] = 0;
  await buildCalibSliders(zeros, state.calibLimits);
  toast("已归零（URDF 零位）");
};

document.getElementById("calib-restore").onclick = async () => {
  if (!state.calibHasSaved || !state.calibBaselineQ) {
    toast("尚无已保存标定可恢复", true);
    return;
  }
  await buildCalibSliders({ ...state.calibBaselineQ }, state.calibLimits);
  toast("已恢复到上次保存的标定");
};

document.getElementById("calib-cancel").onclick = async () => {
  await exitCalibrationMode();
  document.getElementById("calib-card").style.display = "none";
  toast("已取消标定");
  refreshRetargetPanel();
};

document.getElementById("calib-save").onclick = async () => {
  if (!state.robot) return;
  try {
    await API.post("/api/calibration/save", {
      robot: state.robot.name,
      reference: state.reference,
      joint_q: state.calibQ,
      motion_token: state.motion?.token || null,
    });
    state.calibBaselineQ = { ...state.calibQ };
    state.calibHasSaved = true;
    toast("标定已保存");
    await exitCalibrationMode();
    document.getElementById("calib-card").style.display = "none";
    state.calibration = true;
    // Robot still holds the last calibration FK pose until retarget supplies a
    // trajectory; do not resume motion playback with the yellow overlay yet.
    player.setPlaying(false);
    robot.applyStatic();
    setViewVisible(scaledSkel, "tg-scaled", false);
    setViewVisible(scaledEnv, "tg-scaled-env", false);
    refreshRetargetPanel();
    toast("标定已保存 — 请点击 Retarget 后再播放预览");
  } catch (e) { toast(e.message, true); }
};

async function pollJob(jobId, onProgress) {
  while (true) {
    const j = await API.get(`/api/job/${jobId}`);
    if (onProgress) onProgress(j);
    if (j.status === "done") return j;
    if (j.status === "error") throw new Error(j.error || "job failed");
    await new Promise((r) => setTimeout(r, 700));
  }
}

function setRetargetProgress(prog, bar, jp) {
  const p = jp.progress || 0;
  const indet = jp.status === "running" && p < 0.1;
  prog.classList.toggle("indet", indet);
  if (!indet) {
    bar.style.width = `${Math.max(2, p * 100).toFixed(0)}%`;
  }
}

document.getElementById("retarget-btn").onclick = async () => {
  if (!state.motion || !state.robot) return;
  const retargetRobotName = state.robot.name;
  const prog = document.getElementById("rt-progress");
  const bar = prog.querySelector(".bar");
  const status = document.getElementById("rt-status");
  prog.style.display = "block";
  prog.classList.add("indet");
  bar.style.width = "0%";
  const firstHint = !state.robot.ik_prewarmed;
  status.innerHTML = firstHint
    ? `<span class="spin"></span> 正在 retarget…（新机器人首次较慢，进度条可能短暂不动）`
    : `<span class="spin"></span> 正在 retarget…`;
  document.getElementById("retarget-btn").disabled = true;
  setRobotPanelLocked(true);
  try {
    const retargetFps = parseOptionalFps(document.getElementById("rt-retarget-fps"));
    const body = {
      robot: retargetRobotName,
      motion_token: state.motion.token,
      reference: state.reference,
      backend: document.getElementById("rt-backend").value,
      foot_clamp_anti_penetration: footClampAntiPenetrationEnabled(),
    };
    if (retargetFps) body.retarget_fps = retargetFps;
    const { job_id } = await API.post("/api/retarget", body);
    const j = await pollJob(job_id, (jp) => {
      setRetargetProgress(prog, bar, jp);
      const msg = jp.message || (firstHint ? "新机器人首次 retarget 编译中，请耐心等待…" : "正在 retarget…");
      status.innerHTML = `<span class="spin"></span> ${msg}`;
    });
    if (state.robot?.name !== retargetRobotName) {
      prog.classList.remove("indet");
      status.textContent = "";
      toast("Retarget 已完成，但过程中机器人已变更，结果已丢弃。请重新执行 Retarget。", true);
      return;
    }
    prog.classList.remove("indet");
    bar.style.width = "100%";
    if (state.robot) state.robot.ik_prewarmed = true;
    const srcFps = j.result.motion_source_fps ?? state.motion?.framerate;
    const rtFps = j.result.retarget_fps ?? j.result.source_fps;
    status.textContent =
      `完成：${j.result.num_frames} 帧 @ ${(rtFps || 30).toFixed(1)} fps` +
      (srcFps && Math.abs(srcFps - rtFps) > 0.5 ? `（动作原始 ${srcFps.toFixed(1)} fps）` : "");
    state.robotTrajectory = j.result.trajectory;
    robot.setTrajectory(j.result.trajectory);
    // Always restart the shared timeline at t=0.  Previously we only called
    // ``ready`` when inactive, so an in-progress source scrub kept ``t`` near
    // the end — the first "play" of the retarget was already finishing, and
    // the first loop wrap looked like a mysterious global jump.
    player.ready(robot.clipDuration);
    player.refreshFrame();
    document.getElementById("tg-robot").disabled = false;
    if (j.result.scaled_preview) {
      scaledSkel.load(j.result.scaled_preview);
      document.getElementById("tg-scaled").disabled = false;
    } else {
      await refreshScaledPreview();
    }
    if (j.result.scaled_scene) {
      scaledEnv.load(j.result.scaled_scene, state.motion.token);
      document.getElementById("tg-scaled-env").disabled = false;
      setViewVisible(scaledEnv, "tg-scaled-env", true);
    }
    setViewVisible(skel, "tg-skeleton", true);
    setBodyVisible(true);
    setViewVisible(scaledSkel, "tg-scaled", true);
    setViewVisible(robot, "tg-robot", true);
    player.setPlaying(true);
    robot.group.getWorldPosition(_camFocus);
    orbit.target.copy(_camFocus);
    _orbitManualUntil = 0;
    state.exportToken = j.result.export_token;
    state.exportSrcFps = j.result.source_fps;
    state.exportHasScene = j.result.has_scene;
    document.getElementById("rt-export-card").style.display = "block";
    const fpsInput = document.getElementById("rt-export-fps");
    fpsInput.value = "";
    const eff = j.result.retarget_fps ?? j.result.source_fps ?? 30;
    fpsInput.placeholder = `留空 = ${eff.toFixed(0)} fps（Retarget 结果）`;
    const clipSrc = j.result.motion_source_fps ?? state.motion?.framerate;
    let exportHint =
      `当前缓存：<b>${eff.toFixed(1)} fps</b>（Retarget 求解帧率）`;
    if (clipSrc && Math.abs(clipSrc - eff) > 0.5) {
      exportHint += `；动作文件原始 <b>${clipSrc.toFixed(1)} fps</b>`;
    }
    exportHint += "。<b>导出 FPS</b> 仅插值机器人轨迹，不重新求解。";
    const bundleHint = document.getElementById("rt-export-bundle-hint");
    if (bundleHint) bundleHint.style.display = j.result.has_scene ? "block" : "none";
    if (j.result.has_scene) {
      exportHint += " 含地形/物体时将打包为 ZIP（数据文件 + OBJ）。";
    }
    document.getElementById("rt-export-srcfps").innerHTML = exportHint;
    toast("Retarget 完成，可导出");
  } catch (e) {
    status.textContent = "";
    prog.classList.remove("indet");
    toast(e.message, true);
  } finally {
    setRobotPanelLocked(false);
    document.getElementById("retarget-btn").disabled = false;
  }
};
function csvHeaderEnabled(elId) {
  const el = document.getElementById(elId);
  return el ? el.checked : true;
}

document.getElementById("rt-export-btn").onclick = async () => {
  if (!state.exportToken) return;
  const fps = parseFloat(document.getElementById("rt-export-fps").value);
  const fmt = document.getElementById("rt-export-format")?.value || "csv";
  let url = `/api/export/${state.exportToken}?fmt=${encodeURIComponent(fmt)}`;
  if (fps && fps > 0) url += `&fps=${fps}`;
  if (!csvHeaderEnabled("rt-csv-header")) url += "&csv_header=0";
  const name = state.exportHasScene || fmt === "pkl"
    ? `${state.motion?.name || "clip"}_export.zip`
    : `${state.motion?.name || "clip"}.csv`;
  try {
    await triggerBrowserDownload(url, name);
    toast("已开始下载（保存到浏览器默认下载目录）");
  } catch (e) { toast(e.message, true); }
};

// =================================================================  BATCH
let basket = [];
function basketEntryLabel(e) {
  const ds = datasetLabel(e.dataset);
  const ref = referenceLabel(entryReference(e, state.reference || "smpl"));
  const clip = e.origin === "upload"
    ? `${e.export_subdir ? `${e.export_subdir}/` : ""}${e.stem}`
    : `${e.folder_label}/${e.stem}`;
  return `输入 ${ds} → 标定 ${ref} · ${clip}`;
}

async function syncBatchRefHint() {
  const el = document.getElementById("batch-ref-hint");
  if (!el) return;
  if (!basket.length) {
    el.innerHTML = "";
    el.style.display = "none";
    return;
  }
  const groups = new Map();
  for (const e of basket) {
    const ref = entryReference(e, state.reference || "smpl");
    if (!groups.has(ref)) groups.set(ref, { count: 0, datasets: new Set() });
    const g = groups.get(ref);
    g.count += 1;
    g.datasets.add(e.dataset || "unknown");
  }

  const blocks = [];
  for (const [ref, g] of groups) {
    const help = REFERENCE_HELP[ref] || {
      input: `数据集 ${[...g.datasets].map(datasetLabel).join("、")}`,
      calib: `标定参考「${referenceLabel(ref)}」`,
      file: `retarget_calibration_${ref}.yaml`,
    };
    let status = "";
    if (state.robot?.name) {
      try {
        const st = await API.get(
          `/api/calibration/status?robot=${encodeURIComponent(state.robot.name)}`
          + `&reference=${encodeURIComponent(ref)}`,
        );
        status = st.calibrated
          ? '<span class="status-ok">✓ 当前机器人已标定</span>'
          : '<span class="status-warn">✗ 未标定 — 请去左侧「机器人→标定」保存</span>';
      } catch {
        status = "";
      }
    }
    blocks.push(
      `<div class="batch-ref-block">`
      + `<b>${referenceLabel(ref)}</b>（${g.count} 条） ${status}<br>`
      + `<span class="sub">① 输入格式：${help.input}</span><br>`
      + `<span class="sub">② 标定参考：${help.calib}</span><br>`
      + `<span class="sub">③ 标定文件：<code>${help.file}</code>（保存在机器人 URDF 同目录）</span>`
      + `</div>`,
    );
  }
  el.innerHTML = blocks.join("");
  el.style.display = "block";
}

function renderBasket() {
  const list = document.getElementById("basket-list");
  list.innerHTML = "";
  for (const e of basket) {
    const row = document.createElement("div");
    row.className = "basket-row";
    row.innerHTML = `<span>${basketEntryLabel(e)}</span><button class="rm">×</button>`;
    row.querySelector(".rm").onclick = () => { basket = basket.filter((x) => x !== e); syncBasket(); };
    list.appendChild(row);
  }
  document.getElementById("basket-count").textContent = basket.length;
  const badge = document.getElementById("basket-badge");
  badge.textContent = basket.length;
  badge.style.display = basket.length ? "inline-block" : "none";
  document.getElementById("batch-run").disabled = !(basket.length && state.robot);
  void syncBatchRefHint();
}
async function syncBasket() {
  renderBasket();
}
function addToBasket(entries, { silent = false } = {}) {
  for (const e of entries) {
    if (!basket.find((x) => x.source_path === e.source_path)) basket.push(e);
  }
  renderBasket();
  if (!silent) toast(`已加入篮子（${basket.length}）`);
}
document.getElementById("basket-clear").onclick = () => { basket = []; renderBasket(); };

async function ingestBasketFiles(files, profile = "auto") {
  if (!files || !files.length) return;
  showLoading(`上传到会话缓存… (${files.length} 个文件)`);
  try {
    const { job_id } = await uploadFilesXHR(
      "/api/basket/upload",
      files,
      { profile },
      (frac, recv, total) => {
        setLoadingProgress(frac * 0.35, `上传 ${fmtBytes(recv)} / ${fmtBytes(total)}`);
      },
    );
    const payload = await waitMotionJob(job_id, (frac, sub) => {
      setLoadingProgress(0.35 + frac * 0.65, sub);
    }, { uploadFrac: 0.35 });
    const entries = payload.entries || [];
    if (!entries.length) {
      toast("未识别到可 retarget 的 clip", true);
      return;
    }
    addToBasket(entries, { silent: true });
    toast(`已缓存 ${entries.length} 个 clip（关闭 Web 后自动清除）`);
  } catch (e) {
    toast(e.message, true);
  } finally {
    hideLoading();
  }
}

setupDropzone(document.getElementById("basket-drop"), (files) => ingestBasketFiles(files, "auto"));

const BATCH_STAGE_LABELS = { load: "加载", retarget: "重定向", export: "导出" };

function renderBatchFailures(result) {
  const box = document.getElementById("batch-failures");
  if (!box) return;
  const failures = result?.failures || [];
  if (!failures.length) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  box.classList.remove("hidden");
  let html = `<h4>失败明细（${failures.length}）</h4><ul class="batch-fail-list">`;
  for (const f of failures) {
    const stage = BATCH_STAGE_LABELS[f.stage] || f.stage;
    const logLine = f.log_rel
      ? `<div class="sub">已复制 → <code>${escapeHtml(f.log_rel)}</code></div>`
      : (f.stash_error
        ? `<div class="sub warn">未能复制源文件：${escapeHtml(f.stash_error)}</div>`
        : "");
    html += `<li><b>${escapeHtml(f.stem)}</b> <span class="tag">${escapeHtml(stage)}</span>`
      + `<div class="reason">${escapeHtml(f.reason)}</div>${logLine}</li>`;
  }
  html += "</ul>";
  if (result.failure_log) {
    html += `<p class="hint">失败数据目录：<code>${escapeHtml(result.failure_log)}</code><br>`
      + "修复后可将该文件夹（或其中子目录）拖入上方篮子重试；也可打开 "
      + "<code>失败说明.txt</code> / <code>failures.json</code> 查看详情。</p>";
  }
  box.innerHTML = html;
}

function setBatchProgress(jp) {
  const totalProg = document.getElementById("batch-progress-total");
  const clipProg = document.getElementById("batch-progress-clip");
  if (!totalProg || !clipProg) return;
  const totalBar = totalProg.querySelector(".bar");
  const clipBar = clipProg.querySelector(".bar");
  const totalP = jp.progress || 0;
  const clipP = jp.clip_progress ?? 0;
  const totalIndet = jp.status === "running" && totalP < 0.01;
  const clipIndet = jp.status === "running" && clipP < 0.02 && totalP < 0.99;
  totalProg.classList.toggle("indet", totalIndet);
  clipProg.classList.toggle("indet", clipIndet);
  if (!totalIndet) {
    totalBar.style.width = `${Math.max(0, totalP * 100).toFixed(0)}%`;
  } else {
    totalBar.style.width = "0%";
  }
  if (!clipIndet) {
    clipBar.style.width = `${Math.max(0, clipP * 100).toFixed(0)}%`;
  } else {
    clipBar.style.width = "0%";
  }
}

document.getElementById("batch-run").onclick = async () => {
  if (!basket.length || !state.robot) return;
  const batchRobotName = state.robot.name;
  const progStack = document.getElementById("batch-progress-stack");
  const status = document.getElementById("batch-status");
  const failBox = document.getElementById("batch-failures");
  if (failBox) {
    failBox.classList.add("hidden");
    failBox.innerHTML = "";
  }
  progStack?.classList.remove("hidden");
  setBatchProgress({ status: "running", progress: 0, clip_progress: 0 });
  status.innerHTML = `<span class="spin"></span> 批量处理中…`;
  setRobotPanelLocked(true);
  try {
    const batchBody = {
      robot: batchRobotName,
      reference: state.reference || "smpl",
      backend: document.getElementById("batch-backend").value,
      out_dir: document.getElementById("batch-out").value || "batch_export",
      format: document.getElementById("batch-format").value,
      csv_header: csvHeaderEnabled("batch-csv-header"),
      entries: basket,
      foot_clamp_anti_penetration: footClampAntiPenetrationEnabled(),
    };
    const batchSizeRaw = parseInt(document.getElementById("batch-size")?.value, 10);
    if (Number.isFinite(batchSizeRaw) && batchSizeRaw >= 1) {
      batchBody.batch_size = Math.min(256, batchSizeRaw);
    }
    const rtFps = parseOptionalFps(document.getElementById("batch-retarget-fps"));
    const exFps = parseOptionalFps(document.getElementById("batch-export-fps"));
    if (rtFps) batchBody.retarget_fps = rtFps;
    if (exFps) batchBody.export_fps = exFps;
    const { job_id } = await API.post("/api/batch/retarget", batchBody);
    const j = await pollJob(job_id, (jp) => {
      setBatchProgress(jp);
      status.textContent = jp.message || "";
    });
    setBatchProgress({ status: "done", progress: 1, clip_progress: 1 });
    const r = j.result;
    const modeNote = r.solver_mode ? ` · ${r.solver_mode}` : "";
    const partialNote = (r.failures?.length && r.written?.length)
      ? "（ZIP 仅含成功项，失败见下方）" : "";
    status.innerHTML = `完成：${r.written?.length ?? 0} 个 clip` +
      (r.failures?.length ? `，${r.failures.length} 个失败` : "") +
      partialNote +
      modeNote +
      (r.download_name ? ` — 正在下载 <b>${r.download_name}</b>` : "");
    renderBatchFailures(r);
    if (r.download_name) {
      try {
        await triggerBrowserDownload(`/api/job/${job_id}/download`, r.download_name);
      } catch (e) { toast(e.message, true); }
    }
    toast(
      `批量完成：${r.written?.length ?? 0} 个`
      + (r.failures?.length ? `，${r.failures.length} 失败（见下方明细）` : ""),
      !!r.failures?.length,
    );
  } catch (e) {
    status.textContent = "";
    renderBatchFailures(null);
    toast(e.message, true);
  } finally {
    setRobotPanelLocked(false);
  }
};

// =================================================================  PANEL LAYOUT (resize / hide)
function initPanelLayout() {
  const app = document.getElementById("app");
  const key = "hhtools-panel-layout-v1";
  const defaults = { sidebarW: 248, inspectorW: 360, sidebarHidden: false, inspectorHidden: false };
  let layout = { ...defaults };
  try {
    Object.assign(layout, JSON.parse(localStorage.getItem(key) || "{}"));
  } catch { /* ignore */ }
  layout.sidebarW = Math.min(520, Math.max(160, layout.sidebarW || defaults.sidebarW));
  layout.inspectorW = Math.min(640, Math.max(240, layout.inspectorW || defaults.inspectorW));

  const showSb = document.getElementById("show-sidebar");
  const showInsp = document.getElementById("show-inspector");

  const apply = () => {
    app.style.setProperty("--sidebar-w", layout.sidebarHidden ? "0px" : `${layout.sidebarW}px`);
    app.style.setProperty("--inspector-w", layout.inspectorHidden ? "0px" : `${layout.inspectorW}px`);
    app.classList.toggle("sidebar-hidden", layout.sidebarHidden);
    app.classList.toggle("inspector-hidden", layout.inspectorHidden);
    if (showSb) showSb.hidden = !layout.sidebarHidden;
    if (showInsp) showInsp.hidden = !layout.inspectorHidden;
    resize();
  };
  const save = () => localStorage.setItem(key, JSON.stringify(layout));

  const dragCol = (handle, side) => {
    handle.addEventListener("mousedown", (e) => {
      e.preventDefault();
      handle.classList.add("dragging");
      const x0 = e.clientX;
      const w0 = side === "sidebar" ? layout.sidebarW : layout.inspectorW;
      const move = (ev) => {
        const dx = ev.clientX - x0;
        if (side === "sidebar") layout.sidebarW = Math.min(520, Math.max(160, w0 + dx));
        else layout.inspectorW = Math.min(640, Math.max(240, w0 - dx));
        layout.sidebarHidden = false;
        layout.inspectorHidden = false;
        apply();
      };
      const up = () => {
        handle.classList.remove("dragging");
        document.removeEventListener("mousemove", move);
        document.removeEventListener("mouseup", up);
        save();
      };
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
    });
  };

  dragCol(document.getElementById("resize-sidebar"), "sidebar");
  dragCol(document.getElementById("resize-inspector"), "inspector");
  document.getElementById("hide-sidebar")?.addEventListener("click", () => {
    layout.sidebarHidden = true; apply(); save();
  });
  document.getElementById("hide-inspector")?.addEventListener("click", () => {
    layout.inspectorHidden = true; apply(); save();
  });
  showSb?.addEventListener("click", () => { layout.sidebarHidden = false; apply(); save(); });
  showInsp?.addEventListener("click", () => { layout.inspectorHidden = false; apply(); save(); });
  apply();
}

// Wrap every <select> so a CSS chevron can sit outside the native control.
function wrapSelectDropdowns() {
  for (const sel of document.querySelectorAll("select.search")) {
    if (sel.parentElement?.classList.contains("select-wrap")) continue;
    const wrap = document.createElement("div");
    wrap.className = "select-wrap";
    for (const prop of ["flex", "flexGrow", "flexShrink", "flexBasis", "width"]) {
      if (sel.style[prop]) {
        wrap.style[prop] = sel.style[prop];
        sel.style[prop] = "";
      }
    }
    sel.parentNode.insertBefore(wrap, sel);
    wrap.appendChild(sel);
  }
}

// =================================================================  ROBOT-TO-ROBOT (R2R)
// A self-contained module: its own two RobotView instances + state, so it never
// touches the human→robot workflow's `state.robot` / `robot` view. The stage is
// snapshotted on enter and restored on leave so switching panels is lossless.
const r2rSrc = new RobotView();
const r2rTgt = new RobotView();
const r2rSrcSkel = new ScaledSkeletonView();
r2rSrcSkel.color = 0x60a5fa;
const r2rTgtSkel = new ScaledSkeletonView();
r2rTgtSkel.color = 0xffb020;
const r2rSrcEnvGroup = new THREE.Group();
world.add(r2rSrcEnvGroup);
const r2rTgtEnvGroup = new THREE.Group();
world.add(r2rTgtEnvGroup);
const r2rSrcEnv = new ScaledEnvView(r2rSrcEnvGroup);
const r2rTgtEnv = new ScaledEnvView(r2rTgtEnvGroup);
ALL_VIEWS.push(r2rSrc, r2rTgt, r2rSrcSkel, r2rTgtSkel, r2rSrcEnv, r2rTgtEnv);

const r2r = {
  active: false,
  sourceName: null,
  sourcePayload: null,
  targetName: null,
  targetPayload: null,
  sourceToken: null,
  sourceStem: null,
  resultStem: null,
  exportToken: null,
  calibrating: false,
  calibrated: false,
  calibQ: {},
  calibLimits: [],
  calibRows: {},
  calibNeedsCameraFocus: false,
  calibOrbitSaved: null,
  hasScene: false,
  basket: [],
  scaledScene: null,
  tgtScaledScene: null,
};
const r2rVis = {
  srcRobot: true,
  srcSkel: false,
  srcEnv: false,
  tgtRobot: false,
  tgtSkel: false,
  tgtEnv: false,
};

let _r2rMainSnap = null;
const _r2rVec = new THREE.Vector3();

function r2rFocus(view) {
  try {
    view.group.getWorldPosition(_r2rVec);
    orbit.target.copy(_r2rVec);
  } catch { /* ignore */ }
}

function r2rSetToggle(btnId, on) {
  const btn = document.getElementById(btnId);
  if (btn) btn.classList.toggle("on", !!on);
}

function r2rSyncPlayerDuration() {
  let dur = 0.1;
  for (const v of [r2rSrc, r2rTgt, r2rSrcSkel, r2rTgtSkel, r2rSrcEnv, r2rTgtEnv]) {
    if (v.trajectory || v.numFrames > 0) {
      dur = Math.max(dur, v.clipDuration || (v.numFrames / 30));
    }
  }
  player.duration = dur;
  if (player.active) player.refreshFrame();
}

function r2rSceneGlbUrl(token, o) {
  const mesh = o.mesh_file || "";
  if (!token || !mesh) return null;
  let url =
    `/api/r2r/scene_glb?token=${encodeURIComponent(token)}&mesh=${encodeURIComponent(mesh)}`;
  if (o.scale != null && Number.isFinite(o.scale)) {
    url += `&scale=${encodeURIComponent(o.scale)}`;
  }
  return url;
}

function r2rLoadSrcScene(scene, token, duration) {
  if (!scene) {
    r2rSrcEnv.clear();
    r2r.scaledScene = null;
    document.getElementById("r2r-tg-src-env")?.setAttribute("disabled", "");
    return;
  }
  r2r.scaledScene = scene;
  r2rSrcEnv.load(scene, token, {
    duration,
    objectGlbUrl: (o) => r2rSceneGlbUrl(token, o),
  });
  const envBtn = document.getElementById("r2r-tg-src-env");
  if (envBtn) envBtn.disabled = false;
}

function r2rLoadTgtScene(scene, token, duration) {
  if (!scene) {
    r2rTgtEnv.clear();
    r2r.tgtScaledScene = null;
    document.getElementById("r2r-tg-tgt-env")?.setAttribute("disabled", "");
    return;
  }
  r2r.tgtScaledScene = scene;
  r2rTgtEnv.load(scene, token, {
    duration,
    objectGlbUrl: (o) => r2rSceneGlbUrl(token, o),
  });
  const envBtn = document.getElementById("r2r-tg-tgt-env");
  if (envBtn) envBtn.disabled = false;
}

function r2rApplyStage() {
  if (!r2r.active) {
    r2rSrc.group.visible = false;
    r2rTgt.group.visible = false;
    r2rSrcSkel.group.visible = false;
    r2rTgtSkel.group.visible = false;
    r2rSrcEnv.group.visible = false;
    r2rTgtEnv.group.visible = false;
    return;
  }
  for (const v of ALL_VIEWS) {
    if (![r2rSrc, r2rTgt, r2rSrcSkel, r2rTgtSkel, r2rSrcEnv, r2rTgtEnv].includes(v)) {
      v.group.visible = false;
    }
  }
  if (r2r.calibrating) {
    r2rSrc.group.visible = false;
    r2rSrcSkel.group.visible = false;
    r2rSrcEnv.group.visible = false;
    r2rTgtSkel.group.visible = false;
    r2rTgtEnv.group.visible = false;
    r2rTgt.group.visible = (r2rTgt.links?.length || 0) > 0;
    refSkel.group.visible = true;
    revealStage();
    _setPlaybarVisible(false);
    player.setPlaying(false);
    return;
  }
  refSkel.group.visible = false;
  const hasSrc = !!(r2rSrc.trajectory || r2rSrc.links?.length);
  const hasTgt = !!(r2rTgt.trajectory || r2rTgt.links?.length);
  const hasSrcSk = r2rSrcSkel.numFrames > 0;
  const hasTgtSk = r2rTgtSkel.numFrames > 0;
  const hasSrcEnv = r2rSrcEnv.numFrames > 0 || !!r2r.scaledScene?.terrain;
  const hasTgtEnv = r2rTgtEnv.numFrames > 0 || !!r2r.tgtScaledScene?.terrain;
  r2rSrc.group.visible = r2rVis.srcRobot && hasSrc;
  r2rSrcSkel.group.visible = r2rVis.srcSkel && hasSrcSk;
  r2rSrcEnv.group.visible = r2rVis.srcEnv && hasSrcEnv;
  r2rTgt.group.visible = r2rVis.tgtRobot && hasTgt;
  r2rTgtSkel.group.visible = r2rVis.tgtSkel && hasTgtSk;
  r2rTgtEnv.group.visible = r2rVis.tgtEnv && hasTgtEnv;
  r2rSetToggle("r2r-tg-src-robot", r2rSrc.group.visible);
  r2rSetToggle("r2r-tg-src-skel", r2rSrcSkel.group.visible);
  r2rSetToggle("r2r-tg-src-env", r2rSrcEnv.group.visible);
  r2rSetToggle("r2r-tg-tgt-robot", r2rTgt.group.visible);
  r2rSetToggle("r2r-tg-tgt-skel", r2rTgtSkel.group.visible);
  r2rSetToggle("r2r-tg-tgt-env", r2rTgtEnv.group.visible);
  if (hasSrc || hasTgt || hasSrcSk || hasTgtSk || hasSrcEnv || hasTgtEnv) {
    player.active = true;
    revealStage();
    _setPlaybarVisible(true);
    r2rSyncPlayerDuration();
    player.refreshFrame();
  }
}

function r2rEnterPanel() {
  if (r2r.active) { r2rApplyStage(); return; }
  r2r.active = true;
  _r2rMainSnap = {
    vis: ALL_VIEWS.map((v) => v.group.visible),
    refSkel: refSkel.group.visible,
    player: { t: player.t, duration: player.duration, active: player.active },
  };
  for (const v of ALL_VIEWS) {
    if (v !== r2rSrc && v !== r2rTgt) v.group.visible = false;
  }
  player.setPlaying(false);
  document.getElementById("view-hud")?.classList.add("hidden");
  document.getElementById("view-hud-r2r")?.classList.remove("hidden");
  r2rApplyStage();
  void r2rUpdateRetargetBtn();
}

function r2rLeavePanel() {
  if (!r2r.active) return;
  r2r.active = false;
  if (r2r.calibrating) r2rExitCalib();
  r2rSrc.group.visible = false;
  r2rTgt.group.visible = false;
  const s = _r2rMainSnap;
  _r2rMainSnap = null;
  if (s) {
    ALL_VIEWS.forEach((v, i) => {
      if (v !== r2rSrc && v !== r2rTgt) v.group.visible = !!s.vis[i];
    });
    refSkel.group.visible = s.refSkel;
    player.t = s.player.t;
    player.duration = s.player.duration;
    player.active = s.player.active;
    player.setPlaying(false);
    if (player.active) player.refreshFrame();
  } else {
    refSkel.group.visible = false;
  }
  document.getElementById("view-hud-r2r")?.classList.add("hidden");
  document.getElementById("view-hud")?.classList.remove("hidden");
  _restoreViewToggleButtons();
}

// Hook panel switching so the R2R stage is shown/hidden with its tab.
const _r2rOrigSwitch = switchInspectorPanel;
switchInspectorPanel = function r2rSwitch(panelId) {
  const leaving = r2r.active && panelId !== "r2r";
  _r2rOrigSwitch(panelId);
  if (panelId === "r2r") r2rEnterPanel();
  else if (leaving) r2rLeavePanel();
};

function r2rSetCalChip(text, cls) {
  const el = document.getElementById("r2r-cal");
  if (!el) return;
  el.innerHTML = `<span class="status-chip ${cls || ""}"><span class="dot"></span>${text}</span>`;
}

async function r2rUpdateRetargetBtn() {
  const calBtn = document.getElementById("r2r-calib-btn");
  const rtBtn = document.getElementById("r2r-retarget-btn");
  if (calBtn) calBtn.disabled = !(r2r.targetName && r2r.sourceName);
  let calibrated = false;
  if (r2r.targetName && r2r.sourceName) {
    try {
      const st = await API.get(
        `/api/r2r/calibration/status?target=${encodeURIComponent(r2r.targetName)}&source=${encodeURIComponent(r2r.sourceName)}`
      );
      calibrated = !!st.calibrated;
    } catch { /* treat as uncalibrated */ }
  }
  r2r.calibrated = calibrated;
  if (!r2r.targetName || !r2r.sourceName) r2rSetCalChip("—", "");
  else r2rSetCalChip(calibrated ? "已标定" : "未标定 — 请先标定", calibrated ? "ok" : "warn");
  if (rtBtn) rtBtn.disabled = !(r2r.sourceToken && r2r.targetName && calibrated);
}

// --------------------------------------------------------------- robot pickers
async function r2rPopulateSelects() {
  let data;
  try { data = await API.get("/api/robots"); }
  catch { return; }
  const fill = (sel, preferG1) => {
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = "";
    let g1 = null;
    for (const r of data.robots) {
      const opt = document.createElement("option");
      opt.value = r.name;
      opt.textContent = `${r.display_name} (${r.num_dof} DOF)${r.has_urdf ? "" : " — 无URDF"}`;
      opt.disabled = !r.has_urdf;
      sel.appendChild(opt);
      if (preferG1 && !g1 && r.has_urdf && /g1/i.test(r.name + r.display_name)) g1 = r.name;
    }
    if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
    else if (g1) sel.value = g1;
  };
  fill(document.getElementById("r2r-source-select"), true);
  fill(document.getElementById("r2r-target-select"), false);
}

// --------------------------------------------------------------- calibration
let _r2rFkRaf = 0;
let _r2rFkInFlight = false;
let _r2rFkQueued = false;

function r2rCalibCtx() {
  return {
    robotView: r2rTgt,
    getQ: () => r2r.calibQ,
    getSliderRows: () => r2r.calibRows,
    jointChange: (name, val, opts) => r2rSetCalibJointValue(name, val, opts),
    previewFk: (opts) => r2rPreviewCalibPose(opts),
  };
}

function r2rPreviewCalibPose({ live = false, flush = false } = {}) {
  if (!r2r.calibrating || !r2r.targetName) return;
  if (flush) {
    if (_r2rFkRaf) cancelAnimationFrame(_r2rFkRaf);
    _r2rFkRaf = 0;
    void _r2rRunFk();
    return;
  }
  if (_r2rFkRaf) return;
  _r2rFkRaf = requestAnimationFrame(() => {
    _r2rFkRaf = 0;
    void _r2rRunFk();
  });
}

async function _r2rRunFk() {
  if (_r2rFkInFlight) { _r2rFkQueued = true; return; }
  _r2rFkInFlight = true;
  _r2rFkQueued = false;
  try {
    const data = await API.post("/api/robot/fk_preview", {
      robot: r2r.targetName,
      joint_q: r2r.calibQ,
    });
    r2rTgt.applyCalibPose(data.link_transforms, data.ground_offset_z);
    if (calibManip.active) calibManip.updateJointWorld(data.joint_world);
    if (r2r.calibrating && r2r.calibNeedsCameraFocus) {
      r2r.calibNeedsCameraFocus = false;
      applyCalibOrbitLimits({ snapCamera: true });
      focusRobotView({ resetOffset: true });
    }
  } catch (e) {
    console.warn("r2r fk preview", e.message);
  } finally {
    _r2rFkInFlight = false;
    if (_r2rFkQueued) r2rPreviewCalibPose();
  }
}

function r2rSetCalibJointValue(jointName, value, { from, live = false } = {}) {
  const limByName = {};
  for (const L of r2r.calibLimits || []) limByName[L.name] = L;
  const lim = limByName[jointName] || {};
  let lo = lim.lower != null ? lim.lower : -Math.PI;
  let hi = lim.upper != null ? lim.upper : Math.PI;
  if (hi <= lo) { lo = -Math.PI; hi = Math.PI; }
  let x = parseFloat(value);
  if (!Number.isFinite(x)) return;
  x = Math.min(hi, Math.max(lo, x));
  r2r.calibQ[jointName] = x;

  const row = r2r.calibRows[jointName];
  const prec = live ? 4 : 3;
  if (row) {
    if (from === "slider") {
      row.range.value = String(x);
      row.num.value = x.toFixed(prec);
    } else if (from === "number") {
      row.range.value = String(x);
      if (!live) row.num.value = x.toFixed(prec);
    } else if (from !== "hud-input") {
      row.range.value = String(x);
      row.num.value = x.toFixed(prec);
    }
  }
  if (from === "hud-input") {
    calibManip.updateHudValue(jointName, x, { live, syncInput: false });
  } else {
    calibManip.updateHudValue(jointName, x, { live });
  }
  if (from === "slider" || from === "number") calibManip.setSelected(jointName);
  r2rPreviewCalibPose({ live });
}

function r2rBuildSliders(initialQ, limits) {
  const box = document.getElementById("r2r-calib-sliders");
  if (!box) return;
  box.innerHTML = "";
  r2r.calibQ = {};
  r2r.calibRows = {};
  const limByName = {};
  for (const L of limits || []) limByName[L.name] = L;
  const joints = (limits || []).map((L) => L.name).filter(Boolean);
  for (const j of r2r.targetPayload?.actuated_joints || []) {
    if (!limByName[j]) joints.push(j);
  }
  const seen = new Set();
  for (const j of joints) {
    if (seen.has(j)) continue;
    seen.add(j);
    const lim = limByName[j] || {};
    let lo = lim.lower != null ? lim.lower : -Math.PI;
    let hi = lim.upper != null ? lim.upper : Math.PI;
    if (hi <= lo) { lo = -Math.PI; hi = Math.PI; }
    let v = initialQ[j] != null ? parseFloat(initialQ[j]) : 0;
    v = Math.min(hi, Math.max(lo, v));
    r2r.calibQ[j] = v;
    const rowEl = document.createElement("div");
    rowEl.className = "slider-row";
    rowEl.innerHTML = `<label title="${j}">${j}</label>
      <input type="range" min="${lo}" max="${hi}" step="0.001" value="${v}" />
      <input type="number" class="calib-num" min="${lo}" max="${hi}" step="0.001" value="${v.toFixed(3)}" />`;
    const range = rowEl.querySelector('input[type="range"]');
    const num = rowEl.querySelector(".calib-num");
    r2r.calibRows[j] = { row: rowEl, range, num, lo, hi };
    calibManip.updateHudValue(j, v);
    range.oninput = () => r2rSetCalibJointValue(j, range.value, { from: "slider", live: true });
    num.oninput = () => r2rSetCalibJointValue(j, num.value, { from: "number", live: true });
    num.onchange = () => r2rSetCalibJointValue(j, num.value, { from: "number" });
    num.onkeydown = (ev) => {
      if (ev.key === "Enter") { r2rSetCalibJointValue(j, num.value, { from: "number" }); num.blur(); }
    };
    rowEl.onclick = () => {
      calibManip._pickScreen = null;
      calibManip._pickAnchor = null;
      calibManip._hudPinned = null;
      calibManip.setSelected(j, { scrollPanel: true });
    };
    box.appendChild(rowEl);
  }
  r2rPreviewCalibPose();
}

async function r2rStartCalib({ auto = false } = {}) {
  if (!r2r.targetName || !r2r.sourceName) {
    toast("请先加载源机器人与目标机器人", true);
    return;
  }
  if (!auto) toast("准备标定…");
  let session;
  try {
    session = await API.post("/api/r2r/calibration/session", {
      target: r2r.targetName,
      source: r2r.sourceName,
    });
  } catch (e) { toast(e.message, true); return; }
  if (!r2r.targetPayload) {
    try { r2r.targetPayload = await API.post("/api/robot/select", { name: r2r.targetName }); }
    catch (e) { toast(e.message, true); return; }
  }
  switchInspectorPanel("r2r");
  if (!r2r.active) r2rEnterPanel();
  r2r.calibrating = true;
  r2r.calibNeedsCameraFocus = true;
  r2r.calibOrbitSaved = {
    minDistance: orbit.minDistance,
    maxDistance: orbit.maxDistance,
    zoomSpeed: orbit.zoomSpeed,
  };
  orbit.zoomSpeed = 0.022;
  applyCalibOrbitLimits();
  updateR2rCalibBanner();
  document.getElementById("calib-banner")?.classList.remove("hidden");
  r2rSetCalChip("标定中…", "warn");
  document.getElementById("r2r-retarget-btn").disabled = true;

  r2r.calibLimits = session.joint_limits || [];
  await r2rTgt.load(r2r.targetPayload);
  r2rTgt.groundOffset = session.ground_offset_z ?? r2rTgt.groundOffset;
  refSkel.load(session.reference);
  document.getElementById("r2r-calib-edit").style.display = "block";
  r2rApplyStage();
  calibManip.start(r2r.calibLimits, r2rCalibCtx());
  r2rBuildSliders(session.joint_q || {}, r2r.calibLimits);
  document.getElementById("r2r-calib-edit")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  r2rFocus(r2rTgt);
  toast(auto
    ? "目标机器人尚未标定：已自动进入标定模式（点击关节拖动或右侧滑块）"
    : "已进入标定：把目标机器人对齐到蓝色源参考姿态");
}

function r2rExitCalib() {
  r2r.calibrating = false;
  r2r.calibNeedsCameraFocus = false;
  if (r2r.calibOrbitSaved) {
    orbit.minDistance = r2r.calibOrbitSaved.minDistance;
    orbit.maxDistance = r2r.calibOrbitSaved.maxDistance;
    orbit.zoomSpeed = r2r.calibOrbitSaved.zoomSpeed ?? orbit.zoomSpeed;
    r2r.calibOrbitSaved = null;
  }
  calibManip.stop();
  document.getElementById("r2r-calib-edit").style.display = "none";
  document.getElementById("calib-banner")?.classList.add("hidden");
  refSkel.clear();
  refSkel.group.visible = false;
  r2rApplyStage();
}

async function r2rMaybeAutoCalib() {
  if (!r2r.targetName || !r2r.sourceName || r2r.calibrating) return;
  await r2rUpdateRetargetBtn();
  if (!r2r.calibrated) await r2rStartCalib({ auto: true });
}

async function r2rSaveCalib() {
  try {
    await API.post("/api/r2r/calibration/save", {
      target: r2r.targetName,
      source: r2r.sourceName,
      joint_q: r2r.calibQ,
    });
    toast("R2R 标定已保存");
    r2rExitCalib();
    await r2rUpdateRetargetBtn();
  } catch (e) { toast(e.message, true); }
}

// --------------------------------------------------------------- trajectory IO
async function r2rEnsureSourceLoaded() {
  if (r2r.sourceName && r2r.sourcePayload) return true;
  const name = document.getElementById("r2r-source-select")?.value;
  if (!name) {
    toast("请先在「1 · 源机器人」选择并加载 G1（或其它源机器人）", true);
    return false;
  }
  toast("自动加载源机器人…");
  try {
    r2r.sourcePayload = await API.post("/api/robot/select", { name });
    r2r.sourceName = name;
    await r2rSrc.load(r2r.sourcePayload);
    document.getElementById("r2r-source-status").textContent =
      `源机器人：${r2r.sourcePayload.display_name}`;
    return true;
  } catch (e) {
    toast(e.message, true);
    return false;
  }
}

async function r2rUploadTraj(files, profile = "auto") {
  if (!files?.length) return;
  if (!(await r2rEnsureSourceLoaded())) return;
  const st = document.getElementById("r2r-traj-status");
  const prog = document.getElementById("r2r-traj-progress");
  const bar = prog?.querySelector(".bar");
  if (prog) {
    prog.style.display = "block";
    prog.classList.remove("indet");
    if (bar) bar.style.width = "0%";
  }
  st.textContent = "上传中…";
  toast("上传源轨迹…");
  try {
    switchInspectorPanel("r2r");
    if (!r2r.active) r2rEnterPanel();
    const qs = [
      `source_robot=${encodeURIComponent(r2r.sourceName)}`,
      `profile=${encodeURIComponent(profile)}`,
    ].join("&");
    const { job_id } = await uploadFilesXHR(
      `/api/r2r/source/upload?${qs}`,
      files,
      {},
      (frac) => {
        if (bar) bar.style.width = `${Math.max(2, frac * 18).toFixed(0)}%`;
        st.textContent = `上传 ${Math.round(frac * 100)}%…`;
      },
    );
    const data = await waitMotionJob(job_id, (frac, sub) => {
      if (bar) bar.style.width = `${Math.max(2, 18 + frac * 82).toFixed(0)}%`;
      st.textContent = sub;
    }, { uploadFrac: 0.18 });
    r2r.sourceToken = data.token;
    r2r.sourceStem = data.name || (files[0].name || "source").replace(/\.[^.]+$/, "");
    r2r.hasScene = !!data.has_scene;
    if (data.suggested_backend) r2rApplySuggestedBackend(data.suggested_backend);
    await r2rSrc.load(r2r.sourcePayload);
    r2rSrc.setTrajectory(data.trajectory);
    if (data.skeleton_preview) {
      r2rSrcSkel.load(data.skeleton_preview);
      const skBtn = document.getElementById("r2r-tg-src-skel");
      if (skBtn) skBtn.disabled = false;
    }
    const clipDur = Math.max(0.1, (data.num_frames - 1) / (data.framerate || 30));
    r2rLoadSrcScene(data.scaled_scene, data.token, clipDur);
    r2rVis.srcRobot = true;
    r2rVis.srcSkel = false;
    r2rVis.srcEnv = !!data.scaled_scene;
    r2rVis.tgtRobot = false;
    r2rVis.tgtSkel = false;
    r2rVis.tgtEnv = false;
    player.ready(r2rSrc.clipDuration || 1);
    player.seek(0);
    r2rApplyStage();
    r2rFocus(r2rSrc);
    player.setPlaying(true);
    const prof = data.upload_profile ? ` · ${data.upload_profile}` : "";
    st.textContent = `已加载：${data.num_frames} 帧 @ ${data.framerate.toFixed(1)} fps${prof}`;
    if (bar) bar.style.width = "100%";
    toast(`上传成功：${data.num_frames} 帧，正在播放源机器人轨迹`);
    await r2rUpdateRetargetBtn();
  } catch (e) {
    st.textContent = "";
    if (prog) prog.style.display = "none";
    toast(e.message, true);
  }
}

function r2rSuggestedBackendForProfile(profile) {
  const p = (profile || "mimic").toLowerCase();
  if (p === "intermimic" || p === "meshmimic") return "interaction_mesh";
  return "newton";
}

function r2rApplySuggestedBackend(backend) {
  if (!backend) return;
  const rb = document.getElementById("r2r-backend");
  const bb = document.getElementById("r2r-batch-backend");
  if (rb) rb.value = backend;
  if (bb) bb.value = backend;
}

function r2rIngestTraj(files, profile = "auto") {
  if (!files?.length) return;
  if (profile && profile !== "auto") {
    r2rApplySuggestedBackend(r2rSuggestedBackendForProfile(profile));
  }
  void r2rUploadTraj(files, profile);
}

// --------------------------------------------------------------- retarget
async function r2rRunRetarget() {
  if (!r2r.sourceToken || !r2r.targetName) {
    toast("请先上传源轨迹并加载目标机器人", true);
    return;
  }
  await r2rUpdateRetargetBtn();
  if (!r2r.calibrated) {
    toast("目标机器人尚未针对此源机器人标定，请先完成标定", true);
    await r2rStartCalib({ auto: true });
    return;
  }
  const prog = document.getElementById("r2r-progress");
  const bar = prog.querySelector(".bar");
  const status = document.getElementById("r2r-status");
  prog.style.display = "block";
  prog.classList.add("indet");
  bar.style.width = "0%";
  status.innerHTML = `<span class="spin"></span> 正在 retarget…（新机器人首次较慢）`;
  document.getElementById("r2r-retarget-btn").disabled = true;
  try {
    const body = {
      target: r2r.targetName,
      source: r2r.sourceName,
      source_token: r2r.sourceToken,
      backend: document.getElementById("r2r-backend")?.value || "newton",
    };
    const fps = parseOptionalFps(document.getElementById("r2r-retarget-fps"));
    if (fps) body.retarget_fps = fps;
    const { job_id } = await API.post("/api/r2r/retarget", body);
    const j = await pollJob(job_id, (jp) => {
      setRetargetProgress(prog, bar, jp);
      status.innerHTML = `<span class="spin"></span> ${jp.message || "正在 retarget…"}`;
    });
    prog.classList.remove("indet");
    bar.style.width = "100%";
    if (!r2r.targetPayload) {
      r2r.targetPayload = await API.post("/api/robot/select", { name: r2r.targetName });
    }
    await r2rTgt.load(r2r.targetPayload);
    r2rTgt.setTrajectory(j.result.trajectory);
    if (j.result.scaled_preview) {
      r2rTgtSkel.load(j.result.scaled_preview);
      document.getElementById("r2r-tg-tgt-skel").disabled = false;
    }
    const tgtDur = Math.max(
      0.1,
      ((j.result.num_frames || 1) - 1) / (j.result.source_fps || 30),
    );
    r2rLoadTgtScene(j.result.scaled_scene, r2r.sourceToken, tgtDur);
    document.getElementById("r2r-tg-tgt-robot").disabled = false;
    r2r.exportToken = j.result.export_token;
    r2r.exportHasScene = !!j.result.has_scene;
    r2r.resultStem = j.result.stem || r2r.sourceStem || "r2r";
    r2rVis.tgtRobot = true;
    r2rVis.tgtSkel = !!j.result.scaled_preview;
    r2rVis.tgtEnv = !!j.result.scaled_scene;
    player.ready(r2rTgt.clipDuration || 1);
    player.seek(0);
    r2rApplyStage();
    r2rFocus(r2rTgt);
    player.setPlaying(true);
    status.textContent =
      `完成：${j.result.num_frames} 帧 @ ${(j.result.source_fps || 30).toFixed(1)} fps`;
    document.getElementById("r2r-export-card").style.display = "block";
    document.getElementById("r2r-export-fps").value = "";
    const r2rBundleHint = document.getElementById("r2r-export-bundle-hint");
    if (r2rBundleHint) r2rBundleHint.style.display = j.result.has_scene ? "block" : "none";
    toast("R2R Retarget 完成，正在播放目标机器人");
  } catch (e) {
    status.textContent = "";
    prog.classList.remove("indet");
    toast(e.message, true);
  } finally {
    document.getElementById("r2r-retarget-btn").disabled = false;
  }
}

// --------------------------------------------------------------- batch
function r2rRenderBasket() {
  const list = document.getElementById("r2r-basket-list");
  if (!list) return;
  list.innerHTML = "";
  for (const e of r2r.basket) {
    const row = document.createElement("div");
    row.className = "basket-row";
    const label = e.export_subdir ? `${e.export_subdir}/${e.stem}` : e.stem;
    row.innerHTML = `<span>${label} · ${e.upload_profile || "mimic"}</span><button class="rm">×</button>`;
    row.querySelector(".rm").onclick = () => {
      r2r.basket = r2r.basket.filter((x) => x !== e);
      r2rRenderBasket();
    };
    list.appendChild(row);
  }
  document.getElementById("r2r-basket-count").textContent = String(r2r.basket.length);
  const runBtn = document.getElementById("r2r-batch-run");
  if (runBtn) runBtn.disabled = !(r2r.basket.length && r2r.targetName && r2r.sourceName);
}

async function r2rIngestBasket(files, profile = "auto") {
  if (!files?.length) return;
  showLoading(`R2R 批量上传… (${files.length} 个文件)`);
  try {
    const { job_id } = await uploadFilesXHR(
      `/api/r2r/basket/upload?profile=${encodeURIComponent(profile)}`,
      files,
      {},
      (frac) => setLoadingProgress(frac * 0.4, "上传中…"),
    );
    const payload = await waitMotionJob(job_id, (frac, sub) => {
      setLoadingProgress(0.4 + frac * 0.6, sub);
    }, { uploadFrac: 0.4 });
    const entries = payload.entries || [];
    for (const e of entries) {
      if (!r2r.basket.find((x) => x.source_path === e.source_path)) r2r.basket.push(e);
    }
    const last = entries[entries.length - 1];
    if (last?.suggested_backend) r2rApplySuggestedBackend(last.suggested_backend);
    else if (profile && profile !== "auto") {
      r2rApplySuggestedBackend(r2rSuggestedBackendForProfile(profile));
    }
    r2rRenderBasket();
    toast(`已加入篮子：${entries.length} 个 clip（${payload.profile || profile}）`);
  } catch (e) {
    toast(e.message, true);
  } finally {
    hideLoading();
  }
}

// --------------------------------------------------------------- wiring
function r2rInit() {
  void r2rPopulateSelects();
  for (const el of document.querySelectorAll("[data-r2r-profile]")) {
    const prof = el.dataset.r2rProfile || "mimic";
    setupDropzone(el, (files) => r2rIngestTraj(files, prof));
  }
  document.querySelectorAll("[data-r2r-pick]").forEach((btn) => {
    btn.onclick = async () => {
      const prof = btn.dataset.r2rPick || "mimic";
      const folder = btn.dataset.folder === "1";
      await r2rIngestTraj(await pickFiles({ folder }), prof);
    };
  });
  setupDropzone(document.getElementById("r2r-basket-drop"), (files) => r2rIngestBasket(files, "auto"));
  for (const [id, key, view] of [
    ["r2r-tg-src-robot", "srcRobot", r2rSrc],
    ["r2r-tg-src-skel", "srcSkel", r2rSrcSkel],
    ["r2r-tg-src-env", "srcEnv", r2rSrcEnv],
    ["r2r-tg-tgt-robot", "tgtRobot", r2rTgt],
    ["r2r-tg-tgt-skel", "tgtSkel", r2rTgtSkel],
    ["r2r-tg-tgt-env", "tgtEnv", r2rTgtEnv],
  ]) {
    document.getElementById(id)?.addEventListener("click", (ev) => {
      if (ev.currentTarget.disabled) return;
      r2rVis[key] = !r2rVis[key];
      r2rApplyStage();
    });
  }
  document.getElementById("r2r-source-load").onclick = async () => {
    const name = document.getElementById("r2r-source-select").value;
    if (!name) return;
    toast("加载源机器人…");
    try {
      r2r.sourcePayload = await API.post("/api/robot/select", { name });
      r2r.sourceName = name;
      await r2rSrc.load(r2r.sourcePayload);
      switchInspectorPanel("r2r");
      if (!r2r.active) r2rEnterPanel();
      r2rApplyStage();
      r2rFocus(r2rSrc);
      document.getElementById("r2r-source-status").textContent =
        `源机器人：${r2r.sourcePayload.display_name}（上传轨迹后可播放）`;
      toast(`源机器人已加载：${r2r.sourcePayload.display_name}`);
      await r2rMaybeAutoCalib();
      r2rRenderBasket();
    } catch (e) { toast(e.message, true); }
  };
  document.getElementById("r2r-target-load").onclick = async () => {
    const name = document.getElementById("r2r-target-select").value;
    if (!name) return;
    toast("加载目标机器人…");
    try {
      r2r.targetPayload = await API.post("/api/robot/select", { name });
      r2r.targetName = name;
      document.getElementById("r2r-target-status").textContent =
        `目标机器人：${r2r.targetPayload.display_name}`;
      toast(`目标机器人已加载：${r2r.targetPayload.display_name}`);
      await r2rMaybeAutoCalib();
      r2rRenderBasket();
    } catch (e) { toast(e.message, true); }
  };
  document.getElementById("r2r-calib-btn").onclick = () => void r2rStartCalib();
  document.getElementById("r2r-calib-zero").onclick = () => {
    const z = {};
    for (const j of Object.keys(r2r.calibQ)) z[j] = 0;
    r2rBuildSliders(z, r2r.calibLimits);
    toast("已归零（URDF 零位）");
  };
  document.getElementById("r2r-calib-cancel").onclick = () => {
    r2rExitCalib();
    toast("已取消标定");
    void r2rUpdateRetargetBtn();
  };
  document.getElementById("r2r-calib-save").onclick = () => void r2rSaveCalib();
  document.getElementById("r2r-retarget-btn").onclick = () => void r2rRunRetarget();
  document.getElementById("r2r-export-btn").onclick = async () => {
    if (!r2r.exportToken) { toast("请先完成 Retarget", true); return; }
    const fps = parseFloat(document.getElementById("r2r-export-fps").value);
    const fmt = document.getElementById("r2r-export-format")?.value || "csv";
    let url = `/api/export/${r2r.exportToken}?fmt=${encodeURIComponent(fmt)}`;
    if (fps && fps > 0) url += `&fps=${fps}`;
    if (!document.getElementById("r2r-csv-header").checked) url += "&csv_header=0";
    const stem = r2r.resultStem || "r2r";
    const name = r2r.exportHasScene || fmt === "pkl"
      ? `${stem}_export.zip`
      : (fmt === "pkl" ? `${stem}.pkl` : `${stem}.csv`);
    try {
      await triggerBrowserDownload(url, name);
      toast("已开始下载（保存到浏览器默认下载目录）");
    } catch (e) { toast(e.message, true); }
  };
  document.getElementById("r2r-basket-clear")?.addEventListener("click", () => {
    r2r.basket = [];
    r2rRenderBasket();
  });
  document.getElementById("r2r-batch-run")?.addEventListener("click", async () => {
    if (!r2r.basket.length || !r2r.targetName || !r2r.sourceName) return;
    const prog = document.getElementById("r2r-batch-progress");
    const bar = prog?.querySelector(".bar");
    const status = document.getElementById("r2r-batch-status");
    prog.style.display = "block";
    if (bar) bar.style.width = "0%";
    status.innerHTML = `<span class="spin"></span> 批量 R2R 处理中…`;
    try {
      const body = {
        target: r2r.targetName,
        source: r2r.sourceName,
        entries: r2r.basket,
        backend: document.getElementById("r2r-batch-backend")?.value || "newton",
        out_dir: document.getElementById("r2r-batch-out")?.value || "r2r_batch_export",
        format: document.getElementById("r2r-export-format")?.value || "csv",
        csv_header: document.getElementById("r2r-batch-csv-header")?.checked !== false,
      };
      const exFps = parseOptionalFps(document.getElementById("r2r-batch-export-fps"));
      const rtFps = parseOptionalFps(document.getElementById("r2r-retarget-fps"));
      if (exFps) body.export_fps = exFps;
      if (rtFps) body.retarget_fps = rtFps;
      const { job_id } = await API.post("/api/r2r/batch/retarget", body);
      const j = await pollJob(job_id, (jp) => {
        if (bar) bar.style.width = `${Math.max(2, (jp.progress || 0) * 100).toFixed(0)}%`;
        status.textContent = jp.message || "";
      });
      if (bar) bar.style.width = "100%";
      const r = j.result;
      status.textContent = `完成：${r.written?.length ?? 0} 个 clip`;
      if (r.download_name) {
        await triggerBrowserDownload(`/api/job/${job_id}/download`, r.download_name);
        toast("批量 ZIP 已开始下载");
      }
    } catch (e) {
      status.textContent = "";
      toast(e.message, true);
    }
  });
  r2rRenderBasket();
}

// =================================================================  INIT
animate(); // start the render loop now that `player` is initialised
window.__hh = { skel, mesh, skin, scaledSkel, robot, player, scene, world }; // debug handle
window.__hhtoolsReady = true;

// Bridge for the optional dataset-viz module (loaded after this file). Exposes
// the few helpers it needs without making it depend on app.js internals.
window.__hhApp = {
  API,
  toast,
  loadLibraryEntry,
  previewRobotClip,
  populateDvRobotSelect,
  addToBasket,
  switchInspectorPanel,
  getLibrarySourceRoot: () => libSourceRoot,
  uploadFilesXHR,
};

async function verifyUiBuild() {
  try {
    const h = await API.get("/api/health");
    const el = document.getElementById("ui-build");
    if (el) el.textContent = `UI·${h.ui_build || "?"}`;
    if (h.motions_library_root) libMotionsRoot = h.motions_library_root;
    updateMotionsLibraryHint();
    const assetsHint = document.getElementById("motion-assets-hint");
    if (assetsHint && h.source_root) assetsHint.textContent = h.source_root;
    const missingFeatures =
      !h.ui_features?.view_hud ||
      !h.ui_features?.scaled_skeleton_toggle ||
      h.ui_features?.merged_robot_panel === false;
    if (missingFeatures) {
      toast(
        "服务端静态资源可能过旧（缺少新版 UI 特性）。请在仓库根目录执行 uv sync 后 uv run hhtools web 重启。",
        true,
      );
    }
  } catch {
    /* offline / API down — boot overlay handles it */
  }
}

(async function init() {
  wrapSelectDropdowns();
  initPanelLayout();
  document.getElementById("lib-link-path")?.addEventListener("click", () => linkLibraryPath());
  await verifyUiBuild();
  await Promise.all([loadReferenceCatalog(), refreshLibrary(), refreshRobotList()]);
  r2rInit();
  const tour = initTutorial(toast);
  tour.maybeAutoStart();
})();
