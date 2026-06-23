"""Viser application entry point.

Responsibilities:

- Construct the Viser server on ``host:port`` and configure a light studio theme.
- Recursively scan a raw-asset root (default ``assets/motions/``) through
  :func:`hhtools.viewer.library.scan_library` and expose it as a folder-indexed, live
  searchable library. Arbitrary grouping layers (``mimic`` / ``intermimic`` /
  ``meshmimic`` / ...) are transparent — only the innermost dataset folder name
  matters. NPZ conversion is performed **lazily** on first play.
- Manage a per-session :class:`EphemeralCache`: converted files land in a
  ``tempfile.mkdtemp`` under ``/tmp`` and the whole directory is wiped on
  shutdown (either normal quit or ``Ctrl+C``). Whether the user saved or not is
  irrelevant — saves go to a *separate* directory. ``Save clip`` / ``Save folder``
  buttons copy entries into ``assets/save_npz/<FolderLabel>/``, which lives
  outside the cache dir and therefore survives the shutdown wipe.
- Build a :class:`PlaybackPanel`, a :class:`SkeletonRenderer`, a
  :class:`CapsuleMeshRenderer`, and an :class:`ObjectsRenderer`. The three can be
  toggled independently.
- Auto-apply up-axis conversion (Y → Z), snap feet to ground (z_min → 0), optionally
  centre the frame-0 root at the world origin, and drop virtual root nodes such as
  BVH ``Root`` / ``Reference`` placeholders.
- Visibility toggles flip each renderer's ``visible`` flag instead of rebuilding, so
  playback never pauses when the user ticks a checkbox.
- Fit the camera to the frame-0 pose so the character is centred on load.
"""

from __future__ import annotations

import atexit
import os
import signal
import threading
import math
import time
import warnings
from pathlib import Path

import numpy as np

# NumPy 2.4+ warns (and may eventually error under strict filters) when unpickling
# older array metadata (``align=0``) from third-party .npz/.npy under assets/.
warnings.filterwarnings(
    "ignore",
    message="align should be passed as Python or NumPy boolean",
    category=Warning,
)

from hhtools._version import __version__
from hhtools.core.coord import to_up_axis
from hhtools.core.grounding import (
    foot_floor_z_in_positions,
    human_source_floor_z_world,
    terrain_heightfield_z_offset_world,
    use_split_terrain_grounding,
)
from hhtools.core.motion import Motion
from hhtools.core.scene import SceneObject
from hhtools.io import npz
from hhtools.io.base import load_motion
from hhtools.viewer.anatomy import (
    center_motion_root_xy,
    compute_bone_radii,
    dense_rig_viz_exclude_indices,
    degenerate_auxiliary_bone_indices,
    detect_virtual_root,
    exclude_joint_from_compact_scaled_preview,
    exclude_unmapped_head_neck_from_scaled_preview,
    scaler_compact_bead_row_indices,
    snap_motion_to_ground,
)
from hhtools.viewer.cache import EphemeralCache, _attach_library_folder_label
from hhtools.viewer.library import (
    LibraryEntry,
    filter_entries,
    list_folders,
    scan_library,
)
from hhtools.viewer.panels import PlaybackPanel
from hhtools.viewer.renderers import (
    CapsuleMeshRenderer,
    ObjectsRenderer,
    TerrainHeightfieldRenderer,
    ReferenceSkeletonRenderer,
    RobotAnimator,
    ScaledSkeletonRenderer,
    SkeletonRenderer,
    SkinnedMeshRenderer,
)
from hhtools.viewer.theme import PALETTE, TITLEBAR
from hhtools.robot import (
    RobotPreset,
    URDFRobotModel,
    header_columns,
    list_presets as list_robot_presets,
    load_robot,
)

# File extensions whose loaders preserve a skinned-mesh attachment on the Motion.
# When loading these we bypass the NPZ cache for display purposes (the cache is still
# used when the user clicks Save) because the NPZ round-trip strips ``meta["skinned_mesh"]``
# — without this bypass, picking a GLB clip from the library would only render the
# skeleton even though the file actually carries a mesh.
_MESH_CARRYING_EXTS = {".glb", ".gltf"}

# Dataset adapters that drive a parametric body model (SMPL / SMPL-H / SMPL-X) and can
# optionally bake per-frame vertex caches when invoked with ``with_mesh=True``.  The
# viewer routes these through a direct-adapter path (bypassing the NPZ cache) whenever
# the "Skinned mesh" toggle is on, because NPZ caching strips mesh attachments from
# ``motion.meta``.  Any dataset whose ``load_motion`` accepts ``with_mesh=True`` can
# be added here — adding to this set is the only change required.
_SMPL_BAKED_DATASETS: frozenset[str] = frozenset(
    {"amass", "motion_x", "phuma", "gvhmr", "kungfu_athlete"}
)

# Datasets whose motion data follow SMPL-style joint naming — calibration
# reference should default to ``smpl`` (GVHMR / Kungfu athlete included).
_SMPL_FAMILY_DATASETS: frozenset[str] = frozenset(
    {"amass", "motion_x", "phuma", "kungfu_athlete"}
)
_GVHMR_DATASETS: frozenset[str] = frozenset({"gvhmr"})

# Pre-unified NPZ clips shipped as a *bundle* (e.g. ``clip.npz`` + sibling ``*_terrain.obj``).
# The ephemeral cache copies only the NPZ into ``/tmp``; relative ``objects_mesh_paths``
# then resolve against the cache directory where the ``.obj`` is absent, breaking terrain
# and risking viewer instability.  Skip the cache for display — same idea as GLB bypass.
_DIRECT_SOURCE_NPZ_DATASETS: frozenset[str] = frozenset({"unified_npz"})

_FOLDER_ALL = "All"
_MAX_LIBRARY_OPTIONS = 200  # Hard cap so Viser's dropdown doesn't bloat with huge datasets.
_SAVE_LOG_EMPTY = "<span style='opacity:0.55'>No saves yet.</span>"


def _suggested_calibration_reference(
    motion: Motion,
    entry: LibraryEntry | None,
    source_path: Path | None,
) -> str | None:
    """Pick a calibration ``Reference pose`` dropdown value for the loaded clip."""

    if isinstance(entry, LibraryEntry):
        ds = (entry.dataset or "").lower()
        if ds in _GVHMR_DATASETS:
            return "gvhmr"
        # OMOMO + meshmimic are SMPL-H style mocap; ``lafan_bvh`` reference geometry
        # mismatches bind proportions and breaks scaler rest + scale preview.
        if ds == "omomo" or ds.startswith("meshmimic"):
            return "smpl"
    # Prefer the library row's *authored* file (``.glb``) for format;
    # ``source_path`` from callers is often a converted cache ``.npz`` and would
    # miss the extension-based ``glb`` hint.
    ext_path: Path | None = None
    if isinstance(entry, LibraryEntry):
        ext_path = entry.source_path
    elif source_path is not None:
        ext_path = source_path
    if ext_path is not None:
        ext = ext_path.suffix.lower()
        if ext in (".glb", ".gltf"):
            return "glb"
    if isinstance(entry, LibraryEntry):
        ds = (entry.dataset or "").lower()
        if ds in _SMPL_FAMILY_DATASETS:
            return "smpl"
        if ds == "soma":
            return "soma_bvh"
        if ds in ("lafan", "lafan1"):
            return "lafan_bvh"
        if ds == "xsens_mocap":
            return "xsens_mocap"
    from hhtools.retarget.newton_basic.human_aliases import (
        list_detected_rig_type,
    )

    rig = list_detected_rig_type(motion.hierarchy.bone_names)
    return {
        "SMPL/SMPL-H/SMPL-X": "smpl",
        "SOMA BVH": "soma_bvh",
        "Holosoma / SMPL-H mocap": "smpl",
        "Mixamo/CMU/LAFAN": "lafan_bvh",
        "Xsens mocap BVH": "xsens_mocap",
    }.get(rig)


def _dataset_defaults_to_interaction_mesh(
    dataset: str | None,
    motion: Motion | None = None,
    entry: "LibraryEntry | None" = None,
) -> bool:
    """True when the clip has terrain / object interaction that benefits from
    the Laplacian interaction-mesh backend rather than plain Newton IK.

    Detection layers, in order:

    1. **Adapter dataset name** — ``omomo`` and any ``meshmimic*`` flavour
       always need MPC-SQP because they ship terrain / object props.
    2. **Source-path grouping segment** — ``intermimic/<adapter>/...`` and
       ``meshmimic/<adapter>/...`` are the conventional library layouts for
       human-object / human-terrain interaction clips
       (see :mod:`hhtools.viewer.library` docstring), regardless of the
       adapter name they end up registering under.  Triggering on the
       parent directory name lets ``intermimic/OMOMO/<clip>``,
       ``intermimic/AMASS/<clip>`` and any future
       ``intermimic/<X>/<clip>`` clips route to interaction-mesh
       automatically without per-adapter wiring.
    3. **Motion-side hints** — non-empty ``motion.objects`` or a populated
       ``motion.terrain`` heightfield.  This is the catch-all for clips
       that don't live under a recognised grouping directory but still
       carry interaction data, e.g. ``parc_ms`` exports landing under
       ``meshmimic/parc_ms/<clip>/`` as ``unified_npz`` files (the
       ``dataset`` field is ``'unified_npz'`` so layer 1 misses them, and
       layer 2 catches them via the ``meshmimic/`` parent segment, but
       motion-side detection backstops layouts I haven't anticipated).
    """

    ds = (dataset or "").lower()
    if ds == "omomo":
        return True
    if ds.startswith("meshmimic"):
        return True

    # Path-based grouping check.  ``intermimic`` / ``meshmimic`` are
    # *conventional parent directories* used to mark interaction clips;
    # they never appear as ``LibraryEntry.dataset`` (the adapter name
    # comes from the leaf directory) so we have to look at the source
    # path explicitly.  Compare on path *parts* rather than substring
    # matches so a clip name that happens to contain "intermimic" by
    # coincidence doesn't accidentally trigger the routing.
    if entry is not None:
        try:
            parts_lower = {p.lower() for p in entry.source_path.parts}
        except Exception:
            parts_lower = set()
        if "intermimic" in parts_lower or "meshmimic" in parts_lower:
            return True

    if motion is not None:
        if getattr(motion, "objects", None):
            return True
        if getattr(motion, "terrain", None) is not None:
            return True
    return False


def _effective_retarget_backend(
    dropdown_value: str,
    entry: LibraryEntry | None,
    motion: Motion | None = None,
) -> str:
    """Resolve Robot-tab ``Retarget backend`` to ``newton`` or ``interaction_mesh``."""

    if dropdown_value == "Auto":
        ds: str | None = None
        if isinstance(entry, LibraryEntry):
            ds = entry.dataset
        elif motion is not None:
            meta = getattr(motion, "meta", None)
            if isinstance(meta, dict):
                ds = meta.get("dataset")
        return (
            "interaction_mesh"
            if _dataset_defaults_to_interaction_mesh(
                str(ds) if ds else None, motion, entry,
            )
            else "newton"
        )
    if dropdown_value == "Interaction mesh":
        return "interaction_mesh"
    return "newton"


def _html_escape(text: object) -> str:
    """Escape characters that break Viser's MDX-based markdown renderer.

    Viser ≥1.0 renders markdown via ``@mdx-js/mdx`` which treats ``{``/``}``
    as JSX expression delimiters.  ``html.escape`` covers ``<>&"'`` but not
    braces, so we do a second pass to neutralise them as HTML entities.
    """
    import html
    s = html.escape(str(text))
    return s.replace("{", "&#123;").replace("}", "&#125;")
_NOTIFICATION_TTL_SECONDS = 6.0

# Planar offset in metres applied to the robot (and its scaled-skeleton
# preview) so the robot stands well beside the source-motion human rather
# than overlapping it.  +X is "in front" in our Z-up convention; we picked
# +Y so humans and robots sit side-by-side in the default camera view.
# The 5 m gap keeps the robot and its scaled terrain/objects visually
# separate from the source skeleton + original terrain, even for large
# parkour courses or box-climbing scenes.
ROBOT_WORLD_OFFSET: tuple[float, float, float] = (0.0, 5.0, 0.0)


def run_viewer(
    motion: str | Path | None = None,
    motion_dir: str | Path | None = None,
    source_root: str | Path | None = None,
    save_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    keep_cache: bool = False,
    host: str = "127.0.0.1",
    port: int = 8008,
    share: bool = False,
    autoload_first_clip: bool = False,
) -> None:
    """Spawn a Viser viewer.

    Args:
        motion: Optional single NPZ to preload before the UI comes up.
        motion_dir: Optional flat directory of pre-converted NPZs (legacy mode).
        source_root: Raw dataset root to scan (``assets/motions`` by default).
            Scanned recursively — any nested grouping layers are ignored and only
            innermost dataset folder names drive routing.
        save_dir: Destination for persisted NPZs (``assets/save_npz`` by default).
        cache_dir: Per-session NPZ cache directory. ``None`` (the default) mints
            a fresh ``tempfile.mkdtemp`` under ``/tmp`` and rmtrees it on
            shutdown. Pass a path if you want to hand-pick a location — only
            files this session wrote are removed on cleanup in that case.
        keep_cache: Debug escape hatch; if ``True`` we skip :meth:`cleanup` on
            shutdown so developers can inspect what got written. End users
            should never need this.
        host / port / share: Viser transport settings.
        autoload_first_clip: If ``True`` (or env ``HHTOOLS_UI_AUTOLOAD=1``), load the
            first library entry on startup. Default ``False`` avoids brittle assets
            (legacy pickles, strict ``PYTHONWARNINGS=error``, …) killing the Viser
            handshake before the UI becomes usable.
    """
    try:
        import viser
    except ImportError as exc:
        raise ImportError(
            "Viser is required for hhtools ui. Install it with: pip install 'hhtools[viewer]'"
        ) from exc

    source_root_path = Path(source_root) if source_root is not None else Path("assets/motions")
    # ``cache_dir`` stays Optional on purpose: EphemeralCache.create(None) picks a
    # session-owned tempfile that gets fully rmtree'd on shutdown. Only when the
    # caller hands us an explicit path do we treat it as "shared" and prune just
    # the files we wrote.
    cache_dir_param: Path | None = Path(cache_dir) if cache_dir is not None else None
    save_dir_path = Path(save_dir) if save_dir is not None else Path("assets/save_npz")

    cache = EphemeralCache.create(cache_dir=cache_dir_param, save_dir=save_dir_path)
    cache_dir_path = cache.cache_dir  # resolved path (temp dir if cache_dir_param was None)

    # Register cleanup hooks up-front so even a fatal exception during UI setup still
    # clears the ephemeral cache we created in the previous line.
    if not keep_cache:
        atexit.register(cache.cleanup)

        _prev_sigint = signal.getsignal(signal.SIGINT)

        def _on_sigint(signum, frame):  # type: ignore[no-untyped-def]
            cache.cleanup()
            # Restore and re-raise so the normal KeyboardInterrupt still propagates.
            signal.signal(signal.SIGINT, _prev_sigint)
            raise KeyboardInterrupt()

        signal.signal(signal.SIGINT, _on_sigint)

    entries: list[LibraryEntry] = scan_library(source_root_path)
    _want_autoload_first = autoload_first_clip or (
        os.environ.get("HHTOOLS_UI_AUTOLOAD", "").strip().lower() in ("1", "true", "yes")
    )

    server = viser.ViserServer(host=host, port=port, label=TITLEBAR)

    server.gui.configure_theme(
        dark_mode=False,
        brand_color=_hex_to_rgb(PALETTE.accent),
        control_layout="collapsible",
        control_width="large",
        show_logo=False,
        show_share_button=False,
    )

    server.scene.world_axes.visible = False

    # Strong grid + semi-opaque ground plane so feet / capsules vs z=0 is obvious.
    server.scene.add_grid(
        "/ground",
        width=16.0,
        height=16.0,
        plane="xy",
        cell_size=0.25,
        section_size=1.0,
        cell_thickness=1.75,
        section_thickness=2.5,
        cell_color=_hex_to_rgb(PALETTE.grid_cell),
        section_color=_hex_to_rgb(PALETTE.grid_section),
        fade_distance=50.0,
        shadow_opacity=0.18,
    )

    server.scene.add_light_directional(
        "/lights/key", color=(255, 252, 245), intensity=1.35,
        wxyz=(0.924, 0.383, 0.0, 0.0),
    )
    server.scene.add_light_directional(
        "/lights/rim", color=(210, 225, 245), intensity=0.75,
        wxyz=(0.383, -0.383, 0.707, 0.0),
    )
    server.scene.add_light_ambient("/lights/ambient", color=(238, 240, 245), intensity=0.82)

    # ---- Top-level banner ----------------------------------------------------
    # Keep the top of the sidebar airy: a single low-contrast badge in place of the
    # old h2 title. The full brand string still lives in the browser titlebar via
    # ``label=TITLEBAR`` on the Viser server above.
    server.gui.add_markdown(
        f"<div style='padding:4px 0 2px 0;"
        f"color:{PALETTE.text_muted};font-size:0.82em;letter-spacing:0.04em;'>"
        f"<span style='color:{PALETTE.accent};font-weight:600;'>HHTOOLS</span>"
        f" &nbsp;·&nbsp; human ↔ humanoid studio"
        f" &nbsp;·&nbsp; <code style='opacity:0.7'>v{__version__}</code>"
        f"</div>"
    )

    # ---- Motion library (folder + search + clip selector) --------------------
    # We compute initial state, then wire up live filtering. The clip dropdown is
    # rebuilt whenever search / folder change. NPZ conversion only happens when the
    # user actually picks a clip, via EphemeralCache.load_motion.

    # State dict for the library. `entries_filtered[index]` corresponds to the
    # currently-selected dropdown option. Storing it lets the on_update callback
    # grab the full `LibraryEntry` without reparsing the displayed label.
    lib_state: dict[str, object] = {
        "entries": entries,
        "filtered": entries,
        "current": entries[0] if entries else None,
    }

    folder_options = [_FOLDER_ALL] + list_folders(entries)
    initial_filtered = entries[:_MAX_LIBRARY_OPTIONS]
    initial_labels = [_unique_label(e) for e in initial_filtered]
    lib_state["filtered"] = initial_filtered

    # Top-level tab group.  Workflow split:
    #   - Motion: everything existing (Library + Display) — the current default UX.
    #   - Robot:  new in robot_core — URDF preset picker + static T-pose render.
    # Viewer convention: each tab opens with its primary content *expanded* so
    # a first-time visitor sees something useful without hunting.  Secondary
    # groups (Upload, Transformations) stay collapsed to keep the scroll short.
    tabs = server.gui.add_tab_group()

    with tabs.add_tab("Motion", icon="walk"):
        with server.gui.add_folder("Library", expand_by_default=True):
            # Progress row: hidden by default, only materialised during slow work (first
            # NPZ conversion, Save whole folder, etc.). We live-update both widgets AND
            # call server.flush() from the progress helper so the browser paints during
            # long synchronous Python calls.
            progress_label_md = server.gui.add_markdown("")
            progress_bar = server.gui.add_progress_bar(
                value=0.0, visible=False, animated=False,
            )

            search_box = server.gui.add_text(
                "Search", initial_value="",
                hint="Case-insensitive, whitespace-separated tokens match in any order.",
            )
            folder_picker = server.gui.add_dropdown(
                "Folder",
                options=tuple(folder_options),
                initial_value=_FOLDER_ALL,
            )
            clip_picker = server.gui.add_dropdown(
                "Clip",
                options=tuple(initial_labels) if initial_labels else ("(no matches)",),
                initial_value=initial_labels[0] if initial_labels else "(no matches)",
            )
            # Expose the clip picker to the Robot tab *immediately* — otherwise
            # the Robot tab's motion-picker builds against an empty label list
            # (motion_sync would still be a bare dict when that tab renders).
            # ``_load_by_label`` is wired up later in this function because it
            # depends on ``_on_clip_pick`` being defined, but the read-only
            # getters work as soon as the picker exists.
            status_md = server.gui.add_markdown(_make_status(entries, entries, cache))
            save_clip_btn = server.gui.add_button(
                "Save this clip", icon="device-floppy",
                hint="Copy the current clip's NPZ into assets/save_npz.",
            )
            save_folder_btn = server.gui.add_button(
                "Save whole folder", icon="folder-plus",
                hint="Convert and persist every clip in the current folder filter.",
            )
            # Persistent save log (multi-line markdown). Auto-updated by save callbacks
            # with "saved N clip(s) to <path>" so users get permanent feedback even after
            # the transient toast notification disappears.
            save_log_md = server.gui.add_markdown(_SAVE_LOG_EMPTY)

            if not entries:
                save_clip_btn.disabled = True
                save_folder_btn.disabled = True

            # Clip info + Upload nested inside "Library" so the group is self-contained.
            with server.gui.add_folder("Clip info", expand_by_default=True):
                motion_label = server.gui.add_text("Name", initial_value="(none)")
                motion_label.disabled = True
                info_label = server.gui.add_text("Frames · FPS · Bones", initial_value="—")
                info_label.disabled = True
                axis_label = server.gui.add_text("Up axis (src → view)", initial_value="—")
                axis_label.disabled = True
                saved_label = server.gui.add_text("Persisted", initial_value="no")
                saved_label.disabled = True

            with server.gui.add_folder("Upload", expand_by_default=False):
                file_picker = server.gui.add_upload_button(
                    "Upload NPZ", icon="upload", mime_type=".npz"
                )

                @file_picker.on_upload
                def _on_upload(event):  # type: ignore[no-untyped-def]
                    upload = event.target.value
                    if upload is None:
                        return
                    tmp = Path("/tmp") / f"hhtools_upload_{int(time.time() * 1000)}_{upload.name}"
                    tmp.write_bytes(upload.content)
                    lib_state["current"] = None
                    _load_path(tmp)
                    saved_label.value = "upload (not persisted)"

        # ---- Display options (parent group) --------------------------------------
        # Appearance-style toggles and transform toggles share no state but users tend
        # to tweak them together, so keep them under one expandable parent.
        with server.gui.add_folder("Display", expand_by_default=True):
            with server.gui.add_folder("Appearance", expand_by_default=True):
                show_skeleton = server.gui.add_checkbox("Skeleton lines", initial_value=True)
                show_capsules = server.gui.add_checkbox("Capsule body mesh", initial_value=True)
                show_skinned_mesh = server.gui.add_checkbox(
                    "Skinned mesh (GLB/glTF)", initial_value=True,
                    hint=(
                        "Real skinned body mesh deformed per-frame via linear-blend skinning. "
                        "Only active when the clip carries one (GLB/glTF with skin data). "
                        "Automatically hidden for skeleton-only NPZ / BVH / SMPL clips."
                    ),
                )
                show_bone_names = server.gui.add_checkbox(
                    "Bone names", initial_value=False,
                    hint="Display bone/joint names as text labels on the skeleton.",
                )
                show_objects = server.gui.add_checkbox(
                    "Scene objects", initial_value=True,
                    hint="Draw any props (OMOMO largebox, etc.) carried by the subject.",
                )
                show_world_axes = server.gui.add_checkbox("World axes", initial_value=False)
                radius_scale = server.gui.add_slider(
                    "Capsule radius scale", min=0.3, max=2.0, step=0.05, initial_value=1.0,
                    hint="Scales both capsule tubes and joint beads uniformly.",
                )

            with server.gui.add_folder("Transformations", expand_by_default=False):
                snap_ground = server.gui.add_checkbox(
                    "Snap feet to ground (z_min → 0)", initial_value=True
                )
                center_xy = server.gui.add_checkbox(
                    "Centre root at origin (XY)", initial_value=True
                )
                hide_virtual_root = server.gui.add_checkbox(
                    "Hide virtual BVH root", initial_value=True
                )

    # ---- Robot tab (robot_core delivery) -------------------------------------
    # Static T-pose preview only — animation, IK and retargeting belong to the
    # later retarget_newton_basic / retarget_newton_mesh milestones.  Builds on
    # the Robot tab are stateless wrt Motion tab, so no closures leak between
    # them; the one piece of shared state is the viser scene (we draw robot
    # meshes under ``/robot/...`` which is disjoint from motion renderers).
    # Shared state for cross-tab access. The Motion tab populates ``state``
    # below (kept in the same closure); the Robot tab reads ``state['final']``
    # via :func:`_get_current_motion` so the "Retarget to selected motion"
    # button can grab whatever clip is currently on screen.
    state: dict[str, object] = {"raw": None, "final": None}

    def _get_current_motion() -> Motion | None:
        m = state.get("final")
        return m if isinstance(m, Motion) else None

    def _get_current_entry_id() -> str | None:
        """Return the sequence id of the Motion tab's currently-loaded clip, if any.

        Used to pick a default ``.csv`` stem when the user retargets a single
        clip; avoids writing ``hhtools_<robot>_<None>.csv`` when the tab
        label happens to be empty.
        """
        cur = lib_state.get("current")
        return getattr(cur, "sequence_id", None) if cur is not None else None

    def _get_filtered_entries() -> list:  # type: ignore[type-arg]
        """Snapshot of the dropdown's current filter — consumed by batch retarget."""
        return list(lib_state.get("filtered") or [])

    # Shared slot the Robot tab writes into and the main render loop reads on
    # every tick — lets the animator + scaled-skeleton renderers update in
    # lock-step with the playback panel without threading callbacks around.
    robot_state: dict[str, object] = {
        "animator": None,
        "retargeted": None,
        "scaled_renderer": None,
        "scaled_preview": None,
        "calibration_active": False,
        "robot_objects_renderer": None,
        "_prewarm_thread": None,
    }

    # Motion ↔ Robot tab bidirectional sync.  The Robot tab's motion picker
    # calls ``load_by_label`` (forwarding to the Library dropdown's loader),
    # and the Motion tab calls every registered listener after a successful
    # ``_load_entry`` so the Robot picker stays in sync without spinning up
    # a polling loop.  The listener receives the newly-selected label string.
    #
    # We populate the read-only getters (``all_labels``, ``current_label``)
    # here — *before* the Robot tab is built — so its initial label list
    # reflects the Library dropdown's current filter.  Without this ordering
    # the Robot tab's motion picker rendered with an empty option list.
    motion_sync: dict[str, object] = {
        "listeners": [],  # list[Callable[[str], None]]
        "folder_listeners": [],  # list[Callable[[str], None]]
        "search_listeners": [],  # list[Callable[[str], None]]
        "library_refresh_listeners": [],  # list[Callable[[tuple[str,...], str], None]]
        "current_label": lambda: clip_picker.value,
        "all_labels": lambda: list(clip_picker.options),
    }
    motion_sync["get_current_library_entry"] = lambda: lib_state.get("current")
    motion_sync["get_last_loaded_source_path"] = lambda: lib_state.get(
        "_last_loaded_source_path",
    )
    motion_sync.setdefault("on_motion_loaded", [])

    def _fire_motion_loaded_callbacks() -> None:
        for cb in list(motion_sync.get("on_motion_loaded", [])):
            try:
                cb()
            except Exception:
                pass

    def _notify_motion_changed(label: str) -> None:
        for cb in list(motion_sync.get("listeners", [])):
            try:
                cb(label)
            except Exception:
                pass

    def _notify_folder_changed(folder: str) -> None:
        for cb in list(motion_sync.get("folder_listeners", [])):
            try:
                cb(folder)
            except Exception:
                pass

    def _notify_search_changed(query: str) -> None:
        for cb in list(motion_sync.get("search_listeners", [])):
            try:
                cb(query)
            except Exception:
                pass

    _folder_sync_guard = {"active": False}
    _search_sync_guard = {"active": False}
    _clip_pick_guard = {"suppress": False}

    def _set_library_folder(value: str) -> None:
        """Set Motion-tab folder filter; Robot tab calls this for bidirectional sync."""
        if _folder_sync_guard["active"]:
            return
        opts = list(folder_picker.options)
        if value not in opts:
            return
        _folder_sync_guard["active"] = True
        try:
            if folder_picker.value != value:
                folder_picker.value = value
            # Always refresh here — ``folder_picker.on_update`` is suppressed
            # while the guard is active, so we must not rely on it to reload
            # the clip list / auto-load the first match.
            _refresh_clip_dropdown()
            _notify_folder_changed(folder_picker.value)
        finally:
            _folder_sync_guard["active"] = False

    def _set_library_search(value: str) -> None:
        """Set Motion-tab search box; Robot tab calls this for bidirectional sync."""
        if _search_sync_guard["active"]:
            return
        _search_sync_guard["active"] = True
        try:
            if search_box.value != value:
                search_box.value = value
            _refresh_clip_dropdown()
            _notify_search_changed(search_box.value)
        finally:
            _search_sync_guard["active"] = False

    motion_sync["get_folder"] = lambda: folder_picker.value
    motion_sync["folder_options"] = lambda: list(folder_picker.options)
    motion_sync["set_folder"] = _set_library_folder
    motion_sync["get_search"] = lambda: search_box.value
    motion_sync["set_search"] = _set_library_search

    # ``panel`` is built later in this same function (it needs the outer
    # ``state`` dict to already exist).  We pass a getter closure so the
    # Robot tab's callbacks resolve it lazily — by the time the user clicks
    # retarget, the panel is guaranteed to be live.
    _panel_slot: dict[str, object] = {"panel": None}

    def _get_panel():
        return _panel_slot.get("panel")

    # ``_work_lock`` + ``_run_async`` need to exist *before* ``_build_robot_tab``
    # so the Save / Retarget callbacks can capture them via kwargs (the helper
    # function lives at module scope and cannot close over ``run_viewer``'s
    # later-bound locals).  Both are self-contained — no progress-reporter or
    # cache dependency — so promoting them above the tab construction is safe.
    _work_lock = threading.Lock()

    def _run_async(name: str, fn, *args, **kwargs) -> threading.Thread:  # type: ignore[no-untyped-def]
        """Dispatch ``fn(*args, **kwargs)`` onto a named daemon thread and return it.

        Viser ``on_update`` / ``on_click`` callbacks run synchronously on the
        asyncio event-loop thread; long-running adapters / SMPL FK
        imports would freeze the WebSocket and starve the progress bar.
        Kicking work onto a daemon thread keeps the event loop free to ship
        frames and lets the ticker animate in real time.  Daemon threads
        ensure the process exits cleanly even if a worker is mid-IO.
        """
        thread = threading.Thread(
            target=fn, args=args, kwargs=kwargs, name=name, daemon=True,
        )
        thread.start()
        return thread

    def _apply_settings_to_motion(m: Motion) -> Motion:
        if m.up_axis != "Z":
            m = to_up_axis(m, "Z")
        if center_xy.value:
            m = center_motion_root_xy(m)
        if snap_ground.value:
            # Align with soma ``lafan_to_rp1_scaler_config.json`` ground-contact scale
            # (``ground_contact_z`` / body clearance), not extra heuristics.
            m = snap_motion_to_ground(m, margin=0.045)
        return m

    with tabs.add_tab("Robot", icon="robot"):
        _build_robot_tab(
            server,
            get_current_motion=_get_current_motion,
            get_current_entry_id=_get_current_entry_id,
            get_filtered_entries=_get_filtered_entries,
            motion_cache=cache,
            get_panel=_get_panel,
            robot_state=robot_state,
            motion_sync=motion_sync,
            source_root_path=source_root_path,
            save_dir_path=save_dir_path,
            run_async=_run_async,
            work_lock=_work_lock,
            apply_motion_pipeline=_apply_settings_to_motion,
        )

    _mirror_pair = motion_sync.pop("_library_progress_mirror_pair", None)
    _lib_progress_mirrors = [_mirror_pair] if _mirror_pair else []

    skeleton = SkeletonRenderer(
        server,
        line_color=_hex_to_rgb(PALETTE.human),
        joint_color=_hex_to_rgb(PALETTE.joint_bead),
    )
    capsules = CapsuleMeshRenderer(server, color=_hex_to_rgb(PALETTE.robot))
    # Muted brown-grey for the human mesh — reads on the light ground without
    # matching the robot capsule hue (PALETTE.robot).
    skinned_mesh_renderer = SkinnedMeshRenderer(
        server, color=(118, 108, 98), opacity=0.96,
    )
    # Default prop colour + translucency targets the OMOMO use case: small boxes /
    # chairs / mops that would otherwise hide the subject's hands. Large scene
    # geometry (terrain, floors) overrides both via per-object ``SceneObject.opacity``
    # and ``SceneObject.color`` so the ground plane can stay fully opaque and
    # neutral-slate without the prop renderer needing to know about it.
    objects_renderer = ObjectsRenderer(
        server, color=_hex_to_rgb(PALETTE.prop), opacity=0.65
    )
    robot_objects_renderer = ObjectsRenderer(
        server, root_name="/robot_objects", color=(100, 116, 139), opacity=1.0,
    )
    robot_state["robot_objects_renderer"] = robot_objects_renderer
    # Terrain travels through the pipeline as a TerrainHeightfield on
    # Motion.terrain (instead of a SceneObject), so it lives in its own
    # renderer that triangulates the (hf, dx, min_point) grid into a
    # mesh.  One instance for the source-frame Motion tab; another for
    # the robot-frame retarget preview, kept in robot_state so the Robot
    # tab can update it when the user switches motions.
    terrain_renderer = TerrainHeightfieldRenderer(
        server, root_name="/terrain_hf",
    )
    robot_terrain_renderer = TerrainHeightfieldRenderer(
        server, root_name="/robot_terrain_hf",
    )
    robot_state["robot_terrain_renderer"] = robot_terrain_renderer
    panel = PlaybackPanel(server, framerate=30.0, num_frames=1)
    # Publish the panel handle so the Robot tab's callbacks can reach it
    # (auto-play on retarget finish, reconfigure slider range, etc.).
    _panel_slot["panel"] = panel

    # Cross-tab visibility helpers: the Robot tab's calibration mode needs
    # to hide / restore Motion-tab renderers, but those live in a different
    # scope.  Register lazy callbacks in motion_sync so _build_robot_tab
    # can invoke them without holding direct references.
    def _hide_motion_renderers() -> None:
        skeleton.set_visible(False)
        capsules.set_visible(False)
        skinned_mesh_renderer.set_visible(False)
        objects_renderer.set_visible(False)
        terrain_renderer.set_visible(False)
        _clear_bone_labels()

    def _restore_motion_renderers() -> None:
        skeleton.set_visible(bool(show_skeleton.value))
        capsules.set_visible(bool(show_capsules.value))
        skinned_mesh_renderer.set_visible(
            bool(show_skinned_mesh.value) and skinned_mesh_renderer.has_mesh()
        )
        objects_renderer.set_visible(bool(show_objects.value))
        terrain_renderer.set_visible(bool(show_objects.value))

    motion_sync["hide_motion_renderers"] = _hide_motion_renderers
    motion_sync["restore_motion_renderers"] = _restore_motion_renderers

    # -------- Data-pipeline helpers ------------------------------------------

    def _rebuild_renderers(*, reset_playback: bool = False) -> None:
        """Recompute processed motion + per-bone radii + scene geometry.

        ``reset_playback=False`` (default): a Display option was toggled on the same
        clip. We preserve the current frame and play/pause state so playback is never
        interrupted — only the geometry changes under the cursor.

        ``reset_playback=True``: a brand-new motion clip was loaded. We reset the
        cursor to frame 0 and pause.
        """
        if reset_playback:
            inv = motion_sync.get("invalidate_robot_artifacts")
            if callable(inv):
                try:
                    inv()
                except Exception:
                    pass

        raw = state.get("raw")
        if not isinstance(raw, Motion):
            return
        m_final = _apply_settings_to_motion(raw)
        state["final"] = m_final

        exclude: set[int] = (
            degenerate_auxiliary_bone_indices(m_final)
            | dense_rig_viz_exclude_indices(m_final)
        )
        if hide_virtual_root.value and detect_virtual_root(m_final.hierarchy.bone_names):
            exclude = exclude | {0}

        radii = compute_bone_radii(
            m_final.hierarchy.bone_names,
            np.asarray(m_final.hierarchy.parent_indices, dtype=np.int32),
            m_final.positions[0],
        ).astype(np.float32)
        radii = radii * float(radius_scale.value)

        skeleton.set_motion(m_final, exclude_bones=exclude, bone_radii=radii)
        capsules.set_motion(m_final, exclude_bones=exclude, bone_radii=radii)
        skinned_mesh_renderer.set_motion(m_final)
        objects_renderer.set_objects(m_final.objects)
        terrain_renderer.set_terrain(m_final.terrain)
        _clear_bone_labels()
        if show_bone_names.value:
            _create_bone_labels(m_final, current_frame)

        skeleton.set_visible(bool(show_skeleton.value))
        capsules.set_visible(bool(show_capsules.value))
        skinned_mesh_renderer.set_visible(
            bool(show_skinned_mesh.value) and skinned_mesh_renderer.has_mesh()
        )
        objects_renderer.set_visible(bool(show_objects.value))
        terrain_renderer.set_visible(bool(show_objects.value))

        if reset_playback:
            panel.set_motion(m_final.framerate, m_final.num_frames)
            current_frame = 0
        else:
            current_frame = panel.reconfigure(m_final.framerate, m_final.num_frames)

        if show_skeleton.value:
            skeleton.set_frame(current_frame)
        if show_capsules.value:
            capsules.set_frame(current_frame)
        if show_skinned_mesh.value and skinned_mesh_renderer.has_mesh():
            skinned_mesh_renderer.set_frame(current_frame)
        if show_objects.value:
            objects_renderer.set_frame(current_frame)

    # -------- Motion loading --------------------------------------------------

    def _load_path(
        path: Path,
        *,
        progress_pin=None,  # type: ignore[no-untyped-def]
        library_entry: LibraryEntry | None = None,
    ) -> None:
        path = Path(path)
        lib_state["_last_loaded_source_path"] = str(path.resolve())
        inv = motion_sync.get("invalidate_robot_artifacts")
        if callable(inv):
            try:
                inv()
            except Exception:
                pass
        ext = path.suffix.lower()
        kwargs: dict = {}
        if ext in _MESH_CARRYING_EXTS:
            # Direct path (e.g. Upload NPZ → .glb, or legacy flat-dir pointing at .glb).
            # Force with_mesh=True so skinned-mesh rendering lights up end-to-end.
            kwargs["with_mesh"] = True
            # Route per-milestone progress pins into the loader.
            # real stage transitions (parsed, exporting GLB, done) show up on
            # the bar instead of us having to guess from the time curve alone.
            if progress_pin is not None:
                kwargs["progress_callback"] = progress_pin
        if ext == ".npz":
            m = npz.load_npz(path)
        elif ext in _MESH_CARRYING_EXTS:
            m = load_motion(path, **kwargs)
        else:
            m = load_motion(path)
        # Phase boundary: the adapter is done, now we build renderers / scene.
        # Pin the bar at 95% so the user sees the phase hand-off as "almost there"
        # rather than the synthetic curve slowly approaching it from below.
        _esc_pname = _html_escape(path.name)
        if progress_pin is not None:
            progress_pin(f"Building scene for <b>{_esc_pname}</b>", floor=95.0)
        else:
            progress.set_message(f"Building scene for <b>{_esc_pname}</b>")
        state["raw"] = m
        if library_entry is not None:
            _attach_library_folder_label(m, library_entry)
        else:
            pnorm = str(path.resolve()).replace("\\", "/")
            if "meshmimic/20260429_mocap/" in pnorm or "/20260429_mocap/" in pnorm:
                m.meta["library_folder_label"] = "20260429_mocap"
        original_axis = m.up_axis
        _rebuild_renderers(reset_playback=True)
        m_final = state.get("final")
        if isinstance(m_final, Motion):
            motion_label.value = m_final.name
            info_label.value = (
                f"{m_final.num_frames} · {m_final.framerate:.1f} · {m_final.num_bones}"
            )
            axis_label.value = f"{original_axis} → Z"
            _fit_camera_to_motion(server, m_final)
        _fire_motion_loaded_callbacks()

    # ``_work_lock`` + ``_run_async`` are now defined earlier in this same
    # function (just above the Robot tab construction) so they can be passed
    # into ``_build_robot_tab`` as kwargs — see comment there for why.
    progress = _ProgressReporter(
        server,
        progress_bar,
        progress_label_md,
        mirrors=_lib_progress_mirrors,
    )

    _motion_mem_cache: dict[tuple[str, bool], "Motion"] = {}

    def _load_entry(entry: LibraryEntry) -> None:
        """Lazily convert ``entry`` through the ephemeral cache, then display it.

        Authored-rig formats (.glb / .gltf) bypass the NPZ cache *for display*
        because the cache-trip strips ``meta["skinned_mesh"]`` (NPZ has no room for
        skinned geometry in the current schema).  The cache is still populated on Save
        button clicks via :meth:`cache.save_clip`, so "Save" still works — it just
        produces a skeleton-only NPZ.
        """
        src_ext = entry.source_path.suffix.lower()
        _wants_mesh = (
            src_ext in _MESH_CARRYING_EXTS
            or (entry.dataset in _SMPL_BAKED_DATASETS and bool(show_skinned_mesh.value))
        )
        _cache_key = (str(entry.source_path.resolve()), _wants_mesh)
        _cached = _motion_mem_cache.get(_cache_key)
        if _cached is not None:
            with _work_lock:
                inv = motion_sync.get("invalidate_robot_artifacts")
                if callable(inv):
                    try:
                        inv()
                    except Exception:
                        pass
                lib_state["current"] = entry
                lib_state["_last_loaded_source_path"] = str(
                    entry.source_path.resolve(),
                )
                state["raw"] = _cached
                _rebuild_renderers(reset_playback=True)
                m_final = state.get("final")
                if isinstance(m_final, Motion):
                    motion_label.value = m_final.name
                    info_label.value = (
                        f"{m_final.num_frames} · {m_final.framerate:.1f} · "
                        f"{m_final.num_bones}"
                    )
                    axis_label.value = f"{_cached.up_axis} → Z"
                    _fit_camera_to_motion(server, m_final)
                saved_label.value = (
                    "yes" if cache.is_saved(entry) else "no (ephemeral)"
                )
                status_md.content = _make_status(
                    entries, lib_state["filtered"], cache,
                )
                _fire_motion_loaded_callbacks()
            return

        if src_ext in _MESH_CARRYING_EXTS:
            with _work_lock:
                # ``expected_seconds`` drives the synthetic-% curve so the user
                # always sees a climbing number, even during bpy's 25 s black
                # box.  Rough order-of-magnitude averages per format — the
                # curve is asymptotic so a longer-than-expected run just keeps
                # crawling toward 95%, it never stalls or blows past 100.
                hint = ""
                expected_s = 4.0  # GLB/glTF default
                progress.start(
                    f"Parsing <b>{_html_escape(entry.folder_label)}</b>"
                    f" · {_html_escape(entry.stem)}{hint}",
                    indeterminate=False,
                    expected_seconds=expected_s,
                )
                try:
                    lib_state["current"] = entry
                    _load_path(
                        entry.source_path,
                        progress_pin=progress.pin_milestone,
                        library_entry=entry,
                    )
                    _motion_mem_cache[_cache_key] = state["raw"]
                    saved_label.value = "yes" if cache.is_saved(entry) else "no (ephemeral)"
                    status_md.content = _make_status(entries, lib_state["filtered"], cache)
                finally:
                    progress.done()
            return

        # SMPL-family datasets: when the mesh toggle is on, bypass the cache and run the
        # adapter with ``with_mesh=True`` so SMPL forward vertices land on the Motion's
        # ``meta["baked_mesh"]`` attachment. The cache path would otherwise strip those.
        # When the toggle is off we still go through the cache (faster — no forward-pass
        # baking — and size-efficient because the cache only stores joint positions).
        if (
            entry.dataset in _SMPL_BAKED_DATASETS
            and bool(show_skinned_mesh.value)
        ):
            with _work_lock:
                progress.start(
                    f"Baking SMPL mesh for "
                    f"<b>{_html_escape(entry.folder_label)}</b>"
                    f" · {_html_escape(entry.stem)}",
                    indeterminate=False,
                    # SMPL forward on a CPU with ~300 frames lands 3–6 s; GPU
                    # cuts it in half.  Synthetic curve hits ~60% at 5 s which
                    # is a reasonable "most of the way" feeling.
                    expected_seconds=5.0,
                )
                try:
                    from hhtools.io.datasets import registered_datasets
                    adapter_cls = registered_datasets().get(entry.dataset)
                    if adapter_cls is None:
                        raise KeyError(
                            f"No adapter registered for dataset '{entry.dataset}'"
                        )
                    adapter = adapter_cls(entry.source_path.parent)
                    try:
                        motion = adapter.load_motion(
                            entry.adapter_sequence_id, with_mesh=True,
                        )
                    except FileNotFoundError as exc:
                        # Body model weights absent → fall back to cache path which also
                        # would have failed the same way, but at least the error message
                        # is already user-friendly ("No SMPL-H (neutral) weight file
                        # found. Run `hhtools bodymodel setup`...").
                        raise FileNotFoundError(
                            f"Could not bake SMPL mesh for {entry.stem}: {exc}\n"
                            "Either install the body-model weights (see "
                            "`hhtools bodymodel setup`) or untick the 'Skinned mesh' "
                            "checkbox to fall back to skeleton-only rendering."
                        ) from exc
                    state["raw"] = motion
                    _motion_mem_cache[_cache_key] = motion
                    lib_state["current"] = entry
                    _rebuild_renderers(reset_playback=True)
                    m_final = state.get("final")
                    if isinstance(m_final, Motion):
                        motion_label.value = m_final.name
                        info_label.value = (
                            f"{m_final.num_frames} · {m_final.framerate:.1f} · "
                            f"{m_final.num_bones}"
                        )
                        axis_label.value = f"{motion.up_axis} → Z"
                        _fit_camera_to_motion(server, m_final)
                    saved_label.value = "yes" if cache.is_saved(entry) else "no (ephemeral)"
                    status_md.content = _make_status(entries, lib_state["filtered"], cache)
                    _fire_motion_loaded_callbacks()
                finally:
                    progress.done()
            return

        # PARC / unified-npz bundles: always load from the source tree so sibling
        # terrain meshes resolve; never copy through the session cache for display.
        if entry.dataset in _DIRECT_SOURCE_NPZ_DATASETS and src_ext == ".npz":
            with _work_lock:
                progress.start(
                    f"Loading <b>{_html_escape(entry.folder_label)}</b>"
                    f" · {_html_escape(entry.stem)}",
                    indeterminate=False,
                    expected_seconds=1.5,
                )
                try:
                    lib_state["current"] = entry
                    _load_path(entry.source_path, library_entry=entry)
                    _motion_mem_cache[_cache_key] = state["raw"]
                    saved_label.value = "yes" if cache.is_saved(entry) else "no (ephemeral)"
                    status_md.content = _make_status(
                        entries, lib_state["filtered"], cache,
                    )
                finally:
                    progress.done()
            return

        cache_path = cache_dir_path / entry.cache_name
        needs_convert = (
            not cache_path.exists()
            or cache_path.stat().st_mtime < entry.source_path.stat().st_mtime
        )
        with _work_lock:
            if needs_convert:
                progress.start(
                    f"Converting <b>{_html_escape(entry.folder_label)}</b>"
                    f" · {_html_escape(entry.stem)}",
                    indeterminate=False,
                    # BVH / pkl / npy / etc. parse paths typically land in the
                    # 1-8 s range; 4 s is the median we've measured.
                    expected_seconds=4.0,
                )
            try:
                npz_path = cache.get(entry)
            except Exception:
                progress.done()
                raise
            if needs_convert:
                progress.set_message(
                    f"Loading <b>{_html_escape(entry.folder_label)}</b>"
                    f" · {_html_escape(entry.stem)}",
                    value=90.0,
                )
            lib_state["current"] = entry
            _load_path(npz_path, library_entry=entry)
            _motion_mem_cache[_cache_key] = state["raw"]
            saved_label.value = "yes" if cache.is_saved(entry) else "no (ephemeral)"
            status_md.content = _make_status(entries, lib_state["filtered"], cache)
            progress.done()

    # -------- Library callbacks ----------------------------------------------

    def _refresh_clip_dropdown() -> None:
        """Recompute dropdown options from the current search+folder filters."""
        query = search_box.value
        folder = folder_picker.value
        filtered = filter_entries(entries, query, folder=folder)
        # Cap for UI performance — users are expected to keep typing if results > cap.
        shown = filtered[:_MAX_LIBRARY_OPTIONS]
        labels = [_unique_label(e) for e in shown] or ["(no matches)"]
        lib_state["filtered"] = shown
        # When options is empty / identical, Viser may raise; guard with a sentinel.
        clip_picker.options = tuple(labels)
        # Mirror the filter change to every motion-sync subscriber so the
        # Robot tab's motion picker expands / contracts in lock-step with the
        # Library's search+folder filter.
        _notify_motion_changed(clip_picker.value)
        # Keep the current selection if it still matches, otherwise snap to the top.
        current = lib_state.get("current")
        target_label: str | None = None
        if isinstance(current, LibraryEntry):
            for label, entry in zip(labels, shown, strict=False):
                if entry.source_path == current.source_path:
                    target_label = label
                    break
        if target_label is None and labels:
            target_label = labels[0]
        if target_label is not None:
            _clip_pick_guard["suppress"] = True
            try:
                clip_picker.value = target_label
            finally:
                _clip_pick_guard["suppress"] = False
            # Always load explicitly — viser dropdown ``on_update`` is not
            # guaranteed when the value is set programmatically (e.g. Robot-tab
            # folder/search sync while the Motion-tab guard is active).
            _on_clip_pick(None)
        status_md.content = _make_status(entries, filtered, cache)
        _notify_library_refreshed(
            tuple(clip_picker.options), str(clip_picker.value),
        )

    def _notify_library_refreshed(
        options: tuple[str, ...], selected: str,
    ) -> None:
        for cb in list(motion_sync.get("library_refresh_listeners", [])):
            try:
                cb(options, selected)
            except Exception:
                pass

    def _load_by_label(label: str) -> None:
        """Robot-tab hook: load the library entry matching ``label``.

        Drives the Motion-tab dropdown by setting ``clip_picker.value``, so the
        on_update callback fires exactly the same code path as a user click —
        including the background-worker load + the motion_sync notification.
        No-op when the label isn't in the current filter (e.g. the user
        cleared the search box before the Robot tab could react).
        """
        labels = list(clip_picker.options)
        if label in labels:
            _clip_pick_guard["suppress"] = True
            try:
                clip_picker.value = label
            finally:
                _clip_pick_guard["suppress"] = False
            _on_clip_pick(None)

    motion_sync["load_by_label"] = _load_by_label
    # ``current_label`` / ``all_labels`` were seeded earlier (before the Robot
    # tab was built) so its initial option list isn't empty — don't re-bind
    # them here.

    @search_box.on_update
    def _on_search(_):  # type: ignore[no-untyped-def]
        if _search_sync_guard["active"]:
            return
        _search_sync_guard["active"] = True
        try:
            _refresh_clip_dropdown()
            _notify_search_changed(search_box.value)
        finally:
            _search_sync_guard["active"] = False

    @folder_picker.on_update
    def _on_folder(_):  # type: ignore[no-untyped-def]
        if _folder_sync_guard["active"]:
            return
        _folder_sync_guard["active"] = True
        try:
            _refresh_clip_dropdown()
            _notify_folder_changed(folder_picker.value)
        finally:
            _folder_sync_guard["active"] = False

    @clip_picker.on_update
    def _on_clip_pick(_):  # type: ignore[no-untyped-def]
        if _clip_pick_guard["suppress"]:
            return
        shown = lib_state.get("filtered") or []
        if not isinstance(shown, list) or not shown:
            return
        label = clip_picker.value
        labels = [_unique_label(e) for e in shown]
        if label not in labels:
            return
        idx = labels.index(label)
        entry = shown[idx]
        # Mirror the selection to every motion-sync subscriber (Robot tab's
        # motion picker).  We do this *before* starting the worker so the UI
        # reflects the user's intent immediately — the actual load happens
        # in the background.
        _notify_motion_changed(label)

        # Run the load on a worker thread so the viser event loop stays free to
        # ship progress-bar updates while e.g. bpy's FBX subprocess is blocking.
        # See _run_async for why this matters.
        def _worker() -> None:
            try:
                _load_entry(entry)
            except Exception as exc:  # pragma: no cover -- surfaced at runtime
                # Surface *full* error via a toast so users aren't blind to what went
                # wrong.  The Clip Info panel still shows the truncated one-liner for
                # persistence, but the toast carries the actual multi-line message
                # (install instructions for missing FBX backends, etc.).
                motion_label.value = f"(load failed) {type(exc).__name__}"
                info_label.value = str(exc).splitlines()[0][:100]
                _notify_all(
                    server,
                    f"Could not load {entry.stem}",
                    str(exc),
                    color="red",
                )

        _run_async(f"hhtools-load-{entry.stem}", _worker)

    @save_clip_btn.on_click
    def _on_save_clip(_):  # type: ignore[no-untyped-def]
        current = lib_state.get("current")
        if not isinstance(current, LibraryEntry):
            _notify_all(
                server,
                "Nothing to save",
                "Pick a clip in the library first, then try again.",
                color="yellow",
            )
            return
        cache_path = cache_dir_path / current.cache_name
        needs_convert = not cache_path.exists()

        # Worker thread: cache.save_clip may trigger a full adapter conversion
        # for cache misses (seconds-minutes for FBX).  Same rationale as the
        # clip picker — keep the viser event loop free so the ticker updates.
        def _worker() -> None:
            try:
                with _work_lock:
                    if needs_convert:
                        # Save can trigger a full adapter conversion — budget
                        # generously for FBX; anything faster just means the
                        # synthetic curve crawls instead of saturating.
                        est_ext = current.source_path.suffix.lower()
                        est_s = 5.0
                        progress.start(
                            f"Converting & saving "
                            f"<b>{_html_escape(current.folder_label)}</b>"
                            f" · {_html_escape(current.stem)}",
                            indeterminate=False,
                            expected_seconds=est_s,
                        )
                    dst = cache.save_clip(current)
                    progress.done()
            except Exception as exc:  # pragma: no cover
                progress.done()
                msg = f"{type(exc).__name__}: {exc}"
                saved_label.value = f"save failed: {type(exc).__name__}"
                _notify_all(server, "Save failed", msg, color="red")
                return
            rel = _relative_to_repo(dst)
            saved_label.value = f"yes → {rel}"
            status_md.content = _make_status(entries, lib_state["filtered"], cache)
            save_log_md.content = _format_save_log(
                heading="Saved 1 clip",
                lines=[f"`{rel}`"],
            )
            _notify_all(
                server,
                "Saved 1 clip",
                f"{current.folder_label} · {current.stem}\n→ {rel}",
                color="green",
            )

        _run_async(f"hhtools-save-{current.stem}", _worker)

    @save_folder_btn.on_click
    def _on_save_folder(_):  # type: ignore[no-untyped-def]
        folder = folder_picker.value
        bucket = [e for e in entries if folder in (_FOLDER_ALL, e.folder_label)]
        if not bucket:
            _notify_all(
                server,
                "Nothing to save",
                "The current folder filter has no clips to persist.",
                color="yellow",
            )
            return
        label = "all folders" if folder == _FOLDER_ALL else folder
        dst_root = save_dir_path / ("" if folder == _FOLDER_ALL else folder)
        # Determinate progress: one step per entry. cache.save_clip may internally
        # trigger an adapter conversion for cache misses, which is the slow part —
        # the per-step update at least tells users *which* clip is being processed.
        total = len(bucket)

        def _worker() -> None:
            saved: list[Path] = []
            try:
                with _work_lock:
                    progress.start(
                        f"Saving <b>{total}</b> clip(s) from "
                        f"<b>{_html_escape(label)}</b>",
                        indeterminate=False,
                        total=total,
                    )
                    for i, entry in enumerate(bucket, start=1):
                        _elabel = _html_escape(entry.folder_label)
                        _estem = _html_escape(entry.stem)
                        progress.set_message(
                            f"[{i}/{total}] <b>{_elabel}</b> · {_estem}",
                            value=(i - 1) / total * 100.0,
                        )
                        dst = cache.save_clip(entry)
                        saved.append(dst)
                        progress.set_message(
                            f"[{i}/{total}] <b>{_elabel}</b> · {_estem}",
                            value=i / total * 100.0,
                        )
                    progress.done()
            except Exception as exc:  # pragma: no cover
                progress.done()
                msg = f"{type(exc).__name__}: {exc}"
                saved_label.value = f"save folder failed: {type(exc).__name__}"
                _notify_all(server, "Save folder failed", msg, color="red")
                return
            count = len(saved)
            rel_root = _relative_to_repo(dst_root)
            saved_label.value = f"saved {count} clips → {rel_root}"
            status_md.content = _make_status(entries, lib_state["filtered"], cache)
            preview = [f"`{_relative_to_repo(p)}`" for p in saved[:5]]
            if count > 5:
                preview.append(f"… and {count - 5} more")
            save_log_md.content = _format_save_log(
                heading=f"Saved {count} clip(s) from "
                        f"<b>{_html_escape(label)}</b>",
                lines=preview,
            )
            _notify_all(
                server,
                f"Saved {count} clip(s)",
                f"{label} → {rel_root}",
                color="green",
            )

        _run_async(f"hhtools-save-folder-{label}", _worker)

    # ---- Legacy flat-dir dropdown (only when explicitly requested) -----------
    if motion_dir is not None:
        motion_root = Path(motion_dir)
        legacy_candidates = sorted(p for p in motion_root.glob("*.npz") if p.is_file())
        legacy_labels = [p.name for p in legacy_candidates]
        if legacy_labels:
            legacy_dropdown = server.gui.add_dropdown(
                "Motion Library (legacy)",
                options=tuple(legacy_labels),
                initial_value=legacy_labels[0],
            )

            @legacy_dropdown.on_update
            def _on_legacy(_):  # type: ignore[no-untyped-def]
                idx = legacy_labels.index(legacy_dropdown.value)
                lib_state["current"] = None
                _load_path(legacy_candidates[idx])

            if motion is None:
                lib_state["current"] = None
                _load_path(legacy_candidates[0])

    # ---- Display-option callbacks --------------------------------------------
    @show_skeleton.on_update
    def _toggle_skel(_):  # type: ignore[no-untyped-def]
        skeleton.set_visible(bool(show_skeleton.value))

    @show_capsules.on_update
    def _toggle_caps(_):  # type: ignore[no-untyped-def]
        capsules.set_visible(bool(show_capsules.value))

    @show_skinned_mesh.on_update
    def _toggle_skinned(_):  # type: ignore[no-untyped-def]
        # ``has_mesh`` short-circuit: toggling the checkbox on a motion without a
        # skinned mesh does nothing visible, rather than flashing the last rendered
        # mesh's residual handle (which would have been cleared anyway, but this
        # keeps the intent explicit).
        skinned_mesh_renderer.set_visible(
            bool(show_skinned_mesh.value) and skinned_mesh_renderer.has_mesh()
        )

    _bone_label_handles: list[object] = []

    def _clear_bone_labels() -> None:
        for lh in _bone_label_handles:
            try:
                lh.remove()  # type: ignore[union-attr]
            except Exception:
                pass
        _bone_label_handles.clear()

    def _create_bone_labels(motion_obj: Motion, frame: int = 0) -> None:
        _clear_bone_labels()
        if not bool(show_bone_names.value):
            return
        frame = int(np.clip(frame, 0, motion_obj.num_frames - 1))
        positions = motion_obj.positions[frame]
        bone_names = motion_obj.hierarchy.bone_names
        label_offset = np.array([0.0, 0.0, 0.03], dtype=np.float32)
        for i, name in enumerate(bone_names):
            pos = positions[i] + label_offset
            try:
                lh = server.scene.add_label(
                    f"/skeleton/bone_label_{i}",
                    text=name,
                    position=tuple(float(x) for x in pos),
                    visible=True,
                )
                _bone_label_handles.append(lh)
            except Exception:
                pass

    def _update_bone_label_positions(motion_obj: Motion, frame: int) -> None:
        if not _bone_label_handles:
            return
        frame = int(np.clip(frame, 0, motion_obj.num_frames - 1))
        positions = motion_obj.positions[frame]
        label_offset = np.array([0.0, 0.0, 0.03], dtype=np.float32)
        for i, lh in enumerate(_bone_label_handles):
            if i >= positions.shape[0]:
                break
            pos = positions[i] + label_offset
            try:
                lh.position = tuple(float(x) for x in pos)  # type: ignore[union-attr]
            except Exception:
                pass

    @show_bone_names.on_update
    def _toggle_bone_names(_):  # type: ignore[no-untyped-def]
        m_cur = state.get("current_motion")
        if isinstance(m_cur, Motion) and bool(show_bone_names.value):
            current_frame = state.get("frame", 0)
            _create_bone_labels(m_cur, current_frame)
        else:
            _clear_bone_labels()

    @show_objects.on_update
    def _toggle_objects(_):  # type: ignore[no-untyped-def]
        objects_renderer.set_visible(bool(show_objects.value))
        terrain_renderer.set_visible(bool(show_objects.value))

    @snap_ground.on_update
    def _toggle_snap(_):  # type: ignore[no-untyped-def]
        _rebuild_renderers()
        m_final = state.get("final")
        if isinstance(m_final, Motion):
            _fit_camera_to_motion(server, m_final)

    @center_xy.on_update
    def _toggle_center_xy(_):  # type: ignore[no-untyped-def]
        _rebuild_renderers()
        m_final = state.get("final")
        if isinstance(m_final, Motion):
            _fit_camera_to_motion(server, m_final)

    @hide_virtual_root.on_update
    def _toggle_root(_):  # type: ignore[no-untyped-def]
        _rebuild_renderers()

    @show_world_axes.on_update
    def _toggle_world_axes(_):  # type: ignore[no-untyped-def]
        server.scene.world_axes.visible = bool(show_world_axes.value)

    @radius_scale.on_update
    def _on_radius_scale(_):  # type: ignore[no-untyped-def]
        _rebuild_renderers()

    # ---- Initial load --------------------------------------------------------
    # When the first clip needs conversion (SMPL forward pass, BVH parse, ...) it
    # can block for several seconds. We run the auto-load on a background thread so
    # the Viser UI is interactive immediately; the progress bar in the Library
    # folder communicates the wait. ``motion=...`` (explicit preload) also uses the
    # background thread for consistency.
    def _initial_load() -> None:
        try:
            if motion is not None:
                _load_path(Path(motion))
            elif entries and motion_dir is None and _want_autoload_first:
                _load_entry(entries[0])
        except BaseException as exc:  # pragma: no cover - surfaced at runtime
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            import traceback

            traceback.print_exc()
            motion_label.value = f"(load failed) {type(exc).__name__}"
            info_label.value = str(exc)[:80]
            progress.done()

    threading.Thread(target=_initial_load, daemon=True, name="hhtools-initial-load").start()

    if share:
        server.request_share_url()
    # ``port`` is the *requested* port; ``viser.ViserServer`` may bind
    # to a higher one if the requested port is already in use, so always
    # read back the actual port via ``get_port()`` to avoid misleading
    # users about which URL to open in the browser.
    actual_port = server.get_port() if hasattr(server, "get_port") else port
    print(f"[hhtools] viewer ready at http://{host}:{actual_port}")
    if actual_port != port:
        print(
            f"[hhtools] note: requested port {port} was unavailable; "
            f"viser auto-selected {actual_port}",
        )
    print(f"[hhtools] source={source_root_path}  cache={cache_dir_path}  save={save_dir_path}")
    if not keep_cache:
        owns = "auto-temp" if cache.owns_cache_dir else "shared"
        print(f"[hhtools] cache is ephemeral ({owns}); shutdown will wipe it regardless of saves")
    if (
        motion is None
        and motion_dir is None
        and entries
        and not _want_autoload_first
    ):
        print(
            "[hhtools] First library clip is not auto-loaded (safer default). "
            "Pick a clip in the UI, or run: hhtools ui --autoload"
        )

    try:
        while True:
            frame = panel.tick()
            m_cur = state.get("final")
            if isinstance(m_cur, Motion):
                try:
                    if show_skeleton.value:
                        skeleton.set_frame(frame)
                    if show_capsules.value:
                        capsules.set_frame(frame)
                    if show_skinned_mesh.value and skinned_mesh_renderer.has_mesh():
                        skinned_mesh_renderer.set_frame(frame)
                    if show_objects.value:
                        objects_renderer.set_frame(frame)
                    if _bone_label_handles and show_bone_names.value:
                        _update_bone_label_positions(m_cur, frame)
                except Exception:
                    pass
            # ---- Robot tab animation ----
            # Drive the robot + scaled-skeleton renderers from the same
            # playback cursor the Motion tab uses, so scrubbing / playing
            # moves every view in lock-step.  Travel direction matches the
            # source motion: the scaler's ``source_body_quat`` aligns IK to
            # +X robot frame, then the pipeline / preview counter-rotate back
            # to the source heading (same convention as ``Motion`` skeleton
            # lines, which draw raw ``motion.positions``).
            # Frames past the retargeted
            # clip's horizon are clamped to the last solved frame (the robot
            # freezes at the end of its IK window rather than jumping).
            try:
                # During calibration the robot pose is driven only by the
                # joint sliders — do not overwrite from retarget playback.
                if not bool(robot_state.get("calibration_active")):
                    retargeted = robot_state.get("retargeted")
                    animator = robot_state.get("animator")
                    if retargeted is not None and animator is not None and retargeted.num_frames > 0:
                        rf = int(np.clip(frame, 0, retargeted.num_frames - 1))
                        cur_model = state.get("current_model")
                        dof_drive = (
                            cur_model.dof_names()
                            if isinstance(cur_model, URDFRobotModel)
                            else retargeted.dof_names
                        )
                        animator.set_frame_joint_q(
                            retargeted.joint_q[rf],
                            dof_drive,
                        )
                    scaled_pv = robot_state.get("scaled_preview")
                    scaled_rd = robot_state.get("scaled_renderer")
                    if scaled_pv is not None and scaled_rd is not None and scaled_pv.num_frames > 0:
                        sf = int(np.clip(frame, 0, scaled_pv.num_frames - 1))
                        scaled_rd.set_frame(sf)
                    robj_rd = robot_state.get("robot_objects_renderer")
                    if robj_rd is not None:
                        robj_rd.set_frame(frame)
            except Exception:
                # One bad frame shouldn't kill the render loop; surface via
                # log-once later if it turns out to be a recurring issue.
                pass
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        print("[hhtools] viewer shutting down")
    finally:
        if not keep_cache:
            cache.cleanup()


def _unique_label(entry: LibraryEntry) -> str:
    """Build a dropdown-friendly label. Folder prefix disambiguates identical stems."""
    return f"{entry.folder_label} · {entry.stem}"


def _relative_to_repo(p: Path) -> str:
    """Shorten an absolute path to a repo-relative one when possible.

    Falls back to the string form if the path can't be made relative to ``cwd``.
    """
    try:
        return str(Path(p).resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(p)


def _format_save_log(heading: str, lines: list[str]) -> str:
    """Build the markdown shown under the Save buttons after a save operation."""
    body = "<br/>".join(lines) if lines else ""
    return (
        f"<div style='padding:4px 0'>"
        f"<b style='color:{PALETTE.ui_info}'>{heading}</b>"
        + (f"<br/><span style='opacity:0.75;font-size:0.9em'>{body}</span>" if body else "")
        + "</div>"
    )


class _ProgressReporter:
    """Live-updating progress bar + markdown status line inside the Library folder.

    Three operating modes:

    - **indeterminate pulse** (``start(indeterminate=True)``) — animated full bar
      with no %.  Used only when we can't estimate duration *and* have no
      milestones (kept for legacy callers; the Library never uses this now).

    - **synthetic asymptotic %** (``start(indeterminate=False,
      expected_seconds=N)``) — the ticker thread walks the bar along
      ``p(t) = ceiling * (1 - exp(-t/N))`` and re-renders every 400 ms.  This
      gives the user a *concrete* climbing percentage even when the underlying
      work (a bpy subprocess, a big SMPL forward pass) exposes no real progress
      signal.  The curve is intentionally asymptotic: it never reaches 100 on
      its own so a slow bpy run doesn't hit 100% before the data is actually
      ready.  :meth:`pin_milestone` can jump the floor forward when something
      reportable happens (e.g. "FBX parse done, starting GLB export").

    - **fully driven** (``start(indeterminate=False, total=N)``) — caller
      advances via :meth:`set_message(value=...)`.  Used by the Save-whole-folder
      path, where we know how many clips are done.

    :meth:`done` accepts ``success=True`` + ``last_message`` to paint **100%** once
    before hiding (pairs with ``ceiling`` near 100 on retarget so the bar does not
    sit at an asymptotic cap after work has finished).

    Every mutation calls ``server.flush()`` so the browser sees the update even
    when the surrounding Python code is about to re-enter a long-running adapter
    call without returning to the event loop.  Combined with ``_run_async`` at
    every callback entry this keeps the bar moving during a blocking bpy
    subprocess.
    """

    def __init__(self, server, bar, label_md, *, mirrors=None) -> None:  # type: ignore[no-untyped-def]
        self._server = server
        self._bar = bar
        self._label = label_md
        self._mirrors: list = list(mirrors or [])
        self._active = False
        self._total = 0
        self._title = ""
        self._started_at = 0.0
        self._expected_seconds: float | None = None
        # Asymptotic curve caps below 100 so we can still show progress is
        # happening without lying about completion.  95% has just enough room
        # to still "jump to 100" visually when done() fires.
        self._ceiling: float = 95.0
        # Milestones let the FBX backend surface real stages (FBX parsed →
        # starting GLB export → finished GLB).  When a milestone pins the
        # floor, we also reset the curve's anchor time so the asymptotic
        # progression restarts from ``floor`` instead of sitting stuck there
        # until the original curve catches up.  Without this, pinning floor=60
        # at t=1.6s with ``expected_seconds=30`` meant the bar was frozen at
        # 60% for the whole ~25 s of GLB export (``curve(26s) ≈ 55`` < 60).
        self._floor: float = 0.0
        self._floor_at: float = 0.0
        self._lock = threading.Lock()
        self._tick_thread: threading.Thread | None = None
        self._tick_stop: threading.Event | None = None

    def start(
        self,
        title: str,
        *,
        indeterminate: bool = True,
        total: int = 0,
        expected_seconds: float | None = None,
        ceiling: float = 95.0,
    ) -> None:
        with self._lock:
            self._stop_ticker_locked()
            self._active = True
            self._total = max(total, 0)
            self._title = title
            self._started_at = time.monotonic()
            self._expected_seconds = (
                float(expected_seconds) if expected_seconds and expected_seconds > 0 else None
            )
            self._ceiling = max(0.0, min(99.0, float(ceiling)))
            self._floor = 0.0
            self._floor_at = self._started_at
            # When caller supplied an estimate we drive a real percentage even
            # though ``indeterminate`` may have been False already.  The GUI
            # bar.animated flag doubles as "is this a pulse?" so for the
            # estimate path we want the bar filling normally, not pulsing.
            use_estimate = self._expected_seconds is not None
            bar_animated = bool(indeterminate) and not use_estimate
            initial_value = 0.0 if (not indeterminate or use_estimate) else 100.0
            try:
                self._bar.animated = bar_animated
                self._bar.value = initial_value
                self._bar.visible = True
                self._label.content = _progress_md(
                    title, initial_value, bar_animated, elapsed=0.0,
                )
                for mbar, mmd in self._mirrors:
                    try:
                        mbar.animated = bar_animated
                        mbar.value = initial_value
                        mbar.visible = True
                        mmd.content = _progress_md(
                            title, initial_value, bar_animated, elapsed=0.0,
                        )
                    except Exception:
                        pass
                self._server.flush()
            except Exception:
                pass
            # Tick when we have anything to animate on our side: pulse label
            # elapsed (pulse mode) or compute synthetic % (estimate mode).
            if bar_animated or use_estimate:
                self._start_ticker_locked()

    def set_message(self, message: str, *, value: float | None = None) -> None:
        with self._lock:
            if not self._active:
                return
            self._title = message
            try:
                if value is not None and not self._bar.animated:
                    clamped = float(max(0.0, min(100.0, value)))
                    self._bar.value = clamped
                elapsed = time.monotonic() - self._started_at
                self._label.content = _progress_md(
                    message,
                    float(self._bar.value) if not self._bar.animated else 0.0,
                    bool(self._bar.animated),
                    elapsed=elapsed,
                )
                pv = float(self._bar.value) if not self._bar.animated else 0.0
                anim = bool(self._bar.animated)
                for mbar, mmd in self._mirrors:
                    try:
                        if value is not None and not mbar.animated:
                            mbar.value = clamped
                        mmd.content = _progress_md(message, pv, anim, elapsed=elapsed)
                    except Exception:
                        pass
                self._server.flush()
            except Exception:
                pass

    def pin_milestone(self, message: str | None, *, floor: float) -> None:
        """Jump the synthetic-% floor forward to ``floor`` (0..ceiling).

        Use this from loaders that know a real stage boundary (e.g. "bpy
        finished parsing FBX, now exporting GLB" → floor=60).  The ticker
        will then render ``max(curve(t), floor)`` so the bar moves forward
        even if the asymptotic curve hadn't caught up yet.  Passing a
        message also updates the label so users see *what* advanced.
        """
        with self._lock:
            if not self._active:
                return
            new_floor = max(self._floor, min(self._ceiling, float(floor)))
            if new_floor > self._floor:
                # Reset curve anchor so the asymptote climbs from the new floor
                # rather than staying pegged there.  See ``__init__`` comment.
                self._floor = new_floor
                self._floor_at = time.monotonic()
            if message is not None:
                self._title = message
            try:
                value = self._compute_value_locked()
                self._bar.value = value
                elapsed = time.monotonic() - self._started_at
                self._label.content = _progress_md(
                    self._title, value, bool(self._bar.animated), elapsed=elapsed,
                )
                for mbar, mmd in self._mirrors:
                    try:
                        mbar.value = value
                        mmd.content = _progress_md(
                            self._title, value, bool(self._bar.animated),
                            elapsed=elapsed,
                        )
                    except Exception:
                        pass
                self._server.flush()
            except Exception:
                pass

    def done(self, *, success: bool = False, last_message: str | None = None) -> None:
        """Hide the bar.  When ``success`` and ``last_message`` are set, paint 100% once
        before hiding so the UI does not sit pegged at the asymptotic ceiling (e.g. 95%)
        while the worker has already finished — mirrors tqdm's completion tick.
        """
        with self._lock:
            self._active = False
            self._stop_ticker_locked()
            try:
                if success and last_message:
                    self._bar.animated = False
                    self._bar.visible = True
                    self._bar.value = 100.0
                    elapsed = time.monotonic() - self._started_at
                    self._label.content = _progress_md(
                        last_message, 100.0, False, elapsed=elapsed,
                    )
                    for mbar, mmd in self._mirrors:
                        try:
                            mbar.animated = False
                            mbar.visible = True
                            mbar.value = 100.0
                            mmd.content = _progress_md(
                                last_message, 100.0, False, elapsed=elapsed,
                            )
                        except Exception:
                            pass
                    self._server.flush()
                    self._server.flush()
                self._bar.visible = False
                self._bar.animated = False
                self._bar.value = 0.0
                self._label.content = ""
                for mbar, mmd in self._mirrors:
                    try:
                        mbar.visible = False
                        mbar.animated = False
                        mbar.value = 0.0
                        mmd.content = ""
                    except Exception:
                        pass
                self._server.flush()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_value_locked(self) -> float:
        """Synthetic % at ``now``; caller must hold the lock.

        The curve is anchored at the most recent milestone (pin) and climbs
        asymptotically toward ``ceiling``::

            value(t) = floor + (ceiling - floor) * (1 - exp(-(t - floor_at)/expected))

        With no pin yet, ``floor=0`` and ``floor_at=started_at``, so we recover
        the classic ``ceiling * (1 - exp(-t/expected))`` curve.  After each
        milestone, the curve rescales between ``[floor, ceiling]`` over a fresh
        ``expected_seconds`` window — so pinning 60% then waiting 30 s lands
        at ~82%, rather than being stuck at 60% forever.
        """
        if self._expected_seconds is None:
            return self._floor
        elapsed_since_pin = max(0.0, time.monotonic() - self._floor_at)
        span = max(0.0, self._ceiling - self._floor)
        curve = self._floor + span * (1.0 - math.exp(-elapsed_since_pin / self._expected_seconds))
        return float(max(self._floor, min(self._ceiling, curve)))

    def _start_ticker_locked(self) -> None:
        """Caller must hold ``self._lock``."""
        stop_event = threading.Event()
        self._tick_stop = stop_event

        def _tick() -> None:
            # ~400 ms cadence: fast enough to feel live, slow enough to keep the
            # WebSocket from flooding under a hot-reload / busy scene.
            while not stop_event.wait(0.4):
                with self._lock:
                    if not self._active or self._tick_stop is not stop_event:
                        return
                    try:
                        elapsed = time.monotonic() - self._started_at
                        if self._expected_seconds is not None:
                            # Advance synthetic %.
                            self._bar.value = self._compute_value_locked()
                        pv = (
                            float(self._bar.value) if not self._bar.animated else 0.0
                        )
                        self._label.content = _progress_md(
                            self._title,
                            pv,
                            bool(self._bar.animated),
                            elapsed=elapsed,
                        )
                        for mbar, mmd in self._mirrors:
                            try:
                                if self._expected_seconds is not None:
                                    mbar.value = self._compute_value_locked()
                                mmd.content = _progress_md(
                                    self._title,
                                    float(mbar.value) if not mbar.animated else 0.0,
                                    bool(mbar.animated),
                                    elapsed=elapsed,
                                )
                            except Exception:
                                pass
                        self._server.flush()
                    except Exception:
                        return

        thread = threading.Thread(target=_tick, name="hhtools-progress-tick", daemon=True)
        self._tick_thread = thread
        thread.start()

    def _stop_ticker_locked(self) -> None:
        """Caller must hold ``self._lock``."""
        if self._tick_stop is not None:
            self._tick_stop.set()
            self._tick_stop = None
        self._tick_thread = None


def _progress_md(
    title: str,
    value: float,
    indeterminate: bool,
    *,
    elapsed: float = 0.0,
) -> str:
    pct_text = "" if indeterminate else f" &nbsp;·&nbsp; <b>{value:.0f}%</b>"
    spinner = "<span style='opacity:0.8'>⟳</span> " if indeterminate else ""
    if elapsed >= 0.5:
        elapsed_text = (
            f" &nbsp;·&nbsp; <span style='opacity:0.65'>{elapsed:.1f}s</span>"
        )
    else:
        elapsed_text = ""
    return (
        f"<div style='padding:2px 0'>"
        f"{spinner}<span style='font-size:0.92em'>{title}</span>{pct_text}{elapsed_text}"
        f"</div>"
    )


def _build_robot_tab(  # type: ignore[no-untyped-def]
    server,
    *,
    get_current_motion=None,  # Callable[[], Motion | None]
    get_current_entry_id=None,  # Callable[[], str | None]
    get_filtered_entries=None,  # Callable[[], list[LibraryEntry]]
    motion_cache=None,  # EphemeralCache | None — needed for batch loading
    get_panel=None,  # Callable[[], PlaybackPanel | None]
    robot_state=None,  # dict[str, object] shared with main render loop
    motion_sync=None,  # dict[str, object] — Motion-tab load/label helpers
    source_root_path: Path | None = None,  # ``run_viewer``'s ``source_root_path``
    save_dir_path: Path | None = None,     # ``run_viewer``'s ``save_dir_path``
    run_async=None,                         # ``run_viewer``'s ``_run_async``
    work_lock=None,                         # ``run_viewer``'s ``_work_lock``
    apply_motion_pipeline=None,             # ``run_viewer``'s ``_apply_settings_to_motion``
) -> None:
    """Populate the Robot tab: preset picker → animated T-pose → retarget pipeline.

    Responsibilities delivered here:

    1. **Preset picker + stats** – dropdown over every discovered robot
       preset.  Presets without a URDF on disk are shown with a ``⚠`` badge.
    2. **Load / clear** – materialises the URDF into an animated mesh via
       :class:`~hhtools.viewer.renderers.RobotAnimator`, which also handles
       ground alignment (lifts the whole robot so the lowest T-pose vertex
       sits on ``z=0``).  The animator is published to ``robot_state`` so
       the main render loop can drive its pose every tick once a retarget
       result is available.
    3. **Retarget** – single- or batch-mode run of
       :class:`~hhtools.retarget.newton_basic.NewtonBasicPipeline` (Newton IK)
       or :class:`~hhtools.retarget.interaction_mesh.InteractionMeshPipeline`
       (Laplacian MPC), selectable in the UI with Auto routing for OMOMO /
       meshmimic clips; per-frame progress, then auto-play of the retargeted
       motion in the viewer.
    4. **Scaled human skeleton preview** – renders the scaler's pre-IK
       canonical-joint targets (mirrors the soma-retargeter
       ``<…>_scaler_config.json`` sanity dump) so the user can eyeball
       what IK is being asked to chase *before* paying for a full solve.
    5. **Export schema CSV** – dump the column header the retarget CSV will
       use; lets users pin ``dof_order`` in ``robot.yaml`` with confidence.

    Per-tab state lives in the closure-local ``state`` dict.  Cross-tab
    plumbing (``robot_state``, ``get_panel``, ``motion_cache``) is passed
    in so we don't fight viser's one-tab-per-connection model.
    """
    import tempfile

    presets = list_robot_presets()

    if robot_state is None:  # Minimal fallback so standalone unit tests still work.
        robot_state = {}
    robot_state.setdefault("calibration_active", False)
    robot_state.setdefault("_retarget_lock", threading.Lock())
    robot_state.setdefault("_prewarm_thread", None)

    def _await_robot_prewarm(*, timeout: float = 120.0) -> None:
        """Block until the background Newton prewarm thread finishes (if any)."""
        t = robot_state.get("_prewarm_thread")
        if isinstance(t, threading.Thread) and t.is_alive():
            t.join(timeout=max(0.0, float(timeout)))

    state: dict[str, object] = {
        "current_model": None,     # URDFRobotModel once Load succeeded
        "animator": None,          # RobotAnimator owning /robot/<name> handles
        "scaled_renderer": None,   # ScaledSkeletonRenderer (nullable)
    }

    if not presets:
        server.gui.add_markdown(
            f"<div style='opacity:0.7;padding:8px 0;'>"
            f"No robot presets found under <code>configs/robots/</code>.  "
            f"Copy <code>configs/robots/_template/</code> and edit."
            f"</div>"
        )
        return

    # Build one label per preset.  Missing-URDF presets get a warning suffix
    # so the dropdown itself surfaces state; no click required.  We keep the
    # label → preset map around to look up the picked entry on callbacks.
    def _label(p: RobotPreset) -> str:
        return f"{p.display_name} · {p.name}" + ("" if p.has_urdf else "  ⚠ no URDF")

    label_to_preset: dict[str, RobotPreset] = {_label(p): p for p in presets}
    labels = list(label_to_preset)

    picker = server.gui.add_dropdown(
        "Robot preset",
        options=tuple(labels),
        initial_value=labels[0],
        hint=(
            "Discovered from configs/robots/*/robot.yaml.  Presets marked ⚠ "
            "have a robot.yaml but no URDF on disk yet — see README in each "
            "directory for where to drop it."
        ),
    )

    stats_md = server.gui.add_markdown("")

    # Library — identical controls + sync with Motion → Library (Search / Folder / Clip).
    initial_clip_labels: tuple[str, ...] = ("(no matches)",)
    initial_clip_value = "(no matches)"
    initial_folder_labels: tuple[str, ...] = (_FOLDER_ALL,)
    initial_folder_value = _FOLDER_ALL
    initial_search_value = ""
    if motion_sync is not None:
        try:
            all_lbls = list(motion_sync.get("all_labels", lambda: [])())  # type: ignore[arg-type,misc]
            cur = motion_sync.get("current_label", lambda: None)()  # type: ignore[arg-type,misc]
            if all_lbls:
                initial_clip_labels = tuple(all_lbls)
                initial_clip_value = cur if cur in all_lbls else all_lbls[0]
            folder_opts = list(motion_sync.get("folder_options", lambda: [])())  # type: ignore[arg-type,misc]
            cur_folder = motion_sync.get("get_folder", lambda: _FOLDER_ALL)()  # type: ignore[arg-type,misc]
            if folder_opts:
                initial_folder_labels = tuple(folder_opts)
                initial_folder_value = (
                    cur_folder if cur_folder in folder_opts else folder_opts[0]
                )
            initial_search_value = str(
                motion_sync.get("get_search", lambda: "")()  # type: ignore[arg-type,misc]
            )
        except Exception:
            pass

    with server.gui.add_folder("Library", expand_by_default=True):
        robot_lib_prog_md = server.gui.add_markdown("")
        robot_lib_prog_bar = server.gui.add_progress_bar(
            value=0.0, visible=False, animated=False,
        )
        robot_search_box = server.gui.add_text(
            "Search",
            initial_value=initial_search_value,
            hint="Case-insensitive, whitespace-separated tokens match in any order.",
        )
        robot_folder_picker = server.gui.add_dropdown(
            "Folder",
            options=initial_folder_labels,
            initial_value=initial_folder_value,
        )
        robot_clip_picker = server.gui.add_dropdown(
            "Clip",
            options=initial_clip_labels,
            initial_value=initial_clip_value,
        )

    if motion_sync is not None:
        motion_sync["_library_progress_mirror_pair"] = (
            robot_lib_prog_bar,
            robot_lib_prog_md,
        )

    _robot_lib_sync_guard = {"folder": False, "search": False, "clip": False}

    @robot_search_box.on_update
    def _on_robot_search(_):  # type: ignore[no-untyped-def]
        if _robot_lib_sync_guard["search"] or motion_sync is None:
            return
        setter = motion_sync.get("set_search")
        if callable(setter):
            try:
                setter(robot_search_box.value)
            except Exception:
                pass

    @robot_folder_picker.on_update
    def _on_robot_folder(_):  # type: ignore[no-untyped-def]
        if _robot_lib_sync_guard["folder"] or motion_sync is None:
            return
        setter = motion_sync.get("set_folder")
        if callable(setter):
            try:
                setter(robot_folder_picker.value)
            except Exception:
                pass

    @robot_clip_picker.on_update
    def _on_robot_clip_pick(_):  # type: ignore[no-untyped-def]
        if _robot_lib_sync_guard["clip"] or motion_sync is None:
            return
        loader = motion_sync.get("load_by_label")
        if callable(loader):
            try:
                loader(robot_clip_picker.value)
            except Exception:
                pass

    def _on_folder_changed_external(folder: str) -> None:
        """Listener the Motion tab fires when its folder dropdown changes."""
        try:
            if motion_sync is not None:
                folder_opts = list(motion_sync.get("folder_options", lambda: [])())  # type: ignore[arg-type,misc]
                if folder_opts and tuple(folder_opts) != robot_folder_picker.options:
                    robot_folder_picker.options = tuple(folder_opts)
            if (
                folder in robot_folder_picker.options
                and robot_folder_picker.value != folder
            ):
                _robot_lib_sync_guard["folder"] = True
                try:
                    robot_folder_picker.value = folder
                finally:
                    _robot_lib_sync_guard["folder"] = False
        except Exception:
            pass

    def _on_search_changed_external(query: str) -> None:
        """Listener the Motion tab fires when its search box changes."""
        try:
            if robot_search_box.value != query:
                _robot_lib_sync_guard["search"] = True
                try:
                    robot_search_box.value = query
                finally:
                    _robot_lib_sync_guard["search"] = False
        except Exception:
            pass

    def _on_clip_changed_external(label: str) -> None:
        """Listener the Motion tab fires when its clip dropdown changes."""
        if calib_state.get("active"):
            _exit_calibration_mode()
            _notify_all(
                server,
                "Calibration cancelled",
                "Motion changed — exited calibration without saving.",
                color="blue",
            )
        try:
            if motion_sync is not None:
                all_lbls = list(motion_sync.get("all_labels", lambda: [])())  # type: ignore[arg-type,misc]
                if all_lbls and tuple(all_lbls) != robot_clip_picker.options:
                    robot_clip_picker.options = tuple(all_lbls)
            if label in robot_clip_picker.options and robot_clip_picker.value != label:
                _robot_lib_sync_guard["clip"] = True
                try:
                    robot_clip_picker.value = label
                finally:
                    _robot_lib_sync_guard["clip"] = False
        except Exception:
            pass

    def _on_library_refreshed_external(
        options: tuple[str, ...], selected: str,
    ) -> None:
        """Keep Robot-tab clip options in sync after folder/search filter changes."""
        try:
            if options and tuple(options) != robot_clip_picker.options:
                robot_clip_picker.options = options
            if selected in robot_clip_picker.options and robot_clip_picker.value != selected:
                _robot_lib_sync_guard["clip"] = True
                try:
                    robot_clip_picker.value = selected
                finally:
                    _robot_lib_sync_guard["clip"] = False
        except Exception:
            pass

    if motion_sync is not None:
        folder_listeners = motion_sync.setdefault("folder_listeners", [])
        if isinstance(folder_listeners, list):
            folder_listeners.append(_on_folder_changed_external)
        search_listeners = motion_sync.setdefault("search_listeners", [])
        if isinstance(search_listeners, list):
            search_listeners.append(_on_search_changed_external)
        lib_refresh_listeners = motion_sync.setdefault("library_refresh_listeners", [])
        if isinstance(lib_refresh_listeners, list):
            lib_refresh_listeners.append(_on_library_refreshed_external)
        listeners = motion_sync.setdefault("listeners", [])
        if isinstance(listeners, list):
            listeners.append(_on_clip_changed_external)

    load_btn = server.gui.add_button(
        "Load to scene", icon="player-play",
        hint="Parse URDF + compile MJCF + render T-pose mesh (ground-aligned).",
    )
    clear_btn = server.gui.add_button(
        "Clear scene", icon="trash",
        hint="Remove the loaded robot meshes and retarget results from the viewer.",
    )
    show_robot_toggle = server.gui.add_checkbox(
        "Show robot",
        initial_value=True,
        hint=(
            "Hide the robot meshes without removing them.  Useful when you "
            "want to inspect the scaled-skeleton preview or the human motion "
            "without the robot occluding the view."
        ),
    )

    @show_robot_toggle.on_update
    def _on_show_robot(_):  # type: ignore[no-untyped-def]
        animator = state.get("animator")
        if animator is None:
            return
        try:
            animator.set_visible(bool(show_robot_toggle.value))
        except Exception:
            pass

    schema_btn = server.gui.add_button(
        "Export schema CSV", icon="table",
        hint=(
            "Write the column header of the retargeted CSV to /tmp — "
            "useful for pinning dof_order in robot.yaml."
        ),
    )

    # Stage-2 retarget hookup.  The button is disabled until both a robot
    # and a motion are available (and stays disabled on URDF-missing
    # presets).  It runs :class:`~hhtools.retarget.newton_basic.NewtonBasicPipeline`
    # or :class:`~hhtools.retarget.interaction_mesh.InteractionMeshPipeline` in a
    # worker thread (see ``Retarget backend``).
    with server.gui.add_folder("Retarget"):
        # "Current" uses the clip on screen in the Motion tab; "All filtered"
        # iterates over whatever the Motion-tab dropdown is currently
        # showing (honours the search box / folder filter).  We deliberately
        # scope batch mode to the *visible* subset so a user that typed
        # "walk" to narrow things down won't accidentally retarget a 10k
        # AMASS dump by clicking a single button.
        retarget_scope = server.gui.add_dropdown(
            "Target",
            options=("Current motion", "All filtered motions"),
            initial_value="Current motion",
            hint=(
                "Current = whichever clip is loaded in the Motion tab.  "
                "All filtered = every entry currently visible in the "
                "Motion-tab library dropdown."
            ),
        )
        retarget_frames = server.gui.add_number(
            "Max frames (0 = all)",
            initial_value=0,
            min=0,
            max=100000,
            step=1,
            hint=(
                "Cap frames fed to the solver so the first preview lands in "
                "a couple of seconds.  Set to 0 for the whole clip."
            ),
        )
        retarget_iters = server.gui.add_number(
            "IK iterations",
            initial_value=16,
            min=1,
            max=200,
            step=1,
            hint=(
                "Newton: LM iterations per frame.  Interaction mesh: SQP inner "
                "iterations per frame."
            ),
        )
        retarget_human_height = server.gui.add_number(
            "Subject height (m)",
            initial_value=1.7,
            min=0.5,
            max=2.5,
            step=0.01,
            hint="Drives the scaler's height-ratio correction.",
        )
        retarget_backend = server.gui.add_dropdown(
            "Retarget backend",
            options=("Auto", "Newton IK", "Interaction mesh"),
            initial_value="Auto",
            hint=(
                "Auto: clips under intermimic/ or meshmimic/ "
                "(or with terrain/objects) use interaction mesh (Laplacian MPC); "
                "all other datasets use Newton IK.  Override here if needed."
            ),
        )
        retarget_show_scaled = server.gui.add_checkbox(
            "Show scaled human skeleton",
            initial_value=True,
            hint=(
                "Pre-IK scaler targets (orange).  At frame 0 they are solved to "
                "match your calibrated robot pose (the one you dialled against "
                "the blue reference), not a rescaled copy of the clip's raw "
                "frame-0 silhouette — scrub time to see the motion."
            ),
        )
        retarget_preview_btn = server.gui.add_button(
            "Preview scaled skeleton", icon="ruler",
            hint=(
                "Scaler only (no IK).  Frame 0 shows the rest closure onto the "
                "robot; later frames show scaled trajectory.  Fast — no Newton."
            ),
        )
        retarget_btn = server.gui.add_button(
            "Retarget", icon="wand",
            hint=(
                "Run scaler + IK (Newton or interaction mesh per backend) on the "
                "selected target (single or batch) and export CSV(s)."
            ),
        )
        save_robot_clip_btn = server.gui.add_button(
            "Save robot clip (pkl)", icon="device-floppy",
            hint=(
                "Persist the retargeted robot trajectory + scaled scene "
                "(objects, terrain) as separate .pkl files under "
                "assets/save_npz/<dataset>/<folder>/<clip>/.  "
                "Robot pose is [tx,ty,tz,qw,qx,qy,qz, *dof]; objects "
                "and terrain are scaled into the robot frame."
            ),
        )
        save_robot_clip_btn.disabled = True  # enabled once a retarget result lands
        retarget_bar = server.gui.add_progress_bar(
            0.0, animated=False, visible=False,
        )
        retarget_progress_md = server.gui.add_markdown("")
        retarget_status = server.gui.add_markdown(
            "<span style='opacity:0.6'>Select a motion in the Motion tab, "
            "then click Retarget above.</span>"
        )

    # The retarget folder gets its own progress reporter (separate from the
    # Motion-tab one) so a long batch run doesn't steal the Library panel's
    # bar.  Both reporters share the same asymptotic-curve logic.
    retarget_progress = _ProgressReporter(
        server, retarget_bar, retarget_progress_md,
    )

    def _set_retarget_and_playback_gates(*, calibration_active: bool) -> None:
        """Block Retarget + motion Playback while the calibration session is open."""
        lock = bool(calibration_active)
        for w in (
            retarget_scope,
            retarget_frames,
            retarget_iters,
            retarget_human_height,
            retarget_backend,
            retarget_show_scaled,
            retarget_preview_btn,
            retarget_btn,
        ):
            try:
                w.disabled = lock
            except Exception:
                pass
        if callable(get_panel):
            try:
                pn = get_panel()
                if pn is not None and hasattr(pn, "set_calibration_lock"):
                    pn.set_calibration_lock(lock)
            except Exception:
                pass

    # ---- Rig-type tracking for data-source-switch warnings ------------------
    _last_rig_type: dict[str, str | None] = {"value": None}

    def _check_rig_type_switch(motion_obj) -> bool:
        """Check if the rig type changed since last retarget.

        Returns True if the user confirmed (or no change), False if cancelled.
        When a switch is detected, updates retarget_status with a warning,
        auto-switches the calibration reference picker, and returns True.
        """
        from hhtools.retarget.newton_basic.human_aliases import (
            list_detected_rig_type,
            auto_source_to_canonical,
        )
        if motion_obj is None:
            return True
        current_rig = list_detected_rig_type(motion_obj.hierarchy.bone_names)
        prev = _last_rig_type["value"]

        # Same rules as :func:`_suggested_calibration_reference` (motion load):
        # prefer the library's *source* file (e.g. ``.glb``) over
        # ``get_last_loaded_source_path`` (often a converted ``.npz`` in the cache);
        # otherwise the extension-based ``glb`` hint is lost and Mixamo-style
        # rigs are mis-tagged as ``lafan_bvh``.
        cur_le: LibraryEntry | None = None
        if motion_sync is not None:
            entry_get = motion_sync.get("get_current_library_entry")
            if callable(entry_get):
                le = entry_get()
                if isinstance(le, LibraryEntry):
                    cur_le = le
        path_get = (
            motion_sync.get("get_last_loaded_source_path")
            if motion_sync is not None
            else None
        )
        pstr = path_get() if callable(path_get) else None
        suggested_ref = _suggested_calibration_reference(
            motion_obj, cur_le, Path(pstr) if pstr else None,
        )

        src2can = auto_source_to_canonical(motion_obj.hierarchy.bone_names)
        canonical_hits = {
            v for v in src2can.values()
            if v in {
                "hips", "spine", "chest", "neck", "head",
                "left_shoulder", "left_elbow", "left_wrist",
                "right_shoulder", "right_elbow", "right_wrist",
                "left_hip", "left_knee", "left_ankle",
                "right_hip", "right_knee", "right_ankle",
            }
        }
        coverage = len(canonical_hits)
        ref_switched = False
        if suggested_ref and calib_reference_picker.value != suggested_ref:
            try:
                calib_reference_picker.value = suggested_ref
                ref_switched = True
            except Exception:
                pass

        if prev is not None and current_rig != prev:
            esc_prev = _html_escape(prev)
            esc_curr = _html_escape(current_rig)
            ref_note = ""
            if ref_switched and suggested_ref:
                ref_note = (
                    f" Reference pose auto-switched to "
                    f"<b>{_html_escape(suggested_ref)}</b>."
                )
            retarget_status.content = (
                f"<span style='color:{PALETTE.ui_warn}'>⚠ Rig type changed:</span> "
                f"<b>{esc_prev}</b> → <b>{esc_curr}</b>. "
                f"Mapping coverage: <b>{coverage}/17</b> canonical joints."
                f"{ref_note} "
                f"Please verify the <b>calibration</b> is correct "
                f"for this data source."
            )
            _notify_all(
                server, "Data source type changed",
                f"{prev} → {current_rig} · "
                f"Coverage: {coverage}/17 canonical joints. "
                + (f"Reference → {suggested_ref}. " if ref_switched else "")
                + "Verify calibration is suitable.",
                color="yellow",
            )
        elif coverage < 10:
            esc_curr = _html_escape(current_rig)
            retarget_status.content = (
                f"<span style='color:{PALETTE.ui_error}'>⚠ Low mapping coverage:</span> "
                f"rig=<b>{esc_curr}</b>, "
                f"only <b>{coverage}/17</b> canonical joints resolved. "
                f"Scale/retarget results may be incorrect. "
                f"Try selecting a different reference that matches "
                f"your data source's joint naming convention."
            )
            _notify_all(
                server, f"Low mapping coverage ({coverage}/17)",
                f"Rig: {current_rig}. Results may be incorrect.",
                color="orange",
            )
        elif ref_switched and suggested_ref:
            retarget_status.content = (
                f"<span style='color:{PALETTE.ui_ok}'>✓</span> "
                f"Detected rig: <b>{_html_escape(current_rig)}</b>. "
                f"Reference pose auto-switched to "
                f"<b>{_html_escape(suggested_ref)}</b>."
            )

        _last_rig_type["value"] = current_rig
        return True

    # ---- Retarget calibration folder ---------------------------------------
    # Lets the user interactively pose the robot (via actuated-joint sliders)
    # so that its zero/calibrated pose matches a chosen reference human
    # T-pose.  The saved calibration feeds
    # :func:`build_scaler_config_from_calibration` at retarget time — which
    # is *required* now (no heuristic fallback).  See
    # :mod:`hhtools.retarget.calibration` for the derivation.
    # Calibration uses an inline session-folder pattern: Start
    # calibration mounts a new folder inside this one containing the
    # per-joint sliders + Save/Cancel/Reset; Save/Cancel remove the
    # whole session folder.  We deliberately avoid ``add_modal`` here
    # because viser's modal dims the 3D scene with a backdrop, which
    # would obscure the very alignment the user is trying to eyeball.
    # Viser doesn't expose a non-dimming left-sidebar widget, so the
    # session lives in the same right-hand panel as the rest of the
    # Robot tab — no occlusion, no dimming.
    with server.gui.add_folder("Retarget calibration"):
        calib_status_md = server.gui.add_markdown("")
        calib_reference_picker = server.gui.add_dropdown(
            "Reference pose",
            options=(
                "smplx", "smpl", "gvhmr",
                "soma_bvh", "lafan_bvh",
                "glb",
            ),
            initial_value="smpl",
            hint=(
                "smpl / smplx = canonical zero-pose (same joint layout as "
                "SMPL-family clips: AMASS, Motion-X, PHUMA, GVHMR, …); "
                "soma_bvh / lafan_bvh = format-specific rest poses; "
                "glb = frame 0 of the clip loaded in the viewer."
            ),
        )
        calib_show_labels = server.gui.add_checkbox(
            "Show joint names",
            initial_value=False,
            hint="Display joint name labels on the reference skeleton.",
        )
        calib_start_btn = server.gui.add_button(
            "Start calibration", icon="target",
            hint=(
                "Reveal joint sliders below so you can dial the robot's "
                "zero-pose into the reference human T-pose, then derive "
                "per-limb scale/offset on Save.  The reference skeleton "
                "is drawn in blue next to the robot in the 3D scene."
            ),
        )

    def _sync_calib_reference_after_motion_load() -> None:
        if get_current_motion is None or motion_sync is None:
            return
        motion_obj = get_current_motion()
        if motion_obj is None:
            return
        entry = motion_sync.get("get_current_library_entry")
        cur_entry = entry() if callable(entry) else None
        path_raw = motion_sync.get("get_last_loaded_source_path")
        pstr = path_raw() if callable(path_raw) else None
        sug = _suggested_calibration_reference(
            motion_obj,
            cur_entry if isinstance(cur_entry, LibraryEntry) else None,
            Path(pstr) if pstr else None,
        )
        if sug is None:
            return
        opts = tuple(calib_reference_picker.options)
        if sug in opts and calib_reference_picker.value != sug:
            try:
                calib_reference_picker.value = sug
            except Exception:
                pass
        try:
            server.flush()
        except Exception:
            pass

    if motion_sync is not None:
        motion_sync.setdefault("on_motion_loaded", []).append(
            _sync_calib_reference_after_motion_load,
        )

    progress_md = server.gui.add_markdown("")

    def _render_stats(preset: RobotPreset) -> str:
        esc_name = _html_escape(preset.display_name)
        esc_id = _html_escape(preset.name)
        missing_note = (
            "" if preset.has_urdf else
            f"<br/><span style='color:{PALETTE.ui_warn};'>URDF missing — drop file at "
            f"<code>{_html_escape(preset.urdf_path)}</code></span>"
        )
        dof_note = (
            f"{len(preset.dof_order)} declared" if preset.dof_order else
            "unpinned (URDF parse order)"
        )
        auto = bool(preset.meta.get("auto_generated"))
        source_badge = (
            f"<span style='background:{PALETTE.warn};color:#1e293b;padding:1px 6px;"
            f"border-radius:4px;font-size:0.75em;margin-left:8px;'>auto</span>"
            if auto else
            f"<span style='background:{PALETTE.good};color:#1e293b;padding:1px 6px;"
            f"border-radius:4px;font-size:0.75em;margin-left:8px;'>hand</span>"
        )
        return (
            f"<div style='padding:4px 0;font-size:0.88em;line-height:1.55'>"
            f"<b>{esc_name}</b>{source_badge}"
            f" &nbsp;<span style='opacity:0.6'>·</span>&nbsp; "
            f"<code style='font-size:0.85em'>{esc_id}</code><br/>"
            f"<span style='opacity:0.7'>dof_order</span>: {dof_note}<br/>"
            f"<span style='opacity:0.7'>up/forward</span>: "
            f"{_html_escape(preset.up_axis)}-up / +{_html_escape(preset.forward_axis)}<br/>"
            f"<span style='opacity:0.7'>ik_map</span>: {len(preset.ik_map)} entries"
            f"{missing_note}"
            f"</div>"
        )

    def _clear_scaled_preview() -> None:
        sr = state.get("scaled_renderer")
        if sr is not None:
            try:
                sr.clear()
            except Exception:
                pass
        state["scaled_renderer"] = None
        robot_state["scaled_renderer"] = None
        robot_state["scaled_preview"] = None

    def _publish_robot_objects(scaler_cfg, human_h: float) -> None:
        """Show scaled copies of the current motion's objects + terrain near the robot.

        Mirrors the **same** affine chain that the interaction-mesh /
        Newton retarget pipeline applies to the source pose
        (:meth:`InteractionMeshRetargeter._build_scaled_source_pose`):

        1. Floor-normalise (subtract foot-floor ``z_min`` from
           :func:`~hhtools.core.grounding.human_source_floor_z_world`).
        2. Uniform scale by ``model_height / human_h``.
        3. Offset by :data:`ROBOT_WORLD_OFFSET` so the retarget preview
           does not overlap the source view.

        Terrain uses :func:`~hhtools.core.grounding.terrain_heightfield_z_offset_world`
        for ``scaled(..., z_offset=...)`` so low cells are not half-buried when
        ``min(hf)`` is below the feet, matching the interaction-mesh collision asset.

        We deliberately do **not** apply ``source_body_quat`` here.
        ``source_body_quat`` is a yaw rotation that aligns the source
        actor's "forward" with the robot URDF-declared forward, but the
        whole retarget stack (``_build_scaled_source_pose``,
        ``_build_scaled_object_points``, the heightfield, the warm-start
        pelvis quaternion) keeps everything in the **source world frame**
        and lets the robot's free-joint quaternion absorb the heading
        difference.  Applying sbq to the preview here would re-introduce
        the bug it was added to avoid.

        A previous implementation tried to be symmetric — rotate
        position by ``sbq``, apply scale, rotate back by ``sbq⁻¹`` —
        but the ``rotate · scale · unrotate`` cycle on positions
        cancels exactly (the scale is isotropic and z_min normalisation
        is yaw-equivariant), while it also left ``oq · sbq⁻¹`` on the
        quaternion **without** the matching ``sbq · oq`` half on entry.
        The net effect was a quaternion-only ``sbq⁻¹`` rotation: the
        chair's position lined up but its orientation was off by ~90°
        (the typical SMPL→URDF yaw delta), e.g. OMOMO ``woodchair`` had
        the back-rest pointing the wrong way.  Removing the sbq dance
        entirely keeps positions, orientations, and the matching robot
        retarget all in one consistent source frame.

        Terrain travels via :class:`TerrainHeightfield`, scaled with the
        same ``scale`` as the skeleton; ``z_offset`` follows the retarget
        convention above (not necessarily equal to the skeleton floor).
        """
        from hhtools.retarget.calibration.calibration import (
            uniform_overlay_scale,
            uniform_overlay_scale_for_motion,
        )

        robj = robot_state.get("robot_objects_renderer")
        rterrain = robot_state.get("robot_terrain_renderer")
        if robj is None and rterrain is None:
            return
        m = get_current_motion() if get_current_motion is not None else None
        if isinstance(m, Motion):
            mdl = state.get("current_model")
            ik_keys = (
                frozenset(mdl.preset.ik_map.keys())
                if mdl is not None and hasattr(mdl, "preset")
                else frozenset()
            )
            ratio = float(
                uniform_overlay_scale_for_motion(
                    scaler_cfg, float(human_h), m, ik_map_keys=ik_keys,
                )
            )
        else:
            ratio = float(uniform_overlay_scale(scaler_cfg, float(human_h)))

        if not isinstance(m, Motion):
            if robj is not None:
                robj.clear()  # type: ignore[union-attr]
            if rterrain is not None:
                rterrain.clear()  # type: ignore[union-attr]
            return

        pos_all = np.asarray(m.positions, dtype=np.float32)
        z_min = float(human_source_floor_z_world(m))
        z_terrain = float(terrain_heightfield_z_offset_world(m, z_min))
        off = np.array(ROBOT_WORLD_OFFSET, dtype=np.float32)

        if robj is not None:
            scaled: list[SceneObject] = []
            for ob in m.objects:
                op = ob.positions.copy().astype(np.float32)
                op[:, 2] -= z_min
                op *= ratio
                op += off[None, :]
                scaled.append(SceneObject(
                    name=f"robot_{ob.name}",
                    positions=op,
                    quaternions=ob.quaternions.copy(),
                    extents=ob.extents * ratio,
                    mesh_path=ob.mesh_path,
                    scale=ob.scale * ratio,
                    opacity=ob.opacity,
                    color=ob.color,
                ))
            robj.set_objects(scaled)  # type: ignore[union-attr]

        if rterrain is not None:
            if m.terrain is None:
                rterrain.clear()  # type: ignore[union-attr]
            else:
                # Robot frame = source frame scaled by ``ratio`` with
                # the floor at z=0, then shifted by ROBOT_WORLD_OFFSET so
                # the retarget preview doesn't overlap the source view.
                hf_robot = m.terrain.scaled(ratio, z_offset=z_terrain).shifted(
                    dx=float(off[0]),
                    dy=float(off[1]),
                    dz=float(off[2]),
                )
                rterrain.set_terrain(hf_robot)  # type: ignore[union-attr]

    def _clear_robot_objects() -> None:
        robj = robot_state.get("robot_objects_renderer")
        if robj is not None:
            robj.clear()  # type: ignore[union-attr]
        rterrain = robot_state.get("robot_terrain_renderer")
        if rterrain is not None:
            rterrain.clear()  # type: ignore[union-attr]

    def _invalidate_on_source_clip_change() -> None:
        """Drop retarget IK, scaler preview, and robot pose tied to the previous clip."""

        robot_state["retargeted"] = None
        robot_state["retargeted_source_motion"] = None
        robot_state["retargeted_lib_entry"] = None
        robot_state["retargeted_model"] = None
        try:
            save_robot_clip_btn.disabled = True
        except Exception:
            pass
        _clear_scaled_preview()
        _clear_robot_objects()
        anim = robot_state.get("animator")
        model = state.get("current_model")
        if anim is not None and isinstance(model, URDFRobotModel):
            try:
                dof_names = tuple(j.name for j in model.actuated_joints)
                anim.set_frame_joint_q(
                    np.zeros(len(dof_names), dtype=np.float64),
                    dof_names,
                    has_root=False,
                )
            except Exception:
                pass

    if motion_sync is not None:
        motion_sync["invalidate_robot_artifacts"] = _invalidate_on_source_clip_change

    def _clear_scene() -> None:
        """Tear down every viser handle the Robot tab owns.

        Covers three kinds of state: the mesh handles published under
        ``/robot/<preset>/...`` (managed by :class:`RobotAnimator`), the
        scaled-skeleton preview under ``/scaled_human``, and the cross-tab
        ``robot_state`` slots so the main render loop stops trying to
        animate a model that no longer exists.
        """
        anim = state.get("animator")
        if anim is not None:
            try:
                anim.clear()
            except Exception:
                pass
        state["animator"] = None
        state["current_model"] = None
        _clear_scaled_preview()
        _clear_robot_objects()
        robot_state["animator"] = None
        robot_state["retargeted"] = None
        robot_state["retargeted_source_motion"] = None
        robot_state["retargeted_lib_entry"] = None
        robot_state["retargeted_model"] = None
        try:
            save_robot_clip_btn.disabled = True
        except Exception:
            pass

    def _render_robot(model: URDFRobotModel) -> int:
        """Instantiate a :class:`RobotAnimator` for ``model`` and publish it.

        The animator handles per-link mesh handle creation (local vertices,
        handle.position/wxyz controlled externally) AND ground alignment
        (lifts the whole robot so the lowest T-pose vertex sits on ``z=0``).
        We also apply :data:`ROBOT_WORLD_OFFSET` so the robot stands 1 m
        beside the source-motion human rather than overlapping it.

        Returns the number of mesh handles actually placed — useful for log
        output.  The animator lives on both the tab-local ``state`` and the
        shared ``robot_state`` so the render loop can animate it.
        """
        animator = RobotAnimator(server, model, world_offset=ROBOT_WORLD_OFFSET)
        # Honour the Show-robot checkbox straight after load so users aren't
        # surprised by a visible robot appearing when they had it hidden.
        try:
            animator.set_visible(bool(show_robot_toggle.value))
        except Exception:
            pass
        state["animator"] = animator
        robot_state["animator"] = animator
        return animator.num_handles()

    @picker.on_update
    def _on_pick(_):  # type: ignore[no-untyped-def]
        preset = label_to_preset[picker.value]
        stats_md.content = _render_stats(preset)
        load_btn.disabled = not preset.has_urdf

    # Prime the stats + button state.
    stats_md.content = _render_stats(label_to_preset[picker.value])
    load_btn.disabled = not label_to_preset[picker.value].has_urdf

    # Persistent references for modal widgets so they aren't garbage-
    # collected before the user interacts with them.
    _load_modal_refs: dict[str, object] = {}

    def _show_recalibrate_modal(preset: RobotPreset) -> None:
        """Post-load modal asking whether to recalibrate.

        Shown only when the robot already has a saved calibration yaml.
        The modal and its buttons are stashed in ``_load_modal_refs`` to
        prevent GC before the user clicks.
        """
        modal = server.gui.add_modal("Calibration exists")
        _load_modal_refs["modal"] = modal
        with modal:
            server.gui.add_markdown(
                f"<b>{_html_escape(preset.display_name)}</b> already has a "
                "saved calibration.<br>"
                "Do you want to open the calibration editor?"
            )
            btn_yes = server.gui.add_button("Recalibrate", color="green")
            btn_no = server.gui.add_button("No, keep current", color="gray")
            _load_modal_refs["btn_yes"] = btn_yes
            _load_modal_refs["btn_no"] = btn_no

        @btn_yes.on_click
        def _yes(_):  # type: ignore[no-untyped-def]
            modal.close()
            _enter_calibration_mode()

        @btn_no.on_click
        def _no(_):  # type: ignore[no-untyped-def]
            modal.close()

    @load_btn.on_click
    def _on_load(_):  # type: ignore[no-untyped-def]
        preset = label_to_preset[picker.value]

        def _worker() -> None:
            progress_md.content = (
                f"<span style='opacity:0.8'>⟳</span> "
                f"Loading <b>{_html_escape(preset.display_name)}</b>…"
            )
            try:
                server.flush()
            except Exception:
                pass
            try:
                model = load_robot(preset, build_collision_scene=True)
            except Exception as err:
                progress_md.content = (
                    f"<span style='color:{PALETTE.ui_error}'>Load failed:</span> "
                    f"<code>{_html_escape(f'{type(err).__name__}: {err}')}</code>"
                )
                _notify_all(
                    server, "Robot load failed",
                    f"{preset.name}: {err}", color="red",
                )
                return

            _clear_scene()
            state["current_model"] = model
            try:
                n = _render_robot(model)
            except Exception as err:
                progress_md.content = (
                    f"<span style='color:{PALETTE.ui_error}'>Render failed:</span> "
                    f"<code>{_html_escape(f'{type(err).__name__}: {err}')}</code>"
                )
                return
            animator = state["animator"]
            ground_lift = getattr(animator, "ground_offset_z", 0.0)
            progress_md.content = (
                f"<span style='color:{PALETTE.ui_ok}'>✓</span> "
                f"Loaded <b>{_html_escape(preset.display_name)}</b> · "
                f"{len(model.actuated_joints)} DOF · {n} meshes · "
                f"lifted {ground_lift:.3f} m to ground"
            )
            _notify_all(
                server, "Robot loaded",
                f"{preset.display_name}: {len(model.actuated_joints)} DOF, {n} meshes",
                color="green",
            )
            _refresh_calib_status()

            def _prewarm_ik() -> None:
                try:
                    from hhtools.retarget.newton_basic.pipeline import (
                        NewtonBasicPipeline,
                        PipelineConfig,
                    )

                    NewtonBasicPipeline.prewarm_for_robot(
                        model,
                        pipeline_config=PipelineConfig(
                            ik_iterations=max(1, int(retarget_iters.value)),
                            ik_use_cuda_graph=False,
                        ),
                    )
                except Exception:
                    pass

            prev_prewarm = robot_state.get("_prewarm_thread")
            if isinstance(prev_prewarm, threading.Thread) and prev_prewarm.is_alive():
                prev_prewarm.join(timeout=0.5)

            prewarm_thread = threading.Thread(
                target=_prewarm_ik,
                name=f"hhtools-prewarm-{preset.name}",
                daemon=True,
            )
            robot_state["_prewarm_thread"] = prewarm_thread
            prewarm_thread.start()

            # After successful load, check if a calibration already
            # exists and prompt the user about recalibrating.
            from hhtools.retarget.calibration import resolve_calibration_file
            cal_path = (
                resolve_calibration_file(
                    preset.urdf_path.parent,
                    str(calib_reference_picker.value),
                )
                if preset.urdf_path is not None else None
            )
            if cal_path is not None:
                _show_recalibrate_modal(preset)

        threading.Thread(
            target=_worker, name=f"hhtools-robot-load-{preset.name}", daemon=True,
        ).start()

    @clear_btn.on_click
    def _on_clear(_):  # type: ignore[no-untyped-def]
        _exit_calibration_mode()
        _clear_scene()
        progress_md.content = (
            "<span style='opacity:0.6'>Scene cleared.</span>"
        )
        _refresh_calib_status()

    # ---- Calibration session state -----------------------------------------
    # When the user clicks "Start calibration" we materialise a sub-panel
    # worth of joint sliders inside ``_calib_sliders_container`` and stash
    # the live state in ``calib_state``.  Saving / cancelling tears those
    # down, but the top-level calibration folder stays put so the status
    # label keeps reporting the robot's calibration state at a glance.
    calib_state: dict[str, object] = {
        "active": False,           # True while the session panel is on screen
        "ref_renderer": None,      # ReferenceSkeletonRenderer | None
        "session": None,           # viser folder holding session-specific widgets
        "sliders": {},             # name -> slider handle (inside the session)
        "current_q": {},           # name -> radians (what's been dialled in)
        # ik_map editing (Part 2 of the calibration flow): we show one
        # dropdown per ROBOT LINK letting the user pick which canonical
        # human joint from the reference pose should drive it.  Both
        # halves of the session share this state so Save can write a
        # single consolidated update to robot.yaml.
        "ik_map": {},              # canonical -> robot_link (matches preset.ik_map shape)
        "mapping_dropdowns": {},   # robot_link -> dropdown handle
    }

    def _render_calib_status() -> str:
        """Compose the status markdown for the calibration folder.

        Shows whether a calibration yaml already exists for the loaded
        robot, which reference it targeted, and hints at the next action
        (load → start → save) depending on the live state.
        """

        model = state.get("current_model")
        if not isinstance(model, URDFRobotModel):
            return (
                "<span style='opacity:0.6'>Load a robot to see its "
                "calibration state.</span>"
            )
        cal = _resolve_robot_calibration(model)
        active = calib_state["active"]
        if active:
            return (
                f"<span style='color:{PALETTE.ui_warn}'>● Calibration in progress</span> "
                "— move the sliders below, then click <b>Save</b>."
            )
        if cal is None:
            ref_esc = _html_escape(str(calib_reference_picker.value))
            return (
                f"<span style='color:{PALETTE.ui_error}'>● Not calibrated</span> — "
                f"no file for reference <code>{ref_esc}</code> "
                f"(<code>retarget_calibration_{ref_esc}.yaml</code>). "
                "Click <b>Start calibration</b> and save."
            )
        return (
            f"<span style='color:{PALETTE.ui_ok}'>● Calibrated</span> · "
            f"reference <code>{_html_escape(cal.reference)}</code> · "
            f"{len(cal.calibrated_joint_q)} joint(s) non-zero."
        )

    def _refresh_calib_status() -> None:
        try:
            calib_status_md.content = _render_calib_status()
        except Exception:
            pass

    def _clear_calib_sliders() -> None:
        """Tear down the session folder + reference skeleton.  Idempotent.

        The session folder scopes every GUI widget created in the
        "calibration-active" state (sliders + the in-session Save /
        Cancel / Reset buttons).  Calling ``.remove()`` on the folder
        disposes the lot in one go, so we don't need to track individual
        widget handles.  The reference-skeleton renderer is scene-graph
        state (not GUI), so we clear it separately.
        """

        session = calib_state.get("session")
        if session is not None:
            try:
                session.remove()  # type: ignore[attr-defined]
            except Exception:
                pass
        calib_state["session"] = None
        calib_state["sliders"] = {}
        calib_state["mapping_dropdowns"] = {}
        renderer = calib_state.get("ref_renderer")
        if renderer is not None:
            try:
                renderer.clear()
            except Exception:
                pass
        calib_state["ref_renderer"] = None

    def _apply_calib_q() -> None:
        """Re-drive the robot animator from the sliders' current values.

        Called on every slider update + once after the reset button.
        Uses ``set_frame_joint_q(has_root=False)`` so the robot stands at
        its world-offset base rather than being placed by an IK solve
        (which, in calibration mode, hasn't run yet).
        """
        animator = state.get("animator")
        model = state.get("current_model")
        if animator is None or not isinstance(model, URDFRobotModel):
            return
        joint_order = tuple(j.name for j in model.actuated_joints)
        q = np.asarray(
            [calib_state["current_q"].get(n, 0.0) for n in joint_order],
            dtype=np.float64,
        )
        try:
            animator.set_frame_joint_q(q, joint_order, has_root=False)
        except Exception:
            pass

    def _load_existing_calib_into_state(model: URDFRobotModel) -> None:
        """Seed ``calib_state['current_q']`` from a saved calibration file.

        If no yaml exists, every joint stays at 0 — matching the user's
        earlier "initial Q5" choice ("A) 全零").  Missing joints in the
        yaml also default to 0 so partial calibrations are fine.
        """

        joint_order = tuple(j.name for j in model.actuated_joints)
        q = {n: 0.0 for n in joint_order}
        cal = _resolve_robot_calibration(model)
        if cal is not None:
            for name, value in cal.calibrated_joint_q.items():
                if name in q:
                    q[name] = float(value)
        calib_state["current_q"] = q

    def _resolve_robot_calibration(model: URDFRobotModel):
        """Look up the retarget calibration yaml sitting next to the URDF.

        Returns a :class:`~hhtools.retarget.calibration.RobotRetargetCalibration`
        or ``None`` if no yaml exists yet — callers gate retarget on a
        non-None return and surface the missing-calibration error to the
        user (Robot tab's "Start calibration" button is the recovery).
        """
        from hhtools.retarget.calibration import (
            load_calibration,
            resolve_calibration_file,
        )

        preset = model.preset
        if preset.urdf_path is None:
            return None
        cal_path = resolve_calibration_file(
            preset.urdf_path.parent,
            str(calib_reference_picker.value),
        )
        if cal_path is None:
            return None
        try:
            return load_calibration(cal_path)
        except Exception as err:  # noqa: BLE001 — surfaced in UI
            _notify_all(
                server, "Calibration invalid",
                f"{cal_path.name}: {type(err).__name__}: {err}",
                color="red",
            )
            return None

    def _build_scaler_config_or_warn(model, clip, human_h):
        """Resolve the calibration for ``model`` and build a ScalerConfig.

        Returns ``None`` when no calibration is available — callers surface
        a UI banner directing the user at the "Start calibration" button.

        Unlike the ad-hoc first-frame heuristic this replaces, the result
        depends only on the robot's saved calibration yaml + the source
        motion's hierarchy + its frame-0 pose (for yaw alignment); the
        per-limb scales and rotation offsets are derived in closed form
        from the URDF's forward kinematics at the calibrated joint
        configuration.  See :mod:`hhtools.retarget.calibration` for the
        derivation details.
        """

        from hhtools.robot.retarget_profile import build_scaler_config_for_robot

        cal = _resolve_robot_calibration(model)
        if cal is None:
            return None
        try:
            return build_scaler_config_for_robot(
                cal, model, clip, human_height=float(human_h),
            )
        except Exception as err:  # noqa: BLE001 — surfaced in UI
            _notify_all(
                server, "Calibration failed",
                f"{type(err).__name__}: {err}",
                color="red",
            )
            return None

    def _slice_motion(motion_in, max_frames):
        """Return a shallow copy of ``motion_in`` truncated to ``max_frames``.

        ``max_frames == 0`` means "use the whole clip" (and skips the copy
        altogether — the caller owns a shared reference, but the pipeline
        is read-only so that's safe).
        """
        import copy
        if max_frames <= 0 or motion_in.num_frames <= max_frames:
            return motion_in
        clip = copy.copy(motion_in)
        clip.positions = clip.positions[:max_frames]
        clip.quaternions = clip.quaternions[:max_frames]
        return clip

    def _use_source_topology_scaled_preview(model, clip) -> bool:
        """Use scaler joint rows + motion hierarchy for the yellow overlay.

        Always True when the source skeleton has enough joints to form a
        meaningful skeleton (>= 10 bones).  The old ``ik_map + 10``
        threshold excluded OMOMO (24 bones) and parc_ms (15 bones),
        forcing them into per-joint-only display with distorted proportions.
        """
        return int(clip.num_bones) >= 10

    def _scaler_skeleton_segment_indices(
        joint_names: tuple[str, ...],
        hierarchy,
        *,
        ik_map_canonicals: frozenset[str] | None = None,
    ):
        """Edges (parent→child) in *scaler joint index* space, skipping FootMod."""
        pi = np.asarray(hierarchy.parent_indices, dtype=np.int64)
        hnames = list(hierarchy.bone_names)
        h_idx = {hnames[i]: i for i in range(len(hnames))}
        name_to_i = {n: i for i, n in enumerate(joint_names)}
        ik_canons = ik_map_canonicals or frozenset()
        src: list[int] = []
        dst: list[int] = []
        for i, n in enumerate(joint_names):
            if n not in h_idx:
                continue
            if str(n).lower().endswith("footmod"):
                continue
            hi = int(h_idx[n])
            p = int(pi[hi])
            anc_sc = None
            while p >= 0:
                pn = hnames[p]
                if pn in name_to_i:
                    j = int(name_to_i[pn])
                    if j != i:
                        anc_sc = j
                    break
                p = int(pi[p])
            if anc_sc is not None:
                src.append(anc_sc)
                dst.append(i)
        fs: list[int] = []
        fd: list[int] = []
        for s, d in zip(src, dst, strict=True):
            ns, nd = joint_names[s], joint_names[d]
            if exclude_joint_from_compact_scaled_preview(
                ns,
            ) or exclude_joint_from_compact_scaled_preview(nd):
                continue
            if ik_canons and (
                exclude_unmapped_head_neck_from_scaled_preview(
                    ns, ik_map_canonicals=ik_canons,
                )
                or exclude_unmapped_head_neck_from_scaled_preview(
                    nd, ik_map_canonicals=ik_canons,
                )
            ):
                continue
            fs.append(s)
            fd.append(d)
        return (
            np.asarray(fs, dtype=np.int32),
            np.asarray(fd, dtype=np.int32),
        )

    def _compute_scaled_preview(model, clip, human_h):
        """Run the scaler to produce a pre-IK skeleton preview.

        Uses **uniform scaling** (``robot_height / human_height``) for the
        dense source-topology overlay so the yellow figure keeps
        anatomical proportions across the whole motion — matching
        holosoma's approach and what
        :meth:`hhtools.retarget.interaction_mesh.pipeline.InteractionMeshPipeline._build_scaled_source_pose`
        feeds to MPC SQP.  Without this, dense rigs that route through
        MPC SQP (OMOMO, meshmimic, GVHMR-with-objects) show a puffed
        torso + elongated extremities relative to the uniformly-scaled
        terrain rendered next to them.

        The IK-target subset (``transforms`` field with canonical
        ``ik_names``) still comes from the per-joint soma-style scaler so
        Newton IK targets land on the calibrated robot link positions at
        rest.  When the dense overlay is active (``source_transforms`` is
        present) the renderer draws the dense skeleton instead of the
        per-joint beads, so the two representations don't fight for
        screen space.
        """
        from hhtools.core.math import quaternion as _Q
        from hhtools.retarget.newton_basic import ScaledMotionPreview
        from hhtools.retarget.newton_basic.heading_align import (
            align_effector_tensor_to_source_heading,
        )
        from hhtools.retarget.newton_basic.human_aliases import (
            effectors_to_canonical_table,
        )
        from hhtools.retarget.calibration.calibration import (
            uniform_overlay_scale_for_motion,
        )
        from hhtools.retarget.newton_basic.scaler import HumanToRobotScaler
        from hhtools.viewer.anatomy import motion_has_interaction_scene

        if callable(apply_motion_pipeline):
            clip = apply_motion_pipeline(clip)

        scaler_cfg = _build_scaler_config_or_warn(model, clip, human_h)
        if scaler_cfg is None:
            return None
        scaler = HumanToRobotScaler(
            clip.hierarchy, scaler_cfg, human_height=float(human_h),
        )
        scaled = scaler.apply(clip)
        sbq = np.asarray(scaler_cfg.source_body_quat, dtype=np.float32)

        canonical_to_target = effectors_to_canonical_table(
            scaler.joint_names, scaled.transforms, source_to_canonical=None,
        )

        ik_names_list = list(model.preset.ik_map.keys())
        F = scaled.transforms.shape[0]
        out = np.full((F, len(ik_names_list), 7), np.nan, dtype=np.float32)
        for k, canon in enumerate(ik_names_list):
            tgt = canonical_to_target.get(canon)
            if tgt is not None:
                out[:, k, :] = tgt

        ik_names = tuple(ik_names_list)
        out = align_effector_tensor_to_source_heading(
            out, source_body_quat=sbq,
        )

        if _use_source_topology_scaled_preview(model, clip):
            jn = scaler.joint_names
            ik_canons = frozenset(model.preset.ik_map.keys())
            seg_s, seg_d = _scaler_skeleton_segment_indices(
                jn, clip.hierarchy, ik_map_canonicals=ik_canons,
            )
            if int(seg_s.size) > 0:
                bead_idx = scaler_compact_bead_row_indices(jn, clip)
                if ik_canons and "head" not in ik_canons:
                    bead_idx = np.asarray(
                        [
                            int(i)
                            for i in np.asarray(bead_idx, dtype=np.int32).reshape(-1)
                            if not exclude_unmapped_head_neck_from_scaled_preview(
                                jn[int(i)], ik_map_canonicals=ik_canons,
                            )
                        ],
                        dtype=np.int32,
                    )

                M = len(jn)

                # Source-topology yellow preview uses **uniform** scaling
                # (``robot_height / human_height``) so the dense skeleton
                # keeps anatomical proportions across the whole motion.
                #
                # The per-joint soma-style scaler is only exact at the
                # calibration rest pose — every joint's calibrated
                # ``t_offset`` is a constant displacement solved at rest, so
                # on motion frames the per-joint outputs drift from the
                # source's bone lengths (neighbouring joints with different
                # scale[j] no longer agree on segment length, and the
                # constant t_offset for distal bones (wrist→hand,
                # ankle→toe, fingers) shows up as a fixed outward bias).
                # On dense rigs that route through MPC SQP / interaction-
                # mesh (OMOMO, meshmimic, GVHMR-with-objects) this reads
                # visually as a puffed torso + elongated extremities,
                # because the solver itself only sees uniform-scaled
                # positions (see
                # :meth:`InteractionMeshPipeline._build_scaled_source_pose`),
                # and the on-screen terrain / scene objects are uniform-
                # scaled too (see ``_publish_robot_objects``).  Aligning
                # the yellow overlay with the same uniform scaling closes
                # the visual gap and matches the function's own docstring.
                smpl_scale = float(
                    uniform_overlay_scale_for_motion(
                        scaler_cfg,
                        float(human_h),
                        clip,
                        ik_map_keys=ik_canons,
                    )
                )

                src_pos_full = np.asarray(clip.positions, dtype=np.float32).copy()
                z_min_uniform = float(human_source_floor_z_world(clip))
                src_pos_full[:, :, 2] -= z_min_uniform
                src_pos_full *= smpl_scale
                if not motion_has_interaction_scene(clip):
                    from hhtools.web.scaled_preview import (
                        resolve_scaled_overlay_z_correction,
                    )

                    z_corr = float(
                        resolve_scaled_overlay_z_correction(
                            clip, scaler, smpl_scale,
                        )
                    )
                    if abs(z_corr) > 1e-6:
                        src_pos_full[:, :, 2] += np.float32(z_corr)

                src_quat_full = np.asarray(clip.quaternions, dtype=np.float32).copy()

                hname_to_idx_uniform = {
                    n: i for i, n in enumerate(clip.hierarchy.bone_names)
                }
                _parents = np.asarray(
                    clip.hierarchy.parent_indices, dtype=np.int64,
                )
                _root_arr = np.where(_parents < 0)[0]
                _root_idx = int(_root_arr[0]) if _root_arr.size > 0 else 0
                _src_idx_list = [
                    hname_to_idx_uniform.get(n, _root_idx) for n in jn
                ]
                mapped_src_idx = np.asarray(_src_idx_list, dtype=np.int64)

                pos_m = src_pos_full[:, mapped_src_idx, :].astype(
                    np.float32, copy=True,
                )
                quat_m = src_quat_full[:, mapped_src_idx, :].astype(
                    np.float32, copy=True,
                )
                quat_m = _Q.normalize(quat_m.reshape(-1, 4)).reshape(F, M, 4)
                quat_m = _Q.ensure_continuous(quat_m)

                # Add lightweight visual-only contact points (toe/sole/hand tip)
                # derived from the same URDF/MuJoCo collision geometry used by
                # the interaction-mesh solver.  This makes the yellow preview
                # expose the parts that are most likely to collide.
                try:
                    from hhtools.retarget.interaction_mesh.contact_points import (
                        build_contact_mpc_points,
                    )
                    from hhtools.retarget.interaction_mesh.mujoco_scene import MujocoScene
                    from hhtools.retarget.newton_basic.human_aliases import (
                        auto_source_to_canonical as _auto_source_to_canonical,
                    )

                    src2can = _auto_source_to_canonical(clip.hierarchy.bone_names)
                    can2src: dict[str, str] = {}
                    for src_name, can_name in src2can.items():
                        can2src.setdefault(can_name, src_name)
                    hname_to_idx = {n: i for i, n in enumerate(clip.hierarchy.bone_names)}
                    scaler_name_to_idx = {n: i for i, n in enumerate(jn)}
                    c_src_idx: list[int] = []
                    c_links: list[str] = []
                    c_names: list[str] = []
                    for canon, link_name in model.preset.ik_map.items():
                        src_name = can2src.get(canon)
                        if src_name is None or src_name not in hname_to_idx:
                            continue
                        c_src_idx.append(int(hname_to_idx[src_name]))
                        c_links.append(str(link_name))
                        c_names.append(str(canon))

                    mj_scene = MujocoScene.from_robot(model)
                    c_points = build_contact_mpc_points(
                        mj_scene.model, c_links, c_src_idx, c_names,
                    )

                    def _norm_vec(v, fallback, *, ref_norm=None, ratio=0.0):
                        # Per-frame fallback: when ``fallback`` already has
                        # shape ``(F, 3)`` we keep its per-frame value
                        # instead of pinning every degenerate frame to a
                        # single reference vector.  That matters during
                        # 180° pivots — see the matching reasoning on
                        # :meth:`InteractionMeshPipeline._normalise_vectors_per_frame`.
                        # When ``ref_norm`` + ``ratio`` are given, also
                        # fall back when ``norm(v) < ratio * ref_norm``,
                        # i.e. the foot is near-vertical (en pointe / mid
                        # kick) and its XY projection is unreliable.
                        n = np.linalg.norm(v, axis=1, keepdims=True)
                        fb = np.asarray(fallback, dtype=np.float32)
                        if fb.ndim == 1:
                            fb = np.broadcast_to(fb.reshape(1, 3), v.shape)
                        good = n > 1e-6
                        if ref_norm is not None and ratio > 0.0:
                            ref = np.asarray(ref_norm, dtype=np.float32).reshape(-1, 1)
                            good = good & (n > float(ratio) * ref)
                        return np.where(good, v / np.maximum(n, 1e-6), fb).astype(np.float32)

                    def _temporal_smooth(v, window=5):
                        # Replicate-edge box filter along axis 0.  Removes
                        # per-frame XY-projection noise on rigs where the
                        # toe joint is mostly below the ankle (BVH).
                        a = np.asarray(v, dtype=np.float32)
                        F = int(a.shape[0])
                        if F < 2 or window <= 1:
                            return a.copy()
                        pad = int(window) // 2
                        idx = np.arange(F)
                        out = np.zeros_like(a)
                        for k in range(-pad, pad + 1):
                            shifted = np.clip(idx + k, 0, F - 1)
                            out += a[shifted]
                        return (out / float(window)).astype(np.float32)

                    def _continuity_guard(fwd, body_h, max_deg=90.0):
                        # Replace any single-frame outlier (>``max_deg`` from
                        # its predecessor) with that frame's body heading.
                        # See :meth:`InteractionMeshPipeline._enforce_directional_continuity`.
                        out = np.asarray(fwd, dtype=np.float32).copy()
                        bh = np.asarray(body_h, dtype=np.float32)
                        if bh.shape != out.shape:
                            bh = np.broadcast_to(bh.reshape(-1, 3), out.shape)
                        thr = float(np.cos(np.deg2rad(max_deg)))
                        for t in range(1, out.shape[0]):
                            if float(out[t] @ out[t - 1]) < thr:
                                out[t] = bh[t]
                        return out

                    def _global_sign(v, ref):
                        # One global sign decision so ``v`` agrees with
                        # ``ref`` on average — robust to the single-frame
                        # outliers that the previous per-frame
                        # ``_align_to_ref`` + ``_continuous`` pair would
                        # propagate, locking the toe / heel direction to
                        # frame 0's heading whenever the foot lifted
                        # mid-rotation.
                        v_arr = np.asarray(v, dtype=np.float32)
                        r_arr = np.asarray(ref, dtype=np.float32)
                        if v_arr.shape != r_arr.shape:
                            r_arr = np.broadcast_to(r_arr.reshape(1, 3), v_arr.shape)
                        dots = np.sum(v_arr * r_arr, axis=1)
                        med = float(np.median(dots)) if dots.size > 0 else 1.0
                        return -1.0 if med < 0.0 else 1.0

                    def _canonical_indices(names, src_map):
                        res: dict[str, int] = {}
                        priority = {
                            "left_foot": ("toeend", "toe_end", "toebase", "toe", "footmod", "foot"),
                            "right_foot": ("toeend", "toe_end", "toebase", "toe", "footmod", "foot"),
                            "left_wrist": ("hand", "wrist"),
                            "right_wrist": ("hand", "wrist"),
                        }
                        for ii, nn in enumerate(names):
                            cc = src_map.get(nn, nn)
                            res.setdefault(cc, ii)
                        for cc, words in priority.items():
                            best = None
                            for ii, nn in enumerate(names):
                                if src_map.get(nn, nn) != cc:
                                    continue
                                low = nn.lower()
                                rank = next((r for r, w in enumerate(words) if w in low), len(words))
                                if best is None or rank < best[0]:
                                    best = (rank, ii)
                            if best is not None:
                                res[cc] = best[1]
                        return res

                    def _body_heading(pos, cidx):
                        # Per-frame body forward from the hip (or shoulder)
                        # lateral cross product with world-up.  Earlier
                        # versions blended a velocity-based estimate that
                        # introduced threshold-induced discontinuities;
                        # the contact-offset pipeline then propagated
                        # those discontinuities through ``_continuous``,
                        # locking the toe direction across the rest of
                        # the clip.  Hip-lateral × up is smooth across
                        # in-place rotations and follows the same
                        # convention as
                        # :func:`hhtools.retarget.calibration.calibration._forward_from_shoulder_axis`,
                        # which is what ``source_body_quat`` was solved
                        # against.  See the matching note on
                        # :meth:`InteractionMeshPipeline._body_heading_forward`.
                        nf = int(pos.shape[0])
                        fb_init = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                        li = cidx.get("left_hip")
                        ri = cidx.get("right_hip")
                        if li is None or ri is None:
                            li = cidx.get("left_shoulder")
                            ri = cidx.get("right_shoulder")
                        if li is None or ri is None:
                            return np.broadcast_to(
                                fb_init.reshape(1, 3), (nf, 3),
                            ).copy()
                        lat = pos[:, li, :] - pos[:, ri, :]
                        lat[:, 2] = 0.0
                        upv = np.broadcast_to(
                            np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (nf, 3),
                        )
                        return _norm_vec(np.cross(lat, upv).astype(np.float32), fb_init)

                    def _contact_offset(sem, off, pos, anchor_i, cidx):
                        nf = int(pos.shape[0])
                        upv = np.broadcast_to(
                            np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (nf, 3),
                        )
                        sem = sem.lower()
                        if "ankle:toe" in sem or "ankle:heel" in sem:
                            side = "left" if "left_" in sem else "right"
                            foot_i = cidx.get(f"{side}_foot")
                            body_h = _body_heading(pos, cidx)
                            if foot_i is not None and foot_i != anchor_i:
                                raw3d = pos[:, foot_i, :] - pos[:, anchor_i, :]
                                xy = raw3d.copy()
                                xy[:, 2] = 0.0
                                # Mirror the pipeline's two-stage denoise
                                # (temporal box filter + per-frame
                                # body-heading fallback) and the final
                                # continuity guard.  Without these the
                                # BVH-style rigs (LAFAN, Mixamo) flicker
                                # the toe direction by ±180° on
                                # adjacent frames whenever the foot is
                                # near-vertical.
                                xy_s = _temporal_smooth(xy, 5)
                                raw_norm = np.linalg.norm(raw3d, axis=1)
                                fwd = _norm_vec(xy_s, body_h, ref_norm=raw_norm, ratio=0.2)
                                fwd = _continuity_guard(fwd, body_h, 90.0)
                            else:
                                fwd = body_h
                            sign = _global_sign(fwd, body_h)
                            if sign < 0.0:
                                fwd = -fwd
                            side_axis = _norm_vec(
                                np.cross(upv, fwd).astype(np.float32),
                                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                            )
                            return (
                                fwd * float(off[0])
                                + side_axis * float(off[1])
                                + upv * float(off[2])
                            ).astype(np.float32)
                        if "wrist:hand_tip" in sem:
                            side = "left" if "left_" in sem else "right"
                            hand_i = cidx.get(f"{side}_hand", cidx.get(f"{side}_wrist"))
                            elbow_i = cidx.get(f"{side}_elbow")
                            body_h = _body_heading(pos, cidx)
                            if hand_i is not None and hand_i != anchor_i:
                                raw = pos[:, hand_i, :] - pos[:, anchor_i, :]
                            elif elbow_i is not None:
                                raw = pos[:, anchor_i, :] - pos[:, elbow_i, :]
                            else:
                                raw = body_h
                            fwd = _norm_vec(raw, body_h)
                            return (fwd * float(np.linalg.norm(off))).astype(np.float32)
                        return None

                    cidx = _canonical_indices(list(jn), src2can)
                    extra_pos: list[NDArray[np.float32]] = []
                    extra_quat: list[NDArray[np.float32]] = []
                    extra_src: list[int] = []
                    extra_dst: list[int] = []
                    for pt in c_points:
                        off = np.asarray(pt.local_offset, dtype=np.float32).reshape(3)
                        if np.linalg.norm(off) <= 1e-8:
                            continue
                        src_i = int(pt.source_index)
                        if src_i < 0 or src_i >= len(clip.hierarchy.bone_names):
                            continue
                        src_name = clip.hierarchy.bone_names[src_i]
                        anchor_i = scaler_name_to_idx.get(src_name)
                        if anchor_i is None:
                            continue
                        q_anchor = quat_m[:, anchor_i, :]
                        offset_w = _contact_offset(
                            str(getattr(pt, "semantic", "")), off, pos_m, anchor_i, cidx,
                        )
                        if offset_w is None:
                            off_bc = np.broadcast_to(off[None, :], (F, 3))
                            offset_w = _Q.rotate(q_anchor, off_bc)
                        p_extra = pos_m[:, anchor_i, :] + offset_w.astype(np.float32)
                        extra_src.append(anchor_i)
                        extra_dst.append(M + len(extra_pos))
                        extra_pos.append(p_extra.astype(np.float32))
                        extra_quat.append(q_anchor.astype(np.float32))
                    if extra_pos:
                        hide_idx = {
                            int(idx)
                            for canon, idx in cidx.items()
                            if canon in {"left_foot", "right_foot"} and int(idx) < M
                        }
                        if hide_idx:
                            keep_seg = np.asarray(
                                [
                                    int(s) not in hide_idx and int(d) not in hide_idx
                                    for s, d in zip(seg_s, seg_d, strict=False)
                                ],
                                dtype=bool,
                            )
                            seg_s = seg_s[keep_seg]
                            seg_d = seg_d[keep_seg]
                            bead_idx = np.asarray(
                                [int(i) for i in bead_idx if int(i) not in hide_idx],
                                dtype=np.int32,
                            )
                        pos_m = np.concatenate(
                            [pos_m, np.stack(extra_pos, axis=1)], axis=1,
                        )
                        quat_m = np.concatenate(
                            [quat_m, np.stack(extra_quat, axis=1)], axis=1,
                        )
                        seg_s = np.concatenate(
                            [seg_s, np.asarray(extra_src, dtype=np.int32)], axis=0,
                        )
                        seg_d = np.concatenate(
                            [seg_d, np.asarray(extra_dst, dtype=np.int32)], axis=0,
                        )
                        bead_idx = np.concatenate(
                            [
                                bead_idx,
                                np.asarray(extra_dst, dtype=np.int32),
                            ],
                            axis=0,
                        )
                        M = int(pos_m.shape[1])
                except Exception:
                    pass

                # Re-ground after contact offsets so the yellow overlay
                # always meets z=0 at ankle hubs, even when auxiliary toe /
                # sole points sit below or above the drawn skeleton.
                z_floor = foot_floor_z_in_positions(pos_m, tuple(jn))
                if abs(z_floor) > 1e-6:
                    pos_m[:, :, 2] -= np.float32(z_floor)

                src_tf = np.concatenate(
                    [pos_m.astype(np.float32), quat_m.astype(np.float32)],
                    axis=-1,
                )

                return ScaledMotionPreview(
                    joint_names=ik_names,
                    transforms=out,
                    source_joint_names=jn,
                    source_seg_src=seg_s,
                    source_seg_dst=seg_d,
                    source_transforms=src_tf,
                    source_bead_indices=bead_idx,
                )
        return ScaledMotionPreview(joint_names=ik_names, transforms=out)

    def _publish_scaled_preview(preview) -> None:
        """Install ``preview`` as the live scaled-skeleton renderer.

        Idempotent: tears down any previous renderer before adding a new
        one so repeat calls don't accumulate handles in the scene.
        ``preview=None`` just clears the existing renderer (used when
        the calibration is missing — the UI has no meaningful scaled
        skeleton to show).
        """
        _clear_scaled_preview()
        # Always stash the preview — the render loop reads it back for
        # frame-cursor driving, and ``_on_show_scaled`` relies on it to
        # re-materialise the renderer on toggle.
        robot_state["scaled_preview"] = preview
        if preview is None:
            return
        if not retarget_show_scaled.value:
            return
        try:
            renderer = ScaledSkeletonRenderer(
                server, preview, world_offset=ROBOT_WORLD_OFFSET,
            )
        except Exception as err:  # pragma: no cover — surfaced at runtime
            _notify_all(
                server, "Scaled preview failed",
                f"{type(err).__name__}: {err}", color="orange",
            )
            return
        state["scaled_renderer"] = renderer
        robot_state["scaled_renderer"] = renderer

    @retarget_show_scaled.on_update
    def _on_show_scaled(_):  # type: ignore[no-untyped-def]
        """Toggle the scaled-skeleton renderer without recomputing the preview."""
        preview = robot_state.get("scaled_preview")
        if retarget_show_scaled.value and preview is not None:
            sr = state.get("scaled_renderer")
            if sr is not None:
                try:
                    sr.clear()
                except Exception:
                    pass
                state["scaled_renderer"] = None
                robot_state["scaled_renderer"] = None
            try:
                renderer = ScaledSkeletonRenderer(
                    server, preview, world_offset=ROBOT_WORLD_OFFSET,
                )
            except Exception:
                return
            state["scaled_renderer"] = renderer
            robot_state["scaled_renderer"] = renderer
        else:
            # Keep the preview around so re-toggling ON doesn't require the
            # user to click Preview again.
            sr = state.get("scaled_renderer")
            if sr is not None:
                try:
                    sr.clear()
                except Exception:
                    pass
            state["scaled_renderer"] = None
            robot_state["scaled_renderer"] = None

    @retarget_preview_btn.on_click
    def _on_preview_scaled(_):  # type: ignore[no-untyped-def]
        """Scaler-only preview: draws the pre-IK skeleton, no Newton needed."""
        if calib_state.get("active"):
            _notify_all(
                server, "Calibration active",
                "Save or cancel calibration before scaled preview.",
                color="orange",
            )
            return
        model = state.get("current_model")
        if not isinstance(model, URDFRobotModel):
            _notify_all(
                server, "No robot loaded",
                "Click Load first so we know which ik_map to preview against.",
                color="orange",
            )
            return
        if get_current_motion is None or get_current_motion() is None:
            _notify_all(
                server, "No motion selected",
                "Pick a clip in the Motion tab first.", color="orange",
            )
            return
        motion_obj = get_current_motion()
        _check_rig_type_switch(motion_obj)

        def _worker() -> None:
            from hhtools.retarget.newton_basic.human_aliases import (
                list_detected_rig_type,
            )
            try:
                clip = _slice_motion(motion_obj, int(retarget_frames.value or 0))
                if clip.up_axis != "Z":
                    clip = to_up_axis(clip, "Z")
                sc = _build_scaler_config_or_warn(model, clip, 1.7)
                if sc is not None:
                    human_h = float(sc.human_height_assumption)
                    retarget_human_height.value = round(human_h, 3)
                else:
                    human_h = float(retarget_human_height.value)
                rig_label = _html_escape(
                    list_detected_rig_type(clip.hierarchy.bone_names)
                )
                preview = _compute_scaled_preview(model, clip, human_h)
                _publish_scaled_preview(preview)
                if sc is not None:
                    _publish_robot_objects(sc, human_h)
                retarget_status.content = (
                    f"<span style='color:{PALETTE.ui_ok}'>✓</span> "
                    f"Scaled preview · {preview.num_frames} frames · "
                    f"{len(preview.joint_names)} ik_map joints · "
                    f"rig: <b>{rig_label}</b> · "
                    f"height assumption: {human_h:.3f}m"
                )
            except Exception as err:  # noqa: BLE001
                retarget_status.content = (
                    f"<span style='color:{PALETTE.ui_error}'>Scaled preview failed:</span> "
                    f"<code>{_html_escape(f'{type(err).__name__}: {err}')}</code>"
                )

        threading.Thread(
            target=_worker, name="hhtools-scaled-preview", daemon=True,
        ).start()

    def _resolve_batch_targets(model):
        """Return a list of ``(label, motion, stem)`` tuples to retarget.

        "Current motion" resolves to a single-element list using whatever
        the Motion tab has loaded.  "All filtered motions" iterates over
        the library's filtered view and pulls each motion through the
        cache so the Motion tab's state stays untouched.

        Returns ``None`` when the user's selection is empty or the Robot
        tab is missing the hooks needed to resolve it (callers treat this
        as a silent no-op after surfacing an explanatory toast).
        """
        scope = retarget_scope.value
        if scope == "Current motion":
            if get_current_motion is None or get_current_motion() is None:
                _notify_all(
                    server, "No motion selected",
                    "Pick a clip in the Motion tab first.", color="orange",
                )
                return None
            motion_obj = get_current_motion()
            stem = get_current_entry_id() if get_current_entry_id else None
            stem = stem or (motion_obj.name if motion_obj else "motion")
            cur_le: LibraryEntry | None = None
            if motion_sync is not None:
                entry_get = motion_sync.get("get_current_library_entry")
                if callable(entry_get):
                    le = entry_get()
                    if isinstance(le, LibraryEntry):
                        cur_le = le
            return [(stem, motion_obj, stem, cur_le)]

        # Batch mode.
        if get_filtered_entries is None or motion_cache is None:
            _notify_all(
                server, "Batch unavailable",
                "Library access not wired in this viewer build.",
                color="red",
            )
            return None
        entries = get_filtered_entries()
        if not entries:
            _notify_all(
                server, "Filter is empty",
                "No motions in the Motion-tab dropdown.  Adjust search / folder.",
                color="orange",
            )
            return None
        return [
            (entry.sequence_id, None, entry.sequence_id, entry)
            for entry in entries
        ]

    # Inject the few ``run_viewer``-scope helpers the save handler
    # needs.  ``_build_robot_tab`` lives at module scope so it does
    # NOT close over ``run_viewer``'s locals — passing them in
    # explicitly is the only way to reach ``source_root_path`` /
    # ``save_dir_path`` / the async dispatcher.  Fallbacks keep the
    # button useful even if a future caller of ``_build_robot_tab``
    # forgets one of the kwargs (e.g. saves still land under
    # ``./assets/save_npz`` and the worker falls back to a
    # synchronous run with a fresh lock).
    _src_root: Path = (
        Path(source_root_path) if source_root_path is not None else Path("assets/motions")
    )
    _save_root: Path = (
        Path(save_dir_path) if save_dir_path is not None else Path("assets/save_npz")
    )
    _dispatch = run_async if callable(run_async) else (
        lambda name, fn: threading.Thread(target=fn, name=name, daemon=True).start()
    )
    _save_lock = work_lock if work_lock is not None else threading.Lock()

    @save_robot_clip_btn.on_click
    def _on_save_robot_clip(_):  # type: ignore[no-untyped-def]
        """Persist the most recent retarget result + scaled scene as pkls.

        Output layout mirrors the source clip's path within
        ``source_root`` so a ``meshmimic/parc_ms/<clip>/foo.npz``
        source produces
        ``<save_dir>/meshmimic/parc_ms/<clip>/{robot,object_*,terrain}.pkl``.
        Falls back to ``<dataset>/<folder>/<stem>/`` when the source
        path can't be made relative (uploads, ad-hoc files).
        """
        retargeted = robot_state.get("retargeted")
        source_motion = robot_state.get("retargeted_source_motion")
        lib_entry = robot_state.get("retargeted_lib_entry")
        if retargeted is None or source_motion is None:
            _notify_all(
                server, "Nothing to save",
                "Run Retarget first; the save button persists the most "
                "recent retarget result.",
                color="yellow",
            )
            return

        # Resolve the destination directory from the library entry.
        # Mirroring the source path under ``source_root`` keeps the
        # exported pkls discoverable next to where the user picked
        # the clip in the library tree.
        if isinstance(lib_entry, LibraryEntry):
            stem = lib_entry.stem
            try:
                rel = lib_entry.source_path.resolve().relative_to(
                    _src_root.resolve()
                )
                dest_dir = _save_root / rel.parent
            except (ValueError, OSError):
                dest_dir = (
                    _save_root
                    / lib_entry.dataset
                    / lib_entry.folder_label
                    / stem
                )
        else:
            stem = str(getattr(retargeted, "name", "retargeted"))
            dest_dir = _save_root / "retargeted" / stem

        def _worker() -> None:
            try:
                from hhtools.io.parc_export import save_robot_clip_pkls

                with _save_lock:
                    written = save_robot_clip_pkls(
                        retargeted, source_motion, dest_dir,
                    )
                rel_root = _relative_to_repo(dest_dir)
                _notify_all(
                    server,
                    "Saved robot clip",
                    f"{len(written)} pkl → {rel_root}",
                    color="green",
                )
                retarget_status.content = (
                    f"<span style='color:{PALETTE.ui_ok}'>✓</span> "
                    f"Robot clip saved → "
                    f"<code>{_html_escape(str(rel_root))}/</code> "
                    f"({len(written)} pkl: "
                    f"{', '.join(written.keys())})"
                )
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                _notify_all(
                    server, "Save robot clip failed",
                    f"{type(exc).__name__}: {exc}",
                    color="red",
                )

        _dispatch(f"hhtools-save-robot-{stem}", _worker)

    @retarget_btn.on_click
    def _on_retarget(_):  # type: ignore[no-untyped-def]
        if calib_state.get("active"):
            _notify_all(
                server, "Calibration active",
                "Save or cancel calibration before retargeting.",
                color="orange",
            )
            return
        model = state.get("current_model")
        if not isinstance(model, URDFRobotModel):
            _notify_all(
                server, "No robot loaded",
                "Click Load first so we know which robot to target.",
                color="orange",
            )
            return
        if model.preset.urdf_path is not None and model.preset.ik_map:
            from hhtools.robot.kinematics import validate_ik_map

            ik_issues = validate_ik_map(
                model.preset.urdf_path, dict(model.preset.ik_map),
            )
            if ik_issues:
                _notify_all(
                    server,
                    "ik_map invalid — retarget blocked",
                    "\n".join(issue.format() for issue in ik_issues[:4])
                    + (
                        f"\n… and {len(ik_issues) - 4} more"
                        if len(ik_issues) > 4 else ""
                    )
                    + f"\nRun: hhtools robot validate {model.preset.name} --fix",
                    color="red",
                )
                return
        if get_current_motion is None:
            _notify_all(
                server, "Motion unavailable",
                "Robot tab can't reach the Motion tab in this build.",
                color="red",
            )
            return

        batch = _resolve_batch_targets(model)
        if batch is None:
            return

        # Check rig type switch — warn user if data source changed.
        motion_check = get_current_motion() if get_current_motion else None
        if motion_check is not None:
            _check_rig_type_switch(motion_check)

        # Resolve scope eagerly so the worker sees a stable list even if
        # the user changes the dropdown mid-run.
        scope_is_batch = retarget_scope.value == "All filtered motions"
        _first_clip_for_h = batch[0][1] if batch else None
        if _first_clip_for_h is not None:
            _height_clip = (
                to_up_axis(_first_clip_for_h, "Z")
                if _first_clip_for_h.up_axis != "Z"
                else _first_clip_for_h
            )
            _sc_tmp = _build_scaler_config_or_warn(
                model, _height_clip, 1.7,
            )
            if _sc_tmp is not None:
                retarget_human_height.value = round(
                    float(_sc_tmp.human_height_assumption), 3,
                )
        human_h = float(retarget_human_height.value)
        max_frames = int(retarget_frames.value or 0)
        ik_iters = int(retarget_iters.value)

        _rl = robot_state.get("_retarget_lock")
        if isinstance(_rl, threading.Lock) and not _rl.acquire(blocking=False):
            _notify_all(
                server, "Retarget in progress",
                "Wait for the current run to finish before starting another.",
                color="orange",
            )
            return

        retarget_btn.disabled = True
        retarget_preview_btn.disabled = True
        # Drive the bar with explicit ``set_message(..., value=…)`` only — no
        # synthetic exponential ticker.  ``pin_milestone`` + ticker resets made
        # the fill appear to "run again" on every milestone.
        retarget_progress.start(
            f"Retargeting {len(batch)} clip(s) to "
            f"{_html_escape(model.preset.name)}",
            indeterminate=False,
        )

        def _worker() -> None:
            from hhtools.io.robot_csv import save_robot_csv
            from hhtools.retarget.calibration import resolve_calibration_file
            from hhtools.retarget.interaction_mesh import (
                InteractionMeshPipeline,
                InteractionMeshPipelineConfig,
            )
            from hhtools.retarget.newton_basic import (
                NewtonBasicPipeline,
                PipelineConfig,
            )
            from hhtools.retarget.newton_basic._warp_config import (
                configure as configure_warp_cache,
            )
            from hhtools.retarget.newton_basic.human_aliases import (
                list_detected_rig_type,
            )

            completed_ok = False
            done_message: str | None = None
            try:
                configure_warp_cache()
                _await_robot_prewarm()
                _pct_hist = [0.0]

                def _retarget_bump_pct(p: float) -> float:
                    p = float(max(0.0, min(99.0, p)))
                    if p < _pct_hist[0]:
                        return _pct_hist[0]
                    _pct_hist[0] = p
                    return p

                out_dir = Path(tempfile.gettempdir())
                backend_choice = str(retarget_backend.value)

                # ---- Phase 1: load all motions (CPU) -----------------------
                retarget_progress.set_message(
                    "Loading motions…", value=_retarget_bump_pct(1.0),
                )
                clips: list = []
                stems: list[str] = []
                labels: list[str] = []
                clip_entries: list[LibraryEntry | None] = []
                for label_raw, preloaded, stem, lib_entry in batch:
                    if preloaded is None:
                        try:
                            entry: LibraryEntry | None = (
                                lib_entry if isinstance(lib_entry, LibraryEntry) else None
                            )
                            if entry is None:
                                entry = next(
                                    (e for e in get_filtered_entries()
                                     if e.sequence_id == stem),
                                    None,
                                )
                            if entry is None:
                                continue
                            motion_in = motion_cache.load_motion(entry)
                        except Exception as err:
                            retarget_progress.set_message(
                                f"skip {_html_escape(stem)}: "
                                f"{_html_escape(f'{type(err).__name__}: {err}')}"
                            )
                            continue
                    else:
                        motion_in = preloaded
                    if callable(apply_motion_pipeline):
                        motion_in = apply_motion_pipeline(motion_in)
                    if motion_in.up_axis != "Z":
                        motion_in = to_up_axis(motion_in, "Z")
                    clips.append(_slice_motion(motion_in, max_frames))
                    stems.append(stem)
                    labels.append(label_raw)
                    clip_entries.append(
                        lib_entry if isinstance(lib_entry, LibraryEntry) else None
                    )

                if not clips:
                    retarget_status.content = (
                        f"<span style='color:{PALETTE.ui_error}'>Nothing was retargeted.</span>"
                    )
                    return

                # ---- Phase 1b: detect rig type --------------------------------
                rig_label = _html_escape(
                    list_detected_rig_type(clips[0].hierarchy.bone_names)
                )
                backends = [
                    _effective_retarget_backend(backend_choice, ent, clips[i])
                    for i, ent in enumerate(clip_entries)
                ]
                retarget_progress.set_message(
                    f"Detected rig: <b>{rig_label}</b> · "
                    f"{len(clips)} clip(s)",
                    value=_retarget_bump_pct(4.0),
                )

                # ---- Phase 2: build ScalerConfig (same for all clips) ------
                scaler_cfg = _build_scaler_config_or_warn(
                    model, clips[0], human_h,
                )
                if scaler_cfg is None:
                    _notify_all(
                        server, "Robot not calibrated",
                        "Open the Robot tab and click 'Start calibration' "
                        "first.",
                        color="orange",
                    )
                    return

                preset = model.preset
                ref_name = str(calib_reference_picker.value)
                from hhtools.robot.retarget_profile import (
                    build_feet_stabilizer_config,
                    build_pipeline_config_for_preset,
                )

                pipeline_cfg = build_pipeline_config_for_preset(
                    preset, ref_name, ik_iterations=ik_iters,
                )
                feet_cfg = build_feet_stabilizer_config(
                    preset, ref_name, model=model,
                )
                cal_path_str: str | None = None
                if preset.urdf_path is not None:
                    cr = resolve_calibration_file(
                        preset.urdf_path.parent,
                        ref_name,
                    )
                    if cr is not None and cr.is_file():
                        cal_path_str = str(cr)
                if any(b == "interaction_mesh" for b in backends) and cal_path_str is None:
                    _notify_all(
                        server, "Calibration path missing for interaction mesh",
                        "Falling back to Newton IK for all clips.",
                        color="orange",
                    )
                    backends = ["newton"] * len(backends)

                # ---- Phase 3: single or batch IK --------------------------
                total_clips = len(clips)
                use_newton_batch = (
                    scope_is_batch
                    and total_clips > 1
                    and all(b == "newton" for b in backends)
                )
                results: list = []
                result_stems: list[str] = []

                if use_newton_batch:
                    pipeline = NewtonBasicPipeline(
                        model,
                        scaler_config=scaler_cfg,
                        pipeline_config=pipeline_cfg,
                        feet_stabilizer_config=feet_cfg,
                        human_height=human_h,
                        configure_warp=False,
                    )

                    def _batch_frame_cb(done: int, total: int) -> None:
                        t = max(1, int(total))
                        d = min(max(0, int(done)), t)
                        pct = _retarget_bump_pct(min(94.0, 5.0 + 89.0 * d / t))
                        retarget_progress.set_message(
                            f"Solving IK (GPU ×{total_clips}) · frame {d}/{t}",
                            value=pct,
                        )

                    retarget_progress.set_message(
                        f"Building multi-env model ({total_clips} envs)",
                        value=_retarget_bump_pct(3.0),
                    )
                    results = pipeline.run_batch(
                        clips, progress_callback=_batch_frame_cb,
                    )
                    result_stems = stems[: len(results)]
                    _publish_robot_objects(scaler_cfg, human_h)
                else:
                    if not scope_is_batch:
                        try:
                            preview = _compute_scaled_preview(
                                model, clips[0], human_h,
                            )
                            _publish_scaled_preview(preview)
                        except Exception:
                            pass
                    _publish_robot_objects(scaler_cfg, human_h)

                    for i, clip in enumerate(clips):
                        be = backends[i]
                        stem_i = stems[i] if i < len(stems) else "motion"
                        esc_stem = _html_escape(stem_i)

                        if be == "interaction_mesh":
                            assert cal_path_str is not None
                            _im_ok = False
                            try:
                                retarget_progress.set_message(
                                    f"Interaction mesh · <b>{esc_stem}</b> · "
                                    f"clip {i + 1}/{total_clips}…",
                                    value=_retarget_bump_pct(
                                        min(88.0, 5.0 + 85.0 * i / max(1, total_clips)),
                                    ),
                                )
                                im_pipe = InteractionMeshPipeline.from_calibration(
                                    model,
                                    clip,
                                    cal_path_str,
                                    human_height=human_h,
                                    cfg=InteractionMeshPipelineConfig(
                                        sqp_inner_iters=max(1, ik_iters),
                                    ),
                                )

                                def _im_prog(phase: str, cur: int, tot: int) -> None:
                                    t = max(1, int(tot))
                                    c = min(max(0, int(cur)), t)
                                    lo_c = 5.0 + 85.0 * i / max(1, total_clips)
                                    hi_c = 5.0 + 85.0 * (i + 1) / max(1, total_clips)
                                    band_c = max(hi_c - lo_c, 1e-6)
                                    if phase == "precompute":
                                        raw_c = lo_c + 0.42 * band_c * c / t
                                    else:
                                        raw_c = lo_c + 0.42 * band_c + 0.58 * band_c * c / t
                                    pct_c = _retarget_bump_pct(min(97.0, raw_c))
                                    retarget_progress.set_message(
                                        f"Interaction mesh · {esc_stem} · "
                                        f"{'precompute' if phase == 'precompute' else 'MPC'} "
                                        f"{c}/{t}",
                                        value=pct_c,
                                    )

                                results.append(
                                    im_pipe.run(clip, progress_callback=_im_prog),
                                )
                                result_stems.append(stem_i)
                                _im_ok = True
                                retarget_progress.set_message(
                                    f"Interaction mesh · clip {i + 1}/{total_clips} done",
                                    value=_retarget_bump_pct(
                                        min(92.0, 5.0 + 85.0 * (i + 1) / max(1, total_clips)),
                                    ),
                                )
                            except Exception as im_err:  # noqa: BLE001
                                _notify_all(
                                    server,
                                    "Interaction mesh failed — falling back to Newton IK",
                                    f"{type(im_err).__name__}: {im_err}",
                                    color="orange",
                                )
                                retarget_progress.set_message(
                                    f"Interaction mesh failed for {esc_stem}, "
                                    f"falling back to Newton IK…",
                                    value=_retarget_bump_pct(
                                        min(88.0, 5.0 + 85.0 * i / max(1, total_clips)),
                                    ),
                                )
                            if not _im_ok:
                                be = "newton"
                        if be != "interaction_mesh":
                            sc = _build_scaler_config_or_warn(model, clip, human_h)
                            if sc is None:
                                continue
                            newton_pipe = NewtonBasicPipeline(
                                model,
                                scaler_config=sc,
                                pipeline_config=pipeline_cfg,
                                feet_stabilizer_config=feet_cfg,
                                human_height=human_h,
                                configure_warp=False,
                            )
                            nf = max(1, int(clip.num_frames))

                            def _single_frame_cb(
                                done: int,
                                total: int,
                                *,
                                _i: int = i,
                                _n: int = total_clips,
                                _nf: int = nf,
                            ) -> None:
                                tot = max(1, int(total))
                                d = min(max(0, int(done)), tot)
                                frac_clip = float(d) / float(tot)
                                overall = (_i + frac_clip) / float(max(1, _n))
                                pct = _retarget_bump_pct(min(94.0, 5.0 + 89.0 * overall))
                                retarget_progress.set_message(
                                    f"Newton IK · clip {_i + 1}/{_n} · frame {d}/{tot}",
                                    value=pct,
                                )

                            results.append(
                                newton_pipe.run(
                                    clip, progress_callback=_single_frame_cb,
                                )
                            )
                            result_stems.append(stem_i)

                # ---- Phase 4: export CSVs + update UI ----------------------
                exported: list[Path] = []
                last_result = None
                for i, result in enumerate(results):
                    if result.num_frames == 0:
                        continue
                    last_result = result
                    stem = (
                        result_stems[i]
                        if i < len(result_stems)
                        else (stems[i] if i < len(stems) else "motion")
                    )
                    out_path = out_dir / (
                        f"hhtools_{model.preset.name}_"
                        f"{(stem or 'motion').replace('/', '_')}.csv"
                    )
                    save_robot_csv(
                        out_path,
                        robot=model,
                        joint_q=result.joint_q,
                        sample_rate=result.sample_rate,
                        meta={
                            "source_motion": result.name,
                            **{k: str(v) for k, v in result.meta.items()},
                        },
                    )
                    exported.append(out_path)

                    # Also drop a hhtools-schema NPZ + sidecar PARC
                    # pkl pair next to the CSV so users can reload
                    # the retargeted clip in any of our viewers /
                    # consume the heightfield in PARC training,
                    # matching the export spec the user asked for
                    # ("机器人的数据 ↔ NPZ … 地形 ↔ PKL").  We only do
                    # this for the interaction-mesh backend (the
                    # only one that runs against a terrain-aware
                    # collision model and produces a populated
                    # ``smpl_scale`` / ``source_z_min`` in
                    # ``result.meta``).  Failures are non-fatal:
                    # CSV is the contractual artefact, NPZ/PKL are
                    # convenience output.
                    try:
                        from hhtools.io.parc_export import (
                            save_retargeted_motion_npz,
                        )

                        npz_out = out_path.with_suffix(".npz")
                        # ``clips`` is the list of source ``Motion``
                        # objects that fed this batch (one per clip,
                        # zip-aligned with ``results`` / ``stems``);
                        # ``clips[i].terrain`` is the original
                        # heightfield that gets re-emitted as the
                        # sidecar PKL after being re-scaled into
                        # the robot frame inside
                        # ``save_retargeted_motion_npz``.
                        save_retargeted_motion_npz(
                            result, clips[i], model, npz_out,
                        )
                    except Exception as nerr:  # noqa: BLE001
                        import traceback
                        traceback.print_exc()
                        _notify_all(
                            server, "NPZ/PKL export failed",
                            f"{type(nerr).__name__}: {nerr}",
                            color="orange",
                        )

                if last_result is not None:
                    robot_state["retargeted"] = last_result
                    # Track everything the "Save robot clip" button
                    # needs to materialise the per-clip pkl bundle:
                    # the source ``Motion`` (terrain + objects), the
                    # library entry (for the dataset/folder/clip
                    # path layout), and the robot model (for FK +
                    # link names).  We store **only the last
                    # successfully retargeted clip** since the save
                    # button is per-clip — batch saves should pipe
                    # through the dedicated "Save whole folder"
                    # path which the user can wire up later.
                    last_idx = len(results) - 1
                    while last_idx >= 0 and results[last_idx].num_frames == 0:
                        last_idx -= 1
                    if 0 <= last_idx < len(clips):
                        robot_state["retargeted_source_motion"] = clips[last_idx]
                        robot_state["retargeted_lib_entry"] = (
                            clip_entries[last_idx]
                            if last_idx < len(clip_entries) else None
                        )
                        robot_state["retargeted_model"] = model
                    try:
                        save_robot_clip_btn.disabled = False
                    except Exception:
                        pass
                    panel_ = get_panel() if get_panel is not None else None
                    if panel_ is not None and last_result.num_frames > 0:
                        try:
                            panel_.set_motion(
                                framerate=float(last_result.sample_rate),
                                num_frames=int(last_result.num_frames),
                                resume_playing=True,
                            )
                        except Exception as err:
                            import traceback

                            traceback.print_exc()
                            _notify_all(
                                server,
                                "Playback update failed",
                                f"{type(err).__name__}: {err}",
                                color="orange",
                            )

                if not exported:
                    retarget_status.content = (
                        f"<span style='color:{PALETTE.ui_error}'>Nothing was retargeted.</span>"
                    )
                    return

                completed_ok = True
                done_message = (
                    f"Exported {len(exported)} CSV · retarget finished"
                )
                if scope_is_batch:
                    head = exported[0]
                    batch_note = (
                        "GPU-parallel Newton · "
                        if use_newton_batch
                        else "per-clip solvers · "
                    )
                    retarget_status.content = (
                        f"<span style='color:{PALETTE.ui_ok}'>✓</span> "
                        f"Retargeted <b>{len(exported)}</b> clip(s) · "
                        f"{batch_note}"
                        f"rig: <b>{rig_label}</b> · "
                        f"last result playing now · "
                        f"<code>{_html_escape(head.parent)}/</code> "
                        f"(see {_html_escape(head.name)}, …)"
                    )
                    _notify_all(
                        server, f"Retargeted × {len(exported)}",
                        f"CSV dir: {head.parent}", color="green",
                    )
                else:
                    path = exported[0]
                    assert last_result is not None
                    retarget_status.content = (
                        f"<span style='color:{PALETTE.ui_ok}'>✓</span> "
                        f"Retargeted {last_result.num_frames} frames · "
                        f"{len(last_result.dof_names)} DOF · "
                        f"rig: <b>{rig_label}</b> → "
                        f"<code>{_html_escape(path)}</code>"
                    )
                    _notify_all(
                        server, "Retargeted",
                        f"{last_result.num_frames} frames → {path}",
                        color="green",
                    )
            except Exception as err:  # noqa: BLE001
                retarget_status.content = (
                    f"<span style='color:{PALETTE.ui_error}'>Retarget failed:</span> "
                    f"<code>{_html_escape(f'{type(err).__name__}: {err}')}</code>"
                )
                _notify_all(
                    server, "Retarget failed",
                    f"{type(err).__name__}: {err}",
                    color="red",
                )
            finally:
                retarget_progress.done(
                    success=completed_ok,
                    last_message=done_message,
                )
                try:
                    rl = robot_state.get("_retarget_lock")
                    if isinstance(rl, threading.Lock):
                        rl.release()
                except RuntimeError:
                    pass
                if not calib_state.get("active"):
                    retarget_btn.disabled = False
                    retarget_preview_btn.disabled = False
                else:
                    retarget_btn.disabled = True
                    retarget_preview_btn.disabled = True

        threading.Thread(
            target=_worker,
            name=f"hhtools-retarget-{model.preset.name}",
            daemon=True,
        ).start()

    # -------------------- Retarget calibration callbacks --------------------

    def _do_save_calibration() -> None:
        """Gather the session's slider state, derive + persist, teardown UI.

        Side-effects:
          * Writes ``retarget_calibration_<reference>.yaml`` next to the URDF.
          * Mirrors the closed-form scale/offset cache into the yaml for
            diffability (see :func:`save_calibration`).
          * Removes the session folder and clears the reference skeleton.
        """

        model = state.get("current_model")
        if not isinstance(model, URDFRobotModel):
            _notify_all(
                server, "No robot loaded",
                "Calibration save needs a loaded URDF.",
                color="orange",
            )
            return
        preset = model.preset
        if preset.urdf_path is None:
            _notify_all(
                server, "Save failed",
                "Robot preset has no URDF on disk; can't derive a save path.",
                color="red",
            )
            return

        # Only persist joints the user actually moved away from 0 — keeps
        # diffs minimal and makes the yaml easier to reason about at a
        # glance.  Missing joints default to 0 on load.
        non_zero = {
            n: float(v) for n, v in calib_state["current_q"].items()
            if abs(float(v)) > 1e-6
        }
        from hhtools.retarget.calibration import (
            RobotRetargetCalibration,
            calibration_path_for,
            derive_calibration_params,
            save_calibration,
        )

        cal = RobotRetargetCalibration(
            robot=preset.name,
            reference=calib_reference_picker.value,  # type: ignore[arg-type]
            calibrated_joint_q=non_zero,
            notes="Saved via viewer's Retarget calibration panel.",
        )

        # Derive the closed-form scale/offset cache against the live
        # URDF so the yaml captures "what retarget will actually use"
        # for this calibration.  Wrapped in a try so a pathological URDF
        # (missing meshes etc.) still lets the user save the joint_q
        # half of the calibration and re-derive later.
        derived = None
        try:
            motion_live = (
                get_current_motion() if callable(get_current_motion) else None
            )
            derived = derive_calibration_params(
                cal, model, reference_motion=motion_live,
            )
        except Exception as err:  # noqa: BLE001
            _notify_all(
                server, "Scale/offset derivation failed",
                f"{type(err).__name__}: {err}.  Saving joint angles only.",
                color="orange",
            )

        cal_path = calibration_path_for(
            preset.urdf_path.parent,
            reference=str(calib_reference_picker.value),
        )
        try:
            save_calibration(cal, cal_path, derived=derived)
        except Exception as err:  # noqa: BLE001
            _notify_all(
                server, "Save failed",
                f"{type(err).__name__}: {err}",
                color="red",
            )
            return

        yaml_path = preset.meta.get("yaml_path")
        if yaml_path and derived is not None:
            from hhtools.robot.joint_scales import (
                sync_joint_scale_multipliers_to_robot_yaml,
            )

            try:
                sync_joint_scale_multipliers_to_robot_yaml(
                    yaml_path,
                    derived.scales,
                    dict(preset.ik_map),
                )
            except Exception as err:  # noqa: BLE001
                _notify_all(
                    server, "Scale yaml sync failed",
                    f"{type(err).__name__}: {err}",
                    color="orange",
                )

        # --- Also persist the (possibly edited) ik_map to robot.yaml ---
        # We do this after the calibration yaml write so the user has a
        # recovery path if ik_map write fails: the calibration itself is
        # safe on disk, and they can retry or hand-edit the mapping.
        ik_map_edit_summary = _persist_ik_map_if_changed(model)

        n_scales = (
            len(derived.scales) if derived is not None else 0
        )
        _notify_all(
            server, "Calibration saved",
            (
                f"{cal_path}\n"
                f"  · {len(non_zero)} joint(s) non-zero · "
                f"ref {cal.reference} · "
                f"{n_scales} per-limb scale(s) cached"
                + (f"\n  · {ik_map_edit_summary}" if ik_map_edit_summary else "")
            ),
            color="green",
        )
        _exit_calibration_mode()

    def _persist_ik_map_if_changed(model: URDFRobotModel) -> str:
        """Write ``calib_state['ik_map']`` back to ``robot.yaml`` if edited.

        Returns a short human-readable summary of what changed (empty
        string if nothing changed — we skip the round-trip rewrite in
        that case to keep the yaml byte-identical on save).

        Side-effects:
            * Mutates ``model.preset.ik_map`` in place so subsequent
              retargets in this viewer session use the new mapping
              without a registry reload.
            * Rewrites ``preset.meta['yaml_path']`` via
              :func:`update_robot_yaml_ik_map` — preserves comments
              and key order thanks to ruamel round-tripping.
        """

        current = dict(calib_state["ik_map"])  # type: ignore[arg-type]
        previous = dict(model.preset.ik_map)
        if current == previous:
            return ""

        # Detect duplicate canonical keys — the collection step in
        # :func:`_update_ik_map` already deduplicated (last-write-wins),
        # but we want to warn the user if two robot links ended up
        # claiming the same canonical joint in the on-screen rows.
        dropdowns: dict[str, object] = calib_state["mapping_dropdowns"]  # type: ignore[assignment]
        picked_by_link: dict[str, str] = {
            link: str(getattr(w, "value", "")) for link, w in dropdowns.items()
        }
        from collections import Counter

        counts = Counter(v for v in picked_by_link.values() if v)
        duplicates = [c for c, n in counts.items() if n > 1]

        yaml_path = model.preset.meta.get("yaml_path") if model.preset.meta else None
        if not yaml_path:
            _notify_all(
                server, "ik_map write skipped",
                "Preset has no recorded yaml_path; mapping stays in-memory "
                "only.  Retarget calibration still saved.",
                color="orange",
            )
            model.preset.ik_map.clear()
            model.preset.ik_map.update(current)
            return "ik_map updated in memory only (no yaml path recorded)"

        from pathlib import Path as _Path

        from hhtools.robot.yaml_io import update_robot_yaml_ik_map

        try:
            update_robot_yaml_ik_map(_Path(yaml_path), current)
        except Exception as err:  # noqa: BLE001
            _notify_all(
                server, "ik_map write failed",
                f"{type(err).__name__}: {err}.  Mapping stays in-memory only.",
                color="red",
            )
            model.preset.ik_map.clear()
            model.preset.ik_map.update(current)
            return "ik_map updated in memory only (yaml write failed)"

        model.preset.ik_map.clear()
        model.preset.ik_map.update(current)

        added = set(current) - set(previous)
        removed = set(previous) - set(current)
        changed = {
            k for k in set(current) & set(previous)
            if current[k] != previous[k]
        }
        parts: list[str] = []
        if added:
            parts.append(f"added {sorted(added)}")
        if removed:
            parts.append(f"removed {sorted(removed)}")
        if changed:
            parts.append(f"rebound {sorted(changed)}")
        summary = "ik_map: " + ", ".join(parts) if parts else "ik_map: updated"

        if duplicates:
            _notify_all(
                server, "ik_map has duplicates",
                f"Multiple robot links map to {duplicates}.  Retarget will "
                "pick the last one written; rebind the extras to unique "
                "canonical joints.",
                color="orange",
            )
            summary += f" (warning: duplicates for {duplicates})"

        return summary

    def _reset_calib_sliders() -> None:
        """Zero every slider (and the cached ``current_q``).  Stays in mode."""
        for name, sl in calib_state["sliders"].items():
            try:
                sl.value = 0.0  # type: ignore[attr-defined]
            except Exception:
                pass
            calib_state["current_q"][name] = 0.0
        _apply_calib_q()

    _CANONICAL_JOINT_NAMES = (
        "hips", "spine", "chest", "neck", "head",
        "left_shoulder", "left_elbow", "left_wrist",
        "right_shoulder", "right_elbow", "right_wrist",
        "left_hip", "left_knee", "left_ankle",
        "right_hip", "right_knee", "right_ankle",
    )

    def _current_ref_mappings() -> tuple[dict[str, str], dict[str, str]]:
        """Return (canonical_to_native, native_to_canonical) for the current reference.

        When the reference has no ``source_to_canonical`` (e.g.
        ``smplx``), both dicts are identity maps over the
        17 canonical names.
        """
        from hhtools.retarget.calibration.reference import (
            build_motion_reference,
            load_reference_pose,
        )
        try:
            ref_name = calib_reference_picker.value
            motion_obj = state.get("current_motion")
            if ref_name == "glb":
                if motion_obj is not None:
                    ref = build_motion_reference(motion_obj, "glb")
                else:
                    ref = load_reference_pose("smpl")
            else:
                ref = load_reference_pose(ref_name)
        except Exception:
            ref = None

        if ref is not None and ref.source_to_canonical:
            n2c = dict(ref.source_to_canonical)
            c2n: dict[str, str] = {}
            for native, canonical in n2c.items():
                c2n.setdefault(canonical, native)
            return c2n, n2c

        identity = {n: n for n in _CANONICAL_JOINT_NAMES}
        return dict(identity), dict(identity)

    def _reference_joint_options() -> tuple[str, ...]:
        """Joint names shown in the ik_map assignment dropdowns.

        Returns the **native** names from the current reference so
        the user sees the same names as their data source (e.g.
        ``LeftShin`` instead of ``left_knee`` for soma_bvh).
        """
        c2n, _ = _current_ref_mappings()
        return tuple(c2n.get(c, c) for c in _CANONICAL_JOINT_NAMES)

    def _build_mapping_dropdowns(model: URDFRobotModel) -> dict[str, object]:
        """Create a dropdown per ROBOT LINK letting the user rebind which
        human joint should drive it at retarget time.

        Dropdown options show the **native** joint names from the current
        reference (e.g. ``LeftShin`` for soma_bvh) so the user sees the
        same names as their data source.  Internally, the ik_map is
        always stored in canonical names — conversion happens on read
        (canonical→native for display) and write (native→canonical for
        storage).
        """

        handles: dict[str, object] = {}
        options = _reference_joint_options()
        c2n, _ = _current_ref_mappings()

        if not options:
            server.gui.add_markdown(
                f"<span style='color:{PALETTE.ui_error}'>No reference joints available "
                "— mapping editor disabled.</span>"
            )
            return handles

        with server.gui.add_folder(
            "Joint mapping", expand_by_default=False,
        ):
            server.gui.add_markdown(
                "For each robot link, pick which human joint from the "
                f"<b>{_html_escape(calib_reference_picker.value)}</b> "
                "reference should drive it.  Edits here are saved into "
                "<code>robot.yaml</code>'s <code>ik_map</code> block on "
                "<b>Save</b> and take effect on the next retarget."
            )

            link_to_canonical: dict[str, str] = {}
            for canonical, link in calib_state["ik_map"].items():
                link_to_canonical[link] = canonical

            for canonical, link in calib_state["ik_map"].items():
                current_canonical = link_to_canonical.get(link, canonical)
                current_native = c2n.get(current_canonical, current_canonical)

                opts = list(options)
                if current_native not in opts:
                    opts.insert(0, current_native)

                dd = server.gui.add_dropdown(
                    label=link, options=tuple(opts), initial_value=current_native,
                )

                def _make_dd_handler(robot_link: str, widget=dd):
                    def _on_change(_):  # type: ignore[no-untyped-def]
                        _update_ik_map(robot_link, str(widget.value))
                    return _on_change

                dd.on_update(_make_dd_handler(link))
                handles[link] = dd

        return handles

    def _update_ik_map(robot_link: str, new_native_value: str) -> None:
        """Apply a dropdown change to ``calib_state['ik_map']``.

        Dropdown values are in **native** names; we convert to canonical
        before storing so the ik_map stays in canonical form.
        """

        _, n2c = _current_ref_mappings()
        dropdowns: dict[str, object] = calib_state["mapping_dropdowns"]  # type: ignore[assignment]
        new_map: dict[str, str] = {}
        for link, widget in dropdowns.items():
            native_picked = str(getattr(widget, "value", ""))
            if not native_picked:
                continue
            if link == robot_link:
                native_picked = new_native_value
            canonical_picked = n2c.get(native_picked, native_picked)
            new_map[canonical_picked] = link
        calib_state["ik_map"] = new_map

    def _build_calib_session(model: URDFRobotModel) -> None:
        """Create an inline session folder holding sliders + action buttons.

        The folder lives inside the Robot tab (right-hand GUI panel, same
        place as the rest of the controls).  Viser doesn't expose a
        non-dimming "drawer" / "left sidebar" widget, so inline folders
        are the cleanest way to surface the sliders without a ``modal``
        backdrop that darkens the 3D scene.

        The folder itself is the teardown anchor: ``.remove()`` on it
        disposes every widget we added below (sliders + buttons) in one
        call — see :func:`_clear_calib_sliders`.
        """

        ref_name = calib_reference_picker.value
        session = server.gui.add_folder(
            f"Calibration session (ref: {ref_name})",
            expand_by_default=True,
        )
        calib_state["session"] = session

        handles: dict[str, object] = {}
        with session:
            server.gui.add_markdown(
                "Drag each slider so the robot's pose overlays the blue "
                "reference human.  When you click **Save**, the closed-"
                "form per-limb scale + offset will be derived and cached "
                "into the per-format file "
                "<code>retarget_calibration_[reference].yaml</code> "
                "beside the URDF (the segment in brackets is the Reference "
                "pose you picked).  "
                "Cancel reverts without writing."
            )
            save_btn = server.gui.add_button(
                "Save", icon="device-floppy", color="green",
            )
            reset_btn = server.gui.add_button(
                "Reset joints to zero", icon="refresh",
            )
            cancel_btn = server.gui.add_button(
                "Cancel", icon="x",
            )
            with server.gui.add_folder(
                "Actuated joints", expand_by_default=True,
            ):
                for joint in model.actuated_joints:
                    if joint.joint_type == "fixed":
                        continue
                    lo = (
                        float(joint.limit_lower)
                        if joint.limit_lower is not None else -float(np.pi)
                    )
                    hi = (
                        float(joint.limit_upper)
                        if joint.limit_upper is not None else float(np.pi)
                    )
                    # Some exporters emit lower>upper for continuous
                    # joints — fall back to ±π rather than raise.
                    if hi <= lo:
                        lo, hi = -float(np.pi), float(np.pi)
                    init = float(np.clip(
                        calib_state["current_q"].get(joint.name, 0.0), lo, hi,
                    ))
                    slider = server.gui.add_slider(
                        joint.name, min=lo, max=hi, step=0.001,
                        initial_value=init,
                    )

                    def _make_handler(jn: str, sl=slider):
                        def _handler(_):  # type: ignore[no-untyped-def]
                            calib_state["current_q"][jn] = float(sl.value)
                            _apply_calib_q()
                        return _handler

                    slider.on_update(_make_handler(joint.name))
                    handles[joint.name] = slider

            # ---- Joint mapping sub-folder ------------------------------
            # Lets the user re-assign which canonical human joint drives
            # each robot link before retargeting.  See the README's
            # "Retarget calibration · Joint mapping" section for when
            # this matters (typically upper-body mismatches between
            # SMPL-family clips and URDF shoulder conventions).
            mapping_handles = _build_mapping_dropdowns(model)
            calib_state["mapping_dropdowns"] = mapping_handles

        calib_state["sliders"] = handles

        # Wire the in-session buttons.  Save / Reset / Cancel all route
        # back through the helpers defined above — which are responsible
        # for tearing down the session folder when appropriate.
        @save_btn.on_click  # type: ignore[misc]
        def _on_save(_):  # type: ignore[no-untyped-def]
            _do_save_calibration()

        @reset_btn.on_click  # type: ignore[misc]
        def _on_reset(_):  # type: ignore[no-untyped-def]
            _reset_calib_sliders()

        @cancel_btn.on_click  # type: ignore[misc]
        def _on_cancel(_):  # type: ignore[no-untyped-def]
            _exit_calibration_mode()

    def _enter_calibration_mode() -> bool:
        """Reveal the session sliders, draw the reference skeleton, seed values.

        Returns False (and surfaces a toast) when the robot isn't loaded
        — calibration works off ``model.actuated_joints`` and needs the
        URDF parsed first.  True on success.
        """
        model = state.get("current_model")
        if not isinstance(model, URDFRobotModel):
            _notify_all(
                server, "Load a robot first",
                "Calibration needs a loaded URDF to know which joints to "
                "expose as sliders.",
                color="orange",
            )
            return False

        # Stop Motion-tab playback immediately so frame 0 is the single
        # source of truth while calibrating.
        if callable(get_panel):
            try:
                pn = get_panel()
                if pn is not None and hasattr(pn, "pause_at_frame_zero"):
                    pn.pause_at_frame_zero()
            except Exception:
                pass

        # Tear down any half-built state from a previous click.
        _clear_calib_sliders()

        from hhtools.retarget.calibration.reference import (
            build_motion_reference,
            load_reference_pose,
        )

        try:
            ref_name = calib_reference_picker.value
            motion_live = (
                get_current_motion() if callable(get_current_motion) else None
            )
            ref = None
            # Static refs (smpl, …): ``load_reference_pose``.  ``glb``: loaded clip frame 0.
            if ref_name == "glb":
                if motion_live is None or motion_live.num_frames == 0:
                    _notify_all(
                        server, "No motion loaded",
                        "Load a motion first to use 'glb' reference "
                        "(frame 0 of the clip).",
                        color="orange",
                    )
                    return False
                ref = build_motion_reference(motion_live, "glb")
            else:
                ref = load_reference_pose(ref_name)
        except Exception as err:  # noqa: BLE001
            _notify_all(
                server, "Reference pose unavailable",
                f"{calib_reference_picker.value}: {type(err).__name__}: {err}",
                color="red",
            )
            return False
        if getattr(ref, "fallback", False):
            _notify_all(
                server, "Reference fallback in use",
                f"{ref.name}: smplx body model not found; showing canonical "
                "geometry with SMPL-X joint names instead.",
                color="blue",
            )

        # Place the reference skeleton at the robot's world position.
        # The renderer itself shifts positions so the lowest joint
        # touches Z=0, matching the robot's visual ground level.
        world_offset = (
            float(ROBOT_WORLD_OFFSET[0]),
            float(ROBOT_WORLD_OFFSET[1]),
            float(ROBOT_WORLD_OFFSET[2]),
        )

        # Align the reference skeleton to face the same direction as
        # the loaded robot in the scene.  The usual recipe:
        #
        #   1. Robot: shoulder links in URDF (biacromial × +Z) when
        #      available, else ``forward_axis`` on the preset.
        #   2. Reference: ``glb`` uses clip frame 0 — biacromial from shoulder geometry;
        #      back to line across hips.
        #   3. ``heading_rad = robot_yaw − ref_yaw`` rotates the blue
        #      skeleton in the horizontal plane.
        heading_rad = 0.0

        def _yaw_from_biacromial(left_pos, right_pos):
            """Planar (XY) forward yaw from the left–right line × world +Z.

            Matches the right-handed, Z-up screen convention used elsewhere:
            the cross product with ``+Z`` is *not* the coronal forward for every
            rig, but is consistent for comparing robot vs. reference when both
            use the same shoulder/hip line construction.
            """
            shoulder = np.asarray(left_pos, dtype=np.float32) - np.asarray(
                right_pos, dtype=np.float32,
            )
            fwd = np.cross(shoulder, np.array([0.0, 0.0, 1.0], dtype=np.float32))
            fwd[2] = 0.0
            mag = float(np.linalg.norm(fwd))
            if mag < 1e-6:
                return None
            return float(np.arctan2(fwd[1] / mag, fwd[0] / mag))

        # --- Robot forward yaw (do not conflate a computed 0.0 with "unset") --
        robot_fwd_yaw = 0.0
        robot_fwd_from_ik = False
        try:
            from hhtools.retarget.calibration.calibration import (
                _collect_link_transforms_at_q,
            )
            current_q = dict(calib_state.get("current_q") or {})
            link_tx = _collect_link_transforms_at_q(model, current_q)
            ik_map = model.preset.ik_map or {}
            ls_link = ik_map.get("left_shoulder")
            rs_link = ik_map.get("right_shoulder")
            if isinstance(ls_link, dict):
                ls_link = ls_link.get("t_body") or ls_link.get("link")
            if isinstance(rs_link, dict):
                rs_link = rs_link.get("t_body") or rs_link.get("link")
            if ls_link and rs_link:
                T_ls = link_tx.get(str(ls_link))
                T_rs = link_tx.get(str(rs_link))
                if T_ls is not None and T_rs is not None:
                    yaw = _yaw_from_biacromial(
                        T_ls[:3, 3], T_rs[:3, 3],
                    )
                    if yaw is not None:
                        robot_fwd_yaw = float(yaw)
                        robot_fwd_from_ik = True
        except Exception:
            pass
        if not robot_fwd_from_ik:
            fwd_axis = getattr(model.preset, "forward_axis", "X")
            _axis_to_yaw = {"X": 0.0, "Y": np.pi / 2, "-X": np.pi, "-Y": -np.pi / 2}
            robot_fwd_yaw = _axis_to_yaw.get(str(fwd_axis).upper(), 0.0)

        # --- Reference skeleton forward yaw ---
        s2c = ref.source_to_canonical
        can2native: dict[str, str] = {}
        for native, canonical in s2c.items():
            can2native.setdefault(canonical, native)
        jn = list(ref.joint_names)
        jset = frozenset(jn)

        def _ref_biacromial_yaw(
            c_left: str, c_right: str,
        ) -> float | None:
            ln = can2native.get(c_left)
            rn = can2native.get(c_right)
            if not ln or not rn or ln not in jset or rn not in jset:
                return None
            return _yaw_from_biacromial(
                ref.positions[jn.index(ln)],
                ref.positions[jn.index(rn)],
            )

        ref_fwd_yaw = 0.0
        y_ref: float | None = _ref_biacromial_yaw(
            "left_shoulder", "right_shoulder",
        )
        if y_ref is None:
            y_ref = _ref_biacromial_yaw("left_hip", "right_hip")
        if y_ref is not None:
            ref_fwd_yaw = float(y_ref)

        heading_rad = robot_fwd_yaw - ref_fwd_yaw

        calib_ref_exclude: set[int] = set()
        if ref_name == "glb" and (
            motion_live is not None and motion_live.num_bones > 0
        ):
            if detect_virtual_root(list(motion_live.hierarchy.bone_names)):
                calib_ref_exclude.add(0)
        elif detect_virtual_root(list(ref.joint_names)):
            calib_ref_exclude.add(0)

        renderer = ReferenceSkeletonRenderer(
            server, ref, world_offset=world_offset,
            heading_rad=heading_rad,
            show_labels=bool(calib_show_labels.value),
            exclude_bone_indices=calib_ref_exclude if calib_ref_exclude else None,
        )
        calib_state["ref_renderer"] = renderer

        # Hide human motion renderers during calibration so only the
        # reference skeleton and robot are visible.
        if motion_sync is not None:
            hide_fn = motion_sync.get("hide_motion_renderers")
            if callable(hide_fn):
                try:
                    hide_fn()
                except Exception:
                    pass
        _clear_scaled_preview()

        _load_existing_calib_into_state(model)
        # Seed the editable ik_map from the preset's current mapping so
        # dropdowns start where the user's robot.yaml left off.  A
        # deep-copy avoids mutating preset state before the user hits
        # Save — Cancel should leave everything unchanged.
        calib_state["ik_map"] = dict(model.preset.ik_map)
        _build_calib_session(model)
        _apply_calib_q()
        calib_state["active"] = True
        robot_state["calibration_active"] = True
        _set_retarget_and_playback_gates(calibration_active=True)
        _refresh_calib_status()
        return True

    def _exit_calibration_mode() -> None:
        """Remove the session folder + reference skeleton; reset robot to zero."""
        robot_state["calibration_active"] = False
        _clear_calib_sliders()
        calib_state["active"] = False
        model = state.get("current_model")
        animator = state.get("animator")
        if (
            isinstance(model, URDFRobotModel)
            and animator is not None
        ):
            try:
                joint_order = tuple(j.name for j in model.actuated_joints)
                animator.set_frame_joint_q(
                    np.zeros(len(joint_order), dtype=np.float64),
                    joint_order,
                    has_root=False,
                )
            except Exception:
                pass

        # Restore human motion renderers hidden during calibration.
        if motion_sync is not None:
            restore_fn = motion_sync.get("restore_motion_renderers")
            if callable(restore_fn):
                try:
                    restore_fn()
                except Exception:
                    pass

        _set_retarget_and_playback_gates(calibration_active=False)
        _refresh_calib_status()

    @calib_start_btn.on_click
    def _on_calib_start(_):  # type: ignore[no-untyped-def]
        _enter_calibration_mode()

    @calib_reference_picker.on_update
    def _on_calib_ref_change(_):  # type: ignore[no-untyped-def]
        if calib_state["active"]:
            _enter_calibration_mode()
        else:
            model = state.get("current_model")
            if isinstance(model, URDFRobotModel):
                _load_existing_calib_into_state(model)
            _refresh_calib_status()

    @calib_show_labels.on_update
    def _on_calib_labels_change(_):  # type: ignore[no-untyped-def]
        renderer = calib_state.get("ref_renderer")
        if renderer is not None and hasattr(renderer, "set_labels_visible"):
            renderer.set_labels_visible(bool(calib_show_labels.value))
            if bool(calib_show_labels.value) and not renderer._label_handles:
                if calib_state["active"]:
                    _enter_calibration_mode()

    # Prime the status banner on tab open so users know whether to
    # calibrate or not before they even load a robot.
    _refresh_calib_status()

    @schema_btn.on_click
    def _on_schema(_):  # type: ignore[no-untyped-def]
        model = state.get("current_model")
        if not isinstance(model, URDFRobotModel):
            _notify_all(
                server, "No robot loaded",
                "Click Load first so we know which DOF order to dump.",
                color="orange",
            )
            return
        out = Path(tempfile.gettempdir()) / f"hhtools_{model.preset.name}_schema.csv"
        cols = header_columns(model)
        out.write_text(",".join(cols) + "\n", encoding="utf-8")
        _notify_all(
            server, "Schema exported",
            f"{out}  ({len(cols)} columns)",
            color="blue",
        )
        progress_md.content = (
            f"<span style='color:{PALETTE.ui_ok}'>✓</span> "
            f"Wrote <code>{_html_escape(out)}</code> · {len(cols)} columns"
        )


def _notify_all(
    server,  # type: ignore[no-untyped-def]
    title: str,
    body: str,
    *,
    color: str = "green",
) -> None:
    """Send a transient toast to every connected browser.

    Uses Viser's per-client ``add_notification`` so users get immediate visual
    feedback (top-left corner of the viewer) in addition to the persistent Save
    log panel. Exceptions are swallowed — notifications are a nice-to-have and
    must never break the save pipeline itself.
    """
    clients = server.get_clients().values()
    for client in clients:
        try:
            client.add_notification(
                title=title,
                body=body,
                auto_close_seconds=_NOTIFICATION_TTL_SECONDS,
                with_close_button=True,
                color=color,
            )
        except Exception:
            pass


def _make_status(
    all_entries: list[LibraryEntry],
    filtered_entries: list[LibraryEntry],
    cache: EphemeralCache,
) -> str:
    summary = cache.summary()
    written = summary["written"]
    return (
        f"<span style='opacity:0.75'>"
        f"<b>{len(filtered_entries)}</b>/{len(all_entries)} clips"
        f" &nbsp;·&nbsp; <b>{written}</b> NPZ(s) cached this session"
        f"</span>"
    )


def _fit_camera_to_motion(server, motion: Motion) -> None:  # type: ignore[no-untyped-def]
    """Park the camera in front of the subject's frame-0 pose at a comfortable distance."""
    positions = motion.positions
    if positions.size == 0:
        return
    first_frame = positions[0]
    center = first_frame.mean(axis=0)
    extent = float(np.linalg.norm(first_frame.max(axis=0) - first_frame.min(axis=0)))
    distance = max(1.5, extent * 1.6)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    forward = _estimate_body_forward(motion, first_frame, world_up)
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    right = right / right_norm if right_norm > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

    cam = (
        center
        - forward * distance
        + world_up * (distance * 0.25)
        + right * (distance * 0.15)
    )
    cam_pos = (float(cam[0]), float(cam[1]), float(cam[2]))
    look_at = (float(center[0]), float(center[1]), float(center[2]))
    for client in server.get_clients().values():
        client.camera.position = cam_pos
        client.camera.look_at = look_at


def _estimate_body_forward(motion: Motion, frame_pos: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    """Derive a body-forward direction from shoulder / hip spread, fallback to +Y."""
    names = [n.lower() for n in motion.hierarchy.bone_names]

    def _find(keys: tuple[str, ...]) -> int:
        for i, n in enumerate(names):
            if all(k in n for k in keys):
                return i
        return -1

    pairs = [
        (("left", "shoulder"), ("right", "shoulder")),
        (("left", "collar"), ("right", "collar")),
        (("left", "upleg"), ("right", "upleg")),
        (("left", "hip"), ("right", "hip")),
    ]
    for left_keys, right_keys in pairs:
        li = _find(left_keys)
        ri = _find(right_keys)
        if li >= 0 and ri >= 0:
            across = frame_pos[li] - frame_pos[ri]
            forward = np.cross(across, world_up)
            n = float(np.linalg.norm(forward))
            if n > 1e-4:
                return (forward / n).astype(np.float32)
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


__all__ = ["run_viewer"]
