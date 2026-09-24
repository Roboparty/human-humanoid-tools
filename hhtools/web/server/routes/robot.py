"""Robot catalog, selection, upload, and deletion routes."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import File, HTTPException, UploadFile

from hhtools.application.robots import (
    _is_builtin_robot_preset,
    _merge_retarget_references,
    _read_yaml_retarget_references,
    _start_robot_prewarm,
)
from hhtools.web.server.boundary import _safe_upload_directory_name


@dataclass(frozen=True, slots=True)
class RobotRouteOperations:
    serialize_and_store_robot: Callable


def register_robot_routes(app, *, state, uploads) -> RobotRouteOperations:
    _store_uploads = uploads.store

    @app.get("/api/robots")
    def robots() -> dict:
        from hhtools.robot.registry import is_user_installed, list_presets, refresh

        refresh()
        out = []
        for p in list_presets():
            builtin = _is_builtin_robot_preset(p.name)
            out.append(
                {
                    "name": p.name,
                    "display_name": p.display_name,
                    "has_urdf": p.has_urdf,
                    "num_dof": len(p.dof_order),
                    "builtin": builtin,
                    "deletable": is_user_installed(p, state.robot_root) and not builtin,
                }
            )
        return {
            "robots": out,
            "library_dir": str(state.robot_root.resolve()),
        }

    def _serialize_and_store_robot(name: str) -> dict:
        from hhtools.io.scene_serialize import serialize_robot
        from hhtools.robot.loader import load_robot
        from hhtools.robot.registry import get as get_preset

        preset = get_preset(name)
        model = load_robot(preset, compile_mjcf=True)
        if model.mujoco_model is None:
            raise RuntimeError(
                f"URDF for {name!r} did not compile to a MuJoCo model after mesh "
                f"path repair — upload the full robot folder (URDF + meshes/, and "
                f"any mesh/, convex/, or assets/ sidecars). Collada (.dae) meshes "
                f"are auto-converted to STL at ingest."
            )
        state.robots[name] = model
        _start_robot_prewarm(state, model, name)
        payload = serialize_robot(model, name=name)
        try:
            from hhtools.retarget.newton_basic.pipeline import is_newton_ik_prewarmed

            payload["ik_prewarmed"] = is_newton_ik_prewarmed(name)
        except Exception:  # noqa: BLE001
            payload["ik_prewarmed"] = False
        return payload

    @app.post("/api/robot/select")
    async def robot_select(body: dict) -> dict:
        name = body.get("name", "")
        try:
            return _serialize_and_store_robot(name)
        except Exception as err:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"load robot failed: {err}") from err

    @app.post("/api/robot/upload")
    async def robot_upload(files: list[UploadFile] = File(...), name: str | None = None) -> dict:
        """Accept a URDF + mesh files; scaffold + auto-repair, load, serialize."""
        from hhtools.robot.kinematics import prepare_ik_map
        from hhtools.robot.registry import preset_from_dir, refresh
        from hhtools.robot.scaffold import scaffold_yaml_file
        from hhtools.robot.urdf_normalize import (
            ensure_urdf_meshes_resolvable,
            robot_upload_destination,
        )
        from hhtools.robot.yaml_io import update_robot_yaml_ik_map

        urdf_path: Path | None = None
        saved: list[Path] = []
        try:
            drop_name = _safe_upload_directory_name(name, default="uploaded_robot")
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        if _is_builtin_robot_preset(drop_name):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"robot {drop_name!r} is a built-in preset and cannot be overwritten via upload"
                ),
            )
        drop = state.robot_root / drop_name
        # Re-uploading an existing robot rebuilds geometry but must NOT wipe the
        # user's tuned retarget config: keep bundled scalers, calibrations, and
        # the robot.yaml ``retarget.references`` mapping across the rebuild.
        preserved_files: dict[str, bytes] = {}
        preserved_references: dict | None = None
        if drop.exists():
            for pat in ("retarget_calibration_*.yaml", "*scaler_config*.yaml"):
                for f in drop.glob(pat):
                    try:
                        preserved_files[f.name] = f.read_bytes()
                    except OSError:
                        pass
            preserved_references = _read_yaml_retarget_references(drop)
            shutil.rmtree(drop, ignore_errors=True)

        def _robot_upload_path(relative: Path) -> Path:
            rel = relative.as_posix()
            return robot_upload_destination(
                drop,
                rel,
                is_urdf=rel.lower().endswith(".urdf"),
            )

        stored = await _store_uploads(
            files,
            drop,
            default="f",
            destination_for=_robot_upload_path,
        )
        for rel_path, dst in stored:
            is_urdf = rel_path.suffix.lower() == ".urdf"
            saved.append(dst)
            if is_urdf:
                urdf_path = dst
        if urdf_path is None:
            raise HTTPException(status_code=400, detail="no .urdf file in upload")

        try:
            ensure_urdf_meshes_resolvable(
                urdf_path,
                search_dirs=[drop / "meshes", drop],
                output_path=urdf_path,
            )
            # Restore calibration / bundled scalers before scaffold so
            # retarget_calibration_*.yaml survives the URDF replace.
            for fname, data in preserved_files.items():
                try:
                    (drop / fname).write_bytes(data)
                except OSError:
                    pass
            scaffold_yaml_file(urdf_path, overwrite=True, root_dir=drop)
            try:
                preset = preset_from_dir(drop)
            except FileNotFoundError as err:
                raise HTTPException(
                    status_code=400,
                    detail=f"robot ingest failed: {err}",
                ) from err
            refresh()
            repaired, _changes = prepare_ik_map(urdf_path, dict(preset.ik_map))
            yaml_path = preset.meta.get("yaml_path")
            if yaml_path and repaired != dict(preset.ik_map):
                update_robot_yaml_ik_map(yaml_path, repaired)
                refresh()
            if preserved_references:
                _merge_retarget_references(yaml_path, preserved_references)
            if preserved_references:
                refresh()
            return _serialize_and_store_robot(preset.name)
        except HTTPException:
            raise
        except Exception as err:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"robot ingest failed: {err}") from err

    @app.delete("/api/robot/{name}")
    def robot_delete(name: str) -> dict:
        """Remove a user-installed robot from the persistent library."""
        from hhtools.robot.registry import get as get_preset
        from hhtools.robot.registry import is_user_installed, refresh

        try:
            preset = get_preset(name)
        except KeyError as err:
            raise HTTPException(status_code=404, detail=f"unknown robot: {name}") from err
        if _is_builtin_robot_preset(preset.name):
            raise HTTPException(
                status_code=403,
                detail=f"robot {name!r} is a built-in preset and cannot be deleted",
            )
        if not is_user_installed(preset, state.robot_root):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"robot {name!r} is a built-in preset and cannot be deleted from the UI; "
                    "only robots registered via the web UI (under your user library) are removable"
                ),
            )
        target = preset.root_dir.resolve()
        library = state.robot_root.resolve()
        try:
            if not target.is_relative_to(library):
                raise HTTPException(status_code=403, detail="robot is outside the user library")
        except ValueError as err:
            raise HTTPException(
                status_code=403, detail="robot is outside the user library"
            ) from err
        shutil.rmtree(target, ignore_errors=False)
        state.robots.pop(name, None)
        refresh()
        return {"ok": True, "deleted": name}

    return RobotRouteOperations(serialize_and_store_robot=_serialize_and_store_robot)
