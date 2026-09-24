"""Portable job specifications for replaying WebUI work.

The live WebUI uses short-lived tokens for loaded motions and robot-to-robot
sources.  A :class:`JobSpec` deliberately strips those session identifiers and
keeps only inputs that can survive a process restart.  Validation lives in this
small module so API routes, persistence, and tests share the same rules.

This is path-and-parameter replay, not bitwise experiment reproducibility: v1
does not yet record input hashes, the code revision, calibration/preset content,
dependency versions, or device details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

JOB_SPEC_SCHEMA_VERSION = 1
REPLAYABLE_JOB_KINDS = frozenset({"retarget", "batch"})
_SESSION_ONLY_REQUEST_KEYS = frozenset(
    {
        "motion_token",
        "source_token",
        "export_token",
    }
)


class JobSpecError(ValueError):
    """Raised when imported JSON is not a supported job specification."""


def _portable_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _portable_value(item)
            for key, item in value.items()
            if str(key) not in _SESSION_ONLY_REQUEST_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_portable_value(item) for item in value]
    return str(value)


def build_job_spec(kind: str, request: dict[str, Any]) -> dict[str, Any]:
    """Build a versioned, session-independent replay specification."""
    return {
        "schema_version": JOB_SPEC_SCHEMA_VERSION,
        "kind": str(kind).strip(),
        "request": _portable_value(request),
    }


def normalize_job_spec(payload: Any) -> dict[str, Any]:
    """Accept a raw JobSpec or a downloaded WebUI job configuration record."""
    if not isinstance(payload, dict):
        raise JobSpecError("配置根节点必须是 JSON 对象。")

    candidate = payload.get("spec", payload)
    if not isinstance(candidate, dict):
        raise JobSpecError("spec 必须是 JSON 对象。")

    raw_version = candidate.get("schema_version", JOB_SPEC_SCHEMA_VERSION)
    try:
        version = int(raw_version)
    except (TypeError, ValueError) as err:
        raise JobSpecError("schema_version 必须是整数。") from err
    if version != JOB_SPEC_SCHEMA_VERSION:
        raise JobSpecError(f"不支持 JobSpec v{version}；当前只支持 v{JOB_SPEC_SCHEMA_VERSION}。")

    kind = candidate.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise JobSpecError("配置缺少非空的 kind。")
    request = candidate.get("request")
    if not isinstance(request, dict):
        raise JobSpecError("配置缺少 request JSON 对象。")
    return build_job_spec(kind, request)


def _source_paths(  # noqa: PLR0911 - each invalid source shape has a specific reason
    spec: dict[str, Any],
) -> tuple[list[Path], str | None]:
    kind = spec["kind"]
    request = spec["request"]
    if kind == "retarget":
        raw = request.get("source_path")
        if not isinstance(raw, str) or not raw.strip():
            return [], "任务只保留了会话 token，没有可重开的源文件路径。"
        return [Path(raw).expanduser()], None

    raw_source = request.get("source")
    if isinstance(raw_source, str) and raw_source.strip():
        return [Path(raw_source).expanduser()], None

    entries = request.get("entries")
    if not isinstance(entries, list) or not entries:
        return [], "批处理配置没有可重开的 entries。"
    paths: list[Path] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            return [], f"批处理第 {index} 个 entry 不是 JSON 对象。"
        raw = entry.get("source_path")
        if not isinstance(raw, str) or not raw.strip():
            return [], f"批处理第 {index} 个 entry 缺少 source_path。"
        paths.append(Path(raw).expanduser())
    return paths, None


def replay_capability(  # noqa: PLR0911 - each failure mode has a user-facing reason
    spec_payload: Any,
    *,
    ephemeral_root: Path | None = None,
) -> dict[str, Any]:
    """Explain whether this process can reproduce a JobSpec exactly enough."""
    try:
        spec = normalize_job_spec(spec_payload)
    except JobSpecError as err:
        return {"available": False, "reason": str(err), "source_count": 0}

    kind = spec["kind"]
    request = spec["request"]
    if kind not in REPLAYABLE_JOB_KINDS:
        return {
            "available": False,
            "reason": "该任务依赖会话内对象，当前只能复制编辑配置，不能直接重跑。",
            "source_count": 0,
        }
    robot = request.get("robot")
    if not isinstance(robot, str) or not robot.strip():
        return {
            "available": False,
            "reason": "任务配置缺少目标机器人 robot。",
            "source_count": 0,
        }

    paths, source_error = _source_paths(spec)
    if source_error is not None:
        return {"available": False, "reason": source_error, "source_count": 0}

    resolved_ephemeral: Path | None = None
    if ephemeral_root is not None:
        try:
            resolved_ephemeral = ephemeral_root.resolve()
        except OSError:
            resolved_ephemeral = None

    directory_source = (
        kind == "batch"
        and isinstance(request.get("source"), str)
        and bool(str(request["source"]).strip())
    )
    source_count = (
        int(request["entry_count"])
        if directory_source
        and isinstance(request.get("entry_count"), int)
        and request["entry_count"] >= 0
        else len(paths)
    )
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            return {
                "available": False,
                "reason": f"无法解析源文件：{path}",
                "source_count": source_count,
            }
        if directory_source and not resolved.is_dir():
            return {
                "available": False,
                "reason": f"源目录已不存在：{resolved}",
                "source_count": source_count,
            }
        if not directory_source and not resolved.is_file():
            return {
                "available": False,
                "reason": f"源文件已不存在：{resolved}",
                "source_count": source_count,
            }
        if resolved_ephemeral is not None and resolved.is_relative_to(resolved_ephemeral):
            return {
                "available": False,
                "reason": "源文件仍在临时上传目录，请先保存到 Motion Library。",
                "source_count": source_count,
            }

    return {"available": True, "reason": None, "source_count": source_count}
