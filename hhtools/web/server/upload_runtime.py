"""Bounded, atomic staging for multipart Web uploads."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .boundary import (
    _UPLOAD_CHUNK_BYTES,
    _safe_upload_destination,
    _safe_upload_relative_path,
)

if TYPE_CHECKING:
    from fastapi import UploadFile


class UploadStore:
    """Validate, stage, and atomically publish one multipart upload group."""

    def __init__(
        self,
        max_files: int,
        max_file_bytes: int,
        max_request_bytes: int,
    ) -> None:
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.max_request_bytes = max_request_bytes

    def _validated_uploads(
        self,
        files: list[UploadFile],
        *,
        default: str = "upload.bin",
    ) -> list[tuple[UploadFile, Path]]:
        from fastapi import HTTPException

        if len(files) > self.max_files:
            raise HTTPException(
                status_code=413,
                detail=f"too many upload files (limit: {self.max_files})",
            )
        validated: list[tuple[UploadFile, Path]] = []
        try:
            for upload in files:
                validated.append(
                    (
                        upload,
                        _safe_upload_relative_path(upload.filename, default=default),
                    )
                )
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        return validated

    @staticmethod
    def _request_upload_destination(root: Path, relative: Path) -> Path:
        from fastapi import HTTPException

        try:
            return _safe_upload_destination(root, relative)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

    async def store(
        self,
        files: list[UploadFile],
        root: Path,
        *,
        default: str = "upload.bin",
        destination_for: Callable[[Path], Path] | None = None,
    ) -> list[tuple[Path, Path]]:
        """Stage uploads, then publish each complete file atomically.

        The size checks finish before publication starts. Each ``replace`` is
        atomic, but the group is not a filesystem transaction: an I/O failure
        between replacements can still leave a published subset.
        """

        from fastapi import HTTPException

        validated = self._validated_uploads(files, default=default)
        staged: list[tuple[Path, Path, Path]] = []
        total_bytes = 0
        try:
            for upload, relative in validated:
                candidate = destination_for(relative) if destination_for else relative
                destination = self._request_upload_destination(root, candidate)
                destination.parent.mkdir(parents=True, exist_ok=True)
                part = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.upload")
                file_bytes = 0
                try:
                    with part.open("wb") as fp:
                        while chunk := await upload.read(_UPLOAD_CHUNK_BYTES):
                            file_bytes += len(chunk)
                            total_bytes += len(chunk)
                            if file_bytes > self.max_file_bytes:
                                raise HTTPException(
                                    status_code=413,
                                    detail=(
                                        f"upload file exceeds {self.max_file_bytes} bytes: "
                                        f"{relative.as_posix()}"
                                    ),
                                )
                            if total_bytes > self.max_request_bytes:
                                raise HTTPException(
                                    status_code=413,
                                    detail=(
                                        f"upload request exceeds {self.max_request_bytes} bytes"
                                    ),
                                )
                            fp.write(chunk)
                except Exception:
                    part.unlink(missing_ok=True)
                    raise
                staged.append((relative, part, destination))

            # Files become visible only after every item passes the size limits,
            # avoiding a partial folder on a normal 413 rejection. Publication
            # is per-file; an unexpected filesystem failure can still interrupt
            # this loop after an earlier destination has been replaced.
            stored: list[tuple[Path, Path]] = []
            for relative, part, destination in staged:
                part.replace(destination)
                stored.append((relative, destination))
            return stored
        finally:
            for _relative, part, _destination in staged:
                part.unlink(missing_ok=True)
            for upload, _relative in validated:
                await upload.close()
