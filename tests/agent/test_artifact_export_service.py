from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from hhtools.contracts import ApiError, ArtifactDescriptor, ErrorStage
from hhtools.contracts.artifact_exports import ArtifactExportReceipt
from hhtools.services.artifact_exports import ArtifactExportError, ArtifactExportService
from hhtools.services.artifacts import StoredArtifact
from hhtools.services.jobs import JobManagerError

_JOB_ID = "job:export-test"
_ARTIFACT_ID = f"artifact:preview:{'a' * 64}"


def _stored(path: Path, payload: bytes = b"verified artifact bytes") -> StoredArtifact:
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    return StoredArtifact(
        descriptor=ArtifactDescriptor(
            artifact_id=_ARTIFACT_ID,
            job_id=_JOB_ID,
            kind="preview",
            format="glb",
            resource_uri=f"hhtools://jobs/{_JOB_ID}/artifacts/{_ARTIFACT_ID}",
            media_type="model/gltf-binary",
            size_bytes=len(payload),
            sha256=digest,
        ),
        path=path,
    )


class _Jobs:
    def __init__(
        self,
        stored: StoredArtifact | None = None,
        error: Exception | None = None,
    ) -> None:
        self.stored = stored
        self.error = error
        self.calls: list[tuple[str, str, bool]] = []

    def get_artifact(
        self,
        job_id: str,
        artifact_id: str,
        *,
        verify: bool = False,
    ) -> StoredArtifact:
        self.calls.append((job_id, artifact_id, verify))
        if self.error is not None:
            raise self.error
        assert self.stored is not None
        return self.stored


def _destination(export_root: Path, receipt: ArtifactExportReceipt) -> Path:
    return export_root.joinpath(*receipt.relative_path.split("/"))


def test_export_receipt_is_strict_portable_and_contains_no_host_location(tmp_path: Path) -> None:
    stored = _stored(tmp_path / "source.glb")
    jobs = _Jobs(stored)
    export_root = tmp_path / "exports"

    receipt = ArtifactExportService(jobs, export_root).export(_JOB_ID, _ARTIFACT_ID)

    assert jobs.calls == [(_JOB_ID, _ARTIFACT_ID, True)]
    assert _destination(export_root, receipt).read_bytes() == stored.path.read_bytes()
    document = receipt.model_dump(mode="json")
    assert document == {
        "schema_version": "1.0",
        "root_id": "agent-exports",
        "relative_path": receipt.relative_path,
        "job_id": _JOB_ID,
        "artifact_id": _ARTIFACT_ID,
        "kind": "preview",
        "format": "glb",
        "media_type": "model/gltf-binary",
        "size_bytes": stored.descriptor.size_bytes,
        "sha256": stored.descriptor.sha256,
    }
    serialized = json.dumps(document)
    assert str(tmp_path) not in serialized
    assert "artifact-objects" not in serialized
    assert "base64" not in serialized.casefold()
    assert "bytes" not in document

    with pytest.raises(ValidationError):
        ArtifactExportReceipt(**{**document, "relative_path": "../escape.glb"})
    with pytest.raises(ValidationError):
        ArtifactExportReceipt(**{**document, "root_id": "caller-selected"})
    with pytest.raises(ValidationError):
        ArtifactExportReceipt(**{**document, "content_base64": "AAAA"})
    with pytest.raises(ValidationError, match="job and artifact identity"):
        ArtifactExportReceipt(
            **{
                **document,
                "relative_path": (f"jobs/{'b' * 64}/{document['relative_path'].split('/')[-1]}"),
            }
        )


def test_export_preserves_job_membership_errors_without_publishing(tmp_path: Path) -> None:
    jobs = _Jobs(
        error=JobManagerError(
            ApiError(
                code="ARTIFACT_NOT_FOUND",
                message="The job has no artifact with the requested id.",
                stage=ErrorStage.ARTIFACT,
            )
        )
    )
    export_root = tmp_path / "exports"
    service = ArtifactExportService(jobs, export_root)

    with pytest.raises(ArtifactExportError) as captured:
        service.export(_JOB_ID, _ARTIFACT_ID)

    assert captured.value.code == "ARTIFACT_NOT_FOUND"
    assert captured.value.api_error.stage is ErrorStage.ARTIFACT
    assert jobs.calls == [(_JOB_ID, _ARTIFACT_ID, True)]
    assert list(export_root.rglob("*")) == []


def test_unexpected_resolver_io_error_is_mapped_without_leaking_its_path(tmp_path: Path) -> None:
    secret_path = r"C:\Users\Nora\private\artifact.bin"
    service = ArtifactExportService(
        _Jobs(error=OSError(f"could not read {secret_path}")),
        tmp_path / "exports",
    )

    with pytest.raises(ArtifactExportError) as captured:
        service.export(_JOB_ID, _ARTIFACT_ID)

    assert captured.value.code == "INTERNAL_ERROR"
    assert captured.value.api_error.stage is ErrorStage.INTERNAL
    assert captured.value.api_error.details == {"root_id": "agent-exports"}
    assert secret_path not in captured.value.api_error.model_dump_json()


def test_repeat_and_concurrent_exports_are_deterministic_and_leave_no_temps(
    tmp_path: Path,
) -> None:
    stored = _stored(tmp_path / "source.glb", b"concurrent export")
    jobs = _Jobs(stored)
    export_root = tmp_path / "exports"
    service = ArtifactExportService(jobs, export_root)

    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(
            pool.map(
                lambda _index: service.export(_JOB_ID, _ARTIFACT_ID),
                range(16),
            )
        )

    assert all(receipt == receipts[0] for receipt in receipts)
    destination = _destination(export_root, receipts[0])
    assert destination.read_bytes() == stored.path.read_bytes()
    assert [path for path in export_root.rglob("*") if path.is_file()] == [destination]
    assert not list(export_root.rglob("*.tmp"))
    assert jobs.calls == [(_JOB_ID, _ARTIFACT_ID, True)] * 16


def test_existing_identical_export_is_idempotent_and_corruption_is_repaired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"stable export payload"
    stored = _stored(tmp_path / "source.glb", payload)
    service = ArtifactExportService(_Jobs(stored), tmp_path / "exports")
    first = service.export(_JOB_ID, _ARTIFACT_ID)
    destination = _destination(tmp_path / "exports", first)

    def unexpected_copy(*_args: Any, **_kwargs: Any) -> Path:
        raise AssertionError("an identical destination must not be rewritten")

    original_copy = service._copy_to_temporary
    monkeypatch.setattr(service, "_copy_to_temporary", unexpected_copy)
    assert service.export(_JOB_ID, _ARTIFACT_ID) == first

    monkeypatch.setattr(service, "_copy_to_temporary", original_copy)
    destination.write_bytes(b"x" * len(payload))
    assert service.export(_JOB_ID, _ARTIFACT_ID) == first
    assert destination.read_bytes() == payload
    assert not list((tmp_path / "exports").rglob("*.tmp"))


def test_copy_rechecks_exact_source_bytes_after_job_manager_verification(tmp_path: Path) -> None:
    expected = b"expected bytes"
    stored = _stored(tmp_path / "source.glb", expected)
    stored.path.write_bytes(b"tampered bytes")
    service = ArtifactExportService(_Jobs(stored), tmp_path / "exports")

    with pytest.raises(ArtifactExportError) as captured:
        service.export(_JOB_ID, _ARTIFACT_ID)

    assert captured.value.code == "ARTIFACT_HASH_MISMATCH"
    assert captured.value.api_error.retryable is True
    assert not [path for path in (tmp_path / "exports").rglob("*") if path.is_file()]


def test_publish_io_errors_are_structured_and_clean_temporary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = _stored(tmp_path / "source.glb")
    export_root = tmp_path / "exports"
    service = ArtifactExportService(_Jobs(stored), export_root)

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("simulated publish failure")

    monkeypatch.setattr("hhtools.services.artifact_exports.os.replace", fail_replace)
    with pytest.raises(ArtifactExportError) as captured:
        service.export(_JOB_ID, _ARTIFACT_ID)

    assert captured.value.code == "ARTIFACT_EXPORT_FAILED"
    assert captured.value.api_error.retryable is True
    assert captured.value.api_error.details == {
        "artifact_id": _ARTIFACT_ID,
        "job_id": _JOB_ID,
        "root_id": "agent-exports",
    }
    assert str(tmp_path) not in captured.value.api_error.model_dump_json()
    assert not list(export_root.rglob("*.tmp"))
