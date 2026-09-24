from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hhtools.services.artifacts import ArtifactStore, ArtifactStoreError

JOB_ID = "job:0123456789abcdef"


def test_json_artifact_is_canonical_portable_and_survives_restart(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)

    first = store.put_json(
        job_id=JOB_ID,
        kind="job_spec",
        document={"z": 2, "nested": {"b": True, "a": 1}},
        metadata={"schema": "hhtools.job-spec.v2"},
    )
    repeated = store.put_json(
        job_id=JOB_ID,
        kind="job_spec",
        document={"nested": {"a": 1, "b": True}, "z": 2},
        metadata={"schema": "hhtools.job-spec.v2"},
    )

    assert repeated == first
    assert first.resource_uri.endswith(first.artifact_id)
    assert first.sha256 is not None
    assert first.size_bytes is not None
    assert first.created_at is not None
    stored = ArtifactStore(tmp_path).get(first.artifact_id, verify=True)
    assert stored.descriptor == first
    assert stored.path.read_text(encoding="utf-8") == ('{"nested":{"a":1,"b":true},"z":2}')
    persisted = (
        sqlite3.connect(tmp_path / "artifacts.sqlite3")
        .execute(
            "SELECT metadata_json, object_path FROM artifacts WHERE artifact_id = ?",
            (first.artifact_id,),
        )
        .fetchone()
    )
    assert persisted == (
        '{"schema":"hhtools.job-spec.v2"}',
        f"artifact-objects/{first.sha256[:2]}/{first.sha256}",
    )
    assert str(tmp_path) not in json.dumps(first.model_dump(mode="json"))


def test_same_bytes_for_different_jobs_have_distinct_descriptors_and_shared_object(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    first = store.put_bytes(
        job_id=JOB_ID,
        kind="preview",
        payload=b"preview-data",
        format="mp4",
        media_type="video/mp4",
    )
    second = store.put_bytes(
        job_id="job:fedcba9876543210",
        kind="preview",
        payload=b"preview-data",
        format="mp4",
        media_type="video/mp4",
    )

    assert first.artifact_id != second.artifact_id
    assert first.sha256 == second.sha256
    assert store.get(first.artifact_id).path == store.get(second.artifact_id).path


def test_raw_candidates_are_listed_per_job_in_stable_order(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    first = store.put_bytes(job_id=JOB_ID, kind="log_tail", payload=b"first")
    second = store.put_bytes(job_id=JOB_ID, kind="preview", payload=b"second")
    store.put_bytes(
        job_id="job:another",
        kind="preview",
        payload=b"not-in-result",
    )

    descriptors = store.list_candidates_for_job(JOB_ID)

    assert {item.artifact_id for item in descriptors} == {
        first.artifact_id,
        second.artifact_id,
    }
    assert store.list_for_job(JOB_ID) == descriptors


def test_put_file_copies_a_stable_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "solver-output.csv"
    source.write_bytes(b"joint_a,joint_b\n0.0,1.0\n")
    store = ArtifactStore(tmp_path / "managed")

    descriptor = store.put_file(
        job_id=JOB_ID,
        kind="retargeted_motion",
        source=source,
        format="csv",
        media_type="text/csv",
    )
    source.write_bytes(b"changed later")

    assert store.get(descriptor.artifact_id, verify=True).path.read_bytes() == (
        b"joint_a,joint_b\n0.0,1.0\n"
    )


def test_verification_detects_missing_or_modified_managed_bytes(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    descriptor = store.put_bytes(job_id=JOB_ID, kind="preview", payload=b"safe")
    stored = store.get(descriptor.artifact_id)
    stored.path.write_bytes(b"evil")

    with pytest.raises(ArtifactStoreError) as captured:
        store.get(descriptor.artifact_id, verify=True)

    assert captured.value.code == "ARTIFACT_HASH_MISMATCH"
    assert str(tmp_path) not in str(captured.value.api_error.model_dump(mode="json"))


@pytest.mark.parametrize(
    "metadata",
    [
        {"path": "/srv/private/output.csv"},
        {"path": r"C:\\Users\\Nora\\output.csv"},
        {r"D:\\secret": "value"},
        {"number": float("nan")},
    ],
)
def test_metadata_rejects_host_paths_and_non_finite_values(
    tmp_path: Path,
    metadata: dict[str, object],
) -> None:
    with pytest.raises(ArtifactStoreError) as captured:
        ArtifactStore(tmp_path).put_bytes(
            job_id=JOB_ID,
            kind="preview",
            payload=b"safe",
            metadata=metadata,
        )

    assert captured.value.code == "INVALID_PARAMETER"


def test_json_artifact_rejects_host_paths(tmp_path: Path) -> None:
    with pytest.raises(ArtifactStoreError) as captured:
        ArtifactStore(tmp_path).put_json(
            job_id=JOB_ID,
            kind="manifest",
            document={"output": "/private/results/output.csv"},
        )

    assert captured.value.code == "INVALID_PARAMETER"


def test_json_artifact_allows_controlled_resource_uris(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    descriptor = store.put_json(
        job_id=JOB_ID,
        kind="manifest",
        document={
            "artifacts": [
                {
                    "resource_uri": (
                        "hhtools://jobs/job:0123456789abcdef/artifacts/"
                        f"artifact:job_spec:{'a' * 64}"
                    )
                }
            ],
            "documentation": "https://example.invalid/hhtools/jobs",
        },
    )

    assert store.get(descriptor.artifact_id, verify=True).descriptor == descriptor


@pytest.mark.parametrize(
    ("job_id", "kind", "format_name"),
    [
        ("../job", "preview", None),
        (JOB_ID, "Preview", None),
        (JOB_ID, "../preview", None),
        (JOB_ID, "preview", "bad format"),
    ],
)
def test_identifiers_are_strict_uri_segments(
    tmp_path: Path,
    job_id: str,
    kind: str,
    format_name: str | None,
) -> None:
    with pytest.raises(ArtifactStoreError) as captured:
        ArtifactStore(tmp_path).put_bytes(
            job_id=job_id,
            kind=kind,
            payload=b"safe",
            format=format_name,
        )

    assert captured.value.code == "INVALID_PARAMETER"


def test_unknown_artifact_has_stable_error(tmp_path: Path) -> None:
    with pytest.raises(ArtifactStoreError) as captured:
        ArtifactStore(tmp_path).get("artifact:preview:" + "0" * 64)

    assert captured.value.code == "ARTIFACT_NOT_FOUND"


def test_concurrent_identical_publish_creates_one_descriptor(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)

    def publish(_index: int) -> str:
        return store.put_bytes(
            job_id=JOB_ID,
            kind="preview",
            payload=b"same immutable bytes",
            format="mp4",
        ).artifact_id

    with ThreadPoolExecutor(max_workers=12) as pool:
        artifact_ids = list(pool.map(publish, range(40)))

    assert len(set(artifact_ids)) == 1
    assert len(store.list_candidates_for_job(JOB_ID)) == 1


def test_tampered_relative_object_path_is_rejected_without_escape(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    descriptor = store.put_bytes(job_id=JOB_ID, kind="preview", payload=b"safe")
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE artifacts SET object_path = ? WHERE artifact_id = ?",
            ("../outside", descriptor.artifact_id),
        )

    with pytest.raises(ArtifactStoreError) as captured:
        store.get(descriptor.artifact_id)

    assert captured.value.code == "INTERNAL_ERROR"
    assert "outside" not in captured.value.api_error.message
