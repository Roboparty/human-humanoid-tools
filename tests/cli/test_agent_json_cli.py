from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from hhtools.agent.api import router as agent_router
from hhtools.cli import agent as agent_cli
from hhtools.cli import agent_transport
from hhtools.cli.agent import (
    EXIT_INTERNAL_ERROR,
    EXIT_JOB_ERROR,
    EXIT_PARAMETER_ERROR,
    EXIT_PREFLIGHT_ERROR,
    EXIT_SUCCESS,
    run,
)
from hhtools.cli.agent_transport import (
    AgentTransportError,
    HttpAgentTransport,
    StrictJsonError,
    ensure_portable_json,
    loads_strict_json,
)
from hhtools.contracts import (
    AgentCliArgumentDiagnostic,
    AgentCliHelp,
    AgentJobView,
    ApiError,
    ArtifactDescriptor,
    ArtifactListResponse,
    AvailableAssetCatalogEntry,
    AvailableAssetCatalogResponse,
    BatchPreflightResponse,
    CalibrationCandidate,
    CalibrationProposalResponse,
    CalibrationSaveReceipt,
    CalibrationStatusResponse,
    CalibrationValidationReport,
    CapabilityResponse,
    ErrorStage,
    JobProgress,
    PreflightCheck,
    PreflightResponse,
    R2RCalibrationCandidate,
    R2RCalibrationProposalResponse,
    R2RCalibrationSaveReceipt,
    R2RCalibrationStatusResponse,
    R2RPreflightResponse,
    SchedulerCapability,
)

_DIGEST = "a" * 64
_ASSET_ID = f"asset:sha256:{_DIGEST}"
_R2R_TARGET_ASSET_ID = f"asset:sha256:{'b' * 64}"
_PLAN_ID = f"plan:sha256:{_DIGEST}"
_ARTIFACT_ID = "artifact:retargeted_motion:cli-test"
_NOW = datetime(2026, 8, 31, tzinfo=UTC)
_CALIBRATION_CANDIDATE_ID = f"cal-candidate:sha256:{'c' * 64}"


class FakeTransport:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, str, dict[str, Any], dict[str, Any] | None]] = []
        self.downloads: list[tuple[str, Path, ArtifactDescriptor, bool]] = []

    def request_json(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        document: dict[str, Any] | None = None,
    ) -> Any:
        self.requests.append((method, path, dict(query or {}), document))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json", exclude_none=True)
        return response

    def download_artifact(
        self,
        path: str,
        *,
        destination: Path,
        descriptor: ArtifactDescriptor,
        overwrite: bool,
    ) -> None:
        self.downloads.append((path, destination, descriptor, overwrite))
        destination.write_bytes(b"csv")


def _capabilities() -> CapabilityResponse:
    return CapabilityResponse(
        service_version="test",
        scheduler=SchedulerCapability(
            max_running_jobs=0,
            max_queued_jobs=0,
            mode="unlimited",
        ),
        supported_input_formats=["bvh"],
        supported_output_formats=["csv"],
        features={"agent_rest": True},
    )


def _job(job_id: str = "job_cli") -> AgentJobView:
    return AgentJobView(
        job_id=job_id,
        state="queued",
        progress=JobProgress(phase="queued", fraction=0.0),
        submitted_at=_NOW,
        cancellable=True,
        artifact_count=1,
    )


def _descriptor(job_id: str = "job_cli") -> ArtifactDescriptor:
    return ArtifactDescriptor(
        artifact_id=_ARTIFACT_ID,
        job_id=job_id,
        kind="retargeted_motion",
        format="csv",
        resource_uri=f"hhtools://jobs/{job_id}/artifacts/{_ARTIFACT_ID}",
        size_bytes=3,
        sha256=_DIGEST,
    )


def _available_catalog() -> AvailableAssetCatalogResponse:
    return AvailableAssetCatalogResponse(
        assets=[
            AvailableAssetCatalogEntry(
                root_id="source",
                relative_path="AMASS/walk.npz",
                display_name="Walk",
                kind="motion_bundle",
                category="plain_motion",
                dataset="amass",
                reference="smpl",
            )
        ],
        total=1,
        limit=5,
        offset=0,
    )


def _calibration_validation() -> CalibrationValidationReport:
    return CalibrationValidationReport(
        candidate_id=_CALIBRATION_CANDIDATE_ID,
        valid=True,
        score=0.95,
        changed_joint_count=2,
        mapped_slots=16,
        edge_errors_deg={"left_upper_arm": 4.0},
        checks=[
            PreflightCheck(
                code="CALIBRATION_POSE_ALIGNED",
                level="pass",
                message="Calibration pose is aligned.",
            )
        ],
    )


def _invoke(arguments: list[str], transport: Any, *, stdin: str = ""):
    stdout = io.StringIO()
    stderr = io.StringIO()
    selected: list[tuple[str, float]] = []

    def factory(base_url: str, timeout: float) -> FakeTransport:
        selected.append((base_url, timeout))
        return transport

    code = run(
        arguments,
        transport_factory=factory,
        stdin=io.StringIO(stdin),
        stdout=stdout,
        stderr=stderr,
    )
    # Exactly one JSON document and no Rich/progress prose on stdout.
    document = json.loads(stdout.getvalue())
    assert stdout.getvalue().count("\n") == 1
    assert stderr.getvalue() == ""
    return code, document, selected


def test_capabilities_is_one_contract_and_accepts_global_options_anywhere() -> None:
    transport = FakeTransport([_capabilities()])

    code, document, selected = _invoke(
        [
            "capabilities",
            "--json",
            "--timeout",
            "12.5",
            "--base-url",
            "http://127.0.0.1:9000/api/agent/v1",
        ],
        transport,
    )

    assert code == EXIT_SUCCESS
    assert document["schema_version"] == "1.0"
    assert document["service_version"] == "test"
    assert selected == [("http://127.0.0.1:9000/api/agent/v1", 12.5)]
    assert transport.requests == [("GET", "/capabilities", {}, None)]


@pytest.mark.parametrize(
    ("arguments", "command", "required_entry"),
    [
        (["--help"], "hhtools agent", "job"),
        (["job", "--help"], "hhtools agent job", "start"),
        (["job", "start", "--help"], "hhtools agent job start", "--plan"),
        (["job", "wait", "--help"], "hhtools agent job wait", "--after-revision"),
        (
            ["job", "lookup", "--help"],
            "hhtools agent job lookup",
            "--idempotency-key",
        ),
        (
            ["asset", "register", "-h"],
            "hhtools agent asset register",
            "--request",
        ),
        (
            ["artifact", "get", "--help"],
            "hhtools agent artifact get",
            "ARTIFACT_ID",
        ),
        (
            ["asset", "catalog", "--help"],
            "hhtools agent asset catalog",
            "--root-id",
        ),
        (
            ["calibration", "save", "--help"],
            "hhtools agent calibration save",
            "--request",
        ),
        (
            ["calibration", "r2r", "--help"],
            "hhtools agent calibration r2r",
            "status",
        ),
        (
            ["calibration", "r2r", "status", "--help"],
            "hhtools agent calibration r2r status",
            "--request",
        ),
    ],
)
def test_help_is_one_versioned_json_document_without_transport(
    arguments: list[str],
    command: str,
    required_entry: str,
) -> None:
    transport = FakeTransport([])

    code, document, selected = _invoke(arguments, transport)

    help_document = AgentCliHelp.model_validate(document)
    serialized = json.dumps(document)
    assert code == EXIT_SUCCESS
    assert selected == []
    assert help_document.kind == "agent_cli_help"
    assert help_document.command == command
    assert required_entry in serialized
    assert "--help" in help_document.usage
    assert transport.requests == []


@pytest.mark.parametrize(
    ("arguments", "reason_code", "command", "argument"),
    [
        ([], "MISSING_COMMAND", "hhtools agent", "COMMAND"),
        (["not-a-command"], "UNKNOWN_COMMAND", "hhtools agent", "COMMAND"),
        (["job"], "MISSING_COMMAND", "hhtools agent job", "COMMAND"),
        (["job", "not-a-command"], "UNKNOWN_COMMAND", "hhtools agent job", "COMMAND"),
        (["job", "start"], "MISSING_ARGUMENT", "hhtools agent job start", "--plan"),
        (
            ["job", "wait", "job_cli"],
            "MISSING_ARGUMENT",
            "hhtools agent job wait",
            "--after-revision",
        ),
        (
            ["job", "start", "--plan", _PLAN_ID],
            "MISSING_ARGUMENT",
            "hhtools agent job start",
            "--idempotency-key",
        ),
        (
            ["job", "lookup"],
            "MISSING_ARGUMENT",
            "hhtools agent job lookup",
            "--plan",
        ),
        (
            ["job", "lookup", "--plan", _PLAN_ID],
            "MISSING_ARGUMENT",
            "hhtools agent job lookup",
            "--idempotency-key",
        ),
        (
            ["artifact", "list"],
            "MISSING_ARGUMENT",
            "hhtools agent artifact list",
            "JOB_ID",
        ),
    ],
)
def test_parse_failures_have_sanitized_actionable_diagnostics(
    arguments: list[str],
    reason_code: str,
    command: str,
    argument: str,
) -> None:
    transport = FakeTransport([])

    code, document, selected = _invoke(arguments, transport)

    diagnostic = AgentCliArgumentDiagnostic.model_validate(document["details"])
    assert code == EXIT_PARAMETER_ERROR
    assert selected == []
    assert diagnostic.reason_code == reason_code
    assert diagnostic.command == command
    assert diagnostic.argument == argument
    assert diagnostic.expected
    assert diagnostic.usage.startswith(command)
    assert transport.requests == []


def test_request_validation_fails_before_transport_without_echoing_input() -> None:
    transport = FakeTransport([])
    raw = json.dumps(
        {
            "schema_version": "1.0",
            "root_id": "motion-library",
            "relative_path": "walk.bvh",
            "unexpected_secret": "do-not-echo",
        }
    )

    code, document, _selected = _invoke(
        ["asset", "register", "--request", "-", "--json"], transport, stdin=raw
    )

    assert code == EXIT_PARAMETER_ERROR
    assert document["code"] == "INVALID_PARAMETER"
    assert document["details"]["reason_code"] == "REQUEST_CONTRACT_INVALID"
    assert document["details"]["argument"] == "--request"
    assert "do-not-echo" not in json.dumps(document)
    assert transport.requests == []


def test_available_asset_catalog_uses_bounded_query_and_versioned_response() -> None:
    transport = FakeTransport([_available_catalog()])

    code, document, _selected = _invoke(
        [
            "asset",
            "catalog",
            "--root-id",
            "source",
            "--query",
            "walk",
            "--kind",
            "motion_bundle",
            "--limit",
            "5",
        ],
        transport,
    )

    assert code == EXIT_SUCCESS
    assert document["assets"][0]["relative_path"] == "AMASS/walk.npz"
    assert transport.requests == [
        (
            "GET",
            "/assets/available",
            {
                "root_id": "source",
                "query": "walk",
                "kind": "motion_bundle",
                "limit": 5,
                "offset": 0,
            },
            None,
        )
    ]


@pytest.mark.parametrize(
    ("arguments", "argument"),
    [
        (["asset", "catalog", "--limit", "501"], "--limit"),
        (["asset", "catalog", "--offset", "-1"], "--offset"),
        (["asset", "catalog", "--root-id", "../private"], "--root-id"),
        (["asset", "catalog", "--kind", "unknown"], "--kind"),
    ],
)
def test_available_asset_catalog_rejects_invalid_filters_before_transport(
    arguments: list[str],
    argument: str,
) -> None:
    transport = FakeTransport([])

    code, document, _selected = _invoke(arguments, transport)

    assert code == EXIT_PARAMETER_ERROR
    assert document["details"]["command"] == "hhtools agent asset catalog"
    assert document["details"]["argument"] == argument
    assert transport.requests == []


@pytest.mark.parametrize(
    ("arguments", "sensitive_value"),
    [
        (
            ["capabilities", "--unknown", r"C:\Users\Nora\secret.txt"],
            r"C:\Users\Nora\secret.txt",
        ),
        (
            [
                "asset",
                "register",
                "--request",
                r"C:\Users\Nora\does-not-exist.json",
            ],
            r"C:\Users\Nora\does-not-exist.json",
        ),
    ],
)
def test_cli_argument_errors_never_echo_argv_or_request_paths(
    arguments: list[str], sensitive_value: str
) -> None:
    transport = FakeTransport([])

    code, document, _selected = _invoke(arguments, transport)

    assert code == EXIT_PARAMETER_ERROR
    assert document["code"] == "INVALID_PARAMETER"
    assert document["message"] == "The Agent command arguments are invalid."
    assert document["details"]["reason_code"] in {
        "UNKNOWN_ARGUMENT",
        "REQUEST_FILE_UNREADABLE",
    }
    assert document["details"]["usage"].startswith("hhtools agent ")
    assert sensitive_value not in json.dumps(document)
    assert transport.requests == []


def test_unknown_option_never_echoes_option_or_following_secret() -> None:
    secret_option = "--access-token=agent-secret-value"
    secret_path = r"C:\Users\Nora\private\request.json"

    code, document, _selected = _invoke(
        ["capabilities", secret_option, secret_path],
        FakeTransport([]),
    )

    assert code == EXIT_PARAMETER_ERROR
    assert document["details"] == {
        "reason_code": "UNKNOWN_ARGUMENT",
        "command": "hhtools agent capabilities",
        "argument": "<unrecognized>",
        "expected": "Only the documented options and positional arguments are accepted.",
        "usage": (
            "hhtools agent capabilities [--json] [--base-url URL] [--timeout SECONDS] [--help]"
        ),
    }
    serialized = json.dumps(document)
    assert secret_option not in serialized
    assert secret_path not in serialized


def test_inline_json_is_diagnosed_as_request_file_misuse_without_echoing_it() -> None:
    inline = '{"api_key":"agent-secret-value"}'

    code, document, _selected = _invoke(
        ["asset", "register", "--request", inline],
        FakeTransport([]),
    )

    assert code == EXIT_PARAMETER_ERROR
    assert document["details"]["reason_code"] == "REQUEST_FILE_UNREADABLE"
    assert document["details"]["argument"] == "--request"
    assert "JSON_FILE_OR_DASH" in document["details"]["usage"]
    assert inline not in json.dumps(document)


def test_request_file_errors_distinguish_encoding_json_and_contract(
    tmp_path: Path,
) -> None:
    invalid_utf8 = tmp_path / "private-request.json"
    invalid_utf8.write_bytes(b"\xff\xfe")

    encoding_code, encoding_document, _ = _invoke(
        ["asset", "register", "--request", str(invalid_utf8)],
        FakeTransport([]),
    )
    json_code, json_document, _ = _invoke(
        ["asset", "register", "--request", "-"],
        FakeTransport([]),
        stdin="{not-json}",
    )
    contract_code, contract_document, _ = _invoke(
        ["asset", "register", "--request", "-"],
        FakeTransport([]),
        stdin='{"secret_field":"secret-value"}',
    )

    assert {encoding_code, json_code, contract_code} == {EXIT_PARAMETER_ERROR}
    assert encoding_document["details"]["reason_code"] == "REQUEST_ENCODING_INVALID"
    assert json_document["details"]["reason_code"] == "REQUEST_JSON_INVALID"
    assert contract_document["details"]["reason_code"] == "REQUEST_CONTRACT_INVALID"
    serialized = json.dumps([encoding_document, json_document, contract_document])
    assert str(invalid_utf8) not in serialized
    assert "secret_field" not in serialized
    assert "secret-value" not in serialized


@pytest.mark.parametrize(
    ("arguments", "argument"),
    [
        (
            [
                "job",
                "start",
                "--plan",
                "private-plan.json",
                "--idempotency-key",
                "valid-key",
            ],
            "--plan",
        ),
        (
            [
                "job",
                "start",
                "--plan",
                _PLAN_ID,
                "--idempotency-key",
                "secret key with spaces",
            ],
            "--idempotency-key",
        ),
    ],
)
def test_job_start_contract_errors_name_only_the_safe_argument(
    arguments: list[str],
    argument: str,
) -> None:
    code, document, _selected = _invoke(arguments, FakeTransport([]))

    assert code == EXIT_PARAMETER_ERROR
    assert document["details"]["reason_code"] == "INVALID_VALUE"
    assert document["details"]["command"] == "hhtools agent job start"
    assert document["details"]["argument"] == argument
    serialized = json.dumps(document)
    assert "private-plan.json" not in serialized
    assert "secret key with spaces" not in serialized


@pytest.mark.parametrize(
    ("arguments", "argument", "secret"),
    [
        (
            [
                "job",
                "lookup",
                "--plan",
                r"C:\Users\Nora\private-plan.json",
                "--idempotency-key",
                "recover-key",
            ],
            "--plan",
            r"C:\Users\Nora\private-plan.json",
        ),
        (
            [
                "job",
                "lookup",
                "--plan",
                _PLAN_ID,
                "--idempotency-key",
                "secret key with spaces",
            ],
            "--idempotency-key",
            "secret key with spaces",
        ),
        (
            [
                "job",
                "lookup",
                "--plan",
                _PLAN_ID,
                "--idempotency-key",
                "recover-key",
                "--after-revision",
                "-1",
            ],
            "--after-revision",
            "-1",
        ),
    ],
)
def test_job_lookup_contract_errors_never_echo_raw_identifiers(
    arguments: list[str],
    argument: str,
    secret: str,
) -> None:
    code, document, _selected = _invoke(arguments, FakeTransport([]))

    assert code == EXIT_PARAMETER_ERROR
    assert document["details"]["reason_code"] == "INVALID_VALUE"
    assert document["details"]["command"] == "hhtools agent job lookup"
    assert document["details"]["argument"] == argument
    assert secret not in json.dumps(document)


def test_job_lookup_rejects_unrelated_options_without_echoing_their_values() -> None:
    secret_path = r"C:\Users\Nora\private-output.csv"
    code, document, _selected = _invoke(
        [
            "job",
            "lookup",
            "--plan",
            _PLAN_ID,
            "--idempotency-key",
            "recover-key",
            "--output",
            secret_path,
        ],
        FakeTransport([]),
    )

    assert code == EXIT_PARAMETER_ERROR
    assert document["details"]["reason_code"] == "UNKNOWN_ARGUMENT"
    assert document["details"]["argument"] == "<unrecognized>"
    assert secret_path not in json.dumps(document)


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":"1.0","root_id":"a","root_id":"b","relative_path":"x"}',
        '{"schema_version":"1.0","root_id":"a","relative_path":"x","recursive":NaN}',
        '{"schema_version":"1.0","root_id":"a","relative_path":"x","recursive":Infinity}',
        '{"schema_version":"1.0","root_id":"a","relative_path":"x","recursive":1e9999}',
        '{"schema_version":"1.0","root_id":"a","relative_path":"x","recursive":' + "1" * 129 + "}",
        '{"schema_version":"1.0","root_id":"a","relative_path":"x","recursive":0.'
        + "1" * 129
        + "}",
    ],
)
def test_request_json_rejects_ambiguous_or_unbounded_json(raw: str) -> None:
    transport = FakeTransport([])

    code, document, _selected = _invoke(
        ["asset", "register", "--request", "-"], transport, stdin=raw
    )

    assert code == EXIT_PARAMETER_ERROR
    assert document["code"] == "INVALID_PARAMETER"
    assert transport.requests == []


def test_shared_strict_json_loader_also_protects_http_documents() -> None:
    with pytest.raises(StrictJsonError):
        loads_strict_json(b'{"code":"ONE","code":"TWO"}')
    with pytest.raises(StrictJsonError):
        loads_strict_json(b'{"value":-Infinity}')


def test_portable_output_guard_allows_controlled_resource_uris() -> None:
    ensure_portable_json(
        {
            "resource_uri": "hhtools://jobs/job_cli/artifacts/artifact:test:value",
            "documentation": "https://example.test/agent/v1",
            "message": ("See https://example.test/callback?next=/agent/v1 for details."),
        }
    )


@pytest.mark.parametrize(
    "value",
    [
        "prefix%2Fetc%2Fpasswd",
        "https://example.test/report?path=%252Fetc%252Fpasswd",
        "https://prefixC:%5CUsers%5CNora%5Csecret@example.test",
    ],
)
def test_portable_output_guard_rejects_encoded_or_userinfo_paths(value: str) -> None:
    with pytest.raises(agent_transport.PortableJsonError):
        ensure_portable_json({"message": value})


def test_deep_request_json_is_a_parameter_error_instead_of_an_internal_error() -> None:
    transport = FakeTransport([])
    raw = "[" * 2_000 + "0" + "]" * 2_000

    code, document, _selected = _invoke(
        ["asset", "register", "--request", "-"], transport, stdin=raw
    )

    assert code == EXIT_PARAMETER_ERROR
    assert document["code"] == "INVALID_PARAMETER"
    assert transport.requests == []


class _HttpResponse:
    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)

    def __enter__(self):
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


def test_http_transport_rejects_non_strict_success_json(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_transport,
        "urlopen",
        lambda *_args, **_kwargs: _HttpResponse(b'{"schema_version":"1.0","x":1,"x":2}'),
    )
    transport = HttpAgentTransport("http://127.0.0.1:8009/api/agent/v1")

    with pytest.raises(AgentTransportError) as caught:
        transport.request_json("GET", "/capabilities")

    assert caught.value.error.code == "REMOTE_PROTOCOL_ERROR"


def test_http_artifact_download_streams_and_verifies_before_atomic_publish(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"frame,root_x\n0,1.0\n"
    digest = hashlib.sha256(payload).hexdigest()
    descriptor = ArtifactDescriptor(
        artifact_id=_ARTIFACT_ID,
        job_id="job_cli",
        kind="retargeted_motion",
        format="csv",
        resource_uri=f"hhtools://jobs/job_cli/artifacts/{_ARTIFACT_ID}",
        size_bytes=len(payload),
        sha256=digest,
    )
    monkeypatch.setattr(
        agent_transport,
        "urlopen",
        lambda *_args, **_kwargs: _HttpResponse(payload),
    )
    transport = HttpAgentTransport("http://127.0.0.1:8009/api/agent/v1")
    destination = tmp_path / "result.csv"

    transport.download_artifact(
        "/jobs/job_cli/artifacts/artifact%3Aretargeted_motion%3Acli-test/content",
        destination=destination,
        descriptor=descriptor,
        overwrite=False,
    )

    assert destination.read_bytes() == payload
    assert list(tmp_path.glob("*.tmp")) == []


class _WriteFailingStream:
    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def __enter__(self) -> _WriteFailingStream:
        return self

    def __exit__(self, *_args: Any) -> None:
        self._stream.close()

    def write(self, _payload: bytes) -> int:
        raise OSError(r"C:\Users\Nora\private-output.csv")

    def flush(self) -> None:
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()


class _ReadFailingResponse(_HttpResponse):
    def read(self, size: int = -1) -> bytes:
        raise OSError("connection reset")


def test_http_artifact_response_read_failure_remains_a_service_error(
    tmp_path: Path, monkeypatch
) -> None:
    descriptor = _descriptor()
    monkeypatch.setattr(
        agent_transport,
        "urlopen",
        lambda *_args, **_kwargs: _ReadFailingResponse(b""),
    )
    transport = HttpAgentTransport("http://127.0.0.1:8009/api/agent/v1")

    with pytest.raises(AgentTransportError) as caught:
        transport.download_artifact(
            "/jobs/job_cli/artifacts/artifact/content",
            destination=tmp_path / "result.csv",
            descriptor=descriptor,
            overwrite=False,
        )

    assert caught.value.error.code == "AGENT_SERVICE_UNAVAILABLE"
    assert list(tmp_path.glob("*.tmp")) == []


def test_http_artifact_local_write_failure_is_not_a_service_error(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"frame,root_x\n0,1.0\n"
    descriptor = ArtifactDescriptor(
        artifact_id=_ARTIFACT_ID,
        job_id="job_cli",
        kind="retargeted_motion",
        format="csv",
        resource_uri=f"hhtools://jobs/job_cli/artifacts/{_ARTIFACT_ID}",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    original_fdopen = agent_transport.os.fdopen
    monkeypatch.setattr(
        agent_transport,
        "urlopen",
        lambda *_args, **_kwargs: _HttpResponse(payload),
    )
    monkeypatch.setattr(
        agent_transport.os,
        "fdopen",
        lambda fd, mode: _WriteFailingStream(original_fdopen(fd, mode)),
    )
    destination = tmp_path / "result.csv"
    transport = HttpAgentTransport("http://127.0.0.1:8009/api/agent/v1")

    with pytest.raises(AgentTransportError) as caught:
        transport.download_artifact(
            "/jobs/job_cli/artifacts/artifact/content",
            destination=destination,
            descriptor=descriptor,
            overwrite=False,
        )

    assert caught.value.error.code == "OUTPUT_WRITE_FAILED"
    assert caught.value.error.stage is ErrorStage.ARTIFACT
    assert "private-output" not in caught.value.error.model_dump_json()
    assert not destination.exists()
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    ("overwrite", "publish_operation"),
    [(False, "link"), (True, "replace")],
)
def test_http_artifact_publish_failure_has_stable_artifact_error(
    tmp_path: Path,
    monkeypatch,
    overwrite: bool,
    publish_operation: str,
) -> None:
    payload = b"csv"
    descriptor = ArtifactDescriptor(
        artifact_id=_ARTIFACT_ID,
        job_id="job_cli",
        kind="retargeted_motion",
        format="csv",
        resource_uri=f"hhtools://jobs/job_cli/artifacts/{_ARTIFACT_ID}",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(
        agent_transport,
        "urlopen",
        lambda *_args, **_kwargs: _HttpResponse(payload),
    )

    def fail_publish(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(r"C:\Users\Nora\private-output.csv")

    monkeypatch.setattr(agent_transport.os, publish_operation, fail_publish)
    destination = tmp_path / "result.csv"
    transport = HttpAgentTransport("http://127.0.0.1:8009/api/agent/v1")

    with pytest.raises(AgentTransportError) as caught:
        transport.download_artifact(
            "/jobs/job_cli/artifacts/artifact/content",
            destination=destination,
            descriptor=descriptor,
            overwrite=overwrite,
        )

    assert caught.value.error.code == "OUTPUT_WRITE_FAILED"
    assert caught.value.error.stage is ErrorStage.ARTIFACT
    assert "private-output" not in caught.value.error.model_dump_json()
    assert not destination.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_preflight_non_ready_is_structured_and_uses_preflight_exit_code() -> None:
    preflight = PreflightResponse(
        request_id="req_cli",
        status="rejected",
        error=ApiError(
            code="CALIBRATION_REQUIRED",
            message="Calibration is required.",
            stage=ErrorStage.PREFLIGHT,
        ),
    )
    request = {
        "schema_version": "1.0",
        "motion_asset_id": _ASSET_ID,
        "robot_id": "g1_29dof",
        "robot_asset_id": _ASSET_ID,
    }
    transport = FakeTransport([preflight])

    code, document, _selected = _invoke(
        ["preflight", "retarget", "--request", "-"],
        transport,
        stdin=json.dumps(request),
    )

    assert code == EXIT_PREFLIGHT_ERROR
    assert document["status"] == "rejected"
    assert document["error"]["code"] == "CALIBRATION_REQUIRED"
    assert transport.requests[0][0:2] == ("POST", "/preflight/retarget")
    assert transport.requests[0][3] == request | {
        "output_format": "csv",
        "output_policy": "create_new",
        "parameters": {},
    }


def test_r2r_preflight_uses_the_same_strict_json_cli_boundary() -> None:
    preflight = R2RPreflightResponse(
        request_id="req_r2r_cli",
        status="rejected",
        recommended_backend="newton",
        error=ApiError(
            code="R2R_CALIBRATION_REQUIRED",
            message="Pair calibration is required.",
            stage=ErrorStage.PREFLIGHT,
        ),
    )
    request = {
        "schema_version": "1.0",
        "trajectory_asset_id": _ASSET_ID,
        "source_robot_id": "source_bot",
        "source_robot_asset_id": f"asset:sha256:{'b' * 64}",
        "target_robot_id": "target_bot",
        "target_robot_asset_id": f"asset:sha256:{'c' * 64}",
    }
    transport = FakeTransport([preflight])

    code, document, _selected = _invoke(
        ["preflight", "r2r", "--request", "-"],
        transport,
        stdin=json.dumps(request),
    )

    assert code == EXIT_PREFLIGHT_ERROR
    assert document["error"]["code"] == "R2R_CALIBRATION_REQUIRED"
    assert transport.requests[0][0:2] == ("POST", "/preflight/r2r")
    assert transport.requests[0][3] == request | {
        "output_format": "csv",
        "output_policy": "create_new",
        "parameters": {},
    }


def test_batch_preflight_preserves_ordered_child_plan_ids() -> None:
    response = BatchPreflightResponse(
        request_id="req_batch_cli",
        status="rejected",
        error=ApiError(
            code="PLAN_NOT_FOUND",
            message="Child plan unavailable.",
            stage=ErrorStage.PREFLIGHT,
        ),
    )
    plans = [f"plan:sha256:{'a' * 64}", f"plan:sha256:{'b' * 64}"]
    request = {"schema_version": "1.0", "workflow": "h2r", "item_plan_ids": plans}
    transport = FakeTransport([response])

    code, document, _selected = _invoke(
        ["preflight", "batch", "--request", "-"],
        transport,
        stdin=json.dumps(request),
    )

    assert code == EXIT_PREFLIGHT_ERROR
    assert document["error"]["code"] == "PLAN_NOT_FOUND"
    assert transport.requests[0][0:2] == ("POST", "/preflight/batch")
    assert transport.requests[0][3]["item_plan_ids"] == plans
    assert transport.requests[0][3]["output_policy"] == "create_new"


def test_calibration_commands_share_the_strict_request_contract() -> None:
    identity = {
        "schema_version": "1.0",
        "robot_id": "g1_29dof",
        "robot_asset_id": _ASSET_ID,
        "reference": "smplx",
    }
    candidate = CalibrationCandidate(
        candidate_id=_CALIBRATION_CANDIDATE_ID,
        robot_id="g1_29dof",
        robot_asset_id=_ASSET_ID,
        robot_digest=_DIGEST,
        reference="smplx",
        baseline="urdf_zero",
        joint_q={"left_shoulder_roll_joint": 1.2},
    )
    validation = _calibration_validation()
    responses = [
        CalibrationStatusResponse(
            request_id="req_calibration_cli",
            state="missing",
            robot_id="g1_29dof",
            robot_asset_id=_ASSET_ID,
            robot_digest=_DIGEST,
            reference="smplx",
            source="none",
            joint_count=0,
            mapped_slots=16,
            can_propose=True,
            can_silent_save=True,
        ),
        CalibrationProposalResponse(candidate=candidate, validation=validation),
        validation,
        CalibrationSaveReceipt(
            candidate_id=_CALIBRATION_CANDIDATE_ID,
            calibration_id=f"cal:sha256:{'d' * 64}",
            calibration_digest="d" * 64,
            robot_id="g1_29dof",
            reference="smplx",
            save_mode="validated_silent",
            validation=validation,
        ),
    ]
    transport = FakeTransport(responses)

    status_code, status, _selected = _invoke(
        ["calibration", "status", "--request", "-"],
        transport,
        stdin=json.dumps(identity),
    )
    proposal_code, proposal, _selected = _invoke(
        ["calibration", "propose", "--request", "-"],
        transport,
        stdin=json.dumps(identity),
    )
    candidate_request = {
        "schema_version": "1.0",
        "candidate_id": _CALIBRATION_CANDIDATE_ID,
    }
    validation_code, checked, _selected = _invoke(
        ["calibration", "validate", "--request", "-"],
        transport,
        stdin=json.dumps(candidate_request),
    )
    save_request = candidate_request | {"save_mode": "validated_silent"}
    save_code, saved, _selected = _invoke(
        ["calibration", "save", "--request", "-"],
        transport,
        stdin=json.dumps(save_request),
    )

    assert [status_code, proposal_code, validation_code, save_code] == [0, 0, 0, 0]
    assert status["state"] == "missing"
    assert proposal["candidate"]["candidate_id"] == _CALIBRATION_CANDIDATE_ID
    assert checked["valid"] is True
    assert saved["saved"] is True
    assert [request[1] for request in transport.requests] == [
        "/calibrations/status",
        "/calibrations/proposals",
        "/calibrations/validate",
        "/calibrations/save",
    ]


def test_r2r_calibration_commands_use_pair_contracts_and_routes() -> None:
    identity = {
        "schema_version": "1.0",
        "source_robot_id": "source_bot",
        "source_robot_asset_id": _ASSET_ID,
        "target_robot_id": "target_bot",
        "target_robot_asset_id": _R2R_TARGET_ASSET_ID,
    }
    candidate = R2RCalibrationCandidate(
        candidate_id=_CALIBRATION_CANDIDATE_ID,
        source_robot_id="source_bot",
        source_robot_asset_id=_ASSET_ID,
        source_robot_digest=_DIGEST,
        target_robot_id="target_bot",
        target_robot_asset_id=_R2R_TARGET_ASSET_ID,
        target_robot_digest="b" * 64,
        baseline="urdf_zero",
        joint_q={"left_shoulder_roll_joint": 1.2},
    )
    validation = _calibration_validation()
    responses = [
        R2RCalibrationStatusResponse(
            request_id="req_r2r_calibration_cli",
            state="missing",
            source_robot_id="source_bot",
            source_robot_asset_id=_ASSET_ID,
            source_robot_digest=_DIGEST,
            target_robot_id="target_bot",
            target_robot_asset_id=_R2R_TARGET_ASSET_ID,
            target_robot_digest="b" * 64,
            storage="none",
            joint_count=0,
            source_mapped_slots=16,
            target_mapped_slots=16,
            can_propose=True,
            can_silent_save=True,
        ),
        R2RCalibrationProposalResponse(candidate=candidate, validation=validation),
        validation,
        R2RCalibrationSaveReceipt(
            candidate_id=_CALIBRATION_CANDIDATE_ID,
            calibration_id=f"cal:sha256:{'e' * 64}",
            calibration_digest="e" * 64,
            source_robot_id="source_bot",
            target_robot_id="target_bot",
            save_mode="validated_silent",
            validation=validation,
        ),
    ]
    transport = FakeTransport(responses)

    status_code, status, _ = _invoke(
        ["calibration", "r2r", "status", "--request", "-"],
        transport,
        stdin=json.dumps(identity),
    )
    proposal_code, proposal, _ = _invoke(
        ["calibration", "r2r", "propose", "--request", "-"],
        transport,
        stdin=json.dumps(identity),
    )
    candidate_request = {
        "schema_version": "1.0",
        "candidate_id": _CALIBRATION_CANDIDATE_ID,
    }
    validation_code, checked, _ = _invoke(
        ["calibration", "r2r", "validate", "--request", "-"],
        transport,
        stdin=json.dumps(candidate_request),
    )
    save_code, saved, _ = _invoke(
        ["calibration", "r2r", "save", "--request", "-"],
        transport,
        stdin=json.dumps(candidate_request | {"save_mode": "validated_silent"}),
    )

    assert [status_code, proposal_code, validation_code, save_code] == [0, 0, 0, 0]
    assert status["state"] == "missing"
    assert proposal["candidate"]["workflow"] == "r2r"
    assert checked["valid"] is True
    assert saved["saved"] is True
    assert [request[1] for request in transport.requests] == [
        "/calibrations/r2r/status",
        "/calibrations/r2r/proposals",
        "/calibrations/r2r/validate",
        "/calibrations/r2r/save",
    ]


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        (ErrorStage.REQUEST, EXIT_PARAMETER_ERROR),
        (ErrorStage.CALIBRATION, EXIT_PREFLIGHT_ERROR),
        (ErrorStage.PREFLIGHT, EXIT_PREFLIGHT_ERROR),
        (ErrorStage.ADMISSION, EXIT_JOB_ERROR),
        (ErrorStage.ARTIFACT, EXIT_JOB_ERROR),
        (ErrorStage.INTERNAL, EXIT_INTERNAL_ERROR),
    ],
)
def test_service_errors_have_stable_exit_code_mapping(stage: ErrorStage, expected: int) -> None:
    error = AgentTransportError(
        ApiError(code="TEST_ERROR", message="Expected failure.", stage=stage)
    )

    code, document, _selected = _invoke(["capabilities"], FakeTransport([error]))

    assert code == expected
    assert document["code"] == "TEST_ERROR"
    assert document["stage"] == stage.value


@pytest.mark.parametrize(
    "unsafe_value",
    [
        r"debug:C:\Users\Nora\server-secret.log",
        r"path[\\private-host\share\server-secret.log]",
        "debug:/srv/hhtools/private/result.csv",
        r"path[C:\Users\Nora\server-secret.log]",
        "path{/srv/hhtools/private/result.csv}",
        "path|/srv/hhtools/private/result.csv",
        "path[//private-host/share]",
        r"https://example.test/path]C:\Users\Nora\server-secret.log",
        r"https://example.test,C:\Users\Nora\server-secret.log",
        "hhtools://jobs/job_cli/artifacts/report|/srv/hhtools/private/result.csv",
    ],
)
def test_valid_remote_api_error_with_host_path_fails_closed_on_stdout(
    unsafe_value: str,
) -> None:
    remote_error = AgentTransportError(
        ApiError(
            code="REMOTE_FAILURE",
            message="The remote service failed.",
            stage=ErrorStage.INTERNAL,
            details={"debug_path": unsafe_value},
        )
    )

    code, document, _selected = _invoke(["capabilities"], FakeTransport([remote_error]))

    assert code == EXIT_INTERNAL_ERROR
    assert document["code"] == "INTERNAL_ERROR"
    assert document["message"] == ("The Agent response was not safe for portable JSON output.")
    assert document["details"] == {}


def test_valid_remote_success_contract_with_host_path_fails_closed_on_stdout() -> None:
    descriptor = _descriptor().model_copy(
        update={"metadata": {"debug_path": "debug:/srv/hhtools/private/result.csv"}}
    )

    code, document, _selected = _invoke(
        ["artifact", "get", "job_cli", _ARTIFACT_ID],
        FakeTransport([descriptor]),
    )

    assert code == EXIT_INTERNAL_ERROR
    assert document["code"] == "INTERNAL_ERROR"
    assert document["details"] == {}


def test_job_commands_use_public_requests_and_versioned_routes() -> None:
    transport = FakeTransport([_job(), _job(), _job(), _job(), _job(), _job("job_retry")])

    start_code, _, _ = _invoke(
        ["job", "start", "--plan", _PLAN_ID, "--idempotency-key", "cli:start-1"],
        transport,
    )
    lookup_code, lookup_document, _ = _invoke(
        [
            "job",
            "lookup",
            "--plan",
            _PLAN_ID,
            "--idempotency-key",
            "cli:start-1",
            "--after-revision",
            "7",
        ],
        transport,
    )
    get_code, _, _ = _invoke(["job", "get", "job_cli", "--after-revision", "0"], transport)
    wait_code, _, _ = _invoke(
        [
            "job",
            "wait",
            "job_cli",
            "--after-revision",
            "0",
            "--wait-timeout",
            "5",
        ],
        transport,
    )
    cancel_code, _, _ = _invoke(["job", "cancel", "job_cli"], transport)
    retry_code, _, _ = _invoke(
        ["job", "retry", "job_cli", "--idempotency-key", "cli:retry-1"],
        transport,
    )

    assert {start_code, lookup_code, get_code, wait_code, cancel_code, retry_code} == {EXIT_SUCCESS}
    assert lookup_document["job_id"] == "job_cli"
    assert transport.requests == [
        (
            "POST",
            "/jobs",
            {},
            {
                "schema_version": "1.0",
                "plan_id": _PLAN_ID,
                "idempotency_key": "cli:start-1",
            },
        ),
        (
            "POST",
            "/jobs/lookup",
            {},
            {
                "schema_version": "1.0",
                "plan_id": _PLAN_ID,
                "idempotency_key": "cli:start-1",
                "after_revision": 7,
            },
        ),
        ("GET", "/jobs/job_cli", {"after_revision": 0}, None),
        (
            "GET",
            "/jobs/job_cli/wait",
            {"after_revision": 0, "timeout": 5.0},
            None,
        ),
        ("POST", "/jobs/job_cli/cancel", {}, {}),
        (
            "POST",
            "/jobs/job_cli/retry",
            {},
            {"schema_version": "1.0", "idempotency_key": "cli:retry-1"},
        ),
    ]


def test_job_lookup_omits_the_optional_revision_when_not_supplied() -> None:
    transport = FakeTransport([_job()])

    code, document, _selected = _invoke(
        [
            "job",
            "lookup",
            "--plan",
            _PLAN_ID,
            "--idempotency-key",
            "cli:recover-without-revision",
        ],
        transport,
    )

    assert code == EXIT_SUCCESS
    assert document["job_id"] == "job_cli"
    assert transport.requests == [
        (
            "POST",
            "/jobs/lookup",
            {},
            {
                "schema_version": "1.0",
                "plan_id": _PLAN_ID,
                "idempotency_key": "cli:recover-without-revision",
            },
        )
    ]


@pytest.mark.parametrize("wait_timeout", ["-0.1", "60.1", "nan", "inf"])
def test_job_wait_rejects_invalid_timeout_without_transport(wait_timeout: str) -> None:
    code, document, selected = _invoke(
        [
            "job",
            "wait",
            "job_cli",
            "--after-revision",
            "0",
            "--wait-timeout",
            wait_timeout,
        ],
        FakeTransport([]),
    )

    assert code == EXIT_PARAMETER_ERROR
    assert document["details"]["argument"] == "--wait-timeout"
    assert selected == []


def test_job_wait_requires_request_timeout_longer_than_server_wait() -> None:
    code, document, selected = _invoke(
        [
            "job",
            "wait",
            "job_cli",
            "--after-revision",
            "0",
            "--wait-timeout",
            "20",
            "--timeout",
            "20",
        ],
        FakeTransport([]),
    )

    assert code == EXIT_PARAMETER_ERROR
    assert document["details"]["reason_code"] == "INVALID_COMBINATION"
    assert document["details"]["argument"] == "--timeout"
    assert selected == []


def test_artifact_get_requires_job_membership_and_never_writes_bytes_to_json(
    tmp_path: Path,
) -> None:
    descriptor = _descriptor()
    page = ArtifactListResponse(
        job_id="job_cli", artifacts=[descriptor], total=1, limit=100, offset=0
    )
    transport = FakeTransport([page, descriptor])
    destination = tmp_path / "motion.csv"

    list_code, list_document, _ = _invoke(["artifact", "list", "job_cli"], transport)
    get_code, get_document, _ = _invoke(
        [
            "artifact",
            "get",
            "job_cli",
            _ARTIFACT_ID,
            "--verify",
            "--output",
            str(destination),
        ],
        transport,
    )

    assert list_code == get_code == EXIT_SUCCESS
    assert list_document["artifacts"][0]["artifact_id"] == _ARTIFACT_ID
    assert get_document == descriptor.model_dump(mode="json", exclude_none=True)
    assert "output" not in get_document
    assert "base64" not in json.dumps(get_document).casefold()
    assert destination.read_bytes() == b"csv"
    assert transport.requests[1][2] == {"verify": True}
    assert transport.downloads[0][0] == (
        "/jobs/job_cli/artifacts/artifact%3Aretargeted_motion%3Acli-test/content"
    )


def test_artifact_descriptor_identity_mismatch_fails_before_download() -> None:
    transport = FakeTransport([_descriptor(job_id="job_other")])

    code, document, _ = _invoke(["artifact", "get", "job_cli", _ARTIFACT_ID], transport)

    assert code == EXIT_INTERNAL_ERROR
    assert document["code"] == "REMOTE_PROTOCOL_ERROR"
    assert transport.downloads == []


def test_typer_adapter_passes_the_raw_tail_to_strict_json_runner(monkeypatch) -> None:
    transport = FakeTransport([_capabilities()])
    monkeypatch.setattr(
        agent_cli,
        "_default_transport_factory",
        lambda _base_url, _timeout: transport,
    )

    result = CliRunner().invoke(agent_cli.app, ["capabilities", "--json"])

    assert result.exit_code == EXIT_SUCCESS
    assert json.loads(result.stdout)["service_version"] == "test"
    assert result.stdout.count("\n") == 1


def test_typer_parameter_failure_is_one_api_error_without_rich_usage() -> None:
    result = CliRunner().invoke(agent_cli.app, ["job", "start", "--json"])

    assert result.exit_code == EXIT_PARAMETER_ERROR
    assert json.loads(result.stdout)["code"] == "INVALID_PARAMETER"
    assert result.stdout.count("\n") == 1
    assert result.stderr == ""


def test_installed_hhtools_agent_group_keeps_success_and_parse_errors_json(
    monkeypatch,
) -> None:
    # Importing the project root proves the public ``hhtools agent`` group is
    # installed, rather than only exercising this module's inner Typer object.
    from hhtools.cli.main import app as hhtools_app

    transport = FakeTransport([_capabilities()])
    monkeypatch.setattr(
        agent_cli,
        "_default_transport_factory",
        lambda _base_url, _timeout: transport,
    )
    runner = CliRunner()

    success = runner.invoke(hhtools_app, ["agent", "capabilities", "--json"])
    help_result = runner.invoke(hhtools_app, ["agent", "job", "start", "--help"])
    r2r_help = runner.invoke(
        hhtools_app,
        ["agent", "calibration", "r2r", "--help"],
    )
    failure = runner.invoke(hhtools_app, ["agent", "job", "start", "--json"])

    assert success.exit_code == EXIT_SUCCESS
    assert json.loads(success.stdout)["service_version"] == "test"
    assert success.stdout.count("\n") == 1
    assert success.stderr == ""
    assert help_result.exit_code == EXIT_SUCCESS
    assert json.loads(help_result.stdout)["command"] == "hhtools agent job start"
    assert help_result.stdout.count("\n") == 1
    assert help_result.stderr == ""
    assert r2r_help.exit_code == EXIT_SUCCESS
    r2r_help_document = AgentCliHelp.model_validate_json(r2r_help.stdout)
    assert r2r_help_document.command == "hhtools agent calibration r2r"
    assert {item.name for item in r2r_help_document.subcommands} == {
        "status",
        "propose",
        "validate",
        "save",
    }
    assert r2r_help.stderr == ""
    assert failure.exit_code == EXIT_PARAMETER_ERROR
    assert json.loads(failure.stdout)["code"] == "INVALID_PARAMETER"
    assert failure.stdout.count("\n") == 1
    assert failure.stderr == ""

    unknown = runner.invoke(hhtools_app, ["agent", "not-a-command"])
    assert unknown.exit_code == EXIT_PARAMETER_ERROR
    assert json.loads(unknown.stdout)["code"] == "INVALID_PARAMETER"
    assert unknown.stdout.count("\n") == 1
    assert unknown.stderr == ""


def test_installed_agent_accepts_connection_options_before_operation(monkeypatch) -> None:
    from hhtools.cli.main import app as hhtools_app

    def unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("connection refused")

    monkeypatch.setattr(agent_transport, "urlopen", unavailable)
    result = CliRunner().invoke(
        hhtools_app,
        [
            "agent",
            "--base-url",
            "http://127.0.0.1:1/api/agent/v1",
            "--timeout",
            "0.1",
            "capabilities",
        ],
    )

    assert result.exit_code == EXIT_INTERNAL_ERROR
    assert json.loads(result.stdout)["code"] == "AGENT_SERVICE_UNAVAILABLE"
    assert result.stdout.count("\n") == 1
    assert result.stderr == ""


def test_legacy_upgrade_wraps_raw_v1_before_rest() -> None:
    preflight = PreflightResponse(
        request_id="req_legacy_cli",
        status="rejected",
        error=ApiError(
            code="CALIBRATION_REQUIRED",
            message="Calibration is required.",
            stage=ErrorStage.PREFLIGHT,
        ),
    )
    raw_v1 = {
        "schema_version": 1,
        "kind": "retarget",
        "request": {"source_path": "C:/allowed/walk.bvh", "robot": "g1_29dof"},
    }
    transport = FakeTransport(
        [{"schema_version": "1.0", "preflight": preflight.model_dump(mode="json")}]
    )

    code, document, _ = _invoke(
        ["legacy", "upgrade", "--request", "-"],
        transport,
        stdin=json.dumps(raw_v1),
    )

    assert code == EXIT_PREFLIGHT_ERROR
    assert document["preflight"]["status"] == "rejected"
    assert transport.requests[0] == (
        "POST",
        "/legacy/jobspec-v1/upgrade",
        {},
        {"schema_version": "1.0", "payload": raw_v1},
    )


class _ParityJobManager:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.descriptor = _descriptor()

    def start_retarget(self, plan_id: str, *, idempotency_key: str) -> AgentJobView:
        self.calls.append(("start", plan_id, idempotency_key))
        return _job()

    def get_job(self, job_id: str, *, after_revision: int | None = None) -> AgentJobView:
        self.calls.append(("get", job_id, after_revision))
        return _job(job_id)

    def wait_job(
        self,
        job_id: str,
        *,
        after_revision: int,
        timeout: float = 30.0,
    ) -> AgentJobView:
        self.calls.append(("wait", job_id, after_revision, timeout))
        return _job(job_id)

    def lookup_job(
        self,
        plan_id: str,
        *,
        idempotency_key: str,
        after_revision: int | None = None,
    ) -> AgentJobView:
        self.calls.append(("lookup", plan_id, idempotency_key, after_revision))
        return _job()

    def cancel_job(self, job_id: str) -> AgentJobView:
        self.calls.append(("cancel", job_id))
        return _job(job_id)

    def retry_job(self, job_id: str, *, idempotency_key: str) -> AgentJobView:
        self.calls.append(("retry", job_id, idempotency_key))
        return _job("job_retry")

    def list_artifacts(
        self, job_id: str, *, offset: int = 0, limit: int = 100
    ) -> list[ArtifactDescriptor]:
        self.calls.append(("list_artifacts", job_id, offset, limit))
        return [self.descriptor]

    def get_artifact(self, job_id: str, artifact_id: str, *, verify: bool = False) -> Any:
        self.calls.append(("get_artifact", job_id, artifact_id, verify))
        return SimpleNamespace(descriptor=self.descriptor)


class _TestClientTransport:
    """Adapter used only to exercise CLI -> real router -> fake service."""

    def __init__(self, client: TestClient) -> None:
        self.client = client

    def request_json(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        document: dict[str, Any] | None = None,
    ) -> Any:
        response = self.client.request(
            method,
            f"/api/agent/v1{path}",
            params={key: value for key, value in (query or {}).items() if value is not None},
            json=document,
        )
        payload = response.json()
        if response.status_code >= 400:
            raise AgentTransportError(ApiError.model_validate(payload))
        return payload

    def download_artifact(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("parity test does not download artifact bytes")


def test_cli_rest_service_parity_for_jobs_and_artifact_membership() -> None:
    manager = _ParityJobManager()
    app = FastAPI()
    app.state.agent_job_manager = manager
    app.include_router(agent_router)
    client = TestClient(app)
    transport = _TestClientTransport(client)

    start_args = [
        "job",
        "start",
        "--plan",
        _PLAN_ID,
        "--idempotency-key",
        "cli:parity-start",
    ]
    start_code, start_document, _ = _invoke(start_args, transport)
    direct_start = client.post(
        "/api/agent/v1/jobs",
        json={
            "schema_version": "1.0",
            "plan_id": _PLAN_ID,
            "idempotency_key": "cli:parity-start",
        },
    )
    assert start_code == EXIT_SUCCESS
    assert start_document == direct_start.json()
    assert manager.calls[:2] == [
        ("start", _PLAN_ID, "cli:parity-start"),
        ("start", _PLAN_ID, "cli:parity-start"),
    ]

    manager.calls.clear()
    lookup_args = [
        "job",
        "lookup",
        "--plan",
        _PLAN_ID,
        "--idempotency-key",
        "cli:parity-start",
        "--after-revision",
        "7",
    ]
    lookup_code, lookup_document, _ = _invoke(lookup_args, transport)
    direct_lookup = client.post(
        "/api/agent/v1/jobs/lookup",
        json={
            "schema_version": "1.0",
            "plan_id": _PLAN_ID,
            "idempotency_key": "cli:parity-start",
            "after_revision": 7,
        },
    )
    assert lookup_code == EXIT_SUCCESS
    assert lookup_document == direct_lookup.json()
    assert manager.calls == [
        ("lookup", _PLAN_ID, "cli:parity-start", 7),
        ("lookup", _PLAN_ID, "cli:parity-start", 7),
    ]

    manager.calls.clear()
    get_code, get_document, _ = _invoke(
        ["job", "get", "job_cli", "--after-revision", "0"], transport
    )
    direct_get = client.get("/api/agent/v1/jobs/job_cli", params={"after_revision": 0})
    assert get_code == EXIT_SUCCESS
    assert get_document == direct_get.json()
    assert manager.calls == [("get", "job_cli", 0), ("get", "job_cli", 0)]

    manager.calls.clear()
    wait_code, wait_document, _ = _invoke(
        [
            "job",
            "wait",
            "job_cli",
            "--after-revision",
            "7",
            "--wait-timeout",
            "5",
        ],
        transport,
    )
    direct_wait = client.get(
        "/api/agent/v1/jobs/job_cli/wait",
        params={"after_revision": 7, "timeout": 5},
    )
    assert wait_code == EXIT_SUCCESS
    assert wait_document == direct_wait.json()
    assert manager.calls == [
        ("wait", "job_cli", 7, 5.0),
        ("wait", "job_cli", 7, 5.0),
    ]

    manager.calls.clear()
    list_code, list_document, _ = _invoke(
        ["artifact", "list", "job_cli", "--limit", "25", "--offset", "0"],
        transport,
    )
    direct_list = client.get(
        "/api/agent/v1/jobs/job_cli/artifacts",
        params={"limit": 25, "offset": 0},
    )
    assert list_code == EXIT_SUCCESS
    assert list_document == direct_list.json()
    assert manager.calls == [
        ("list_artifacts", "job_cli", 0, 25),
        ("get", "job_cli", None),
        ("list_artifacts", "job_cli", 0, 25),
        ("get", "job_cli", None),
    ]

    manager.calls.clear()
    descriptor_code, descriptor_document, _ = _invoke(
        ["artifact", "get", "job_cli", _ARTIFACT_ID, "--verify"], transport
    )
    direct_descriptor = client.get(
        f"/api/agent/v1/jobs/job_cli/artifacts/{_ARTIFACT_ID}",
        params={"verify": True},
    )
    assert descriptor_code == EXIT_SUCCESS
    assert descriptor_document == direct_descriptor.json()
    assert manager.calls == [
        ("get_artifact", "job_cli", _ARTIFACT_ID, True),
        ("get_artifact", "job_cli", _ARTIFACT_ID, True),
    ]
