from __future__ import annotations

import base64
import json

import pytest

from hhtools.contracts.portability import (
    MAX_INLINE_BASE64_DECODED_BYTES,
    MAX_PORTABLE_CONTAINER_ITEMS,
    MAX_PORTABLE_DEPTH,
    MAX_PORTABLE_DOCUMENT_BYTES,
    MAX_PORTABLE_DOCUMENT_STRING_BYTES,
    MAX_PORTABLE_STRING_BYTES,
    PortableJsonError,
    looks_like_host_path,
    validate_portable_json,
)


@pytest.mark.parametrize(
    "value",
    [
        "C:\\Users\\Nora\\secret.txt",
        "/srv/hhtools/private.json",
        "\\\\server\\share\\private.bin",
        "file:///etc/passwd",
        "data:application/octet-stream;base64,SGVsbG8=",
        "%2Fetc%2Fpasswd",
        "%252Fetc%252Fpasswd",
        "prefix/etc/passwd",
        "prefix%2Fetc%2Fpasswd",
        "prefix/usr/local/bin/tool",
        "prefix/data/model.bin",
        "prefix/workspace/model.bin",
        "prefix/proc/self/environ",
        "prefixC:\\Users\\Nora\\secret.txt",
        "%43%3A%5CUsers%5CNora%5Csecret.txt",
        "file%3A%2F%2F%2Fetc%2Fpasswd",
        "%FF",
        "https://example.invalid/report?path=%FF",
        "https://C:%5CUsers%5CNora%5Csecret.txt@example.invalid/report",
        "https://example.invalid:abc/usr/local/bin",
        "hhtools://capabilities:abc/usr/local/bin",
        "hhtools://capabilities:99999",
        "hhtools://[abc]/x",
        "https://[::::]/docs",
        "See https://example.invalid/docs|/etc/passwd",
        "https://example.invalid/docs]/srv/secret",
        "https://example.invalid/docs`/srv/secret",
        "https://example.invalid/report?path=/etc/passwd",
        "https://example.invalid/report?path=%2Fetc%2Fpasswd",
        "https://example.invalid/report?path=%252Fetc%252Fpasswd",
        "https://example.invalid/report#C:%5CUsers%5CNora%5Csecret.txt",
        "hhtools://jobs/job:one/status?path=%2Fetc%2Fpasswd",
        "hhtools://jobs/job:one/status#file%3A%2F%2F%2Fetc%2Fpasswd",
    ],
)
def test_host_paths_and_encoded_uri_payloads_are_rejected(value: str) -> None:
    assert looks_like_host_path(value)
    with pytest.raises(PortableJsonError, match="host path"):
        validate_portable_json({"value": value})


@pytest.mark.parametrize(
    "value",
    [
        "hhtools://capabilities",
        "hhtools://schemas/agent/v1/api-error",
        "hhtools://robots/g1",
        "hhtools://assets/asset:sha256:abcd/manifest",
        "hhtools://plans/plan:sha256:abcd",
        "hhtools://jobs/job:one/status",
        "hhtools://jobs/job:one/artifacts/artifact:preview:one",
        "https://example.invalid/docs/agent/v1",
        "https://[2001:db8::1]/docs",
        "https://example.invalid/docs?next=https%3A%2F%2Fdocs.example%2Fguide",
        "https://example.invalid/callback?next=/agent/v1",
        "See https://example.invalid/callback?next=/agent/v1 for details.",
    ],
)
def test_portable_resources_and_normal_text_are_allowed(value: str) -> None:
    assert not looks_like_host_path(value)
    validate_portable_json({"value": value})


def test_normal_long_prose_is_not_mistaken_for_base64() -> None:
    value = "ordinary prose may be comfortably longer than a Base64 digest " * 2_000

    assert not looks_like_host_path(value)
    validate_portable_json({"value": value})


@pytest.mark.parametrize(
    "url",
    [
        "/?panel=h2r&calibrate=smplx",
        "http://127.0.0.1:8009/?panel=h2r&calibrate=smplx",
        "http://localhost:8009/?view=calibration",
        "http://[::1]:8009/?view=calibration",
    ],
)
def test_controlled_loopback_next_action_is_allowed(url: str) -> None:
    validate_portable_json({"actor": "human", "action": "open_calibration_ui", "url": url})


def test_next_action_without_a_url_is_allowed() -> None:
    validate_portable_json({"actor": "agent", "action": "continue_polling", "url": None})


def test_https_documentation_next_action_is_allowed() -> None:
    validate_portable_json(
        {
            "actor": "human",
            "action": "read_documentation",
            "url": "https://docs.example.invalid/guide?next=/agent/v1",
        }
    )


def test_unhashable_actor_is_validated_as_ordinary_json() -> None:
    validate_portable_json({"actor": {"name": "human"}, "action": "describe"})


def test_invalid_unicode_is_normalized_to_a_portable_error() -> None:
    with pytest.raises(PortableJsonError, match="UTF-8"):
        validate_portable_json({"value": "\ud800"})


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8009/?return=%2Fetc%2Fpasswd",
        "http://127.0.0.1:8009/?return=file%253A%252F%252F%252Fetc%252Fpasswd",
        "http://127.0.0.1:8009/#C:%5CUsers%5CNora",
        "http://example.invalid:8009/?view=calibration",
    ],
)
def test_next_action_cannot_expand_the_loopback_or_path_boundary(url: str) -> None:
    with pytest.raises(PortableJsonError, match="host path"):
        validate_portable_json({"actor": "human", "action": "open_calibration_ui", "url": url})


@pytest.mark.parametrize(
    "key",
    [
        "session_token",
        "sessionToken",
        "motion_token",
        "api_key",
        "apiKey",
        "authorization",
        "password",
    ],
)
def test_sensitive_keys_are_never_public_json(key: str) -> None:
    with pytest.raises(PortableJsonError, match="invalid object key"):
        validate_portable_json({key: "top-secret"})

    with pytest.raises(PortableJsonError, match="host path"):
        validate_portable_json(
            {
                "actor": "human",
                "action": "open_calibration_ui",
                "url": f"http://127.0.0.1:8009/?{key}=top-secret",
            }
        )


def test_large_base64_is_rejected_but_small_base64_is_allowed() -> None:
    small = base64.b64encode(b"x" * MAX_INLINE_BASE64_DECODED_BYTES).decode("ascii")
    large = base64.b64encode(b"x" * (MAX_INLINE_BASE64_DECODED_BYTES + 1)).decode("ascii")

    validate_portable_json({"digest_material": small})
    with pytest.raises(PortableJsonError, match="base64"):
        validate_portable_json({"embedded_binary": large})


def test_large_unpadded_urlsafe_base64_is_rejected() -> None:
    large = (
        base64.urlsafe_b64encode(b"x" * (MAX_INLINE_BASE64_DECODED_BYTES + 1))
        .decode("ascii")
        .rstrip("=")
    )

    assert len(large) % 4 in {2, 3}
    with pytest.raises(PortableJsonError, match="base64"):
        validate_portable_json({"embedded_binary": large})


@pytest.mark.parametrize(
    "prefix",
    ["base64:", "application/octet-stream;base64,"],
)
def test_explicit_base64_with_mime_whitespace_is_bounded(prefix: str) -> None:
    encoded = base64.b64encode(
        b"x" * (MAX_INLINE_BASE64_DECODED_BYTES + 1)
    ).decode("ascii")
    wrapped = " \n\t".join(encoded[index : index + 76] for index in range(0, len(encoded), 76))

    with pytest.raises(PortableJsonError, match="base64"):
        validate_portable_json({"embedded_binary": prefix + wrapped})


def test_near_two_mib_base64_and_oversized_text_are_rejected() -> None:
    near_two_mib = base64.b64encode(b"x" * (2 * 1024 * 1024 - 128)).decode("ascii")
    oversized_text = "human-readable text " * (MAX_PORTABLE_STRING_BYTES // 10)

    with pytest.raises(PortableJsonError, match="too large"):
        validate_portable_json({"embedded_binary": near_two_mib})
    with pytest.raises(PortableJsonError, match="too large"):
        validate_portable_json({"message": oversized_text})


def test_document_budget_cannot_be_evaded_with_many_strings() -> None:
    chunk = "normal prose with spaces " * 1_000
    count = MAX_PORTABLE_DOCUMENT_STRING_BYTES // len(chunk.encode("utf-8")) + 2

    with pytest.raises(PortableJsonError, match="document string budget"):
        validate_portable_json({"chunks": [chunk] * count})


def test_report_sized_document_can_use_more_than_one_mib_of_safe_text() -> None:
    chunk = "bounded report prose with spaces " * 23_000
    document = {"messages": [chunk, chunk]}

    assert (
        MAX_PORTABLE_STRING_BYTES
        < len(json.dumps(document, separators=(",", ":")).encode("utf-8"))
        < MAX_PORTABLE_DOCUMENT_BYTES
    )
    validate_portable_json(document)


def test_non_string_bulk_cannot_evade_the_document_budget() -> None:
    with pytest.raises(PortableJsonError, match="container item budget"):
        validate_portable_json({"numbers": [0] * (MAX_PORTABLE_CONTAINER_ITEMS + 1)})


def test_deep_empty_containers_are_bounded() -> None:
    document: object = None
    for _ in range(MAX_PORTABLE_DEPTH + 1):
        document = [document]

    with pytest.raises(PortableJsonError, match="nesting too deep"):
        validate_portable_json(document)
