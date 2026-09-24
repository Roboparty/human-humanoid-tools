from __future__ import annotations

import pytest

from hhtools.web import dependencies


def test_missing_web_runtime_dependencies_reports_distribution_names(monkeypatch) -> None:
    present = {"fastapi", "starlette"}
    monkeypatch.setattr(
        dependencies,
        "find_spec",
        lambda import_name: object() if import_name in present else None,
    )

    assert dependencies.missing_web_runtime_dependencies() == (
        "uvicorn",
        "python-multipart",
    )


def test_require_web_runtime_dependencies_provides_recovery_commands(monkeypatch) -> None:
    monkeypatch.setattr(
        dependencies,
        "missing_web_runtime_dependencies",
        lambda: ("uvicorn",),
    )

    with pytest.raises(dependencies.MissingWebDependenciesError) as error:
        dependencies.require_web_runtime_dependencies()

    assert error.value.missing == ("uvicorn",)
    assert "uv sync --locked --extra web" in str(error.value)
    assert "uv run hhtools web" in str(error.value)


def test_browser_server_checks_dependencies_before_startup(tmp_path, monkeypatch) -> None:
    from hhtools.web import server
    from hhtools.web.server import launch as server_launch

    def fail_preflight() -> None:
        raise dependencies.MissingWebDependenciesError(("uvicorn",))

    monkeypatch.setattr(server_launch, "require_web_runtime_dependencies", fail_preflight)

    with pytest.raises(dependencies.MissingWebDependenciesError):
        server.run_web(source_root=tmp_path, save_dir=tmp_path)


def test_desktop_sidecar_checks_dependencies_before_startup(tmp_path, monkeypatch) -> None:
    from hhtools.web import server
    from hhtools.web.server import launch as server_launch

    def fail_preflight() -> None:
        raise dependencies.MissingWebDependenciesError(("fastapi",))

    monkeypatch.setattr(server_launch, "require_web_runtime_dependencies", fail_preflight)

    with pytest.raises(dependencies.MissingWebDependenciesError):
        server.run_desktop_sidecar(
            source_root=tmp_path,
            save_dir=tmp_path,
            cache_dir=tmp_path,
            host="127.0.0.1",
            port=43123,
            session_secret="unit-test-secret",
        )
