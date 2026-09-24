"""Distribution and project setup contracts for the public Agent adapters."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _requirement_name(requirement: str) -> str:
    return re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0].casefold()


def test_mcp_extra_owns_its_product_runtime_dependencies() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    requirements = extras["mcp"]
    names = {_requirement_name(requirement) for requirement in requirements}
    execution_runtime_names = {
        _requirement_name(requirement)
        for extra in ("formats", "smpl", "robot", "retarget", "retarget-interaction")
        for requirement in extras[extra]
    }

    assert {"mcp", *execution_runtime_names} <= names
    assert {"fastapi", "uvicorn", "python-multipart"}.isdisjoint(names)
    assert all(not requirement.startswith("hhtools[") for requirement in requirements)


def test_codex_project_mcp_uses_portable_isolated_runtime_paths() -> None:
    config = tomllib.loads((REPO_ROOT / ".codex" / "config.toml").read_text(encoding="utf-8"))
    server = config["mcp_servers"]["hhtools"]

    assert server["command"] == "uv"
    assert server["cwd"] == "."
    assert server["enabled"] is True
    assert server["required"] is True
    assert server["default_tools_approval_mode"] == "writes"
    assert server["args"][:4] == ["run", "--frozen", "--no-sync", "hhtools-mcp"]
    assert server["args"][server["args"].index("--save-dir") + 1] == ".hhtools/agent/save"
    assert server["args"][server["args"].index("--cache") + 1] == ".hhtools/agent/cache"
    assert (
        server["args"][server["args"].index("--job-settings") + 1]
        == ".hhtools/agent/job-settings.json"
    )
    assert server["args"][server["args"].index("--web-ui-url") + 1].endswith(":8010")
    assert all(not Path(value).is_absolute() for value in server["args"] if "://" not in value)
    assert ".hhtools/" in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_readmes_discover_the_public_agent_guide() -> None:
    guide = (REPO_ROOT / "docs" / "agent.md").read_text(encoding="utf-8")
    assert "uv sync --locked --extra mcp" in guide
    assert "--managed-python --python 3.12" not in guide
    assert "uv run hhtools agent capabilities" in guide
    assert "hhtools agent asset catalog" in guide
    assert "list_available_assets" in guide
    assert "uv run --frozen --no-sync" in guide
    assert "Only one local runtime may own a `save-dir`" in guide
    assert "run_mode: smoke" in guide
    assert "../.codex/config.toml" in guide

    for readme in ("README.md", "README_cn.md"):
        contents = (REPO_ROOT / readme).read_text(encoding="utf-8")
        assert "(docs/agent.md)" in contents
        assert "uv sync --locked" in contents
        assert "--managed-python --python 3.12" not in contents
