from __future__ import annotations

import json
import subprocess
import sys

from click import unstyle
from typer.testing import CliRunner

from hhtools.cli import doctor


def test_doctor_help_lists_repeatable_requirements() -> None:
    result = CliRunner().invoke(doctor.app, ["--help"])
    help_text = unstyle(result.stdout)

    assert result.exit_code == 0
    assert "--json" in help_text
    assert "--require" in help_text
    assert "web|robot|retarget|mcp" in help_text
    assert "models|gvhmr" in help_text


def test_doctor_json_is_one_document_with_empty_stderr() -> None:
    result = CliRunner().invoke(doctor.app, ["--json"])

    assert result.exit_code == 0
    assert result.stderr == ""
    assert result.stdout.count("\n") == 1
    document = json.loads(result.stdout)
    assert document["schema_version"] == "1.0"
    assert document["kind"] == "hhtools_doctor"
    assert document["base_ready"] is True
    assert document["requested"] == []
    assert [check["name"] for check in document["checks"]] == [
        "base",
        "web",
        "robot",
        "retarget",
        "mcp",
        "bodymodels",
        "gvhmr",
    ]


def test_doctor_required_unavailable_group_exits_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor,
        "_check_web",
        lambda: doctor.DoctorCheck("web", False, "not installed", {"missing": ["fastapi"]}),
    )

    result = CliRunner().invoke(doctor.app, ["--json", "--require", "web"])

    assert result.exit_code == 1
    assert result.stderr == ""
    document = json.loads(result.stdout)
    assert document["requested"] == ["web"]
    assert document["base_ready"] is True
    assert document["ready"] is False


def test_doctor_human_output_allows_unavailable_optional_groups() -> None:
    result = CliRunner().invoke(doctor.app, [])

    assert result.exit_code == 0
    assert result.stderr == ""
    assert result.stdout.startswith("hhtools doctor ")
    assert "[OK] base (required)" in result.stdout
    assert "(optional)" in result.stdout
    assert result.stdout.rstrip().endswith("Result: ready")


def test_doctor_gvhmr_check_never_runs_an_external_probe(monkeypatch) -> None:
    from hhtools.integrations import gvhmr

    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("doctor must not run the GVHMR environment probe")

    monkeypatch.setattr(gvhmr, "_run_probe", unexpected_probe)

    check = doctor._check_gvhmr()

    assert check.details["probe_scope"] == "configuration_only"


def test_doctor_is_lazy_and_does_not_initialize_solver_modules() -> None:
    script = """
import json
import sys
sys.argv = ["hhtools", "--version"]
import hhtools.cli.main
version_lazy = "hhtools.cli.doctor" not in sys.modules
from hhtools.cli.doctor import build_report
build_report([])
print(json.dumps({
    "version_lazy": version_lazy,
    "warp_not_imported": "warp" not in sys.modules,
    "newton_not_imported": "newton" not in sys.modules,
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {
        "version_lazy": True,
        "warp_not_imported": True,
        "newton_not_imported": True,
    }
