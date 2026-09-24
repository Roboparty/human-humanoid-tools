from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = REPO_ROOT / "install.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _release_assets(root: Path, *, version: str = "1.2.3") -> Path:
    assets = root / "assets"
    assets.mkdir()
    filenames = [
        f"hhtools-{version}-py3-none-any.whl",
        "requirements-all.txt",
        "installer-uv.toml",
    ]
    for filename in filenames:
        (assets / filename).write_text(f"fixture for {filename}\n", encoding="utf-8")
    checksums = "".join(
        f"{hashlib.sha256((assets / filename).read_bytes()).hexdigest()}  {filename}\n"
        for filename in filenames
    )
    (assets / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    return assets


def _fake_uv(root: Path) -> tuple[Path, Path]:
    arguments = root / "uv-arguments.txt"
    executable = root / "uv"
    _write_executable(
        executable,
        """#!/bin/sh
set -eu
printf '%s\\n' "$@" > "$HHTOOLS_TEST_UV_ARGUMENTS"
mkdir -p "$UV_TOOL_BIN_DIR"
mkdir -p "$UV_TOOL_DIR/hhtools/bin"
cat > "$UV_TOOL_DIR/hhtools/bin/python" <<'EOF'
#!/bin/sh
exit 0
EOF
chmod +x "$UV_TOOL_DIR/hhtools/bin/python"
cat > "$UV_TOOL_BIN_DIR/hhtools" <<'EOF'
#!/bin/sh
case "${1:-}" in
    --version) printf '%s\\n' 'hhtools 1.2.3' ;;
    doctor) [ "${HHTOOLS_TEST_DOCTOR_FAIL:-0}" != '1' ] ;;
    *) exit 0 ;;
esac
EOF
chmod +x "$UV_TOOL_BIN_DIR/hhtools"
cat > "$UV_TOOL_BIN_DIR/hhtools-mcp" <<'EOF'
#!/bin/sh
exit 0
EOF
chmod +x "$UV_TOOL_BIN_DIR/hhtools-mcp"
""",
    )
    return executable, arguments


def _environment(tmp_path: Path, assets: Path, uv_bin: Path, arguments: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(tmp_path / "home"),
            "HHTOOLS_ASSET_DIR": str(assets),
            "HHTOOLS_UV_BIN": str(uv_bin),
            "HHTOOLS_TEST_UV_ARGUMENTS": str(arguments),
            "HHTOOLS_VERSION": "1.2.3",
        }
    )
    if os.geteuid() == 0:
        environment.update(
            {
                "HHTOOLS_INSTALL_ROOT": str(tmp_path / "system" / "hhtools"),
                "HHTOOLS_BIN_DIR": str(tmp_path / "system" / "bin"),
                "HHTOOLS_CACHE_DIR": str(tmp_path / "system" / "cache"),
            }
        )
    return environment


def _installer_command() -> list[str]:
    command = ["sh", str(INSTALLER)]
    if os.geteuid() == 0:
        command.append("--system")
    return command


def _installed_bin_dir(tmp_path: Path) -> Path:
    if os.geteuid() == 0:
        return tmp_path / "system" / "bin"
    return tmp_path / "home" / ".local" / "bin"


def _installed_root(tmp_path: Path) -> Path:
    if os.geteuid() == 0:
        return tmp_path / "system" / "hhtools"
    return tmp_path / "home" / ".local" / "share" / "hhtools"


def test_installer_is_posix_sh_and_documents_model_boundaries() -> None:
    contents = INSTALLER.read_text(encoding="utf-8")

    assert contents.startswith("#!/bin/sh\n")
    assert "[[" not in contents
    assert "BASH_SOURCE" not in contents
    assert "${HHTOOLS_PYTHON:-'>=3.12,<3.14'}" in contents
    assert "GVHMR, SMPL-family weights, robot model archives" in contents


def test_install_uses_verified_release_assets_and_isolated_uv_tool(tmp_path: Path) -> None:
    assets = _release_assets(tmp_path)
    uv_bin, arguments = _fake_uv(tmp_path)

    completed = subprocess.run(
        _installer_command(),
        env=_environment(tmp_path, assets, uv_bin, arguments),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "HHTools 1.2.3 is ready." in completed.stdout
    assert "were not downloaded" in completed.stdout
    bin_dir = _installed_bin_dir(tmp_path)
    assert (bin_dir / "hhtools").is_file()
    assert (bin_dir / "hhtools-mcp").is_file()
    install_root = _installed_root(tmp_path)
    assert (install_root / "runtime-version").read_text(encoding="utf-8") == "1.2.3\n"
    assert (install_root / "tools" / "hhtools" / "bin" / "python").is_file()

    passed_arguments = arguments.read_text(encoding="utf-8").splitlines()
    assert passed_arguments[:2] == ["tool", "install"]
    assert passed_arguments[passed_arguments.index("--python") + 1] == ">=3.12,<3.14"
    assert "--with-requirements" in passed_arguments
    assert passed_arguments[-1].startswith("hhtools @ file://")


def test_failed_runtime_verification_does_not_publish_completion_marker(
    tmp_path: Path,
) -> None:
    assets = _release_assets(tmp_path)
    uv_bin, arguments = _fake_uv(tmp_path)
    install_root = _installed_root(tmp_path)
    install_root.mkdir(parents=True)
    marker = install_root / "runtime-version"
    marker.write_text("stale\n", encoding="utf-8")
    environment = _environment(tmp_path, assets, uv_bin, arguments)
    environment["HHTOOLS_TEST_DOCTOR_FAIL"] = "1"

    completed = subprocess.run(
        _installer_command(),
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode != 0
    assert not marker.exists()


def test_installer_rejects_a_release_checksum_mismatch_before_running_uv(tmp_path: Path) -> None:
    assets = _release_assets(tmp_path)
    (assets / "requirements-all.txt").write_text("tampered\n", encoding="utf-8")
    uv_bin, arguments = _fake_uv(tmp_path)

    completed = subprocess.run(
        _installer_command(),
        env=_environment(tmp_path, assets, uv_bin, arguments),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode != 0
    assert "release asset verification failed" in completed.stderr
    assert not arguments.exists()
