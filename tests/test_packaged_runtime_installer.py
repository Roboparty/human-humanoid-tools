from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INSTALLER_TEMPLATE = REPOSITORY_ROOT / "desktop" / "scripts" / "install-packaged-runtime.sh"
VERSION = "0.1.0"
WHEEL = f"hhtools-{VERSION}-py3-none-any.whl"
RUNTIME_ID = f"{VERSION}+sha256.0123456789abcdefabcd"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _bootstrap(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "bootstrap"
    assets = root / "assets"
    bin_dir = root / "bin"
    assets.mkdir(parents=True)
    bin_dir.mkdir()
    (assets / WHEEL).write_text("wheel fixture\n", encoding="utf-8")
    (assets / "requirements-all.txt").write_text("dependency==1.0\n", encoding="utf-8")
    (assets / "installer-uv.toml").write_text("", encoding="utf-8")
    (root / "RUNTIME_ID").write_text(f"{RUNTIME_ID}\n", encoding="utf-8")

    arguments = tmp_path / "uv-arguments.txt"
    uv = bin_dir / "uv"
    _write_executable(
        uv,
        """#!/bin/sh
set -eu
printf '%s\\n' "$@" > "$HHTOOLS_TEST_UV_ARGUMENTS"
mkdir -p "$UV_TOOL_DIR/hhtools/bin" "$UV_TOOL_BIN_DIR"
cat > "$UV_TOOL_DIR/hhtools/bin/python" <<'EOF'
#!/bin/sh
exit "${HHTOOLS_TEST_IMPORT_STATUS:-0}"
EOF
chmod +x "$UV_TOOL_DIR/hhtools/bin/python"
cat > "$UV_TOOL_BIN_DIR/hhtools" <<'EOF'
#!/bin/sh
case "${1:-}" in
    --version) printf '%s\\n' 'hhtools 0.1.0' ;;
    doctor) exit 0 ;;
    *) exit 0 ;;
esac
EOF
chmod +x "$UV_TOOL_BIN_DIR/hhtools"
""",
    )

    checksum_paths = [
        f"assets/{WHEEL}",
        "assets/requirements-all.txt",
        "assets/installer-uv.toml",
        "bin/uv",
        "RUNTIME_ID",
    ]
    checksums = "".join(
        f"{hashlib.sha256((root / relative).read_bytes()).hexdigest()}  {relative}\n"
        for relative in checksum_paths
    )
    (root / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    installer = INSTALLER_TEMPLATE.read_text(encoding="utf-8")
    installer = installer.replace("embedded_version=''", f"embedded_version='{VERSION}'")
    installer = installer.replace("embedded_wheel=''", f"embedded_wheel='{WHEEL}'")
    installer = installer.replace("embedded_runtime_id=''", f"embedded_runtime_id='{RUNTIME_ID}'")
    installer_path = root / "install.sh"
    _write_executable(installer_path, installer)
    return installer_path, arguments


def _environment(tmp_path: Path, arguments: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(tmp_path / "home"),
            "HHTOOLS_BIN_DIR": str(tmp_path / "installed" / "bin"),
            "HHTOOLS_CACHE_DIR": str(tmp_path / "installed" / "cache"),
            "HHTOOLS_INSTALL_ROOT": str(tmp_path / "installed" / "runtime"),
            "HHTOOLS_TEST_UV_ARGUMENTS": str(arguments),
        }
    )
    return environment


def _command(installer: Path) -> list[str]:
    command = ["sh", str(installer)]
    if os.geteuid() == 0:
        command.append("--system")
    return command


def test_packaged_installer_uses_local_inputs_and_publishes_runtime_marker(
    tmp_path: Path,
) -> None:
    installer, arguments = _bootstrap(tmp_path)

    completed = subprocess.run(
        _command(installer),
        env=_environment(tmp_path, arguments),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "[1/5] Verifying bundled HHTools 0.1.0" in completed.stdout
    assert "[5/5] HHTools runtime installation completed." in completed.stdout
    assert "curl" not in INSTALLER_TEMPLATE.read_text(encoding="utf-8")
    assert (tmp_path / "installed" / "runtime" / "runtime-version").read_text(
        encoding="utf-8"
    ) == f"{RUNTIME_ID}\n"
    passed_arguments = arguments.read_text(encoding="utf-8").splitlines()
    assert passed_arguments[:2] == ["tool", "install"]
    assert "--with-requirements" in passed_arguments
    assert passed_arguments[-1].startswith("hhtools @ file://")


def test_packaged_installer_rejects_tampered_local_asset(tmp_path: Path) -> None:
    installer, arguments = _bootstrap(tmp_path)
    (installer.parent / "assets" / "requirements-all.txt").write_text(
        "tampered\n", encoding="utf-8"
    )

    completed = subprocess.run(
        _command(installer),
        env=_environment(tmp_path, arguments),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode != 0
    assert "bundled installation file verification failed" in completed.stderr
    assert not arguments.exists()


def _mac_environment(tmp_path: Path, arguments: Path) -> dict[str, str]:
    environment = _environment(tmp_path, arguments)
    commands = tmp_path / "commands"
    commands.mkdir()
    _write_executable(
        commands / "uname", '#!/bin/sh\ncase "$1" in -s) echo Darwin;; -m) echo arm64;; esac\n',
    )
    _write_executable(commands / "id", '#!/bin/sh\necho 1000\n')
    environment["PATH"] = f"{commands}{os.pathsep}{environment['PATH']}"
    for key in ("HHTOOLS_INSTALL_ROOT", "HHTOOLS_BIN_DIR", "HHTOOLS_CACHE_DIR"):
        environment.pop(key)
    return environment


def test_mac_installer_uses_application_support_and_private_commands(tmp_path: Path) -> None:
    installer, arguments = _bootstrap(tmp_path)
    environment = _mac_environment(tmp_path, arguments)
    # XDG directories must not change the macOS runtime path used by Electron.
    environment["XDG_DATA_HOME"] = str(tmp_path / "linux-data")
    completed = subprocess.run(
        ["sh", str(installer)], env=environment, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    install_root = tmp_path / "home" / "Library" / "Application Support" / "hhtools"
    assert (install_root / "runtime-version").read_text().strip() == RUNTIME_ID
    assert (install_root / "bin" / "hhtools").is_file()
    passed_arguments = arguments.read_text().splitlines()
    assert passed_arguments[passed_arguments.index("--python") + 1] == "3.12"
    assert not (tmp_path / "linux-data").exists()


def test_mac_installer_does_not_mark_failed_first_launch_imports_as_ready(tmp_path: Path) -> None:
    installer, arguments = _bootstrap(tmp_path)
    environment = _mac_environment(tmp_path, arguments)
    environment["HHTOOLS_TEST_IMPORT_STATUS"] = "1"
    completed = subprocess.run(
        ["sh", str(installer)], env=environment, capture_output=True, text=True, check=False,
    )
    assert completed.returncode != 0
    assert "Preparing the first macOS launch" in completed.stdout
    install_root = tmp_path / "home" / "Library" / "Application Support" / "hhtools"
    assert not (install_root / "runtime-version").exists()


@pytest.mark.parametrize("failure", ["system", "platform", "architecture"])
def test_mac_installer_rejects_incompatible_requests(tmp_path: Path, failure: str) -> None:
    installer, arguments = _bootstrap(tmp_path)
    environment = _mac_environment(tmp_path, arguments)
    command = ["sh", str(installer)]
    if failure == "system":
        command.append("--system")
    else:
        content = installer.read_text()
        content = content.replace(
            "embedded_platform=''" if failure == "platform" else "embedded_arch=''",
            "embedded_platform='Linux'" if failure == "platform" else "embedded_arch='x86_64'",
        )
        installer.write_text(content)
    completed = subprocess.run(
        command, env=environment, capture_output=True, text=True, check=False,
    )
    assert completed.returncode != 0
    assert not arguments.exists()
    assert not (tmp_path / "home").exists()
