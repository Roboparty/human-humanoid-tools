#!/bin/sh
set -eu

umask 022

installer_name='HHTools desktop runtime installer'
embedded_version=''
embedded_wheel=''
embedded_runtime_id=''
embedded_platform=''
embedded_arch=''
python_request=${HHTOOLS_PYTHON:-'>=3.12,<3.14'}
system_install=0

say() {
    printf '%s\n' "$*"
}

die() {
    printf '%s: %s\n' "$installer_name" "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Install the HHTools runtime bundled with the Linux or macOS desktop package.

Usage:
  install.sh [--system]

Options:
  --system    Linux only: install under /opt/hhtools and link commands in /usr/local/bin.
              This mode must be run as root.
  -h, --help  Show this help.

The HHTools wheel, dependency lock, installer configuration, and uv executable
are read from this desktop package. Python and third-party packages are fetched
by uv from the configured package indexes; no unpublished Release is required.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --system)
            system_install=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

host_platform=$(uname -s)
host_arch=$(uname -m)
case "$host_platform:$host_arch" in
    Linux:x86_64|Linux:amd64) checksum_command=sha256sum ;;
    Darwin:arm64)
        checksum_command=shasum
        [ "$system_install" -eq 0 ] || die 'macOS supports current-user installation only'
        # Match the Python minor version validated by desktop development and CI.
        python_request=${HHTOOLS_PYTHON:-3.12}
        ;;
    *) die 'this installer supports Linux x86_64 and macOS Apple Silicon only' ;;
esac
[ -z "$embedded_platform" ] || [ "$host_platform" = "$embedded_platform" ] \
    || die 'the bundled runtime was built for a different operating system'
[ -z "$embedded_arch" ] || [ "$host_arch" = "$embedded_arch" ] \
    || die 'the bundled runtime was built for a different CPU architecture'

for command_name in cat cp dirname id mkdir mktemp mv rm "$checksum_command" uname; do
    command -v "$command_name" >/dev/null 2>&1 \
        || die "required command not found: $command_name"
done

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
asset_dir="$script_dir/assets"
uv_bin="$script_dir/bin/uv"
checksum_file="$script_dir/SHA256SUMS"
runtime_id_file="$script_dir/RUNTIME_ID"
version=$embedded_version
wheel_name=$embedded_wheel
runtime_id=$embedded_runtime_id

[ -n "$version" ] || die 'the desktop package has no embedded runtime version'
[ -n "$wheel_name" ] || die 'the desktop package has no embedded HHTools wheel'
[ -n "$runtime_id" ] || die 'the desktop package has no embedded runtime identity'
[ -f "$runtime_id_file" ] || die "bundled runtime identity not found: $runtime_id_file"
[ "$(cat "$runtime_id_file")" = "$runtime_id" ] \
    || die 'bundled runtime identity differs from the installer'
[ -f "$checksum_file" ] || die "bundled checksum file not found: $checksum_file"
[ -x "$uv_bin" ] || die "bundled uv executable not found: $uv_bin"
[ -f "$asset_dir/$wheel_name" ] || die "bundled wheel not found: $wheel_name"
[ -f "$asset_dir/requirements-all.txt" ] \
    || die 'bundled dependency lock not found: requirements-all.txt'
[ -f "$asset_dir/installer-uv.toml" ] \
    || die 'bundled uv configuration not found: installer-uv.toml'

if [ "$system_install" -eq 1 ]; then
    [ "$(id -u)" -eq 0 ] || die '--system must be run as root'
    install_root=${HHTOOLS_INSTALL_ROOT:-/opt/hhtools}
    bin_dir=${HHTOOLS_BIN_DIR:-/usr/local/bin}
    cache_dir=${HHTOOLS_CACHE_DIR:-/var/cache/hhtools/uv}
else
    [ "$(id -u)" -ne 0 ] || die 'do not run as root without --system'
    [ -n "${HOME:-}" ] || die 'HOME is required for a user installation'
    if [ "$host_platform" = Darwin ]; then
        install_root=${HHTOOLS_INSTALL_ROOT:-"$HOME/Library/Application Support/hhtools"}
        bin_dir=${HHTOOLS_BIN_DIR:-"$install_root/bin"}
        cache_dir=${HHTOOLS_CACHE_DIR:-"$HOME/Library/Caches/hhtools/uv"}
    else
        data_home=${XDG_DATA_HOME:-"$HOME/.local/share"}
        cache_home=${XDG_CACHE_HOME:-"$HOME/.cache"}
        install_root=${HHTOOLS_INSTALL_ROOT:-"$data_home/hhtools"}
        bin_dir=${HHTOOLS_BIN_DIR:-"${XDG_BIN_HOME:-$HOME/.local/bin}"}
        cache_dir=${HHTOOLS_CACHE_DIR:-"$cache_home/hhtools/uv"}
    fi
fi

say "[1/5] Verifying bundled HHTools $version installation files..."
(
    cd "$script_dir"
    if [ "$checksum_command" = shasum ]; then
        shasum -a 256 -c SHA256SUMS
    else
        sha256sum -c SHA256SUMS
    fi
) || die 'bundled installation file verification failed'

say '[2/5] Preparing the isolated runtime directories...'
mkdir -p "$install_root" "$bin_dir" "$cache_dir"
runtime_marker="$install_root/runtime-version"
rm -f -- "$runtime_marker"
temporary_parent=${TMPDIR:-/tmp}
work_dir=$(mktemp -d "$temporary_parent/hhtools-desktop-install.XXXXXX")
cleanup() {
    rm -rf -- "$work_dir"
}
trap cleanup EXIT HUP INT TERM
cp "$asset_dir/$wheel_name" "$work_dir/$wheel_name"
cp "$asset_dir/requirements-all.txt" "$work_dir/requirements-all.txt"
cp "$asset_dir/installer-uv.toml" "$work_dir/installer-uv.toml"

say "[3/5] Installing Python $python_request and locked dependencies with bundled uv..."
UV_TOOL_DIR="$install_root/tools" \
UV_TOOL_BIN_DIR="$bin_dir" \
UV_PYTHON_INSTALL_DIR="$install_root/python" \
UV_PYTHON_BIN_DIR="$install_root/python-bin" \
UV_CACHE_DIR="$cache_dir" \
    "$uv_bin" tool install \
        --force \
        --python "$python_request" \
        --config-file "$work_dir/installer-uv.toml" \
        --with-requirements "$work_dir/requirements-all.txt" \
        "hhtools @ file://$work_dir/$wheel_name"

say '[4/5] Verifying the installed CLI, WebUI, retargeting, and Agent runtime...'
hhtools_bin="$bin_dir/hhtools"
[ -x "$hhtools_bin" ] || die "installed command not found: $hhtools_bin"
"$hhtools_bin" --version
"$hhtools_bin" doctor --require web --require retarget --require mcp
runtime_python="$install_root/tools/hhtools/bin/python"
[ -x "$runtime_python" ] || die "installed runtime Python not found: $runtime_python"

if [ "$host_platform" = Darwin ]; then
    # A cold macOS load of scientific native libraries can exceed the sidecar's
    # startup deadline. Finish that work while installation progress is visible.
    say '[4/5] Preparing the first macOS launch; loading scientific libraries may take a few minutes...'
    "$runtime_python" -c 'from hhtools.web.server.launch import run_desktop_sidecar'
fi

runtime_marker_tmp="$install_root/.runtime-version.$$"
printf '%s\n' "$runtime_id" > "$runtime_marker_tmp"
mv "$runtime_marker_tmp" "$runtime_marker"

say '[5/5] HHTools runtime installation completed.'
say 'GVHMR and SMPL-family weights were not installed.'
