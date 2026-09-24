#!/bin/sh
set -eu

umask 022

# Tagged release assets replace these three values before publishing.  The
# source copy can still resolve the latest release, or be pointed at local
# assets by the release smoke test.
embedded_repository='Roboparty/human-humanoid-tools'
embedded_version=''
embedded_wheel=''

installer_name='HHTools installer'
uv_version='0.12.9'
python_request=${HHTOOLS_PYTHON:-'>=3.12,<3.14'}
repository=${HHTOOLS_REPOSITORY:-$embedded_repository}
version=${HHTOOLS_VERSION:-$embedded_version}
asset_dir=${HHTOOLS_ASSET_DIR:-}
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
Install the complete HHTools Python/Web/Agent runtime on Linux.

Usage:
  install.sh [--system] [--version VERSION]

Options:
  --system          Install under /opt/hhtools and link commands in /usr/local/bin.
                    This mode must be run as root.
  --version VALUE   Install one tagged HHTools release (without the leading "v").
  -h, --help        Show this help.

Environment overrides:
  HHTOOLS_PYTHON        Python request passed to uv (default: >=3.12,<3.14).
  HHTOOLS_REPOSITORY    GitHub owner/repository used for release downloads.
  HHTOOLS_INSTALL_ROOT  Runtime storage directory.
  HHTOOLS_BIN_DIR       Directory receiving hhtools and hhtools-mcp.

GVHMR, SMPL-family weights, robot model archives, and NVIDIA drivers are not
downloaded by this installer.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --system)
            system_install=1
            shift
            ;;
        --version)
            [ "$#" -ge 2 ] || die '--version requires a value'
            version=$2
            shift 2
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

[ "$(uname -s)" = 'Linux' ] || die 'this installer currently supports Linux only'
case "$(uname -m)" in
    x86_64|amd64) ;;
    *) die 'the complete runtime is currently verified only on Linux x86_64' ;;
esac

for command_name in id mktemp sha256sum; do
    command -v "$command_name" >/dev/null 2>&1 || die "required command not found: $command_name"
done

if [ "$system_install" -eq 1 ]; then
    [ "$(id -u)" -eq 0 ] || die '--system must be run as root (for example, with sudo)'
    install_root=${HHTOOLS_INSTALL_ROOT:-/opt/hhtools}
    bin_dir=${HHTOOLS_BIN_DIR:-/usr/local/bin}
    cache_dir=${HHTOOLS_CACHE_DIR:-/var/cache/hhtools/uv}
else
    [ "$(id -u)" -ne 0 ] || die 'do not run as root without the explicit --system option'
    [ -n "${HOME:-}" ] || die 'HOME is required for a user installation'
    data_home=${XDG_DATA_HOME:-"$HOME/.local/share"}
    cache_home=${XDG_CACHE_HOME:-"$HOME/.cache"}
    install_root=${HHTOOLS_INSTALL_ROOT:-"$data_home/hhtools"}
    bin_dir=${HHTOOLS_BIN_DIR:-"${XDG_BIN_HOME:-$HOME/.local/bin}"}
    cache_dir=${HHTOOLS_CACHE_DIR:-"$cache_home/hhtools/uv"}
fi

case "$repository" in
    ''|/*|*/|*//*|*/*/*) die "invalid GitHub repository: $repository" ;;
esac
repository_owner=${repository%%/*}
repository_name=${repository#*/}
case "$repository_owner$repository_name" in
    *[!A-Za-z0-9_.-]*) die "invalid GitHub repository: $repository" ;;
esac

fetch_https() {
    fetch_url=$1
    fetch_output=$2
    command -v curl >/dev/null 2>&1 || die 'curl is required for network installation'
    curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 1 -fsSL \
        "$fetch_url" -o "$fetch_output"
}

if [ -z "$version" ]; then
    command -v curl >/dev/null 2>&1 || die 'curl is required to resolve the latest release'
    latest_url=$(curl --proto '=https' --tlsv1.2 --retry 3 -fsSIL \
        -o /dev/null -w '%{url_effective}' \
        "https://github.com/$repository/releases/latest")
    latest_tag=${latest_url##*/}
    version=${latest_tag#v}
fi

case "$version" in
    ''|*[!0-9A-Za-z._+-]*) die "invalid release version: $version" ;;
esac

if [ -n "$embedded_wheel" ] && [ "$version" = "$embedded_version" ]; then
    wheel_name=$embedded_wheel
else
    wheel_name="hhtools-$version-py3-none-any.whl"
fi

mkdir -p "$install_root" "$bin_dir" "$cache_dir"
# A marker represents a fully verified runtime, so hide any previous marker
# before replacing files. A concurrent app launch will keep showing setup until
# this installation finishes successfully.
runtime_marker="$install_root/runtime-version"
rm -f -- "$runtime_marker"
temporary_parent=${TMPDIR:-/tmp}
work_dir=$(mktemp -d "$temporary_parent/hhtools-install.XXXXXX")
cleanup() {
    rm -rf -- "$work_dir"
}
trap cleanup EXIT HUP INT TERM

release_url="https://github.com/$repository/releases/download/v$version"
copy_asset() {
    asset_name=$1
    if [ -n "$asset_dir" ]; then
        [ -f "$asset_dir/$asset_name" ] || die "release asset not found: $asset_dir/$asset_name"
        cp "$asset_dir/$asset_name" "$work_dir/$asset_name"
    else
        fetch_https "$release_url/$asset_name" "$work_dir/$asset_name"
    fi
}

say "Installing HHTools $version for $python_request"
copy_asset 'SHA256SUMS'
copy_asset "$wheel_name"
copy_asset 'requirements-all.txt'
copy_asset 'installer-uv.toml'

(
    cd "$work_dir"
    sha256sum -c SHA256SUMS
) || die 'release asset verification failed'

if [ -n "${HHTOOLS_UV_BIN:-}" ]; then
    uv_bin=$HHTOOLS_UV_BIN
    [ -x "$uv_bin" ] || die "HHTOOLS_UV_BIN is not executable: $uv_bin"
else
    uv_home="$install_root/uv"
    uv_bin="$uv_home/uv"
    uv_ready=0
    if [ -x "$uv_bin" ]; then
        case "$("$uv_bin" --version 2>/dev/null || true)" in
            "uv $uv_version"*) uv_ready=1 ;;
        esac
    fi
    if [ "$uv_ready" -eq 0 ]; then
        fetch_https "https://astral.sh/uv/$uv_version/install.sh" "$work_dir/uv-install.sh"
        mkdir -p "$uv_home"
        env UV_UNMANAGED_INSTALL="$uv_home" UV_NO_MODIFY_PATH=1 \
            sh "$work_dir/uv-install.sh"
    fi
    [ -x "$uv_bin" ] || die 'the private uv bootstrap did not produce an executable'
fi

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

hhtools_bin="$bin_dir/hhtools"
[ -x "$hhtools_bin" ] || die "installed command not found: $hhtools_bin"
"$hhtools_bin" --version
"$hhtools_bin" doctor --require web --require retarget --require mcp >/dev/null

# Publish the completion marker only after the installed interpreter and all
# required runtime imports have passed validation. The desktop shell ignores a
# partial tool environment without this marker.
runtime_python="$install_root/tools/hhtools/bin/python"
[ -x "$runtime_python" ] || die "installed runtime Python not found: $runtime_python"
runtime_marker_tmp="$install_root/.runtime-version.$$"
printf '%s\n' "$version" > "$runtime_marker_tmp"
mv "$runtime_marker_tmp" "$runtime_marker"

say ''
say "HHTools $version is ready."
say 'GVHMR, SMPL-family weights, and robot model archives were not downloaded.'
case ":${PATH:-}:" in
    *":$bin_dir:"*) ;;
    *) say "Add $bin_dir to PATH, then open a new shell." ;;
esac
say 'Start the WebUI with: hhtools web'
