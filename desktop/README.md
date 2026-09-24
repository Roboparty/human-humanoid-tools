# Human-Humanoid Tools

This directory contains the Electron GUI shell for the existing FastAPI and three.js WebUI.
The desktop app keeps the current HTTP routes and Python business logic while supervising its own
local Python sidecar.

## Development

Prerequisites:

- Node.js 22.12 or newer
- The repository `.venv` with the hhtools Web dependencies installed
- A working browser version of `uv run hhtools web`

From this directory:

```powershell
npm install
npm run dev
```

On Linux and macOS, the equivalent commands are:

```bash
npm install
npm run dev
```

The shell discovers the repository by walking upward from the current app path. Override runtime
locations when needed:

```powershell
$env:HHTOOLS_REPO_ROOT = 'C:\path\to\human-humanoid-tools'
$env:HHTOOLS_PYTHON = 'C:\path\to\python.exe'
npm run dev
```

```bash
HHTOOLS_REPO_ROOT=/path/to/human-humanoid-tools \
HHTOOLS_PYTHON=/path/to/python3 npm run dev
```

Optional data path overrides are `HHTOOLS_SOURCE_ROOT`, `HHTOOLS_SAVE_DIR`,
`HHTOOLS_CACHE_DIR`, and `HHTOOLS_LOG_DIR`.

Background-job admission is optional. Both settings have a factory default of `0`, preserving
unlimited concurrency for expert users:

```powershell
$env:HHTOOLS_MAX_RUNNING_JOBS = '1'
$env:HHTOOLS_MAX_QUEUED_JOBS = '32'
npm run dev
```

A positive running value enables the FIFO waiting queue. A queued value of `0` means unlimited
waiting; it has no effect while running is `0`. The same values are editable in local Electron
under **Settings → Background-job scheduling**; Save persists and hot-applies them without restarting the sidecar
or Electron. Active jobs are not interrupted. Explicit environment values remain startup
overrides and win again on the next launch. `HHTOOLS_WEB_SETTINGS_PATH` can redirect the
persistent JSON file for portable installs and isolated tests.
The Electron environment filter forwards these three named settings, not arbitrary variables or
secrets.
This cap covers scheduled Web jobs, not the optional Warp/Newton robot prewarm thread; it is not
a process-wide GPU concurrency guarantee.

## Verification

```powershell
npm run typecheck
npm test
npm run test:e2e
npm run dist:win
npm run dist:linux
npm run dist:mac
```

`test:e2e` builds and launches the real Electron application, checks the existing WebUI, captures
a screenshot, closes the app, and verifies that the supervised Python process exits.

## Desktop packages

Prebuilt installers are available in the [0.1.0 preview release](https://github.com/Eleanor1018/human-humanoid-tools/releases/tag/v0.1.0%28beta%29).
For macOS 12+ on Apple Silicon, [download the DMG](https://github.com/Eleanor1018/human-humanoid-tools/releases/download/v0.1.0%28beta%29/hhtools-0.1.0-mac-arm64.dmg),
drag **Human-Humanoid Tools** into **Applications**, and launch it to complete the online first-run setup.
See the signing and compatibility notes below before installing this preview.

The desktop targets use different runtime delivery models:

- Windows stages the current all-extras `.venv` and tracked HHTools application files into the
  installer, so the installed EXE does not require a checkout or system Python.
- Linux keeps Python outside the Debian package. On first launch, a native setup page installs the
  bundled HHTools wheel and its explicitly locked dependencies either for the current user
  (recommended) or under `/opt/hhtools`. The package also carries the pinned uv executable, so this
  path does not depend on an unpublished GitHub Release bootstrap. System installation uses the
  operating system's `pkexec` authentication dialog; HHTools never reads or forwards the password.
- macOS uses the same verified first-run bootstrap for the current user. The DMG and ZIP target
  Apple Silicon (arm64), macOS 12 or newer, using Python 3.12. Python lives in
  `~/Library/Application Support/hhtools`, with its private commands under `bin` and its download
  cache in `~/Library/Caches/hhtools/uv`. No system Python, checkout, Homebrew, or administrator
  password is required on the destination Mac. First setup requires internet access.
  Setup also imports the full Web server once before marking the runtime ready, so macOS can
  finish the slow first load of scientific native libraries before the normal startup deadline.
  Installation and startup temporarily prevent macOS app suspension; the activity is released
  when the operation finishes or fails, so switching windows does not stall setup.

All packages include only the 30 motions and six robot bundles selected by
`configs/builtin-assets.json`. Before packaging, install the pinned robot bundles and point the
stager at that library:

```bash
uv run python scripts/install_builtin_robots.py --destination /path/to/release-robots
export HHTOOLS_BUNDLED_ROBOT_DIR=/path/to/release-robots
```

Then invoke electron-builder through the package scripts:

```bash
npm run dist:linux   # release/hhtools-0.1.0-amd64.deb
npm run dist:win     # release/hhtools-0.1.0-x64-setup.exe
npm run dist:mac     # release/hhtools-0.1.0-mac-arm64.dmg (and .zip)
```

`npm run dist:win` must run on Windows after `uv sync --extra all --no-dev`; the runtime stager
rejects another host platform and excludes untracked files, development packages, caches, model
weights, and source-map files. `npm run dist:linux` never stages that runtime.

Build `dist:mac` on an Apple Silicon Mac with native arm64 Node.js 22.12+, Xcode Command Line
Tools, and uv **0.12.9** on PATH (or set `HHTOOLS_PACKAGING_UV_BIN`). Install the build environment
with `uv sync --frozen --extra all --extra dev --python 3.12`, then `npm ci` in this directory.
Prepare the pinned robot library as above before packaging. Open the DMG, drag
**Human-Humanoid Tools** into **Applications**, and launch it to finish setup.

The macOS preview uses the default **ad-hoc signature**, with hardened runtime and notarization
disabled. It has not been notarized by Apple. A downloaded copy may need approval under
**System Settings → Privacy & Security → Open Anyway**. To produce a Developer ID-signed and
notarized release, configure those signing options explicitly. The bootstrap uv retains its original
signature and is excluded from Electron's signing pass so its embedded checksum remains valid;
if re-signing uv for notarization, do so before `prepare-bootstrap` computes those checksums.

The pinned Warp build supports Apple Silicon CPU execution, not Metal/CUDA acceleration.
Intel Mac packages are not provided because the pinned Warp wheel has no macOS x64 build.
GVHMR's NVIDIA GPU pipeline is not supported locally on Apple Silicon. SMPL-family weights remain
a separate user installation, just as on the other platforms.

To verify the actual macOS package in an isolated temporary home, without checkout or Python
overrides (the test downloads the first-run dependencies):

```bash
HHTOOLS_E2E_EXECUTABLE="$PWD/release/mac-arm64/Human-Humanoid Tools.app/Contents/MacOS/Human-Humanoid Tools" \
  npx playwright test --config playwright.config.ts e2e/mac-package.spec.ts
```

Before publishing the Windows installer, verify the final EXE stays below
[GitHub Releases' 2 GiB per-file limit](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas).
If the all-extras GPU runtime exceeds it, choose a smaller core runtime or a
separately downloaded, checksummed optional GPU payload rather than silently producing an
unpublishable release asset.

SMPL-family model files and GVHMR weights are not part of the default distributable. The separate
`npm run prepare:models` command exists only for a locally authorized build; a user checkbox cannot
grant redistribution rights for a model file.

Install the Linux package with `sudo apt install ./release/hhtools-0.1.0-amd64.deb`, then launch
`hhtools-desktop`. The first-run setup may download several gigabytes, shows live output, verifies
the bundled checksums, and restarts the app only after `hhtools doctor` succeeds. Network access is
still required for uv to acquire Python and the third-party packages named by the bundled lock. The
Debian package does not install or replace the separate `hhtools` CLI command. GVHMR remains
optional: use its dedicated setup from the Video to Motion view after the core application starts.

## Runtime model

1. Electron allocates a random `127.0.0.1` port and a per-launch session secret.
2. `SidecarSupervisor` starts `python -m hhtools.cli.desktop_sidecar`.
3. Electron waits for `/api/health`, injects the session header into requests, and only then shows
   the existing WebUI.
4. Closing Electron stops the full Python process tree before the app exits.

Development builds use the checkout and `.venv`; packaged Windows builds prefer their bundled
runtime, and packaged Linux/macOS builds prefer a completed managed installation.
`HHTOOLS_REPO_ROOT` remains an explicit development/support override. The sidecar receives only an
allowlisted environment rather than Electron's complete environment, and bundled runtimes do not
inherit the host `PYTHONPATH`.
Linux display/session and native-library variables such as `DISPLAY`,
`WAYLAND_DISPLAY`, `DBUS_SESSION_BUS_ADDRESS`, `XDG_RUNTIME_DIR`, `LD_LIBRARY_PATH`, `MUJOCO_GL`,
and `PYOPENGL_PLATFORM` are retained so GNOME, MuJoCo, and GPU runtimes can initialize normally.
