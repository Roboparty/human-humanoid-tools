# Installation

**[中文](./install_cn.md)**

- [Desktop app](#desktop-app): Windows, Linux, and macOS packages
- [From source](#from-source): full or minimal environment with uv

## Desktop app

Download a package from the [latest release](https://github.com/Roboparty/human-humanoid-tools/releases/latest). Every 0.1.6 package includes 30 built-in motions and seven robots. **ROBOTO 01** (`rp1`) is listed first, ahead of Unitree G1; **ROBOTO_ORIGIN** (`roboto_origin`) remains a separate model.

GVHMR and separately licensed SMPL-family weights are optional on all platforms and are not included. Build details are in [`desktop/README.md`](../desktop/README.md#desktop-packages).

| Platform | Package | Checksum file | Size |
|----------|---------|---------------|------|
| Windows x64 | [hhtools-0.1.6-x64-setup.exe](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-x64-setup.exe) | [SHA-256](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-x64-setup.exe.sha256) | 433 MiB |
| Linux amd64 | [hhtools-0.1.6-linux-amd64.deb](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-linux-amd64.deb) | [SHA-256](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-linux-amd64.deb.sha256) | 233 MiB |
| macOS 12+, Apple Silicon | [hhtools-0.1.6-mac-arm64.dmg](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-mac-arm64.dmg) | [SHA-256](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-mac-arm64.dmg.sha256) | 287 MiB |

### Windows

1. Download and run the EXE. The default current-user installation does not require administrator privileges.
2. If Windows shows a SmartScreen or untrusted-publisher warning, that is expected: this preview is **self-signed**, not signed with a publicly trusted commercial certificate.

The installer bundles Python, the application dependencies, and the Microsoft C++ runtime, so the core application starts without a checkout, system Python, or first-run dependency downloads. It was installed and launched on Windows x64 before release.

To check a downloaded copy in PowerShell:

```powershell
Get-FileHash .\hhtools-0.1.6-x64-setup.exe -Algorithm SHA256
```

Expected SHA-256: `50b57519bfe0cce1df110385198fb9f7a795cada808794d220e65a739212f8c9`

The bundled Torch build is CPU-only. Optional GVHMR video-to-motion uses Docker and needs a compatible NVIDIA GPU, its image and checkpoints, and licensed SMPL-X assets. These are not included, and GPU inference was not validated for this installer.

### Linux

```bash
sha256sum -c hhtools-0.1.6-linux-amd64.deb.sha256
sudo apt install ./hhtools-0.1.6-linux-amd64.deb
hhtools-desktop
```

The Debian package contains Electron plus the curated built-in motions and robots. On first launch, its setup page uses the HHTools wheel, locked dependency list, uv configuration, and uv executable carried inside the package. The recommended per-user option needs no administrator password; the all-users option opens the operating system authentication dialog.

The package was installed and launched on Linux before release.

Expected SHA-256: `6c0a0864a197c34cc1ea05582dfeb1d14a284d8d620e4e6e6742337a669ecd8a`

### macOS

1. Open the DMG and drag **Human-Humanoid Tools** into **Applications**.
2. Launch the app from Applications. This preview is ad-hoc signed and has not been notarized by Apple. If macOS blocks it, approve it under **System Settings → Privacy & Security → Open Anyway**.
3. Follow the first-run setup and keep an internet connection while it downloads an isolated Python 3.12 environment. No existing Python, source checkout, or Homebrew installation is needed, and the runtime is installed for the current user without an administrator password.

Expected SHA-256: `bd9df0930b1ae6d2a312ee2a0e6ccaaf66a6d4c8f16e1644cf0790beacdf5fc6`

Platform limits:

- Warp retargeting runs on CPU. Intel Macs and the local NVIDIA GVHMR pipeline are unsupported.
- Desktop builds hide Video → Motion, video import, V2M batch, and GVHMR setup. H2R, R2R, existing-motion imports, and SMPL-family motion support remain available. Windows, Linux, and the ordinary WebUI keep their video-to-motion entry points.

## From source

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then clone the repository. HHTools supports Python 3.12 or newer; uv selects a compatible interpreter and can install one when needed.

```bash
git clone https://github.com/Roboparty/human-humanoid-tools.git
cd human-humanoid-tools
# Only needed when no compatible Python is installed:
uv python install 3.12
```

**Requirements:** Linux and Windows support **NVIDIA GPU (CUDA 12)** acceleration. Video-to-motion additionally needs a separate CUDA-capable [GVHMR installation](gvhmr.md).

### Full environment

The project-maintained `all` extra covers formats, robots, retargeting, the WebUI, and Agent/MCP:

```bash
uv sync --locked --extra all
uv run hhtools --help
uv run hhtools web
```

Open `http://127.0.0.1:8009`. GVHMR and separately licensed SMPL-family model weights remain external; the `all` extra installs their Python integration libraries, not those model files.

### Built-in robots

The source repository contains the presets and download instructions, not the large robot model assets. Install the robots you need into your user Robot Library:

```bash
# Install all seven curated robots.
uv run python scripts/install_builtin_robots.py
# Install one preset only (repeat --only to select several).
uv run python scripts/install_builtin_robots.py --only rp1
```

The installer downloads pinned official archives and keeps the selected URDFs, referenced meshes, and upstream license files where provided, plus a checksummed `SOURCE.json`. ROBOTO 01's URDF and meshes come from the pinned public [`Roboparty/rp1_description`](https://github.com/Roboparty/rp1_description/tree/84529a55649b5c9796d49093c266208c879f9f29) repository; that revision (`84529a55649b5c9796d49093c266208c879f9f29`) does not provide a `LICENSE` file.

`--replace` atomically replaces local changes in the selected preset. Desktop bundles can point `HHTOOLS_BUNDLED_ROBOT_DIR` at an audited Robot Library.

### Smaller environments

Install only the workflow you need. The core terminal commands require no extra:

```bash
uv sync --locked
uv run hhtools --help
```

For WebUI preview and Newton retargeting:

```bash
uv sync --locked --extra web --extra retarget
uv run hhtools web
```

For a preview-only WebUI, omit `--extra retarget`. If a required package is absent, startup exits with the missing package names and the exact recovery command instead of an import traceback.

### Troubleshooting

After `git pull`, run `uv sync` and restart `uv run hhtools web` (not an older system-wide install), then hard-refresh the browser.
