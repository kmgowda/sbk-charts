# Portable sbk-charts distributions

sbk-charts supports two self-bootstrapping delivery models:

1. A source checkout or source archive uses the root launchers and prepares a
   Python 3.10+ virtual or Conda environment when necessary.
2. A standalone portable archive bundles the Python runtime, sbk-charts, and
   its Python dependencies. The destination host does not need Python, pip,
   venv, or Conda.

## Standalone release archives

Choose the archive matching the destination host:

| Archive suffix | Host |
|---|---|
| `linux-amd64.tar.gz` | Linux x86-64 |
| `macos-arm64.tar.gz` | macOS Apple silicon |
| `windows-amd64.zip` | Windows x86-64 |

Download the archive and its adjacent `.sha256` file from the GitHub release.
Verify the checksum before extraction:

```bash
sha256sum --check sbk-charts-<version>-linux-amd64.tar.gz.sha256
```

On macOS, use `shasum -a 256` and compare the printed value with the checksum
file. On Windows PowerShell, use `Get-FileHash -Algorithm SHA256`.

Extract the complete directory and run the platform executable:

```bash
./sbk-charts -i /path/to/results.csv -o report.xlsx
```

```powershell
.\sbk-charts.exe -i C:\path\to\results.csv -o report.xlsx
```

Do not copy only the executable. PyInstaller's `_internal` directory is part
of the application and must remain next to it. Each bundle also contains the
license, README, documentation, and a `manifest.json` with SHA-256 hashes for
every bundled file.

Portable archives are native builds rather than cross-compiled artifacts.
Release automation builds and smoke-tests each archive on its target GitHub
runner. The Linux build deliberately uses the official CPU-only PyTorch wheel
so the archive does not contain unused CUDA runtimes; PyTorch-based analysis
remains available on the CPU. Archives are checksummed but are not code-signed
or notarized; apply your organization's signing and trust policies when
required.

## Building locally

Install the reviewed build-tool versions into a Python 3.10+ environment that
already has sbk-charts installed, then run:

```bash
python -m pip install "pyinstaller==6.22.0" "pyinstaller-hooks-contrib==2026.6"
python scripts/build_portable.py
```

The archive and checksum are written to `dist/portable/`. A build is valid
only for the operating system and processor architecture on which it ran. The
builder runs the frozen executable with `--help` before creating the archive.
For a release-sized Linux build, install the CPU-only PyTorch wheel before
installing sbk-charts, matching `.github/workflows/portable.yml`.

## Operational notes

- All normal sbk-charts command-line arguments are supported.
- Input CSV and output workbook paths remain external to the bundle.
- Cloud AI plugins still require their normal credentials and network access.
- Local AI plugins retain their model and hardware requirements.
- Upgrade by extracting a newer archive into a new directory; do not overlay
  `_internal` directories from different versions.
