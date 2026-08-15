#!/usr/bin/env python3
"""Build a self-contained, checksummed sbk-charts archive for the current host."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPLICATION_NAME = "sbk-charts"


def application_version() -> str:
    """Read the sbk-charts version without importing application dependencies."""
    namespace: dict[str, object] = {}
    version_file = ROOT / "src" / "version" / "sbk_version.py"
    exec(version_file.read_text(encoding="utf-8"), namespace)
    return str(namespace["__sbk_version__"])


def current_platform() -> str:
    """Return the portable target identifier for the current native host."""
    operating_system = {"darwin": "macos", "linux": "linux", "win32": "windows"}.get(sys.platform)
    if operating_system is None:
        raise SystemExit(f"Unsupported portable-build operating system: {sys.platform}.")
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        architecture = "arm64"
    elif machine in {"amd64", "x86_64", "x64"}:
        architecture = "amd64"
    else:
        raise SystemExit(f"Unsupported portable-build architecture: {platform.machine()}.")
    return f"{operating_system}-{architecture}"


def sha256(path: Path) -> str:
    """Calculate a file SHA-256 digest without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_executable(directory: Path, target: str) -> Path:
    """Resolve the frozen executable inside a PyInstaller onedir output."""
    suffix = ".exe" if target.startswith("windows-") else ""
    return directory / f"{APPLICATION_NAME}{suffix}"


def build_bundle(output_directory: Path) -> Path:
    """Build, smoke-test, manifest, archive, and checksum a portable bundle."""
    version = application_version()
    target = current_platform()
    bundle_name = f"{APPLICATION_NAME}-{version}-{target}"
    output_directory.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sbk-charts-portable-") as temporary:
        work = Path(temporary)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "PyInstaller",
                "--noconfirm",
                "--clean",
                "--onedir",
                "--name",
                APPLICATION_NAME,
                "--paths",
                str(ROOT),
                "--collect-data",
                "src.main",
                "--collect-data",
                "src.images",
                "--collect-submodules",
                "src.custom_ai",
                "--distpath",
                str(work / "dist"),
                "--workpath",
                str(work / "work"),
                "--specpath",
                str(work),
                str(ROOT / "scripts" / "sbk_charts_portable_entry.py"),
            ],
            check=True,
            cwd=ROOT,
        )

        frozen = work / "dist" / APPLICATION_NAME
        subprocess.run(
            [str(portable_executable(frozen, target)), "--help"],
            check=True,
            cwd=frozen,
        )

        bundle = work / bundle_name
        shutil.copytree(frozen, bundle)
        shutil.copy2(ROOT / "LICENSE", bundle / "LICENSE")
        shutil.copy2(ROOT / "README.md", bundle / "README.md")
        shutil.copytree(ROOT / "docs", bundle / "docs")

        files = {
            path.relative_to(bundle).as_posix(): sha256(path)
            for path in sorted(bundle.rglob("*"))
            if path.is_file()
        }
        (bundle / "manifest.json").write_text(
            json.dumps({"version": version, "platform": target, "files": files}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if target.startswith("windows-"):
            archive = output_directory / f"{bundle_name}.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
                for path in sorted(bundle.rglob("*")):
                    if path.is_file():
                        output.write(path, Path(bundle_name) / path.relative_to(bundle))
        else:
            archive = output_directory / f"{bundle_name}.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                output.add(bundle, arcname=bundle_name)

    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    return archive


def main() -> int:
    """Parse build options and create the current platform archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "portable")
    selected = parser.parse_args()
    archive = build_bundle(selected.output.resolve())
    print(f"Built {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
