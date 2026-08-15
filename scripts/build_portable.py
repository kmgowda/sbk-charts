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

from scripts.project_policy import ProjectPolicy, load_policy

ROOT = Path(__file__).resolve().parents[1]
POLICY = load_policy()


def application_version() -> str:
    """Read the sbk-charts version without importing application dependencies."""
    namespace: dict[str, object] = {}
    version_file = ROOT / POLICY.application.version_file
    exec(version_file.read_text(encoding="utf-8"), namespace)
    return str(namespace["__sbk_version__"])


def current_platform(policy: ProjectPolicy = POLICY) -> str:
    """Return the portable target identifier for the current native host."""
    operating_system = policy.portable.platforms.get(sys.platform)
    if operating_system is None:
        raise SystemExit(f"Unsupported portable-build operating system: {sys.platform}.")
    machine = platform.machine().lower()
    architecture = policy.portable.architectures.get(machine)
    if architecture is None:
        raise SystemExit(f"Unsupported portable-build architecture: {platform.machine()}.")
    target = f"{operating_system}-{architecture}"
    if target not in policy.portable.targets:
        raise SystemExit(f"Unsupported portable-build target: {target}.")
    return target


def sha256(path: Path) -> str:
    """Calculate a file SHA-256 digest without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_executable(directory: Path, target: str, policy: ProjectPolicy = POLICY) -> Path:
    """Resolve the frozen executable inside a PyInstaller onedir output."""
    suffix = ".exe" if target.startswith("windows-") else ""
    return directory / f"{policy.application.name}{suffix}"


def copy_bundle_paths(bundle: Path, policy: ProjectPolicy) -> None:
    """Copy every policy-declared documentation or metadata path."""
    for relative_name in policy.portable.bundle_paths:
        source = ROOT / relative_name
        destination = bundle / relative_name
        if source.is_dir():
            shutil.copytree(source, destination)
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        else:
            raise FileNotFoundError(f"Portable bundle path does not exist: {source}")


def build_bundle(output_directory: Path, policy: ProjectPolicy = POLICY) -> Path:
    """Build, smoke-test, manifest, archive, and checksum a portable bundle."""
    version = application_version()
    target = current_platform(policy)
    application_name = policy.application.name
    bundle_name = f"{application_name}-{version}-{target}"
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
                application_name,
                "--paths",
                str(ROOT),
                *[
                    argument
                    for package in policy.package_data
                    for argument in ("--collect-data", package)
                ],
                *[
                    argument
                    for package in policy.portable.collect_submodules
                    for argument in ("--collect-submodules", package)
                ],
                "--distpath",
                str(work / "dist"),
                "--workpath",
                str(work / "work"),
                "--specpath",
                str(work),
                str(ROOT / policy.portable.entry_script),
            ],
            check=True,
            cwd=ROOT,
        )

        frozen = work / "dist" / application_name
        subprocess.run(
            [str(portable_executable(frozen, target, policy)), "--help"],
            check=True,
            cwd=frozen,
        )

        bundle = work / bundle_name
        shutil.copytree(frozen, bundle)
        copy_bundle_paths(bundle, policy)

        files = {
            path.relative_to(bundle).as_posix(): sha256(path)
            for path in sorted(bundle.rglob("*"))
            if path.is_file()
        }
        (bundle / policy.portable.manifest_name).write_text(
            json.dumps(
                {
                    "application": application_name,
                    "archive_format": policy.portable.archive_formats[target],
                    "files": files,
                    "hash_algorithm": policy.portable.hash_algorithm,
                    "platform": target,
                    "version": version,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )

        archive_format = policy.portable.archive_formats[target]
        if archive_format == "zip":
            archive = output_directory / f"{bundle_name}.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
                for path in sorted(bundle.rglob("*")):
                    if path.is_file():
                        output.write(path, Path(bundle_name) / path.relative_to(bundle))
        elif archive_format == "tar.gz":
            archive = output_directory / f"{bundle_name}.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                output.add(bundle, arcname=bundle_name)
        else:
            raise ValueError(f"Unsupported portable archive format: {archive_format}")

    checksum = archive.with_suffix(archive.suffix + policy.portable.checksum_suffix)
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
