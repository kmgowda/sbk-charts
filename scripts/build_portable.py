#!/usr/bin/env python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""Build one self-extracting, checksummed sbk-charts application for this host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import struct
import zipfile
from pathlib import Path, PurePath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.project_policy import ProjectPolicy, application_version, load_policy

POLICY = load_policy()
COPY_CHUNK_SIZE = 1024 * 1024
LOCK_OWNER_GRACE_ATTEMPTS = 5


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
        for chunk in iter(lambda: source.read(COPY_CHUNK_SIZE), b""):
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


def zip_member_name(bundle_name: str, relative_path: PurePath) -> str:
    """Return a ZIP-standard member name for paths from any host OS."""
    return f"{bundle_name}/{relative_path.as_posix()}"


def create_payload(bundle: Path, target: str, destination: Path, policy: ProjectPolicy) -> None:
    """Create the compressed payload embedded in a self-extracting application."""
    archive_format = policy.portable.archive_formats[target]
    if archive_format == "zip":
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    output.write(path, path.relative_to(bundle).as_posix())
    elif archive_format == "tar.gz":
        with tarfile.open(destination, "w:gz") as output:
            for path in sorted(bundle.iterdir()):
                output.add(path, arcname=path.name)
    else:
        raise ValueError(f"Unsupported portable payload format: {archive_format}")


def unix_launcher(
    policy: ProjectPolicy,
    version: str,
    target: str,
    payload_sha256: str,
) -> bytes:
    """Return a POSIX self-extracting launcher prefix for Linux or macOS."""
    expected_kernel = "Linux" if target.startswith("linux-") else "Darwin"
    expected_machine = "x86_64" if target.endswith("-amd64") else "arm64"
    script = f'''#!/bin/sh
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##
set -eu
app_name='{policy.application.name}'
version='{version}'
target='{target}'
state_schema='{policy.portable.runtime_state_schema}'
payload_sha256='{payload_sha256}'
payload_line=__PAYLOAD_LINE__
lock_timeout='{policy.portable.bootstrap_lock_timeout_seconds}'
expected_kernel='{expected_kernel}'
expected_machine='{expected_machine}'
self_path=$0
case "$self_path" in /*) ;; *) self_path=$(pwd)/$self_path ;; esac
fail() {{ printf '%s: ERROR: %s\n' "$app_name" "$*" >&2; exit 1; }}
[ "$(uname -s)" = "$expected_kernel" ] || fail "This application requires $target"
machine=$(uname -m)
case "$expected_machine:$machine" in
    x86_64:x86_64|arm64:arm64|arm64:aarch64) ;;
    *) fail "This application requires $target; detected $machine" ;;
esac
if [ -n "${{SBK_CHARTS_PORTABLE_ROOT:-}}" ]; then
    runtime_root=$SBK_CHARTS_PORTABLE_ROOT
elif [ "$expected_kernel" = Darwin ]; then
    [ -n "${{HOME:-}}" ] || fail "HOME is unset; set SBK_CHARTS_PORTABLE_ROOT"
    runtime_root="$HOME/Library/Caches/{policy.portable.runtime_directory}"
else
    [ -n "${{XDG_CACHE_HOME:-${{HOME:-}}}}" ] || fail "HOME is unset; set SBK_CHARTS_PORTABLE_ROOT"
    runtime_root="${{XDG_CACHE_HOME:-$HOME/.cache}}/{policy.portable.runtime_directory}"
fi
install_dir="$runtime_root/$version/$target/$payload_sha256"
state_file="$runtime_root/state-$target"
lock_dir="$runtime_root/bootstrap-$target.lock"
executable="$install_dir/{policy.application.name}"
marker="$install_dir/.payload.sha256"
state_is_ready() {{
    [ -x "$executable" ] && [ -r "$marker" ] &&
    [ "$(sed -n '1p' "$marker")" = "$payload_sha256" ] && [ -r "$state_file" ] &&
    grep -Fqx "schema=$state_schema" "$state_file" &&
    grep -Fqx "target=$target" "$state_file" &&
    grep -Fqx "version=$version" "$state_file" &&
    grep -Fqx "payload_sha256=$payload_sha256" "$state_file" &&
    grep -Fqx "install_dir=$install_dir" "$state_file"
}}
sha256_file() {{
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{{print $1}}'
    elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{{print $1}}'
    else fail "sha256sum or shasum is required for payload verification"; fi
}}
lock_acquired=no
release_lock() {{
    [ "$lock_acquired" = yes ] || return 0
    owner=$(sed -n '1p' "$lock_dir/pid" 2>/dev/null || true)
    if [ "$owner" = "$$" ]; then rm -rf "$lock_dir"; fi
    lock_acquired=no
}}
temporary=
cleanup() {{
    if [ -n "${{temporary:-}}" ] && [ -d "$temporary" ]; then rm -rf "$temporary"; fi
    release_lock
}}
created=no
reused=no
selection_source=self-extract-cache
if state_is_ready; then
    reused=yes
else
    mkdir -p "$runtime_root"
    attempts=0
    unowned_attempts=0
    trap cleanup EXIT HUP INT TERM
    while ! mkdir "$lock_dir" 2>/dev/null; do
        lock_has_owner=no
        if [ -r "$lock_dir/pid" ]; then
            owner=$(sed -n '1p' "$lock_dir/pid")
            case "$owner" in
                *[!0-9]*|'') ;;
                *)
                    lock_has_owner=yes
                    if ! kill -0 "$owner" 2>/dev/null; then
                        rm -rf "$lock_dir"
                        continue
                    fi
                    ;;
            esac
        fi
        if [ "$lock_has_owner" = yes ]; then
            unowned_attempts=0
        else
            unowned_attempts=$((unowned_attempts + 1))
            if [ "$unowned_attempts" -ge {LOCK_OWNER_GRACE_ATTEMPTS} ]; then
                rm -rf "$lock_dir"
                unowned_attempts=0
                continue
            fi
        fi
        attempts=$((attempts + 1))
        [ "$attempts" -lt "$lock_timeout" ] || fail "Timed out waiting for $lock_dir"
        sleep 1
    done
    if ! printf '%s\n' "$$" > "$lock_dir/pid"; then
        rm -rf "$lock_dir"
        fail "Cannot record lock owner in $lock_dir"
    fi
    lock_acquired=yes
    if state_is_ready; then
        reused=yes
    else
        temporary=$(mktemp -d "$runtime_root/.install.XXXXXX") || fail "Cannot create temporary directory"
        archive="$temporary/payload.tar.gz"
        tail -n "+$payload_line" "$self_path" > "$archive" || fail "Cannot read embedded payload"
        [ "$(sha256_file "$archive")" = "$payload_sha256" ] || fail "Embedded payload failed SHA-256 verification"
        mkdir "$temporary/content"
        tar -xzf "$archive" -C "$temporary/content" || fail "Cannot extract embedded payload"
        [ -x "$temporary/content/{policy.application.name}" ] || fail "Extracted application is not executable"
        printf '%s\n' "$payload_sha256" > "$temporary/content/.payload.sha256"
        if [ -e "$install_dir" ]; then rm -rf "$install_dir"; fi
        mkdir -p "$(dirname "$install_dir")"
        mv "$temporary/content" "$install_dir" || fail "Cannot publish extracted application"
        rm -rf "$temporary"
        temporary=
        temporary_state="$runtime_root/.state-$$.tmp"
        {{
            printf 'schema=%s\n' "$state_schema"
            printf 'target=%s\n' "$target"
            printf 'version=%s\n' "$version"
            printf 'payload_sha256=%s\n' "$payload_sha256"
            printf 'install_dir=%s\n' "$install_dir"
        }} > "$temporary_state"
        mv "$temporary_state" "$state_file"
        created=yes
        selection_source=self-extract-created
    fi
    release_lock
    trap - EXIT HUP INT TERM
fi
export SBK_CHARTS_PORTABLE_SELECTION_SOURCE=$selection_source
export SBK_CHARTS_PORTABLE_REUSED=$reused
export SBK_CHARTS_PORTABLE_CREATED=$created
export SBK_CHARTS_PORTABLE_PREFIX=$install_dir
exec "$executable" "$@"
'''
    payload_line = script.count("\n") + 1
    return script.replace("__PAYLOAD_LINE__", str(payload_line)).encode("utf-8")


def windows_csharp_compiler() -> Path:
    """Find the .NET Framework C# compiler available on Windows runners."""
    direct = shutil.which("csc")
    if direct:
        return Path(direct)
    windows = Path(os.environ.get("WINDIR", r"C:\Windows"))
    candidates = (
        windows / "Microsoft.NET" / "Framework64" / "v4.0.30319" / "csc.exe",
        windows / "Microsoft.NET" / "Framework" / "v4.0.30319" / "csc.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("The Windows .NET Framework C# compiler was not found")


def windows_launcher_source(
    policy: ProjectPolicy,
    version: str,
    target: str,
    payload_sha256: str,
) -> str:
    """Render the reviewed native Windows extractor template."""
    template = (ROOT / "scripts" / "windows_self_extractor.cs").read_text(encoding="utf-8")
    replacements = {
        "@@APP_NAME@@": policy.application.name,
        "@@VERSION@@": version,
        "@@TARGET@@": target,
        "@@PAYLOAD_SHA256@@": payload_sha256,
        "@@STATE_SCHEMA@@": str(policy.portable.runtime_state_schema),
        "@@RUNTIME_DIRECTORY@@": policy.portable.runtime_directory.replace("/", r"\\"),
        "@@LOCK_TIMEOUT_SECONDS@@": str(policy.portable.bootstrap_lock_timeout_seconds),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    if "@@" in template:
        raise ValueError("The Windows self-extractor template has unresolved placeholders")
    return template


def compile_windows_launcher(
    destination: Path,
    policy: ProjectPolicy,
    version: str,
    target: str,
    payload_sha256: str,
) -> None:
    """Compile the small native Windows extractor from its reviewed template."""
    source_text = windows_launcher_source(
        policy, version, target, payload_sha256
    )
    source = destination.with_suffix(".cs")
    source.write_text(source_text, encoding="utf-8")
    try:
        subprocess.run(
            [
                str(windows_csharp_compiler()),
                "/nologo",
                "/target:exe",
                f"/out:{destination}",
                "/reference:System.IO.Compression.dll",
                "/reference:System.IO.Compression.FileSystem.dll",
                str(source),
            ],
            check=True,
            cwd=ROOT,
        )
    finally:
        source.unlink(missing_ok=True)


def create_self_extracting_application(
    bundle: Path,
    output_directory: Path,
    version: str,
    target: str,
    policy: ProjectPolicy,
) -> Path:
    """Wrap a frozen bundle in one verified persistent self-extracting file."""
    payload_suffix = ".zip" if target.startswith("windows-") else ".tar.gz"
    payload = bundle.parent / f"payload{payload_suffix}"
    create_payload(bundle, target, payload, policy)
    payload_digest = sha256(payload)
    extension = policy.portable.self_extracting_extensions[target]
    artifact = output_directory / f"{policy.application.name}-{version}-{target}.{extension}"
    if target.startswith("windows-"):
        compile_windows_launcher(artifact, policy, version, target, payload_digest)
        with artifact.open("ab") as output, payload.open("rb") as source:
            shutil.copyfileobj(source, output, length=COPY_CHUNK_SIZE)
            output.write(struct.pack("<Q", payload.stat().st_size))
    else:
        with artifact.open("wb") as output, payload.open("rb") as source:
            output.write(unix_launcher(policy, version, target, payload_digest))
            shutil.copyfileobj(source, output, length=COPY_CHUNK_SIZE)
        artifact.chmod(0o755)
    return artifact


def build_bundle(output_directory: Path, policy: ProjectPolicy = POLICY) -> Path:
    """Build, smoke-test, manifest, and wrap one portable application."""
    version = application_version(policy)
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

        archive = create_self_extracting_application(
            bundle, output_directory, version, target, policy
        )

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
