#!/usr/bin/env python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""Prepare or publish a complete sbk-charts GitHub release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.project_policy import ProjectPolicy, application_version, load_policy

DEFAULT_BRANCH = "main"
DEFAULT_REMOTE = "origin"
DEFAULT_TIMEOUT_SECONDS = 3600
DEFAULT_POLL_SECONDS = 15
VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+){2,3}$")
FORBIDDEN_TRACKED_PARTS = frozenset(
    {
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".sbk-runtime",
        ".tox",
        ".venv",
        ".vscode",
        ".idea",
        ".continue",
        "__pycache__",
        "build",
        "dist",
        "htmlcov",
        "node_modules",
        "sbk-charts-venv",
        "venv-sbk-charts",
    }
)
FORBIDDEN_TRACKED_NAMES = frozenset(
    {
        ".DS_Store",
        ".coverage",
        ".env",
        ".sbk-charts-runtime",
        "coverage.xml",
        "out.xls",
        "out.xlsx",
        "SHA256SUMS",
    }
)
FORBIDDEN_TRACKED_SUFFIXES = frozenset({".pyc", ".pyo", ".whl", ".run", ".exe"})


@dataclass(frozen=True)
class ReleaseAssets:
    """Locally built package assets and remotely built portable asset names."""

    wheel: Path
    source_distribution: Path
    checksums: Path
    expected_names: frozenset[str]


def run(
    arguments: list[str],
    *,
    capture: bool = False,
    check: bool = True,
    cwd: Path = ROOT,
) -> subprocess.CompletedProcess[str]:
    """Run one command with consistent diagnostics."""
    print("+", " ".join(arguments), flush=True)
    return subprocess.run(
        arguments,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=capture,
    )


def captured(arguments: list[str], *, check: bool = True) -> str:
    """Run a command and return stripped standard output."""
    return run(arguments, capture=True, check=check).stdout.strip()


def repository_slug(policy: ProjectPolicy) -> str:
    """Resolve the GitHub owner/repository name from application policy."""
    parsed = urlparse(policy.application.url)
    if parsed.hostname != "github.com":
        raise ValueError("Application URL must identify a github.com repository")
    slug = parsed.path.strip("/").removesuffix(".git")
    if slug.count("/") != 1:
        raise ValueError("Application URL must contain a GitHub owner and repository")
    return slug


def remote_repository_slug(remote_url: str) -> str:
    """Resolve an owner/repository slug from a GitHub HTTPS or SSH remote."""
    if remote_url.startswith("git@github.com:"):
        slug = remote_url.split(":", 1)[1]
    else:
        parsed = urlparse(remote_url)
        if parsed.hostname != "github.com":
            raise ValueError("Release remote must be hosted on github.com")
        slug = parsed.path.strip("/")
    slug = slug.removesuffix(".git")
    if slug.count("/") != 1:
        raise ValueError("Release remote must contain a GitHub owner and repository")
    return slug


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one release asset."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def forbidden_tracked_files(tracked_files: list[str]) -> tuple[str, ...]:
    """Return repository paths that must not enter source archives or releases."""
    forbidden: list[str] = []
    for name in tracked_files:
        path = Path(name)
        if (
            any(part in FORBIDDEN_TRACKED_PARTS for part in path.parts)
            or any(part.endswith(".egg-info") for part in path.parts)
            or path.name in FORBIDDEN_TRACKED_NAMES
            or path.name.startswith("~$")
            or path.suffix.lower() in FORBIDDEN_TRACKED_SUFFIXES
            or (path.name.startswith("sbk_charts-") and path.name.endswith(".tar.gz"))
        ):
            forbidden.append(name)
    return tuple(sorted(forbidden))


def verify_tracked_release_sources() -> None:
    """Reject generated or checkout-only files tracked by Git."""
    tracked_output = captured(["git", "ls-files", "-z"])
    tracked_files = [name for name in tracked_output.split("\0") if name]
    forbidden = forbidden_tracked_files(tracked_files)
    if forbidden:
        raise RuntimeError(
            "Generated or checkout-only files are tracked by Git: "
            + ", ".join(forbidden)
        )


def expected_portable_asset_names(
    policy: ProjectPolicy,
    version: str,
) -> frozenset[str]:
    """Return every native application and checksum expected from CI."""
    names: set[str] = set()
    for target in policy.portable.targets:
        extension = policy.portable.self_extracting_extensions[target]
        application = f"{policy.application.name}-{version}-{target}.{extension}"
        names.add(application)
        names.add(application + policy.portable.checksum_suffix)
    return frozenset(names)


def verify_portable_asset_directory(
    policy: ProjectPolicy,
    version: str,
    directory: Path,
) -> tuple[Path, ...]:
    """Validate the exact native release asset set and every checksum sidecar."""
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"Unsupported release version: {version}")
    if version != application_version(policy):
        raise ValueError("Portable assets do not match the canonical application version")
    expected_names = expected_portable_asset_names(policy, version)
    if not directory.is_dir():
        raise ValueError(f"Portable asset directory does not exist: {directory}")
    entries = tuple(directory.iterdir())
    invalid_entries = sorted(path.name for path in entries if not path.is_file())
    if invalid_entries:
        raise ValueError(
            "Portable asset directory contains non-files: "
            + ", ".join(invalid_entries)
        )
    actual_names = frozenset(path.name for path in entries)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ValueError("Invalid portable release asset set (" + "; ".join(details) + ")")

    checksum_suffix = policy.portable.checksum_suffix
    applications = sorted(
        (directory / name)
        for name in expected_names
        if not name.endswith(checksum_suffix)
    )
    for application in applications:
        checksum_path = directory / (application.name + checksum_suffix)
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
        expected_line = f"{sha256(application)}  {application.name}"
        if lines != [expected_line]:
            raise ValueError(f"Invalid checksum file: {checksum_path.name}")
    return tuple(sorted(directory / name for name in expected_names))


def package_asset_names(policy: ProjectPolicy, version: str) -> tuple[str, str]:
    """Return the wheel and source-distribution filenames built by setuptools."""
    normalized = policy.application.distribution_name.replace("-", "_")
    return (
        f"{normalized}-{version}-py3-none-any.whl",
        f"{normalized}-{version}.tar.gz",
    )


def verify_release_checkout(
    policy: ProjectPolicy,
    version: str,
    remote: str,
    branch: str,
) -> str:
    """Require a clean checkout exactly matching the selected remote branch."""
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"Unsupported release version: {version}")
    if version != application_version(policy):
        raise ValueError("Requested release does not match the canonical application version")
    for tool in ("git", "gh"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"Required release tool is not available: {tool}")
    if captured(["git", "status", "--porcelain"]):
        raise RuntimeError("Release checkout must not contain uncommitted changes")
    verify_tracked_release_sources()
    if captured(["git", "branch", "--show-current"]) != branch:
        raise RuntimeError(f"Release must run from the {branch} branch")
    canonical_repository = repository_slug(policy)
    configured_repository = remote_repository_slug(
        captured(["git", "remote", "get-url", remote])
    )
    if configured_repository != canonical_repository:
        raise RuntimeError(
            f"Release remote must identify {canonical_repository}, not {configured_repository}"
        )
    run(["git", "fetch", "--tags", remote, branch])
    head = captured(["git", "rev-parse", "HEAD"])
    remote_head = captured(["git", "rev-parse", f"{remote}/{branch}"])
    if head != remote_head:
        raise RuntimeError(f"HEAD must exactly match {remote}/{branch}")
    run(["gh", "auth", "status", "--hostname", "github.com"])
    return head


def run_verification(python: str) -> None:
    """Run the release checks that do not require native hosted runners."""
    run([python, "-m", "unittest", "discover", "-s", "tests", "-v"])
    run(
        [
            python,
            "-m",
            "flake8",
            ".",
            "--count",
            "--select=E9,F63,F7,F82",
            "--show-source",
            "--statistics",
            "--exclude=venv-sbk-charts,.sbk-runtime,dist,build",
        ]
    )
    run(["bash", "-n", "sbk-charts"])
    run(["./sbk-charts", "-h"])
    with tempfile.TemporaryDirectory(prefix="sbk-charts-release-smoke-") as temporary:
        report = Path(temporary) / "report.xlsx"
        run(
            [
                "./sbk-charts",
                "-i",
                "samples/charts/sbk-file-read.csv",
                "-o",
                str(report),
                "noai",
            ]
        )
        run(
            [
                python,
                "-c",
                "import openpyxl,sys; "
                "w=openpyxl.load_workbook(sys.argv[1]); "
                "assert 'Summary' in w.sheetnames",
                str(report),
            ]
        )


def build_package_assets(
    policy: ProjectPolicy,
    version: str,
    output_directory: Path,
    python: str,
) -> ReleaseAssets:
    """Build wheel/sdist in isolation and write their combined checksum file."""
    output_directory.mkdir(parents=True, exist_ok=True)
    wheel_name, source_name = package_asset_names(policy, version)
    with tempfile.TemporaryDirectory(prefix="sbk-charts-release-build-") as temporary:
        build_directory = Path(temporary)
        run(
            [
                python,
                "-m",
                "build",
                "--wheel",
                "--sdist",
                "--outdir",
                str(build_directory),
            ]
        )
        wheel_source = build_directory / wheel_name
        source_archive = build_directory / source_name
        if not wheel_source.is_file() or not source_archive.is_file():
            raise RuntimeError("Package builder did not produce the expected wheel and sdist")
        wheel = Path(shutil.copy2(wheel_source, output_directory / wheel_name))
        source_distribution = Path(
            shutil.copy2(source_archive, output_directory / source_name)
        )
    checksums = output_directory / "SHA256SUMS"
    checksums.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in (wheel, source_distribution)
        ),
        encoding="utf-8",
    )
    expected = {
        wheel.name,
        source_distribution.name,
        checksums.name,
        *expected_portable_asset_names(policy, version),
    }
    return ReleaseAssets(
        wheel=wheel,
        source_distribution=source_distribution,
        checksums=checksums,
        expected_names=frozenset(expected),
    )


def previous_tag(head: str) -> str | None:
    """Return the nearest earlier tag, if this repository has one."""
    result = run(
        ["git", "describe", "--tags", "--abbrev=0", f"{head}^"],
        capture=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def change_summary(previous: str | None, head: str) -> list[str]:
    """Return release-note bullets from commit subjects since the prior tag."""
    revision = f"{previous}..{head}" if previous else head
    subjects = captured(["git", "log", "--format=%s", "--no-merges", revision])
    return [subject for subject in subjects.splitlines() if subject]


def generated_release_notes(
    policy: ProjectPolicy,
    version: str,
    repository: str,
    head: str,
    assets: ReleaseAssets,
) -> str:
    """Generate detailed, current release notes from policy and Git history."""
    earlier = previous_tag(head)
    changes = change_summary(earlier, head)
    portable_lines = []
    for target in policy.portable.targets:
        extension = policy.portable.self_extracting_extensions[target]
        portable_lines.append(
            f"- `{policy.application.name}-{version}-{target}.{extension}` -- {target}"
        )
    history = "\n".join(f"- {subject}" for subject in changes)
    if not history:
        history = "- Release packaging and maintenance updates."
    comparison = (
        f"https://github.com/{repository}/compare/{earlier}...{version}"
        if earlier
        else f"https://github.com/{repository}/commits/{version}"
    )
    return f"""# {policy.application.name} {version}

This release provides the sbk-charts command-line application, Python packages,
and native self-extracting applications for every supported portable target.

## Changes

{history}

## Downloads

### Native self-extracting applications

These files include sbk-charts, Python, and all supported AI dependencies. The
destination computer does not need a separate Python, pip, venv, or Conda setup.

{chr(10).join(portable_lines)}

Each native application has an adjacent `.sha256` file. Verify it before use.
Native files are not code-signed or notarized, so the operating system may show
a security warning.

### Python packages and source

- `{assets.wheel.name}` -- installable Python wheel
- `{assets.source_distribution.name}` -- Python source distribution
- `SHA256SUMS` -- SHA-256 values for the wheel and source distribution
- GitHub source archives -- exact repository source for this tag

The wheel and source distribution require Python {policy.runtime.minimum_python}
or newer.

## Quick start

Linux or macOS portable application:

```bash
chmod +x {policy.application.name}-{version}-<target>.run
./{policy.application.name}-{version}-<target>.run -i results.csv -o report.xlsx
```

Windows portable application:

```powershell
.\\{policy.application.name}-{version}-windows-amd64.exe -i results.csv -o report.xlsx
```

Python wheel:

```bash
python -m pip install {assets.wheel.name}
{policy.application.name} -i results.csv -o report.xlsx
```

## Release verification

- The release script runs all unit tests and focused Flake8 checks.
- The source launcher help path and sample CSV-to-XLSX path are tested.
- Wheel and source distributions are built from the tagged checkout.
- Native CI builds run on each policy-declared operating system.
- Every native build tests concurrent first extraction, saved reuse, and workbook creation.

See the [README](https://github.com/{repository}/blob/{version}/README.md),
[portable guide](https://github.com/{repository}/blob/{version}/docs/PORTABLE.md),
and [architecture guide](https://github.com/{repository}/blob/{version}/docs/ARCHITECTURE.md).

**Full changelog:** {comparison}
"""


def release_information(repository: str, version: str) -> dict[str, object] | None:
    """Return an existing GitHub release, or None when it does not exist."""
    result = run(
        [
            "gh",
            "release",
            "view",
            version,
            "--repo",
            repository,
            "--json",
            "tagName,isDraft,isPrerelease,assets,url",
        ],
        capture=True,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.strip()
        if "release not found" in error.lower() or "not found" in error.lower():
            return None
        raise RuntimeError(f"Cannot inspect GitHub release {version}: {error}")
    return json.loads(result.stdout)


def release_asset_names(release: dict[str, object]) -> frozenset[str]:
    """Return normalized asset names from a GitHub release response."""
    assets = release.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError("GitHub release assets must be a list")
    return frozenset(
        str(asset["name"])
        for asset in assets
        if isinstance(asset, dict) and "name" in asset
    )


def remote_tag_commit(remote_tags: str, version: str) -> str | None:
    """Return the commit for one exact remote tag, preferring its peeled ref."""
    exact_ref = f"refs/tags/{version}"
    peeled_ref = exact_ref + "^{}"
    matching_refs: dict[str, str] = {}
    for line in remote_tags.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] in {exact_ref, peeled_ref}:
            matching_refs[fields[1]] = fields[0]
    return matching_refs.get(peeled_ref) or matching_refs.get(exact_ref)


def ensure_tag(version: str, head: str, remote: str) -> None:
    """Create and push the version tag, or validate an existing matching tag."""
    result = run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{version}"], capture=True, check=False)
    if result.returncode == 0:
        tagged_commit = captured(["git", "rev-list", "-n", "1", version])
        if tagged_commit != head:
            raise RuntimeError(f"Existing tag {version} does not identify HEAD")
    else:
        run(["git", "tag", "--annotate", version, "--message", f"sbk-charts {version}"])
    remote_tags = captured(
        [
            "git",
            "ls-remote",
            "--tags",
            remote,
            f"refs/tags/{version}",
            f"refs/tags/{version}^{{}}",
        ],
        check=False,
    )
    remote_commit = remote_tag_commit(remote_tags, version)
    if remote_commit:
        if remote_commit != head:
            raise RuntimeError(f"Remote tag {version} does not identify HEAD")
    else:
        run(["git", "push", remote, f"refs/tags/{version}"])


def publish_release(
    repository: str,
    version: str,
    notes_file: Path,
    assets: ReleaseAssets,
    resume: bool,
) -> tuple[str, bool]:
    """Create or resume a release and report whether it was already published."""
    existing = release_information(repository, version)
    already_published = existing is not None and not bool(existing["isDraft"])
    upload_paths = [assets.wheel, assets.source_distribution, assets.checksums]
    if existing is None:
        run(
            [
                "gh",
                "release",
                "create",
                version,
                "--repo",
                repository,
                "--verify-tag",
                "--draft",
                "--title",
                f"sbk-charts {version}",
                "--notes-file",
                str(notes_file),
                *[str(path) for path in upload_paths],
            ]
        )
        existing = release_information(repository, version)
    elif not resume:
        raise RuntimeError(
            f"GitHub release {version} already exists; inspect it and rerun with --resume"
        )
    if existing is None:
        raise RuntimeError("GitHub release was not visible after creation")
    if bool(existing["isPrerelease"]):
        raise RuntimeError("Existing release is unexpectedly marked as a prerelease")
    if resume:
        run(
            [
                "gh",
                "release",
                "upload",
                version,
                "--repo",
                repository,
                "--clobber",
                *[str(path) for path in upload_paths],
            ]
        )
    if bool(existing["isDraft"]):
        run(
            [
                "gh",
                "release",
                "edit",
                version,
                "--repo",
                repository,
                "--title",
                f"sbk-charts {version}",
                "--notes-file",
                str(notes_file),
                "--draft=false",
            ]
        )
    published = release_information(repository, version)
    if published is None or bool(published["isDraft"]):
        raise RuntimeError("GitHub release was not published")
    return str(published["url"]), already_published


def restart_portable_workflow(repository: str, version: str) -> None:
    """Rebuild native assets for an existing immutable release tag."""
    run(
        [
            "gh",
            "workflow",
            "run",
            "portable.yml",
            "--repo",
            repository,
            "--ref",
            version,
            "--field",
            f"release_tag={version}",
        ]
    )


def wait_for_assets(
    repository: str,
    version: str,
    expected_names: frozenset[str],
    timeout_seconds: int,
    poll_seconds: int,
) -> None:
    """Wait until hosted native builds attach every required release asset."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        release = release_information(repository, version)
        if release is None:
            raise RuntimeError("Published release disappeared while waiting for assets")
        actual = release_asset_names(release)
        missing = sorted(expected_names - actual)
        if not missing:
            print(f"All {len(expected_names)} required release assets are attached.")
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for release assets: " + ", ".join(missing))
        print("Waiting for native release assets:", ", ".join(missing), flush=True)
        time.sleep(poll_seconds)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse release preparation and publication options."""
    parser = argparse.ArgumentParser(prog="create_github_release.py", description=__doc__)
    parser.add_argument("--publish", action="store_true", help="Push the tag and publish on GitHub")
    parser.add_argument("--resume", action="store_true", help="Resume an existing matching release")
    parser.add_argument("--version", help="Release version; defaults to the canonical project version")
    parser.add_argument("--notes-file", type=Path, help="Use reviewed release notes instead of generated notes")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "release")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python for tests, lint, and package builds; the launcher selects its own runtime",
    )
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    selected = parser.parse_args(arguments)
    if selected.resume and not selected.publish:
        parser.error("--resume requires --publish")
    if selected.timeout_seconds < 1 or selected.poll_seconds < 1:
        parser.error("timeouts must be positive integers")
    return selected


def main(arguments: list[str] | None = None) -> int:
    """Prepare release assets and optionally publish the complete release."""
    selected = parse_args(arguments)
    policy = load_policy()
    version = selected.version or application_version(policy)
    repository = repository_slug(policy)
    head = verify_release_checkout(
        policy,
        version,
        DEFAULT_REMOTE,
        DEFAULT_BRANCH,
    )
    run_verification(selected.python)
    assets = build_package_assets(policy, version, selected.output.resolve(), selected.python)
    notes_file = selected.output.resolve() / "RELEASE_NOTES.md"
    if selected.notes_file:
        notes_file.write_text(selected.notes_file.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        notes_file.write_text(
            generated_release_notes(policy, version, repository, head, assets),
            encoding="utf-8",
        )
    print(f"Prepared release {version} in {selected.output.resolve()}")
    if not selected.publish:
        print("Preparation complete. Review RELEASE_NOTES.md, then rerun with --publish.")
        return 0
    ensure_tag(version, head, DEFAULT_REMOTE)
    release_url, already_published = publish_release(
        repository,
        version,
        notes_file,
        assets,
        selected.resume,
    )
    portable_names = expected_portable_asset_names(policy, version)
    published = release_information(repository, version)
    if published is None:
        raise RuntimeError("Published release disappeared before asset verification")
    if selected.resume and already_published and not portable_names.issubset(
        release_asset_names(published)
    ):
        restart_portable_workflow(repository, version)
    wait_for_assets(
        repository,
        version,
        assets.expected_names,
        selected.timeout_seconds,
        selected.poll_seconds,
    )
    print(f"Release complete: {release_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
