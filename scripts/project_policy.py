#!/usr/bin/env python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""Read and validate shared sbk-charts runtime and artifact policy."""

from __future__ import annotations

import argparse
import ast
import configparser
import importlib
import json
import os
import platform
import re
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_FILE = ROOT / "sbk-charts.ini"
ENVIRONMENT_KINDS = frozenset({"managed", "venv", "conda"})
SELECTION_SOURCES = frozenset(
    {
        "unknown",
        "explicit-venv",
        "saved-state",
        "active-venv",
        "active-conda",
        "project-venv",
        "managed-cache",
        "named-conda",
        "created-managed",
        "created-venv",
        "created-conda",
    }
)


def _items(value: str) -> tuple[str, ...]:
    """Return a normalized tuple from a comma-separated policy value."""
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class ApplicationMetadata:
    """Package and command metadata shared by builders and packaging."""

    name: str
    distribution_name: str
    module: str
    entry_point: str
    description: str
    url: str
    license: str
    author: str
    author_email: str
    version_file: str
    runtime_requirements: str


@dataclass(frozen=True)
class RuntimePolicy:
    """Python and environment-selection policy shared by launchers."""

    minimum_python: str
    managed_python: str
    default_conda_environment: str
    default_profile: str
    runtime_state_file: str
    runtime_state_schema: int
    virtual_environment_names: tuple[str, ...]
    managed_runtime_directory: str
    lock_directory: str
    bootstrap_lock_timeout_seconds: int
    unix_python_commands: tuple[str, ...]
    windows_python_launchers: tuple[str, ...]


@dataclass(frozen=True)
class BootstrapPolicy:
    """Pinned standalone runtime-manager download policy."""

    manager: str
    manager_version: str
    download_base_url: str
    connect_timeout_seconds: int
    download_timeout_seconds: int
    download_retries: int
    archives: dict[str, str]
    checksums: dict[str, str]


@dataclass(frozen=True)
class PortableArtifactPolicy:
    """Naming, contents, hashing, and target rules for portable applications."""

    targets: tuple[str, ...]
    manifest_name: str
    checksum_suffix: str
    hash_algorithm: str
    build_python: str
    runtime_state_schema: int
    runtime_directory: str
    bootstrap_lock_timeout_seconds: int
    bundle_paths: tuple[str, ...]
    entry_script: str
    collect_submodules: tuple[str, ...]
    platforms: dict[str, str]
    architectures: dict[str, str]
    archive_formats: dict[str, str]
    self_extracting_extensions: dict[str, str]
    runners: dict[str, str]


@dataclass(frozen=True)
class ProjectPolicy:
    """Complete deployment policy loaded from ``sbk-charts.ini``."""

    application: ApplicationMetadata
    runtime: RuntimePolicy
    bootstrap: BootstrapPolicy
    portable: PortableArtifactPolicy
    package_data: dict[str, tuple[str, ...]]
    ai_requirements: dict[str, str]


def load_policy(path: Path = POLICY_FILE) -> ProjectPolicy:
    """Load the central policy file and fail clearly when it is incomplete."""
    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(path, encoding="utf-8"):
        raise FileNotFoundError(f"Project policy file not found: {path}")

    application = ApplicationMetadata(**dict(parser["application"]))
    runtime_section = parser["runtime"]
    runtime = RuntimePolicy(
        minimum_python=runtime_section["minimum_python"],
        managed_python=runtime_section["managed_python"],
        default_conda_environment=runtime_section["default_conda_environment"],
        default_profile=runtime_section["default_profile"],
        runtime_state_file=runtime_section["runtime_state_file"],
        runtime_state_schema=runtime_section.getint("runtime_state_schema"),
        virtual_environment_names=_items(runtime_section["virtual_environment_names"]),
        managed_runtime_directory=runtime_section["managed_runtime_directory"],
        lock_directory=runtime_section["lock_directory"],
        bootstrap_lock_timeout_seconds=runtime_section.getint("bootstrap_lock_timeout_seconds"),
        unix_python_commands=_items(runtime_section["unix_python_commands"]),
        windows_python_launchers=_items(runtime_section["windows_python_launchers"]),
    )
    bootstrap_section = parser["bootstrap"]
    bootstrap_archives = {
        key.removesuffix("-archive"): value
        for key, value in bootstrap_section.items()
        if key.endswith("-archive")
    }
    bootstrap_checksums = {
        key.removesuffix("-sha256"): value
        for key, value in bootstrap_section.items()
        if key.endswith("-sha256")
    }
    bootstrap = BootstrapPolicy(
        manager=bootstrap_section["manager"],
        manager_version=bootstrap_section["manager_version"],
        download_base_url=bootstrap_section["download_base_url"],
        connect_timeout_seconds=bootstrap_section.getint("connect_timeout_seconds"),
        download_timeout_seconds=bootstrap_section.getint("download_timeout_seconds"),
        download_retries=bootstrap_section.getint("download_retries"),
        archives=bootstrap_archives,
        checksums=bootstrap_checksums,
    )
    portable_section = parser["portable"]
    portable = PortableArtifactPolicy(
        targets=_items(portable_section["targets"]),
        manifest_name=portable_section["manifest_name"],
        checksum_suffix=portable_section["checksum_suffix"],
        hash_algorithm=portable_section["hash_algorithm"],
        build_python=portable_section["build_python"],
        runtime_state_schema=portable_section.getint("runtime_state_schema"),
        runtime_directory=portable_section["runtime_directory"],
        bootstrap_lock_timeout_seconds=portable_section.getint(
            "bootstrap_lock_timeout_seconds"
        ),
        bundle_paths=_items(portable_section["bundle_paths"]),
        entry_script=portable_section["entry_script"],
        collect_submodules=_items(portable_section["collect_submodules"]),
        platforms=dict(parser["portable.platforms"]),
        architectures=dict(parser["portable.architectures"]),
        archive_formats=dict(parser["portable.archive_formats"]),
        self_extracting_extensions=dict(parser["portable.self_extracting_extensions"]),
        runners=dict(parser["portable.runners"]),
    )

    missing_formats = set(portable.targets) - set(portable.archive_formats)
    unknown_formats = set(portable.archive_formats) - set(portable.targets)
    if missing_formats or unknown_formats:
        raise ValueError(
            "Portable targets and archive formats differ: "
            f"missing={sorted(missing_formats)}, unknown={sorted(unknown_formats)}"
        )
    if set(portable.targets) != set(portable.runners):
        raise ValueError("Every portable target must have exactly one native runner")
    if set(portable.targets) != set(portable.self_extracting_extensions):
        raise ValueError(
            "Every portable target must have exactly one self-extracting extension"
        )
    expected_extensions = {
        target: "exe" if target.startswith("windows-") else "run"
        for target in portable.targets
    }
    if portable.self_extracting_extensions != expected_extensions:
        raise ValueError(
            "Portable self-extracting extensions must be run on Unix and exe on Windows"
        )
    if portable.hash_algorithm != "sha256":
        raise ValueError(f"Unsupported portable hash algorithm: {portable.hash_algorithm}")
    if not runtime.virtual_environment_names:
        raise ValueError("At least one virtual environment name is required")
    if not runtime.default_profile.strip():
        raise ValueError("Default dependency profile must not be empty")
    if runtime.bootstrap_lock_timeout_seconds < 1:
        raise ValueError("Bootstrap lock timeout must be at least one second")
    if runtime.runtime_state_schema < 1:
        raise ValueError("Runtime state schema must be at least one")
    if portable.runtime_state_schema < 1:
        raise ValueError("Portable runtime state schema must be at least one")
    if portable.bootstrap_lock_timeout_seconds < 1:
        raise ValueError("Portable bootstrap lock timeout must be at least one second")
    if not portable.runtime_directory.strip() or Path(portable.runtime_directory).is_absolute():
        raise ValueError("Portable runtime directory must be a non-empty relative path")
    if set(bootstrap.archives) != set(bootstrap.checksums):
        raise ValueError("Every bootstrap archive must have exactly one SHA-256 checksum")
    if bootstrap.manager != "uv":
        raise ValueError(f"Unsupported bootstrap manager: {bootstrap.manager}")
    if bootstrap.connect_timeout_seconds < 1 or bootstrap.download_timeout_seconds < 1:
        raise ValueError("Bootstrap download timeouts must be at least one second")
    if bootstrap.download_retries < 1:
        raise ValueError("Bootstrap download retries must be at least one")
    if not re.fullmatch(r"\d+\.\d+\.\d+", bootstrap.manager_version):
        raise ValueError("Bootstrap manager version must be an exact X.Y.Z version")
    if not re.fullmatch(r"\d+\.\d+\.\d+", runtime.managed_python):
        raise ValueError("Managed Python must be an exact X.Y.Z version")
    invalid_checksums = {
        target for target, checksum in bootstrap.checksums.items()
        if not re.fullmatch(r"[0-9a-f]{64}", checksum)
    }
    if invalid_checksums:
        raise ValueError(
            f"Bootstrap targets have invalid SHA-256 checksums: {sorted(invalid_checksums)}"
        )
    invalid_archives = {
        target for target, archive in bootstrap.archives.items()
        if not archive.endswith(".zip" if target.startswith("windows-") else ".tar.gz")
    }
    if invalid_archives:
        raise ValueError(
            f"Bootstrap targets have invalid archive formats: {sorted(invalid_archives)}"
        )
    missing_bootstrap_targets = set(portable.targets) - set(bootstrap.archives)
    if missing_bootstrap_targets:
        raise ValueError(
            f"Portable targets lack bootstrap support: {sorted(missing_bootstrap_targets)}"
        )
    package_data = {package: _items(paths) for package, paths in parser["package_data"].items()}
    ai_requirements = dict(parser["ai.requirements"])
    return ProjectPolicy(
        application=application,
        runtime=runtime,
        bootstrap=bootstrap,
        portable=portable,
        package_data=package_data,
        ai_requirements=ai_requirements,
    )


def application_version(policy: ProjectPolicy, root: Path = ROOT) -> str:
    """Read one module-level literal version without executing application code."""
    version_file = root / policy.application.version_file
    source = version_file.read_text(encoding="utf-8")
    try:
        module = ast.parse(source, filename=str(version_file))
    except SyntaxError as error:
        raise RuntimeError(f"Could not parse configured version file {version_file}: {error}") from error

    assignments: list[ast.expr] = []
    for statement in module.body:
        if isinstance(statement, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "__sbk_version__" for target in statement.targets):
                assignments.append(statement.value)
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "__sbk_version__"
        ):
            assignments.append(statement.value)

    if len(assignments) != 1:
        raise RuntimeError(
            f"Expected exactly one module-level __sbk_version__ assignment in {version_file}; "
            f"found {len(assignments)}"
        )
    value = assignments[0]
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        raise RuntimeError(f"__sbk_version__ must be assigned a string literal in {version_file}")
    return value.value


def load_requirements(path: Path) -> list[str]:
    """Read requirement entries while preserving URL fragment identifiers."""
    requirements: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
        if line:
            requirements.append(line)
    return requirements


def environment_matches_policy(
    policy: ProjectPolicy,
    root: Path = ROOT,
    required_backend: str = "",
) -> bool:
    """Return whether the installed application matches the checked-out source."""
    try:
        installed_version = distribution_version(policy.application.distribution_name)
    except PackageNotFoundError:
        return False
    if installed_version != application_version(policy, root):
        return False

    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    try:
        importlib.import_module(policy.application.module)
        if required_backend and required_backend != "noai":
            registry = importlib.import_module("src.ai.registry")
            registry.load_backend_class(required_backend)
    except Exception:
        return False
    return True


def runtime_details(
    policy: ProjectPolicy,
    environment_kind: str,
    environment_prefix: str,
    environment_profile: str | None = None,
    selection_source: str = "unknown",
    saved_environment_reused: bool = False,
    environment_created: bool = False,
) -> tuple[str, ...]:
    """Return standardized launcher details for the selected runtime."""
    if environment_kind not in ENVIRONMENT_KINDS:
        raise ValueError(f"Unsupported environment kind: {environment_kind}")
    if selection_source not in SELECTION_SOURCES:
        raise ValueError(f"Unsupported environment selection source: {selection_source}")
    if environment_profile is None:
        environment_profile = policy.runtime.default_profile
    label = policy.application.name
    environment_label = {
        "managed": "managed venv",
        "venv": "venv",
        "conda": "conda",
    }.get(environment_kind, environment_kind)
    return (
        f"{label}: Operating system: {platform.platform(aliased=True)}",
        f"{label}: Python: {platform.python_version()} ({sys.executable})",
        f"{label}: Environment: {environment_label} ({environment_prefix})",
        f"{label}: Dependency profile: {environment_profile}",
        f"{label}: Selection source: {selection_source}",
        f"{label}: Saved environment reused: {'yes' if saved_environment_reused else 'no'}",
        f"{label}: Environment created this run: {'yes' if environment_created else 'no'}",
    )


def remember_environment(
    environment_kind: str,
    environment_prefix: str,
    state_file: Path,
    fingerprint: str = "",
    profile: str | None = None,
    state_schema: int | None = None,
) -> None:
    """Atomically persist the last validated launcher environment."""
    if environment_kind not in ENVIRONMENT_KINDS:
        raise ValueError(f"Unsupported environment kind: {environment_kind}")
    if not environment_prefix:
        raise ValueError("Environment prefix must not be empty")
    if state_schema is None:
        policy = load_policy()
        state_schema = policy.runtime.runtime_state_schema
    else:
        policy = None
    if profile is None:
        profile = (policy or load_policy()).runtime.default_profile

    temporary = state_file.with_name(f".{state_file.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            f"schema={state_schema}\n"
            f"kind={environment_kind}\n"
            f"prefix={environment_prefix}\n"
            f"fingerprint={fingerprint}\n"
            f"profile={profile}\n",
            encoding="utf-8",
        )
        temporary.replace(state_file)
    finally:
        temporary.unlink(missing_ok=True)


def load_remembered_environment(
    state_file: Path,
    supported_schema: int | None = None,
) -> tuple[str, str] | None:
    """Load a valid remembered launcher environment, if one exists."""
    if supported_schema is None:
        supported_schema = load_policy().runtime.runtime_state_schema
    try:
        values = dict(
            line.split("=", 1)
            for line in state_file.read_text(encoding="utf-8").splitlines()
            if "=" in line
        )
    except OSError:
        return None
    environment_kind = values.get("kind", "")
    environment_prefix = values.get("prefix", "")
    state_schema = values.get("schema", str(supported_schema))
    if state_schema != str(supported_schema):
        return None
    if environment_kind not in ENVIRONMENT_KINDS or not environment_prefix:
        return None
    return environment_kind, environment_prefix


def github_matrix(policy: ProjectPolicy) -> dict[str, list[dict[str, str]]]:
    """Return the native portable build matrix declared by project policy."""
    return {
        "include": [
            {
                "target": target,
                "runner": policy.portable.runners[target],
                "python": policy.portable.build_python,
            }
            for target in policy.portable.targets
        ]
    }


def main() -> int:
    """Expose policy-derived values needed by automation."""
    parser = argparse.ArgumentParser(description=__doc__)
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument(
        "--environment-ready",
        nargs="?",
        const="",
        metavar="BACKEND",
    )
    outputs.add_argument("--github-matrix", action="store_true")
    outputs.add_argument("--minimum-python", action="store_true")
    outputs.add_argument(
        "--runtime-details",
        nargs="+",
        metavar="VALUE",
    )
    outputs.add_argument(
        "--remember-environment",
        nargs="+",
        metavar="VALUE",
    )
    selected = parser.parse_args()
    if selected.environment_ready is not None:
        return 0 if environment_matches_policy(
            load_policy(), required_backend=selected.environment_ready
        ) else 1
    if selected.github_matrix:
        print(json.dumps(github_matrix(load_policy()), separators=(",", ":")))
        return 0
    if selected.minimum_python:
        print(load_policy().runtime.minimum_python)
        return 0
    if selected.runtime_details:
        if len(selected.runtime_details) not in {2, 6}:
            parser.error("--runtime-details expects 2 or 6 values")
        if len(selected.runtime_details) == 2:
            runtime_arguments: list[object] = [*selected.runtime_details]
        else:
            environment_kind, environment_prefix, profile, source, saved, created = (
                selected.runtime_details
            )
            if saved not in {"yes", "no"} or created not in {"yes", "no"}:
                parser.error("runtime saved/created values must be yes or no")
            runtime_arguments = [
                environment_kind,
                environment_prefix,
                profile,
                source,
                saved == "yes",
                created == "yes",
            ]
        print("\n".join(runtime_details(load_policy(), *runtime_arguments)))
        return 0
    if selected.remember_environment:
        if len(selected.remember_environment) not in {3, 4, 5}:
            parser.error("--remember-environment expects 3, 4, or 5 values")
        environment_kind, environment_prefix, state_file = selected.remember_environment[:3]
        optional_values = selected.remember_environment[3:]
        policy = load_policy()
        if len(optional_values) == 2:
            fingerprint, profile = optional_values
        elif len(optional_values) == 1:
            fingerprint, profile = "", optional_values[0]
        else:
            fingerprint, profile = "", policy.runtime.default_profile
        remember_environment(
            environment_kind,
            environment_prefix,
            Path(state_file),
            fingerprint,
            profile,
            policy.runtime.runtime_state_schema,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
