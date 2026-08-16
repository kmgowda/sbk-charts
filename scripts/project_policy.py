#!/usr/bin/env python3
"""Read and validate shared sbk-charts runtime and artifact policy."""

from __future__ import annotations

import argparse
import ast
import configparser
import importlib
import json
import re
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_FILE = ROOT / "sbk-charts.ini"


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
    default_conda_environment: str
    virtual_environment_names: tuple[str, ...]
    unix_python_commands: tuple[str, ...]
    windows_python_launchers: tuple[str, ...]


@dataclass(frozen=True)
class PortableArtifactPolicy:
    """Naming, contents, hashing, and target rules for portable archives."""

    targets: tuple[str, ...]
    manifest_name: str
    checksum_suffix: str
    hash_algorithm: str
    build_python: str
    bundle_paths: tuple[str, ...]
    entry_script: str
    collect_submodules: tuple[str, ...]
    platforms: dict[str, str]
    architectures: dict[str, str]
    archive_formats: dict[str, str]
    runners: dict[str, str]


@dataclass(frozen=True)
class ProjectPolicy:
    """Complete deployment policy loaded from ``sbk-charts.ini``."""

    application: ApplicationMetadata
    runtime: RuntimePolicy
    portable: PortableArtifactPolicy
    package_data: dict[str, tuple[str, ...]]


def load_policy(path: Path = POLICY_FILE) -> ProjectPolicy:
    """Load the central policy file and fail clearly when it is incomplete."""
    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(path, encoding="utf-8"):
        raise FileNotFoundError(f"Project policy file not found: {path}")

    application = ApplicationMetadata(**dict(parser["application"]))
    runtime_section = parser["runtime"]
    runtime = RuntimePolicy(
        minimum_python=runtime_section["minimum_python"],
        default_conda_environment=runtime_section["default_conda_environment"],
        virtual_environment_names=_items(runtime_section["virtual_environment_names"]),
        unix_python_commands=_items(runtime_section["unix_python_commands"]),
        windows_python_launchers=_items(runtime_section["windows_python_launchers"]),
    )
    portable_section = parser["portable"]
    portable = PortableArtifactPolicy(
        targets=_items(portable_section["targets"]),
        manifest_name=portable_section["manifest_name"],
        checksum_suffix=portable_section["checksum_suffix"],
        hash_algorithm=portable_section["hash_algorithm"],
        build_python=portable_section["build_python"],
        bundle_paths=_items(portable_section["bundle_paths"]),
        entry_script=portable_section["entry_script"],
        collect_submodules=_items(portable_section["collect_submodules"]),
        platforms=dict(parser["portable.platforms"]),
        architectures=dict(parser["portable.architectures"]),
        archive_formats=dict(parser["portable.archive_formats"]),
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
    if portable.hash_algorithm != "sha256":
        raise ValueError(f"Unsupported portable hash algorithm: {portable.hash_algorithm}")
    if not runtime.virtual_environment_names:
        raise ValueError("At least one virtual environment name is required")
    package_data = {package: _items(paths) for package, paths in parser["package_data"].items()}
    return ProjectPolicy(
        application=application,
        runtime=runtime,
        portable=portable,
        package_data=package_data,
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


def environment_matches_policy(policy: ProjectPolicy, root: Path = ROOT) -> bool:
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
    except Exception:
        return False
    return True


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
    outputs.add_argument("--environment-ready", action="store_true")
    outputs.add_argument("--github-matrix", action="store_true")
    outputs.add_argument("--minimum-python", action="store_true")
    selected = parser.parse_args()
    if selected.environment_ready:
        return 0 if environment_matches_policy(load_policy()) else 1
    if selected.github_matrix:
        print(json.dumps(github_matrix(load_policy()), separators=(",", ":")))
        return 0
    if selected.minimum_python:
        print(load_policy().runtime.minimum_python)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
