#!/usr/bin/env python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""Validate native workflow artifacts before uploading a GitHub release."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.create_github_release import verify_portable_asset_directory
from scripts.project_policy import application_version, load_policy


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse the portable artifact validation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--version", help="Expected version; defaults to project version")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Validate and list the complete portable release asset set."""
    selected = parse_args(arguments)
    policy = load_policy()
    version = selected.version or application_version(policy)
    assets = verify_portable_asset_directory(
        policy,
        version,
        selected.directory.resolve(),
    )
    print(f"Validated {len(assets)} portable release files for {version}:")
    for asset in assets:
        print(f"- {asset.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
