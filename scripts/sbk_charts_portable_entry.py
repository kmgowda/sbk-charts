#!/usr/bin/env python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""Entry point for a self-contained sbk-charts executable."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def main(arguments: list[str] | None = None) -> int:
    """Run the normal sbk-charts CLI while preserving all supplied arguments."""
    selected = list(sys.argv[1:] if arguments is None else arguments)
    application_name = Path(sys.executable).stem
    selection_source = os.environ.get(
        "SBK_CHARTS_PORTABLE_SELECTION_SOURCE", "extracted-portable"
    )
    reused = os.environ.get("SBK_CHARTS_PORTABLE_REUSED", "no")
    created = os.environ.get("SBK_CHARTS_PORTABLE_CREATED", "no")
    prefix = os.environ.get(
        "SBK_CHARTS_PORTABLE_PREFIX", str(Path(sys.executable).resolve().parent)
    )
    print(f"{application_name}: Operating system: {platform.platform(aliased=True)}")
    print(f"{application_name}: Python: {platform.python_version()} ({sys.executable})")
    print(f"{application_name}: Environment: portable ({prefix})")
    print(f"{application_name}: Dependency profile: all-ai")
    print(f"{application_name}: Selection source: {selection_source}")
    print(f"{application_name}: Saved environment reused: {reused}")
    print(f"{application_name}: Environment created this run: {created}")
    sys.argv = [application_name, *selected]

    from src.main.sbk_charts import sbk_charts

    sbk_charts()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
