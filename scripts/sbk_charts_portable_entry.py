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

import sys
from pathlib import Path


def main(arguments: list[str] | None = None) -> int:
    """Run the normal sbk-charts CLI while preserving all supplied arguments."""
    selected = list(sys.argv[1:] if arguments is None else arguments)
    sys.argv = [Path(sys.executable).stem, *selected]

    from src.main.sbk_charts import sbk_charts

    sbk_charts()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
