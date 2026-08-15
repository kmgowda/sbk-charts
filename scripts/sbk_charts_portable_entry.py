#!/usr/bin/env python3
"""Entry point for a self-contained sbk-charts executable."""

from __future__ import annotations

import sys


def main(arguments: list[str] | None = None) -> int:
    """Run the normal sbk-charts CLI while preserving all supplied arguments."""
    selected = list(sys.argv[1:] if arguments is None else arguments)
    sys.argv = ["sbk-charts", *selected]

    from src.main.sbk_charts import sbk_charts

    sbk_charts()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
