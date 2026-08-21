<!--
Copyright (c) KMG. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
-->

# Add an sbk-charts chart

Use this skill for a new metric visualization, comparison sheet, or chart family.

## Read first

- `AGENTS.md`
- `docs/ARCHITECTURE.md`, chart and workbook-order sections
- `docs/AGENT_RECIPES.md`, recipes 3 through 5
- `src/charts/charts.py`, `multicharts.py`, `constants.py`, and `utils.py`

## Procedure

1. Decide whether data comes from R interval sheets or T total sheets.
2. Add a new exact CSV header to `src/charts/constants.py` if needed.
3. Find the closest current chart and reuse its series builder and chart factory.
4. Put reusable or per-run behavior in `SbkCharts` and cross-run orchestration in `SbkMultiCharts`.
5. Create a stable sheet name. Never rename an existing sheet without an explicit compatibility decision.
6. Register the method in the intended position of `SbkMultiCharts.create_graphs()`.
7. Use shared table and chart themes.

## Verification

```bash
./sbk-charts -i samples/charts/sbk-file-read.csv -o /tmp/chart-one.xlsx
./sbk-charts \
  -i samples/charts/sbk-file-read.csv,samples/charts/sbk-rocksdb-read.csv \
  -o /tmp/chart-two.xlsx
```

Load each workbook with openpyxl to verify its structure and expected sheet
order. Because openpyxl does not render charts, also open each workbook in
Excel, LibreOffice, or an equivalent spreadsheet viewer. Visually check the
titles, axes, units, ranges, legends, series, colors, fonts, dimensions,
placement, and sheet order.
