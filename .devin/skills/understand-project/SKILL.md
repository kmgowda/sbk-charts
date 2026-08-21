<!--
Copyright (c) KMG. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
-->

# Understand or review sbk-charts

Use this skill for repository onboarding, architecture reviews, impact analysis, or explaining how a command becomes an Excel workbook.

## Read first

- `README.md` for user-visible behavior and current commands
- `docs/DEVELOPMENT.md` for the shortest contributor walkthrough
- `docs/ARCHITECTURE.md` for the complete data and runtime design
- `AGENTS.md` for invariants, safety rules, and verification requirements

## Procedure

1. Run `git status --short --branch` and preserve unrelated work.
2. Identify whether the question concerns delivery, CLI, sheets, charts, AI, RAG, policy, or packaging.
3. Read the owning source file, its direct caller, and its constants. Do not infer behavior from filenames alone.
4. Trace one concrete sample through the layer. For workbook behavior, use `samples/charts/sbk-file-read.csv`.
5. Search for all consumers before calling a value duplicated or hardcoded.
6. Separate architectural invariants from implementation details that can safely change.
7. Report evidence, risks, and untested assumptions. Do not edit during a read-only review.

Useful searches:

```bash
rg -n "create_sheets|create_graphs|add_performance_details" src
rg -n "R_PREFIX|T_PREFIX|LATENCY_TIME_UNIT" src
rg -n "BACKENDS|load_backend_class|SbkGenAI" src
rg -n "policy_value|load_policy|bundle_paths" sbk-charts sbk-charts.ps1 scripts sbk-charts.ini
```

## Expected explanation

A good project explanation answers:

- what the user provides and receives;
- which layer owns each transformation;
- why R/T sheet names and workbook stage order are stable contracts;
- when optional AI code is imported;
- how the source launcher creates and remembers an environment;
- how source packages differ from portable archives;
- which tests prove structure and which checks require visual or native-platform inspection.

Use links to exact files and keep recommendations separate from confirmed current behavior.
