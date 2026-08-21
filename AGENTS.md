<!--
Copyright (c) KMG. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
-->

# Software-agent and contributor guide

This is the repository entry point for Codex, Devin, Cursor, Windsurf, Copilot, Aider, and human contributors. It states what the project does, where behavior lives, which rules must remain true, and how to prove a change works.

Read this file before editing. Then read the task-specific source and the matching recipe in [docs/AGENT_RECIPES.md](docs/AGENT_RECIPES.md). The [architecture guide](docs/ARCHITECTURE.md) explains the complete data flow.

## 1. Project in one minute

sbk-charts is a Python 3.10+ command-line application. It reads one or more SBK benchmark CSV files and creates an `.xlsx` report containing:

- an `SBK` cover sheet, plus one `R<n>` interval-data sheet and one `T<n>` total-data sheet per input;
- a Summary sheet and a Durations sheet;
- throughput, latency, percentile, percentile-count, data-volume, and timeout charts;
- optional AI-written analyses and optional interactive chat.

The main pipeline is fixed:

```text
CSV files -> R/T sheets -> Summary and charts -> optional AI text -> optional chat
```

On supported targets, source launchers can install an exact managed Python and locked dependency profile without preinstalled Python, venv, or Conda. They also reuse valid virtual and Conda environments. Portable releases bundle Python and dependencies.

## 2. Read the right document

| Need | Read |
|---|---|
| User commands and examples | [README.md](README.md) |
| First development change | [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) |
| Full module and runtime design | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Step-by-step change procedures | [docs/AGENT_RECIPES.md](docs/AGENT_RECIPES.md) |
| New AI plugin contract | [docs/PLUGIN_SPECIFICATION.md](docs/PLUGIN_SPECIFICATION.md) |
| Shared configuration ownership | [docs/POLICY.md](docs/POLICY.md) |
| Standalone archives | [docs/PORTABLE.md](docs/PORTABLE.md) |
| Backend setup and flags | [src/custom_ai/README.md](src/custom_ai/README.md) |

## 3. Code map

| Path | Owns | Change it when |
|---|---|---|
| `src/main/` | CLI orchestration | Changing top-level stage order or startup output |
| `src/parser/` | Base `-i` and `-o` flags | Changing non-AI input/output syntax |
| `src/sheets/` | CSV-to-R/T split and initial workbook | Changing data-sheet creation |
| `src/charts/` | Summary, charts, themes, CSV header constants | Adding or changing workbook visuals |
| `src/stat/` | Frozen `StorageStat` | Changing the AI-facing statistics shape |
| `src/genai/` | Shared AI interface and prompts | Changing every backend's analysis contract or prompt |
| `src/ai/` | Backend defaults, lazy registry, scheduling, Excel AI text, chat | Changing AI defaults, execution, or layout |
| `src/rag/` | Retrieval and grounding | Changing chat context selection |
| `src/custom_ai/<name>/` | One AI adapter | Adding or fixing one backend |
| `src/version/` | Canonical version | Cutting an approved release |
| `scripts/` | Policy, portable builder, frozen entry, and native Windows extractor | Changing release/runtime tooling |
| `sbk-charts.ini` | Shared runtime/artifact metadata | Changing a value consumed by multiple delivery systems |
| `.github/workflows/` | CI and release automation | Changing verification or native builds |
| `tests/` | Policy/portable unit tests | Changing launchers, policy, packaging, or archives |

## 4. First actions for any task

1. Run `git status --short --branch`. Preserve unrelated user changes.
2. Identify the owning module from the table above.
3. Read that module, its direct callers, its constants, and the matching architecture section.
4. Search before assuming. Prefer `rg` and `rg --files`.
5. Choose the verification commands before editing.
6. Make the smallest coherent change and update its documentation.
7. Run targeted checks, then the end-to-end sample.
8. Inspect the generated workbook when chart or Summary behavior changed.
9. Review `git diff --check`, `git status`, and the staged file list before committing.

Do not rely on old terminal examples in issues or reviews when the current parser and source can answer the question.

## 5. Development setup

The simplest source setup is:

```bash
python3 -m venv venv-sbk-charts
source venv-sbk-charts/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r requirements-dev.txt
```

You can also let `./sbk-charts` bootstrap and validate an application environment. Run tests with the explicit development interpreter you installed above:

```bash
venv-sbk-charts/bin/python -m unittest discover -s tests -v
```

Put core runtime dependencies in `requirements.txt`, managed build backends in `requirements-bootstrap.txt`, backend-only dependencies in `requirements-ai/<backend>.txt`, contributor tools in `requirements-dev.txt`, and portable build tools in `requirements-portable.txt`. Regenerate affected exact hashed files in `requirements-lock/` whenever a runtime input changes.

## 6. Definition of done

Every code change must pass checks proportional to its risk. The minimum application smoke test is:

```bash
./sbk-charts -i samples/charts/sbk-file-read.csv -o /tmp/sbk-charts-out.xlsx
```

Confirm the workbook can be read and contains Summary:

```bash
venv-sbk-charts/bin/python -c \
  "import openpyxl; w=openpyxl.load_workbook('/tmp/sbk-charts-out.xlsx'); print(len(w.sheetnames), 'Summary' in w.sheetnames)"
```

Run the available unit tests:

```bash
venv-sbk-charts/bin/python -m unittest discover -s tests -v
```

Run syntax and undefined-name checks on changed Python files, or use the CI scope:

```bash
venv-sbk-charts/bin/python -m flake8 . \
  --count --select=E9,F63,F7,F82 --show-source --statistics
```

Always finish with:

```bash
git diff --check
git status --short
```

Additional requirements by change type:

| Change | Additional proof |
|---|---|
| Chart or table | Open the workbook in Excel, LibreOffice, or equivalent and inspect size, labels, colors, series, and order. |
| AI plugin | Run that backend end to end, or prove missing credentials/service fail clearly and still save the workbook. |
| Shared prompt | Exercise representative cloud and local backends when credentials/services are available. |
| Launcher | Run `-h`; test reuse and creation paths relevant to the OS; run `bash -n sbk-charts` for Bash changes. |
| Windows launcher | Run PowerShell and batch smoke tests on Windows; static inspection on Linux is not sufficient release proof. |
| Policy or portable build | Run `tests/test_portable.py`, policy CLI outputs, wheel, sdist, and a native self-extracting build when feasible. Execute the artifact twice and create a workbook through it. |
| Packaging | Build wheel and sdist and inspect required assets. |
| Mermaid diagram | Render with Mermaid CLI v11+ when available. |
| Documentation only | Check links, code syntax, Mermaid syntax, spelling, current names, and current flags. |

## 7. Core invariants

### Workbook stages

Do not reorder the three creation stages without updating every consumer:

1. `SbkMultiSheets` creates R/T data sheets with XlsxWriter.
2. `SbkMultiCharts` reopens the file with openpyxl and creates Summary and charts.
3. `SbkAI` optionally appends analysis to the existing Summary sheet.

### R/T addressing

Each input CSV must map to one `R<n>` and one `T<n>` sheet. Chart and AI code use `is_r_num_sheet()` and `is_t_num_sheet()` to find data. Renaming these sheets is a cross-project compatibility change.

### Summary has two writers

`src/charts/multicharts.py` owns the main Summary layout. `src/ai/sbk_ai.py` owns the AI block in columns G and H. Check both when changing rows, columns, widths, or anchors.

### One latency unit per comparison

Comparison charts require all R sheets to use the same latency unit. Do not remove this check without adding correct unit conversion.

### Shared prompts belong in the framework

The four standard prompts live in `src/genai/genai.py`. A change there affects every backend. Provider-specific request formatting belongs in the provider plugin.

### Statistics are constructed once

`StorageStat` is frozen. Build its regular and total mappings before construction and treat them as read-only after construction.

### Fail one selected backend, not the application

The lightweight registry must not import provider SDKs. A missing optional dependency should fail with a useful message only after its backend is selected; it must not break help or core chart generation.

### Zero-only RAG metrics are skipped

The simple retrieval layer intentionally ignores all-zero metrics. This avoids irrelevant read or write fields in single-direction workloads.

## 8. Coding conventions

- Support Python 3.10 and newer.
- Use standard-library imports first, third-party imports second, and `src.*` imports last.
- Do not add wildcard imports.
- Add type hints when modifying public functions and methods.
- Write short docstrings for public behavior.
- Use logging for new diagnostic output. User-facing CLI progress may follow the surrounding output style.
- Do not add emojis to code, comments, or documentation.
- Use exact SBK CSV header constants from `src/charts/constants.py`; never repeat strings such as `MB/Sec` inline.
- Use sheet constants from `src/sheets/constants.py`.
- Keep plugin directory and module names lower snake case.
- The plugin class uses PascalCase. The explicit key in `src/ai/registry.py` defines its subcommand.
- Prefix plugin flags with the backend name when possible.
- Pin normal dependencies with compatible-release constraints (`~=`) unless a package has a documented reason for another form.
- Preserve Apache 2.0 headers. Do not modify `LICENSE` without explicit approval.

## 9. Plugin rules

A backend lives at `src/custom_ai/<directory>/<directory>.py` and subclasses `SbkGenAI`. Register it with a lightweight descriptor in `src/ai/registry.py`; do not edit `src/parser/sbk_parser.py` for plugin-only flags. The descriptor defines the command, implementation module and class, and plugin-specific flags without importing the implementation or its optional SDK.

Defaults used by both a registry flag and its adapter belong in `src/ai/defaults.py`. Do not add a second `add_args()` implementation to the provider class; the registry is the only runtime argument-registration path.

Every production backend should:

- declare its flags in the registry descriptor and consume them in `parse_args()`;
- return `(True, text)` or `(False, readable_error)`;
- implement all four canonical analyses;
- implement chat response behavior if chat is supported;
- reuse prompt builders from `SbkGenAI`;
- document authentication, model, threading, and resource needs;
- clean up sessions or model resources in `close()` when necessary.

If a selected backend fails to import, run:

```bash
venv-sbk-charts/bin/python -c \
  "from src.ai.registry import load_backend_class; print(load_backend_class('<backend>'))"
```

Help is registry-driven and should not import optional SDKs. Import the selected module directly for a traceback.

## 10. Chart rules

- Decide whether the source is an R sheet, a T sheet, or both.
- Reuse the series builders and chart factories in `SbkCharts`.
- Add any new CSV header to `src/charts/constants.py` first.
- Keep existing sheet names stable; users may have automation or saved formulas that depend on them.
- Put broad comparison sheets before detailed sheets.
- Use the shared theme functions so dimensions, fonts, backgrounds, and line colors remain consistent.
- Test one input and multiple inputs.
- Inspect the rendered workbook; a valid openpyxl file is not proof that a chart is readable.

## 11. Bootstrap, policy, and release rules

`sbk-charts.ini` owns shared runtime and artifact values. `scripts/project_policy.py` provides typed Python access and validation. Bash and PowerShell have small native INI readers because bootstrap cannot assume Python exists.

When changing policy:

- update all consumers or keep the existing key contract;
- extend tests for validation and generated CI data;
- update [docs/POLICY.md](docs/POLICY.md);
- smoke-test Linux/macOS Bash and Windows PowerShell/batch paths as applicable.

Launcher changes must preserve temporary-path cleanup, common selection order, provenance reporting, and fallback behavior. A system Python candidate is usable only when it can create a temporary venv with working `ensurepip` and `pip`. An explicit `SBK_CHARTS_VENV` disables creation of a new managed environment. Windows PowerShell 5.1 can drop empty native-command arguments, so legacy state persistence uses the policy helper's profile-without-fingerprint form. State without a schema remains backward compatible; unknown schemas must not be trusted.

The version is declared once in `src/version/sbk_version.py`. Do not copy it into examples that will become stale; use `<version>` in release procedures.

## 12. Mermaid rules

Use simple Mermaid syntax that renders consistently:

- use ASCII text in node and sequence labels;
- quote labels when punctuation could be parsed as syntax;
- use `<br/>` for a line break;
- do not use `++` in sequence messages;
- avoid Unicode arrows and em dashes in sequence messages;
- avoid unquoted participant aliases containing parentheses;
- render diagrams with `mmdc` v11+ when it is installed.

## 13. Repository safety

Preserve unrelated work in a dirty tree. Stage explicit files rather than relying on `git add -A`.

Never commit:

- `out.xlsx` or other generated reports;
- `venv-sbk-charts/`, `.venv/`, or Conda environments;
- `.sbk-runtime/` managed Python, tools, environments, and caches;
- `dist/`, `build/`, wheels, source archives, or portable applications;
- downloaded model files or caches;
- `.sbk-charts-runtime`;
- API keys, tokens, credentials, or secret-bearing logs.

The following actions need the user's explicit approval for the specific action:

- pushing commits, tags, artifacts, or releases;
- changing the version string;
- modifying license text;
- adding a new top-level Python package beside existing `src/*` packages;
- making a major-version upgrade to pandas, openpyxl, torch, google-genai, or another key dependency;
- force-pushing, rewriting history, or deleting branches.

Do not claim Windows, macOS, GPU, provider API, or portable-runtime testing if it was not actually performed. State the exact limitation.

## 14. Task routing for software agents

| User request | Primary files | Required reading |
|---|---|---|
| Improve chart appearance | `src/charts/charts.py`, `multicharts.py` | Architecture sections 7 and 8; chart recipe |
| Add or reorder charts | `src/charts/multicharts.py` | Workbook order and chart recipe |
| Fix CSV conversion | `src/sheets/` | Architecture section 6 |
| Add an AI backend | `src/custom_ai/` | Plugin specification and plugin recipe |
| Fix backend imports | plugin, registry, optional requirements and lock | AI section and troubleshooting recipe |
| Change AI text layout | `src/ai/sbk_ai.py` | Summary two-writer rule |
| Change prompt content | `src/genai/genai.py` | Canonical-prompt rule |
| Improve chat grounding | `src/rag/sbk_rag.py` | Architecture section 12 |
| Fix self-bootstrap | root launchers, policy helper | Policy and bootstrap recipes |
| Change portable applications | `scripts/build_portable.py`, policy, workflow | Portable guide and tests |
| Cut a release | version, packaging, workflows | Release recipe; explicit approval required |
| Update docs | relevant Markdown plus code source | Verify every command and link against current code |
| Understand or review the repository | `README.md`, `docs/DEVELOPMENT.md`, owning modules | Architecture overview and understand-project skill |

## 15. Handoff format

At completion, report:

1. the user-visible outcome;
2. important implementation choices;
3. exact verification commands and results;
4. anything not tested and why;
5. changed-file links;
6. commit, branch, push, or PR details only when those actions were requested and completed.

Do not hide failed checks. Explain whether they indicate a code problem, environment limitation, unavailable credential, unsupported platform, or unrelated pre-existing issue.
