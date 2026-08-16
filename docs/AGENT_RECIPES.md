<!--
Copyright (c) KMG. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
-->

# Contributor and software-agent recipes

These recipes turn common requests into repeatable edits and verification. Read [AGENTS.md](../AGENTS.md) first. Use [ARCHITECTURE.md](ARCHITECTURE.md) when a recipe mentions an invariant you do not yet understand.

## 1. Add an AI backend

### Choose the closest existing adapter

| New backend type | Start from |
|---|---|
| Cloud SDK with an API key | `src/custom_ai/gemini/` or `anthropic/` |
| Local HTTP service | `src/custom_ai/ollama/` |
| Local service with its own SDK | `src/custom_ai/lm_studio/` |
| In-process Transformers model | `src/custom_ai/pytorch_llm/` |

For a substantial plugin, fill in [PLUGIN_SPECIFICATION.md](PLUGIN_SPECIFICATION.md) before coding.

### Implement

Create:

```text
src/custom_ai/<plugin_name>/
    __init__.py
    <plugin_name>.py
    README.md
```

The module must define one concrete `SbkGenAI` subclass. Directory and module names use lower snake case, and the class uses PascalCase. The explicit registry key defines the command; it is not derived from the class name.

Register plugin flags in the lightweight descriptor and consume them in `parse_args()`. Do not import the optional SDK from the registry.

Implement these result methods using the `(bool, text)` contract:

```python
def get_model_description(self) -> tuple[bool, str]: ...
def get_throughput_analysis(self) -> tuple[bool, str]: ...
def get_latency_analysis(self) -> tuple[bool, str]: ...
def get_total_mb_analysis(self) -> tuple[bool, str]: ...
def get_percentile_histogram_analysis(self) -> tuple[bool, str]: ...
def get_response(self, query: str) -> tuple[bool, str]: ...
```

Reuse the base prompt builders. Read credentials from environment variables, never command history or committed files. Add the SDK to `requirements-ai/<command>.txt`, register it in `sbk-charts.ini`, and regenerate its exact hashed lock.

### Verify

```bash
./sbk-charts -h
./sbk-charts -i samples/charts/sbk-file-read.csv <plugin-command> -h
```

Test the missing-auth or unavailable-service path. It should produce clear failures and still save a workbook:

```bash
./sbk-charts -i samples/charts/sbk-file-read.csv \
  -o /tmp/plugin-missing-service.xlsx <plugin-command>
```

Then test the configured backend end to end. Confirm all four analyses finish or fail individually with useful text. If chat is supported, run with `-chat` and ask one question whose answer must mention a value or storage name from the sample.

## 2. Change an existing AI backend

| Change | Files |
|---|---|
| New backend flag | Registry descriptor, plugin `parse_args()`, and README |
| Default model or endpoint | Plugin constants and README |
| SDK update | Plugin imports/calls, optional requirements, hashed lock, README |
| Provider-specific response parsing | Plugin request helper |
| Shared analysis wording | `src/genai/genai.py`, not the plugin |

Run backend help and import only the affected backend so an SDK regression is visible:

```bash
venv-sbk-charts/bin/python -c \
  "from src.ai.registry import load_backend_class; print(load_backend_class('<backend>'))"
```

## 3. Add a chart

### Decide the data level

- Use R sheets for interval values and time-series comparisons.
- Use T sheets for totals and run-level summaries.
- Use `SbkCharts` for reusable series and per-run behavior.
- Use `SbkMultiCharts` for cross-run sheets and workbook ordering.

If the metric is new, add its exact CSV header once to `src/charts/constants.py`.

### Implement

1. Find an existing chart with the same shape.
2. Reuse `create_line_chart()`, `create_bar_chart()`, and existing series helpers.
3. Create a stable worksheet name that follows current naming conventions.
4. Add the method call to the intended position in `SbkMultiCharts.create_graphs()`.
5. Let `apply_table_theme()` and `apply_chart_theme()` style it.

Do not rename existing chart sheets. Do not hard-code column indexes when header lookup can find them.

### Verify one and multiple runs

```bash
./sbk-charts -i samples/charts/sbk-file-read.csv \
  -o /tmp/chart-single.xlsx

./sbk-charts \
  -i samples/charts/sbk-file-read.csv,samples/charts/sbk-rocksdb-read.csv \
  -o /tmp/chart-compare.xlsx
```

List sheets programmatically, then inspect both files visually:

```bash
venv-sbk-charts/bin/python -c \
  "import openpyxl; print(openpyxl.load_workbook('/tmp/chart-compare.xlsx').sheetnames)"
```

Check chart size, title, axes, legend, series colors, adjacent-line contrast, fonts, data range, and placement.

## 4. Change workbook styling

Presentation settings are concentrated in `src/charts/charts.py`, with Summary-specific formatting in `src/charts/multicharts.py` and AI text formatting in `src/ai/sbk_ai.py`.

Before editing, identify every consumer of the value with `rg`. Change shared theme functions when the requirement applies to all sheets. Change one chart method only when the requirement is local.

Test at least:

- a dense line chart;
- a bar chart;
- a percentile chart;
- R and T tables;
- Summary without AI;
- Summary with AI if AI formatting changed.

Visual inspection is mandatory. Library load success does not prove the text fits, the chart is large enough, or neighboring colors are distinguishable.

## 5. Reorder workbook sheets

Sheet creation order is controlled near the bottom of `src/charts/multicharts.py` in `create_graphs()`.

1. Map the requested conceptual order to the exact `create_*` calls.
2. Preserve `create_summary_sheet()` and `create_sbk_date_sheet()` at the beginning.
3. Keep theme application and workbook save at the end.
4. Move call sites, not generated sheets after the fact.
5. Generate a workbook and compare the complete `sheetnames` list with the requested order.

Remember that some methods create several numbered sheets.

## 6. Change the Summary sheet

The Summary sheet is written twice:

- `SbkMultiCharts.create_summary_sheet()` writes metadata and benchmark timing.
- `SbkAI.add_ai_analysis()` appends AI content in columns G and H.

If you add ordinary rows, the AI block follows `max_row`. If you change columns, merged ranges, widths, or the meaning of columns G/H, update and test both writers.

Run once without a backend and once with an available backend. Inspect values, fonts, wrapping, row heights, and separation between the two sections.

## 7. Change a shared prompt

Edit one of these methods in `src/genai/genai.py`:

- `get_throughput_prompt()`;
- `get_latency_prompt()`;
- `get_total_mb_prompt()`;
- `get_percentile_histogram_prompt()`.

Keep metric data in the prompt and make unit instructions explicit. A prompt change affects all production backends, so test representative cloud and local implementations when available. If services or credentials are unavailable, test prompt construction directly and state the integration gap.

Do not copy a shared prompt into every plugin.

## 8. Debug a missing backend

### Load the selected backend

```bash
venv-sbk-charts/bin/python -c \
  "from src.ai.registry import load_backend_class; print(load_backend_class('<backend>'))"
```

### Import the missing module directly

```bash
venv-sbk-charts/bin/python -c \
  "import src.custom_ai.<directory>.<module>"
```

Common causes are a stale backend lock, a package installed into a different environment, an upstream import path change, or a syntax error. Fix the root cause and test the selected backend. Help is lazy and should list declared backends without their SDKs.

## 9. Debug AI timeouts or memory errors

1. Increase the total budget, for example `-secs 600`.
2. Add `-nothreads` before the backend command.
3. Check provider rate limits or local-server health.
4. For PyTorch, check RAM, accelerator memory, device choice, and model size.
5. Distinguish timeout text from provider error text saved in Summary.

Example:

```bash
./sbk-charts -i input.csv -o /tmp/slow-model.xlsx \
  -secs 600 -nothreads pytorchllm --pt-model <smaller-model>
```

The timeout is a total analysis budget, not a guaranteed hard termination of provider or native-library work already running.

## 10. Change RAG behavior

The default path is `SbkSimpleRAGPipeline` in `src/rag/sbk_rag.py`. The Chroma implementation is optional and is not selected by `SbkAI` today.

When changing retrieval:

1. test ingestion from multiple `StorageStat` values;
2. preserve intentional all-zero metric filtering unless the requirement changes it;
3. test direct metric questions and storage-comparison questions;
4. inspect retrieved context, not only the final model response;
5. ensure empty data and empty results fail safely.

Do not add ChromaDB as a required dependency merely because the alternative implementation exists.

## 11. Change source bootstrap behavior

Relevant files are `sbk-charts`, `sbk-charts.ps1`, `sbk-charts.bat`, `sbk-charts.ini`, and `scripts/project_policy.py`.

Preserve the main selection priorities. An explicit venv is tried first. Otherwise the launcher tries remembered, active, project-local, and named Conda environments before the fingerprinted managed environment. On a supported target it then verifies pinned `uv`, installs the exact managed Python, installs the selected hashed lock, self-checks, publishes atomically, and remembers it. Legacy venv/Conda creation remains a fallback for unsupported managed targets. Validate Unix `bin/python`, Windows `Scripts\python.exe`, fingerprints, profiles, locking, checksum rejection, first-run creation, and second-run offline reuse.

Required checks include:

```bash
bash -n sbk-charts
./sbk-charts -h
venv-sbk-charts/bin/python -m unittest discover -s tests -v
```

Test the changed selection branch, runtime detail output, state persistence, and argument forwarding. Windows changes require a real Windows PowerShell/batch smoke test before release approval.

## 12. Change centralized policy

Use `sbk-charts.ini` when a value is shared by launchers, setup, CI, or portable artifacts. Keep domain-only values in their modules.

After a policy edit, run:

```bash
venv-sbk-charts/bin/python scripts/project_policy.py --minimum-python
venv-sbk-charts/bin/python scripts/project_policy.py --github-matrix
venv-sbk-charts/bin/python -m unittest discover -s tests -v
```

Update [POLICY.md](POLICY.md), extend validation tests, and check every consumer of the changed key.

## 13. Build and verify packages

Build without deleting unrelated user artifacts:

```bash
venv-sbk-charts/bin/python -m build
```

Inspect the wheel and source archive with `unzip -l` and `tar -tzf`. Confirm at least:

- `src/main/banner.txt`;
- `src/images/sbk-logo.png`;
- `sbk-charts.ini`;
- source-distribution scripts and documentation expected by `MANIFEST.in`.

For high-confidence release verification, install the wheel into a fresh temporary environment and run it from outside the repository.

## 14. Build a portable archive

Install build-only requirements into a suitable native environment:

```bash
python -m pip install -r requirements-portable.txt
python scripts/build_portable.py
```

The builder identifies the current target, runs PyInstaller, smoke-tests the frozen executable with `--help`, copies declared documentation/metadata, generates `manifest.json`, creates the native archive, and writes an external `.sha256` file.

Inspect the archive root, executable, `_internal` directory, manifest paths, file hashes, and checksum. A build proves only its current operating system and architecture.

## 15. Update documentation

1. Read the code that owns the behavior.
2. Prefer short sentences and define project-specific terms.
3. Put beginner instructions before internal details.
4. Use real current commands; copy option names from argparse code or help output.
5. Use `<version>` in release examples unless documenting the current release explicitly.
6. Link to the owning guide instead of duplicating long explanations.
7. Use Mermaid only when it makes relationships easier to understand.
8. Render diagrams when `mmdc` is available.
9. Search for stale names, versions, paths, flags, and custom reference tags.
10. Run the documented commands that are safe and local.

Useful audit searches:

```bash
rg -n "3\\.26|sbk-analytics|-nothreads true|pytorch_llm|--model |--input|--output" \
  --glob '*.md'
rg -n "<ref_file|<ref_snippet|file:///" --glob '*.md'
```

## 16. Cut a release

Changing the version, creating tags, pushing, and publishing require explicit user approval.

1. Set the approved version once in `src/version/sbk_version.py`.
2. Run tests, lint, source CLI, workbook inspection, wheel, and sdist checks.
3. Build native portable artifacts through the release workflow or on each native target.
4. Confirm documentation uses the intended version where a current-version statement is necessary.
5. Commit and tag only after reviewing the exact diff and artifact list.
6. Push and publish only when specifically authorized.

Use placeholders in reusable instructions:

```bash
git tag <version>
git push origin <version>
```

## 17. Pull-request handoff

A useful PR description contains:

- the user problem;
- the implemented behavior;
- important design decisions and compatibility notes;
- exact commands and results;
- platform, provider, or visual checks not performed;
- screenshots for workbook appearance changes when practical;
- documentation and test updates;
- follow-up work that is intentionally out of scope.
