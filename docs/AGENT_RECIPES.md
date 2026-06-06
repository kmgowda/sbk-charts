<!--
Copyright (c) KMG. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
-->

# AGENT_RECIPES.md — Step-by-step task playbooks for sbk-charts

> **Audience.** AI coding agents and human contributors who need
> concrete, copy-pasteable procedures for common tasks in this
> repository. Each recipe lists the **exact files to touch**, the
> **exact commands to run**, and **what success looks like**.
>
> Read <ref_file file="/root/projects/sbk-charts/AGENTS.md" /> first
> for repo-wide conventions and gotchas, and
> <ref_file file="/root/projects/sbk-charts/docs/ARCHITECTURE.md" />
> for the design background.

---

## Index

1. [Add a new AI plugin (backend)](#1-add-a-new-ai-plugin-backend)
2. [Modify an existing AI plugin (add a CLI flag, swap an SDK)](#2-modify-an-existing-ai-plugin)
3. [Add a new chart to the workbook](#3-add-a-new-chart-to-the-workbook)
4. [Add a new column / metric to `charts/constants.py`](#4-add-a-new-column--metric-to-chartsconstantspy)
5. [Change a prompt template (affects all plugins)](#5-change-a-prompt-template-affects-all-plugins)
6. [Tweak the Summary sheet](#6-tweak-the-summary-sheet)
7. [Debug a plugin that silently disappears from `-h`](#7-debug-a-plugin-that-silently-disappears-from--h)
8. [Debug "AI timed out" / partial Summary output](#8-debug-ai-timed-out--partial-summary-output)
9. [Bump a Python dependency](#9-bump-a-python-dependency)
10. [Cut a new release](#10-cut-a-new-release)

---

## 1. Add a new AI plugin (backend)

**Goal:** Make `sbk-charts -i run.csv <newplugin>` work, with the AI
calling the new model/service to generate the four canonical analyses.

### 1.1 Prerequisites

Before you start, confirm:

- You have a Python client/SDK for the target model (cloud API or
  local server).
- The library is on **PyPI** (or installable with pip).
- You have read the existing plugins under
  <ref_file file="/root/projects/sbk-charts/src/custom_ai/" />. The
  closest reference is usually:
  - **For a cloud API** → mirror `gemini/gemini.py` or
    `anthropic/anthropic.py`.
  - **For a local HTTP server** → mirror `ollama/ollama.py` or
    `lm_studio/lm_studio.py`.
  - **For an in-process model** → mirror `pytorch_llm/pytorch_llm.py`.

### 1.2 Files to create

Pick a plugin name. The convention (see <ref_file file="/root/projects/sbk-charts/AGENTS.md" />
§3) is:

- Directory name: lowercase, snake_case (`mistral` or `azure_openai`).
- Module filename: same as the directory (`mistral.py`).
- Class name: PascalCase (`Mistral`, `AzureOpenAI`).
- CLI subcommand: lowercased class name (`mistral`, `azureopenai`).

```
src/custom_ai/mistral/
├── __init__.py        # empty is fine
└── mistral.py         # contains `class Mistral(SbkGenAI):`
```

### 1.3 Step-by-step

**Step A — Create the directory + empty `__init__.py`.**

```bash
mkdir -p src/custom_ai/mistral
touch src/custom_ai/mistral/__init__.py
```

**Step B — Add the plugin dependency to `requirements.txt`.**

Pin with `~=` (compatible release):

```
mistralai~=1.5.0
```

Install it:

```bash
pip install -r requirements.txt
```

**Step C — Implement the plugin.** Copy the template below into
`src/custom_ai/mistral/mistral.py` and adapt the four `_call_*`
helpers to your SDK.

```python
#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Mistral AI Integration Module."""

import os
from typing import Tuple
from mistralai import Mistral as MistralClient   # or whatever your SDK exports

from src.genai.genai import SbkGenAI

DEFAULT_MODEL = "mistral-large-latest"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.4


class Mistral(SbkGenAI):
    """Mistral AI analysis backend."""

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.model = DEFAULT_MODEL
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.temperature = DEFAULT_TEMPERATURE
        self._client = None
        if self.api_key:
            try:
                self._client = MistralClient(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Failed to init Mistral client: {e}")

    def add_args(self, parser):
        parser.add_argument("--mistral-model",
                            default=DEFAULT_MODEL,
                            help=f"Mistral model (default: {DEFAULT_MODEL})")
        parser.add_argument("--mistral-max-tokens", type=int,
                            default=DEFAULT_MAX_TOKENS)
        parser.add_argument("--mistral-temperature", type=float,
                            default=DEFAULT_TEMPERATURE)

    def parse_args(self, args):
        self.model = args.mistral_model
        self.max_tokens = args.mistral_max_tokens
        self.temperature = args.mistral_temperature

    def get_model_description(self) -> Tuple[bool, str]:
        if not self.api_key:
            return False, ("Mistral analysis is not available "
                           "(missing MISTRAL_API_KEY).")
        return True, (f"Mistral API\n"
                      f" Model: {self.model}\n"
                      f" Temperature: {self.temperature}\n"
                      f" Max Tokens: {self.max_tokens}")

    def _call(self, prompt: str) -> Tuple[bool, str]:
        if not self.api_key:
            return False, "Mistral API key not configured."
        try:
            resp = self._client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return True, resp.choices[0].message.content.strip()
        except Exception as e:
            return False, f"Error calling Mistral API: {e}"

    def get_throughput_analysis(self) -> Tuple[bool, str]:
        return self._call(self.get_throughput_prompt())

    def get_latency_analysis(self) -> Tuple[bool, str]:
        return self._call(self.get_latency_prompt())

    def get_total_mb_analysis(self) -> Tuple[bool, str]:
        return self._call(self.get_total_mb_prompt())

    def get_percentile_histogram_analysis(self) -> Tuple[bool, str]:
        return self._call(self.get_percentile_histogram_prompt())

    def get_response(self, query) -> Tuple[bool, str]:
        prompt = (
            "You are a storage performance engineer. Analyze the following "
            f"query in the context of the loaded benchmark data:\nQuery: {query}\n"
            "Provide a focused technical answer."
        )
        return self._call(self._enhance_prompt_with_rag(prompt, query))

    def close(self, args):
        # SDK has no explicit close; provided for interface compatibility.
        pass
```

**Step D — Verify discovery.** No registration step is needed —
`SbkAI.__init__` will pick the plugin up automatically. Confirm:

```bash
./sbk-charts -h
```

You should see `mistral` listed in the subcommand list.

If it does not appear, run:

```bash
python3 -c "from src.ai.discover import discover_custom_ai_classes; print(discover_custom_ai_classes())"
```

— the first import error in the output is the cause (see Recipe §7).

**Step E — End-to-end smoke test.**

```bash
export MISTRAL_API_KEY=...
./sbk-charts -i samples/charts/sbk-file-read.csv -o /tmp/out.xlsx \
    mistral --mistral-model mistral-large-latest
```

Expected output ends with `File updated with graphs and AI documentation: /tmp/out.xlsx`
and all four analyses logged as `✓ Completed get_*_analysis`.

If the API key is missing, the plugin must **fail gracefully** —
each analysis returns `(False, "Mistral API key not configured.")`
and the Summary sheet still gets written, just with error messages
in the AI block.

**Step F — Update docs.** Add the new row to the plugin comparison
table in <ref_file file="/root/projects/sbk-charts/docs/ARCHITECTURE.md" />
§5.9 and the seven-plugin count throughout that file.

### 1.4 Acceptance checklist

- [ ] Plugin appears in `sbk-charts -h`.
- [ ] All four analyses succeed against a sample CSV with a valid API
  key.
- [ ] Without an API key, the four analyses each return a clear
  `(False, "...")` and the workbook still saves cleanly.
- [ ] `requirements.txt` lists the new dep, pinned with `~=`.
- [ ] No code change leaks into other plugins.
- [ ] Plugin name in the comparison table in ARCHITECTURE.md §5.9.
- [ ] `gemini/anthropic` etc. continue to work — discovery is silent
  on import errors but the existence of your plugin should not regress
  them.

---

## 2. Modify an existing AI plugin

**Goal:** Add a new CLI flag, swap an SDK version, or fix a bug in one
plugin.

### 2.1 Files to touch

| Change | Touch |
|---|---|
| New CLI flag | The plugin's `add_args()` and `parse_args()` |
| New default value | The `DEFAULT_*` constants at the top of the plugin |
| SDK swap | The plugin's imports + the request/response handling, and `requirements.txt` |
| Bug in a specific analysis | The relevant `get_*_analysis()` method or its helper |

**Do not** modify prompts in the plugin. Prompts live in
<ref_file file="/root/projects/sbk-charts/src/genai/genai.py" /> — see
Recipe §5.

### 2.2 Verification

```bash
./sbk-charts -i samples/charts/sbk-file-read.csv -o /tmp/out.xlsx <plugin> [...new flags...]
```

Open the produced workbook, scroll to the Summary sheet column H, and
confirm the analyses reflect the change.

### 2.3 Worked example: the Gemini SDK migration

The recent change in
<ref_file file="/root/projects/sbk-charts/src/custom_ai/gemini/gemini.py" />
swapped `google.ai.generativelanguage` → `google.genai` (the modern
`google-genai` 1.62+ Client API). The pattern was:

1. Update the import (`import google.genai as genai`).
2. Replace `GenerativeServiceClient(...)` constructions with
   `genai.Client(api_key=...)`.
3. Replace request/response shaping with the new
   `client.models.generate_content(model=..., contents=..., config=GenerateContentConfig(...))`
   form.
4. `requirements.txt` already had `google-genai~=1.62.0`.

End-to-end verification was a single CLI run with `GEMINI_API_KEY` set.

---

## 3. Add a new chart to the workbook

**Goal:** Produce a new chart sheet in `out.xlsx`. For example, a "Total
Records / Second by Storage" comparison.

### 3.1 Files to touch

- <ref_file file="/root/projects/sbk-charts/src/charts/multicharts.py" /> —
  add the chart method on `SbkMultiCharts`.
- (Optional) <ref_file file="/root/projects/sbk-charts/src/charts/charts.py" /> —
  if there is a single-run analog, add it on the base class.
- <ref_file file="/root/projects/sbk-charts/src/charts/multicharts.py" />
  `create_graphs()` at the bottom — register your new method in the
  sequence.

### 3.2 Step-by-step

**Step A — Decide R-sheet or T-sheet.**

- *R-sheet (per-interval)* chart → iterate `if is_r_num_sheet(name)`.
- *T-sheet (totals)* chart → iterate `if is_t_num_sheet(name)`.

**Step B — Pick a series builder from `SbkCharts`.** Common ones:

| Builder | Returns | Use for |
|---|---|---|
| `get_throughput_mb_series(ws, ws_name)` | single MB/Sec series | line/bar of MB/Sec |
| `get_throughput_write_request_mb_series` / `get_throughput_read_request_mb_series` | request-side MB/Sec | write/read split |
| `get_latency_series(ws, ws_name)` | dict of latency-column series | latency line charts |
| `get_latency_percentile_series(ws, ws_name, names_list)` | row-wise percentile series | percentile line charts (use with `slc_percentile_names`) |
| `get_latency_percentile_count_series(ws, ws_name, names_list)` | row-wise percentile-count series | histogram |
| `get_avg_latency_series` / `get_min_latency_series` / `get_max_latency_series` | single cell-series | bar charts of totals |
| `get_write_timeout_events_series` / `get_read_timeout_events_series` | timeout event series | bar charts of timeouts |

If the chart you want isn't covered, add a new
`__get_column_series(ws, ws_name, column_name)` wrapper on `SbkCharts`
and reference it from your new method.

**Step C — Add the method.** Template for a *bar chart comparing a
single T-column across all T-sheets*:

```python
def create_total_records_per_sec_compare_graph(self):
    """Create a bar chart comparing total Records/Sec across T-sheets."""
    chart = None
    for name in self.wb.sheetnames:
        if is_t_num_sheet(name):
            ws = self.wb[name]
            if chart is None:
                action = get_action_name_from_worksheet(ws)
                chart = self.create_bar_chart(
                    "Total Records / Second Comparison",
                    action, "Records / Second", 25, 50)
            prefix = name + "-" + get_storage_name_from_worksheet(ws)
            chart.append(self.__get_column_series(ws, prefix, constants.RECORDS_PER_SEC))
    if chart is not None:
        sheet = self.wb.create_sheet("Total_Records_PerSec")
        sheet.add_chart(chart)
        return sheet
    return None
```

**Step D — Register in `create_graphs()`.** At the bottom of
<ref_file file="/root/projects/sbk-charts/src/charts/multicharts.py" />:

```python
def create_graphs(self):
    if self.check_time_units():
        self.create_summary_sheet()
        # ... existing calls ...
        self.create_total_records_per_sec_compare_graph()   # <-- new
        self.wb.save(self.file)
```

**Step E — Verify.**

```bash
./sbk-charts -i samples/charts/sbk-file-read.csv -o /tmp/out.xlsx
python3 -c "import openpyxl; print('Total_Records_PerSec' in openpyxl.load_workbook('/tmp/out.xlsx').sheetnames)"
```

Should print `True`. Open the workbook and inspect the new sheet.

### 3.3 Acceptance checklist

- [ ] New sheet appears in `wb.sheetnames`.
- [ ] Chart axes, title, and series legend render correctly.
- [ ] Multi-CSV runs (e.g. `-i a.csv,b.csv`) show one series per input.
- [ ] No regression to the existing chart set.

---

## 4. Add a new column / metric to `charts/constants.py`

**Goal:** Reference a CSV column that isn't yet a named constant. For
example, a new SBK version emits an `EffectiveQueueDepth` column you
want to plot.

### 4.1 Files to touch

1. <ref_file file="/root/projects/sbk-charts/src/charts/constants.py" /> —
   add the constant.
2. Whoever wants to *use* the column (a chart method, a prompt builder,
   etc.) imports the constant and references it.

### 4.2 Step-by-step

**Step A — Add the constant.** Use the **exact** header string from the
CSV as the value:

```python
# in src/charts/constants.py
EFFECTIVE_QUEUE_DEPTH = "EffectiveQueueDepth"
```

**Step B — Use it.** Never hard-code the string anywhere; always
reference `constants.EFFECTIVE_QUEUE_DEPTH`.

**Step C — Verify.** The column will be picked up automatically by
`get_columns_from_worksheet()` because that function reads whatever
headers actually exist in the worksheet. The constant is for *code*
to use, not for the parser.

---

## 5. Change a prompt template (affects all plugins)

**Goal:** Reword one of the four canonical prompts (throughput,
latency, total MB, percentile histogram) — e.g. add an instruction to
output in Markdown, or to use a specific time unit in the explanation.

### 5.1 Files to touch

- <ref_file file="/root/projects/sbk-charts/src/genai/genai.py" /> —
  the four `get_*_prompt()` methods.

### 5.2 Step-by-step

**Step A — Identify the prompt.** Each prompt builder returns the
prompt string that every plugin sends to its model:

| Builder | Used by | Lines |
|---|---|---|
| `get_throughput_prompt` | `get_throughput_analysis` in every plugin | ~152–200 |
| `get_latency_prompt` | `get_latency_analysis` | ~202–293 |
| `get_total_mb_prompt` | `get_total_mb_analysis` | ~296–326 |
| `get_percentile_histogram_prompt` | `get_percentile_histogram_analysis` | ~328–403 |

**Step B — Edit the prompt string.** Keep the structure (persona line,
task list, embedded metrics table, "Now write the analysis…"). Only
change the parts you intend to change.

**Step C — Verify with at least two backends.** Cloud and local should
both produce sensible output. The user's existing budget (`-secs`) does
not change.

```bash
./sbk-charts -i samples/charts/sbk-file-read.csv -o /tmp/out-gemini.xlsx gemini
./sbk-charts -i samples/charts/sbk-file-read.csv -o /tmp/out-ollama.xlsx ollama
```

Open both Summary sheets and confirm the analyses reflect the
prompt change.

### 5.3 Acceptance checklist

- [ ] Only `src/genai/genai.py` is touched.
- [ ] All four prompts still embed their metrics table.
- [ ] At least one cloud and one local backend produce sensible output.
- [ ] No regression in the chat-mode `get_response` flow.

---

## 6. Tweak the Summary sheet

**Goal:** Add a row, column, or new section to the Summary sheet.

### 6.1 Files to touch

- <ref_file file="/root/projects/sbk-charts/src/charts/multicharts.py" /> —
  `create_summary_sheet()` writes the metadata, drivers/actions, time
  unit, and benchmark date/time table.
- <ref_file file="/root/projects/sbk-charts/src/ai/sbk_ai.py" /> —
  `add_ai_analysis()` writes the four AI narratives into column H of
  the same sheet, using `sheet.max_row + N` anchors.

### 6.2 Gotcha — the two-phase Summary

See <ref_file file="/root/projects/sbk-charts/AGENTS.md" /> §4.2. If
you add rows in stage 2, the AI block at stage 3 *automatically*
pushes down because it uses `sheet.max_row`. If you add a new column
or reposition column H, you must update the AI side too.

### 6.3 Verification

```bash
./sbk-charts -i samples/charts/sbk-file-read.csv -o /tmp/out.xlsx
python3 -c "
import openpyxl
ws = openpyxl.load_workbook('/tmp/out.xlsx')['Summary']
for r in range(1, ws.max_row + 1):
    row = [ws.cell(row=r, column=c).value for c in range(7, 12)]
    if any(v is not None for v in row):
        print(r, row)
"
```

Confirm new content appears in the expected row range and the AI block
(if you ran with a backend) still lands below it.

### 6.4 Worked example

The recent benchmark date/time table (Sheet Name / Storage / Start /
End / Duration columns) was added by extending `create_summary_sheet()`
to iterate every R-sheet, read the first and last `Date`+`Time` cells,
compute the duration, and write a 5-column table just below the
action-to-storage block. The AI block continued to land correctly
because it anchors on `sheet.max_row + N`.

---

## 7. Debug a plugin that silently disappears from `-h`

**Symptom.** `sbk-charts -h` shows a smaller plugin list than expected.
The plugin's directory exists, but its subcommand is missing.

### 7.1 Root cause

The discovery layer (<ref_file file="/root/projects/sbk-charts/src/ai/discover.py" />)
catches **any** exception during a plugin's `import_module()` call and
prints `Importing module <name> failed with error: <e>` to stdout. The
discoverer then proceeds without the broken plugin. **This is by
design** — a broken plugin should not break the others — but it makes
import errors silent.

### 7.2 Procedure

**Step A — List discovered plugins explicitly.**

```bash
python3 -c "from src.ai.discover import discover_custom_ai_classes; \
    print(discover_custom_ai_classes())"
```

Stdout *above* the dict will contain the import errors.

**Step B — Re-run the actual import** by hand to get the full
traceback:

```bash
python3 -c "import src.custom_ai.<plugin>.<plugin>"
```

This will produce a clean traceback you can act on. Typical causes:

- Missing dependency in `requirements.txt`.
- Renamed module/class in an upstream SDK (the recent Gemini case:
  `google.ai.generativelanguage` was deprecated).
- Syntax error in the plugin file.

**Step C — Fix and re-verify.** After the fix, `discover_custom_ai_classes()`
should return a dict containing your plugin's lowercased class name as
a key, and `./sbk-charts -h` should list its subcommand.

---

## 8. Debug "AI timed out" / partial Summary output

**Symptom.** One or more of the four analyses come back as
`(False, "Analysis timed out")` in the Summary sheet.

### 8.1 Diagnosis order

1. **Is the model just slow?** Re-run with `-secs 600` (10 minutes).
   If it now succeeds, the prompt is fine; you just hit the default
   120 s wall.
2. **Is one analysis blocking the others?** Re-run with `-nothreads`.
   In sequential mode you'll see exactly which analysis stalls.
3. **Is the model OOM-ing or being throttled?** For `PyTorchLLM` on a
   single GPU: definitely use `-nothreads`. For cloud APIs: check the
   provider's rate-limit response.
4. **Is the prompt too long?** Each prompt embeds a metrics table.
   With many storage systems × many percentiles, the latency prompt
   in particular can grow. If you suspect this, log the prompt length:
   ```python
   print(f"DEBUG prompt len = {len(prompt)} chars")
   ```
   inside the plugin before sending.
5. **Is the model returning empty / unparseable text?** Each plugin
   wraps response parsing in `try/except`. If the model returns nothing
   useful, the plugin returns `(False, "No content in response")` —
   that is *not* the same as a timeout. Distinguish the two when
   reporting back to the user.

---

## 9. Bump a Python dependency

**Goal:** Move a package in `requirements.txt` to a newer release.

### 9.1 Step-by-step

**Step A — Pick the bump.** Patch (`1.2.3 → 1.2.4`) and minor
(`1.2.3 → 1.3.0`) bumps are fine. **Major** bumps (`1.x → 2.x`) for
key deps (`pandas`, `openpyxl`, `torch`, `google-genai`, `anthropic`)
require user approval per
<ref_file file="/root/projects/sbk-charts/AGENTS.md" /> §7.

**Step B — Edit `requirements.txt`.** Keep the `~=` pin form:

```
google-genai~=1.63.0
```

**Step C — Install and run.**

```bash
pip install -r requirements.txt --upgrade
./sbk-charts -i samples/charts/sbk-file-read.csv -o /tmp/out.xlsx
```

If the dep is plugin-specific (e.g. `anthropic`), also run with that
plugin to confirm the API surface hasn't changed.

**Step D — If the API has changed**, follow Recipe §2.3 (worked
example: Gemini migration) — update the plugin's imports and method
shape to match the new SDK.

---

## 10. Cut a new release

> Requires explicit user approval per
> <ref_file file="/root/projects/sbk-charts/AGENTS.md" /> §7.

**Step A — Bump the version.** Edit
<ref_file file="/root/projects/sbk-charts/src/version/sbk_version.py" />:

```python
__sbk_version__ = "3.26.2.2"
```

**Step B — Build the wheel and sdist.**

```bash
python -m build
ls dist/
# dist/sbk_charts-3.26.2.2-py3-none-any.whl
# dist/sbk_charts-3.26.2.2.tar.gz
```

**Step C — Smoke-test the wheel in a clean venv.**

```bash
python3 -m venv /tmp/sbk-release-test
source /tmp/sbk-release-test/bin/activate
pip install dist/sbk_charts-3.26.2.2-py3-none-any.whl
sbk-charts -i samples/charts/sbk-file-read.csv -o /tmp/out.xlsx
deactivate
```

**Step D — Tag, push, and upload to GitHub Releases.** *Only with
explicit user approval.*

```bash
git tag 3.26.2.2
git push origin 3.26.2.2
# upload dist/* to https://github.com/kmgowda/sbk-charts/releases/new
```

---

*If a task you need to do is not in this file, add it here when
you're done so the next agent benefits.*
