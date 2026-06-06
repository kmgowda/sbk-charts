<!--
Copyright (c) KMG. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
-->

# AGENTS.md — AI Agent Guide for the sbk-charts Repository

> **Audience.** This file is the standard entry point for AI coding agents
> (Devin, Claude Code, Cursor, GitHub Copilot, Continue, Aider, OpenAI
> Codex, etc.) working in this repository. It tells the agent **what
> sbk-charts is, how to run and verify it, what conventions to follow,
> where things live, and what the common gotchas are**.
>
> Humans: see <ref_file file="/root/projects/sbk-charts/README.md" /> for
> the end-user manual, and <ref_file file="/root/projects/sbk-charts/docs/ARCHITECTURE.md" />
> for the internal design.

---

## 1. What this repository is

**sbk-charts** is a Python application that converts one or more
[SBK](https://github.com/kmgowda/SBK) benchmark CSV files into a
richly-formatted `.xlsx` workbook containing:

1. **R/T worksheets** — for each input CSV, two sheets are produced:
   - `R<n>` — *per-interval* rows (rows where the `Type` column is anything
     other than `Total`).
   - `T<n>` — *total/summary* rows (rows where `Type == "Total"`).
2. **~20 chart sheets** — throughput, latency, percentile, percentile-count
   histograms, write/read variations, and timeout-event comparisons.
3. **A Summary sheet** — metadata (sbk-charts version, generation date/time,
   drivers, time unit, the benchmark date/time table), plus optional
   AI-generated narrative analysis.

**Languages & runtime.** Python 3.10+. Pure-Python (no native code besides
PyTorch wheels in optional plugins).

**License.** Apache 2.0.

**Default branch.** `main`. Active development happens on feature branches
and is merged to `main` via PR.

### Top-level package map (memorise this)

| Package | Role | When you edit it |
|---|---|---|
| `src/main/` | CLI entry point (`sbk_charts()` orchestrator). | When changing top-level flow (sheets → charts → AI). |
| `src/parser/` | Base argparse setup (`-i`, `-o`). | Rarely. |
| `src/sheets/` | CSV → R/T worksheets. | When changing how CSVs map to sheets, or sheet naming. |
| `src/charts/` | All chart generation (per-run and multi-run). ~2 100 LoC. | When adding/changing a chart, the Summary sheet, or chart constants. **Most common change after plugins.** |
| `src/stat/` | `StorageStat` immutable dataclass. | Rarely. Only when adding a new top-level statistic. |
| `src/genai/` | `SbkGenAI` abstract base + prompt builders. | When changing the prompt templates (affects every backend). |
| `src/ai/` | `SbkAI` orchestrator (parallel execution, chat, plugin discovery). | When changing how AI analyses run, time out, or are written into Excel. |
| `src/rag/` | Simple RAG (default) + ChromaDB RAG (optional). | When changing retrieval logic or grounding. |
| `src/custom_ai/<name>/` | One subdirectory per AI backend. **7 plugins today.** | When adding or fixing an AI backend. **Most common change.** |
| `src/version/` | `__sbk_version__` string. | When cutting a release. |

**For new AI plugins, see <ref_file file="/root/projects/sbk-charts/docs/PLUGIN_SPECIFICATION.md" />
(spec template + worked example) and
<ref_file file="/root/projects/sbk-charts/docs/AGENT_RECIPES.md" />
("Add a new AI plugin" recipe).**

---

## 2. Build, run, and verify

### Set up the dev environment

```bash
# from the repo root
python3 -m venv venv-sbk-charts
source venv-sbk-charts/bin/activate
pip install -r requirements.txt
```

### Run the tool from source

```bash
# Single CSV
./sbk-charts -i /path/to/run.csv -o out.xlsx

# Multiple CSVs (driver comparison)
./sbk-charts -i run-a.csv,run-b.csv,run-c.csv -o compare.xlsx

# With an AI backend (Gemini)
export GEMINI_API_KEY=...
./sbk-charts -i run.csv gemini --gemini-model gemini-2.5-flash

# With a local backend (Ollama)
./sbk-charts -i run.csv ollama --ollama-model llama3.1

# With chat mode after the analyses
./sbk-charts -i run.csv gemini -chat

# Disable threading for plugin debugging
./sbk-charts -i run.csv pytorchllm -nothreads

# List available AI backends and base flags
./sbk-charts -h
```

### Build a distributable wheel

```bash
python -m build
# produces dist/sbk_charts-<version>-py3-none-any.whl
```

### Verification — what counts as "done"

A change is **done** only after all of the following succeed:

1. **The CLI runs to completion** on a sample CSV:
   ```bash
   ./sbk-charts -i samples/charts/sbk-file-read.csv -o /tmp/out.xlsx
   ```
   No tracebacks, exit code 0, and `/tmp/out.xlsx` is created.

2. **If you touched plugin code**, run the affected plugin end-to-end:
   ```bash
   ./sbk-charts -i samples/charts/sbk-file-read.csv -o /tmp/out.xlsx <plugin>
   ```
   The four analyses must all complete (or fail gracefully with a clear
   error message — e.g. missing API key) within the 120 s budget.

3. **If you touched chart code**, open the generated `.xlsx` and verify
   the new/changed chart looks right:
   ```bash
   python3 -c "import openpyxl; wb = openpyxl.load_workbook('/tmp/out.xlsx'); print(wb.sheetnames)"
   ```

4. **No new pip dependency** introduced without adding it to
   `requirements.txt`.

> There are **no unit tests** in the repo today. End-to-end CLI runs are
> the verification mechanism.

---

## 3. Repository conventions

### File-system conventions

| Path | Convention |
|---|---|
| `src/custom_ai/<name>/__init__.py` | Can be empty. |
| `src/custom_ai/<name>/<name>.py` | The plugin module. Lower-case filename matches the directory name. |
| Class name inside the plugin | PascalCase of the directory name. `gemini/` → class `Gemini`; `hugging_face/` → class `HuggingFace`; `pytorch_llm/` → class `PyTorchLLM`. **The discovery layer keys plugins by lower-cased class name**, so a class named `HuggingFace` becomes the subcommand `huggingface`. |
| `src/charts/constants.py` | Add new column-name constants here. Use the exact CSV header string as the value. |
| `src/sheets/constants.py` | Sheet naming constants (`R_PREFIX`, `T_PREFIX`, `TYPE`). |
| `src/version/sbk_version.py` | Single-line `__sbk_version__ = "X.Y.Z.N"`. |
| `requirements.txt` | All runtime deps. Pin with `~=` (PEP 440 compatible release). |
| `setup.py` | Picks up `requirements.txt` automatically; do not duplicate deps. |

### Coding conventions

- **Python 3.10+**. Use `typing` features (`Optional`, `List`, `Tuple`, etc.).
- **Type hints** on public methods. Existing code uses sparse hints; add
  them when modifying a function. Don't strip existing hints.
- **Docstrings** are encouraged on every public function/method. Follow
  the existing style: short summary, `Parameters:`/`Returns:` blocks.
- **No emojis** in code, comments, or docs unless the user explicitly
  requests them. (Existing `print()` calls do contain some emoji
  decorations — match the surrounding style when editing those.)
- **No `print()` for new debug output** — if you need diagnostics, route
  through `logging` like `src/rag/sbk_rag.py` does.
- **Imports**: stdlib first, third-party second, `src.*` last. Do not
  add wildcard imports.
- **Constants for column names**: never hard-code a CSV header string
  like `"MB/Sec"` or `"AvgLatency"` inline. Always use a constant from
  `src/charts/constants.py`. Add a new constant if the column is new.

### Naming conventions

- **AI plugin subcommand** = lower-cased class name. `Gemini` → `gemini`;
  `LmStudio` → `lmstudio`.
- **CLI flags for a plugin** are prefixed by the plugin name:
  `--gemini-model`, `--anthropic-max-tokens`, `--pt-device`. Pick the
  shortest unambiguous prefix.
- **Chart sheet names** are pre-chosen; do not rename existing ones —
  user-saved workbooks depend on them. New charts should follow the
  pattern `Total_<MetricName>` for T-sheet charts and `<MetricName>` or
  `<MetricName>-<n>` for R-sheet charts.

---

## 4. Known gotchas (in priority order)

### 4.1 Plugin discovery silently swallows ImportError

`src.ai.discover.discover_custom_ai_classes` calls
`importlib.import_module` on every submodule of `src.custom_ai`. If the
import fails (typically a missing third-party dep), the discoverer prints
`Importing module <name> failed with error: <e>` and **continues** so
the other plugins remain usable. **The result is that a broken plugin
silently disappears from `sbk-charts -h`.**

If the user reports "I can't see my plugin in the help output", run:

```bash
python3 -c "from src.ai.discover import discover_custom_ai_classes; print(discover_custom_ai_classes())"
```

The first import error in the stdout is your culprit. This is exactly
how the recent Gemini fix surfaced (`google.ai.generativelanguage` →
`google.genai`).

### 4.2 The Summary sheet is built in two phases

`SbkMultiCharts.create_summary_sheet()` (stage 2) writes the version,
date/time, drivers/actions table, and the benchmark date/time table.
Then `SbkAI.add_ai_analysis()` (stage 3) appends the four AI narratives
into column H of the **same** sheet.

If you change the Summary layout, you must verify **both** writers still
agree on which row to start at. The AI step uses `sheet.max_row + 3` and
`sheet.max_row + 2` as anchors, so adding rows in stage 2 automatically
pushes the AI block down — but adding columns or repositioning column H
will break the AI block.

### 4.3 The R/T sheet split is the universal addressing scheme

Every downstream module (charts, AI orchestration, RAG) addresses data
by R-sheet/T-sheet name and uses `is_r_num_sheet` / `is_t_num_sheet`
(<ref_file file="/root/projects/sbk-charts/src/charts/utils.py" />) to
classify a worksheet. If you rename, prefix, or otherwise rearrange the
data sheets, you must update those regexes and every caller.

Sheets named anything other than `R<digits>` or `T<digits>` are ignored
by the chart and AI code paths. This is why `Summary`, `Throughput_MB`,
etc. coexist safely in the same workbook.

### 4.4 Prompts live in `SbkGenAI`, not in plugins

The four prompt-builder methods (`get_throughput_prompt`,
`get_latency_prompt`, `get_total_mb_prompt`,
`get_percentile_histogram_prompt`) live in
<ref_file file="/root/projects/sbk-charts/src/genai/genai.py" />. **All
seven plugins call them.** Changing a prompt template therefore changes
the output of every backend simultaneously — that is the design.

If you want a plugin-specific prompt tweak, override the relevant
`get_*_analysis()` method and build a custom prompt there. Do not
duplicate the canonical prompts in the plugin.

### 4.5 ThreadPoolExecutor is the default; some plugins need `-nothreads`

`SbkAI.add_ai_analysis()` submits 4 analyses to a 4-worker pool. For
cloud APIs this is fine. For `PyTorchLLM` on a single GPU it can OOM
the device — the user must pass `-nothreads` to fall back to sequential
execution. Document this in any new plugin's README if it has this
property.

### 4.6 The 120-second budget is a hard wall

`-secs/--seconds` defaults to 120. Once exceeded, in-flight futures are
cancelled and their results become `(False, "Analysis timed out")`. A
slow local model on CPU **will** hit this — surface this clearly to
users with `print()` warnings, not silent partial output.

### 4.7 RAG ingestion of zero-valued metrics is intentionally skipped

`SbkSimpleRAGPipeline._process_storage_stat` skips metrics whose values
are all zero (e.g. write metrics during a pure-read benchmark). This
prevents the RAG index from being polluted with meaningless rows.
Do not "fix" this unless you have a specific reason — the keyword scorer
relies on the assumption that ingested data is non-trivial.

### 4.8 `StorageStat` is a frozen dataclass

You cannot reassign `stat.regular[...] = …` after construction. Build
the full `regular` and `total` dicts before calling `StorageStat(...)`.
If you need an in-place "edit", construct a new `StorageStat` with the
changed fields.

### 4.9 Mermaid diagrams in `docs/ARCHITECTURE.md`

If you edit the architecture doc's diagrams, test them with `mmdc`
(mermaid-cli v11+ on Node 18+). Pitfalls:

- HTML entities like `&#91;` / `&lt;` are rendered **literally** in some
  versions. Use plain ASCII inside `[" "]` node labels, or `<br/>` for
  line breaks (which is supported).
- `++` is a reserved token in sequence-diagram messages. Use words
  instead (e.g. "plus" or "and").
- Em-dash (`—`) and Unicode arrow (`→`) inside sequence-diagram messages
  cause parse errors. Use `--` and `then`/`to`.
- `participant X as Some (Name)` with unquoted parens fails — drop the
  parens from aliases.

### 4.10 Do not commit `out.xlsx`, `venv-sbk-charts/`, or generated wheels

These are working artefacts. `.gitignore` already covers them but
agents using `git add -A` can accidentally pick up stray files. Stage
specific files only.

---

## 5. Where to look for deeper documentation

| Topic | Read |
|---|---|
| End-user manual | <ref_file file="/root/projects/sbk-charts/README.md" /> |
| Internal architecture, data flow, design decisions, open research problems | <ref_file file="/root/projects/sbk-charts/docs/ARCHITECTURE.md" /> |
| Step-by-step recipes (add an AI plugin, add a chart, fix a bug, …) | <ref_file file="/root/projects/sbk-charts/docs/AGENT_RECIPES.md" /> |
| Fillable spec template for spec-driven plugin development | <ref_file file="/root/projects/sbk-charts/docs/PLUGIN_SPECIFICATION.md" /> |
| SBK upstream (the benchmark engine that produces the CSVs) | <https://github.com/kmgowda/SBK> |

---

## 6. The two AI-development workflows this repo supports

This repository works equally well for both styles of AI-assisted
development:

### 6.1 Vibe coding (informal, iterative)

For quick fixes, single-file edits, debugging:

1. Agent reads the relevant file + this `AGENTS.md` + the relevant
   `AGENT_RECIPES.md` recipe.
2. Agent makes the change.
3. Agent verifies with the end-to-end CLI run from §2.
4. Agent reports results to the human.

**Loop is small.** No spec document. Suitable for: bugfixes, prompt
tweaks, README updates, small refactors, adding a new chart column to
`constants.py`.

### 6.2 Spec-driven development (formal, repeatable)

For larger work (a new AI plugin, a new chart, a new feature in
`SbkAI`):

1. Human (or AI assistant) writes a spec by filling in
   <ref_file file="/root/projects/sbk-charts/docs/PLUGIN_SPECIFICATION.md" />
   (for plugins) or a similar markdown template.
2. Spec is reviewed / refined by the human.
3. Agent reads the spec + `AGENTS.md` + `AGENT_RECIPES.md`.
4. Agent generates code and updates docs according to the spec.
5. Agent runs the verification checklist; iterates on failures.
6. Spec stays in version control as the source of truth for the
   feature.

**Loop is larger** but produces auditable artefacts.

The spec template explicitly cross-references the recipes, so the
agent has a single deterministic path from spec → working code.

---

## 7. Things that are out of scope for an AI agent without explicit user approval

The following actions require explicit user confirmation **for every
specific action** (not blanket approval):

- Running `git push`, `git tag`, or any operation that publishes to a
  remote.
- Modifying the Apache 2.0 license headers or `LICENSE` file.
- Changing the version string in
  <ref_file file="/root/projects/sbk-charts/src/version/sbk_version.py" />.
- Cutting a GitHub release or uploading wheels.
- Adding a new top-level Python package (i.e. something parallel to
  `src/ai/`, `src/charts/`, etc.). New *AI plugins* under `src/custom_ai/`
  are fine.
- Upgrading a major version of a key dependency (`pandas`, `openpyxl`,
  `torch`, `google-genai`). Patch-level / minor bumps are fine.
- Force-pushing, rewriting history, or deleting branches.

For everything else inside `src/custom_ai/`, `src/charts/`, `src/sheets/`,
`src/genai/`, `src/ai/`, `src/rag/`, `docs/`, and the top-level CLI,
normal edit-and-verify flow is fine.

---

## 8. Quick agent self-check before starting

Before making any change, the agent should be able to answer these
questions for the change at hand. If the agent can't answer them, it
should re-read this file and the relevant linked docs.

1. Which package does my change live in (`src/custom_ai/<name>/`,
   `src/charts/`, …)?
2. What's the verification command that proves my change is correct
   (typically a CLI run on `samples/charts/sbk-file-read.csv`)?
3. Have I touched any of the gotcha areas in §4? Did I update both
   sides of any cross-cutting change (e.g. Summary sheet writers in
   stage 2 and stage 3)?
4. If I added a new dependency, did I add it to `requirements.txt`?
5. If I added a new AI plugin, did I follow the directory + class +
   subcommand naming convention from §3?
6. If I changed a prompt, did I check that all seven plugins still
   produce a sensible response (or at least that my change is one
   place — `SbkGenAI` — and applies uniformly)?
7. Are there any architectural invariants from
   <ref_file file="/root/projects/sbk-charts/docs/ARCHITECTURE.md" /> §8
   my change must preserve (three-stage ordering, R/T addressing,
   prompts-in-framework, frozen `StorageStat`)?

When in doubt, **prefer reading existing code over making assumptions**.
The 7 existing plugins, in particular, are the canonical reference for
how a plugin should look.
