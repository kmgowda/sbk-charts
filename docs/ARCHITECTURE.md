<!--
Copyright (c) KMG. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
-->

# Architecture and internals

This guide explains how sbk-charts works from the command line down to individual workbook cells. Start with the first three sections if you are new to the project. The later sections describe extension points, invariants, release tooling, and failure behavior for experienced engineers.

For installation and usage examples, read the [main README](../README.md). If this is your first change, follow the [development guide](DEVELOPMENT.md). For detailed change procedures, read [AGENTS.md](../AGENTS.md) and [AGENT_RECIPES.md](AGENT_RECIPES.md).

## 1. Purpose and boundaries

SBK is the benchmark producer. It writes wide CSV files containing metadata, per-interval performance values, and total values. sbk-charts is the reporting consumer. It does not run a benchmark and it does not change benchmark results. It converts those results into an Excel report and can optionally ask an AI model to explain them.

The application has five main responsibilities:

1. parse input and output options;
2. split each CSV into regular and total worksheets;
3. create summary tables and charts;
4. optionally create four AI narratives;
5. optionally answer interactive questions grounded in the loaded statistics.

The launchers and release scripts form a separate delivery layer. They find or bundle Python, install dependencies, and start the same application entry point.

## 2. System view

```mermaid
flowchart LR
    U[User] --> L[Source launcher or portable executable]
    L --> M[CLI entry point]
    C[SBK CSV files] --> S[Sheet builder]
    M --> S
    S --> W[Excel workbook]
    W --> G[Chart builder]
    G --> W
    M --> A[AI orchestrator]
    P[AI backend plugin] --> A
    R[Simple RAG] --> A
    A --> W
    A --> Q[Interactive chat]
```

There is one important implementation detail: the workbook is written in stages by two Excel libraries.

- `pandas` reads CSV data.
- `XlsxWriter` creates the first workbook and the R/T data sheets.
- `openpyxl` reopens that workbook, adds summary and chart sheets, applies styles, and later appends AI text.

## 3. End-to-end sequence

The main function is `sbk_charts()` in `src/main/sbk_charts.py`.

```mermaid
sequenceDiagram
    actor User
    participant Main as CLI entry point
    participant AI as SbkAI
    participant Sheets as SbkMultiSheets
    participant Charts as SbkMultiCharts
    participant Book as XLSX workbook
    participant Plugin as Selected AI plugin

    User->>Main: input CSV paths and output XLSX path
    Main->>AI: register lightweight backend subcommands
    Main->>Sheets: create_sheets
    Sheets->>Book: write R1 T1 R2 T2 sheets
    Main->>Charts: create_graphs
    Charts->>Book: add Summary duration charts and themes
    Main->>AI: parse arguments and open backend
    alt AI backend selected
        AI->>Book: load R and T statistics
        AI->>Plugin: run four analyses
        Plugin-->>AI: success flag and text
        AI->>Book: append analysis to Summary
    else no backend selected
        AI-->>User: explain that charts are complete without AI
    end
    opt chat requested
        AI->>AI: initialize Simple RAG
        User->>AI: ask questions
        AI->>Plugin: send question with retrieved context
        Plugin-->>User: answer
    end
```

The stage order is an architectural invariant. Chart code expects R/T sheets to exist. AI code expects the Summary sheet and charts stage to have finished.

## 4. Repository map

| Path | Responsibility | Main abstractions |
|---|---|---|
| `src/main/` | Application entry point and banner | `sbk_charts()` |
| `src/parser/` | Base input/output arguments | `get_sbk_parser()` |
| `src/sheets/` | CSV parsing and initial workbook creation | `SbkSheets`, `SbkMultiSheets` |
| `src/charts/` | Data lookup, tables, all charts, Summary sheet | `SbkCharts`, `SbkMultiCharts` |
| `src/stat/` | Immutable AI-facing data transfer object | `StorageStat` |
| `src/genai/` | Backend interface and shared prompts | `SbkGenAI` |
| `src/ai/` | Lazy registry, execution, timeout, Excel AI layout, chat | `SbkAI`, `BACKENDS` |
| `src/rag/` | Retrieval for chat and prompt grounding | `SbkSimpleRAGPipeline`, optional `SbkRAGPipeline` |
| `src/custom_ai/` | Provider and local-model adapters | Seven `SbkGenAI` subclasses |
| `src/version/` | Canonical release version | `__sbk_version__` |
| `scripts/` | Policy reader, portable entry point, portable builder, Windows extractor template | `project_policy.py`, `build_portable.py`, `windows_self_extractor.cs` |
| `tests/` | Runtime-policy and portable-build unit tests | `test_portable.py` |
| `.github/workflows/` | Linux lint/tests, Windows launcher smoke tests, native portable builds | GitHub Actions jobs |
| `.devin/skills/` | Task instructions usable by Devin and other capable agents | Onboarding, build, chart, plugin, troubleshooting skills |

## 5. Command-line construction

`src/parser/sbk_parser.py` creates only the base parser:

- `-i` / `--ifiles`, required and comma-separated;
- `-o` / `--ofile`, default `out.xlsx`.

`SbkAI.add_args()` adds the cross-backend AI options:

- `-secs` / `--seconds`;
- `-nothreads`;
- `-chat`.

It reads a static lightweight registry and creates one argparse subcommand per backend without importing provider SDKs. Each descriptor explicitly provides the command name, implementation module and class, and argument-registration function. Those flags must match the implementation's `parse_args()` contract.

```mermaid
flowchart TD
    A[get_sbk_parser] --> B[Base input and output options]
    B --> C[SbkAI.add_args]
    D[Static BACKENDS registry] --> C
    C --> E[Global AI options]
    C --> F[One subparser per backend descriptor]
    F --> G[Plugin-specific options]
    G --> H[argparse Namespace]
```

Argparse treats the backend name as a boundary. Global flags go before the backend and plugin flags go after it.

## 6. CSV-to-sheet layer

`SbkMultiSheets.create_sheets()` opens an XlsxWriter workbook, creates an `SBK` cover sheet with the project logo, and processes input files in command-line order. File one becomes `R1` and `T1`, file two becomes `R2` and `T2`, and so on.

`wb_add_two_sheets()` reads the `Type` column:

```text
Type == "Total"  -> T<n>
every other Type -> R<n>
```

The CSV header is copied to row 1. Column widths begin at the header width and expand when a value is longer. The SBK logo and cover sheet are inserted once during initial workbook creation.

The R/T naming scheme is the universal address system for the rest of the program:

```mermaid
flowchart LR
    C1[CSV file 1] --> R1[R1 interval rows]
    C1 --> T1[T1 total rows]
    C2[CSV file 2] --> R2[R2 interval rows]
    C2 --> T2[T2 total rows]
    R1 --> X[Charts AI and RAG]
    T1 --> X
    R2 --> X
    T2 --> X
```

`src/charts/utils.py` recognizes only names matching `R<digits>` and `T<digits>`. Other sheets are safely ignored when statistics are collected.

## 7. Chart layer

### 7.1 Base and multi-run classes

`SbkCharts` in `src/charts/charts.py` provides reusable operations:

- map worksheet headers to column indexes;
- create openpyxl `Series` objects from named metrics;
- create line and bar charts;
- split percentile sets into readable groups;
- create per-run latency, percentile, histogram, and throughput sheets;
- apply shared chart and table themes.

`SbkMultiCharts` extends that behavior across every R/T pair. It creates the Summary and Durations sheets, comparisons across runs, totals, and the final workbook order.

Exact SBK column names and ordered percentile chart groups live in `src/charts/constants.py`. Stable sheet names, sheet prefixes, and the `Type == Total` value live in `src/sheets/constants.py`. Chart code should never repeat an SBK header or workbook sheet name inline.

### 7.2 Workbook order

`SbkMultiCharts.create_graphs()` deliberately places large comparisons before fine-grained views:

```mermaid
flowchart TD
    A[Summary] --> B[Durations]
    B --> C[Broad throughput and write read comparisons]
    C --> D[Grouped latency comparisons]
    D --> E[Total percentile histogram]
    E --> F[Total MB and throughput summaries]
    F --> G[Timeout comparisons]
    G --> H[Total min avg max latency]
    H --> I[Total percentile curves]
    I --> J[Per-run and fine-grained latency views]
```

Representative generated names include:

- `Summary` and `Durations`;
- `Throughput_MB` and `Throughput_Records`;
- `Write_Read_MB` and `Write_Read_Records`;
- `Total_Percentiles_Histogram`;
- `Total_MB`, `Total_Throughput_MB`, and `Total_Throughput_Records`;
- `RW_TimeoutEvents` and `RW_TimeoutEvents_Per_Sec`;
- `Total_Min_Latency`, `Total_Avg_Latency`, and `Total_Max_Latency`;
- `Total_RW_TimeoutEvents`;
- grouped and per-run latency and percentile sheets.

Some long chart families are split across numbered sheets so labels and lines remain readable.

### 7.3 Time-unit safety

Before adding graphs, `check_time_units()` reads the latency unit from every R sheet. If units differ, comparison charts are not created. This prevents a visually convincing but invalid comparison, such as plotting nanoseconds and milliseconds on the same axis.

### 7.4 Presentation policy

Chart dimensions, font sizes, line colors, table fills, row heights, widths, legends, and grid styles are presentation policy. They are grouped in the chart implementation rather than in global runtime configuration. `apply_table_theme()` and `apply_chart_theme()` run after sheet creation so all generated sheets receive consistent formatting.

## 8. Summary sheet ownership

The Summary sheet has two writers.

1. `SbkMultiCharts.create_summary_sheet()` creates report metadata, driver/action information, and the latency unit. `create_sbk_date_sheet()` writes benchmark start, end, and elapsed time to the separate Durations sheet.
2. `SbkAI.add_ai_analysis()` later appends the AI warning, model description, and four narratives in columns G and H.

The AI block uses `sheet.max_row` plus spacing to choose its start row. Adding rows to the chart-owned section automatically moves AI content down. Moving the AI columns or changing their meaning requires coordinated changes in both modules.

## 9. Statistics model

`SbkAI.get_storage_stats()` visits every R sheet, finds its matching T sheet, and converts both into one `StorageStat`:

```text
StorageStat(
    storage=<driver name>,
    timeunit=<latency unit>,
    action=<read or write action>,
    regular={<metric>: [interval values...]},
    total={<metric>: [total values...]},
)
```

Metadata columns such as ID, Header, Type, Storage, Action, and Latency Time Unit are excluded from the metric mappings.

`StorageStat` is a frozen dataclass. This makes the container safe to share between analysis workers, but its nested dictionaries are not deeply immutable. Existing code builds complete mappings before construction and treats them as read-only afterward.

## 10. AI plugin system

### 10.1 Lazy registry

`src/ai/registry.py` declares the stable command name, implementation module, class, and CLI argument builder for each backend. Dependency-free defaults shared by the registry and adapters live in `src/ai/defaults.py`. The registry does not import optional provider modules. `load_backend_class()` imports only the command selected by the user:

```text
gemini      -> Gemini
huggingface -> HuggingFace
lmstudio    -> LmStudio
pytorchllm  -> PyTorchLLM
```

This keeps `sbk-charts -h` and core chart generation independent of optional SDKs. `SbkAI.parse_args()` imports only the selected descriptor's implementation, so a missing package fails after selection rather than hiding commands from help. `src/ai/discover.py` remains a developer diagnostic, not the runtime command registry.

### 10.2 Interface

`SbkGenAI` defines lifecycle hooks, model description, four analysis methods, and chat response behavior. A backend normally implements:

- descriptor-owned argument registration and implementation-owned `parse_args(args)`;
- `open(args)` and `close(args)` when it owns resources;
- `get_model_description()`;
- `get_throughput_analysis()`;
- `get_latency_analysis()`;
- `get_total_mb_analysis()`;
- `get_percentile_histogram_analysis()`;
- `get_response(query)` for chat.

All result methods use `(success: bool, text: str)`. Expected provider, authentication, network, or model errors should be returned as readable failure text rather than escaping to the orchestrator.

### 10.3 Canonical prompts

The shared prompt builders live in `src/genai/genai.py`, not in individual plugins. They format the same `StorageStat` data for every backend. This gives cloud and local models the same analytical task and keeps prompt improvements in one place.

| Prompt | Main metrics |
|---|---|
| Throughput | MB/sec, records/sec, write/read rates |
| Latency | minimum, average, maximum, and latency series |
| Total MB | total transferred data |
| Percentile histogram | latency percentile and percentile-count distributions |

Plugins can add provider-specific transport and chat framing, but should reuse the canonical analysis prompts unless a reviewed requirement says otherwise.

### 10.4 Current backends

| Command | Adapter | Location | Main external need |
|---|---|---|---|
| `anthropic` | Anthropic Python SDK | `src/custom_ai/anthropic/` | API key and network |
| `gemini` | Google Gen AI SDK | `src/custom_ai/gemini/` | API key and network |
| `huggingface` | Hugging Face inference client | `src/custom_ai/hugging_face/` | API token and network |
| `lmstudio` | LM Studio SDK | `src/custom_ai/lm_studio/` | Reachable LM Studio server |
| `ollama` | HTTP requests to Ollama | `src/custom_ai/ollama/` | Reachable Ollama server and downloaded model |
| `pytorchllm` | Transformers and PyTorch | `src/custom_ai/pytorch_llm/` | Model files and enough CPU, RAM, or GPU memory |
| `noai` | Deterministic placeholder | `src/custom_ai/no_ai/` | Nothing |

## 11. AI execution and timeout behavior

By default, `SbkAI.add_ai_analysis()` submits four analysis methods to a `ThreadPoolExecutor` with four workers. It polls for completed futures and applies one total time budget, default 120 seconds. Remaining results receive timeout text when the budget is exceeded.

With `-nothreads`, the same methods run sequentially. The elapsed budget is checked after each method. Sequential mode is useful for debugging and for models that cannot safely handle four simultaneous calls, especially a large model on one GPU.

```mermaid
flowchart TD
    A[Four analysis methods] --> B{nothreads?}
    B -- No --> C[Four worker executor]
    B -- Yes --> D[Run one method at a time]
    C --> E[Collect success or error tuples]
    D --> E
    E --> F[Fill missing or timed-out results]
    F --> G[Write all four sections to Summary]
```

The application saves provider error text into the report so chart generation is still useful when AI is unavailable.

## 12. Retrieval and chat

Chat mode uses `SbkSimpleRAGPipeline` by default. It is an in-memory retrieval system and needs no vector database.

At initialization it:

1. receives `StorageStat` objects;
2. turns non-trivial total metrics into text documents and metadata;
3. skips metrics whose values are all zero;
4. extracts storage-system names and semantic tags;
5. stores the documents in memory.

For a question it extracts keywords, recognizes comparison intent, scores stored documents, returns relevant measurements, and formats them as context. `SbkGenAI._enhance_prompt_with_rag()` adds that context before the plugin sends the request to its model.

```mermaid
flowchart LR
    S[StorageStat objects] --> I[In-memory ingestion]
    Q[User question] --> K[Keyword and intent extraction]
    I --> R[Scored retrieval]
    K --> R
    R --> C[Formatted benchmark context]
    C --> P[Backend prompt]
    Q --> P
    P --> A[AI answer]
```

`src/rag/sbk_chroma_rag.py` contains an optional ChromaDB-based implementation, but the normal application path does not instantiate it. Do not describe ChromaDB as a runtime requirement.

## 13. Self-bootstrap architecture

There are three source launchers:

- `sbk-charts` for Bash on Linux and macOS;
- `sbk-charts.ps1` for PowerShell;
- `sbk-charts.bat`, which delegates to PowerShell for Command Prompt users.

They use native shell logic to read `sbk-charts.ini` because Python may not exist yet. A supported launcher can download a pinned, SHA-256-verified `uv`, use it to install an exact project-managed Python, and build an environment from a hashed lock. After choosing Python, the launcher uses `scripts/project_policy.py` for shared validation, runtime reporting, and remembered-environment state.

When `SBK_CHARTS_VENV` is not set, runtime selection happens in stages:

1. try the last validated environment from the state file;
2. try the active virtual or Conda environment and known project-local virtual environments;
3. try the reusable fingerprinted managed venv, then the configured named-Conda environment;
4. on a supported managed target, create the exact managed Python and locked environment;
5. if managed setup is unsupported or fails, probe system Python candidates for venv support and then prepare the named Conda environment.

Both native implementations use the same stage order. The remembered state remains the first preference, while the exact project-owned managed cache is preferred over a named Conda environment when neither was remembered.

When `SBK_CHARTS_VENV` is set, the launcher tries that explicit path first and skips remembered, active, and project-local candidates. It does not create a new managed environment while the override is set. If the explicit venv is unusable, it may reuse an existing managed or named Conda environment, create the requested normal venv with a suitable system Python, or fall back to Conda.

Legacy environment creation checks Python candidates in policy order. A candidate is selected only after it creates a temporary venv whose `ensurepip` and `pip` commands work; failed probes are removed before the next interpreter is tried. Bootstrap-manager downloads and unpublished managed environments are also temporary and are removed on failure.

An environment is reusable only when it has a supported Python, the installed distribution version matches the source version, the application and selected backend import, and the dependency check succeeds. A managed environment also must match the fingerprint of target, exact Python, selected profile, and lock contents. Creation is serialized by a bootstrap lock with a policy-controlled wait timeout and published by directory rename only after self-validation. The lock is released before the application process replaces the launcher. The successful runtime is written atomically to `.sbk-charts-runtime` immediately before the application starts.

Selection provenance is captured before the state file is rewritten. The launcher passes the environment profile, selection source, saved-state reuse flag, and creation flag to `project_policy.py`. This produces the same structured runtime report on Bash, PowerShell, and the CMD shim. A managed environment is reported as a `managed venv`, making clear that it is an isolated virtual environment whose Python is owned by sbk-charts.

The static backend registry lets `--help` and core chart generation run without importing optional provider SDKs. Selecting a backend changes the dependency profile and imports only that implementation. Exact, hashed environments live in `requirements-lock/`; human-maintained inputs live in `requirements.txt` and `requirements-ai/`.

Managed source targets cover glibc Linux x86-64/ARM64, macOS Intel/Apple silicon, and Windows x86-64/ARM64. Portable release targets are a smaller independent list.

## 14. Runtime and artifact policy

`sbk-charts.ini` is the shared source of truth for values used by launchers, packaging, CI, and portable builds:

- application and distribution identity;
- Python minimum, exact managed Python, and interpreter search order;
- environment names, runtime state filename, and state schema;
- pinned runtime manager downloads and checksums;
- AI dependency-profile inputs and exact lock directory;
- entry point, requirements file, version file, and package data;
- portable targets, runners, formats, manifest, checksum, and bundle paths.

`scripts/project_policy.py` loads these values into frozen dataclasses, validates cross-field consistency, reads requirements safely, reads the version with Python's AST, emits the CI matrix, and provides runtime-state helpers.

Domain settings remain in their owning modules. CSV headers and percentile groups belong in chart constants, workbook sheet identities belong in sheet constants, visual sizes and colors belong in chart or AI layout code, shared backend defaults belong in `src/ai/defaults.py`, and retrieval scores belong in the RAG algorithm. See [POLICY.md](POLICY.md) for the ownership rules.

## 15. Packaging and portable releases

`setup.py` consumes project policy rather than repeating package name, entry point, requirements path, version path, or package-data values. A wheel installs a normal `sbk-charts` console script.

Portable builds use PyInstaller in one-directory mode as an internal payload, then wrap that payload in one persistent self-extracting application:

```mermaid
flowchart TD
    A[sbk-charts.ini] --> B[Native GitHub runner matrix]
    C[Source and requirements] --> D[PyInstaller onedir build]
    B --> D
    D --> E[Run executable with help]
    E --> F[Copy license README policy and docs]
    F --> G[Create manifest with file hashes]
    G --> H[Compress native payload]
    H --> I[Prepend native self-extracting launcher]
    I --> J[Write external SHA-256 checksum]
```

Supported targets are currently Linux x86-64, macOS Apple silicon, and Windows x86-64. Each application is built on its native GitHub runner. Unix releases are executable `.run` files containing TAR.GZ payloads. Windows releases are native `.exe` extractors containing ZIP payloads. The Linux workflow installs the CPU-only PyTorch wheel to avoid bundling unused CUDA runtimes.

On first execution, the launcher verifies the embedded payload and atomically publishes it under the user's OS cache. State is keyed by schema, version, target, and payload checksum. Later executions validate and reuse that saved payload. Concurrent executions share a short-lived extraction lock, and the Python application is always started after lock release.

Portable applications are checksummed but not code-signed or notarized.

## 16. Tests and CI

`tests/test_portable.py` covers policy parsing and validation, safe requirements parsing, AST version lookup, portable argument forwarding and provenance, self-extractor generation, real Unix first-run extraction and saved reuse, Windows payload generation, environment validation, backward-compatible state schemas, remembered state including profile-only legacy state, and launcher/workflow contracts.

The main CI workflow:

- reads the minimum Python version from policy;
- runs flake8 syntax and undefined-name checks;
- runs the portable-policy unit tests;
- verifies package builds and the Bash launcher on Linux;
- verifies fresh and offline managed bootstrap on Linux, macOS Apple silicon, and Windows;
- asserts first-run creation and second-run saved-state provenance output;
- creates a Windows virtual environment and smoke-tests both Windows launchers;
- verifies that launchers skip a version-compatible Python that cannot create a working venv;
- verifies failed bootstrap downloads and unpublished environments do not leave temporary directories.

The portable workflow:

- reads its native build matrix from policy;
- installs application and portable build requirements;
- reruns tests;
- builds and smoke-tests one self-extracting application;
- executes it twice to prove first-run extraction and saved reuse;
- creates a sample workbook through the saved bundled runtime;
- uploads build artifacts and, for release events, attaches them to the release.

The Excel pipeline still requires an end-to-end sample run because unit tests do not verify chart rendering or all workbook behavior.

## 17. Architectural invariants

Preserve these rules unless a reviewed design explicitly changes them:

1. Workbook stages run in this order: data sheets, charts and Summary, optional AI text.
2. Data sheets are named `R<digits>` and `T<digits>`.
3. Every R sheet has a corresponding T sheet with the same numeric suffix.
4. Compared R sheets use one latency time unit.
5. Exact CSV headers come from `src/charts/constants.py`.
6. Shared analysis prompts live in `SbkGenAI`, not provider plugins.
7. Optional backend imports are lazy and do not stop help or unrelated backends.
8. AI methods return `(bool, text)` and failures remain visible in the workbook.
9. `StorageStat` is constructed completely and then treated as read-only.
10. Summary layout changes account for both chart and AI writers.
11. Runtime and artifact metadata shared across systems belongs in `sbk-charts.ini`.
12. Generated workbooks, environments, models, and build artifacts are not committed.

## 18. Common extension paths

| Goal | Start here | Also inspect |
|---|---|---|
| Add a chart | `src/charts/multicharts.py` | `charts.py`, `constants.py`, `utils.py` |
| Change workbook styling | `src/charts/charts.py` | Summary styling in `multicharts.py` and `sbk_ai.py` |
| Change CSV split behavior | `src/sheets/sheets.py` | `sheets/constants.py`, every R/T caller |
| Add an AI backend | plugin, registry, optional requirements and lock | `genai.py`, plugin spec |
| Change all AI prompts | `src/genai/genai.py` | every backend's response limits |
| Change AI scheduling | `src/ai/sbk_ai.py` | timeout and thread-safety behavior |
| Change chat retrieval | `src/rag/sbk_rag.py` | `genai.py`, `sbk_ai.py` |
| Change bootstrap policy | `sbk-charts.ini` | both launchers, policy tests, docs |
| Add a portable target | `sbk-charts.ini` | workflow, builder, tests, portable docs |

Detailed procedures and acceptance checks are in [AGENT_RECIPES.md](AGENT_RECIPES.md).

## 19. Failure model

| Failure | Expected behavior |
|---|---|
| No managed runtime can be provisioned and no venv-capable Python or usable Conda exists | Launcher exits with an explanation. |
| Environment is stale or dependencies fail validation | Launcher attempts repair or another environment. |
| Input CSV cannot be read | Sheet creation fails; no valid report can be produced. |
| Compared latency units differ | Graph generation is skipped to avoid invalid charts. |
| A selected plugin cannot import | Its command remains visible through the lazy registry; startup reports the backend import failure without importing unrelated plugins. |
| AI credential or service is unavailable | Backend returns readable failure text; charts remain usable. |
| AI exceeds its time budget | Missing results are marked timed out. |
| Chat retrieval finds little context | The backend still receives the question, but its answer may be more general. |
| Portable executable fails its help smoke test | Archive creation fails. |

## 20. Glossary

| Term | Meaning |
|---|---|
| SBK | Storage Benchmark Kit, the upstream benchmark producer. |
| R sheet | Regular per-interval rows for one input CSV. |
| T sheet | Total rows for the matching input CSV. |
| Percentile | A latency boundary, such as p99, below which that percentage of observations falls. |
| Percentile count | The count associated with a latency percentile bucket in SBK output. |
| Backend | An adapter that sends shared prompts to a cloud API, local server, or local model. |
| RAG | Retrieval-augmented generation: adding relevant benchmark facts to an AI prompt. |
| Source launcher | A shell script that finds or creates Python and installs the checkout. |
| Portable application | One native self-extracting release file containing the application, Python, and dependencies. |
| Policy | Shared runtime and artifact metadata stored in `sbk-charts.ini`. |
