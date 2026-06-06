<!--
Copyright (c) KMG. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
-->

# PLUGIN_SPECIFICATION.md — Spec-driven development for new sbk-charts AI plugins

> **Purpose.** This file gives you (a human contributor or an AI coding
> agent driven by a human) a **fillable template** to specify a new
> sbk-charts AI plugin before any code is written. Once the spec is
> complete, the agent turns it into working code by following
> <ref_file file="/root/projects/sbk-charts/docs/AGENT_RECIPES.md" />
> §1 ("Add a new AI plugin").
>
> **The spec is the contract.** Anything the spec says, the code must
> implement. Anything the spec is silent on, the code uses sensible
> defaults from the existing plugin patterns
> (<ref_file file="/root/projects/sbk-charts/src/custom_ai/" />).
>
> Before reading further, agents should have read
> <ref_file file="/root/projects/sbk-charts/AGENTS.md" /> and the
> "Add a new AI plugin" recipe.

---

## Table of contents

1. [How to use this template](#1-how-to-use-this-template)
2. [The spec template (fillable)](#2-the-spec-template-fillable)
3. [Worked example — the Gemini plugin spec](#3-worked-example--the-gemini-plugin-spec)
4. [Acceptance checklist](#4-acceptance-checklist)

---

## 1. How to use this template

### 1.1 Two workflows

**Workflow A — "Vibe coding"** (you know what you want; iterate fast):

1. Copy §2 into a scratch buffer or a new branch's commit message.
2. Fill in the minimum needed: plugin name, target model/service, one
   or two key config flags.
3. Hand to an AI agent with the prompt: *"Implement this plugin
   following AGENT_RECIPES.md §1."*
4. Iterate against the verification checklist in
   <ref_file file="/root/projects/sbk-charts/docs/AGENT_RECIPES.md" />
   §1.4.

**Workflow B — "Spec-driven"** (formal; auditable; multi-person):

1. Copy §2 into `docs/specs/<plugin-name>.spec.md` (or similar; the
   path is up to your team).
2. Fill the **whole** template, including test plan and acceptance
   criteria.
3. Review with the team / commit to the branch.
4. Hand to an agent: *"Implement the spec in
   `docs/specs/<plugin-name>.spec.md`."*
5. Agent generates the plugin, updates `requirements.txt`, and updates
   the comparison table in `docs/ARCHITECTURE.md` §5.9.
6. End-to-end CLI verification (Recipes §1.3 Step E + §1.4) and the
   acceptance checklist (§4 below) gate the merge.
7. The spec stays in the repo as the source of truth for the feature.

### 1.2 Template conventions

- **Bold field names** are required.
- *Italic field names* are optional; agents will use sensible defaults
  if you leave them blank.
- `code-style` field values are taken verbatim into code or config.
- Sections with `→` show explicit cross-references to the Python file
  the agent will produce.

---

## 2. The spec template (fillable)

> Copy everything inside the fenced block to a new file and fill it in.
> Anything in `<…>` is a placeholder.

```markdown
# Plugin spec — <Display Name>

## 0. Metadata

- **Plugin display name**: <e.g. "Mistral", "Azure OpenAI">
- **Subcommand on the CLI**: `<lowercase, e.g. mistral, azureopenai>`
- **Directory name** under `src/custom_ai/`: `<snake_case, e.g. mistral, azure_openai>`
- **Class name** inside the plugin: `<PascalCase, e.g. Mistral, AzureOpenAI>`
- **One-line summary**: <e.g. "Cloud API for Mistral's instruct models">
- *Author / contact*: <name or handle>
- *Spec status*: draft | reviewed | implemented

## 1. Hosting model

Pick ONE:

- [ ] Cloud SaaS API (e.g. Gemini, Anthropic)
- [ ] Local HTTP server the user runs (e.g. Ollama, LM Studio)
- [ ] In-process model load (e.g. PyTorchLLM)
- [ ] Other: <describe>

This determines which existing plugin you should copy the structure
from (see <ref_file file="/root/projects/sbk-charts/docs/AGENT_RECIPES.md" /> §1.1).

## 2. SDK / transport

- **Python library name** (PyPI): `<e.g. mistralai>`
- **Library version pin** (`~=`): `<e.g. ~=1.5.0>`
- **Import path inside the library** that the plugin uses:
  `<e.g. from mistralai import Mistral as MistralClient>`
- *Underlying protocol* (only for local servers): <REST / gRPC / SDK-managed>
- *Server URL* (only for local servers, with default): `<e.g. http://localhost:1234>`

## 3. Authentication

Pick ONE:

- [ ] Environment variable: `<NAME>` (must be a single var the user sets)
- [ ] None (purely local)
- [ ] Other: <describe>

→ The plugin's `__init__` reads this in `os.getenv("<NAME>")`.

## 4. Configuration surface (CLI flags)

List every flag the plugin will register on its subparser. Use the
`--<plugin>-<field>` prefix convention.

| Flag | Type | Default | What it controls |
|---|---|---|---|
| `--<plugin>-model` | str | `<e.g. mistral-large-latest>` | Model identifier |
| `--<plugin>-temperature` | float | `0.4` | Sampling temperature |
| `--<plugin>-max-tokens` | int | `2048` | Max response tokens |
| *…optional, list more if needed…* | | | |

> The default temperature **must** be `0.4` unless you have a justified
> reason — that is the project's chosen value (see ARCHITECTURE.md
> §5.9). Justify any deviation in the spec.

## 5. Model description (`get_model_description`)

Multi-line string returned to the Summary sheet header. Template:

```
<API or transport name>
 Model: <model id>
 Temperature: <temp>
 Max Tokens: <max>
```

If auth fails: return `(False, "<plugin> analysis is not available (missing <ENV VAR>)")`.

## 6. Analysis behaviour

For each of the four canonical analyses, declare whether the plugin
uses the framework-provided prompt or a custom one. The default and
strongly-preferred choice is "framework".

| Analysis | Source of prompt | Notes |
|---|---|---|
| `get_throughput_analysis` | framework `get_throughput_prompt()` | |
| `get_latency_analysis` | framework `get_latency_prompt()` | |
| `get_total_mb_analysis` | framework `get_total_mb_prompt()` | |
| `get_percentile_histogram_analysis` | framework `get_percentile_histogram_prompt()` | |
| `get_response(query)` | persona prompt + `_enhance_prompt_with_rag(prompt, query)` | for chat mode |

> If any row says "custom", the spec must include the full custom
> prompt as a code-style block. Reviewer must explicitly approve.

## 7. Error handling

The plugin's `get_*_analysis()` methods must follow this contract:

- Return `(True, <text>)` on success.
- Return `(False, <human-readable error>)` on failure.
- **Never raise** to the orchestrator. Wrap your SDK call in `try/except`.
- Network errors, timeouts, auth failures, and parse errors all result
  in `(False, <message>)` — different messages for different causes.

Declare known failure modes specific to this plugin (e.g. "Mistral
returns 429 on free tier above 1 RPS; we surface that as 'rate limited
— retry later'").

## 8. Threading

Declare one of:

- [ ] Thread-safe — works under the default `ThreadPoolExecutor(4)`.
- [ ] **Not** thread-safe — users must pass `-nothreads`. (Common for
  in-process models on a single GPU.)

If "not thread-safe", document this in the plugin's docstring so
`sbk-charts -h <plugin>` users see it.

## 9. Resource cleanup

What does `close(args)` do?

- [ ] Nothing (no resources to close)
- [ ] Closes an HTTP session
- [ ] Frees model memory / GPU memory
- [ ] Other: <describe>

## 10. Test plan

Minimal test plan to consider the plugin "done":

1. Discovery — `sbk-charts -h` lists the new subcommand.
2. Help — `sbk-charts -i x.csv <plugin> -h` lists all the flags from §4.
3. Auth-missing path — without the env var (§3), all four analyses
   return `(False, ...)` and the workbook still saves.
4. Happy path — with valid auth, all four analyses return `(True, ...)`
   within the default 120 s budget against the sample CSV
   <ref_file file="/root/projects/sbk-charts/samples/charts/sbk-file-read.csv" />.
5. Chat mode — `-chat` followed by a free-form query produces a
   coherent answer that references the loaded benchmark data.
6. *Optional* — multi-CSV run with `-i a.csv,b.csv` mentions both
   storage drivers by name.

## 11. Out of scope

Things this plugin will **not** do (record explicitly so future
reviewers don't expect them):

- <e.g. "No streaming responses — the framework collects full text.">
- <e.g. "No multimodal input — prompts are text-only today.">
- <e.g. "No retries on 5xx — leave to the SDK.">

## 12. Open questions

- <e.g. "Is there a free tier model id worth defaulting to?">
- <e.g. "Should we expose a region / endpoint flag for enterprise tenants?">

```

---

## 3. Worked example — the Gemini plugin spec

This is the spec as it *would* read for the Gemini plugin that ships
today in <ref_file file="/root/projects/sbk-charts/src/custom_ai/gemini/gemini.py" />.
Use it as a complete reference of "what good looks like".

```markdown
# Plugin spec — Gemini

## 0. Metadata

- Plugin display name: Gemini
- Subcommand on the CLI: `gemini`
- Directory name: `gemini`
- Class name: `Gemini`
- One-line summary: Google Gemini cloud API via google-genai SDK.
- Spec status: implemented

## 1. Hosting model

- [x] Cloud SaaS API

## 2. SDK / transport

- Python library: `google-genai`
- Version pin: `~=1.62.0`
- Import path: `import google.genai as genai`
- Underlying protocol: SDK-managed REST

## 3. Authentication

- [x] Environment variable: `GEMINI_API_KEY`

## 4. Configuration surface

| Flag | Type | Default | What it controls |
|---|---|---|---|
| `--gemini-model` | str | `gemini-2.5-flash` | Model identifier |
| `--gemini-temperature` | float | `0.4` | Sampling temperature |
| `--gemini-max-tokens` | int | `2048` | Max output tokens |

## 5. Model description

```
Google Gemini API
 Model: gemini-2.5-flash
 Temperature: 0.4
 Max Tokens: 2048
```

On missing API key: `(False, "Google AI API key not found. Please set GEMINI_API_KEY environment variable.")`.

## 6. Analysis behaviour

All four analyses use the framework prompts. `get_response(query)`
builds a persona prompt and enhances it with
`_enhance_prompt_with_rag(prompt, query)`.

## 7. Error handling

- API errors → `(False, "Error calling Gemini API: <e>")`.
- Empty content → `(False, "No content in response")`.
- No API key → `(False, "Gemini analysis is not available (missing GEMINI_API_KEY...).")`.
- Never raises to the orchestrator.

## 8. Threading

- [x] Thread-safe — the `google.genai.Client` is thread-safe under the
  Python SDK contract.

## 9. Resource cleanup

`close(args)` calls `self._client.close()` if a client was created.

## 10. Test plan

All six items in §10 of the template apply.

## 11. Out of scope

- No streaming.
- No multimodal input today (text only).
- No Vertex AI / GCP credentials path — only `GEMINI_API_KEY`.

## 12. Open questions

- Resolved during implementation: the `google-genai` 1.62+ Client API
  replaced the deprecated `google.ai.generativelanguage` path.
```

---

## 4. Acceptance checklist

A plugin built from a spec is "done" when **all** of the following are
true:

### 4.1 Functional

- [ ] Plugin appears in `sbk-charts -h`.
- [ ] All flags from §4 appear in `sbk-charts -i x.csv <plugin> -h`.
- [ ] With valid auth (or no auth, for local plugins), all four
  analyses return `(True, <text>)` against
  `samples/charts/sbk-file-read.csv` within the default 120 s budget.
- [ ] Without auth, the four analyses return `(False, <clear message>)`
  and the workbook still saves cleanly.
- [ ] Chat mode produces a sensible response that references the
  loaded benchmark data (verified by asking
  *"Which storage has the worst p99 latency?"* — answer should name
  the storage from the sample data).

### 4.2 Code quality

- [ ] `requirements.txt` lists the new dep with `~=` pin.
- [ ] No code change leaks outside `src/custom_ai/<plugin>/`,
  `requirements.txt`, and the comparison table in `docs/ARCHITECTURE.md`
  §5.9.
- [ ] Plugin file has the Apache 2.0 copyright header (copy from any
  existing plugin).
- [ ] No emoji introduced in code or comments.
- [ ] Constants for `DEFAULT_MODEL`, `DEFAULT_MAX_TOKENS`,
  `DEFAULT_TEMPERATURE` declared at module top.
- [ ] No prompt strings live in the plugin (except the chat persona
  in `get_response`).
- [ ] All six abstract methods of `SbkGenAI` are implemented:
  `get_model_description`, `get_throughput_analysis`,
  `get_latency_analysis`, `get_total_mb_analysis`,
  `get_percentile_histogram_analysis`, `get_response`.

### 4.3 Documentation

- [ ] Plugin row added to the comparison table in
  `docs/ARCHITECTURE.md` §5.9 (library / env var / default model /
  max tokens / temperature / notes).
- [ ] If §11 of the spec lists out-of-scope items, those are
  paraphrased in the plugin's class docstring.
- [ ] If §8 says "not thread-safe", the class docstring **and** the
  `add_args` help text both surface this.

### 4.4 Reproducibility (Workflow B only)

- [ ] The completed spec is committed to `docs/specs/<plugin>.spec.md`
  (or wherever your team agreed).
- [ ] The PR description links to the spec.

---

*If you find yourself unable to fit a plugin into this template, that
is a signal that the SPI itself may need to evolve. Raise the gap
explicitly — do not stretch the template silently. Edits to
`SbkGenAI` itself are out of scope for a single-plugin PR.*
