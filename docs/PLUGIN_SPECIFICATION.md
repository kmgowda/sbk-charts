<!--
Copyright (c) KMG. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
-->

# AI backend specification template

Use this template before adding a substantial AI backend. It gives humans and software agents an agreed contract for command names, dependencies, authentication, model behavior, failures, threading, documentation, and tests.

Read [AGENTS.md](../AGENTS.md), the [architecture AI section](ARCHITECTURE.md#10-ai-plugin-system), and the [add-backend recipe](AGENT_RECIPES.md#1-add-an-ai-backend) first.

## How to use the template

For a small experiment, fill in the essential fields in a branch description. For reviewed work, copy the template into `docs/specs/<plugin-name>.md`, complete it, and keep it with the implementation.

The plugin implementation normally changes:

```text
src/custom_ai/<directory>/__init__.py
src/custom_ai/<directory>/<directory>.py
src/custom_ai/<directory>/README.md
requirements.txt
src/custom_ai/README.md
docs/ARCHITECTURE.md
```

Plugin arguments belong in the plugin's `add_args()` and `parse_args()` methods. No central registration edit is needed because discovery is automatic.

## Fillable template

````markdown
# Backend specification: <display name>

## 1. Identity

- Display name: <human-readable name>
- Directory: `<lower_snake_case>`
- Module: `<lower_snake_case>.py`
- Class: `<PascalCase>`
- CLI command: `<lowercase class name>`
- One-sentence purpose: <what service or runtime it connects to>
- Status: draft | reviewed | implemented

## 2. Hosting and transport

- Hosting: cloud API | local HTTP service | local SDK service | in-process model
- Python package: `<PyPI name>`
- Compatible-release pin: `<package>~=X.Y.Z`
- Import used by the module: `<Python import>`
- Default endpoint, if any: `<URL or none>`
- Internet required: yes | no | only for model download

Choose the closest implementation pattern:

- Cloud SDK: `gemini` or `anthropic`
- Local HTTP: `ollama`
- Local service SDK: `lm_studio`
- In-process model: `pytorch_llm`

## 3. Authentication

- Method: environment variable | local service | none | other
- Environment variable: `<NAME or none>`
- Missing-auth message: `<plain action the user can take>`
- Secret-handling notes: <never print or persist the secret>

## 4. CLI options

Global options such as `-secs`, `-nothreads`, and `-chat` already exist. List plugin-only options here.

| Flag | Type | Default | Validation | Purpose |
|---|---|---|---|---|
| `--<plugin>-model` | string | `<model>` | non-empty | Provider model ID |
| `--<plugin>-temperature` | float | `0.4` | provider range | Response sampling |
| `--<plugin>-max-tokens` | integer | `2048` | positive | Response limit |

Example final command:

```bash
./sbk-charts -i input.csv -o output.xlsx -secs 300 \
  <plugin> --<plugin>-model <model>
```

## 5. Lifecycle

- Work performed in `__init__`: <keep network/model work minimal>
- Work performed in `open(args)`: <client, session, or model setup>
- Work performed in `close(args)`: <session close or memory cleanup>
- Reusable across four calls: yes | no

## 6. Model description

Define the successful text written near the AI analysis in Summary:

```text
<provider or runtime>
Model: <model>
Temperature: <value>
Maximum tokens: <value>
```

Define the failure result when configuration is unavailable:

```text
(False, "<clear configuration message>")
```

## 7. Analysis behavior

The default is to use all framework prompts.

| Method | Prompt source | Provider call | Expected result |
|---|---|---|---|
| `get_throughput_analysis()` | `get_throughput_prompt()` | <call> | `(bool, text)` |
| `get_latency_analysis()` | `get_latency_prompt()` | <call> | `(bool, text)` |
| `get_total_mb_analysis()` | `get_total_mb_prompt()` | <call> | `(bool, text)` |
| `get_percentile_histogram_analysis()` | `get_percentile_histogram_prompt()` | <call> | `(bool, text)` |
| `get_response(query)` | chat persona plus RAG context | <call> | `(bool, text)` |

If a custom analysis prompt is required, include the requirement and obtain review approval. Do not silently duplicate and fork the shared prompt.

## 8. Response conversion

- Provider response field containing text: `<field/path>`
- Empty-response behavior: `<failure message>`
- Streaming: unsupported | collected into one string | other
- Formatting cleanup: <whitespace, Markdown, or none>

## 9. Errors and retries

List expected failures and the exact category of message returned to the user.

| Failure | Returned behavior | Retry behavior |
|---|---|---|
| Missing credential | `(False, actionable text)` | none |
| Authentication rejected | `(False, provider-safe text)` | none |
| Rate limit | `(False, retry-later text)` | SDK/default/custom |
| Network timeout | `(False, timeout text)` | SDK/default/custom |
| Model not found | `(False, model text)` | none |
| Empty response | `(False, empty-response text)` | none |

Do not expose credentials in exceptions or logs. Expected provider failures should not escape to `SbkAI` as uncaught exceptions.

## 10. Concurrency and resources

- Safe for four parallel analysis calls: yes | no | unknown
- Recommended `-nothreads`: yes | no
- CPU requirement: <estimate>
- RAM requirement: <estimate>
- GPU requirement: <estimate or none>
- Model disk requirement: <estimate or none>
- Provider quota or cost note: <link to provider docs rather than hard-coded prices>

## 11. Chat and RAG

- Chat supported: yes | no
- Calls `_enhance_prompt_with_rag`: yes | no
- Behavior when retrieval returns no context: <describe>
- Example grounded question: `<question that names a metric>`

## 12. Security and privacy

- Benchmark data leaves the machine: yes | no
- Destination service: <provider or none>
- Data-retention documentation: <official link or unknown>
- Proxy/TLS configuration: <SDK behavior>
- Sensitive fields that must not be logged: <list>

## 13. Documentation

- [ ] Add backend row to `src/custom_ai/README.md`.
- [ ] Add focused `src/custom_ai/<directory>/README.md`.
- [ ] Add backend row to `docs/ARCHITECTURE.md`.
- [ ] Include setup, exact flags, examples, data/privacy note, resource note, and troubleshooting.
- [ ] Avoid claims about current model availability or price unless linked and dated.

## 14. Verification

- [ ] Discovery lists the command.
- [ ] Backend help lists every flag.
- [ ] Missing configuration fails clearly and workbook saving completes.
- [ ] Valid configuration completes all four analyses on the sample CSV.
- [ ] Multi-input analysis mentions all expected runs.
- [ ] Chat answer uses retrieved benchmark context.
- [ ] Parallel mode tested, or `-nothreads` documented and tested.
- [ ] Dependency is present in a clean installation.
- [ ] No secret appears in output, workbook, or git diff.

Record exact commands, operating system, Python version, model, elapsed time, and any untested platform.

## 15. Out of scope and open questions

- Out of scope: <explicit exclusions>
- Open question: <decision still needed>
- Follow-up: <future improvement that does not block this implementation>
````

## Worked example: Gemini metadata

This short example shows how current code maps to the template. It is not a replacement for a complete reviewed specification.

| Field | Gemini value |
|---|---|
| Directory/module/class | `gemini` / `gemini.py` / `Gemini` |
| CLI command | `gemini` |
| Hosting | Cloud API through `google-genai` |
| Authentication | `GEMINI_API_KEY` |
| Default model | `gemini-2.5-flash` |
| Plugin flags | `--gemini-model`, `--gemini-max-tokens`, `--gemini-temperature` |
| Prompt ownership | All four standard prompts come from `SbkGenAI` |
| Chat | Adds Simple RAG context through `_enhance_prompt_with_rag()` |
| Result contract | `(True, response_text)` or `(False, readable_error)` |

Verification examples:

```bash
./sbk-charts -h
./sbk-charts -i samples/charts/sbk-file-read.csv gemini -h
read -r -s -p "GEMINI_API_KEY: " GEMINI_API_KEY
printf '\n'
export GEMINI_API_KEY
./sbk-charts \
  -i samples/charts/sbk-file-read.csv \
  -o /tmp/gemini.xlsx \
  gemini --gemini-model gemini-2.5-flash
```

## Review checklist

A reviewer should be able to answer these questions from the spec:

1. What command will users type?
2. What must they install or start?
3. Where does benchmark data go?
4. Which credential is required and how is it protected?
5. What are the exact defaults and flags?
6. Does the backend tolerate four concurrent calls?
7. How do expected errors appear in the workbook?
8. Does chat use benchmark context?
9. What resources, quota, or cost should a user expect?
10. Which end-to-end checks prove the implementation?

If the backend cannot fit this contract, propose an interface change separately. Do not hide a framework change inside one provider adapter.
