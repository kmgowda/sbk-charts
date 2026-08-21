<!--
Copyright (c) KMG. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
-->

# Anthropic backend

The `anthropic` command sends the four shared SBK analysis prompts to Anthropic through the `anthropic` Python SDK.

## Setup

Set the credential in the environment:

```bash
export ANTHROPIC_API_KEY="your-key"
```

On PowerShell:

```powershell
$env:ANTHROPIC_API_KEY = "your-key"
```

Benchmark-derived prompt data is sent to Anthropic. Review Anthropic's current data, region, and billing policies before use.

## Run

```bash
./sbk-charts -i input.csv -o anthropic-report.xlsx anthropic
```

Choose configuration explicitly:

```bash
./sbk-charts -i input.csv -o anthropic-report.xlsx -secs 300 \
  anthropic \
  --anthropic-model anthropic-sonnet-4-20250514 \
  --anthropic-max-tokens 4096 \
  --anthropic-temperature 0.4
```

| Flag | Code default |
|---|---|
| `--anthropic-model` | `anthropic-sonnet-4-20250514` |
| `--anthropic-max-tokens` | `2048` |
| `--anthropic-temperature` | `0.4` |

Provider model availability changes. If the default is unavailable to your account, pass an available model ID with `--anthropic-model`.

## How it works

`Anthropic` subclasses `SbkGenAI`. `open()` creates the client when a key exists. Each analysis calls `client.messages.create()` and returns the first text block. `close()` closes the client. Chat adds retrieved benchmark context before making the same style of request.

## Troubleshooting

- Missing-key messages: confirm the variable exists in the same shell that starts sbk-charts.
- Authentication or model errors: verify the key, account access, and exact model ID.
- Rate or timeout errors: increase `-secs` when the provider call is slow, but also check provider limits.
- Backend import failure after selection: reinstall dependencies and run the direct-load command in the parent [AI backend guide](../README.md).

Do not paste keys into issues, workbooks, logs, or committed files.
