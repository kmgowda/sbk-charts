<!--
Copyright (c) KMG. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
-->

# Hugging Face backend

The `huggingface` command uses `huggingface_hub.InferenceClient` and its chat-completion API.

## Setup

Create a token with the access needed for the selected inference model, then export it:

```bash
export HUGGINGFACE_API_TOKEN="your-token"
```

On PowerShell:

```powershell
$env:HUGGINGFACE_API_TOKEN = "your-token"
```

The exact variable is `HUGGINGFACE_API_TOKEN`.

## Run

```bash
./sbk-charts -i input.csv -o huggingface-report.xlsx huggingface
```

Select another model:

```bash
./sbk-charts -i input.csv -o huggingface-report.xlsx \
  huggingface --model_id meta-llama/Llama-3.1-8B-Instruct
```

| Flag | Code default |
|---|---|
| `-id`, `--model_id` | `meta-llama/Llama-3.1-8B-Instruct` |

The adapter requests up to 5000 response tokens with temperature 0.4 and top-p 0.9. Those values are currently code defaults rather than CLI options.

## How it works

Each analysis creates an `InferenceClient` for the selected model and token, sends one shared prompt through `chat_completion()`, and reads the first choice. Chat adds Simple RAG context first.

## Troubleshooting

- Missing token: check the exact environment-variable spelling.
- Access denied or gated model: accept the model's terms or select a model your token can access.
- Model does not support chat completion: choose a compatible instruct/chat model.
- Backend import failure after selection: reinstall `huggingface_hub` through the backend requirements and run the direct-load command in the parent [AI backend guide](../README.md).

Benchmark prompt data is sent to Hugging Face or the inference provider serving the chosen model. Review current provider policy before use.
