<!--
Copyright (c) KMG. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
-->

# AI backends

sbk-charts can create the workbook without AI. If a backend command is selected, it also writes four plain-English analyses to the Summary sheet and can start interactive chat.

All backends receive the same throughput, latency, total-MB, and percentile-histogram prompts from `src/genai/genai.py`. A backend's job is to configure a model, send those prompts, convert the response to text, and return `(success, text)`.

Canonical model, endpoint, and generation defaults live in `src/ai/defaults.py`. The lightweight registry and the selected adapter both consume those values, so changing a default does not require duplicate edits.

## Choose a backend

| Command | Runs | Setup | Default |
|---|---|---|---|
| `anthropic` | Anthropic cloud | `ANTHROPIC_API_KEY` | `anthropic-sonnet-4-20250514` |
| `gemini` | Google cloud | `GEMINI_API_KEY` | `gemini-2.5-flash` |
| `huggingface` | Hugging Face inference service | `HUGGINGFACE_API_TOKEN` | `meta-llama/Llama-3.1-8B-Instruct` |
| `lmstudio` | Local LM Studio process | Start LM Studio and load a model | Server-selected model |
| `ollama` | Local or remote Ollama server | Start Ollama and pull a model | `llama3.1` |
| `pytorchllm` | Current Python process | Model files and sufficient compute/memory | `openai/gpt-oss-20b` |
| `noai` | Current process | None | Placeholder failure text |

Use a cloud backend for simple setup when benchmark data may be sent to that provider. Use Ollama or LM Studio when you want a locally served model. Use PyTorchLLM only when the selected model fits your machine; the default model is large.

## Common syntax

```text
./sbk-charts -i <input.csv> -o <output.xlsx> \
  [-secs <seconds>] [-nothreads] [-chat] \
  <backend> [backend options]
```

Global flags must appear before the backend. Backend flags appear after it.

```bash
./sbk-charts -i input.csv -o report.xlsx -secs 300 -nothreads \
  ollama --ollama-model llama3.1
```

Show current commands:

```bash
./sbk-charts -h
```

Show one backend's current options:

```bash
./sbk-charts -i samples/charts/sbk-file-read.csv <backend> -h
```

## Backend flags

| Backend | Flags |
|---|---|
| Anthropic | `--anthropic-model`, `--anthropic-max-tokens`, `--anthropic-temperature` |
| Gemini | `--gemini-model`, `--gemini-max-tokens`, `--gemini-temperature` |
| Hugging Face | `-id`, `--model_id` |
| LM Studio | `--url`, `--lm-model`, `--lm-temperature`, `--lm-max-tokens` |
| Ollama | `-url`/`--ollama-url`, `-model`/`--ollama-model`, `-tmp`/`--ollama-temperature`, `-timeout`/`--ollama-timeout` |
| PyTorchLLM | `--pt-model`, `--pt-train`, `--pt-device`, `--pt-max-length`, `--pt-temperature`, `--pt-top-p` |
| NoAI | No backend-specific flags |

`-secs` is the orchestrator's total four-analysis budget. `--ollama-timeout` is the timeout for an individual Ollama HTTP request. They are different controls.

## Chat

Add `-chat` before the backend command:

```bash
./sbk-charts -i input.csv -o report.xlsx -chat gemini
```

Chat initializes the in-memory Simple RAG pipeline from workbook statistics. Retrieved benchmark facts are added to the question before the selected backend is called. Press Enter on an empty line to submit a question and Control-D to exit.

## Threading

The four analyses run concurrently by default. Use `-nothreads` for a local service that behaves inconsistently under simultaneous requests, for step-by-step debugging, or for an in-process model on one GPU.

```bash
./sbk-charts -i input.csv -o report.xlsx -nothreads lmstudio
```

`-nothreads` is a flag and takes no `true` or `false` value.

## Data and security

Cloud backends send prompt text derived from benchmark measurements to their provider. Review the provider's privacy, retention, region, and billing rules before use. Never place API keys in a command committed to source control. Use environment variables or your organization's secret manager.

Local backends avoid a cloud provider only when the server and model are actually local. A remotely hosted Ollama or LM Studio endpoint still sends data over the network.

## Failure behavior

If no backend command is selected, chart creation completes and the program prints that AI is disabled. If a selected backend lacks a key, service, model, or resource, each analysis should return readable failure text and the workbook should still be saved.

Help is generated from lightweight descriptors and does not import backend implementations. If a selected backend fails to import, load that backend directly to see the original error:

```bash
venv-sbk-charts/bin/python -c \
  "from src.ai.registry import load_backend_class; print(load_backend_class('<backend>'))"
```

## Detailed guides

- [Anthropic](anthropic/README.md)
- [Gemini](gemini/README.md)
- [Hugging Face](hugging_face/README.md)
- [LM Studio](lm_studio/README.md)
- [Ollama](ollama/README.md)
- [PyTorchLLM](pytorch_llm/README.md)
- [NoAI](no_ai/README.md)

To design another backend, use [the plugin specification](../../docs/PLUGIN_SPECIFICATION.md) and [the contributor recipe](../../docs/AGENT_RECIPES.md#1-add-an-ai-backend).
