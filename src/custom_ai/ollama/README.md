# Ollama backend

The `ollama` command sends shared SBK analysis prompts to an Ollama server over HTTP. The server can run on the same computer or on a reachable network host.

## Setup

Install Ollama, download a model, and start the service:

```bash
ollama pull llama3.1
ollama serve
```

Confirm the API responds:

```bash
curl http://localhost:11434/api/tags
```

## Run

```bash
./sbk-charts -i input.csv -o ollama-report.xlsx ollama
```

```bash
./sbk-charts -i input.csv -o ollama-report.xlsx -secs 600 \
  ollama \
  --ollama-url http://localhost:11434 \
  --ollama-model llama3.1 \
  --ollama-temperature 0.4 \
  --ollama-timeout 120
```

| Flag | Code default |
|---|---|
| `-url`, `--ollama-url` | `http://localhost:11434` |
| `-model`, `--ollama-model` | `llama3.1` |
| `-tmp`, `--ollama-temperature` | `0.4` |
| `-timeout`, `--ollama-timeout` | `120` seconds per HTTP request |

The global `-secs` value is the total budget for all four analyses. `--ollama-timeout` applies to one HTTP request.

## How it works

The adapter checks `/api/tags` for service health and sends chat requests to `/api/chat`. It reads the assistant message from the JSON response. Chat mode adds retrieved benchmark measurements to the prompt.

## Troubleshooting

- Connection refused: run `ollama serve` and verify the URL.
- Model not found: run `ollama pull <model>` and use the same name in `--ollama-model`.
- Slow or timed-out calls: choose a smaller model, increase both relevant timeouts, or add `-nothreads`.
- Out of memory: choose a smaller or more strongly quantized model.
- Remote endpoint: check firewall and routing, and remember benchmark prompt data leaves the local machine.

The implementation uses the `requests` package; it does not require the Ollama Python client.
