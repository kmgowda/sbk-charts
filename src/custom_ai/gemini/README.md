# Gemini backend

The `gemini` command sends the four shared SBK analysis prompts to Google's Gemini API through the `google-genai` SDK.

## Setup

Set the API key:

```bash
export GEMINI_API_KEY="your-key"
```

On PowerShell:

```powershell
$env:GEMINI_API_KEY = "your-key"
```

Benchmark-derived prompt data is sent to Google. Review Google's current data, region, quota, and billing policies before use.

## Run

```bash
./sbk-charts -i input.csv -o gemini-report.xlsx gemini
```

```bash
./sbk-charts -i input.csv -o gemini-report.xlsx -secs 300 \
  gemini \
  --gemini-model gemini-2.5-flash \
  --gemini-max-tokens 4096 \
  --gemini-temperature 0.4
```

| Flag | Code default |
|---|---|
| `--gemini-model` | `gemini-2.5-flash` |
| `--gemini-max-tokens` | `2048` |
| `--gemini-temperature` | `0.4` |

Model availability is controlled by Google and may vary by account or region. Override the default when necessary.

## How it works

`Gemini` subclasses `SbkGenAI` and creates `google.genai.Client` when a key exists. Analysis uses `client.models.generate_content()` with the selected model and generation configuration. Chat uses the same client after adding retrieved benchmark context. The client is closed during backend cleanup.

## Troubleshooting

- Missing-key message: export `GEMINI_API_KEY` in the launching shell.
- Model unavailable: pass a model available to your account with `--gemini-model`.
- Quota or network failure: inspect the returned provider error and current account quota.
- Import error for `google.genai`: reinstall the project requirements in the selected launcher environment.
- Backend import failure after selection: run the direct-load command in the parent [AI backend guide](../README.md).

Do not enable debug output that prints credentials or complete sensitive benchmark prompts.
