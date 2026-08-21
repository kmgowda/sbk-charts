<!--
Copyright (c) KMG. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
-->

# LM Studio backend

The `lmstudio` command sends analysis requests to a model managed by LM Studio. It uses the `lmstudio` Python SDK and does not need a cloud API key.

## Setup

1. Install and start LM Studio.
2. Download and load a model that can follow chat instructions.
3. Start LM Studio's local server.
4. Confirm the server is reachable from the machine running sbk-charts.

The code default endpoint is `http://localhost:1234/api/v0`.

## Run

Use the model selected in LM Studio:

```bash
./sbk-charts -i input.csv -o lmstudio-report.xlsx -nothreads lmstudio
```

Request a named model and change generation settings:

```bash
./sbk-charts -i input.csv -o lmstudio-report.xlsx -secs 600 -nothreads \
  lmstudio \
  --url http://localhost:1234/api/v0 \
  --lm-model <loaded-model-name> \
  --lm-temperature 0.4 \
  --lm-max-tokens 1800
```

| Flag | Code default |
|---|---|
| `--url` | `http://localhost:1234/api/v0` |
| `--lm-model` | Empty, meaning LM Studio's selected model |
| `--lm-temperature` | `0.4` |
| `--lm-max-tokens` | `1800` |

`-nothreads` is recommended when the loaded model or server does not handle four simultaneous analyses reliably. The flag takes no value.

## How it works

`open()` asks the SDK for the selected or named model handle. Each analysis calls `respond()` with the shared prompt and generation configuration. `close()` drops the model reference. Chat adds retrieved workbook facts before calling the model.

## Troubleshooting

- Connection failure: start the LM Studio server and verify its port.
- Model error: load the model or make `--lm-model` match LM Studio's identifier.
- Inconsistent parallel responses: add `-nothreads` before `lmstudio`.
- Timeout: increase `-secs` and inspect server/model performance.

If LM Studio is on another machine, benchmark prompt data travels to that machine even though this is called a local backend.
