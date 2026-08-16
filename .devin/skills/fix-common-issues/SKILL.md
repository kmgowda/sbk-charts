# Diagnose common sbk-charts problems

Use this skill for missing plugins, bootstrap failures, chart failures, AI timeouts, and packaging errors. Diagnose before editing.

## Backend missing from help

```bash
venv-sbk-charts/bin/python -c \
  "from src.ai.discover import discover_custom_ai_classes; print(discover_custom_ai_classes())"
venv-sbk-charts/bin/python -c \
  "import src.custom_ai.<directory>.<module>"
```

Fix the missing dependency, changed SDK import, or code error. Discovery intentionally skips only the broken backend.

## Launcher selects or repairs an environment repeatedly

Inspect `.sbk-charts-runtime`, `SBK_CHARTS_VENV`, `SBK_CHARTS_CONDA_ENV`, and `SBK_CHARTS_STATE_FILE`. A remembered environment is reused only when Python is supported, the installed distribution version matches source, the application imports, and dependency validation passes.

Run `./sbk-charts -h` and read the printed OS, interpreter, environment kind, and prefix. Do not remove validation merely to hide a real dependency conflict.

## Charts are missing

Check that the workbook contains matching `R<digits>` and `T<digits>` sheets, required exact headers, and one latency time unit across compared R sheets. Use `get_columns_from_worksheet()` rather than assumed column positions.

## AI analysis times out or exhausts memory

```bash
./sbk-charts -i input.csv -o /tmp/analysis.xlsx \
  -secs 600 -nothreads <backend>
```

Check provider/service health. For PyTorch, choose a smaller model and confirm device, RAM, accelerator memory, disk, and wheel compatibility.

## Logo or banner missing after installation

Check `sbk-charts.ini` package data, `setup.py`, `MANIFEST.in`, and file paths relative to module `__file__`. Build wheel and sdist, inspect their contents, then test a fresh installation from outside the repository.

## RAG gives generic context

Confirm storage statistics exist and ingestion produced documents. All-zero metrics are intentionally skipped. Test retrieval output directly before blaming the model. The default application uses Simple RAG, not ChromaDB.

## Reporting

Record the exact command, OS, Python executable/version, environment kind/path, complete error, and smallest reproducible input. State whether the issue is code, dependency availability, credential/service state, hardware capacity, or an untested platform.
