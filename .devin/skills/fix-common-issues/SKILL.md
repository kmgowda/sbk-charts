# Diagnose common sbk-charts problems

Use this skill for missing plugins, bootstrap failures, chart failures, AI timeouts, and packaging errors. Diagnose before editing.

## Selected backend fails to import

```bash
venv-sbk-charts/bin/python -c \
  "from src.ai.registry import load_backend_class; print(load_backend_class('<backend>'))"
venv-sbk-charts/bin/python -c \
  "import src.custom_ai.<directory>.<module>"
```

Help uses the lazy registry and should list every declared backend without its SDK installed. Fix the selected profile, changed SDK import, or code error.

## Launcher selects or repairs an environment repeatedly

Inspect `.sbk-charts-runtime`, `.sbk-runtime/`, `SBK_CHARTS_VENV`, `SBK_CHARTS_CONDA_ENV`, and launcher overrides. A managed environment also requires the current profile and lock fingerprint. Test first-run creation and a second run with network access disabled.

Run `./sbk-charts -h` and read the printed OS, interpreter, environment kind, prefix, profile, selection source, saved-state reuse flag, and creation flag. Do not remove validation merely to hide a real dependency conflict.

If legacy fallback selects no Python, test every configured candidate instead of stopping at the first version-compatible executable. Each candidate must create a temporary venv with working `ensurepip` and `pip`. Failed probes, `.tool.*` downloads, and unpublished `.env-*` directories should be removed automatically. On Windows, also verify the CMD shim remembers a backend profile when the legacy environment has no managed fingerprint.

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
