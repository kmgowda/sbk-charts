# Fix Common Issues

## Overview
This skill provides solutions to common issues encountered when working with sbk-charts.

## When to use this skill
Use this skill when:
- Encountering import errors
- Plugin not appearing in help output
- Chart generation failures
- AI analysis not working
- Build/packaging issues

## Common Issues and Solutions

### Plugin Not Appearing in Help Output

**Symptom:**
```bash
sbk-charts -h
# Plugin is missing from the list
```

**Cause:** Import error during plugin discovery. The discoverer silently swallows ImportError.

**Solution:**
```bash
python3 -c "from src.ai.discover import discover_custom_ai_classes; print(discover_custom_ai_classes())"
```
Look for import errors in the output. Common causes:
- Missing dependency in `requirements.txt`
- Incorrect import path in the plugin
- Syntax error in plugin code

### Import Error: No module named 'google.ai' (or similar)

**Symptom:**
```
Importing module src.custom_ai.gemini.gemini failed with error : No module named 'google.ai'
```

**Cause:** Import path mismatch. The package name in `requirements.txt` doesn't match the import path.

**Solution:**
1. Check `requirements.txt` for the package name
2. Check the plugin's import statements
3. Update the import to match the actual package:
   - `google-genai` uses `google.generativeai` or `google.genai`
   - NOT `google.ai.generativelanguage`

### Chart Generation Fails with Header Offset Issues

**Symptom:**
Charts are misaligned or data is read from wrong rows.

**Cause:** Dynamic header row offset not handled correctly in chart generation.

**Solution:**
Ensure chart generation uses `get_columns_from_worksheet()` to get actual column positions, not hardcoded row numbers.

### Logo or Banner Not Found

**Symptom:**
```
SBK logo Image not found: ./images/sbk-logo.png
```

**Cause:** Relative paths don't work when running from different directories or after installation.

**Solution:**
Use `__file__`-relative paths:
```python
img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images', 'sbk-logo.png')
```

### Build Fails with ModuleNotFoundError

**Symptom:**
```
ModuleNotFoundError: No module named 'src'
```

**Cause:** `setup.py` imports from the source package during build phase.

**Solution:**
Read version from file instead of importing:
```python
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'src', 'version', 'sbk_version.py')
    with open(version_file, 'r') as f:
        content = f.read()
        match = re.search(r"__sbk_version__\s*=\s*['\"]([^'\"]+)['\"]", content)
        if match:
            return match.group(1)
    raise RuntimeError("Could not find version")
```

### Assets Not Bundled in Wheel

**Symptom:**
Logo or banner.txt missing from the wheel after build.

**Cause:** Incorrect `package_data` configuration or missing `pyproject.toml`.

**Solution:**
1. Ensure `pyproject.toml` exists with build backend specified
2. Verify `package_data` keys match actual package names:
   ```python
   package_data={
       'src.main': ['banner.txt'],
       'src.images': ['sbk-logo.png'],
   }
   ```
3. Add `include_package_data=True` in setup.py
4. Ensure `MANIFEST.in` includes the files

### AI Analysis Times Out

**Symptom:**
```
Analysis timed out
```

**Cause:** Analysis exceeds the 120-second budget.

**Solution:**
- Increase timeout with `-secs` flag
- Use a faster model
- For GPU-bound plugins, use `-nothreads` flag
- Check if the API is responding slowly

### RAG Pipeline Returns Empty Results

**Symptom:**
Chat mode returns generic answers without specific data context.

**Cause:** RAG ingestion skipped zero-valued metrics (intentional behavior).

**Solution:**
This is by design. Metrics with all-zero values are skipped to avoid polluting the index.

### ThreadPoolExecutor OOM on GPU

**Symptom:**
Out of memory errors when using PyTorchLLM or other GPU-bound plugins.

**Cause:** Multiple analyses trying to use GPU simultaneously.

**Solution:**
Use the `-nothreads` flag to run analyses sequentially:
```bash
sbk-charts -i input.csv pytorchllm -nothreads
```

## Debugging Tips

### Enable verbose output
```bash
sbk-charts -i input.csv -o output.xlsx --verbose
```

### Check Python path
```bash
python3 -c "import sys; print('\n'.join(sys.path))"
```

### Verify package installation
```bash
pip list | grep sbk-charts
pip show sbk-charts
```

### Test with minimal input
Use the sample CSV to isolate the issue:
```bash
sbk-charts -i samples/charts/sbk-file-read.csv -o /tmp/test.xlsx
```

## When to Ask for Help
If none of these solutions work:
1. Check the GitHub Issues for similar problems
2. Create a minimal reproducible example
3. Include the exact error message
4. Specify your Python version and environment
5. Open a new issue on GitHub
