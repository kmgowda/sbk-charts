# Add New AI Plugin

## Overview
This skill guides through adding a new AI backend plugin to sbk-charts.

## When to use this skill
Use this skill when:
- Adding support for a new AI provider (e.g., OpenAI, Claude, local models)
- Creating a custom AI backend
- Extending the AI analysis capabilities

## Prerequisites
- Read `docs/PLUGIN_SPECIFICATION.md` for the spec template
- Read `docs/AGENT_RECIPES.md` for the "Add a new AI plugin" recipe
- Have the AI provider's API documentation available

## Steps

### 1. Create the plugin directory
```bash
mkdir -p src/custom_ai/<plugin_name>
touch src/custom_ai/<plugin_name>/__init__.py
```

### 2. Create the plugin module
Create `src/custom_ai/<plugin_name>/<plugin_name>.py` with:
- Class name: PascalCase of the plugin name (e.g., `OpenAI`, `Claude`)
- Inherit from `SbkGenAI` (from `src.genai.genai`)
- Implement the four analysis methods:
  - `get_throughput_analysis()`
  - `get_latency_analysis()`
  - `get_total_mb_analysis()`
  - `get_percentile_histogram_analysis()`

### 3. Add CLI arguments
Update `src/parser/sbk_parser.py` to add plugin-specific arguments:
- Prefix with plugin name: `--<plugin>-model`, `--<plugin>-api-key`, etc.
- Add to the appropriate subparser

### 4. Update requirements.txt
Add the required dependencies for your plugin to `requirements.txt`

### 5. Test the plugin
```bash
# Install in editable mode
pip install -e .

# Test with a sample CSV
sbk-charts -i samples/charts/sbk-file-read.csv <plugin_name>
```

### 6. Update documentation
- Add a README in `src/custom_ai/<plugin_name>/README.md`
- Update `AGENTS.md` if there are plugin-specific gotchas
- Update `docs/PLUGIN_SPECIFICATION.md` if you added new patterns

## Naming Conventions
- Directory: lowercase with underscores (e.g., `open_ai`, `claude`)
- Module: same as directory (e.g., `open_ai.py`)
- Class: PascalCase (e.g., `OpenAI`, `Claude`)
- CLI subcommand: lowercase class name (e.g., `openai`, `claude`)
- CLI flags: prefixed with plugin name (e.g., `--openai-model`)

## Common Patterns

### API Key Handling
Most plugins need an API key. Pattern:
```python
import os
api_key = os.environ.get('PLUGIN_API_KEY') or args.plugin_api_key
if not api_key:
    raise ValueError("API key required")
```

### Model Configuration
Allow model selection via CLI:
```python
model = args.plugin_model or "default-model-name"
```

### Timeout Handling
Respect the global timeout:
```python
timeout = args.seconds or 120
```

### Thread Safety
If the plugin uses GPU resources, note that users should use `-nothreads` flag.

## Verification
- Plugin appears in `sbk-charts -h` output
- All four analyses complete successfully
- Output is written to Excel Summary sheet
- No import errors in discovery phase
