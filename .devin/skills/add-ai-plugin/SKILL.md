# Add an sbk-charts AI backend

Use this skill when adding a cloud provider, local model server, or in-process language model.

## Read first

- `AGENTS.md`
- `docs/ARCHITECTURE.md`, especially the AI plugin system
- `docs/PLUGIN_SPECIFICATION.md`
- `docs/AGENT_RECIPES.md`, recipe 1

## Procedure

1. Choose the closest existing backend pattern.
2. For a substantial integration, complete a plugin specification.
3. Create `src/custom_ai/<name>/__init__.py`, `<name>.py`, and `README.md`.
4. Define one concrete `SbkGenAI` subclass. Its lowercased class name becomes the command automatically.
5. Add plugin flags in its `add_args()` and consume them in `parse_args()`. Do not edit the base parser and do not create a registry.
6. Reuse all four shared prompt builders from `SbkGenAI`.
7. Return `(True, text)` on success and `(False, actionable_error)` for expected failure.
8. Implement chat with `_enhance_prompt_with_rag()` when supported.
9. Close sessions or release model resources in `close()`.
10. Add the dependency to `requirements.txt` and update the backend index and architecture table.

## Verification

```bash
./sbk-charts -h
./sbk-charts -i samples/charts/sbk-file-read.csv <backend> -h
./sbk-charts -i samples/charts/sbk-file-read.csv \
  -o /tmp/backend.xlsx <backend>
```

Test missing authentication/service, configured happy path, all four analyses, chat when supported, and `-nothreads` when the model is not safe for four concurrent calls. Confirm the workbook saves even when provider calls fail clearly.

Never print, store, or commit credentials.
