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
4. Define one concrete `SbkGenAI` subclass and add its lightweight descriptor to `src/ai/registry.py`.
5. Add the same plugin flags to the registry descriptor and consume them in `parse_args()`. The registry must not import the optional SDK.
6. Reuse all four shared prompt builders from `SbkGenAI`.
7. Return `(True, text)` on success and `(False, actionable_error)` for expected failure.
8. Implement chat with `_enhance_prompt_with_rag()` when supported.
9. Close sessions or release model resources in `close()`.
10. Add dependencies to `requirements-ai/<name>.txt`, map the profile in `sbk-charts.ini`, and regenerate its exact hashed lock.
11. Add the backend README to `MANIFEST.in` and the portable `bundle_paths` in `sbk-charts.ini`.
12. Update the backend index, architecture table, and portable-policy tests that assert the bundled guide set.

## Verification

```bash
./sbk-charts -h
./sbk-charts -i samples/charts/sbk-file-read.csv <backend> -h
./sbk-charts -i samples/charts/sbk-file-read.csv \
  -o /tmp/backend.xlsx <backend>
```

Test missing authentication/service, configured happy path, all four analyses, chat when supported, and `-nothreads` when the model is not safe for four concurrent calls. Confirm the workbook saves even when provider calls fail clearly.

Use a fresh `SBK_CHARTS_RUNTIME_ROOT` to confirm the launcher creates the backend-specific managed profile. Run it again offline to confirm the saved profile and lock fingerprint are reused. Do not claim that check when the backend lock has no distribution for the test platform.

Never print, store, or commit credentials.
