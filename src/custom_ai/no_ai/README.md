# NoAI backend

`noai` is a deterministic placeholder that implements the AI interface without calling a model.

```bash
./sbk-charts -i input.csv -o noai-report.xlsx noai
```

It writes clear "AI is not enabled" failure text for the model description and four analyses. It has no backend-specific options, credentials, network calls, or model requirements.

This differs from omitting a backend command:

- no backend command creates charts and skips the AI block;
- `noai` selects an AI implementation, so the Summary receives placeholder analysis sections.

NoAI is useful for exercising the AI orchestration and Summary layout without contacting a provider.
