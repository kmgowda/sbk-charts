# PyTorchLLM backend

The `pytorchllm` command loads a Hugging Face causal language model directly in the sbk-charts Python process through Transformers and PyTorch. It offers local inference but has the highest memory and startup requirements.

## Resource warning

The code default is `openai/gpt-oss-20b`, a large model that does not fit many developer machines. Model memory depends on parameter count, data type, device, and loading behavior. Choose a smaller compatible model unless you know the default fits.

The backend automatically chooses CUDA, then Apple MPS, then CPU. You can override that choice.

## Run

Use a model that fits the machine and run the analyses sequentially on one accelerator:

```bash
./sbk-charts -i input.csv -o pytorch-report.xlsx \
  -secs 1800 -nothreads \
  pytorchllm \
  --pt-model <hugging-face-model-id-or-local-path> \
  --pt-device cpu
```

Configuration example:

```bash
./sbk-charts -i input.csv -o pytorch-report.xlsx \
  -secs 1800 -nothreads \
  pytorchllm \
  --pt-model <model> \
  --pt-device mps \
  --pt-max-length 1024 \
  --pt-temperature 0.4 \
  --pt-top-p 0.9
```

| Flag | Code default |
|---|---|
| `--pt-model` | `openai/gpt-oss-20b` |
| `--pt-train` | Disabled |
| `--pt-device` | CUDA, otherwise MPS, otherwise CPU |
| `--pt-max-length` | `2048` |
| `--pt-temperature` | `0.4` |
| `--pt-top-p` | `0.9` |

`--pt-train` changes model state and can require substantially more memory and time. Use it only when you understand the plugin's training and local-save behavior.

## How it works

The plugin first looks for a saved model under its `saved_models` directory. Otherwise it downloads the selected model and tokenizer through Transformers. It moves the model to the chosen device, generates responses for shared prompts, and releases model references and accelerator caches during cleanup.

Model downloads require network access unless a local path or populated cache is used. Generated or downloaded model files must not be committed.

## Troubleshooting

- Out of memory: select a smaller model, reduce `--pt-max-length`, use CPU if practical, and keep `-nothreads` enabled.
- Unsupported PyTorch distribution: use a Python/platform combination with a matching wheel or let the source launcher fall back to Conda.
- MPS or CUDA failure: verify the installed PyTorch build supports the accelerator, or pass `--pt-device cpu`.
- Model access failure: authenticate with Hugging Face when required and accept gated-model terms.
- Very slow execution: increase `-secs`; CPU inference for large models can take much longer than the default budget.

Local execution keeps prompts on the machine after any required model download, subject to the behavior of installed libraries and configured caches.
