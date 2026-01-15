<!--
Copyright (c) KMG. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
-->
# PyTorch LLM Implementation for SBK Charts

This document describes the PyTorch LLM implementation for SBK Charts, which enables local AI-powered analysis of storage benchmark results using PyTorch and Hugging Face models.

## Overview

The PyTorch LLM implementation allows SBK Charts to leverage local language models through PyTorch's inference capabilities.
This implementation is particularly useful for users who want to run AI analysis locally without relying on external APIs.
This implementation is based on the Hugging Face Transformers library and PyTorch.
you can train the model by providing '--pt-train' option, but the default model : 'openai/gpt-oss-20b' consumes more than 200GB RAM.

## Implementation Details

The PyTorch LLM implementation is located in the `src/custom_ai/pytorch_llm` directory and extends the base AI interface defined in `src/genai/genai.py`.

### Key Features

1. **Local Model Inference**: Runs entirely on your hardware using PyTorch
2. **Hugging Face Integration**: Supports any Causal Language Model from the Hugging Face Hub
3. **Hardware Acceleration**: Automatically utilizes CUDA, MPS, or CPU based on availability
4. **Memory Efficient**: Uses 16-bit or 32-bit precision based on hardware support
5. **Configurable Parameters**: Adjust model parameters like temperature and top-p sampling

## Prerequisites

### Python Dependencies

- PyTorch (with CUDA support recommended for GPU acceleration)
- Transformers library from Hugging Face
- A compatible pre-trained language model (e.g., from Hugging Face Hub)

### Hardware Requirements

- CPU: Modern x86-64 or ARM processor
- RAM: At least 32GB (128GB+ recommended for larger models)
- GPU: NVIDIA GPU with CUDA support recommended for better performance
- Disk Space: 10GB+ for model storage (varies by model size)

## Installation

1. Install PyTorch (with CUDA if available):
   ```bash
   # For CUDA 11.8
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU-only
   # pip3 install torch torchvision torchaudio
   ```

2. Install the Transformers library:
   ```bash
   pip install transformers
   ```

3. (Optional) Install additional dependencies for specific models:
   ```bash
   pip install accelerate bitsandbytes
   ```

## Configuration

The implementation supports the following configuration options:

- **Model**: Any Hugging Face model ID or local path (default: `openai/gpt-oss-20b`)
- **Device**: Automatically detects CUDA/MPS/CPU (can be overridden)
- **Max Length**: Maximum sequence length for generation (default: 2048)
- **Temperature**: Controls randomness (default: 0.4)
- **Top-p**: Nucleus sampling parameter (default: 0.9)

## Usage

### Basic Usage

```bash
# Process a single CSV file with default settings
sbk-charts -i input.csv -o output.xlsx pytorch_llm

# Process multiple CSV files
sbk-charts -i file1.csv,file2.csv -o output.xlsx pytorch_llm
```

### Advanced Options

```bash
# Specify a different model
sbk-charts -i input.csv -o output.xlsx pytorch_llm --pt-model mistralai/Mistral-7B-v0.1

# Adjust generation parameters
sbk-charts -i input.csv -o output.xlsx pytorch_llm \
    --pt-temperature 0.7 \
    --pt-top-p 0.95 \
    --pt-max-length 1024

# Force CPU usage
sbk-charts -i input.csv -o output.xlsx pytorch_llm --pt-device cpu
```

## Example Commands

```bash
# Process file with default settings
sbk-charts -i ./samples/charts/sbk-file-read.csv -o ./samples/charts/sbk-file-read-pytorch.xlsx pytorch_llm

# Use a smaller model with custom parameters
sbk-charts -i ./samples/charts/sbk-file-read.csv -o ./samples/charts/sbk-file-read-mistral.xlsx pytorch_llm \
    --pt-model mistralai/Mistral-7B-v0.1 \
    --pt-temperature 0.5 \
    --pt-max-length 1024

# Process multiple files
sbk-charts -i ./samples/charts/sbk-file-read.csv,./samples/charts/sbk-rocksdb-read.csv \
    -o ./samples/charts/sbk-combined-pytorch.xlsx pytorch_llm
```

## Model Management

### Using Local Models

1. Download a model from Hugging Face Hub:
   ```bash
   from transformers import AutoModelForCausalLM, AutoTokenizer
   
   model_name = "mistralai/Mistral-7B-v0.1"
   model = AutoModelForCausalLM.from_pretrained(model_name)
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   
   # Save locally
   save_path = "./saved_models/mistral-7b"
   model.save_pretrained(save_path)
   tokenizer.save_pretrained(save_path)
   ```

2. Use the local model:
   ```bash
   sbk-charts -i input.csv -o output.xlsx pytorch_llm --pt-model ./saved_models/mistral-7b
   ```

## Performance Tips

1. **Use GPU Acceleration**: Ensure CUDA is properly installed for best performance
2. **Quantization**: For large models, consider 4-bit or 8-bit quantization
3. **Batch Processing**: Process multiple files in a single command when possible
4. **Model Size**: Choose an appropriately sized model for your hardware
5. **Execution time**: choose the execution time at least 30 minutes ; you can use the parameter '-secs' parameter

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM) Errors**:
   - Reduce `--pt-max-length`
   - Use a smaller model
   - Enable gradient checkpointing
   - Use quantization

2. **Model Loading Issues**:
   - Ensure the model name is correct
   - Check internet connection if downloading
   - Verify disk space is available

3. **CUDA Errors**:
   - Check CUDA installation: `nvidia-smi`
   - Ensure PyTorch was installed with CUDA support
   - Try reducing batch size or model size

## Directory Structure

```
src/
└── custom_ai/
    └── pytorch_llm/
        ├── __init__.py
        ├── pytorch_llm.py      # Main implementation
        └── README.md           # This document
```
## License

Apache License 2.0
