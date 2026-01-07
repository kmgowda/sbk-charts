<!--
Copyright (c) KMG. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
-->
# Ollama Implementation for SBK Charts

This document describes the Ollama implementation for SBK Charts, which enables AI-powered analysis of storage benchmark results using the Ollama API.

## Overview

The Ollama implementation allows SBK Charts to leverage local LLMs through the Ollama API for generating descriptive summaries of throughput, latency, and other performance metrics. This implementation provides an alternative to cloud-based AI services and enables offline analysis capabilities.

## Implementation Details

The Ollama implementation is located in the `src/custom_ai/ollama` directory and extends the base AI interface defined in `src/custom_ai/base_ai.py`.

### Key Features

1. **Local LLM Integration**: Uses Ollama API to run local language models
2. **Offline Processing**: No internet connection required for analysis
3. **Flexible Model Support**: Works with any Ollama-supported model
4. **Consistent Interface**: Maintains compatibility with existing SBK Charts AI framework

### Configuration

The Ollama implementation requires the following configuration:

- **Ollama Server**: Must be running locally or accessible via network
- **Model Selection**: Specify the desired model (e.g., `llama3`, `mistral`, etc.)
- **API Endpoint**: Default is `http://localhost:11434` (can be customized)

### Usage

To use the Ollama implementation, run SBK Charts with the `ollama` subcommand:

```bash
# Basic usage with default settings
sbk-charts -i input.csv -o output.xlsx ollama

# With custom model
sbk-charts -i input.csv -o output.xlsx ollama --model llama3

# With custom endpoint
sbk-charts -i input.csv -o output.xlsx ollama --url http://localhost:11434

```

# Process single CSV file with Ollama
sbk-charts -i ./samples/charts/sbk-file-read.csv -o ./samples/charts/sbk-file-read-ollama.xlsx ollama

# Process multiple CSV files with Ollama
sbk-charts -i ./samples/charts/sbk-file-read.csv,./samples/charts/sbk-rocksdb-read.csv -o ./samples/charts/sbk-file-rocksdb-read-ollama.xlsx ollama

# With custom model
sbk-charts -i ./samples/charts/sbk-file-read.csv -o ./samples/charts/sbk-file-read-ollama.xlsx ollama --model mistral


