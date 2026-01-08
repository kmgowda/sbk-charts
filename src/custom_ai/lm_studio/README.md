# LM Studio AI Implementations
This directory contains LM  AI implementations for the sbk-charts application.

To use the LM studio for analysis make sure that you hare LMG studio server is up and running locally in your machine or in your reachable network.
you need to ensure that model is loaded properly at LM studio.


- Inherits from `SbkGenAI` class for consistent interface
- Implements all required analysis methods:
  - `get_throughput_analysis()`
  - `get_latency_analysis()`
  - `get_total_mb_analysis()`
  - `get_percentile_histogram_analysis()`
- Uses LM Studio's local AI inference capabilities for analysis
- Supports local model hosting without internet dependency

## Prerequisites

1. Install [LM Studio](https://lmstudio.ai/)
2. Download and host a suitable model (e.g., Mistral 7B, Llama 2, etc.)
3. Start the LM Studio server
4. Ensure LM Studio server is running and accessible

## Configuration

The LM Studio AI backend requires the following configuration:

- LM Studio server URL (default: `http://localhost:1234`)
- Model name to use for analysis
- API key (if required by your LM Studio setup)

## Usage

The example command to use the Hugging Face implementation is:

```
(sbk-charts-venv) kmg@Mac-Studio sbk-charts % sbk-charts -i ./samples/charts/sbk-file-read.csv,./samples/charts/sbk-rocksdb-read.csv -nothreads true lmstudio

   _____   ____    _  __            _____   _    _              _____    _______    _____
  / ____| |  _ \  | |/ /           / ____| | |  | |     /\     |  __ \  |__   __|  / ____|
 | (___   | |_) | | ' /   ______  | |      | |__| |    /  \    | |__) |    | |    | (___
  \___ \  |  _ <  |  <   |______| | |      |  __  |   / /\ \   |  _  /     | |     \___ \
  ____) | | |_) | | . \           | |____  | |  | |  / ____ \  | | \ \     | |     ____) |
 |_____/  |____/  |_|\_\           \_____| |_|  |_| /_/    \_\ |_|  \_\    |_|    |_____/

Sbk Charts Version : 3.26.1.0
Input Files :  ./samples/charts/sbk-file-read.csv,./samples/charts/sbk-rocksdb-read.csv
Output File :  out.xlsx
SBK logo image found: /Users/kmg/projects/sbk-charts/images/sbk-logo.png
xlsx file : out.xlsx created
Time Unit : NANOSECONDS
Reading : FILE, ROCKSDB
file : out.xlsx updated with graphs
Starting AI analysis. This may take a few minutes...
Running analysis sequentially (no threads)...
Running get_throughput_analysis...
✓ Completed get_throughput_analysis
Running get_latency_analysis...
✓ Completed get_latency_analysis
Running get_total_mb_analysis...
✓ Completed get_total_mb_analysis
Running get_percentile_histogram_analysis...
✓ Completed get_percentile_histogram_analysis
Analysis completed in 29.49 seconds
File updated with graphs and AI documentation: out.xlsx
```

Note that option "-nothreads true" is used. this is because , the parallel threads with local LM Studio server has the consistency issues with python threads implementation.


The -help option can be used to get the help message for the Hugging Face implementation.

```

usage: sbk-charts lmstudio [-h] [--url URL] [--lm-model LM_MODEL] [--lm-temperature LM_TEMPERATURE] [--lm-max-tokens LM_MAX_TOKENS]

options:
  -h, --help            show this help message and exit
  --url URL             server url (default: http://localhost:1234/v1)
  --lm-model LM_MODEL   Model name or path to use (default: openai/gpt-oss-20b, uses LM Studio's selected model)
  --lm-temperature LM_TEMPERATURE
                        Sampling temperature (default: 0.4)
  --lm-max-tokens LM_MAX_TOKENS
                        Maximum number of tokens to generate (default: 1800)

```


### Model Selection

The LM Studio Face model can be selected using the `--lm-model` option. The default model is `openai/gpt-oss-20b`.
But you need ensure that chosen model is loaded at the LM studio server.

