<!--
Copyright (c) KMG. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
-->
# SBK Charts 
The sbk-charts is the python application software to create the xlsx file for given single or multiple CSV file containing the SBK performance 
results. The generated xlsx file contains the graphs of latency percentile variations and throughput variations.
The [SBK framework](https://github.com/kmgowda/SBK) can be used to benchmark the performance of various storage engines like RocksDB, LMDB, LevelDB, etc. and to generate the performance results in CSV format.
The sbk-charts application can be used to visualize these results in a more user-friendly way.

**sbk-charts uses AI to generate descriptive summaries about throughput and latency analysis**

## AI Backends

SBK Charts supports multiple AI backends for analysis:

1. **LM Studio** - For local AI inference with LM Studio
2. **Ollama** - For running local LLMs through the Ollama API
3. **Hugging Face** - For cloud-based AI analysis (default)

### LM Studio Setup

1. Install [LM Studio](https://lmstudio.ai/)
2. Download and host a suitable model (e.g., Mistral 7B, Llama 2)
3. Start the LM Studio server

Example usage:
```bash
sbk-charts -i input.csv -o output.xlsx lmstudio --lm-model mistral
```

### Ollama Setup

1. Install [Ollama](https://ollama.com/)
2. Pull required models:
   ```bash
   ollama pull llama3
   ollama pull mistral
   ```

Example usage:
```bash
sbk-charts -i input.csv -o output.xlsx ollama --model llama3
```

For more details, see the documentation in `src/custom_ai/<backend>/README.md`

---

## Running SBK Charts:

```
<SBK directory>./sbk-charts
...
(sbk-charts-venv) kmg@Mac-Studio sbk-charts % sbk-charts -h
usage: sbk-charts [-h] -i IFILES [-o OFILE] {huggingface,noai} ...

sbk charts

positional arguments:
  {huggingface,noai}   Available sub-commands

options:
  -h, --help           show this help message and exit
  -i, --ifiles IFILES  Input CSV files, separated by ','
  -o, --ofile OFILE    Output xlsx file

Please report issues at https://github.com/kmgowda/sbk-charts

```

# Single CSV file processing

Example command with single CSV file
```
kmg@kmgs-MBP SBK % ./sbk-charts -i ./samples/charts/sbk-file-read.csv -o ./samples/charts/sbk-file-read.xlsx 

   _____   ____    _  __            _____   _    _              _____    _______    _____
  / ____| |  _ \  | |/ /           / ____| | |  | |     /\     |  __ \  |__   __|  / ____|
 | (___   | |_) | | ' /   ______  | |      | |__| |    /  \    | |__) |    | |    | (___
  \___ \  |  _ <  |  <   |______| | |      |  __  |   / /\ \   |  _  /     | |     \___ \
  ____) | | |_) | | . \           | |____  | |  | |  / ____ \  | | \ \     | |     ____) |
 |_____/  |____/  |_|\_\           \_____| |_|  |_| /_/    \_\ |_|  \_\    |_|    |_____/

Sbk Charts Version : 0.96
Input Files :  ./samples/charts/sbk-file-read.csv
Output File :  ./samples/charts/sbk-file-read.xlsx
xlsx file : ./samples/charts/sbk-file-read.xlsx created
Time Unit : NANOSECONDS
Reading : FILE

```
you can see the sample [fil read in csv](./samples/charts/sbk-file-read.csv) as input file and the generated output file is [file read graphs](./samples/charts/sbk-file-read.xlsx)


## Multiple CSV files processing

Example command with multiple CSV files
```
kmg@kmgs-MBP SBK % ./sbk-charts -i ./samples/charts/sbk-file-read.csv,./samples/charts/sbk-rocksdb-read.csv -o ./samples/charts/sbk-file-rocksdb-read.xlsx

   _____   ____    _  __            _____   _    _              _____    _______    _____
  / ____| |  _ \  | |/ /           / ____| | |  | |     /\     |  __ \  |__   __|  / ____|
 | (___   | |_) | | ' /   ______  | |      | |__| |    /  \    | |__) |    | |    | (___
  \___ \  |  _ <  |  <   |______| | |      |  __  |   / /\ \   |  _  /     | |     \___ \
  ____) | | |_) | | . \           | |____  | |  | |  / ____ \  | | \ \     | |     ____) |
 |_____/  |____/  |_|\_\           \_____| |_|  |_| /_/    \_\ |_|  \_\    |_|    |_____/

Sbk Charts Version : 0.96
Input Files :  ./samples/charts/sbk-file-read.csv,./samples/charts/sbk-rocksdb-read.csv
Output File :  ./samples/charts/sbk-file-rocksdb-read.xlsx
xlsx file : ./samples/charts/sbk-file-rocksdb-read.xlsx created
Time Unit : NANOSECONDS
Reading : FILE, ROCKSDB

```
you can see the sample [fil read in csv](./samples/charts/sbk-file-read.csv) and the [rocksdb red in csv](./samples/charts/sbk-rocksdb-read.csv) as input files and the generated output file is [file and rocksdb read comparesion](./samples/charts/sbk-file-rocksdb-read.xlsx)

## Python Virtual Environment Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

```
#create the env
python3 -m venv venv-sbk-charts

#set the env
source venv-sbk-charts/bin/activate

# install required packages
pip install -e .

# build the sbk-charts
python3 -m build  
``` 

to deactivate from the venv 

```
# deactivate the venv
deactivate
```

## Generative AI-Powered Analysis

SBK Charts includes AI-powered analysis descriptions to provide deeper insights into your storage benchmark results.
As of today, The analysis is performed using the Hugging Face model and includes:

### Available AI Analyses

1. **Throughput Analysis**
   - Analyzes MB/s and records/s metrics
   - Identifies performance patterns and anomalies
   - Compares performance across different storage systems

2. **Latency Analysis**
   - Examines latency distributions
   - Highlights tail latency patterns
   - Provides comparative analysis between different storage configurations

3. **Total MB Analysis**
   - Analyzes total data transferred
   - Identifies throughput patterns over time
   - Compares data transfer efficiency

4. **Percentile Histogram Analysis**
   - Detailed analysis of latency percentiles
   - Identifies performance bottlenecks
   - Compares percentile distributions across storage systems

### Usage

To use AI analysis, run the tool with one of the available AI subcommands:

```bash
# Using Hugging Face model (default)
sbk-charts -i input.csv -o output.xlsx huggingface

# Example
sbk-charts -i ./samples/charts/sbk-file-read.csv,./samples/charts/sbk-rocksdb-read.csv huggingface


# Using NoAI (fallback with error messages)
sbk-charts -i input.csv -o output.xlsx noai
# Example
sbk-charts -i ./samples/charts/sbk-file-read.csv,./samples/charts/sbk-rocksdb-read.csv noai

```

for further details on custom AI implementations, please refer to the [custom AI](./src/custom_ai/README.md) directory.

