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

## Running SBK Charts:

```
<SBK directory>./sbk-charts
...
(venv-sbk-charts) kmg@Mac-Studio sbk-charts % sbk-charts -h
usage: sbk-charts [-h] -i IFILES [-o OFILE] [-secs SECONDS] [-nothreads NOTHREADS] {huggingface,lmstudio,noai,ollama} ...

SBK Charts - Storage Benchmark Visualization Tool

positional arguments:
  {huggingface,lmstudio,noai,ollama}
                        Available sub-commands

options:
  -h, --help            show this help message and exit
  -i, --ifiles IFILES   Comma-separated list of input CSV files containing benchmark results
  -o, --ofile OFILE     Output XLSX file path (default: out.xlsx)
  -secs, --seconds SECONDS
                        Timeout seconds, default : 120
  -nothreads, --nothreads NOTHREADS
                        No parallel threads, default : False

Please report issues at https://github.com/kmgowda/sbk-charts

```

# Single CSV file processing

Example command with single CSV file
```
(venv-sbk-charts) kmg@Mac-Studio sbk-charts % sbk-charts -i ./samples/charts/sbk-file-read.csv -o ./samples/charts/sbk-file-read.xlsx

   _____   ____    _  __            _____   _    _              _____    _______    _____
  / ____| |  _ \  | |/ /           / ____| | |  | |     /\     |  __ \  |__   __|  / ____|
 | (___   | |_) | | ' /   ______  | |      | |__| |    /  \    | |__) |    | |    | (___
  \___ \  |  _ <  |  <   |______| | |      |  __  |   / /\ \   |  _  /     | |     \___ \
  ____) | | |_) | | . \           | |____  | |  | |  / ____ \  | | \ \     | |     ____) |
 |_____/  |____/  |_|\_\           \_____| |_|  |_| /_/    \_\ |_|  \_\    |_|    |_____/

Sbk Charts Version : 3.26.1.0
Input Files :  ./samples/charts/sbk-file-read.csv
Output File :  ./samples/charts/sbk-file-read.xlsx
SBK logo image found: /Users/kmg/projects/sbk-charts/images/sbk-logo.png
xlsx file : ./samples/charts/sbk-file-read.xlsx created
Time Unit : NANOSECONDS
Reading : FILE
file : ./samples/charts/sbk-file-read.xlsx updated with graphs
AI is not enabled!. you can use the subcommands [huggingface lmstudio noai ollama] to enable it.
```
you can see the sample [fil read in csv](./samples/charts/sbk-file-read.csv) as input file and the generated output file is [file read graphs](./samples/charts/sbk-file-read.xlsx)


## Multiple CSV files processing

Example command with multiple CSV files
```
(venv-sbk-charts) kmg@Mac-Studio sbk-charts % sbk-charts -i ./samples/charts/sbk-file-read.csv,./samples/charts/sbk-rocksdb-read.csv -o ./samples/charts/sbk-file-rocksdb-read.xlsx

   _____   ____    _  __            _____   _    _              _____    _______    _____
  / ____| |  _ \  | |/ /           / ____| | |  | |     /\     |  __ \  |__   __|  / ____|
 | (___   | |_) | | ' /   ______  | |      | |__| |    /  \    | |__) |    | |    | (___
  \___ \  |  _ <  |  <   |______| | |      |  __  |   / /\ \   |  _  /     | |     \___ \
  ____) | | |_) | | . \           | |____  | |  | |  / ____ \  | | \ \     | |     ____) |
 |_____/  |____/  |_|\_\           \_____| |_|  |_| /_/    \_\ |_|  \_\    |_|    |_____/

Sbk Charts Version : 3.26.1.0
Input Files :  ./samples/charts/sbk-file-read.csv,./samples/charts/sbk-rocksdb-read.csv
Output File :  ./samples/charts/sbk-file-rocksdb-read.xlsx
SBK logo image found: /Users/kmg/projects/sbk-charts/images/sbk-logo.png
xlsx file : ./samples/charts/sbk-file-rocksdb-read.xlsx created
Time Unit : NANOSECONDS
Reading : FILE, ROCKSDB
file : ./samples/charts/sbk-file-rocksdb-read.xlsx updated with graphs
AI is not enabled!. you can use the subcommands [huggingface lmstudio noai ollama] to enable it.

```
you can see the sample [fil read in csv](./samples/charts/sbk-file-read.csv) and the [rocksdb red in csv](./samples/charts/sbk-rocksdb-read.csv) as input files and the generated output file is [file and rocksdb read comparesion](./samples/charts/sbk-file-rocksdb-read.xlsx)

## Python Virtual Environment Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

#### Setup with Python virtual environment 

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

#### Setup with conda

```
# Create a new conda environment with Python 3.14 or higher
conda create -n conda-sbk-charts python=3.14 -y

# Activate the environment
conda activate conda-sbk-charts

# Install pip if not already installed
conda install pip -y

# Install the project in editable mode using pip
pip install -e .

# Build the sbk-charts package
python -m build
```

To deactivate

```
conda deactivate
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

## AI Backends

SBK Charts supports multiple AI backends for analysis:

1. **LM Studio** - For local AI inference with LM Studio
2. **Ollama** - For running local LLMs through the Ollama API
3. **Hugging Face** - For cloud-based AI analysis (default)

### LM Studio Setup

1. Install [LM Studio](https://lmstudio.ai/)
2. Download and host a suitable model (e.g., Mistral 7B, Llama 3.1)
3. Start the LM Studio server

Example usage:
```bash
sbk-charts -i input.csv -o output.xlsx lmstudio 
```

### Ollama Setup

1. Install [Ollama](https://ollama.com/)
2. Pull required models:
   ```bash
   ollama pull llama3.1
   ```

Example usage:
```bash
sbk-charts -i input.csv -o output.xlsx ollama
```

For more details, see the documentation in [custom AI models](src/custom_ai/README.md)

## Contributing

We welcome and appreciate contributions from the open-source community!
Whether you're interested in improving the code, enhancing documentation, or adding new AI backend models, your contributions help make SBK Charts better for everyone.

### How to Contribute

1. **Fork the repository** and create your feature branch (`git checkout -b feature/amazing-feature`)
2. **Commit your changes** with clear, descriptive messages
3. **Push to the branch** (`git push origin feature/amazing-feature`)
4. **Open a Pull Request** with a clear description of your changes

### Areas Needing Contributions

#### 1. Code Improvements
- Performance optimizations
- Bug fixes
- New features and enhancements
- Test coverage improvements

#### 2. Documentation
- Improve existing documentation
- Add usage examples
- Create tutorials or guides
- Translate documentation to other languages

#### 3. AI Model Integrations
We're particularly interested in expanding our AI capabilities. You can help by:
- Adding support for new AI providers (e.g., OpenAI, Anthropic, local LLMs)
- Improving prompt engineering for better analysis
- Adding new types of performance analysis
- Supporting more benchmark result formats

### Setting Up for Development
Set up the development environment as described in the [Python Virtual Environment Setup](#python-virtual-environment-setup) section

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use type hints for better code clarity
- Write docstrings for all public functions and classes
- Keep commits atomic and focused

### Reporting Issues

Found a bug or have a feature request? Please open an issue on our [GitHub Issues](https://github.com/kmgowda/sbk-charts/issues) page with:
- A clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Any relevant screenshots or logs

### License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

---

