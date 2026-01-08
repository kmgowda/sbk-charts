# Custom AI Implementations for SBK Charts

This directory contains custom AI implementations for analyzing storage benchmark results in the SBK Charts application. Each implementation provides AI-powered analysis of performance metrics including throughput, latency, and other storage-related statistics.

## Available AI Backends

### 1. Hugging Face
- **Description**: Cloud-based AI analysis using Hugging Face's Inference API
- **Requirements**:
  - `HUGGINGFACE_API_TOKEN` environment variable set with a valid API key
  - Internet access to Hugging Face's API endpoints
- **Configuration**:
  - `--model_id`: Specify the Hugging Face model ID (default: `google/gemma-2-2b-it`)
- **Usage**:
  ```bash
  export HUGGINGFACE_API_TOKEN=your_api_token_here
  sbk-charts -i ./samples/charts/sbk-file-read.csv huggingface
  ```

For more details, see the documentation in [Hugging face](hugging_face/README.md)

### 2. LM Studio
- **Description**: Local AI analysis using LM Studio's local inference server
- **Requirements**:
  - LM Studio application running locally
  - Compatible LLM model loaded in LM Studio
  - Network access to the LM Studio server (default: localhost:1234)
- **Configuration**:
  - `--url`: LM Studio server URL (default: `http://localhost:1234/v1`)
  - `--lm-model`: Model name (default: `openai/gpt-oss-20b`)
  - `--lm-temperature`: Sampling temperature (0.0-1.0, default: 0.4)
  - `--lm-max-tokens`: Maximum tokens to generate (default: 1800)
- **Usage**:
  ```bash
  sbk-charts -i ./samples/charts/sbk-file-read.csv lmstudio
  ```
For more details, see the documentation in [LM Studio](lm_studio/README.md)


### 3. Ollama
- **Description**: Local AI analysis using Ollama's local model serving
- **Requirements**:
  - Ollama server running locally
  - Compatible model pulled via Ollama
- **Configuration**:
  - `--ollama-url`: Ollama server URL (default: `http://localhost:11434`)
  - `--ollama-model`: Model name (default: `llama3.1`)
  - `--ollama-temperature`: Sampling temperature (0.0-1.0, default: 0.4)
  - `--ollama-timeout`: Request timeout in seconds (default: 120)
- **Usage**:
  ```bash
  sbk-charts -i ./samples/charts/sbk-file-read.csv ollama --ollama-model llama3.1 --ollama-temperature 0.4
  ```

For more details, see the documentation in [ollama](ollama/README.md)

### 4. NoAI (Default)
- **Description**: Placeholder implementation that returns error messages
- **Usage**:
  ```bash
  sbk-charts -i ./samples/charts/sbk-file-read.csv noai
  ```
  This will display a message indicating that AI analysis is not enabled.

## Extending with Custom AI Implementations

To create a new AI backend, create a new Python module in this directory that implements the `SbkGenAI` abstract base class. Your implementation must provide the following methods:

```python
class MyCustomAI(SbkGenAI):
    def get_model_description(self) -> Tuple[bool, str]:
        """Return a description of the AI model being used."""
        pass
        
    def get_throughput_analysis(self) -> Tuple[bool, str]:
        """Generate analysis of throughput metrics."""
        pass
        
    def get_latency_analysis(self) -> Tuple[bool, str]:
        """Generate analysis of latency metrics."""
        pass
        
    def get_total_mb_analysis(self) -> Tuple[bool, str]:
        """Generate analysis of total MB processed."""
        pass
        
    def get_percentile_histogram_analysis(self) -> Tuple[bool, str]:
        """Generate analysis of percentile histogram data."""
        pass
```

## Common Configuration

All AI backends support the following common configuration options:

- `-i, --input`: Comma-separated list of input CSV files
- `-o, --output`: Output file path (optional, prints to console if not specified)
- `--log-level`: Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Example Commands

1. **Basic usage with Hugging Face**:
   ```bash
   export HUGGINGFACE_API_TOKEN=your_token_here
   sbk-charts -i file1.csv,file2.csv huggingface
   ```

2. **Using LM Studio with custom parameters**:
   ```bash
   sbk-charts -i data.csv lmstudio --lm-temperature 0.7 --lm-max-tokens 2000
   ```

3. **Saving output to a file**:
   ```bash
   sbk-charts -i data.csv ollama -o analysis_results.txt
   ```

## Troubleshooting

- **Connection Issues**: Ensure the AI service (LM Studio/Ollama) is running and accessible
- **API Key Errors**: Verify that environment variables like `HUGGINGFACE_API_TOKEN` are set correctly
- **Model Loading**: Make sure the specified model is available in your local/remote environment
- **Timeout Errors**: Increase the timeout value for large datasets or complex analyses

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

