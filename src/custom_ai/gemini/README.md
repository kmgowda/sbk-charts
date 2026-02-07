# Gemini AI Backend

This module provides integration with Google's Gemini AI models for generating AI-powered analysis of storage benchmark results.

## Features

- Cloud-based AI analysis using Google's Gemini API
- Support for multiple Gemini models (Pro, Flash, etc.)
- Configurable model parameters (temperature, max tokens)
- Automatic error handling and retry logic
- RAG-enhanced context for custom queries
- Uses official Google AI SDK for reliable API communication

## Requirements

- `google-genai` Python package (included in requirements.txt)
- Valid Google AI API key
- Internet connection to Google's API endpoints

## Installation

The module uses the official `google-genai` Python SDK. Install dependencies:

```bash
pip install -r requirements.txt
```

The `google-genai` package will be installed automatically.

## Configuration

Set the required environment variable:

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

You can get an API key from the [Google AI Studio](https://makersuite.google.com/app/apikey).

## Usage

### Basic Usage

```python
from src.custom_ai.gemini import Gemini

# Initialize the Gemini backend
gemini = Gemini()

# Check if properly configured
success, description = gemini.get_model_description()
if success:
    print(description)
else:
    print(f"Configuration error: {description}")

# Generate throughput analysis
success, analysis = gemini.get_throughput_analysis()
if success:
    print(analysis)
else:
    print(f"Analysis failed: {analysis}")
```

### Command Line Usage

```bash
# Use default model (gemini-2.5-flash)
python sbk_charts.py --ai-backend gemini

# Specify different model
python sbk_charts.py --ai-backend gemini --gemini-model gemini-1.5-pro

# Adjust model parameters
python sbk_charts.py --ai-backend gemini --gemini-temperature 0.7 --gemini-max-tokens 4096
```

## Supported Models

- `gemini-2.5-flash` (default) - Latest fast and efficient model
- `gemini-1.5-pro` - More capable for complex analysis
- `gemini-1.5-flash` - Fast and efficient for most use cases
- `gemini-1.0-pro` - Legacy model support

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--gemini-model` | Gemini model to use | `gemini-2.5-flash` |
| `--gemini-temperature` | Sampling temperature (0.0-1.0) | `0.4` |
| `--gemini-max-tokens` | Maximum tokens to generate | `2048` |

## Error Handling

The module provides comprehensive error handling:

- Missing API key detection
- Network connectivity issues
- Model availability problems
- Invalid response handling

## Integration with SBK Charts

The Gemini backend integrates seamlessly with the SBK Charts framework:

1. Extends the `SbkGenAI` base class
2. Implements all required analysis methods
3. Supports RAG-enhanced context for custom queries
4. Provides consistent error handling and logging

## Examples

### Analyzing Throughput Performance

```python
gemini = Gemini()
success, analysis = gemini.get_throughput_analysis()
if success:
    print("Throughput Analysis:")
    print(analysis)
```

### Custom Query Analysis

```python
success, response = gemini.get_response("Which storage system has better read performance?")
if success:
    print("Custom Analysis:")
    print(response)
```

## Implementation Details

### API Communication

The module uses the official `google-genai` Python SDK for communication with Google's Gemini API:

- **Client Management**: Creates and manages `GenerativeServiceClient` instances
- **Request Building**: Uses structured `GenerateContentRequest` objects
- **Response Parsing**: Extracts text content from structured API responses
- **Error Handling**: Comprehensive exception handling for API failures

### Key Components

1. **Gemini Class**: Main implementation extending `SbkGenAI`
2. **Client Initialization**: Automatic client setup with API key
3. **Content Generation**: Structured request/response handling
4. **Configuration Management**: Dynamic parameter updates


### Common Issues

1. **Missing API Key**: Ensure `GEMINI_API_KEY` environment variable is set
2. **Network Issues**: Check internet connectivity to Google's API endpoints
3. **Model Unavailable**: Verify the specified model is available in your region
4. **Quota Exceeded**: Check your Google AI API quota and usage limits
5. **SDK Installation**: Ensure `google-genai` package is properly installed

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This module is licensed under the Apache License 2.0, same as the parent project.
