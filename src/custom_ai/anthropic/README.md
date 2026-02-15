# Anthropic Claude AI Implementation for SBK Charts

This directory contains the Anthropic Claude AI implementation for analyzing storage benchmark results in the SBK Charts application. This implementation leverages Claude's advanced reasoning capabilities to provide detailed, technical performance analysis.

## Overview

The Anthropic backend uses Claude's API to generate AI-powered analysis of storage performance metrics including throughput, latency, and percentile distributions. Claude excels at understanding complex technical data and providing actionable insights for storage engineers.

## Features

- **Advanced Analysis**: Utilizes Claude's state-of-the-art language understanding for detailed technical insights
- **Multiple Models**: Support for Claude Sonnet, Opus, and Haiku models
- **Cloud-Based**: No local infrastructure required
- **Reliable**: Anthropic's enterprise-grade API with high availability
- **Configurable**: Adjust model parameters for different use cases

## Prerequisites

### API Key

You need an Anthropic API key to use this backend:

1. Sign up at https://console.anthropic.com/
2. Create an API key in your account settings
3. Set the environment variable:
   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

### Python Dependencies

Install the Anthropic Python SDK:

```bash
pip install anthropic
```

## Configuration

The Anthropic backend supports the following configuration options:

- **--anthropic-model**: Claude model to use (default: `claude-sonnet-4-20250514`)
- **--anthropic-temperature**: Sampling temperature, 0.0-1.0 (default: `0.4`)
- **--anthropic-max-tokens**: Maximum tokens to generate (default: `2048`)

### Available Models

- **claude-sonnet-4-20250514**: Latest Sonnet model - balanced performance and speed (recommended)
- **claude-opus-4-20250514**: Most capable model - best for complex analysis
- **claude-3-5-sonnet-20241022**: Previous Sonnet version
- **claude-3-5-haiku-20241022**: Fastest model - good for simple analysis

## Usage

### Basic Usage

```bash
# Set your API key
export ANTHROPIC_API_KEY=your_api_key_here

# Run with default settings (Claude Sonnet 4)
sbk-charts -i ./samples/charts/sbk-file-read.csv -o output.xlsx anthropic
```

### Advanced Usage

```bash
# Use Claude Opus for more detailed analysis
sbk-charts -i input.csv -o output.xlsx anthropic --anthropic-model anthropic-opus-4-20250514

# Adjust temperature for more creative responses
sbk-charts -i input.csv -o output.xlsx anthropic --anthropic-temperature 0.7

# Increase max tokens for longer responses
sbk-charts -i input.csv -o output.xlsx anthropic --anthropic-max-tokens 4096

# Process multiple files
sbk-charts -i file1.csv,file2.csv -o combined.xlsx anthropic
```

### Example with All Options

```bash
export ANTHROPIC_API_KEY=your_api_key_here

sbk-charts \
  -i ./samples/charts/sbk-file-read.csv,./samples/charts/sbk-rocksdb-read.csv \
  -o ./samples/charts/sbk-comparison.xlsx \
  anthropic \
  --anthropic-model anthropic-sonnet-4-20250514 \
  --anthropic-temperature 0.4 \
  --anthropic-max-tokens 2048
```

## Output

The Anthropic backend generates four types of analysis that are added to your Excel workbook:

1. **Throughput Analysis**: Examines MB/s and records/s metrics across storage systems
2. **Latency Analysis**: Analyzes latency distributions and tail latencies
3. **Total MB Analysis**: Evaluates total data transferred and processing patterns
4. **Percentile Histogram Analysis**: Detailed breakdown of latency percentile distributions

Each analysis is formatted and added to the "Summary" sheet in the output Excel file.

## Example Output

```
   _____   ____    _  __            _____   _    _              _____    _______    _____
  / ____| |  _ \  | |/ /           / ____| | |  | |     /\     |  __ \  |__   __|  / ____|
 | (___   | |_) | | ' /   ______  | |      | |__| |    /  \    | |__) |    | |    | (___
  \___ \  |  _ <  |  <   |______| | |      |  __  |   / /\ \   |  _  /     | |     \___ \
  ____) | | |_) | | . \           | |____  | |  | |  / ____ \  | | \ \     | |     ____) |
 |_____/  |____/  |_|\_\           \_____| |_|  |_| /_/    \_\ |_|  \_\    |_|    |_____/

Sbk Charts Version : 3.26.2.0
Input Files :  ./samples/charts/sbk-file-read.csv
Output File :  output.xlsx
SBK logo image found: /path/to/sbk-logo.png
xlsx file : output.xlsx created
Time Unit : NANOSECONDS
Reading : FILE
file : output.xlsx updated with graphs
Starting AI analysis. This may take a few minutes...
Running analysis in parallel with timeout: 120 seconds...
✓ Completed get_throughput_analysis
✓ Completed get_latency_analysis
✓ Completed get_total_mb_analysis
✓ Completed get_percentile_histogram_analysis
Analysis completed in 15.23 seconds
File updated with graphs and AI documentation: output.xlsx
```

## Model Selection Guide

Choose the appropriate model based on your needs:

### Claude Sonnet 4 (Default - Recommended)
- **Best for**: Most use cases
- **Speed**: Fast
- **Quality**: Excellent
- **Cost**: Moderate

### Claude Opus 4
- **Best for**: Complex analysis requiring deep reasoning
- **Speed**: Slower
- **Quality**: Best
- **Cost**: Higher

### Claude Haiku
- **Best for**: Quick analysis, high-volume processing
- **Speed**: Fastest
- **Quality**: Good
- **Cost**: Lower

## Error Handling

The implementation includes robust error handling:

- **Missing API Key**: Clear error message with instructions
- **API Failures**: Automatic error reporting with details
- **Network Issues**: Timeout handling and retry logic
- **Invalid Responses**: Graceful degradation with error messages

## Troubleshooting

### "API key not found" Error

**Solution**: Set the ANTHROPIC_API_KEY environment variable:
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

### Connection Errors

**Possible causes**:
- No internet connection
- Firewall blocking Anthropic API
- API service temporarily unavailable

**Solution**: Check your network connection and try again

### Rate Limiting

If you see rate limit errors:
- Wait a few moments before retrying
- Consider using a lower-tier model (Haiku) for high-volume analysis
- Contact Anthropic support to increase your rate limits

### Invalid Model Name

**Solution**: Use one of the supported model names:
- `claude-sonnet-4-20250514`
- `claude-opus-4-20250514`
- `claude-3-5-sonnet-20241022`
- `claude-3-5-haiku-20241022`

## Performance Tips

1. **Use Sonnet for most tasks**: It offers the best balance of speed and quality
2. **Parallel processing**: The tool runs analysis in parallel by default (4 concurrent tasks)
3. **Adjust timeout**: For large datasets, use `-secs` parameter to increase timeout:
   ```bash
   sbk-charts -i input.csv -o output.xlsx -secs 300 anthropic
   ```
4. **Sequential processing**: For debugging, use `-nothreads true`:
   ```bash
   sbk-charts -i input.csv -o output.xlsx -nothreads true anthropic
   ```

## Cost Considerations

Anthropic charges based on tokens processed:
- Input tokens (your prompts)
- Output tokens (Claude's responses)

Typical analysis costs:
- Single file: ~5,000-10,000 tokens total
- Multiple files: ~15,000-30,000 tokens total

Check current pricing at: https://www.anthropic.com/pricing

## Integration with SBK Charts

The Anthropic backend integrates seamlessly with the SBK Charts framework:

- Inherits from `SbkGenAI` base class
- Automatically discovered by the plugin system
- Uses same prompt templates as other backends
- Returns data in standard `(success, result)` tuple format

## Security Best Practices

1. **Never commit API keys**: Use environment variables or secret managers
2. **Rotate keys regularly**: Generate new API keys periodically
3. **Use separate keys**: Different keys for development and production
4. **Monitor usage**: Check your Anthropic dashboard for unexpected usage


## Support and Resources

- **Anthropic Documentation**: https://docs.anthropic.com/
- **API Reference**: https://docs.anthropic.com/claude/reference/
- **SBK Charts Issues**: https://github.com/kmgowda/sbk-charts/issues
- **Anthropic Support**: https://support.anthropic.com/

## License

This implementation is part of the SBK Charts project and is licensed under the Apache License 2.0.

## Contributing

Contributions are welcome! Please see the main SBK Charts README for contribution guidelines.
