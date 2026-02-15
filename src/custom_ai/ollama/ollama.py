#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""Ollama AI Integration Module

This module provides integration with the Ollama server for local LLM inference,
enabling AI-powered analysis of storage benchmark results without requiring
cloud services. It supports any model compatible with the Ollama runtime.

Key Features:
- Local LLM inference using Ollama
- Support for any Ollama-compatible models (e.g., llama3, mistral)
- Configurable model parameters and timeouts
- Automatic server health checking
- Connection pooling for performance

Requirements:
- Ollama server running locally (https://ollama.ai/)
- Desired models pre-downloaded via 'ollama pull <model>'
- Network access to the Ollama server (default: localhost:11434)
"""

import requests
from src.genai.genai import SbkGenAI

# Default Ollama configuration
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"  # Default model


class Ollama(SbkGenAI):
    """Ollama AI Analysis Backend
    
    This class implements the SbkGenAI interface to provide AI-powered analysis
    using a locally running Ollama server. It handles all communication with
    the Ollama REST API and formats benchmark data for effective analysis.
    
    Configuration Options:
    - Server URL: Configurable via --ollama-url (default: http://localhost:11434)
    - Model: Specify with --ollama-model (default: llama3.1)
    - Temperature: Control response randomness with --ollama-temperature (0.0-1.0)
    - Timeout: Set request timeout with --ollama-timeout (seconds)
    
    The implementation includes automatic reconnection and error handling to
    ensure robust operation even if the Ollama server is temporarily unavailable.
    """

    def __init__(self):
        super().__init__()
        self.base_url = DEFAULT_BASE_URL
        self.model = DEFAULT_MODEL
        self.temperature = 0.4
        self.timeout = 120  # seconds
        self.session = None

    def _create_session(self):
        """Create a requests session with proper headers."""
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update({
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })

    def _check_server(self) -> bool:
        """Check if Ollama server is running and accessible.

        Returns:
            bool: True if server is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def add_args(self, parser):
        """Add command-line arguments for Ollama configuration."""
        parser.add_argument(
            "-url",
            "--ollama-url",
            help=f"Ollama server URL (default: {DEFAULT_BASE_URL})",
            default=DEFAULT_BASE_URL
        )
        parser.add_argument(
            "-model",
            "--ollama-model",
            help=f"Model name to use (default: {DEFAULT_MODEL})",
            default=DEFAULT_MODEL
        )
        parser.add_argument(
            "-tmp",
            "--ollama-temperature",
            type=float,
            help="Sampling temperature (default: 0.4)",
            default=0.4
        )
        parser.add_argument(
            "-timeout",
            "--ollama-timeout",
            type=int,
            help="Request timeout in seconds (default: 120)",
            default=120
        )

    def parse_args(self, args):
        """Parse command-line arguments."""
        self.base_url = args.ollama_url.rstrip('/')  # Remove trailing slash if present
        self.model = args.ollama_model
        self.temperature = args.ollama_temperature
        self.timeout = args.ollama_timeout
        self._create_session()

    def get_model_description(self) -> tuple[bool, str]:
        """Get a description of the current Ollama configuration.

        Returns:
            tuple: (success, description) where success is a boolean and
                  description is a string describing the configuration
        """
        server_status = "running" if self._check_server() else "not responding"
        desc = (f"Ollama at {self.base_url} ({server_status}), "
                f"Model: {self.model}, "
                f"Temperature: {self.temperature}, "
                f"Timeout: {self.timeout}s")
        return True, desc

    def _call_ollama(self, prompt: str) -> tuple[bool, str]:
        """Send a prompt to the Ollama API and get the response.

        Args:
            prompt: The prompt to send to the model

        Returns:
            tuple: (success, response) where success is a boolean and
                   response is either the generated text or an error message
        """
        if not self._check_server():
            return False, "Ollama server is not running or not accessible"

        try:
            # Ollama's chat completion API endpoint
            url = f"{self.base_url}/api/chat"

            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "options": {
                    "temperature": self.temperature
                },
                "stream": False  # Explicitly disable streaming
            }

            # Make the request with a longer timeout
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            # Debug: Print raw response for troubleshooting
            response_text = response.text

            # Try to parse the response
            try:
                result = response.json()
            except ValueError as e:
                # If JSON parsing fails, try to handle the response as text
                if response_text.strip().startswith('{') and 'message' in response_text:
                    # Try to extract content from malformed JSON
                    try:
                        import re
                        content_match = re.search(r'"content":"(.*?)"', response_text)
                        if content_match:
                            return True, content_match.group(1).replace('\\n', '\n')
                    except:
                        pass
                return False, f"Failed to parse Ollama response: {str(e)}. Response: {response_text[:200]}"

            # Handle different response formats
            if isinstance(result, dict):
                if 'message' in result and 'content' in result['message']:
                    return True, result['message']['content'].strip()
                elif 'response' in result:
                    return True, result['response'].strip()
                elif 'content' in result:
                    return True, result['content'].strip()
                else:
                    return False, f"Unexpected response format from Ollama: {str(result)[:200]}"
            elif isinstance(result, str):
                return True, result.strip()
            else:
                return False, f"Unexpected response type from Ollama: {type(result)}"

        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.text
                    error_msg = f"{error_msg} - Response: {error_detail[:200]}"
                except:
                    error_msg = f"{error_msg} - Status: {e.response.status_code}"
            return False, f"Ollama API error: {error_msg}"
        except Exception as e:
            return False, f"Error in Ollama API call: {str(e)}"


    def _call_analysis(self, prompt: str) -> tuple[bool, str]:
        """Wrapper for analysis calls with retry logic."""
        max_retries = 2
        for attempt in range(max_retries):
            success, result = self._call_ollama(prompt)
            if success or attempt == max_retries - 1:
                return success, result
            print(f"Retrying ({attempt + 1}/{max_retries})...")
        return False, "Max retries reached"

    def get_throughput_analysis(self) -> tuple[bool, str]:
        """Generate throughput analysis using Ollama."""
        try:
            prompt = self.get_throughput_prompt()
            return self._call_analysis(prompt)
        except Exception as e:
            return False, f"Failed to generate throughput analysis: {str(e)}"

    def get_latency_analysis(self) -> tuple[bool, str]:
        """Generate latency analysis using Ollama."""
        try:
            prompt = self.get_latency_prompt()
            return self._call_analysis(prompt)
        except Exception as e:
            return False, f"Failed to generate latency analysis: {str(e)}"

    def get_total_mb_analysis(self) -> tuple[bool, str]:
        """Generate total MB processed analysis using Ollama."""
        try:
            prompt = self.get_total_mb_prompt()
            return self._call_analysis(prompt)
        except Exception as e:
            return False, f"Failed to generate total MB analysis: {str(e)}"

    def get_percentile_histogram_analysis(self) -> tuple[bool, str]:
        """Generate percentile histogram analysis using Ollama."""
        try:
            prompt = self.get_percentile_histogram_prompt()
            return self._call_analysis(prompt)
        except Exception as e:
            return False, f"Failed to generate percentile histogram analysis: {str(e)}"

    def get_response(self, query: str) -> tuple[bool, str]:
        """
        Generate a response for a custom query using RAG-enhanced context.

        Args:
            query: The query string to analyze

        Returns:
            tuple: (success, response) where success is a boolean and
                   response is either the generated text or an error message
        """
        try:
            # Create a prompt for the custom query
            prompt = f"""You are a storage performance engineer. Please analyze the following query based on the provided context:

Query: {query}

Please provide a short technical analysis that addresses the query comprehensively. Use the contextual information provided to give specific and accurate insights."""
            
            # Enhance with RAG context
            enhanced_prompt = self._enhance_prompt_with_rag(prompt, query)
            
            return self._call_analysis(enhanced_prompt)
            
        except Exception as e:
            return False, f"Failed to generate response for query: {str(e)}"

    def __del__(self):
        """Clean up resources."""
        if self.session:
            self.session.close()