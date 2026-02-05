#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""Anthropic Claude AI Integration Module

This module provides integration with Anthropic's Claude API for generating
AI-powered analysis of storage benchmark results. It uses Claude's advanced
reasoning capabilities to provide detailed performance insights.

Key Features:
- Cloud-based AI analysis using Anthropic's Claude API
- Support for multiple Claude models (Sonnet, Opus, Haiku)
- Configurable model parameters (temperature, max tokens)
- Automatic error handling and retry logic

Requirements:
- Anthropic Python SDK (anthropic package)
- Valid Anthropic API key
- Internet connection to Anthropic's API endpoints
"""

import os
from typing import Tuple
from anthropic import Anthropic as AnthropicClient
from src.genai.genai import SbkGenAI

# Default Anthropic configuration
DEFAULT_MODEL = "anthropic-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.4


class Anthropic(SbkGenAI):
    """Anthropic Claude AI Analysis Backend
    
    This class implements the SbkGenAI interface to provide AI-powered analysis
    using Anthropic's Claude models via their API. It handles all communication
    with the Anthropic API and formats benchmark data for effective analysis.
    
    Configuration:
    - API Key: Set via ANTHROPIC_API_KEY environment variable
    - Model: Specify with --anthropic-model (default: anthropic-sonnet-4-20250514)
    - Temperature: Control response randomness with --anthropic-temperature (0.0-1.0)
    - Max Tokens: Set maximum response length with --anthropic-max-tokens
    
    The implementation includes automatic error handling and provides detailed
    technical analysis suitable for storage performance engineering.
    
    Attributes:
        api_key (str): Anthropic API key from environment variable
        model (str): Claude model identifier
        max_tokens (int): Maximum tokens in response
        temperature (float): Sampling temperature for response generation
        client: Anthropic API client instance
    """

    def __init__(self):
        """Initialize the Anthropic backend with default configuration."""
        super().__init__()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = DEFAULT_MODEL
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.temperature = DEFAULT_TEMPERATURE
        self.client = None

    def _initialize_client(self) -> bool:
        """Initialize the Anthropic API client.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self.client is not None:
            return True
            
        if not self.api_key:
            return False
            
        try:
            self.client = AnthropicClient(api_key=self.api_key)
            return True
        except Exception as e:
            print(f"Failed to initialize Anthropic client: {str(e)}")
            return False

    def add_args(self, parser):
        """Add command-line arguments for Anthropic configuration.
        
        Args:
            parser: ArgumentParser instance to add arguments to
        """
        parser.add_argument(
            "--anthropic-model",
            help=f"Anthropic Claude model to use (default: {DEFAULT_MODEL})",
            default=DEFAULT_MODEL
        )
        parser.add_argument(
            "--anthropic-max-tokens",
            type=int,
            help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})",
            default=DEFAULT_MAX_TOKENS
        )
        parser.add_argument(
            "--anthropic-temperature",
            type=float,
            help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
            default=DEFAULT_TEMPERATURE
        )

    def parse_args(self, args):
        """Parse command-line arguments.
        
        Args:
            args: Parsed arguments from ArgumentParser
        """
        self.model = args.anthropic_model
        self.max_tokens = args.anthropic_max_tokens
        self.temperature = args.anthropic_temperature

    def open(self, args) -> None:
        if not self.api_key:
            print("Anthropic API key not found. Please set the ANTHROPIC_API_KEY, environment variable with your API key." )
            return
        self._initialize_client()


    def close(self, args):
        """Close the Anthropic client connection."""
        if self.client is not None:
            self.client.close()
            self.client = None

    def get_model_description(self) -> Tuple[bool, str]:
        """Get a description of the current Anthropic configuration.

        Returns:
            tuple: (success, description) where success is a boolean and
                  description is a string describing the configuration
        """
        if not self.api_key:
            return False, "Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable."
        
        desc = (f"Anthropic Claude API\n"
                f" Model: {self.model}\n"
                f" Temperature: {self.temperature}\n"
                f" Max Tokens: {self.max_tokens}")
        return True, desc

    def _call_claude(self, prompt: str) -> Tuple[bool, str]:
        """Send a prompt to Claude and get the response.

        Args:
            prompt: The prompt to send to Claude

        Returns:
            tuple: (success, response) where success is a boolean and
                   response is either the generated text or an error message
        """


        try:
            # Create a message using Claude's API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract the text from the response
            if message.content and len(message.content) > 0:
                # The content is a list of ContentBlock objects
                text_content = message.content[0].text
                return True, text_content.strip()
            else:
                return False, "Empty response from Claude API"

        except Exception as e:
            error_msg = f"Error calling Anthropic API: {str(e)}"
            print(error_msg)
            return False, error_msg

    def get_throughput_analysis(self) -> Tuple[bool, str]:
        """Generate throughput analysis using Claude.
        
        Returns:
            tuple: (success, analysis) where success is a boolean and
                   analysis is the generated text or error message
        """
        try:
            prompt = self.get_throughput_prompt()
            return self._call_claude(prompt)
        except Exception as e:
            return False, f"Failed to generate throughput analysis: {str(e)}"

    def get_latency_analysis(self) -> Tuple[bool, str]:
        """Generate latency analysis using Claude.
        
        Returns:
            tuple: (success, analysis) where success is a boolean and
                   analysis is the generated text or error message
        """
        try:
            prompt = self.get_latency_prompt()
            return self._call_claude(prompt)
        except Exception as e:
            return False, f"Failed to generate latency analysis: {str(e)}"

    def get_total_mb_analysis(self) -> Tuple[bool, str]:
        """Generate total MB processed analysis using Claude.
        
        Returns:
            tuple: (success, analysis) where success is a boolean and
                   analysis is the generated text or error message
        """
        try:
            prompt = self.get_total_mb_prompt()
            return self._call_claude(prompt)
        except Exception as e:
            return False, f"Failed to generate total MB analysis: {str(e)}"

    def get_percentile_histogram_analysis(self) -> Tuple[bool, str]:
        """Generate percentile histogram analysis using Claude.
        
        Returns:
            tuple: (success, analysis) where success is a boolean and
                   analysis is the generated text or error message
        """
        try:
            prompt = self.get_percentile_histogram_prompt()
            return self._call_claude(prompt)
        except Exception as e:
            return False, f"Failed to generate percentile histogram analysis: {str(e)}"

    def get_response(self, query) -> Tuple[bool, str]:
        """
          Generate a response for a custom query using RAG-enhanced context.

          Args:
              query: The query string to analyze

          Returns:
              Tuple[bool, str]: A tuple containing:
                  - bool: True if analysis was successful, False otherwise
                  - str: The analysis text or error message if analysis failed
          """
        try:
            # Create a prompt for the custom query
            prompt = f"""You are a storage performance engineer. Please analyze the following query based on the provided context:
                  Query: {query}
                  provide a short technical analysis that addresses the query comprehensively."""

            # Enhance with RAG context
            enhanced_prompt = self._enhance_prompt_with_rag(prompt, query)

            return  self._call_claude(enhanced_prompt)

        except Exception as e:
            return False, f"Failed to generate response for query: {str(e)}"

