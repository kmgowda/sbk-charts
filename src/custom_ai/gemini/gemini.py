#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""Gemini AI Integration Module

This module provides integration with Google's Gemini AI models for generating
AI-powered analysis of storage benchmark results. It uses Gemini's advanced
reasoning capabilities to provide detailed performance insights.

Key Features:
- Cloud-based AI analysis using Google's Gemini API
- Support for multiple Gemini models (Pro, Flash, etc.)
- Configurable model parameters (temperature, max tokens)
- Automatic error handling and retry logic

Requirements:
- Valid Google AI API key
- Internet connection to Google's API endpoints
- google-genai Python package
"""

import os
import google.ai.generativelanguage as genai
from typing import Tuple
from src.genai.genai import SbkGenAI

# Default Gemini configuration
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.4


def _test_api_access(api_key):
    """Test basic API access to help debug issues."""
    try:
        # Create client with the API key
        client = genai.GenerativeServiceClient(client_options={"api_key": api_key})
        
        # List available models using the SDK
        request = genai.ListModelsRequest()
        response = client.list_models(request)
        
        model_names = [model.name for model in response.models]
        print(f"DEBUG: Available models: {model_names}")
        return True, f"Available models: {model_names}"
            
    except Exception as e:
        return False, f"API test exception: {str(e)}"


class Gemini(SbkGenAI):
    """Google Gemini AI Analysis Backend
    
    This class implements SbkGenAI interface to provide AI-powered analysis
    using Google's Gemini models via their API. It handles all communication
    with the Gemini API and formats benchmark data for effective analysis.
    
    Configuration:
    - API Key: Set via GEMINI_API_KEY environment variable
    - Model: Specify with --gemini-model (default: gemini-1.5-flash)
    - Temperature: Control response randomness with --gemini-temperature (0.0-1.0)
    - Max Tokens: Set maximum response length with --gemini-max-tokens
    
    The implementation includes automatic error handling and provides detailed
    technical analysis suitable for storage performance engineering.
    
    Attributes:
        api_key (str): Google AI API key from environment variable
        model (str): Gemini model identifier
        max_tokens (int): Maximum tokens in response
        temperature (float): Sampling temperature for response generation
    """

    def __init__(self):
        """Initialize the Gemini backend with default configuration."""
        super().__init__()
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = DEFAULT_MODEL
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.temperature = DEFAULT_TEMPERATURE
        self._client = None
        
        # Initialize the Google AI SDK client if API key is available
        if self.api_key:
            try:
                self._client = genai.GenerativeServiceClient(client_options={"api_key": self.api_key})
            except Exception as e:
                print(f"Warning: Failed to initialize Gemini client: {str(e)}")

    def add_args(self, parser):
        """Add command-line arguments for Gemini configuration.
        
        Args:
            parser: ArgumentParser instance to add arguments to
        """
        parser.add_argument(
            "--gemini-model",
            help=f"Gemini model to use (default: {DEFAULT_MODEL})",
            default=DEFAULT_MODEL
        )
        parser.add_argument(
            "--gemini-max-tokens",
            type=int,
            help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})",
            default=DEFAULT_MAX_TOKENS
        )
        parser.add_argument(
            "--gemini-temperature",
            type=float,
            help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
            default=DEFAULT_TEMPERATURE
        )

    def parse_args(self, args):
        """Parse command-line arguments.
        
        Args:
            args: Parsed arguments from ArgumentParser
        """
        self.model = args.gemini_model
        self.max_tokens = args.gemini_max_tokens
        self.temperature = args.gemini_temperature
        
        # Reinitialize the client instance with new parameters
        if self.api_key:
            try:
                self._client = genai.GenerativeServiceClient(client_options={"api_key": self.api_key})
            except Exception as e:
                print(f"Warning: Failed to reinitialize Gemini client: {str(e)}")

    def get_model_description(self) -> Tuple[bool, str]:
        """Get a description of the current Gemini configuration.

        Returns:
            tuple: (success, description) where success is a boolean and
                  description is a string describing the configuration
        """
        if not self.api_key:
            return False, "Google AI API key not found. Please set GEMINI_API_KEY environment variable."
        
        desc = (f"Google Gemini API\n"
                f" Model: {self.model}\n"
                f" Temperature: {self.temperature}\n"
                f" Max Tokens: {self.max_tokens}")
        return True, desc

    def _call_gemini_for_analysis(self, model_id, prompt):
        """Send a prompt to the configured Gemini model and return the reply.

        Parameters
        - prompt (str): the textual prompt to send to the model. Prompts are
          expected to be short technical instructions and metric tables.

        Returns
        - tuple: (True, <analysis string>) on success
                (False, <error message>) if no API key is configured or
                if another precondition prevents calling the API.

        Notes
        - Uses google-genai SDK for communication with Gemini API
        - Returns a two-element tuple following the project's existing
          convention for success/failure and message payloads.
        """
        if not self.api_key:
            return [False, (
                "Gemini analysis is not available (missing GEMINI_API_KEY environment variable). "
                "Configure to API key to enable Gemini-based analysis."
            )]

        try:
            # Use the existing client instance or create a new one if needed
            if self._client is None:
                self._client = genai.GenerativeServiceClient(client_options={"api_key": self.api_key})
            
            # Create the content request
            content = genai.Content(
                parts=[genai.Part(text=prompt)]
            )
            
            # Create the generation request
            request = genai.GenerateContentRequest(
                model=f"models/{model_id}",
                contents=[content],
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            
            # Generate content
            response = self._client.generate_content(request)
            
            # Extract the text from the response
            if response.candidates and response.candidates[0].content:
                text_parts = [part.text for part in response.candidates[0].content.parts if part.text]
                if text_parts:
                    return (True, "".join(text_parts).strip())
            
            return (False, "No content in response")
                
        except Exception as e:
            error_msg = f"Error calling Gemini API: {str(e)}"
            return (False, error_msg)

    def get_throughput_analysis(self) -> Tuple[bool, str]:
        """Generate throughput analysis using Gemini.
        
        Returns:
            tuple: (success, analysis) where success is a boolean and
                   analysis is the generated text or error message
        """
        try:
            prompt = self.get_throughput_prompt()
            return self._call_gemini_for_analysis(self.model, prompt)
        except Exception as e:
            return False, f"Failed to generate throughput analysis: {str(e)}"

    def get_latency_analysis(self) -> Tuple[bool, str]:
        """Generate latency analysis using Gemini.
        
        Returns:
            tuple: (success, analysis) where success is a boolean and
                   analysis is the generated text or error message
        """
        try:
            prompt = self.get_latency_prompt()
            return self._call_gemini_for_analysis(self.model, prompt)
        except Exception as e:
            return False, f"Failed to generate latency analysis: {str(e)}"

    def get_total_mb_analysis(self) -> Tuple[bool, str]:
        """Generate total MB processed analysis using Gemini.
        
        Returns:
            tuple: (success, analysis) where success is a boolean and
                   analysis is the generated text or error message
        """
        try:
            prompt = self.get_total_mb_prompt()
            return self._call_gemini_for_analysis(self.model, prompt)
        except Exception as e:
            return False, f"Failed to generate total MB analysis: {str(e)}"

    def get_percentile_histogram_analysis(self) -> Tuple[bool, str]:
        """Generate percentile histogram analysis using Gemini.
        
        Returns:
            tuple: (success, analysis) where success is a boolean and
                   analysis is the generated text or error message
        """
        try:
            prompt = self.get_percentile_histogram_prompt()
            return self._call_gemini_for_analysis(self.model, prompt)
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
                Provide a short technical analysis that addresses the query comprehensively."""

            # Enhance with RAG context
            enhanced_prompt = self._enhance_prompt_with_rag(prompt, query)

            return self._call_gemini_for_analysis(self.model, enhanced_prompt)
        except Exception as e:
            return False, f"Failed to generate response for query: {str(e)}"

    def close(self, args):
        """Close the Gemini client connection.
        
        Note: REST API doesn't require explicit connection closing,
        but this method is provided for interface compatibility.
        """
        pass
