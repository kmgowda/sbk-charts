#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##
"""LM Studio AI Integration Module

This module provides integration with LM Studio's local inference server for
generating AI-powered analysis of storage benchmark results. It allows running
AI analysis locally without requiring cloud-based AI services.

Key Features:
- Local model inference using LM Studio
- Support for various open-source LLMs
- Configurable model parameters (temperature, max tokens)
- Automatic reconnection on connection failures

Requirements:
- LM Studio application running locally
- Compatible LLM model loaded in LM Studio
- Network access to the LM Studio server (default: localhost:1234)
"""

import lmstudio
from src.genai.genai import SbkGenAI

# Default LM Studio configuration
BASE_URL = "http://localhost:1234/api/v0"


class LmStudio(SbkGenAI):
    """LM Studio AI Analysis Backend
    
    This class implements the SbkGenAI interface to provide AI-powered analysis
    using a locally running LM Studio instance. It handles communication with
    the LM Studio server and formats the benchmark data for analysis.
    
    Configuration:
    - Server URL: Configurable, defaults to http://localhost:1234/api/v0
    - Model: Can be specified via command line (default: LM Studio's loaded model)
    - Temperature: Controls randomness (default: 0.4)
    - Max Tokens: Limits response length (default: 1800)
    
    The class automatically handles connection management and error recovery.
    """

    def __init__(self):
        super().__init__()
        self.url = BASE_URL
        self.model = ""  # Will use LM Studio's default loaded model
        self.temperature = 0.4
        self.max_tokens = 1800
        self.llm_model = None

    def add_args(self, parser):
        """Add command-line arguments for LM Studio configuration."""
        parser.add_argument(
            "--url",
            help=f"server url (default: {BASE_URL})",
            default=BASE_URL
        )
        parser.add_argument(
            "--lm-model",
            help="Model name or path to use (default: LM Studio's loaded model)",
            default=""
        )
        parser.add_argument(
            "--lm-temperature",
            type=float,
            help="Sampling temperature (default: 0.4)",
            default=0.4
        )
        parser.add_argument(
            "--lm-max-tokens",
            type=int,
            help="Maximum number of tokens to generate (default: 1800)",
            default=1800
        )

    def parse_args(self, args):
        """Parse command-line arguments."""
        self.url = args.url
        self.model = args.lm_model
        self.temperature = args.lm_temperature
        self.max_tokens = args.lm_max_tokens

    def open(self, args) -> None:
        # Get the model handle (use default or specified model)
        if self.model:
            self.llm_model = lmstudio.llm(self.model)
        else:
            self.llm_model = lmstudio.llm()

    def close(self, args) -> None:
        """Close the LM Studio model and clean up resources."""
        try:
            # LM Studio's LLM object doesn't have a close method
            # Just clear the reference
            self.llm_model = None
        except Exception as e:
            print(f"Warning: Error closing LM Studio model: {str(e)}")
            self.llm_model = None


    def get_model_description(self) -> tuple[bool, str]:
        """Get a description of the current LM Studio configuration.

        Returns:
            tuple: (success, description) where success is a boolean and
                  description is a string describing the configuration
        """
        desc = (f" LM Studio at {self.url},\n"
                f" Model: {self.model or 'LM Studio default'},\n"
                f" Temperature: {self.temperature}, Max Tokens: {self.max_tokens}")
        return True, desc

    def _call_analysis(self, prompt: str) -> tuple[bool, str]:
        """Send a prompt to the local LM Studio instance and get the response.

        Args:
            prompt: The prompt to send to the model

        Returns:
            tuple: (success, response) where success is a boolean and
                   response is either the generated text or an error message
        """
        try:

            # Create a chat with the prompt
            chat = lmstudio.Chat()
            chat.add_user_message(prompt)
            
            # Configure inference parameters
            config = {
                "temperature": self.temperature,
                "maxTokens": self.max_tokens
            }
            
            # Generate response
            response = self.llm_model.respond(chat, config=config)
            
            # Extract the generated text from PredictionResult
            generated_text = response.content.strip()
            return True, generated_text

        except Exception as e:
            return False, f"Error in LM Studio analysis: {str(e)}"

    def get_throughput_analysis(self) -> tuple[bool, str]:
        """Generate throughput analysis using LM Studio."""
        try:
            prompt = self.get_throughput_prompt()
            return self._call_analysis(prompt)
        except Exception as e:
            return False, f"Failed to generate throughput analysis: {str(e)}"

    def get_latency_analysis(self) -> tuple[bool, str]:
        """Generate latency analysis using LM Studio."""
        try:
            prompt = self.get_latency_prompt()
            return self._call_analysis(prompt)
        except Exception as e:
            return False, f"Failed to generate latency analysis: {str(e)}"

    def get_total_mb_analysis(self) -> tuple[bool, str]:
        """Generate total MB processed analysis using LM Studio."""
        try:
            prompt = self.get_total_mb_prompt()
            return self._call_analysis(prompt)
        except Exception as e:
            return False, f"Failed to generate total MB analysis: {str(e)}"

    def get_percentile_histogram_analysis(self) -> tuple[bool, str]:
        """Generate percentile histogram analysis using LM Studio."""
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

Please provide a detailed technical analysis that addresses the query comprehensively. Use the contextual information provided to give specific and accurate insights."""
            
            # Enhance with RAG context
            enhanced_prompt = self._enhance_prompt_with_rag(prompt, query)
            
            return self._call_analysis(enhanced_prompt)
            
        except Exception as e:
            return False, f"Failed to generate response for query: {str(e)}"