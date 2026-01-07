#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""sbk_charts.custom_ai.local_lm

Integration with LM Studio's Python SDK for local model inference.
This module provides the LocalLMAnalysis class that interfaces with
a locally running LM Studio instance for generating AI analysis.
"""

from lmstudio import LMStudioClient
from src.genai.genai import SbkGenAI

# Default LM Studio configuration
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1234
DEFAULT_TIMEOUT = 60  # seconds


class lm_studio(SbkGenAI):
    """Adapter for using LM Studio's Python SDK for local model inference.

    This class provides an interface to a locally running LM Studio instance
    for generating AI-powered analysis of storage benchmark results.
    """

    def __init__(self):
        super().__init__()
        self.host = DEFAULT_HOST
        self.port = DEFAULT_PORT
        self.timeout = DEFAULT_TIMEOUT
        self.model = None
        self.temperature = 0.4
        self.max_tokens = 1800
        self.client = None

    def _connect(self) -> bool:
        """Establish connection to the LM Studio server.

        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            self.client = LMStudioClient(host=self.host, port=self.port, timeout=self.timeout)
            return True
        except Exception as e:
            print(f"Failed to connect to LM Studio: {str(e)}")
            return False

    def add_args(self, parser):
        """Add command-line arguments for LM Studio configuration."""
        parser.add_argument(
            "--lm-host",
            help=f"LM Studio server host (default: {DEFAULT_HOST})",
            default=DEFAULT_HOST
        )
        parser.add_argument(
            "--lm-port",
            type=int,
            help=f"LM Studio server port (default: {DEFAULT_PORT})",
            default=DEFAULT_PORT
        )
        parser.add_argument(
            "--lm-timeout",
            type=int,
            help=f"Connection timeout in seconds (default: {DEFAULT_TIMEOUT})",
            default=DEFAULT_TIMEOUT
        )
        parser.add_argument(
            "--lm-model",
            help="Model name or path to use (default: None, uses LM Studio's selected model)",
            default=None
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
        self.host = args.lm_host
        self.port = args.lm_port
        self.timeout = args.lm_timeout
        self.model = args.lm_model
        self.temperature = args.lm_temperature
        self.max_tokens = args.lm_max_tokens

    def get_model_description(self) -> tuple[bool, str]:
        """Get a description of the current LM Studio configuration.

        Returns:
            tuple: (success, description) where success is a boolean and
                  description is a string describing the configuration
        """
        desc = (f"LM Studio at {self.host}:{self.port}\n"
                f"Model: {self.model or 'LM Studio default'}\n"
                f"Temperature: {self.temperature}, Max Tokens: {self.max_tokens}")
        return True, desc

    def _call_analysis(self, prompt: str) -> tuple[bool, str]:
        """Send a prompt to the local LM Studio instance and get the response.

        Args:
            prompt: The prompt to send to the model

        Returns:
            tuple: (success, response) where success is a boolean and
                   response is either the generated text or an error message
        """
        if not self._connect():
            return False, "Failed to connect to LM Studio server"

        try:
            # Create a chat completion
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )

            # Extract the generated text
            generated_text = response.choices[0].message.content.strip()
            return True, generated_text

        except Exception as e:
            return False, f"Error in LM Studio analysis: {str(e)}"

        finally:
            # Clean up the client
            try:
                if self.client:
                    self.client.close()
            except:
                pass

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