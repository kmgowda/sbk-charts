#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##
"""PyTorch LLM Integration Module

This module provides integration with local PyTorch-based language models for
generating AI-powered analysis of storage benchmark results. It allows running
AI analysis locally using PyTorch's native inference capabilities.

Key Features:
- Local model inference using PyTorch
- Support for Hugging Face models through PyTorch
- Configurable model parameters (temperature, max tokens, device)
- Automatic model loading and management

Requirements:
- PyTorch
- Transformers library (for Hugging Face models)
- A compatible pre-trained language model (e.g., from Hugging Face)
"""

import torch
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from src.genai.genai import SbkGenAI

# Default model configuration
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() 
    else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
    else "cpu"
)
DEFAULT_MAX_LENGTH = 2048
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.9


class PyTorchLLM(SbkGenAI):
    """PyTorch LLM Analysis Backend
    
    This class implements the SbkGenAI interface to provide AI-powered analysis
    using locally loaded PyTorch models. It supports any causal language model
    from the Hugging Face model hub that's compatible with PyTorch.
    
    Configuration:
    - Model: Any Hugging Face model ID or local path (default: gpt2)
    - Device: 'cuda', 'mps', or 'cpu' (auto-detects CUDA by default)
    - Max Length: Maximum sequence length for generation (default: 512)
    - Temperature: Controls randomness (default: 0.7)
    - Top-p: Nucleus sampling parameter (default: 0.9)
    """

    def __init__(self):
        super().__init__()
        self.model_name = DEFAULT_MODEL
        self.device = DEFAULT_DEVICE
        self.max_length = DEFAULT_MAX_LENGTH
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P
        self.model = None
        self.tokenizer = None
        self._is_initialized = False

    def _initialize_model(self) -> bool:
        """Initialize the PyTorch model and tokenizer.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self._is_initialized:
            return True
            
        try:
            print(f"Loading model {self.model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # In _initialize_model, update the device and dtype handling:
            if 'cuda' in self.device and torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16  # Use BF16 if supported
                else:
                    dtype = torch.float16  # Fall back to FP16
            else:
                dtype = torch.float32

            # Load model with appropriate device mapping
            if self.device.startswith('cuda'):
                # For CUDA, use device_map='auto' for better memory management
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=dtype,
                    device_map='auto',
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            else:
                # For CPU/MPS, first load to CPU then move to device if needed
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=dtype,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                
                # Move to device using to_empty() for meta tensors
                if any(t.is_meta for t in self.model.parameters()):
                    self.model = self.model.to_empty(device=self.device)
                    self.model.load_state_dict(
                        AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            dtype=dtype,
                            device_map=None,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True
                        ).state_dict(),
                        assign=True
                    )
                else:
                    self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            self._is_initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize model: {str(e)}")
            self.model = None
            self.tokenizer = None
            self._is_initialized = False
            return False

    def add_args(self, parser):
        """Add command-line arguments for PyTorch LLM configuration."""
        parser.add_argument(
            "--pt-model",
            help=f"Hugging Face model ID or path (default: {DEFAULT_MODEL})",
            default=DEFAULT_MODEL
        )
        parser.add_argument(
            "--pt-device",
            help=f"Device to run the model on (default: {DEFAULT_DEVICE})",
            default=DEFAULT_DEVICE
        )
        parser.add_argument(
            "--pt-max-length",
            type=int,
            help=f"Maximum sequence length (default: {DEFAULT_MAX_LENGTH})",
            default=DEFAULT_MAX_LENGTH
        )
        parser.add_argument(
            "--pt-temperature",
            type=float,
            help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
            default=DEFAULT_TEMPERATURE
        )
        parser.add_argument(
            "--pt-top-p",
            type=float,
            help=f"Top-p sampling parameter (default: {DEFAULT_TOP_P})",
            default=DEFAULT_TOP_P
        )

    def parse_args(self, args):
        """Parse command-line arguments."""
        self.model_name = args.pt_model
        self.device = args.pt_device
        self.max_length = args.pt_max_length
        self.temperature = args.pt_temperature
        self.top_p = args.pt_top_p
        
        # Reinitialize model if needed
        if not self._is_initialized:
            self._initialize_model()
            self._is_initialized = True

    def get_model_description(self) -> Tuple[bool, str]:
        """Get a description of the current PyTorch LLM configuration.

        Returns:
            tuple: (success, description) where success is a boolean and
                  description is a string describing the configuration
        """
        if not self._is_initialized:
            return False, "Model initialization failed"
            
        desc = (f"PyTorch LLM: {self.model_name}\n"
                f"Device: {self.device}, Max Length: {self.max_length}\n"
                f"Temperature: {self.temperature}, Top-p: {self.top_p}\n"
                f"Model Class: {self.model.__class__.__name__ if self.model else 'Not loaded'}")
        return True, desc

    def _generate_text(self, prompt: str) -> Tuple[bool, str]:
        """Generate text using the loaded PyTorch model.
        
        Args:
            prompt: The input prompt for text generation
            
        Returns:
            tuple: (success, response) where success is a boolean and
                   response is either the generated text or an error message
        """
        if not self._is_initialized and not self._initialize_model():
            return False, "Model initialization failed"
            
        try:
            # Ensure model is on the correct device
            device = torch.device(self.device)
            self.model = self.model.to(device)

            prompt =  "Role: user, Content: " + prompt

            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_token_type_ids=False
            )
            # Move all input tensors to the same device as the model
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Calculate max_new_tokens, ensuring it's at least 1
            max_new_tokens = max(1, min(4096, self.max_length - inputs['input_ids'].shape[1]))

            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                )

            # Decode and clean up the output
            full_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            # Remove the input prompt from the output
            generated_text = full_text
            if full_text.startswith(prompt):
                generated_text = full_text[len(prompt):].strip()
            
            return True, generated_text
            
        except Exception as e:
            return False, f"Error in text generation: {str(e)}"

    def get_throughput_analysis(self) -> Tuple[bool, str]:
        """Generate throughput analysis using the PyTorch LLM."""
        try:
            prompt = self.get_throughput_prompt()
            return self._generate_text(prompt)
        except Exception as e:
            return False, f"Failed to generate throughput analysis: {str(e)}"

    def get_latency_analysis(self) -> Tuple[bool, str]:
        """Generate latency analysis using the PyTorch LLM."""
        try:
            prompt = self.get_latency_prompt()
            return self._generate_text(prompt)
        except Exception as e:
            return False, f"Failed to generate latency analysis: {str(e)}"

    def get_total_mb_analysis(self) -> Tuple[bool, str]:
        """Generate total MB processed analysis using the PyTorch LLM."""
        try:
            prompt = self.get_total_mb_prompt()
            return self._generate_text(prompt)
        except Exception as e:
            return False, f"Failed to generate total MB analysis: {str(e)}"

    def get_percentile_histogram_analysis(self) -> Tuple[bool, str]:
        """Generate percentile histogram analysis using the PyTorch LLM."""
        try:
            prompt = self.get_percentile_histogram_prompt()
            return self._generate_text(prompt)
        except Exception as e:
            return False, f"Failed to generate percentile histogram analysis: {str(e)}"

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
