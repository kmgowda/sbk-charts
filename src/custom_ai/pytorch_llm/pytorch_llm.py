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
from typing import Tuple, Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils.hub import cached_file, TRANSFORMERS_CACHE
from src.genai.genai import SbkGenAI
import traceback
import re
import os
import glob
import logging

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

# Configure logging
logger = logging.getLogger(__name__)


class PyTorchLLM(SbkGenAI):
    """PyTorch LLM Analysis Backend
    
    This class implements the SbkGenAI interface to provide AI-powered analysis
    using locally loaded PyTorch models. It supports any causal language model
    from the Hugging Face model hub that's compatible with PyTorch.
    
    Configuration:
    - Model: Any Hugging Face model ID or local path (default: openai/gpt-oss-20b)
    - Device: 'cuda', 'mps', or 'cpu' (auto-detects CUDA by default)
    - Max Length: Maximum sequence length for generation (default: 2048)
    - Temperature: Controls randomness (default: 0.4)
    - Top-p: Nucleus sampling parameter (default: 0.9)
    
    Attributes:
        model_name (str): Name or path of the loaded model
        device (str): Device type for model execution
        max_length (int): Maximum sequence length
        temperature (float): Sampling temperature
        top_p (float): Top-p sampling parameter
        model: The loaded PyTorch model
        tokenizer: The loaded tokenizer
        _is_initialized (bool): Whether the model is initialized
        output_list (list): List of generated outputs for training
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_name = DEFAULT_MODEL
        self.device = DEFAULT_DEVICE
        self.max_length = DEFAULT_MAX_LENGTH
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P
        self.model = None
        self.tokenizer = None
        self._is_initialized = False
        self.output_list = []

    def _initialize_model(self) -> bool:
        """Initialize the PyTorch model and tokenizer.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self._is_initialized:
            return True
            
        try:
            logger.info(f"Loading model {self.model_name} on {self.device}...")
            # Check if we have a saved model
            saved_model_dir = os.path.join(os.path.dirname(__file__), 'saved_models', self.model_name.split('/')[-1])

            # In _initialize_model, update the device and dtype handling:
            if 'cuda' in self.device and torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16  # Use BF16 if supported
                else:
                    dtype = torch.float16  # Fall back to FP16
            else:
                dtype = torch.float32

            if os.path.exists(saved_model_dir):
                logger.info(f"Loading model from {saved_model_dir}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    saved_model_dir,
                    device_map="auto",
                    dtype=dtype,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    saved_model_dir,
                    trust_remote_code=True
                )
            else:
                logger.info(f"Downloading model {self.model_name}")
                # Get the default cache directory
                logger.info(f"Default cache directory: {TRANSFORMERS_CACHE}")
                
                # Download the tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                # Download the model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    dtype=dtype,
                    trust_remote_code=True
                )
                
                # Print the model's cache location
                try:
                    # Try to get the model's configuration file path
                    config_file = None
                    if hasattr(self.model.config, 'name_or_path'):
                        # For most models, the config is in the same directory as the model
                        model_name = self.model.config.name_or_path
                        if os.path.exists(model_name):
                            # If it's a local path
                            config_file = os.path.join(model_name, 'config.json')
                        else:
                            # If it's a model name, check the cache
                            try:
                                config_file = cached_file(model_name, 'config.json')
                            except:
                                pass
                    
                    if config_file and os.path.exists(config_file):
                        model_dir = os.path.dirname(config_file)
                        logger.info(f"✅ Model files cached at: {model_dir}")
                    else:
                        # Fallback: try to find any .bin or .safetensors file in the cache
                        cache_files = glob.glob(os.path.join(TRANSFORMERS_CACHE, '**/*.bin'), recursive=True) + \
                                     glob.glob(os.path.join(TRANSFORMERS_CACHE, '**/*.safetensors'), recursive=True)
                        if cache_files:
                            model_dir = os.path.dirname(cache_files[0])
                            logger.info(f"✅ Found model weights at: {model_dir}")
                        else:
                            logger.warning(f"⚠️ Could not determine exact cache location. Using default: {TRANSFORMERS_CACHE}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not determine exact cache location: {str(e)}")
                    logger.info(f"Using default cache directory: {TRANSFORMERS_CACHE}")

            self.model = self.model.to(self.device)

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Set model to evaluation mode
            self.model.eval()

            self._is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            self.model = None
            self.tokenizer = None
            self._is_initialized = False
            return False

    def _get_device_type(self) -> str:
        """Get standardized device type string.
        
        Returns:
            str: 'cuda', 'mps', or 'cpu'
        """
        device = torch.device(self.device)
        device_str = str(device).lower()
        if 'cuda' in device_str:
            return 'cuda'
        elif 'mps' in device_str:
            return 'mps'
        else:
            return 'cpu'

    def add_args(self, parser) -> None:
        """Add command-line arguments for PyTorch LLM configuration."""
        parser.add_argument(
            "--pt-model",
            help=f"Hugging Face model ID or path (default: {DEFAULT_MODEL})",
            default=DEFAULT_MODEL
        )
        parser.add_argument(
            "--pt-train",
            action="store_true",
            help="Enable training mode (default: False)",
            default=False
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

    def parse_args(self, args) -> None:
        """Parse command-line arguments."""
        self.model_name = args.pt_model
        self.device = args.pt_device
        self.max_length = args.pt_max_length
        self.temperature = args.pt_temperature
        self.top_p = args.pt_top_p
        
        # Store training flag
        self.train_mode = args.pt_train
        

    def open(self, args) -> None:
        # Reinitialize model if needed
        if not self._is_initialized:
            if not self._initialize_model():
                logger.error("Failed to initialize model during open")
                return


    def close(self, args) -> None:
        loss = None
        # If in training mode and we have a target, train on the generated output
        if self.train_mode:
            self.model.train()
            logger.info("\n" + "=" * 50)
            logger.info("🚀 Starting training on generated output")

            for output_text in self.output_list:
                logger.info(f"📄 Generated length: {len(output_text)} chars")
                loss = self._train_on_output(output_text)
                if loss is None:
                    break
            logger.info("=" * 50)
            # Save the model after training
            if loss is not None:
                logger.info("💾 Saving trained model...")
                save_success = self._save_model()
                if save_success:
                    logger.info("✅ Model saved successfully")
                else:
                    logger.error("❌ Failed to save model")


    def _save_model(self, output_dir: str = None) -> bool:
        """Save the model and tokenizer to disk.
        
        Args:
            output_dir: Directory to save the model. If None, uses model_name.
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            if output_dir is None:
                output_dir = os.path.join(os.path.dirname(__file__), 'saved_models', self.model_name.split('/')[-1])
            
            # Validate output directory path
            if not output_dir or not isinstance(output_dir, str):
                logger.error("Invalid output directory path")
                return False
                
            # Sanitize path to prevent directory traversal
            output_dir = os.path.normpath(output_dir)
            if '..' in output_dir:
                logger.error("Directory traversal detected in output path")
                return False
            
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"💾 Saving model to {output_dir}...")
            
            # Save model and tokenizer
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"✅ Model successfully saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {str(e)}")
            return False
            
    def _train_on_output(self, generated_text: str) -> Optional[float]:
        """Train the model on the generated output.
        
        Args:
            generated_text: The text generated by the model
            
        Returns:
            float: The loss value if successful, None otherwise
        """
        # Input validation
        if not generated_text or not isinstance(generated_text, str):
            logger.error("Invalid generated text for training")
            return None
            
        if len(generated_text.strip()) == 0:
            logger.error("Empty generated text for training")
            return None
        try:
            logger.info(f"🔄 Starting training on generated output (length: {len(generated_text)} chars)...")

            # Get model's dtype for consistent typing
            model_dtype = next(self.model.parameters()).dtype
            
            # Tokenize the training text
            inputs = self.tokenizer(
                generated_text,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            
            # Move inputs to the correct device and type
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Create labels (shifted input) and ensure correct dtype
            labels = inputs["input_ids"].clone()
            
            # Ensure attention mask is the same dtype as the model
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'].to(dtype=model_dtype)
            
            # Initialize optimizer if not already done
            if not hasattr(self, 'optimizer'):
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
            
            # Zero gradients
            self.optimizer.zero_grad()

            device = torch.device(self.device)

            device_type = self._get_device_type()
            
            # Forward pass with autocast for mixed precision training
            with torch.autocast(device_type=device_type,
                              dtype=model_dtype if model_dtype in [torch.float16, torch.bfloat16] else None):
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    labels=labels
                )
            
            # Backward pass and optimize
            loss = outputs.loss
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                # Print training progress
                logger.info(f"✅ Training complete - Loss: {loss.item():.4f}")
                return loss.item()

            logger.warning("⚠️ No loss computed during training")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error during training: {str(e)}")
            traceback.print_exc()
            return None
    
    def _generate_text(self, prompt: str) -> Tuple[bool, str]:
        """Generate text using the loaded PyTorch model.
        
        Args:
            prompt: The input prompt for text generation
            
        Returns:
            tuple: (success, response) where success is a boolean and
                   response is either the generated text or an error message
        """
        # Input validation
        if not prompt or not isinstance(prompt, str):
            return False, "Invalid prompt: must be a non-empty string"
            
        if len(prompt.strip()) == 0:
            return False, "Invalid prompt: cannot be empty or whitespace only"
            
        if not self._is_initialized and not self._initialize_model():
            return False, "Model initialization failed"

        try:
            # Ensure model is on the correct device and in evaluation mode
            device = torch.device(self.device)
            
            # Get model's dtype for consistent typing
            model_dtype = next(self.model.parameters()).dtype

            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_token_type_ids=False
            )
            
            # Move inputs to the correct device and type
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs['input_ids'] = inputs['input_ids'].to(dtype=torch.long)
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'].to(dtype=torch.long)
            
            # Calculate max_new_tokens, ensuring it's at least DEFAULT_MAX_LENGTH
            max_new_tokens = max(DEFAULT_MAX_LENGTH, inputs['input_ids'].shape[1])
            
            # Generate text with appropriate settings
            with torch.no_grad():
                device_type = self._get_device_type()
                with torch.autocast(device_type=device_type, 
                                 enabled=device_type != 'cpu',
                                 dtype=model_dtype if model_dtype in [torch.float16, torch.bfloat16] else None):
                    outputs = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs.get('attention_mask', None),
                        max_new_tokens=max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        no_repeat_ngram_size=3,
                    )
            
            # Decode the generated text
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the output if it appears at the beginning
            generated_text = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text
            
            # Clean up the generated text
            cleaned_text = self._clean_generated_text(generated_text)

            self.output_list.append(cleaned_text)
            
            return True, cleaned_text
            
        except Exception as e:
            error_msg = f"Error in text generation: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return False, error_msg


    def get_model_description(self) -> Tuple[bool, str]:
        """Get a description of the current PyTorch LLM configuration.

        Returns:
            tuple: (success, description) where success is a boolean and
                  description is a string describing the configuration
        """
        if not self._is_initialized:
            return False, "Model initialization failed"
            
        desc = (f"PyTorch LLM: {self.model_name}"
                f", Device: {self.device}, Max Length: {self.max_length}"
                f", Temperature: {self.temperature}, Top-p: {self.top_p}"
                f", Model Class: {self.model.__class__.__name__ if self.model else 'Not loaded'}\n")
        return True, desc

    def _clean_generated_text(self, text: str) -> str:
        """Clean up generated text by removing multiple consecutive non-text symbols.
        
        Args:
            text: The text to clean
            
        Returns:
            str: Cleaned text with normalized punctuation and symbols
        """
        if not text:
            return text
            
        cleaned = text
        # List of special characters to clean up
        special_chars = ['.', '|', '%', '?']
        
        for char in special_chars:
            # Escape special regex characters
            escaped_char = re.escape(char)
            # Replace multiple consecutive characters with a single one
            cleaned = re.sub(fr'{escaped_char}+', char, cleaned)
            # Replace characters separated by whitespace with a single one
            cleaned = re.sub(fr'({escaped_char}\s*)+', char, cleaned)

        
        return cleaned.strip()

    def _train_on_example(self, prompt: str, target: str) -> Optional[float]:
        """Train the model on a single example.
        
        Args:
            prompt: The input prompt
            target: The target response
            
        Returns:
            float: The loss value if successful, None otherwise
        """
        try:
            # Combine prompt and target with separator
            text = f"{prompt}{target}{self.tokenizer.eos_token}"
            
            # Get model's dtype for consistent typing
            model_dtype = next(self.model.parameters()).dtype
            
            # Tokenize the input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            
            # Move inputs to the correct device and type
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Create labels (shifted input) and ensure correct dtype
            labels = inputs["input_ids"].clone()
            
            # Ensure attention mask is the same dtype as the model
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'].to(dtype=model_dtype)
            
            # Forward pass with autocast for mixed precision training
            with torch.autocast(device_type='cuda' if 'cuda' in str(self.device) else 'cpu', 
                              dtype=model_dtype):
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    labels=labels
                )
            
            # Backward pass and optimize
            loss = outputs.loss
            if loss is not None:
                loss.backward()
                return loss.item()
            return None
            
        except Exception as e:
            return None, f"Error during training: {str(e)}"

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

    def get_response(self, query: str) -> Tuple[bool, str]:
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
            
            return self._generate_text(enhanced_prompt)
            
        except Exception as e:
            return False, f"Failed to generate response for query: {str(e)}"

    def __del__(self) -> None:
        """Clean up resources when the object is destroyed."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
