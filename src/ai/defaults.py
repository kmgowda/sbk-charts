# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""Dependency-free defaults shared by AI CLI descriptors and adapters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationDefaults:
    """Common text-generation settings for a provider backend."""

    model: str
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class HuggingFaceDefaults:
    """Defaults for Hugging Face hosted inference."""

    model: str
    max_tokens: int
    temperature: float
    top_p: float


@dataclass(frozen=True)
class LmStudioDefaults:
    """Defaults for an LM Studio server."""

    url: str
    model: str
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class OllamaDefaults:
    """Defaults for an Ollama server."""

    url: str
    model: str
    temperature: float
    request_timeout_seconds: int
    health_timeout_seconds: int


@dataclass(frozen=True)
class PyTorchDefaults:
    """Defaults for the in-process PyTorch backend."""

    model: str
    train: bool
    device: str
    max_length: int
    temperature: float
    top_p: float


ANTHROPIC_DEFAULTS = GenerationDefaults(
    model="anthropic-sonnet-4-20250514",
    max_tokens=2048,
    temperature=0.4,
)
GEMINI_DEFAULTS = GenerationDefaults(
    model="gemini-2.5-flash",
    max_tokens=2048,
    temperature=0.4,
)
HUGGING_FACE_DEFAULTS = HuggingFaceDefaults(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_tokens=5000,
    temperature=0.4,
    top_p=0.9,
)
LM_STUDIO_DEFAULTS = LmStudioDefaults(
    url="http://localhost:1234/api/v0",
    model="",
    max_tokens=1800,
    temperature=0.4,
)
OLLAMA_DEFAULTS = OllamaDefaults(
    url="http://localhost:11434",
    model="llama3.1",
    temperature=0.4,
    request_timeout_seconds=120,
    health_timeout_seconds=5,
)
PYTORCH_DEFAULTS = PyTorchDefaults(
    model="openai/gpt-oss-20b",
    train=False,
    device="auto",
    max_length=2048,
    temperature=0.4,
    top_p=0.9,
)
