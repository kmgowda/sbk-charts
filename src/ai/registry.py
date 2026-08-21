# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""Static AI backend registry with lazy implementation imports."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

from src.ai.defaults import (
    ANTHROPIC_DEFAULTS,
    GEMINI_DEFAULTS,
    HUGGING_FACE_DEFAULTS,
    LM_STUDIO_DEFAULTS,
    OLLAMA_DEFAULTS,
    PYTORCH_DEFAULTS,
)


@dataclass(frozen=True)
class BackendDescriptor:
    """Describe one backend without importing its optional SDK."""

    module: str
    class_name: str
    add_arguments: Callable[[object], None]

    def load_class(self) -> type:
        """Import and return the backend implementation class."""
        module = importlib.import_module(self.module)
        return getattr(module, self.class_name)


def _anthropic(parser: object) -> None:
    parser.add_argument("--anthropic-model", default=ANTHROPIC_DEFAULTS.model)
    parser.add_argument(
        "--anthropic-max-tokens", type=int, default=ANTHROPIC_DEFAULTS.max_tokens
    )
    parser.add_argument(
        "--anthropic-temperature", type=float, default=ANTHROPIC_DEFAULTS.temperature
    )


def _gemini(parser: object) -> None:
    parser.add_argument("--gemini-model", default=GEMINI_DEFAULTS.model)
    parser.add_argument("--gemini-max-tokens", type=int, default=GEMINI_DEFAULTS.max_tokens)
    parser.add_argument(
        "--gemini-temperature", type=float, default=GEMINI_DEFAULTS.temperature
    )


def _huggingface(parser: object) -> None:
    parser.add_argument(
        "-id",
        "--model_id",
        default=HUGGING_FACE_DEFAULTS.model,
        help="Hugging Face model ID",
    )


def _lmstudio(parser: object) -> None:
    parser.add_argument("--url", default=LM_STUDIO_DEFAULTS.url)
    parser.add_argument("--lm-model", default=LM_STUDIO_DEFAULTS.model)
    parser.add_argument(
        "--lm-temperature", type=float, default=LM_STUDIO_DEFAULTS.temperature
    )
    parser.add_argument("--lm-max-tokens", type=int, default=LM_STUDIO_DEFAULTS.max_tokens)


def _no_arguments(_parser: object) -> None:
    return


def _ollama(parser: object) -> None:
    parser.add_argument("-url", "--ollama-url", default=OLLAMA_DEFAULTS.url)
    parser.add_argument("-model", "--ollama-model", default=OLLAMA_DEFAULTS.model)
    parser.add_argument(
        "-tmp", "--ollama-temperature", type=float, default=OLLAMA_DEFAULTS.temperature
    )
    parser.add_argument(
        "-timeout",
        "--ollama-timeout",
        type=int,
        default=OLLAMA_DEFAULTS.request_timeout_seconds,
    )


def _pytorchllm(parser: object) -> None:
    parser.add_argument("--pt-model", default=PYTORCH_DEFAULTS.model)
    parser.add_argument(
        "--pt-train", action="store_true", default=PYTORCH_DEFAULTS.train
    )
    parser.add_argument("--pt-device", default=PYTORCH_DEFAULTS.device)
    parser.add_argument("--pt-max-length", type=int, default=PYTORCH_DEFAULTS.max_length)
    parser.add_argument(
        "--pt-temperature", type=float, default=PYTORCH_DEFAULTS.temperature
    )
    parser.add_argument("--pt-top-p", type=float, default=PYTORCH_DEFAULTS.top_p)


BACKENDS: dict[str, BackendDescriptor] = {
    "anthropic": BackendDescriptor("src.custom_ai.anthropic.anthropic", "Anthropic", _anthropic),
    "gemini": BackendDescriptor("src.custom_ai.gemini.gemini", "Gemini", _gemini),
    "huggingface": BackendDescriptor(
        "src.custom_ai.hugging_face.hugging_face", "HuggingFace", _huggingface
    ),
    "lmstudio": BackendDescriptor("src.custom_ai.lm_studio.lm_studio", "LmStudio", _lmstudio),
    "noai": BackendDescriptor("src.custom_ai.no_ai.no_ai", "NoAI", _no_arguments),
    "ollama": BackendDescriptor("src.custom_ai.ollama.ollama", "Ollama", _ollama),
    "pytorchllm": BackendDescriptor(
        "src.custom_ai.pytorch_llm.pytorch_llm", "PyTorchLLM", _pytorchllm
    ),
}


def load_backend_class(name: str) -> type:
    """Load the selected backend implementation and no other provider SDK."""
    try:
        descriptor = BACKENDS[name]
    except KeyError as error:
        raise ValueError(f"Unknown AI backend: {name}") from error
    return descriptor.load_class()
