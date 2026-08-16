"""Static AI backend registry with lazy implementation imports."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable


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
    parser.add_argument("--anthropic-model", default="anthropic-sonnet-4-20250514")
    parser.add_argument("--anthropic-max-tokens", type=int, default=2048)
    parser.add_argument("--anthropic-temperature", type=float, default=0.4)


def _gemini(parser: object) -> None:
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--gemini-max-tokens", type=int, default=2048)
    parser.add_argument("--gemini-temperature", type=float, default=0.4)


def _huggingface(parser: object) -> None:
    parser.add_argument(
        "-id",
        "--model_id",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model ID",
    )


def _lmstudio(parser: object) -> None:
    parser.add_argument("--url", default="http://localhost:1234/api/v0")
    parser.add_argument("--lm-model", default="")
    parser.add_argument("--lm-temperature", type=float, default=0.4)
    parser.add_argument("--lm-max-tokens", type=int, default=1800)


def _no_arguments(_parser: object) -> None:
    return


def _ollama(parser: object) -> None:
    parser.add_argument("-url", "--ollama-url", default="http://localhost:11434")
    parser.add_argument("-model", "--ollama-model", default="llama3.1")
    parser.add_argument("-tmp", "--ollama-temperature", type=float, default=0.4)
    parser.add_argument("-timeout", "--ollama-timeout", type=int, default=120)


def _pytorchllm(parser: object) -> None:
    parser.add_argument("--pt-model", default="openai/gpt-oss-20b")
    parser.add_argument("--pt-train", action="store_true", default=False)
    parser.add_argument("--pt-device", default="auto")
    parser.add_argument("--pt-max-length", type=int, default=2048)
    parser.add_argument("--pt-temperature", type=float, default=0.4)
    parser.add_argument("--pt-top-p", type=float, default=0.9)


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
