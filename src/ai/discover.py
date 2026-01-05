#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import Dict

from src.genai.genai import SbkGenAI


def discover_custom_ai_classes(package_name: str = "src.custom_ai") -> Dict[str, type]:
    """
    Discover and import all concrete subclasses of SbkGenAI defined in the
    `package_name` package (by default `src.custom_ai`).

    Behavior and notes:
    - Attempts to import the package and each submodule found in the package
      directory using pkgutil.iter_modules.
    - If the `src` package is not on sys.path, the function will add the
      repository root (one level above the `src` directory) to sys.path so the
      import can succeed when running from the repository root.
    - Only returns classes that are subclasses of `SbkGenAI` (excluding the
      base class itself) and are not abstract (inspect.isabstract == False).

    Returns:
        Dict[str, type]: mapping from class name to the class object for each
        discovered concrete SbkGenAI implementation.
    """

    discovered: Dict[str, type] = {}

    # Ensure the repo root (parent of `src`) is on sys.path so imports like
    # `src.custom_ai` succeed when running tools from the project root.
    try:
        pkg = importlib.import_module(package_name)
    except Exception:
        # Try to add the parent of the `src` directory to sys.path
        this_file = Path(__file__).resolve()
        repo_root = this_file.parents[1]  # parent of `src`
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        pkg = importlib.import_module(package_name)

    # Iterate modules in the package and import them
    if not hasattr(pkg, "__path__"):
        return discovered

    for finder, name, ispkg in pkgutil.iter_modules(pkg.__path__):
        # import the module
        full_name = f"{pkg.__name__}.{name}"
        try:
            module = importlib.import_module(full_name)
        except Exception:
            # Silently skip modules that fail to import; callers can decide how
            # to handle missing dependencies.
            continue

        # inspect classes in the module
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            # Ensure class is defined in the module we just imported
            if obj.__module__ != module.__name__:
                continue

            try:
                is_subclass = issubclass(obj, SbkGenAI)
            except Exception:
                is_subclass = False

            if is_subclass and obj is not SbkGenAI and not inspect.isabstract(obj):
                discovered[obj.__name__] = obj

    return discovered
