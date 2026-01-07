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
    `package_name` package and its subpackages.

    Behavior and notes:
    - Recursively searches through all subpackages in the specified package
    - Attempts to import each module found in the package and subpackages
    - Only returns classes that are concrete subclasses of `SbkGenAI`
    - Handles both Python files and package directories

    Args:
        package_name: The root package name to search in (default: "src.custom_ai")

    Returns:
        Dict[str, type]: mapping from class name to the class object for each
        discovered concrete SbkGenAI implementation.
    """
    discovered: Dict[str, type] = {}
    processed_modules = set()

    # Ensure the repo root (parent of `src`) is on sys.path so imports like
    # `src.custom_ai` succeed when running tools from the project root.
    try:
        pkg = importlib.import_module(package_name)
    except ImportError:
        # Try to add the parent of the `src` directory to sys.path
        this_file = Path(__file__).resolve()
        repo_root = this_file.parents[1]  # parent of `src`
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        pkg = importlib.import_module(package_name)

    if not hasattr(pkg, "__path__"):
        return discovered

    def process_module(module_name: str):
        """Helper function to process a single module and its submodules"""
        if module_name in processed_modules:
            return
        processed_modules.add(module_name)

        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            print(f"Importing module {module_name} failed with error : {str(e)}")
            return

        # Inspect classes in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Ensure class is defined in the module we just imported
            if obj.__module__ != module.__name__:
                continue

            try:
                is_subclass = issubclass(obj, SbkGenAI)
            except Exception:
                is_subclass = False

            if is_subclass and obj is not SbkGenAI and not inspect.isabstract(obj):
                discovered[obj.__name__.lower()] = obj

    # Process the root package
    process_module(package_name)

    # Process all submodules and subpackages
    for finder, name, ispkg in pkgutil.walk_packages(pkg.__path__, package_name + '.'):
        if ispkg:
            # If it's a package, process it
            process_module(name)
        else:
            # If it's a module, process it
            process_module(name)

    return discovered