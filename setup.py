#!/usr/bin/env python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

import importlib.util
import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

project_root = os.path.abspath(os.path.dirname(__file__))
policy_reader_path = os.path.join(project_root, 'scripts', 'project_policy.py')
policy_spec = importlib.util.spec_from_file_location('sbk_charts_project_policy', policy_reader_path)
if policy_spec is None or policy_spec.loader is None:
    raise RuntimeError(f"Could not load project policy reader: {policy_reader_path}")
policy_reader = importlib.util.module_from_spec(policy_spec)
sys.modules[policy_spec.name] = policy_reader
policy_spec.loader.exec_module(policy_reader)
load_policy = policy_reader.load_policy

# Read the configured version module without importing application dependencies.
def get_version() -> str:
    return policy_reader.application_version(project_policy, Path(project_root))

project_policy = load_policy()
__sbk_version__ = get_version()


# Get the absolute path to the package directory
package_dir = project_root

# Read the canonical runtime requirements; a missing file is a packaging error.
req_file = os.path.join(package_dir, *project_policy.application.runtime_requirements.split('/'))
with open(req_file, encoding='utf-8') as requirements_file:
    required = [
        line.strip()
        for line in requirements_file
        if line.strip() and not line.lstrip().startswith('#')
    ]

setup(
    name=project_policy.application.distribution_name,
    version=__sbk_version__,
    python_requires=f">={project_policy.runtime.minimum_python}",
    # Install the 'src' package as a top-level package so imports like
    # `from src.ai.sbk_ai import SbkAI` work at runtime.
    package_dir={'': '.'},  # Look for packages in the project root (will include 'src' package)
    packages=find_packages(where='.'),
    include_package_data=True,
    package_data={package: list(paths) for package, paths in project_policy.package_data.items()},
    entry_points={
        'console_scripts': [
            # Point the console script at the module inside the 'src' package
            f'{project_policy.application.name}={project_policy.application.entry_point}',
        ],
    },
    url=project_policy.application.url,
    license=project_policy.application.license,
    author=project_policy.application.author,
    author_email=project_policy.application.author_email,
    description=project_policy.application.description,
    install_requires=required,
)
