#!/usr/bin/env python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

import os
from setuptools import setup, find_packages
from src.version.sbk_version import __sbk_version__


# Get the absolute path to the package directory
package_dir = os.path.abspath(os.path.dirname(__file__))

# Read requirements.txt if it exists
required = []
req_file = os.path.join(package_dir, 'requirements.txt')
if os.path.exists(req_file):
    with open(req_file) as f:
        required = f.read().splitlines()
else:
    # Fallback to hardcoded requirements if file not found
    required = [
        'openpyxl~=3.1.5',
        'pandas~=2.2.3',
        'XlsxWriter~=3.2.3',
        'ordered-set~=4.1.0',
        'jproperties~=2.1.1',
        'pillow~=12.0.0',
        'openpyxl-image-loader~=1.0.5'
    ]

setup(
    name='sbk-charts',
    version=__sbk_version__,
    # Install the 'src' package as a top-level package so imports like
    # `from src.ai.sbk_ai import SbkAI` work at runtime.
    package_dir={'': '.'},  # Look for packages in the project root (will include 'src' package)
    packages=find_packages(where='.'),
    package_data={
        # Include banner.txt in the 'main' package (located at src/main/banner.txt)
        'main': ['banner.txt'],
    },
    entry_points={
        'console_scripts': [
            # Point the console script at the module inside the 'src' package
            'sbk-charts=src.main.sbk_charts:sbk_charts',
        ],
    },
    url='https://github.com/kmgowda/sbk-charts',
    license='Apache License Version 2.0',
    author='KMG',
    author_email='keshava.gowda@gmail.com',
    description='SBK Charts',
    install_requires=required,
)
