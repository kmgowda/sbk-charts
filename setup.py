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
from src.main.sbk_version import __sbk_version__


# Get the absolute path to the package directory
package_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(package_dir, 'src')

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
    package_dir={'': 'src'},  # Tell setuptools to look for packages in src/
    packages=find_packages(where='src'),  # Find all packages under src/
    package_data={
        'charts': ['banner.txt'],  # Include banner.txt in the package
    },
    entry_points={
        'console_scripts': [
            'sbk-charts=charts.sbk_charts:sbk_charts',
        ],
    },
    url='https://github.com/kmgowda/sbk-charts',
    license='Apache License Version 2.0',
    author='KMG',
    author_email='keshava.gowda@gmail.com',
    description='SBK Charts',
    install_requires=required,
)
