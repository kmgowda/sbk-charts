#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""SBK Command Line Parser Module

This module provides command-line argument parsing for the SBK Charts tool.
It defines the argument parser configuration and provides a clean interface
for accessing command-line arguments.

Features:
- Application version reporting without input files
- Input file specification (CSV format)
- Output file configuration (XLSX format)
- AI backend selection and configuration
- Threading and performance options

Example Usage:
    parser = get_sbk_parser()
    args = parser.parse_args()
    print(f"Input files: {args.ifiles}")
    print(f"Output file: {args.ofile}")
"""

import argparse

from src.version.sbk_version import __sbk_version__


def get_sbk_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for SBK Charts.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser with SBK-specific arguments.
        
    The parser includes the following arguments:
        -version, --version: Print the application version and exit
        -i, --ifiles: Comma-separated list of input CSV files (required)
        -o, --ofile: Output XLSX file path (default: 'out.xlsx')
    """
    parser = argparse.ArgumentParser(
        description='SBK Charts - Storage Benchmark Visualization and AI analytics Tool',
        epilog='Please report issues at https://github.com/kmgowda/sbk-charts'
    )
    parser.add_argument(
        '-version', '--version',
        action='version',
        version=f'Sbk Charts Version : {__sbk_version__}',
        help='Show the sbk-charts version and exit'
    )
    parser.add_argument(
        '-i', '--ifiles',
        help="Comma-separated list of input CSV files containing benchmark results",
        required=True
    )
    parser.add_argument(
        '-o', '--ofile',
        help='Output XLSX file path (default: %(default)s)',
        default="out.xlsx"
    )
    return parser
