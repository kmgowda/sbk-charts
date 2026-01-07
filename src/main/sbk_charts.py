#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""sbk_charts.main.sbk_charts

Command-line entry point glue for the sbk-charts tool.

This module exposes a single function, `sbk_charts()`, which is responsible
for parsing command-line arguments, creating the Excel workbook from one or
more CSV input files, and invoking the chart-generation AI pipeline to add
charts to the workbook. The function performs I/O and prints progress to
stdout.

"""

import os
import sys
# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ai.sbk_ai import SbkAI
from src.sheets.sheets import SbkMultiSheets
from src.version.sbk_version import __sbk_version__
from src.parser.sbk_parser import get_sbk_parser

SBK_BANNER_FILE = os.path.join(os.path.curdir, 'src/main', 'banner.txt')

def sbk_charts():
    """Top-level orchestration for sbk-charts CLI.

    Behavior
    - Parse CLI arguments using `SbkParser` (expects `-i/--ifiles` and `-o/--ofile`).
    - Print banner and version information.
    - Create R/T worksheets for each input CSV file using `SbkMultiSheets`.
    - Instantiate the AI/charting pipeline and generate charts into the
      previously-created workbook.

    Side effects
    - Writes an .xlsx output file (as specified by the `-o/--ofile` argument).
    - Prints informational messages to stdout.

    Returns
    - None
    """
    parser = get_sbk_parser()
    ch = SbkAI(__sbk_version__)
    ch.add_args(parser)
    args = parser.parse_args()
    print(open(SBK_BANNER_FILE, 'r').read())
    print("Sbk Charts Version : " + __sbk_version__)
    print('Input Files : ', args.ifiles)
    print('Output File : ', args.ofile)
    sh = SbkMultiSheets(args.ifiles.split(","), args.ofile)
    sh.create_sheets()
    ch.parse_args(args)
    ch.create_graphs()
