#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

import os

from src.ai.sbk_ai import SbkAI
from src.sheets.sheets import SbkMultiSheets
from src.charts.multicharts import SbkMultiCharts
from src.main.sbk_version import __sbk_version__
from src.parser.sbk_parser import SbkParser

SBK_BANNER_FILE = os.path.join(os.path.curdir, 'src/main', 'banner.txt')

def sbk_charts():
    parser = SbkParser()
    args = parser.parse_args()
    print(open(SBK_BANNER_FILE, 'r').read())
    print("Sbk Charts Version : " + __sbk_version__)
    print('Input Files : ', args.ifiles)
    print('Output File : ', args.ofile)
    sh = SbkMultiSheets(args.ifiles.split(","), args.ofile)
    sh.create_sheets()
    ch = SbkAI(__sbk_version__, args.ofile)
    ch.create_graphs()
