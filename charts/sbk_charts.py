#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

import argparse
import os
from charts.sheets import SbkMultiSheets
from charts.multicharts import SbkMultiCharts
from charts.version import __version__

SBK_BANNER_FILE = os.path.join(os.path.curdir, 'charts', 'banner.txt')

def sbk_charts():
    parser = argparse.ArgumentParser(description='sbk charts',
                                     epilog='Please report issues at https://github.com/kmgowda/SBK')
    parser.add_argument('-i', '--ifiles', help="Input CSV files, separated by ','", required=True)
    parser.add_argument('-o', '--ofile', help='Output xlsx file', default="out.xlsx")
    args = parser.parse_args()
    print(open(SBK_BANNER_FILE, 'r').read())
    print("Sbk Charts Version : " + __version__)
    print('Input Files : ', args.ifiles)
    print('Output File : ', args.ofile)
    sh = SbkMultiSheets(args.ifiles.split(","), args.ofile)
    sh.create_sheets()
    ch = SbkMultiCharts(__version__, args.ofile)
    ch.create_graphs()
