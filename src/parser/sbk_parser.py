#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""sbk_charts.parser.sbk_parser

Simple command-line argument parser wrapper used by the sbk-charts CLI.
This module provides the `SbkParser` class which encapsulates an
argparse.ArgumentParser instance and exposes a small convenience API to
register and parse arguments for the SBK charts command-line tool.

"""

import argparse


def get_sbk_parser():
    parser = argparse.ArgumentParser(description='sbk charts',
                                            epilog='Please report issues at https://github.com/kmgowda/sbk-charts')
    parser.add_argument('-i', '--ifiles', help="Input CSV files, separated by ','", required=True)
    parser.add_argument('-o', '--ofile', help='Output xlsx file', default="out.xlsx")
    return parser
