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

class SbkParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='sbk charts',
                                         epilog='Please report issues at https://github.com/kmgowda/sbk-charts')
        self.parser.add_argument('-i', '--ifiles', help="Input CSV files, separated by ','", required=True)
        self.parser.add_argument('-o', '--ofile', help='Output xlsx file', default="out.xlsx")

    def add_argument(self, short_name, name, help_msg, required = False, default = None):
        self.parser.add_argument(short_name, name, help=help_msg, required=required, default=default)

    def parse_args(self):
        return self.parser.parse_args()