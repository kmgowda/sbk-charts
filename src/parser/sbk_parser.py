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

class SbkParser:
    """Wrapper around argparse.ArgumentParser for the SBK Charts CLI.

    Responsibilities
    - Provide a pre-configured ArgumentParser with the common `-i/--ifiles`
      and `-o/--ofile` options used by the tool.
    - Offer a small convenience method `add_argument` to register additional
      options in a consistent way.

    Example usage
        p = SbkParser()
        p.add_argument('-v', '--verbose', help_msg='Enable verbose output')
        args = p.parse_args()

    The class intentionally keeps behavior minimal and delegates to
    argparse for the heavy lifting.
    """

    def __init__(self):
        """Create and configure the underlying argparse.ArgumentParser.

        The default parser includes:
        - `-i/--ifiles`: required, comma-separated input CSV files
        - `-o/--ofile`: optional, output xlsx file path (defaults to 'out.xlsx')
        """
        self.__parser = argparse.ArgumentParser(description='sbk charts',
                                                epilog='Please report issues at https://github.com/kmgowda/sbk-charts')
        self.__parser.add_argument('-i', '--ifiles', help="Input CSV files, separated by ','", required=True)
        self.__parser.add_argument('-o', '--ofile', help='Output xlsx file', default="out.xlsx")

    def add_subparsers(self, dest, help, required = False):
        return self.__parser.add_subparsers(dest=dest, help=help, required=required)

    def add_argument(self, short_name, name, help_msg, required = False, default = None):
        """Add an argument to the underlying ArgumentParser.

        Parameters
        - short_name (str): short flag (e.g. '-v')
        - name (str): long flag (e.g. '--verbose')
        - help_msg (str): help text used in the CLI usage message
        - required (bool): whether the argument is required (default False)
        - default: default value for the argument if not supplied
        """
        self.__parser.add_argument(short_name, name, help=help_msg, required=required, default=default)

    def parse_args(self):
        """Parse command-line arguments and return the populated namespace.

        Returns
        - argparse.Namespace: parsed arguments as attributes (e.g. args.ifiles)
        """
        return self.__parser.parse_args()

    def set_defaults(self, **kwargs):
        self.__parser.set_defaults(**kwargs)