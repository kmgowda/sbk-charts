#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""sbk_charts.sheets

Utilities to convert SBK CSV output into an Excel workbook with separate
R (raw/interval) and T (total/summary) worksheets. This module is a small
helper used by the SBK CLI to create .xlsx files that are later consumed by
chart generation code.

Public functions and classes
- wb_add_two_sheets(wb, r_name, t_name, df): write two worksheets (R and T)
  from a pandas DataFrame.
- SbkSheets: create a single-workbook from one CSV file.
- SbkMultiSheets: final class to create a single workbook from multiple CSV files.

Note: This file performs I/O (reads CSV via pandas and writes .xlsx via
xlsxwriter). Docstrings were added/expanded only — no executable code was
modified.
"""

# SBK-sheets :  Storage Benchmark Kit - Sheets

from pandas import read_csv
from xlsxwriter import Workbook
from typing import final

from src.sheets.logo import add_sbk_logo
from src.sheets import constants


def wb_add_two_sheets(wb, r_name, t_name, df):
    """Add two worksheets to an xlsxwriter Workbook using data from a DataFrame.

    This helper writes header row(s) and then splits the DataFrame rows into
    two worksheets: one for interval-level rows (R sheet) and one for total/summary
    rows (T sheet). The function preserves column widths based on header and
    cell content lengths.

    Parameters
    - wb: xlsxwriter.Workbook instance to which sheets will be added
    - r_name (str): name for the R sheet (e.g. 'R1')
    - t_name (str): name for the T sheet (e.g. 'T1')
    - df (pandas.DataFrame): the data to write; expects a 'Type' column used to
      separate interval vs total rows (see src.sheets.constants.TYPE)

    Returns
    - None (works by side effect on the provided Workbook)
    """
    header = df.columns.values
    r_ws = wb.add_worksheet(r_name)
    t_ws = wb.add_worksheet(t_name)
    for c, h in enumerate(header):
        r_ws.set_column(c, c, len(h))
        t_ws.set_column(c, c, len(h))
        r_ws.write(0, c, h)
        t_ws.write(0, c, h)
    r_row = 1
    t_row = 1
    for row in df.iterrows():
        if row[1][constants.TYPE] == constants.TYPE_TOTAL:
            ws, row_num = t_ws, t_row
            t_row += 1
        else:
            ws, row_num = r_ws, r_row
            r_row += 1
        for c, h in enumerate(header):
            col_size = len(str(row[1][h])) + 1
            if col_size > len(h):
                ws.set_column(c, c, col_size)
            try:
                ws.write(row_num, c, row[1][h])
            except Exception as ex:
                pass


class SbkSheets:
    """Create an Excel workbook with SBK-formatted R/T sheets from a CSV.

    This class reads a single input CSV file (via pandas.read_csv) and writes
    an .xlsx workbook with two worksheets named according to constants.R_PREFIX
    and constants.T_PREFIX. It also inserts the SBK logo into the workbook.

    Usage
    - Instantiate with input CSV file path and desired output .xlsx path.
    - Call `create_sheets()` to produce the workbook on disk.
    """
    def __init__(self, i_file, o_file):
        self.iFile = i_file
        self.oFile = o_file

    def create_sheets(self):
        """Produce an .xlsx workbook from the configured input CSV file.

        The method will:
        - open an xlsxwriter.Workbook for the configured output path
        - insert the SBK logo
        - read the CSV input into a pandas DataFrame
        - use wb_add_two_sheets() to populate R and T worksheets
        - close the workbook to flush it to disk

        Returns
        - None (side effect: file written at self.oFile)
        """
        wb = Workbook(self.oFile)
        add_sbk_logo(wb)
        df = read_csv(self.iFile)
        wb_add_two_sheets(wb, constants.R_PREFIX + "1", constants.T_PREFIX + "1", df)
        wb.close()
        print("xlsx file %s created" % self.oFile)


@final
class SbkMultiSheets(SbkSheets):

    """Final class to create a single workbook from multiple CSV inputs.

    This class is declared final to prevent further subclassing. It collects
    multiple CSV files and writes each as a pair of R/T worksheets into the
    same output workbook. The resulting workbook will contain R1/T1, R2/T2, ...

    Usage
    - Instantiate with a list of input CSV file paths and an output .xlsx path.
    - Call `create_sheets()` to write the combined workbook.
    """

    # This should be final class ; just create sheets
    def __init_subclass__(cls, **kwargs):
        raise TypeError("Cannot create subclass for SbkMultiSheets")

    def __init__(self, i_files_list, o_file):
        super().__init__(i_files_list[0], o_file)
        self.iFilesList = i_files_list

    def create_sheets(self):
        """Create a single .xlsx workbook containing sheets for all input files.

        For each CSV in the provided list, this method adds a pair of worksheets
        with names R<n> and T<n> where <n> is the 1-based index of the file in
        the list. The SBK logo is added once at workbook creation.

        Returns
        - None (writes file to self.oFile)
        """
        wb = Workbook(self.oFile)
        add_sbk_logo(wb)
        for i, file in enumerate(self.iFilesList):
            wb_add_two_sheets(wb, constants.R_PREFIX + str(i + 1), constants.T_PREFIX + str(i + 1), read_csv(file))
        wb.close()
        print("xlsx file : %s created" % self.oFile)
