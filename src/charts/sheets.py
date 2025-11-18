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

# SBK-sheets :  Storage Benchmark Kit - Sheets

from pandas import read_csv
from xlsxwriter import Workbook

from . import constants


def wb_add_two_sheets(wb, r_name, t_name, df):
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

# The logo insertion fucntion works fine only if the package pillow is installed.
def add_sbk_logo(wb):
    ws = wb.add_worksheet("SBK")
    img_path = os.path.abspath("./images/sbk-logo.png")
    if os.path.exists(img_path):
        print(f"SBK logo image found: {img_path}")
        try:
            ws.insert_image('K7', img_path, {'x_scale': 0.5, 'y_scale': 0.5})
        except Exception as ex:
            print(f"Failed to insert image: {ex}")
    else:
        print(f"SBK logo Image not found: {img_path}")

class SbkSheets:
    def __init__(self, i_file, o_file):
        self.iFile = i_file
        self.oFile = o_file

    def create_sheets(self):
        wb = Workbook(self.oFile)
        add_sbk_logo(wb)
        df = read_csv(self.iFile)
        wb_add_two_sheets(wb, constants.R_PREFIX + "1", constants.T_PREFIX + "1", df)
        wb.close()
        print("xlsx file %s created" % self.oFile)


class SbkMultiSheets(SbkSheets):
    def __init__(self, i_files_list, o_file):
        super().__init__(i_files_list[0], o_file)
        self.iFilesList = i_files_list

    def create_sheets(self):
        wb = Workbook(self.oFile)
        add_sbk_logo(wb)
        for i, file in enumerate(self.iFilesList):
            wb_add_two_sheets(wb, constants.R_PREFIX + str(i + 1), constants.T_PREFIX + str(i + 1), read_csv(file))
        wb.close()
        print("xlsx file : %s created" % self.oFile)
