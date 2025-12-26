#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

# sbk_charts :  Storage Benchmark Kit - Charts
import re
import os
from collections import OrderedDict
from typing import final

from openpyxl import load_workbook
from openpyxl.chart import LineChart, BarChart, Reference, Series
from openpyxl.drawing.image import Image
from openpyxl.drawing.text import CharacterProperties, ParagraphProperties
from openpyxl_image_loader import SheetImageLoader

import src.sheets.constants as sheets_constants
from src.charts import constants

class SbkCharts:
    def __init__(self, version, file):
        self.version = version
        self.file = file
        self.wb = load_workbook(self.file)
        self.time_unit = self.get_time_unit(self.wb[sheets_constants.R_PREFIX + "1"])
        self.n_latency_charts = 5
        self.latency_groups = [
            [constants.MIN_LATENCY, constants.PERCENTILE_5],
            [constants.PERCENTILE_5, constants.PERCENTILE_10, constants.PERCENTILE_15, constants.PERCENTILE_20, 
             constants.PERCENTILE_25, constants.PERCENTILE_30, constants.PERCENTILE_35, constants.PERCENTILE_40,
             constants.PERCENTILE_45, constants.PERCENTILE_50],
            [constants.PERCENTILE_50, constants.AVG_LATENCY],
            [constants.PERCENTILE_50, constants.PERCENTILE_55, constants.PERCENTILE_60, constants.PERCENTILE_65,
             constants.PERCENTILE_70, constants.PERCENTILE_75, constants.PERCENTILE_80, constants.PERCENTILE_85,
             constants.PERCENTILE_90],
            [constants.PERCENTILE_92_5, constants.PERCENTILE_95, constants.PERCENTILE_97_5, constants.PERCENTILE_99,
             constants.PERCENTILE_99_25, constants.PERCENTILE_99_5, constants.PERCENTILE_99_75, constants.PERCENTILE_99_9,
             constants.PERCENTILE_99_95, constants.PERCENTILE_99_99]]
        self.slc_percentile_names = [[constants.PERCENTILE_5, constants.PERCENTILE_10, constants.PERCENTILE_15, 
                                      constants.PERCENTILE_20, constants.PERCENTILE_25, constants.PERCENTILE_30, 
                                      constants.PERCENTILE_35, constants.PERCENTILE_40, constants.PERCENTILE_50],
                                     [constants.PERCENTILE_50, constants.PERCENTILE_55, constants.PERCENTILE_60,
                                      constants.PERCENTILE_65, constants.PERCENTILE_70, constants.PERCENTILE_75,
                                      constants.PERCENTILE_80, constants.PERCENTILE_85, constants.PERCENTILE_90,
                                      constants.PERCENTILE_92_5, constants.PERCENTILE_95, constants.PERCENTILE_97_5,
                                      constants.PERCENTILE_99, constants.PERCENTILE_99_25, constants.PERCENTILE_99_5,
                                      constants.PERCENTILE_99_75, constants.PERCENTILE_99_9, constants.PERCENTILE_99_95,
                                      constants.PERCENTILE_99_99]]
        self.percentile_count_names = [constants.PERCENTILE_COUNT_5, constants.PERCENTILE_COUNT_10, 
                                      constants.PERCENTILE_COUNT_15, constants.PERCENTILE_COUNT_20,
                                      constants.PERCENTILE_COUNT_25, constants.PERCENTILE_COUNT_30, 
                                      constants.PERCENTILE_COUNT_35, constants.PERCENTILE_COUNT_40,
                                      constants.PERCENTILE_COUNT_50, constants.PERCENTILE_COUNT_55, 
                                      constants.PERCENTILE_COUNT_60, constants.PERCENTILE_COUNT_65,
                                      constants.PERCENTILE_COUNT_70, constants.PERCENTILE_COUNT_75, 
                                      constants.PERCENTILE_COUNT_80, constants.PERCENTILE_COUNT_85,
                                      constants.PERCENTILE_COUNT_90, constants.PERCENTILE_COUNT_92_5, 
                                      constants.PERCENTILE_COUNT_95, constants.PERCENTILE_COUNT_97_5,
                                      constants.PERCENTILE_COUNT_99, constants.PERCENTILE_COUNT_99_25, 
                                      constants.PERCENTILE_COUNT_99_5, constants.PERCENTILE_COUNT_99_75,
                                      constants.PERCENTILE_COUNT_99_9, constants.PERCENTILE_COUNT_99_95, 
                                      constants.PERCENTILE_COUNT_99_99]

    @final
    def is_r_num_sheet(self, name):
        return re.match(r'^' + sheets_constants.R_PREFIX + r'\d+$', name)

    @final
    def is_t_num_sheet(self, name):
        return re.match(r'^' + sheets_constants.T_PREFIX + r'\d+$', name)

    @final
    def get_t_num_sheet_name(self, r_num_name):
        return sheets_constants.T_PREFIX + r_num_name[1:]

    @final
    def get_columns_from_worksheet(self, ws):
        ret = OrderedDict()
        for cell in ws[1]:
            if cell.value:
                ret[cell.value] = cell.column
        return ret

    def get_latency_percentile_columns(self, ws):
        columns = self.get_columns_from_worksheet(ws)
        ret = OrderedDict()
        for key in columns.keys():
            if key.startswith("Percentile_") and not key.startswith("Percentile_Count"):
                ret[key] = columns[key]
        return ret

    def get_latency_percentile_count_columns(self, ws):
        columns = self.get_columns_from_worksheet(ws)
        ret = OrderedDict()
        for key in columns.keys():
            if key.startswith("Percentile_Count"):
                ret[key] = columns[key]
        return ret

    def get_latency_columns(self, ws):
        columns = self.get_columns_from_worksheet(ws)
        ret = OrderedDict()
        ret[constants.AVG_LATENCY] = columns[constants.AVG_LATENCY]
        ret[constants.MIN_LATENCY] = columns[constants.MIN_LATENCY]
        ret[constants.MAX_LATENCY] = columns[constants.MAX_LATENCY]
        ret.update(self.get_latency_percentile_columns(ws))
        return ret

    @final
    def get_time_unit(self, ws):
        names = self.get_columns_from_worksheet(ws)
        return str(ws.cell(row=2, column=names[constants.LATENCY_TIME_UNIT]).value).upper()

    @final
    def get_storage_name(self, ws):
        names = self.get_columns_from_worksheet(ws)
        return str(ws.cell(row=2, column=names[constants.STORAGE]).value).upper()

    @final
    def get_action_name(self, ws):
        names = self.get_columns_from_worksheet(ws)
        return str(ws.cell(row=2, column=names[constants.ACTION]).value)

    @final
    def __add_chart_attributes(self, chart, title, x_title, y_title, height, width):
        # Set the title of the chart with font size
        # Set chart title
        chart.title = title
        chart.title.tx.rich.p[0].pPr.defRPr = CharacterProperties(sz=3600, b=True)  # 36pt, bold
        
        # Set x-axis title
        chart.x_axis.title = x_title
        if not hasattr(chart.x_axis.title.tx.rich.p[0], 'pPr'):
            chart.x_axis.title.tx.rich.p[0].pPr = ParagraphProperties()
        chart.x_axis.title.tx.rich.p[0].pPr.defRPr = CharacterProperties(sz=1800)  # 18pt
        
        # Set y-axis title
        chart.y_axis.title = y_title
        if not hasattr(chart.y_axis.title.tx.rich.p[0], 'pPr'):
            chart.y_axis.title.tx.rich.p[0].pPr = ParagraphProperties()
        chart.y_axis.title.tx.rich.p[0].pPr.defRPr = CharacterProperties(sz=1800)  # 18pt

        chart.height = height
        chart.width = width
        chart.x_axis.delete = False
        chart.y_axis.delete = False

    @final
    def create_line_chart(self, title, x_title, y_title, height, width):
        chart = LineChart()
        self.__add_chart_attributes(chart, title, x_title, y_title, height, width)
        return chart

    @final
    def create_bar_chart(self, title, x_title, y_title, height, width):
        chart = BarChart()
        self.__add_chart_attributes(chart, title, x_title, y_title, height, width)
        return chart

    def create_latency_line_graph(self, title):
        return self.create_line_chart(title, "Intervals", "Latency time in " + self.time_unit, 25, 50)

    def get_latency_series(self, ws, ws_name):
        latencies = self.get_latency_columns(ws)
        data_series = OrderedDict()
        for x in latencies:
            data_series[x] = Series(Reference(ws, min_col=latencies[x], min_row=2,
                                              max_col=latencies[x], max_row=ws.max_row),
                                    title=ws_name + "-" + x)
        return data_series

    def get_latency_percentile_series(self, ws, ws_name, names_list):
        latencies = self.get_latency_percentile_columns(ws)
        data_series = OrderedDict()
        min_col = latencies[names_list[0]]
        max_col = latencies[names_list[-1]]
        for r in range(2, ws.max_row + 1):
            data_series[r] = Series(Reference(ws, min_col=min_col, min_row=r, max_col=max_col, max_row=r),
                                    title=ws_name + "_" + str(r-1))
        return data_series

    def get_latency_percentile_count_series(self, ws, ws_name, names_list):
        latencies = self.get_latency_percentile_count_columns(ws)
        data_series = OrderedDict()
        min_col = latencies[names_list[0]]
        max_col = latencies[names_list[-1]]
        for r in range(2, ws.max_row + 1):
            data_series[r] = Series(Reference(ws, min_col=min_col, min_row=r, max_col=max_col, max_row=r),
                                    title=ws_name + "_" + str(r-1))
        return data_series

    @final
    def __get_column_values(self, ws, column_name):
        cols = self.get_columns_from_worksheet(ws)
        values = []
        for row in range(2, ws.max_row + 1):
            cell_value = ws.cell(row=row, column=cols[column_name]).value
            values.append(cell_value)
        return values

    @final
    def get_throughput_mb_values(self, ws):
        return self.__get_column_values(ws, constants.MB_PER_SEC)

    @final
    def get_throughput_write_request_mb_values(self, ws):
        return self.__get_column_values(ws, constants.WRITE_REQUEST_MB_PER_SEC)

    @final
    def get_throughput_read_request_mb_values(self, ws):
        return self.__get_column_values(ws, constants.READ_REQUEST_MB_PER_SEC)

    @final
    def get_throughput_records_values(self, ws):
        return self.__get_column_values(ws, constants.RECORDS_PER_SEC)

    @final
    def get_throughput_write_request_records_values(self, ws):
        return self.__get_column_values(ws, constants.WRITE_REQUEST_RECORDS_PER_SEC)

    @final
    def get_throughput_read_request_records_values(self, ws):
        return self.__get_column_values(ws, constants.READ_REQUEST_RECORDS_PER_SEC)

    @final
    def get_records_values(self, ws):
        return self.__get_column_values(ws, constants.RECORDS)

    @final
    def get_mb_values(self, ws):
        return self.__get_column_values(ws, constants.MB)

    @final
    def get_write_request_mb_values(self, ws):
        return self.__get_column_values(ws, constants.WRITE_REQUEST_MB)

    @final
    def get_write_request_records_values(self, ws):
        return self.__get_column_values(ws, constants.WRITE_REQUEST_RECORDS)

    @final
    def get_write_response_pending_mb_values(self, ws):
        return self.__get_column_values(ws, constants.WRITE_RESPONSE_PENDING_MB)

    @final
    def get_write_response_pending_records_values(self, ws):
        return self.__get_column_values(ws, constants.WRITE_RESPONSE_PENDING_RECORDS)

    @final
    def get_read_request_records_values(self, ws):
        return self.__get_column_values(ws, constants.READ_REQUEST_RECORDS)

    @final
    def get_read_request_mb_values(self, ws):
        return self.__get_column_values(ws, constants.READ_REQUEST_MB)

    @final
    def get_read_response_pending_mb_values(self, ws):
        return self.__get_column_values(ws, constants.READ_RESPONSE_PENDING_MB)

    @final
    def get_read_response_pending_records_values(self, ws):
        return self.__get_column_values(ws, constants.READ_RESPONSE_PENDING_RECORDS)

    @final
    def get_write_read_request_pending_mb_values(self, ws):
        return self.__get_column_values(ws, constants.WRITE_READ_REQUEST_PENDING_MB)

    @final
    def get_write_read_request_pending_records_values(self, ws):
        return self.__get_column_values(ws, constants.WRITE_READ_REQUEST_PENDING_RECORDS)

    @final
    def get_avg_latency_values(self, ws, ws_name):
        return self.__get_column_values(ws, constants.AVG_LATENCY)

    @final
    def get_min_latency_values(self, ws, ws_name):
        return self.__get_column_values(ws, constants.MIN_LATENCY)

    @final
    def get_max_latency_values(self, ws, ws_name):
        return self.__get_column_values(ws, constants.MAX_LATENCY)

    @final
    def get_write_timeout_events_values(self, ws):
        return self.__get_column_values(ws, constants.WRITE_TIMEOUT_EVENTS)

    @final
    def get_write_timeout_events_per_sec_values(self, ws):
        return self.__get_column_values(ws, constants.WRITE_TIMEOUT_EVENTS_PER_SEC)

    @final
    def get_read_timeout_events_values(self, ws):
        return self.__get_column_values(ws, constants.READ_TIMEOUT_EVENTS)

    @final
    def get_read_timeout_events_per_sec_values(self, ws):
        return self.__get_column_values(ws, constants.READ_TIMEOUT_EVENTS_PER_SEC)

    @final
    def __get_column_series(self, ws, ws_name, column_name):
        cols = self.get_columns_from_worksheet(ws)
        return Series(Reference(ws, min_col=cols[column_name], min_row=2,
                                max_col=cols[column_name], max_row=ws.max_row),
                      title=ws_name + "-" + column_name)

    @final
    def get_throughput_mb_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.MB_PER_SEC)

    @final
    def get_throughput_write_request_mb_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.WRITE_REQUEST_MB_PER_SEC)

    @final
    def get_throughput_read_request_mb_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.READ_REQUEST_MB_PER_SEC)

    @final
    def get_throughput_records_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.RECORDS_PER_SEC)

    @final
    def get_throughput_write_request_records_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.WRITE_REQUEST_RECORDS_PER_SEC)

    @final
    def get_throughput_read_request_records_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.READ_REQUEST_RECORDS_PER_SEC)

    @final
    def get_records_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.RECORDS)

    @final
    def get_mb_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.MB)

    @final
    def get_write_request_mb_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.WRITE_REQUEST_MB)

    @final
    def get_write_request_records_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.WRITE_REQUEST_RECORDS)

    @final
    def get_write_response_pending_mb_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.WRITE_RESPONSE_PENDING_MB)

    @final
    def get_write_response_pending_records_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.WRITE_RESPONSE_PENDING_RECORDS)

    @final
    def get_read_request_records_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.READ_REQUEST_RECORDS)

    @final
    def get_read_request_mb_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.READ_REQUEST_MB)

    @final
    def get_read_response_pending_mb_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.READ_RESPONSE_PENDING_MB)

    @final
    def get_read_response_pending_records_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.READ_RESPONSE_PENDING_RECORDS)

    @final
    def get_write_read_request_pending_mb_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.WRITE_READ_REQUEST_PENDING_MB)

    @final
    def get_write_read_request_pending_records_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.WRITE_READ_REQUEST_PENDING_RECORDS)

    @final
    def get_avg_latency_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.AVG_LATENCY)

    @final
    def get_min_latency_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.MIN_LATENCY)

    @final
    def get_max_latency_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.MAX_LATENCY)

    @final
    def get_write_timeout_events_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.WRITE_TIMEOUT_EVENTS)

    @final
    def get_write_timeout_events_per_sec_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.WRITE_TIMEOUT_EVENTS_PER_SEC)

    @final
    def get_read_timeout_events_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.READ_TIMEOUT_EVENTS)

    @final
    def get_read_timeout_events_per_sec_series(self, ws, ws_name):
        return self.__get_column_series(ws, ws_name, constants.READ_TIMEOUT_EVENTS_PER_SEC)

    def create_latency_compare_graphs(self, ws, prefix):
        charts, sheets = [], []
        for i in range(self.n_latency_charts):
            charts.append(self.create_latency_line_graph("Latency Variations"))
            sheets.append(self.wb.create_sheet("Latencies-" + prefix + "-" + str(i + 1)))
        latency_series = self.get_latency_series(ws, prefix)
        for x in latency_series:
            for i, g in enumerate(self.latency_groups):
                if x in g:
                    charts[i].append(latency_series[x])
        for i, ch in enumerate(charts):
            sheets[i].add_chart(ch)
        return sheets

    def create_latency_graphs(self, ws, prefix):
        sheets = []
        latency_series = self.get_latency_series(ws, prefix)
        for x in latency_series:
            chart = self.create_latency_line_graph(x + " Variations")
            # adding data
            chart.append(latency_series[x])
            # add chart to the sheet
            sheet = self.wb.create_sheet(x)
            sheet.add_chart(chart)
            sheets.append(sheet)
        return sheets

    def create_total_latency_percentile_graphs(self, ws, prefix):
        title = "Total Percentiles"
        latency_cols = self.get_latency_percentile_columns(ws)
        sheets = []
        for i, percentile_names in enumerate(self.slc_percentile_names):
            chart = self.create_line_chart(title, "Percentiles", "Latency time in " + self.time_unit, 25, 50)
            latency_series = self.get_latency_percentile_series(ws, prefix, percentile_names)
            for x in latency_series:
                chart.append(latency_series[x])
            # Add x-axis labels
            percentiles = Reference(ws, min_col=latency_cols[percentile_names[0]], min_row=1,
                                    max_col=latency_cols[percentile_names[-1]], max_row=1)
            chart.set_categories(percentiles)
            # add chart to the sheet
            sheet = self.wb.create_sheet("Total_Percentiles_" + str(i + 1))
            sheet.add_chart(chart)
            sheets.append(sheet)
        return sheets

    def create_total_latency_percentile_count_graphs(self, ws, prefix):
        title = "Total Percentiles Histogram"
        latency_cols = self.get_latency_percentile_count_columns(ws)
        chart = self.create_bar_chart(title, "Percentiles", "Count ", 25, 50)
        latency_series = self.get_latency_percentile_count_series(ws, prefix, self.percentile_count_names)
        for x in latency_series:
            chart.append(latency_series[x])
        # Add x-axis labels
        percentiles_counts = Reference(ws, min_col=latency_cols[self.percentile_count_names[0]], min_row=1,
                                max_col=latency_cols[self.percentile_count_names[-1]], max_row=1)
        chart.set_categories(percentiles_counts)
        # add chart to the sheet
        sheet = self.wb.create_sheet("Total_Percentiles_Histogram" )
        sheet.add_chart(chart)
        return sheet


    def create_throughput_mb_graph(self, ws, prefix):
        chart = self.create_line_chart("Throughput Variations in Mega Bytes / Seconds",
                                       "Intervals", "Throughput in MB/Sec", 25, 50)
        # adding data
        chart.append(self.get_throughput_write_request_mb_series(ws, prefix))
        chart.append(self.get_throughput_read_request_mb_series(ws, prefix))
        chart.append(self.get_throughput_mb_series(ws, prefix))
        # add chart to the sheet
        sheet = self.wb.create_sheet("MB_Sec")
        sheet.add_chart(chart)
        return sheet

    def create_throughput_records_graph(self, ws, prefix):
        chart = self.create_line_chart("Throughput Variations in Records / Seconds",
                                       "Intervals", "Throughput in Records/Sec", 25, 50)
        # adding data
        chart.append(self.get_throughput_write_request_records_series(ws, prefix))
        chart.append(self.get_throughput_read_request_records_series(ws, prefix))
        chart.append(self.get_throughput_records_series(ws, prefix))
        # add chart to the sheet
        sheet = self.wb.create_sheet("Records_Sec")
        sheet.add_chart(chart)
        return sheet

    def ensure_sbk_logo(self, img_path='./images/sbk-logo.png', cell='K7', scale=0.5):
        ws = self.wb['SBK']
        # Put your sheet in the loader
        image_loader = SheetImageLoader(ws)

        if image_loader.image_in(cell):
            print(f"SBK logo Image already exists in SBK sheet")
            return

        # Check if image already exists in the sheet
        img_abs_path = os.path.abspath(img_path)
        # Add image if not present
        if os.path.exists(img_abs_path):
            img = Image(img_abs_path)
            img.width *= scale
            img.height *= scale
            ws.add_image(img, cell)
            print(f"Inserted image at {img_abs_path} in SBK sheet")
        else:
            print(f"Image not found: {img_abs_path}")

    def create_graphs(self):
        r_name = sheets_constants.R_PREFIX + "1"
        r_ws = self.wb[r_name]
        r_prefix = r_name + self.get_storage_name(r_ws)
        t_name = sheets_constants.T_PREFIX + "1"
        t_ws = self.wb[t_name]
        t_prefix = t_name + self.get_storage_name(t_ws)
        self.create_throughput_mb_graph(r_ws, r_prefix)
        self.create_throughput_records_graph(r_ws, r_prefix)
        self.create_latency_compare_graphs(r_ws, r_prefix)
        self.create_latency_graphs(r_ws, r_prefix)
        self.create_total_latency_percentile_graphs(t_ws, t_prefix)
        self.wb.save(self.file)
        print("file : %s updated with graphs" % self.file)
