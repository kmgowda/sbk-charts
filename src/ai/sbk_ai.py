#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

from src.charts.multicharts import SbkMultiCharts
from src.custom_ai.hugging_face import HuggingFace
from openpyxl.utils import range_boundaries
# Add a border around the analysis section
from openpyxl.styles import Font, Border, Side, Alignment   
import textwrap
from openpyxl.worksheet.dimensions import ColumnDimension

class SbkAI(SbkMultiCharts):
    def __init__(self, version, file):
        super().__init__(version, file)
        self.ai = HuggingFace()

    def __series_values_to_list(self, ws, series):
        """
        Convert an openpyxl.chart.series.Series 'val' to a list of Python values.
        ws    : openpyxl worksheet containing the data
        series: openpyxl.chart.series.Series instance
        """
        # The underlying NumRef -> 'f' attribute contains the range string like 'Sheet1'!$B$2:$B$10
        num_ref = series.val.numRef
        range_str = num_ref.f  # e.g. 'Sheet1'!$B$2:$B$10

        # Remove sheet name if present
        if "!" in range_str:
            _, coord = range_str.split("!", 1)
        else:
            coord = range_str

        min_col, min_row, max_col, max_row = range_boundaries(coord)

        values = []
        for row in ws.iter_rows(min_row=min_row, max_row=max_row,
                                min_col=min_col, max_col=max_col):
            for cell in row:
                values.append(cell.value)
        return values

    def create_summary_sheet(self):
        sheet = super().create_summary_sheet()
        if sheet is None:
            print("Warning: Could not create summary sheet")
            return None
            
        throughputs = dict()
        for name in self.wb.sheetnames:
            if self.is_rnum_sheet(name):
                ws = self.wb[name]
                prefix = name + "-" + self.get_storage_name(ws)
                throughputs[self.get_storage_name(ws)] = self.__series_values_to_list(ws, self.get_throughput_mb_series(ws, prefix))
        status, analysis = self.ai.get_throughput_mb_analysis(throughputs)
        #print(analysis)
        if not status:
            print(analysis)
            return sheet
        
        # Add analysis text to the summary sheet
        if analysis:
            try:
                # Find the next available row after existing content
                max_row = sheet.max_row + 2  # Add some spacing
                
                # Add a title for the analysis section
                title_cell = sheet.cell(row=max_row, column=7)
                title_cell.value = "AI Performance Analysis"
                title_cell.font = Font(size=14, bold=True, color="FF0000")  # Red, bold, 14pt

                # Add the Throughput analysis section
                cell = sheet.cell(row=max_row + 2, column=7)
                cell.value = "Throughput Analysis"
                cell.font = Font(size=12, bold=True, color = "EE00FF")
                
                # Set column width to fit 80 characters
                # Using a larger multiplier to ensure 120 characters fit comfortably
                sheet.column_dimensions['H'].width = 120 * 0.90  # Increased from 0.14 to 0.20 for better fit
                
                # Add the analysis text with word wrap
                cell = sheet.cell(row=max_row + 2, column=8)
                cell.value = analysis
                cell.font = Font(size=12)
                cell.border = Border(left=Side(style='thin'),
                                   right=Side(style='thin'),
                                   top=Side(style='thin'),
                                   bottom=Side(style='thin'))

                # Enable text wrapping and set row height to auto-adjust
                cell.alignment = Alignment(wrap_text=True, vertical='top')
                
                # Calculate required row height based on text length and wrap at 80 characters
                wrapped_lines = []
                for line in analysis.split('\n'):
                    wrapped_lines.extend(textwrap.wrap(line, width=120))

                # Set row height (20 points per line, minimum 20)
                row_height = max(20, len(wrapped_lines) * 20)
                sheet.row_dimensions[max_row + 2].height = row_height

            except Exception as e:
                print(f"Error adding analysis to summary sheet: {str(e)}")
        
        return sheet


    def create_graphs(self):
        if self.check_time_units():
            self.create_summary_sheet()
            self.create_multi_throughput_mb_graph()
            self.create_multi_throughput_records_graph()
            self.create_all_latency_compare_graphs()
            self.create_multi_latency_compare_graphs()
            self.create_multi_latency_graphs()
            self.create_multi_write_read_records_graph()
            self.create_multi_write_read_mb_graph()
            self.create_multi_write_read_timeout_events_graph()
            self.create_multi_write_read_timeout_events_per_sec_graph()
            self.create_total_multi_latency_percentile_graphs()
            self.create_total_multi_latency_percentile_count_graphs()
            self.create_total_mb_compare_graph()
            self.create_total_throughput_mb_compare_graph()
            self.create_total_throughput_records_compare_graph()
            self.create_total_min_latency_compare_graph()
            self.create_total_avg_latency_compare_graph()
            self.create_total_max_latency_compare_graph()
            self.create_total_write_read_timeout_events_compare_graph()
            self.wb.save(self.file)
            print("file : %s updated with graphs and AI documentation" % self.file)
