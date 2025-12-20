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

    def create_multi_throughput_mb_graph(self ):
        sheet = super().create_multi_throughput_mb_graph()
        throughputs = dict()
        for name in self.wb.sheetnames:
            if self.is_rnum_sheet(name):
                ws = self.wb[name]
                prefix = name + "-" + self.get_storage_name(ws)
                throughputs[self.get_storage_name(ws)] = self.__series_values_to_list(ws, self.get_throughput_mb_series(ws, prefix))
        analysis = self.ai.get_throughput_mb_analysis(throughputs)
        print(analysis)


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
