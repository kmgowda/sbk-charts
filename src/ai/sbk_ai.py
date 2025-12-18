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


class SbkAI(SbkMultiCharts):
    def __init__(self, version, file):
        super().__init__(version, file)

    def create_multi_throughput_mb_graph(self ):
        super().create_multi_throughput_mb_graph()


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
