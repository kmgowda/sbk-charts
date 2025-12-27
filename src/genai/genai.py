#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

from typing import final


class SbkGenAI:
    def __init__(self):
        self.storage_stats = None
        pass

    @final
    def set_storage_stats(self, stats):
        self.storage_stats = stats
        pass

    def get_model_description(self):
        pass

    def get_throughput_analysis(self):
        pass

    def get_latency_analysis(self):
        pass


