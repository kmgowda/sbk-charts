#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

class StorageStat:
    def __init__(self, storage=None, timeunit=None, action=None):
        self.storage = storage
        self.timeunit = timeunit
        self.action = action
        self.regular = dict()
        self.total = dict()





