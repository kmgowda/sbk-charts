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
from dataclasses import dataclass
from typing import Final, Dict, Any, Optional

@dataclass(frozen=True)
class StorageStat:
    storage: Final[Optional[str]]
    timeunit: Final[Optional[str]]
    action: Final[Optional[str]]
    regular: Final[Optional[Dict[str, Any]]]
    total: Final[Optional[Dict[str, Any]]]

    def __post_init__(self):
        # Convert regular and total to empty dict if None
        if self.regular is None:
            object.__setattr__(self, 'regular', {})
        if self.total is None:
            object.__setattr__(self, 'total', {})

    @final
    def get_total_sum_value(self, name):
        return sum(self.total[name])

    @final
    def get_total_avg_value(self, name):
        return self.get_total_sum_value(name)/len(self.total)

    @final
    def get_regular_sum_value(self, name):
        return sum(self.regular[name])

    @final
    def get_regular_avg_value(self, name):
        return self.get_regular_sum_value(name)/len(self.regular)






