#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""sbk_charts.stat.storage

Data classes representing per-storage statistics collected from SBK runs.

This module defines the immutable `StorageStat` dataclass which stores both
interval-level (`regular`) and aggregated/total (`total`) measurements for
a particular storage system and action. Small convenience methods are
provided to compute sums and averages of recorded numeric series.

"""

from typing import final
from dataclasses import dataclass
from typing import Final, Dict, Any, Optional

@dataclass(frozen=True)
class StorageStat:
    """Immutable container for a storage system's benchmark statistics.

    Attributes
    - storage: Optional[str] name of the storage/driver (e.g. 'minio')
    - timeunit: Optional[str] latency time unit (e.g. 'MS' or 'US')
    - action: Optional[str] action performed (e.g. 'read' or 'write')
    - regular: Optional[Dict[str, Any]] mapping of interval/regular metrics
      (lists of numeric values) collected across measurement intervals.
    - total: Optional[Dict[str, Any]] mapping of aggregated/total metrics
      (lists or single-value summaries) for the entire run.

    Notes
    - The dataclass is frozen to make instances hashable and safe to share.
    - The __post_init__ method ensures `regular` and `total` are dictionaries
      (replacing None with empty dict) so downstream code can index them
      without guarding for None.
    """
    storage: Final[Optional[str]]
    timeunit: Final[Optional[str]]
    action: Final[Optional[str]]
    regular: Final[Optional[Dict[str, Any]]]
    total: Final[Optional[Dict[str, Any]]]

    def __post_init__(self):
        """Normalize optional mapping fields after object creation.

        If either `regular` or `total` is None at construction time, this
        replaces them with an empty dict so other helper methods can operate
        without additional None checks.
        """
        # Convert regular and total to empty dict if None
        if self.regular is None:
            object.__setattr__(self, 'regular', {})
        if self.total is None:
            object.__setattr__(self, 'total', {})

    @final
    def get_total_sum_value(self, name):
        """Return the sum of the values stored under `name` in `total`.

        Parameters
        - name (str): key name in the `total` mapping whose values will be summed.

        Returns
        - numeric: sum(self.total[name])

        Raises
        - KeyError if `name` is not present in the `total` mapping.
        """
        return sum(self.total[name])

    @final
    def get_total_avg_value(self, name):
        """Return the average of the values stored under `name` in `total`.

        The average is computed as the sum divided by the number of elements
        present in the `total` mapping. This method assumes that
        `self.total[name]` is an iterable of numeric values.
        """
        return self.get_total_sum_value(name)/len(self.total[name])

    @final
    def get_regular_sum_value(self, name):
        """Return the sum of the values stored under `name` in `regular`.

        Parameters
        - name (str): key name in the `regular` mapping whose values will be summed.

        Returns
        - numeric: sum(self.regular[name])

        Raises
        - KeyError if `name` is not present in the `regular` mapping.
        """
        return sum(self.regular[name])

    @final
    def get_regular_avg_value(self, name):
        """Return the average of the values stored under `name` in `regular`.

        The average is computed as the sum divided by the number of elements
        present in the `regular` mapping. This method assumes that
        `self.regular[name]` is an iterable of numeric values.
        """
        return self.get_regular_sum_value(name)/len(self.regular[name])
