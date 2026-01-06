#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##
"""
SBK Generative AI Interface Module

This module defines the abstract base class `SbkGenAI` which serves as an interface
for different AI model implementations to provide performance analysis for storage benchmarks.

Classes:
    SbkGenAI: Abstract base class defining the interface for AI analysis of storage metrics.
"""
from abc import ABC, abstractmethod
from typing import final, List, Tuple, Optional, Dict, Any


class SbkGenAI(ABC):
    """
    Abstract base class for AI-powered storage benchmark analysis.
    
    This class defines the interface that all AI analysis implementations must follow
    to provide performance analysis for storage benchmarks. Concrete implementations
    should inherit from this class and implement all abstract methods.
    
    Attributes:
        storage_stats: List of storage statistics to be analyzed by the AI model.
    """
    def __init__(self):
        """Initialize the SbkGenAI instance with default values."""
        self.storage_stats: Optional[List[Any]] = None

    @final
    def set_storage_stats(self, stats: List[Any]) -> None:
        """
        Set the storage statistics to be analyzed.
        
        This method is marked as final to ensure consistent behavior across all subclasses.
        
        Args:
            stats: A list of storage statistics objects containing benchmark data.
        """
        self.storage_stats = stats

    def get_class_name(self):
        return self.__name__


    def add_args(self, parser):
        pass

    def parse_args(self, args):
        pass

    @abstractmethod
    def get_model_description(self) -> Tuple[bool, str]:
        """
        Get a description of the AI model being used for analysis.
        
        Returns:
            bool: True if analysis was successful, False otherwise
            str: A string describing the AI model, including its name and version.
        """
        pass

    @abstractmethod
    def get_throughput_analysis(self) -> Tuple[bool, str]:
        """
        Generate analysis of throughput metrics from the benchmark data.
        
        Returns:
            Tuple[bool, str]: A tuple containing:
                - bool: True if analysis was successful, False otherwise
                - str: The analysis text or error message if analysis failed
        """
        pass

    @abstractmethod
    def get_latency_analysis(self) -> Tuple[bool, str]:
        """
        Generate analysis of latency metrics from the benchmark data.
        
        Returns:
            Tuple[bool, str]: A tuple containing:
                - bool: True if analysis was successful, False otherwise
                - str: The analysis text or error message if analysis failed
        """
        pass

    @abstractmethod
    def get_total_mb_analysis(self) -> Tuple[bool, str]:
        pass

    @abstractmethod
    def get_percentile_histogram_analysis(self) -> Tuple[bool, str]:
        pass
