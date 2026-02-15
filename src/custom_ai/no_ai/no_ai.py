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
No AI Implementation Module

This module provides a NoAI class that implements the SbkGenAI interface
but returns error messages for all analysis methods. This is used as a fallback
when AI analysis is not available or not enabled.
"""
from typing import Tuple
from src.genai.genai import SbkGenAI


class NoAI(SbkGenAI):
    """
    A no-op implementation of the SbkGenAI interface.
    
    This class provides implementations for all required methods but always
    returns an error message indicating that AI analysis is not enabled.
    """
    
    def get_model_description(self) -> Tuple[bool, str]:
        """
        Return an error message indicating AI is not enabled.
        
        Returns:
            Tuple[bool, str]: (False, error_message)
        """
        return False, "AI analysis is not enabled. Please use one of the available AI implementations."
    
    def get_throughput_analysis(self) -> Tuple[bool, str]:
        """
        Return an error message for throughput analysis.
        
        Returns:
            Tuple[bool, str]: (False, error_message)
        """
        return False, "Throughput analysis is not available: AI is not enabled."
    
    def get_latency_analysis(self) -> Tuple[bool, str]:
        """
        Return an error message for latency analysis.
        
        Returns:
            Tuple[bool, str]: (False, error_message)
        """
        return False, "Latency analysis is not available: AI is not enabled."
    
    def get_total_mb_analysis(self) -> Tuple[bool, str]:
        """
        Return an error message for total MB analysis.
        
        Returns:
            Tuple[bool, str]: (False, error_message)
        """
        return False, "Total MB analysis is not available: AI is not enabled."
    
    def get_percentile_histogram_analysis(self) -> Tuple[bool, str]:
        """
        Return an error message for percentile histogram analysis.
        
        Returns:
            Tuple[bool, str]: (False, error_message)
        """
        return False, "Percentile histogram analysis is not available: AI is not enabled."

    def get_response(self, query: str) -> Tuple[bool, str]:
        """
        Return an error message for custom query analysis.
        
        Args:
            query: The query string to analyze
            
        Returns:
            Tuple[bool, str]: (False, error_message)
        """
        return False, f"Custom query analysis is not available: AI is not enabled. Query was: {query}"
