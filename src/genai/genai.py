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
from statistics import mean
from typing import final, List, Tuple, Optional, Any
from src.charts import constants


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
        self.rag_pipeline = None

    @final
    def set_storage_stats(self, stats: List[Any]) -> None:
        """
        Set the storage statistics to be analyzed.
        
        This method is marked as final to ensure consistent behavior across all subclasses.
        
        Args:
            stats: A list of storage statistics objects containing benchmark data.
        """
        self.storage_stats = stats

    @final
    def set_rag_pipeline(self, rag_pipeline) -> None:
        """
        Set the RAG pipeline for context-enhanced analysis.
        
        This method is marked as final to ensure consistent behavior across all subclasses.
        
        Args:
            rag_pipeline: A RAG pipeline instance for retrieving contextual data.
        """
        self.rag_pipeline = rag_pipeline

    def _enhance_prompt_with_rag(self, base_prompt: str, query: str = None) -> str:
        """
        Enhance a prompt with relevant context from the RAG pipeline.
        
        Args:
            base_prompt: The original prompt to enhance
            query: Specific query for RAG retrieval (optional)
            
        Returns:
            str: Enhanced prompt with RAG context
        """
        if not self.rag_pipeline:
            return base_prompt
        
        try:
            # Use the query or derive one from the base prompt
            search_query = query or self._extract_query_from_prompt(base_prompt)
            
            # Retrieve relevant context
            context_list = self.rag_pipeline.retrieve_context(search_query, n_results=3)
            
            if context_list:
                # Format context for the prompt
                context_text = self.rag_pipeline.format_context_for_prompt(context_list)
                
                # Create enhanced prompt
                enhanced_prompt = f"""{base_prompt}

CONTEXTUAL INFORMATION:
{context_text}

Please use the above contextual information along with your analysis to provide more accurate and detailed insights."""
                return enhanced_prompt
            
        except Exception as e:
            # If RAG enhancement fails, return the original prompt
            print(f"Warning: Failed to enhance prompt with RAG context: {str(e)}")
        
        return base_prompt

    def _extract_query_from_prompt(self, prompt: str) -> str:
        """
        Extract a meaningful search query from the prompt.
        
        Args:
            prompt: The prompt to extract query from
            
        Returns:
            str: Extracted query string
        """
        # Simple keyword extraction - can be enhanced
        keywords = []
        
        # Look for performance-related keywords
        performance_terms = ['throughput', 'latency', 'performance', 'mb/s', 'iops', 'storage']
        for term in performance_terms:
            if term.lower() in prompt.lower():
                keywords.append(term)
        
        # If no specific keywords found, use first 100 characters
        if not keywords:
            return prompt[:100].strip()
        
        return " ".join(keywords)

    def add_args(self, parser):
        pass

    def parse_args(self, args):
        pass

    def open(self, args):
        pass

    def close(self, args):
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

    def get_throughput_prompt(self):
        """Generate a prompt for throughput analysis.
        Returns:
            str: A formatted prompt for the AI model to analyze throughput data.
        """
        if not self.storage_stats:
            raise RuntimeError("Storage stats not available")

        # 1) Compute per-system stats
        stats = list()
        for stat in self.storage_stats:
            stats.append({
                "storage": stat.storage,
                "min": min(stat.regular[constants.MB_PER_SEC]),
                "max": max(stat.regular[constants.MB_PER_SEC]),
                "avg": mean(stat.regular[constants.MB_PER_SEC]),
                "count": len(stat.regular[constants.MB_PER_SEC]),
            }
            )

        # 2) Rank by average throughput (descending)
        ranked = sorted(stats, key=lambda kv: kv["avg"], reverse=True)

        # 3) Build a compact metrics table for the prompt
        lines = []
        for rank, s in enumerate(ranked, start=1):
            lines.append(
                f"{rank}. {s['storage']}: avg={s['avg']:.2f} MB/s, "
                f"min={s['min']:.2f}, max={s['max']:.2f}, n={s['count']}"
            )
        metrics_block = "\n".join(lines)

        # 4) Prompt engineering for a local instruct model
        prompt = (
            "You are a storage performance Engineer. "
            "I need a detailed technical analysis of storage system throughput based on the following metrics. "
            "Analyze the following throughput benchmark results for different storage systems. "
            "Throughput numbers are in MB/s, and higher values are better.\n\n"
            "Tasks:\n"
            "- Identify which storage systems have the highest and lowest minimum, average and maximum throughput.\n"
            "- Quantify relative differences roughly (for example, 'about 2x higher').\n"
            "- Mention any big gaps or interesting patterns.\n"
            "- If any sentence of paragraph is of more than 70 characters, break it into multiple sentences.\n"
            "Here are the measurements:\n"
            f"{metrics_block}\n\n"
            "Now write the analysis in clear, technical English."
        )
        
        # Enhance with RAG context if available
        return self._enhance_prompt_with_rag(prompt, "throughput analysis storage performance")

    def get_latency_prompt(self):
        """Generate a prompt for latency analysis.
          Returns:
              str: A formatted prompt for the AI model to analyze latency data.
          """
        if not self.storage_stats:
            raise RuntimeError("Storage stats not available")

        # Define the key latency metrics we want to analyze
        latency_metrics = [
            constants.AVG_LATENCY,
            constants.MIN_LATENCY,
            constants.MAX_LATENCY,
            constants.PERCENTILE_50,  # Median
            constants.PERCENTILE_90,
            constants.PERCENTILE_95,
            constants.PERCENTILE_99,
            constants.PERCENTILE_99_9,
            constants.PERCENTILE_99_99
        ]

        # Collect latency statistics for each storage system
        stats = []
        for stat in self.storage_stats:
            if not stat.total:
                continue

            storage_stats = {
                constants.STORAGE: stat.storage,
                "action": stat.action,
                "metrics": {},
                "total_records": sum(stat.total.get(constants.RECORDS, [0]))
            }

            # Get values for each latency metric
            for metric in latency_metrics:
                if metric in stat.total:
                    values = stat.total[metric]
                    if values:
                        storage_stats["metrics"][metric] = {
                            "min": min(values),
                            "max": max(values),
                            "avg": sum(values) / len(values)
                        }

            # Add time unit for context
            storage_stats[constants.LATENCY_TIME_UNIT] = stat.timeunit
            stats.append(storage_stats)

        if not stats:
            raise RuntimeError("No valid latency data available for analysis.")

        # Build a comparison table for the prompt
        def format_value(metric_data, key):
            return f"{metric_data[key]:.2f}" if metric_data else "N/A"

        # Create a table with the most important metrics
        table_rows = []
        for stat in stats:
            metrics = stat["metrics"]
            row = [
                f"{stat[constants.STORAGE]} ({stat['action']})",
                format_value(metrics.get(constants.AVG_LATENCY, {}), "avg"),
                format_value(metrics.get(constants.PERCENTILE_50, {}), "avg"),
                format_value(metrics.get(constants.PERCENTILE_95, {}), "avg"),
                format_value(metrics.get(constants.PERCENTILE_99_9, {}), "avg"),
                format_value(metrics.get(constants.MAX_LATENCY, {}), "max"),
                f"{stat['total_records']:,}",
                stat[constants.LATENCY_TIME_UNIT]
            ]
            table_rows.append(" | ".join(row))

        # Create the prompt for the LLM
        prompt = (
            "I need a detailed technical analysis of storage system latencies based on the following metrics. "
            "The data represents various latency measurements across different storage systems.\n\n"
            "### Latency Metrics (all values in their respective time units):\n"
            "| Storage (Action) | Avg Latency | p50 (Median) | p95 | p99.9 | Max Latency | Total Records | Time Unit |\n"
            "|------------------|-------------|--------------|-----|-------|-------------|----------------|-----------|\n"
            f"{chr(10).join(table_rows)}\n\n"
            "### Analysis Instructions:\n"
            "1. Compare the latency profiles and total records across different storage systems\n"
            "2. Identify which storage system performs best for different percentiles\n"
            "3. Note any significant differences between average and tail latencies for given total records\n"
            "4. Highlight any anomalies or interesting patterns in the data\n"
            "5. Consider the impact of the time unit (microseconds, milliseconds, etc.) and total records on the interpretation\n\n"
            "if any sentence of paragraph is of more than 70 characters, break it into multiple sentences.\n"
            "Provide a clear, concise technical analysis that would be useful for a storage engineer "
            "evaluating these systems. Focus on the most significant findings and their implications."
        )

        # Enhance with RAG context if available
        return self._enhance_prompt_with_rag(prompt, "latency analysis storage performance percentiles")


    def get_total_mb_prompt(self):

        if not self.storage_stats:
            raise RuntimeError("Storage stats not available")

        # Build the prompt
        lines = []
        for i, s in enumerate(self.storage_stats, 1):
            lines.append(
                f"{i}. {s.storage}: "
                f"MB: {sum(s.total[constants.MB])} "
                f"Time: {sum(s.total[constants.REPORT_SECONDS])} "
                f"n= {len(s.total[constants.MB])} "
            )

        prompt = (
            "You are a storage performance engineer. Analyze the following data processing metrics:\n\n"
            "For each storage system, you'll see:\n"
            "- MB: Total megabytes processed\n"
            "- Time: Total reporting intervals in seconds\n"
            "- n: Number of data points\n\n"
            f"{chr(10).join(lines)}\n\n"
            "Please provide a technical analysis that includes:\n"
            "1. Which storage system processed the most/least data in total\n"
            "2. Any correlations between data volume and reporting time\n"
            "3. Potential bottlenecks or anomalies in the data processing\n"
            "4. Recommendations for optimizing data processing based on the patterns observed\n\n"
            "Keep the analysis concise, technical, and focused on actionable insights."
        )
        
        # Enhance with RAG context if available
        return self._enhance_prompt_with_rag(prompt, "total MB data processing analysis")

    def get_percentile_histogram_prompt(self):

        if not self.storage_stats:
            raise RuntimeError("Storage stats not available")

        # List of all percentile count constants we want to analyze
        percentile_count_consts = [
            constants.PERCENTILE_COUNT_5, constants.PERCENTILE_COUNT_10,
            constants.PERCENTILE_COUNT_15, constants.PERCENTILE_COUNT_20, constants.PERCENTILE_COUNT_25,
            constants.PERCENTILE_COUNT_30, constants.PERCENTILE_COUNT_35, constants.PERCENTILE_COUNT_40,
            constants.PERCENTILE_COUNT_45, constants.PERCENTILE_COUNT_50, constants.PERCENTILE_COUNT_55,
            constants.PERCENTILE_COUNT_60, constants.PERCENTILE_COUNT_65, constants.PERCENTILE_COUNT_70,
            constants.PERCENTILE_COUNT_75, constants.PERCENTILE_COUNT_80, constants.PERCENTILE_COUNT_85,
            constants.PERCENTILE_COUNT_90, constants.PERCENTILE_COUNT_92_5, constants.PERCENTILE_COUNT_95,
            constants.PERCENTILE_COUNT_97_5, constants.PERCENTILE_COUNT_99, constants.PERCENTILE_COUNT_99_25,
            constants.PERCENTILE_COUNT_99_5, constants.PERCENTILE_COUNT_99_75, constants.PERCENTILE_COUNT_99_9,
            constants.PERCENTILE_COUNT_99_95, constants.PERCENTILE_COUNT_99_99
        ]

        # Build the prompt with percentile count data
        lines = []
        for i, stat in enumerate(self.storage_stats, 1):
            if not stat.total:
                continue

            # Get all percentile count values for this storage
            percentile_counts = []
            for const in percentile_count_consts:
                try:
                    # Get the percentile count values from stat.total
                    values = stat.total[const]
                    if values:  # Only process if there are values
                        # Calculate the total count for this percentile
                        total_count = sum(values)
                        # Extract the percentile number from the constant name
                        percentile_str = const.replace('Percentile_Count_', '').replace('_', '.')
                        try:
                            percentile = float(percentile_str)
                            percentile_counts.append((percentile, total_count))
                        except (ValueError, TypeError) as e:
                            raise ValueError(f"Invalid percentile value: {percentile_str}, error: {e}")
                except (KeyError, AttributeError) as e:
                    raise ValueError(f"Invalid total percentile value: {const}, error: {e}")

            if not percentile_counts:
                continue

            # Sort by percentile
            percentile_counts.sort()

            # Format the data for the prompt
            storage_line = f"{i}. {stat.storage} ({stat.action}):\n"
            for percentile, count in percentile_counts:
                storage_line += f"   - {percentile}%: {count:,} samples\n"
            lines.append(storage_line)

        if not lines:
            raise RuntimeError("No valid percentile count data available for analysis.")

        prompt = (
            "You are a storage performance engineer. Analyze the following percentile histogram data:\n\n"
            "For each storage system, you'll see the number of samples at each percentile level.\n"
            "This data shows the distribution of request latencies across different percentiles.\n\n"
            f"{chr(10).join(lines)}\n\n"
            "Please provide a technical analysis that includes:\n"
            "1. The overall latency distribution pattern for each storage system\n"
            "2. Any significant spikes or anomalies in the percentile distribution\n"
            "3. How the distributions compare across different storage systems\n"
            "4. Potential performance bottlenecks or optimization opportunities\n"
            "5. Any patterns indicating specific performance characteristics (e.g., long tail latencies)\n\n"
            "Focus on actionable insights and keep the analysis concise and technical.\n"
            "Note: The values represent the number of operations that completed within each latency percentile. "
            "Higher counts in higher percentiles may indicate performance issues."
        )
        
        # Enhance with RAG context if available
        return self._enhance_prompt_with_rag(prompt, "percentile histogram distribution analysis")


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

    @abstractmethod
    def get_response(self, query)-> Tuple[bool, str]:
        pass