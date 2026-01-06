#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

"""sbk_charts.custom_ai.hugging_face

Integration wrapper for Hugging Face Inference APIs used by the SBK
analysis subsystem.

This module provides a small adapter class, `HuggingFace`, that delegates
prompt-based analysis requests to a specified Hugging Face model via the
`huggingface_hub.InferenceClient`.

Key behavior
- Reads the HUGGINGFACE_API_TOKEN environment variable for authentication.
- Builds prompts from benchmark statistics provided by the parent class
  (`SbkGenAI`) and requests chat-style completions from the model.
- Returns analysis text or an error tuple when the token is not available or
  when there is no data to analyze.

"""

import  os
from statistics import mean
from huggingface_hub import InferenceClient
from src.genai.genai import SbkGenAI
from src.charts import constants

HF_MODEL_ID = "google/gemma-2-2b-it"


def _call_llm_for_analysis(prompt):
    """Send a prompt to the configured Hugging Face model and return the reply.

    Parameters
    - prompt (str): the textual prompt to send to the model. Prompts are
      expected to be short technical instructions and metric tables.

    Returns
    - list: [True, <analysis string>] on success
            [False, <error message>] if no API token is configured or
            if another precondition prevents calling the API.

    Notes
    - Uses the `huggingface_hub.InferenceClient` and the chat_completion
      interface to obtain the model's response. The method intentionally
      returns a two-element list following the project's existing
      convention for success/failure and message payloads.
    """
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not api_token:
        return [False, (
            "LLM analysis is not available (missing HUGGINGFACE_API_TOKEN environment variable). "
            "Configure the token to enable Hugging Face-based analysis."
        )]

    client = InferenceClient(model=HF_MODEL_ID, token=api_token)

    completion = client.chat_completion(  # ← key change
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1800,
        temperature=0.4,
        top_p=0.9,
    )
    # OpenAI‑style response schema
    return (True, completion.choices[0].message["content"].strip())


class HuggingFace(SbkGenAI):
    """Adapter for calling a Hugging Face chat-style model for analysis.

    Inherits from `SbkGenAI` which provides the storage statistics data
    (self.storage_stats) assembled from SBK run results. This class turns
    those statistics into compact prompts and calls the Hugging Face
    Inference API to obtain human-readable technical analyses.

    Behavior
    - If the `HUGGINGFACE_API_TOKEN` environment variable is not set, calls
      to the LLM return an error tuple [False, message].
    - The `_call_llm_for_analysis` method uses chat_completion to keep the
      request aligned with conversational model APIs.
    """
    def __init__(self):
        super().__init__()
        self.model_id = HF_MODEL_ID


    def add_args(self, parser):
        parser.add_argument("-id", "--model_id", help="Hugging Face model ID; default model: "+HF_MODEL_ID, default=HF_MODEL_ID)
        parser.set_defaults(model_id=HF_MODEL_ID)

    def parse_args(self, args):
#        if hasattr(args, 'model_id') and args.model_id is not None:
            self.model_id = args.model_id

    def get_model_description(self):
        """Return a short description of the configured Hugging Face model.

        Returns
         - tuple: (True, <model string>) on success or (False, <message>) on
          failure.
        """
        return (True, "Hugging Face Inference APIs with model ID: " + HF_MODEL_ID)


    def get_throughput_analysis(self):
        """Generate a prompt from throughput statistics and request analysis.

        The method:
        1. Computes per-storage statistics (min/max/avg/count) from
           `self.storage_stats`.
        2. Ranks storage systems by average throughput.
        3. Builds a compact text table and a carefully-worded prompt for the
           model.
        4. Delegates to `_call_llm_for_analysis(prompt)` and returns its
           result.

        Returns
        - tuple: (True, <analysis string>) on success or (False, <message>) on
          failure (for example, when there are no throughput values).
        """

        # 1) Compute per-system stats
        stats = list()
        for stat in self.storage_stats:
            stats.append( {
                "storage": stat.storage,
                "min": min(stat.regular[constants.MB_PER_SEC]),
                "max": max(stat.regular[constants.MB_PER_SEC]),
                "avg": mean(stat.regular[constants.MB_PER_SEC]),
                "count": len(stat.regular[constants.MB_PER_SEC]),
            }
            )

        if not stats:
            return (False, "No valid throughput values available for analysis.")

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
            "- if any sentence of paragraph is of more than 70 characters, break it into multiple sentences.\n"
            "Here are the measurements:\n"
            f"{metrics_block}\n\n"
            "Now write the analysis in clear, technical English."
        )

        return _call_llm_for_analysis(prompt)


    def get_latency_analysis(self):
        """Builds a latency-focused prompt from stored percentile metrics.

        The method collects selected latency percentiles and summary metrics
        from `self.storage_stats`, formats them into a markdown-style table,
        and requests the Hugging Face model to produce a concise technical
        analysis. The returned analysis should be suitable for a storage
        engineer evaluating tail latencies and percentile behavior.

        Returns
        - list: [True, <analysis string>] on success or [False, <message>] on
          failure (for example, when latency data is missing).
        """
        if not self.storage_stats:
            return (False, "No storage statistics available for latency analysis.")

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
            return (False, "No valid latency data available for analysis.")

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

        return _call_llm_for_analysis(prompt)

    def get_total_mb_analysis(self):
        pass

    def get_percentile_histogram_analysis(self):
        pass

    def get_performance_summary(self):
        pass