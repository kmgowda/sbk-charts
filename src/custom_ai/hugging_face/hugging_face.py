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
from typing import Tuple
from huggingface_hub import InferenceClient
from src.genai.genai import SbkGenAI

HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def _call_llm_for_analysis(model_id, prompt):
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

    client = InferenceClient(model=model_id, token=api_token)

    completion = client.chat_completion(  # ← key change
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
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
        return (True, "Hugging Face Inference APIs with model ID: " + self.model_id)


    def get_throughput_analysis(self):
        """Generate  throughput statistics  analysis.

        Returns
        - tuple: (True, <analysis string>) on success or (False, <message>) on
          failure (for example, when there are no throughput values).
        """

        return _call_llm_for_analysis(self.model_id, self.get_throughput_prompt())


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

        return _call_llm_for_analysis(self.model_id, self.get_latency_prompt())

    def get_total_mb_analysis(self):
        """Generate analysis of total MB processed and reporting intervals.

        Returns:
            tuple: (success, analysis) where:
                - success (bool): True if analysis was successful
                - analysis (str): The analysis text or error message
        """

        return _call_llm_for_analysis(self.model_id, self.get_total_mb_prompt())

    def get_percentile_histogram_analysis(self):
        """Generate analysis of percentile histogram data.

        Returns:
            tuple: (success, analysis) where:
                - success (bool): True if analysis was successful
                - analysis (str): The analysis text or error message
        """

        return _call_llm_for_analysis(self.model_id, self.get_percentile_histogram_prompt())

    def get_response(self, query) -> Tuple[bool, str]:
        """
        Generate a response for a custom query using RAG-enhanced context.

        Args:
            query: The query string to analyze

        Returns:
            Tuple[bool, str]: A tuple containing:
                - bool: True if analysis was successful, False otherwise
                - str: The analysis text or error message if analysis failed
        """
        try:

            # Check if this is a storage comparison query
            is_comparison = self._is_storage_comparison_query(query)
            
            # Create a prompt for the custom query
            if is_comparison:
                # Special handling for "which storage system is doing better" type questions
                query_lower = query.lower()
                if 'doing better' in query_lower or 'better' in query_lower:
                    prompt = f"""You are a storage performance engineer specializing in comparative analysis. Please analyze the following query about storage system performance:

Query: {query}

Please provide a detailed comparative analysis focusing on READ PERFORMANCE and THROUGHPUT:

Primary Analysis Requirements:
1. Focus on READ THROUGHPUT performance as the primary comparison metric
2. Identify which storage system has better read throughput (higher MB/s is better)
3. Compare read performance across all storage systems that have read operations
4. If a storage system only has write operations, clearly state this limitation
5. Provide specific throughput numbers and performance rankings

Analysis Structure:
1. **Read Throughput Comparison**: Compare storage systems with read operations
   - List storage systems in order of read throughput performance (best to worst)
   - Provide actual throughput values (MB/s) for each system
   - Highlight the best performing system for read operations

2. **Performance Classification**: 
   - Categorize systems as high/medium/low performance based on throughput
   - Explain what makes a system "better" in terms of read performance

3. **Workload Considerations**:
   - Note which systems have read-only, write-only, or mixed workloads
   - Explain how workload type affects the comparison

4. **Clear Recommendation**:
   - State clearly: "Storage system X is doing better in terms of read throughput"
   - Provide the specific performance difference

Important Guidelines:
- Higher throughput (MB/s) = Better performance for read operations
- Focus on READ performance unless the query specifically asks about write performance
- Use the contextual benchmark data to provide specific, data-driven insights
- If comparing systems with different workload types, explain the limitations

Use the contextual information from the benchmark data to provide specific, data-driven insights and recommendations."""
                else:
                    prompt = f"""You are a storage performance engineer specializing in comparative analysis. Please analyze the following query about storage system performance:

Query: {query}

Please provide a detailed comparative analysis that:
1. Identifies the specific storage systems being compared
2. Analyzes performance metrics (throughput, latency, IOPS, etc.) for each system
3. Considers the workload types (read-only, write-only, mixed) and their impact on performance
4. Takes into account time units and action types (reading, writing, read_write) when making comparisons
5. Provides clear recommendations based on the data and specific use cases
6. Explains the technical reasons for performance differences
7. Highlights which storage system performs better for specific workloads

Important comparison guidelines:
- Compare systems with similar workload types (read-only vs read-only, write-only vs write-only, mixed vs mixed)
- Consider time units when comparing latency (MS vs US)
- Account for the fact that some systems may only have write operations or only read operations
- Use the contextual information from benchmark data to provide specific, data-driven insights
- If comparing systems with different workload characteristics, explain the limitations of the comparison

Use the contextual information from the benchmark data to provide specific, data-driven insights and recommendations."""
            else:
                prompt = f"""You are a storage performance engineer. Please analyze the following query based on the provided context:

Query: {query}

Please provide a detailed technical analysis that addresses the query comprehensively. Use the contextual information provided to give specific and accurate insights."""
            
            # Enhance with RAG context
            enhanced_prompt = self._enhance_prompt_with_rag(prompt, query)
            
            return _call_llm_for_analysis(self.model_id, enhanced_prompt)
            
        except Exception as e:
            return False, f"Failed to generate response for query: {str(e)}"
    
    def _is_storage_comparison_query(self, query: str) -> bool:
        """
        Check if the query is asking for a storage system comparison.
        
        Args:
            query: The query string
            
        Returns:
            bool: True if this is a storage comparison query
        """
        comparison_indicators = [
            'better', 'worse', 'compare', 'comparison', 'versus', 'vs', 'against',
            'which', 'what', 'best', 'worst', 'faster', 'slower', 'higher', 'lower',
            'perform', 'performance', 'recommend', 'choose', 'select', 'winner',
            'outperform', 'superior', 'inferior', 'preferable'
        ]
        
        storage_indicators = [
            'storage', 'system', 'device', 'drive', 'ssd', 'hdd', 'nvme', 'filesystem',
            'minio', 's3', 'gcs', 'azure', 'blob', 'bucket'
        ]
        
        action_indicators = [
            'reading', 'writing', 'read', 'write', 'read_write', 'readwrite', 'mixed',
            'throughput', 'latency', 'iops', 'mb/s', 'bandwidth'
        ]
        
        query_lower = query.lower()
        has_comparison = any(indicator in query_lower for indicator in comparison_indicators)
        has_storage = any(indicator in query_lower for indicator in storage_indicators)
        has_action = any(indicator in query_lower for indicator in action_indicators)
        
        # Consider it a comparison query if it has comparison indicators AND either storage or action indicators
        return has_comparison and (has_storage or has_action)
