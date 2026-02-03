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
SBK Simple RAG Pipeline Module

This module provides a simple RAG (Retrieval-Augmented Generation) pipeline that
doesn't require ChromaDB. It uses a simple keyword-based search approach as a
fallback when ChromaDB is not available.

Key Features:
- StorageStat data ingestion and chunking
- Simple keyword-based search (no embeddings required)
- Fallback implementation for systems without ChromaDB
- Context retrieval for AI prompts
- Support for multiple storage systems and metrics
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import re
from collections import defaultdict
from src.stat.storage import StorageStat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SbkSimpleRAGPipeline:
    """
    Simple RAG Pipeline for SBK AI Analysis (ChromaDB-free)
    
    This class provides a basic RAG implementation that doesn't require
    external vector databases. It uses keyword-based search and simple
    text matching for retrieving relevant context from storage benchmark data.
    
    Attributes:
        documents: List of processed documents from StorageStat objects
        metadata: List of metadata corresponding to each document
        storage_systems: Set of unique storage system names found in the data
        is_initialized: Flag indicating if the pipeline is ready
    """
    
    def __init__(self, collection_name: str = "sbk_benchmark_data"):
        """
        Initialize the simple RAG pipeline.
        
        Args:
            collection_name: Name for the collection (for compatibility with main RAG class)
        """
        self.collection_name = collection_name
        self.documents = []
        self.metadata = []
        self.is_initialized = False
        self.storage_systems = set()  # Store unique storage system names
    
    def initialize(self) -> bool:
        """
        Initialize the simple RAG pipeline.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Simple RAG pipeline (ChromaDB-free)")
            self.is_initialized = True
            logger.info("Simple RAG pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Simple RAG pipeline: {str(e)}")
            return False
    
    def ingest_storage_stats(self, storage_stats: List[StorageStat]) -> bool:
        """
        Ingest data from StorageStat objects into the simple RAG pipeline.
        
        Args:
            storage_stats: List of StorageStat objects to ingest
            
        Returns:
            bool: True if ingestion successful, False otherwise
        """
        if not self.is_initialized:
            logger.error("Simple RAG pipeline not initialized. Call initialize() first.")
            return False
            
        try:
            all_documents = []
            all_metadatas = []
            
            for storage_stat in storage_stats:
                logger.info(f"Processing storage stat for: {storage_stat.storage}")
                documents, metadatas = self._process_storage_stat(storage_stat)
                
                all_documents.extend(documents)
                all_metadatas.extend(metadatas)
            
            if not all_documents:
                logger.warning("No valid data found in storage stats")
                return False
            
            # Store in memory
            self.documents = all_documents
            self.metadata = all_metadatas
            
            # Extract storage system names from metadata
            self._extract_storage_systems()
            
            logger.info(f"Successfully ingested {len(all_documents)} data points from {len(storage_stats)} storage systems")
            if self.storage_systems:
                logger.info(f"Detected storage systems: {', '.join(sorted(self.storage_systems))}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest storage stats: {str(e)}")
            return False
    
    def _process_storage_stat(self, storage_stat: StorageStat, is_regular: bool = False) -> Tuple[List[str], List[Dict]]:
        """
        Process a single StorageStat object and extract documents and metadata.
        
        Args:
            storage_stat: StorageStat object to process
            
        Returns:
            Tuple of (documents, metadatas)
        """
        try:
            documents = []
            metadatas = []
            
            # Create base metadata
            base_metadata = {
                'storage': storage_stat.storage,
                'timeunit': storage_stat.timeunit,
                'action': storage_stat.action
            }
            
            # Process regular metrics
            if storage_stat.regular and is_regular:
                for metric_name, values in storage_stat.regular.items():
                    if values and len(values) > 0:
                        # Skip if all values are zero
                        if all(v == 0 for v in values):
                            logger.debug(f"Skipping zero-valued metric: {storage_stat.storage} {storage_stat.action} regular {metric_name}")
                            continue
                        
                        # Create document for regular metrics
                        doc_text = self._storage_stat_to_text(
                            storage_stat.storage, storage_stat.action, 
                            'regular', metric_name, values
                        )
                        
                        metadata = base_metadata.copy()
                        metadata.update({
                            'metric_type': 'regular',
                            'metric_name': metric_name,
                            'values': values,
                            'count': len(values),
                            'avg': sum(values) / len(values) if values else 0
                        })
                        
                        documents.append(doc_text)
                        metadatas.append(metadata)
            
            # Process total metrics
            if storage_stat.total:
                for metric_name, values in storage_stat.total.items():
                    if values and len(values) > 0:
                        # Skip if all values are zero
                        if all(v == 0 for v in values):
                            logger.debug(f"Skipping zero-valued metric: {storage_stat.storage} {storage_stat.action} total {metric_name}")
                            continue
                        
                        # Create document for total metrics
                        doc_text = self._storage_stat_to_text(
                            storage_stat.storage, storage_stat.action, 
                            'total', metric_name, values
                        )
                        
                        metadata = base_metadata.copy()
                        metadata.update({
                            'metric_type': 'total',
                            'metric_name': metric_name,
                            'values': values,
                            'count': len(values),
                            'avg': sum(values) / len(values) if values else 0
                        })
                        
                        documents.append(doc_text)
                        metadatas.append(metadata)
            
            return documents, metadatas
            
        except Exception as e:
            logger.error(f"Error processing storage stat {storage_stat.storage}: {str(e)}")
            return [], []
    
    def _storage_stat_to_text(self, storage: str, action: str, metric_type: str, 
                              metric_name: str, values: List[Any]) -> str:
        """
        Convert StorageStat data to a text representation for searching with enhanced comparison support.
        
        Args:
            storage: Storage system name
            action: Action type (read/write)
            metric_type: Type of metric (regular/total)
            metric_name: Name of the metric
            values: List of metric values
            
        Returns:
            str: Text representation of the storage stat
        """
        if not values:
            return f"No data available for {storage} {action} {metric_name}"
        
        avg_val = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        
        # Determine workload characteristics
        action_lower = action.lower()
        workload_type = "unknown"
        if 'read' in action_lower and 'write' in action_lower:
            workload_type = "mixed_workload"
        elif 'read' in action_lower:
            workload_type = "read_workload"
        elif 'write' in action_lower:
            workload_type = "write_workload"
        
        # Build base text parts
        text_parts = [
            f"Storage: {storage}",
            f"Action: {action}",
            f"Workload: {workload_type}",
            f"Metric Type: {metric_type}",
            f"Metric: {metric_name}",
            f"Average: {avg_val:.2f}",
            f"Min: {min_val:.2f}",
            f"Max: {max_val:.2f}",
            f"Count: {len(values)}"
        ]

        # Enhanced performance classification with emphasis on throughput and read performance
        metric_name_lower = metric_name.lower()
        
        # Special handling for throughput metrics - more specific detection
        is_throughput_metric = (
            'throughput' in metric_name_lower or 
            'mb/s' in metric_name_lower or 
            'mbs' in metric_name_lower or
            'mb/sec' in metric_name_lower or
            'mbps' in metric_name_lower or
            'bytes/sec' in metric_name_lower or
            'transfer_rate' in metric_name_lower or
            'bandwidth' in metric_name_lower
        )
        
        # Exclude request-based metrics from throughput classification
        is_request_metric = (
            'request' in metric_name_lower or
            'req' in metric_name_lower or
            'readrequest' in metric_name_lower or
            'writerequest' in metric_name_lower
        )
        
        # Only classify as throughput if it's actually a throughput metric and not a request metric
        if is_throughput_metric and not is_request_metric:
            if avg_val > 1000:
                text_parts.append("Performance: high_throughput")
                text_parts.append("Performance_Indicator: excellent")
            elif avg_val > 100:
                text_parts.append("Performance: medium_throughput")
                text_parts.append("Performance_Indicator: good")
            else:
                text_parts.append("Performance: low_throughput")
                text_parts.append("Performance_Indicator: poor")
        
        # Special handling for read operations
        if 'read' in action_lower:
            text_parts.append("Operation_Type: read_operation")
            if is_throughput_metric and not is_request_metric:
                text_parts.append("Read_Throughput_Performance: primary_metric")
                text_parts.append("Comparison_Metric: read_throughput")
            elif 'percentile' in metric_name_lower:
                text_parts.append("Read_Latency_Performance: important_metric")
        
        # Special handling for write operations
        if 'write' in action_lower:
            text_parts.append("Operation_Type: write_operation")
            if is_throughput_metric and not is_request_metric:
                text_parts.append("Write_Throughput_Performance: primary_metric")
            elif 'percentile' in metric_name_lower:
                text_parts.append("Write_Latency_Performance: important_metric")
        
        # Latency classification
        if 'percentile' in metric_name_lower:
            if avg_val < 1:
                text_parts.append("Performance: low_latency")
                text_parts.append("Performance_Indicator: excellent")
            elif avg_val < 10:
                text_parts.append("Performance: medium_latency")
                text_parts.append("Performance_Indicator: good")
            else:
                text_parts.append("Performance: high_latency")
                text_parts.append("Performance_Indicator: poor")
        
        # Add comparison importance flags for actual throughput metrics
        if 'read' in action_lower and is_throughput_metric and not is_request_metric:
            text_parts.append("Comparison_Importance: high")
            text_parts.append("Storage_Comparison_Key: read_throughput")
        
        return " | ".join(text_parts)
    
    def retrieve_context(self, query: str, n_results: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from the ingested data using keyword matching with enhanced read throughput prioritization.
        
        Args:
            query: Query string to search for
            n_results: Number of results to retrieve (increased default for comprehensive data)
            
        Returns:
            List of dictionaries containing retrieved context
        """
        if not self.is_initialized:
            logger.error("Simple RAG pipeline not initialized. Call initialize() first.")
            return []
        
        try:
            # Extract keywords from query
            query_keywords = self._extract_keywords(query.lower())
            
            # Score documents based on keyword matches with special emphasis on read throughput
            scored_docs = []
            query_lower = query.lower()
            is_better_query = 'better' in query_lower or 'doing better' in query_lower
            
            for i, doc in enumerate(self.documents):
                doc_lower = doc.lower()
                score = 0
                
                # Count keyword matches
                for keyword in query_keywords:
                    if keyword in doc_lower:
                        score += doc_lower.count(keyword)
                
                # Special scoring for "which storage system is doing better" queries
                if is_better_query:
                    # Boost score for read throughput data (actual throughput, not request metrics)
                    if 'read_throughput' in doc_lower or 'read_operation' in doc_lower:
                        score += 10
                    if ('throughput' in doc_lower and 'read' in doc_lower) and 'request' not in doc_lower:
                        score += 8
                    if 'mb/sec' in doc_lower and 'read' in doc_lower:
                        score += 9
                    if 'mb/s' in doc_lower and 'read' in doc_lower and 'request' not in doc_lower:
                        score += 7
                    if 'comparison_importance: high' in doc_lower:
                        score += 5
                    if 'storage_comparison_key: read_throughput' in doc_lower:
                        score += 7
                    
                    # Penalize request-based metrics for throughput comparisons
                    if 'request' in doc_lower and ('throughput' in doc_lower or 'mb/s' in doc_lower):
                        score -= 5
                
                # Always include some data from each storage system for comprehensive comparison
                metadata = self.metadata[i]
                storage_name = metadata.get('storage', '')
                if storage_name:
                    # Ensure we get at least some data from each storage system
                    if score == 0:
                        score = 1  # Minimum score to include all storage systems
                
                if score > 0:
                    scored_docs.append((score, i, doc, self.metadata[i]))
            
            # Sort by score (descending) and take top n_results
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            top_docs = scored_docs[:n_results]
            
            # Ensure we have balanced representation from all storage systems
            if len(top_docs) < n_results and self.storage_systems:
                # Add more documents to ensure each storage system is represented
                storage_represented = set()
                for score, idx, doc, metadata in top_docs:
                    storage_represented.add(metadata.get('storage', ''))
                
                # Add documents from underrepresented storage systems
                for score, idx, doc, metadata in scored_docs[n_results:]:
                    storage_name = metadata.get('storage', '')
                    if storage_name not in storage_represented and len(top_docs) < n_results:
                        top_docs.append((score, idx, doc, metadata))
                        storage_represented.add(storage_name)
            
            # Format results
            context_list = []
            for score, idx, doc, metadata in top_docs:
                context_item = {
                    'text': doc,
                    'metadata': metadata,
                    'distance': 1.0 / (1.0 + score)  # Convert score to distance-like metric
                }
                context_list.append(context_item)
            
            logger.info(f"Retrieved {len(context_list)} context items for query: {query[:50]}...")
            return context_list
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from a query with enhanced storage comparison support.
        
        Args:
            query: The query string
            
        Returns:
            List of keywords
        """
        query_lower = query.lower()
        
        # Simple keyword extraction - split on common delimiters and filter
        keywords = []
        
        # Split on common delimiters
        for word in re.split(r'[\s,;:!?]+', query):
            if len(word) >= 3:  # Only keep words with 3+ characters
                keywords.append(word)
        
        # Add common storage/performance terms with better throughput detection
        performance_terms = [
            'throughput', 'latency', 'percentile', 'percentile_count', 'performance', 'iops', 'storage', 'read', 'write',
            'mb/s', 'mbs', 'mb/sec', 'mbps', 'bytes/sec', 'transfer_rate', 'bandwidth', "histogram"
        ]
        for term in performance_terms:
            if term in query_lower:
                keywords.append(term)
        
        # Add action-specific keywords
        action_keywords = ['reading', 'writing', 'read_write', 'readwrite', 'mixed']
        for action in action_keywords:
            if action in query_lower:
                keywords.append(action)
        
        # Add time unit keywords
        time_units = ['ms', 'us', 'microseconds', 'milliseconds', 'seconds']
        for unit in time_units:
            if unit in query_lower:
                keywords.append(unit)
        
        # Special handling for "which storage system is doing better" type queries
        if 'better' in query_lower or 'best' in query_lower or 'doing better' in query_lower:
            # Always add throughput and read keywords for better performance queries
            keywords.extend(['throughput', 'read', 'performance'])
            
            # Add storage system names
            keywords.extend(self.storage_systems)
            
            # If no specific action mentioned, assume read performance is important
            if not any(action in query_lower for action in ['write', 'writing', 'write_only']):
                keywords.extend(['reading', 'read_workload', 'read_performance'])
        
        # Check for comparison queries and add storage system names
        if self._is_comparison_query(query):
            keywords.extend(self.storage_systems)
            
            # Add workload type keywords for better comparison context
            if hasattr(self, 'storage_info'):
                for storage_name, info in self.storage_info.items():
                    if info['workload_type'] in query_lower:
                        keywords.append(info['workload_type'])
        
        return list(set(keywords))  # Remove duplicates
    
    def format_context_for_prompt(self, context_list: List[Dict[str, Any]], max_context_length: int = 2000) -> str:
        """
        Format retrieved context into a string suitable for AI prompts.
        
        Args:
            context_list: List of context items from retrieve_context
            max_context_length: Maximum length of formatted context
            
        Returns:
            str: Formatted context string
        """
        if not context_list:
            return "No relevant context found."
        
        context_parts = ["Relevant context from benchmark data:"]
        
        for i, item in enumerate(context_list):
            context_text = f"{i+1}. {item['text']}"
            context_parts.append(context_text)
            
            # Check if we've exceeded the maximum length
            current_length = len("\n".join(context_parts))
            if current_length > max_context_length:
                # Truncate and break
                context_parts = context_parts[:-1]  # Remove the last addition
                context_parts.append(f"... (truncated, showing first {i} items)")
                break
        
        return "\n".join(context_parts)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.
        
        Returns:
            Dict containing collection statistics
        """
        if not self.is_initialized:
            return {'error': 'Pipeline not initialized'}
        
        try:
            return {
                'collection_name': self.collection_name,
                'document_count': len(self.documents),
                'is_initialized': self.is_initialized,
                'pipeline_type': 'Simple RAG (ChromaDB-free)'
            }
        except Exception as e:
            return {'error': f'Failed to get stats: {str(e)}'}
    
    def clear_collection(self) -> bool:
        """
        Clear all data from the collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized:
            logger.error("Pipeline not initialized")
            return False
        
        try:
            self.documents = []
            self.metadata = []
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            return False
    
    def close(self, cleanup_local_data: bool = False) -> bool:
        """
        Close the simple RAG pipeline and clean up resources.
        
        Args:
            cleanup_local_data: Not used in simple implementation (for compatibility)
            
        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        try:
            # Clear data from memory
            self.documents = []
            self.metadata = []
            
            # Mark as not initialized
            self.is_initialized = False
            
            logger.info("Simple RAG pipeline closed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error closing Simple RAG pipeline: {str(e)}")
            return False
    
    def _extract_storage_systems(self):
        """
        Extract unique storage system names and their characteristics from the metadata.
        
        This method looks for storage system information and analyzes their workload patterns
        (read-only, write-only, or mixed) to provide better context for comparisons.
        """
        storage_info = {}  # storage_name -> {'actions': set, 'timeunits': set, 'has_reads': bool, 'has_writes': bool}
        
        for metadata in self.metadata:
            storage_name = metadata.get('storage')
            if not storage_name or storage_name.lower() == 'nan':
                continue
                
            action = metadata.get('action', '').lower()
            timeunit = metadata.get('timeunit', '').lower()
            metric_name = metadata.get('metric_name', '').lower()
            
            # Initialize storage info if not exists
            if storage_name not in storage_info:
                storage_info[storage_name] = {
                    'actions': set(),
                    'timeunits': set(),
                    'has_reads': False,
                    'has_writes': False,
                    'workload_type': 'unknown'
                }
            
            # Update storage characteristics
            if action:
                storage_info[storage_name]['actions'].add(action)
                
                # Determine if it's read or write operation
                if 'read' in action:
                    storage_info[storage_name]['has_reads'] = True
                if 'write' in action:
                    storage_info[storage_name]['has_writes'] = True
            
            if timeunit:
                storage_info[storage_name]['timeunits'].add(timeunit)
            
            # Add storage name to the set
            self.storage_systems.add(storage_name)
        
        # Determine workload types for each storage system
        for storage_name, info in storage_info.items():
            if info['has_reads'] and info['has_writes']:
                info['workload_type'] = 'mixed'
            elif info['has_reads']:
                info['workload_type'] = 'read_only'
            elif info['has_writes']:
                info['workload_type'] = 'write_only'
            
            logger.info(f"Storage system '{storage_name}': "
                       f"actions={info['actions']}, "
                       f"timeunits={info['timeunits']}, "
                       f"workload={info['workload_type']}")
        
        # Store detailed storage info for enhanced context
        self.storage_info = storage_info
        
        # Debug: Print all metadata keys to help identify storage columns
        if self.metadata and not self.storage_systems:
            logger.error("No storage systems found. Available metadata columns:")
            if self.metadata:
                for key in self.metadata[0].keys():
                    logger.info(f"  - {key}")
    
    def _is_comparison_query(self, query: str) -> bool:
        """
        Check if the query is asking for a comparison between storage systems.
        
        Args:
            query: The query string
            
        Returns:
            bool: True if this is a comparison query
        """
        comparison_indicators = [
            'better', 'worse', 'compare', 'comparison', 'versus', 'vs', 'against',
            'which', 'what', 'best', 'worst', 'faster', 'slower', 'higher', 'lower',
            'perform', 'performance', 'recommend', 'choose', 'select'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in comparison_indicators)
    
    def get_storage_systems(self) -> List[str]:
        """
        Get the list of storage systems found in the CSV files.
        
        Returns:
            List[str]: Sorted list of unique storage system names
        """
        return sorted(list(self.storage_systems))
    
    def debug_storage_extraction(self):
        """
        Debug method to print information about storage system extraction.
        """
        print(f"Total metadata entries: {len(self.metadata)}")
        print(f"Found storage systems: {self.get_storage_systems()}")
        
        if self.metadata:
            print(f"Available metadata columns: {list(self.metadata[0].keys())}")
            
            # Show sample storage values if available
            storage_columns = ['storage', 'storage_name', 'system', 'storage_system', 'device']
            for metadata in self.metadata[:3]:  # Check first 3 entries
                for col in metadata:
                    if col.lower() in [sc.lower() for sc in storage_columns]:
                        print(f"Sample {col}: {metadata[col]}")
                        break
    
    def format_context_for_prompt(self, context_list: List[Dict[str, Any]], max_context_length: int = 10000000) -> str:
        """
        Format retrieved context into a string suitable for AI prompts with enhanced storage comparison support.
        
        This enhanced version includes detailed storage system information including
        workload types, actions, and time units for better comparative analysis.
        Optimized to include comprehensive storage system data without premature truncation.
        
        Args:
            context_list: List of context items from retrieve_context
            max_context_length: Maximum length of formatted context (default very high)
            
        Returns:
            str: Formatted context string
        """
        if not context_list:
            context_parts = ["No relevant context found."]
            
            # If we have storage systems but no specific context, mention them with details
            if self.storage_systems and hasattr(self, 'storage_info'):
                context_parts.append("\nAvailable Storage Systems:")
                for storage_name in sorted(self.storage_systems):
                    info = self.storage_info.get(storage_name, {})
                    actions_str = ", ".join(sorted(info.get('actions', []))) if info.get('actions') else "unknown"
                    workload_str = info.get('workload_type', 'unknown')
                    timeunits_str = ", ".join(sorted(info.get('timeunits', []))) if info.get('timeunits') else "unknown"
                    context_parts.append(f"- {storage_name}: workload={workload_str}, actions={actions_str}, timeunits={timeunits_str}")
            
            return "\n".join(context_parts)
        
        context_parts = ["Relevant context from benchmark data:"]
        
        # Group context by storage system for better organization
        storage_contexts = {}
        for item in context_list:
            storage_name = item.get('metadata', {}).get('storage', 'Unknown')
            if storage_name not in storage_contexts:
                storage_contexts[storage_name] = []
            storage_contexts[storage_name].append(item)
        
        # Add storage systems information with detailed characteristics
        if self.storage_systems and hasattr(self, 'storage_info'):
            context_parts.append("\nAvailable Storage Systems:")
            for storage_name in sorted(self.storage_systems):
                info = self.storage_info.get(storage_name, {})
                actions_str = ", ".join(sorted(info.get('actions', []))) if info.get('actions') else "unknown"
                workload_str = info.get('workload_type', 'unknown')
                timeunits_str = ", ".join(sorted(info.get('timeunits', []))) if info.get('timeunits') else "unknown"
                context_parts.append(f"- {storage_name}: workload={workload_str}, actions={actions_str}, timeunits={timeunits_str}")
        
        context_parts.append("\nDetailed Performance Data by Storage System:")
        
        # Add performance data grouped by storage system
        for storage_name in sorted(storage_contexts.keys()):
            context_parts.append(f"\n=== {storage_name.upper()} ===")
            
            storage_items = storage_contexts[storage_name]
            
            # Group by action type (read/write) for better organization
            action_groups = {}
            for item in storage_items:
                action = item.get('metadata', {}).get('action', 'unknown')
                if action not in action_groups:
                    action_groups[action] = []
                action_groups[action].append(item)
            
            for action in sorted(action_groups.keys()):
                context_parts.append(f"\n{action.upper()} Operations:")
                
                for item in action_groups[action]:
                    context_text = item['text']
                    metadata = item.get('metadata', {})
                    
                    # Format with more detailed information
                    formatted_text = f"  • {context_text}"
                    if 'avg' in metadata:
                        formatted_text += f" (avg: {metadata['avg']:.2f})"
                    if 'metric_type' in metadata:
                        formatted_text += f" [{metadata['metric_type']}]"
                    
                    context_parts.append(formatted_text)
                    
                    # Check length less frequently to allow more data
                    if len(context_parts) % 100 == 0:  # Check every 100 items instead of every item
                        current_length = len("\n".join(context_parts))
                        if current_length > max_context_length * 0.9:  # Use 90% of max to be safe
                            context_parts.append(f"\n... (context truncated at {len(context_parts)} items to stay within limits)")
                            return "\n".join(context_parts)
        
        # Final length check
        current_length = len("\n".join(context_parts))
        if current_length > max_context_length:
            # If still too long, provide a summary instead
            summary_parts = ["Relevant context from benchmark data (summary due to length limits):"]
            summary_parts.append(f"Total storage systems: {len(storage_contexts)}")
            for storage_name, items in storage_contexts.items():
                summary_parts.append(f"- {storage_name}: {len(items)} performance metrics")
            return "\n".join(summary_parts)
        
        return "\n".join(context_parts)
