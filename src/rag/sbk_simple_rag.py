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
- CSV data ingestion and chunking
- Simple keyword-based search (no embeddings required)
- Fallback implementation for systems without ChromaDB
- Context retrieval for AI prompts
- Support for multiple CSV files
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SbkSimpleRAGPipeline:
    """
    Simple RAG Pipeline for SBK AI Analysis (ChromaDB-free)
    
    This class provides a basic RAG implementation that doesn't require
    external vector databases. It uses keyword-based search and simple
    text matching for retrieving relevant context.
    
    Attributes:
        documents: List of processed documents from CSV files
        metadata: List of metadata corresponding to each document
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
    
    def ingest_csv_files(self, csv_files: List[str]) -> bool:
        """
        Ingest data from multiple CSV files into the simple RAG pipeline.
        
        Args:
            csv_files: List of CSV file paths to ingest
            
        Returns:
            bool: True if ingestion successful, False otherwise
        """
        if not self.is_initialized:
            logger.error("Simple RAG pipeline not initialized. Call initialize() first.")
            return False
            
        try:
            all_documents = []
            all_metadatas = []
            
            for file_path in csv_files:
                if not os.path.exists(file_path):
                    logger.warning(f"CSV file not found: {file_path}")
                    continue
                    
                logger.info(f"Processing CSV file: {file_path}")
                documents, metadatas, ids = self._process_csv_file(file_path)
                
                all_documents.extend(documents)
                all_metadatas.extend(metadatas)
            
            if not all_documents:
                logger.warning("No valid data found in CSV files")
                return False
            
            # Store in memory
            self.documents = all_documents
            self.metadata = all_metadatas
            
            # Extract storage system names from metadata
            self._extract_storage_systems()
            
            logger.info(f"Successfully ingested {len(all_documents)} data points from {len(csv_files)} files")
            if self.storage_systems:
                logger.info(f"Detected storage systems: {', '.join(sorted(self.storage_systems))}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest CSV files: {str(e)}")
            return False
    
    def _process_csv_file(self, file_path: str) -> Tuple[List[str], List[Dict], List[str]]:
        """
        Process a single CSV file and extract documents, metadata, and IDs.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Tuple of (documents, metadatas, ids)
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)

            documents = []
            metadatas = []
            ids = []
            
            file_name = os.path.basename(file_path)
            
            # Process each row as a separate document
            for idx, row in df.iterrows():
                # Create a text representation of the row
                row_text = self._row_to_text(row, file_name, idx)
                
                # Create metadata
                metadata = {
                    'file_name': file_name,
                    'row_index': idx,
                    'file_path': file_path
                }
                
                # Add column values to metadata
                for col in df.columns:
                    if pd.notna(row[col]):
                        metadata[col] = str(row[col])
                
                # Create unique ID
                doc_id = f"{file_name}_{idx}"
                
                documents.append(row_text)
                metadatas.append(metadata)
                ids.append(doc_id)
            
            return documents, metadatas, ids
            
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {str(e)}")
            return [], [], []
    
    def _row_to_text(self, row: pd.Series, file_name: str, row_idx: int) -> str:
        """
        Convert a CSV row to a text representation for searching.
        
        Args:
            row: Pandas Series representing a row
            file_name: Name of the source file
            row_idx: Row index
            
        Returns:
            str: Text representation of the row
        """
        text_parts = [f"Data from {file_name}, row {row_idx}:"]
        
        for col, value in row.items():
            if pd.notna(value):
                text_parts.append(f"{col}: {value}")
        
        return " | ".join(text_parts)
    
    def retrieve_context(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from the ingested data using keyword matching.
        
        Args:
            query: Query string to search for
            n_results: Number of results to retrieve
            
        Returns:
            List of dictionaries containing retrieved context
        """
        if not self.is_initialized:
            logger.error("Simple RAG pipeline not initialized. Call initialize() first.")
            return []
        
        try:
            # Extract keywords from query
            query_keywords = self._extract_keywords(query.lower())
            
            # Score documents based on keyword matches
            scored_docs = []
            for i, doc in enumerate(self.documents):
                doc_lower = doc.lower()
                score = 0
                
                # Count keyword matches
                for keyword in query_keywords:
                    if keyword in doc_lower:
                        score += doc_lower.count(keyword)
                
                if score > 0:
                    scored_docs.append((score, i, doc, self.metadata[i]))
            
            # Sort by score (descending) and take top n_results
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            top_docs = scored_docs[:n_results]
            
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
        Extract meaningful keywords from a query.
        
        Args:
            query: The query string
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction - split on common delimiters and filter
        keywords = []
        
        # Split on common delimiters
        for word in re.split(r'[\s,;:!?]+', query):
            if len(word) >= 3:  # Only keep words with 3+ characters
                keywords.append(word)
        
        # Add common storage/performance terms
        performance_terms = ['throughput', 'latency', 'performance', 'mb/s', 'iops', 'storage', 'read', 'write']
        for term in performance_terms:
            if term in query:
                keywords.append(term)
        
        # Check for comparison queries and add storage system names
        if self._is_comparison_query(query):
            keywords.extend(self.storage_systems)
        
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
        Extract unique storage system names from the metadata.
        
        This method looks for common storage system identifier columns
        like 'STORAGE', 'storage_name', 'system', etc. in the metadata.
        """
        storage_columns = ['storage', 'storage_name', 'system', 'storage_system', 'device']
        
        for metadata in self.metadata:
            for col in metadata:
                # Check if this column matches any storage column (case-insensitive)
                if col.lower() in [sc.lower() for sc in storage_columns]:
                    storage_name = str(metadata[col]).strip()
                    if storage_name and storage_name.lower() != 'nan':
                        self.storage_systems.add(storage_name)
        
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
    
    def format_context_for_prompt(self, context_list: List[Dict[str, Any]], max_context_length: int = 2000) -> str:
        """
        Format retrieved context into a string suitable for AI prompts.
        
        This enhanced version includes storage system information when relevant.
        
        Args:
            context_list: List of context items from retrieve_context
            max_context_length: Maximum length of formatted context
            
        Returns:
            str: Formatted context string
        """
        if not context_list:
            context_parts = ["No relevant context found."]
            
            # If we have storage systems but no specific context, mention them
            if self.storage_systems:
                context_parts.append(f"Available storage systems in the dataset: {', '.join(self.get_storage_systems())}")
            
            return "\n".join(context_parts)
        
        context_parts = ["Relevant context from benchmark data:"]
        
        # Add storage systems information if this might be a comparison query
        if self.storage_systems:
            context_parts.append(f"Available storage systems: {', '.join(self.get_storage_systems())}")
        
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
