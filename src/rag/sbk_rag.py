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
SBK RAG Pipeline Module

This module provides a RAG (Retrieval-Augmented Generation) pipeline for enhancing
AI analysis with contextual information from CSV data files. It uses ChromaDB
as the vector database to store and retrieve relevant data points.

Key Features:
- CSV data ingestion and chunking
- Vector embeddings using sentence-transformers
- ChromaDB for efficient similarity search
- Context retrieval for AI prompts
- Support for multiple CSV files
"""

import os
import shutil
import pandas as pd
import chromadb
import tempfile
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SbkRAGPipeline:
    """
    RAG Pipeline for SBK AI Analysis
    
    This class handles the complete workflow of ingesting CSV data, creating embeddings,
    storing them in ChromaDB, and retrieving relevant context for AI analysis.
    
    Attributes:
        collection_name: Name of the ChromaDB collection
        embedding_model: Sentence transformer model for embeddings
        chroma_client: ChromaDB client instance
        collection: ChromaDB collection for storing/retrieving embeddings
        is_initialized: Flag indicating if the pipeline is ready
    """
    
    def __init__(self, collection_name: str = "sbk_benchmark_data"):
        """
        Initialize the RAG pipeline.
        
        Args:
            collection_name: Name for the ChromaDB collection
        """
        self.collection_name = collection_name
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize the RAG pipeline with embedding model and ChromaDB.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize the embedding model
            logger.info("Loading sentence transformer model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize ChromaDB client
            logger.info("Initializing ChromaDB client...")
            self.chroma_client = chromadb.Client()
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.chroma_client.create_collection(name=self.collection_name)
                logger.info(f"Created new collection: {self.collection_name}")
            
            self.is_initialized = True
            logger.info("RAG pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            return False
    
    def ingest_csv_files(self, csv_files: List[str]) -> bool:
        """
        Ingest data from multiple CSV files into the RAG pipeline.
        
        Args:
            csv_files: List of CSV file paths to ingest
            
        Returns:
            bool: True if ingestion successful, False otherwise
        """
        if not self.is_initialized:
            logger.error("RAG pipeline not initialized. Call initialize() first.")
            return False
            
        try:
            all_documents = []
            all_metadatas = []
            all_ids = []
            
            for file_path in csv_files:
                if not os.path.exists(file_path):
                    logger.warning(f"CSV file not found: {file_path}")
                    continue
                    
                logger.info(f"Processing CSV file: {file_path}")
                documents, metadatas, ids = self._process_csv_file(file_path)
                
                all_documents.extend(documents)
                all_metadatas.extend(metadatas)
                all_ids.extend(ids)
            
            if not all_documents:
                logger.warning("No valid data found in CSV files")
                return False
            
            # Add to ChromaDB collection
            logger.info(f"Adding {len(all_documents)} documents to ChromaDB...")
            self.collection.add(
                documents=all_documents,
                metadatas=all_metadatas,
                ids=all_ids
            )
            
            logger.info(f"Successfully ingested {len(all_documents)} data points from {len(csv_files)} files")
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
        Convert a CSV row to a text representation for embedding.
        
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
    
    def retrieve_context(self, query: str, n_results: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from the ingested data based on a query.
        
        Args:
            query: Query string to search for
            n_results: Number of results to retrieve
            
        Returns:
            List of dictionaries containing retrieved context
        """
        if not self.is_initialized:
            logger.error("RAG pipeline not initialized. Call initialize() first.")
            return []
        
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            context_list = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    context_item = {
                        'text': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0
                    }
                    context_list.append(context_item)
            
            logger.info(f"Retrieved {len(context_list)} context items for query: {query[:50]}...")
            return context_list
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def format_context_for_prompt(self, context_list: List[Dict[str, Any]], max_context_length: int = 10000000) -> str:
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
        if not self.is_initialized or not self.collection:
            return {'error': 'Pipeline not initialized'}
        
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'is_initialized': self.is_initialized
            }
        except Exception as e:
            return {'error': f'Failed to get stats: {str(e)}'}
    
    def clear_collection(self) -> bool:
        """
        Clear all data from the collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized or not self.collection:
            logger.error("Pipeline not initialized")
            return False
        
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.create_collection(name=self.collection_name)
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            return False

    def _cleanup_local_data(self) -> bool:
        """
        Clean up all local ChromaDB data from disk.

        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        try:

            # Default ChromaDB storage locations
            chroma_paths = [
                os.path.join(os.getcwd(), "chroma_db"),
                os.path.join(os.path.expanduser("~"), ".chroma"),
                os.path.join(tempfile.gettempdir(), "chroma"),
            ]

            # Also check for any persistent ChromaDB directories
            for path in chroma_paths:
                if os.path.exists(path):
                    logger.info(f"Removing ChromaDB data from: {path}")
                    try:
                        shutil.rmtree(path)
                        logger.info(f"Successfully removed: {path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {path}: {str(e)}")

            # Clear any remaining in-memory collections
            if self.chroma_client:
                try:
                    # Try to delete the collection if it still exists
                    self.chroma_client.delete_collection(name=self.collection_name)
                except Exception:
                    # Collection might not exist or client might be closed
                    pass

            logger.info("Local ChromaDB data cleanup completed")
            return True

        except Exception as e:
            logger.error(f"Error cleaning up local data: {str(e)}")
            return False

    def close(self, cleanup_local_data: bool = False) -> bool:
        """
        Close the RAG pipeline and clean up resources.
        
        Args:
            cleanup_local_data: If True, removes all local ChromaDB data from disk
            
        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        try:
            # Clear collection from memory
            if self.collection:
                self.collection = None
            
            # Close ChromaDB client
            if self.chroma_client:
                self.chroma_client = None
            
            # Clean up embedding model
            if self.embedding_model:
                self.embedding_model = None
            
            # Mark as not initialized
            self.is_initialized = False
            
            # Optionally clean up local data
            if cleanup_local_data:
                self._cleanup_local_data()
            
            logger.info("RAG pipeline closed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error closing RAG pipeline: {str(e)}")
            return False
    
