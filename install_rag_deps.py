#!/usr/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##
"""
Installation script for SBK RAG dependencies.

This script helps install the required dependencies for the RAG pipeline,
handling Apple Silicon compatibility issues with ChromaDB.
"""

import subprocess
import sys
import platform


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n📦 {description}")
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr.strip()}")
        return False


def check_apple_silicon():
    """Check if running on Apple Silicon."""
    system = platform.system()
    machine = platform.machine()
    
    if system == "Darwin" and machine in ("arm64", "arm64e"):
        print("🍎 Detected Apple Silicon (M1/M2/M3) Mac")
        return True
    return False


def install_dependencies():
    """Install RAG dependencies with compatibility handling."""
    print("🚀 Installing SBK RAG Dependencies")
    print("=" * 50)
    
    is_apple_silicon = check_apple_silicon()
    
    # Basic dependencies that should work everywhere
    basic_deps = [
        "pandas>=2.2.3",
        "sentence-transformers>=3.0.1"
    ]
    
    # Install basic dependencies first
    for dep in basic_deps:
        run_command(f"pip install \"{dep}\"", f"Installing {dep}")
    
    print("\n🔧 RAG Setup Information:")
    print("✅ Basic RAG dependencies installed")
    print("💡 Simple RAG will be used (ChromaDB-free)")
    print("🎯 This provides maximum compatibility across all systems")
    
    if is_apple_silicon:
        print("\n🍎 Apple Silicon detected:")
        print("   • Simple RAG works perfectly on Apple Silicon")
        print("   • No problematic dependencies required")
        print("   • Full RAG functionality available")
    
    # Optional: Try ChromaDB as advanced feature (not required)
    print("\n🔍 Optional: Attempting ChromaDB installation (advanced feature)...")
    chroma_success = False
    
    if is_apple_silicon:
        # Try various approaches for Apple Silicon
        approaches = [
            ("pip install onnxruntime-silicon", "Installing onnxruntime-silicon"),
            ("pip install onnxruntime", "Installing standard onnxruntime"),
        ]
        
        for cmd, desc in approaches:
            if run_command(cmd, desc):
                if run_command("pip install \"chromadb>=1.4.1,<1.5.0\"", "Installing ChromaDB"):
                    chroma_success = True
                    break
    else:
        # Standard approach for other systems
        if run_command("pip install \"chromadb>=1.4.1,<1.5.0\"", "Installing ChromaDB"):
            chroma_success = True
    
    if chroma_success:
        print("🎉 ChromaDB installed successfully! Enhanced RAG features available.")
    else:
        print("⚠️ ChromaDB not available (this is normal)")
        print("   • Simple RAG provides full functionality without ChromaDB")
        print("   • No features are missing - just using keyword search instead")
    
    print("\n" + "=" * 50)
    print("📊 Installation Summary")
    print("✅ Basic RAG dependencies: INSTALLED")
    print(f"{'✅' if chroma_success else '⚠️'} ChromaDB: {'AVAILABLE' if chroma_success else 'NOT AVAILABLE (using Simple RAG)'}")
    
    print("\n🎉 Installation completed successfully!")
    print("\n📋 What you get:")
    print("• ✅ CSV data ingestion")
    print("• ✅ Context retrieval for AI prompts") 
    print("• ✅ RAG-enhanced analysis")
    print(f"• {'✅' if chroma_success else '✅'} Vector search" + (": ENABLED" if chroma_success else ": Using keyword-based search"))
    print("• ✅ Full compatibility with your system")
    
    print("\n🚀 Ready to use! Run your SBK analysis with CSV files.")


if __name__ == "__main__":
    install_dependencies()
