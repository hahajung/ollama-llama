#!/usr/bin/env python3
"""
Windows-compatible cleanup script
"""
import os
import shutil
from pathlib import Path
import subprocess
import sys

def cleanup_windows():
    print("=" * 60)
    print("WINDOWS AI SYSTEM CLEANUP")
    print("=" * 60)
    
    # Clean old databases
    paths_to_clean = [
        "./technical_optimized_chroma_db",
        "./technical_chroma_db", 
        "./chroma_db",
        "./vector_db",
        "./cache",
        "./old_cache",
        "./__pycache__"
    ]
    
    for path_str in paths_to_clean:
        path = Path(path_str)
        if path.exists():
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                print(f"✓ Removed: {path}")
            except Exception as e:
                print(f"✗ Could not remove {path}: {e}")
    
    # Create new directories
    Path("./optimized_technical_db").mkdir(exist_ok=True)
    Path("./response_cache").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)
    print("✓ Created directory structure")
    
    # Check Ollama installation
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("✓ Ollama is installed")
        else:
            print("✗ Ollama not found. Please install from: https://ollama.ai/download/windows")
    except:
        print("✗ Ollama not found. Please install from: https://ollama.ai/download/windows")
    
    print("\n✓ Cleanup complete!")

if __name__ == "__main__":
    cleanup_windows()