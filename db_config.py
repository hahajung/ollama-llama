# Database Configuration
# This file defines the paths for the vector database

import os
from pathlib import Path

# Primary database path (created by fix_and_optimize.py)
PRIMARY_DB_PATH = "./optimized_technical_db"

# Alternative paths for compatibility
ALTERNATIVE_PATHS = [
    "./technical_optimized_chroma_db",
    "./technical_chroma_db",
    "./chroma_db"
]

def get_database_path():
    """Get the first available database path"""
    # Check primary path first
    if Path(PRIMARY_DB_PATH).exists():
        return PRIMARY_DB_PATH
    
    # Check alternatives
    for path in ALTERNATIVE_PATHS:
        if Path(path).exists():
            return path
    
    # Return primary as default
    return PRIMARY_DB_PATH

# Export the path
DB_PATH = get_database_path()
