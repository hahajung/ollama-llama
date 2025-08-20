#!/usr/bin/env python3
"""
Fix database path configuration for the enhanced chatbot
"""

import os
import shutil
from pathlib import Path

def fix_database_paths():
    """
    Fix the database path issues for the enhanced chatbot
    """
    print("="*60)
    print("DATABASE PATH CONFIGURATION FIX")
    print("="*60)
    
    # Check existing databases
    existing_dbs = [
        "optimized_technical_db",
        "technical_optimized_chroma_db",
        "technical_chroma_db",
        "chroma_db"
    ]
    
    print("\nChecking for existing databases:")
    found_db = None
    for db in existing_dbs:
        if Path(db).exists():
            print(f"  [FOUND] {db}")
            if not found_db:
                found_db = db
        else:
            print(f"  [MISSING] {db}")
    
    if not found_db:
        print("\nNo database found! Please run: python fix_and_optimize.py")
        return False
    
    print(f"\nUsing database: {found_db}")
    
    # Find and update the chatbot configuration
    files_to_check = [
        "web_searching_chatbot.py",
        "app/core/improved_chatbot.py",
        "app/core/enhanced_retriever.py",
        "complete_technical_pipeline.py",
        "app/enhanced_ui.py"
    ]
    
    print("\nUpdating configuration files:")
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"  Checking {file_path}...")
            
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check what database path it's using
            if "technical_optimized_chroma_db" in content:
                print(f"    -> File expects: technical_optimized_chroma_db")
                # Create a copy or symlink
                if not Path("technical_optimized_chroma_db").exists():
                    print(f"    -> Creating copy from {found_db}")
                    shutil.copytree(found_db, "technical_optimized_chroma_db")
                    print(f"    -> Created technical_optimized_chroma_db")
            
            elif "chroma_db" in content and "technical" not in content:
                print(f"    -> File expects: chroma_db")
                if not Path("chroma_db").exists():
                    print(f"    -> Creating copy from {found_db}")
                    shutil.copytree(found_db, "chroma_db")
                    print(f"    -> Created chroma_db")
    
    print("\n" + "="*60)
    print("Configuration complete!")
    print("="*60)
    return True

def create_config_file():
    """
    Create a configuration file for consistent database paths
    """
    config = """# Database Configuration
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
    \"\"\"Get the first available database path\"\"\"
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
"""
    
    with open("db_config.py", "w") as f:
        f.write(config)
    
    print("Created db_config.py configuration file")

def update_chatbot_to_use_config():
    """
    Create a wrapper for the chatbot that uses the correct database
    """
    wrapper_code = """#!/usr/bin/env python3
\"\"\"
Enhanced chatbot wrapper with correct database configuration
\"\"\"

import os
import sys
from pathlib import Path

# Set the correct database path
os.environ['CHROMA_DB_PATH'] = './optimized_technical_db'

# Import and run the original chatbot
try:
    # Try to import the enhanced chatbot
    from app.core.improved_chatbot import EnhancedTechnicalChatbot
    
    print("Starting Enhanced Technical Chatbot...")
    print(f"Using database: {os.environ['CHROMA_DB_PATH']}")
    
    # Initialize with correct path
    chatbot = EnhancedTechnicalChatbot(
        qa_db_path='./optimized_technical_db',
        search_db_path='./optimized_technical_db'
    )
    
    # Run interactive mode
    print("\\nChatbot ready! Type 'exit' to quit.\\n")
    
    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            
            response = chatbot.chat(query)
            print(f"\\nAssistant: {response}\\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            
except ImportError:
    # Fallback to web searching chatbot
    print("Enhanced chatbot not found, trying web_searching_chatbot...")
    
    # Import the optimized AI system instead
    from fix_and_optimize import OptimizedAISystem
    
    ai_system = OptimizedAISystem(
        db_path="./optimized_technical_db",
        cache_path="./response_cache",
        use_cloud_api=False
    )
    
    print("\\nOptimized AI System ready! Type 'exit' to quit.\\n")
    
    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            
            result = ai_system.query_with_tiered_models(query)
            print(f"\\nAssistant: {result['response']}\\n")
            print(f"[Model: {result['model_used']} | Time: {result['response_time']:.2f}s]\\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

print("\\nGoodbye!")
"""
    
    with open("run_chatbot.py", "w") as f:
        f.write(wrapper_code)
    
    print("Created run_chatbot.py wrapper script")

def main():
    """Main function"""
    print("Fixing database path configuration...")
    
    # Step 1: Fix paths
    if fix_database_paths():
        print("\nDatabase paths fixed successfully!")
    
    # Step 2: Create config file
    create_config_file()
    
    # Step 3: Create wrapper
    update_chatbot_to_use_config()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nYou can now run the chatbot with:")
    print("  python run_chatbot.py")
    print("\nOr if you have the enhanced UI:")
    print("  python app/enhanced_ui.py")
    print("="*60)

if __name__ == "__main__":
    main()