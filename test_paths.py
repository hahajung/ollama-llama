#!/usr/bin/env python3
"""
Simple script to test where the database actually is vs where Streamlit is looking
"""

import os
import sys
from pathlib import Path

def test_database_locations():
    """Test all possible database locations."""
    
    print("=== Path Testing for Database Location ===")
    
    # Show current environment
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {__file__}")
    print(f"Python executable: {sys.executable}")
    print()
    
    # Test various path combinations
    test_paths = [
        "./technical_optimized_chroma_db",
        "../technical_optimized_chroma_db", 
        "technical_optimized_chroma_db",
        "./app/../technical_optimized_chroma_db",
        os.path.join(os.getcwd(), "technical_optimized_chroma_db"),
        os.path.join(os.path.dirname(os.getcwd()), "technical_optimized_chroma_db"),
    ]
    
    print("Testing database paths:")
    found_paths = []
    
    for test_path in test_paths:
        exists = os.path.exists(test_path)
        is_dir = os.path.isdir(test_path) if exists else False
        
        status = "✅ EXISTS (DIR)" if exists and is_dir else "❌ MISSING"
        print(f"  {test_path} -> {status}")
        
        if exists and is_dir:
            found_paths.append(test_path)
            
            # Check if it has Chroma files
            chroma_files = []
            try:
                for item in os.listdir(test_path):
                    if item.endswith('.sqlite3') or 'data_level0.bin' in item:
                        chroma_files.append(item)
                print(f"    Chroma files: {len(chroma_files)} found")
            except:
                print(f"    Cannot read directory contents")
    
    print()
    print(f"Found {len(found_paths)} working database paths:")
    for path in found_paths:
        print(f"  ✅ {path}")
    
    # Show directory contents
    print("\n=== Directory Contents ===")
    
    cwd = os.getcwd()
    print(f"\nCurrent directory ({cwd}):")
    try:
        for item in os.listdir(cwd):
            item_path = os.path.join(cwd, item)
            item_type = "DIR" if os.path.isdir(item_path) else "FILE"
            marker = " 🎯" if "chroma" in item.lower() else ""
            print(f"  {item} ({item_type}){marker}")
    except Exception as e:
        print(f"  Error reading directory: {e}")
    
    # Check parent directory
    parent_dir = os.path.dirname(cwd)
    print(f"\nParent directory ({parent_dir}):")
    try:
        for item in os.listdir(parent_dir):
            item_path = os.path.join(parent_dir, item)
            item_type = "DIR" if os.path.isdir(item_path) else "FILE"
            marker = " 🎯" if "chroma" in item.lower() else ""
            print(f"  {item} ({item_type}){marker}")
    except Exception as e:
        print(f"  Error reading directory: {e}")
    
    # Test Streamlit-specific paths
    print("\n=== Streamlit Environment Test ===")
    
    # Simulate what Streamlit might see
    try:
        # This mimics what our Streamlit app does
        app_file = Path("app/enhanced_ui.py")
        if app_file.exists():
            app_dir = app_file.parent
            root_dir = app_dir.parent
            
            print(f"App file: {app_file}")
            print(f"App directory: {app_dir}")
            print(f"Root directory: {root_dir}")
            
            db_path = root_dir / "technical_optimized_chroma_db"
            print(f"Expected database path: {db_path}")
            print(f"Database exists: {db_path.exists()}")
            
        else:
            print("app/enhanced_ui.py not found - run from wrong directory?")
            
    except Exception as e:
        print(f"Streamlit path test failed: {e}")
    
    print("\n=== Recommendations ===")
    
    if found_paths:
        print("✅ Database found! Use this path in your Streamlit app:")
        best_path = found_paths[0]
        print(f"   {best_path}")
        
        print("\n💡 Quick fix for enhanced_ui.py:")
        print(f"   Replace the database path with: '{best_path}'")
        
    else:
        print("❌ No database found!")
        print("   1. Run: python complete_technical_pipeline.py")
        print("   2. Check if pipeline completed successfully")
        print("   3. Look for 'technical_optimized_chroma_db' directory")

if __name__ == "__main__":
    test_database_locations()