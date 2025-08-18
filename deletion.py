#!/usr/bin/env python3
"""
Cleanup Script for Itential RAG Project
Safely removes old/unnecessary files and folders.
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """Clean up old and unnecessary files."""
    
    print("🧹 Itential RAG Project Cleanup")
    print("=" * 40)
    
    # Files to delete
    files_to_delete = [
        "scraper_diagnostic_tool.py",
        "database_inspector.py", 
        "setup_enhanced.py",
        "migration_script.py",
        "run_enhanced_pipeline.py",
        "rag_diagnosis.py",
        "retrieval_diagnostics.py",
        "qa_preprocessing_pipeline.py",
        "embedder_only.py",
        "create_qa_vector_database.py",
        "test_embeddings.py",
        "create_chroma_db.py",
        "enhanced_itential_docs.jsonl",
        "itential_qa_pairs.jsonl",
        "data.txt",
        "thumbnail_small.png"
    ]
    
    # Folders to delete
    folders_to_delete = [
        "enhanced_pipeline",
        "logs", 
        "scripts",
        "__pycache__"
    ]
    
    # Optional folders (ask before deleting)
    optional_folders = [
        "backups"
    ]
    
    deleted_files = 0
    deleted_folders = 0
    
    # Delete files
    print("\n📄 Cleaning up files...")
    for file_name in files_to_delete:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"  ✅ Deleted: {file_name}")
                deleted_files += 1
            except Exception as e:
                print(f"  ❌ Failed to delete {file_name}: {e}")
        else:
            print(f"  ⏭️  Skipped: {file_name} (not found)")
    
    # Delete folders
    print("\n📁 Cleaning up folders...")
    for folder_name in folders_to_delete:
        folder_path = Path(folder_name)
        if folder_path.exists() and folder_path.is_dir():
            try:
                shutil.rmtree(folder_path)
                print(f"  ✅ Deleted folder: {folder_name}")
                deleted_folders += 1
            except Exception as e:
                print(f"  ❌ Failed to delete folder {folder_name}: {e}")
        else:
            print(f"  ⏭️  Skipped: {folder_name} (not found)")
    
    # Handle optional folders
    print("\n📂 Optional folders...")
    for folder_name in optional_folders:
        folder_path = Path(folder_name)
        if folder_path.exists() and folder_path.is_dir():
            # Check if folder has content
            try:
                contents = list(folder_path.iterdir())
                if contents:
                    print(f"  📋 {folder_name} contains {len(contents)} items:")
                    for item in contents[:5]:  # Show first 5 items
                        print(f"    - {item.name}")
                    if len(contents) > 5:
                        print(f"    ... and {len(contents) - 5} more")
                    
                    response = input(f"  ❓ Delete {folder_name}? (y/N): ").strip().lower()
                    if response in ['y', 'yes']:
                        shutil.rmtree(folder_path)
                        print(f"  ✅ Deleted folder: {folder_name}")
                        deleted_folders += 1
                    else:
                        print(f"  ⏭️  Kept: {folder_name}")
                else:
                    # Empty folder, delete it
                    folder_path.rmdir()
                    print(f"  ✅ Deleted empty folder: {folder_name}")
                    deleted_folders += 1
            except Exception as e:
                print(f"  ❌ Error checking {folder_name}: {e}")
        else:
            print(f"  ⏭️  Skipped: {folder_name} (not found)")
    
    # Clean up Python cache files in app folder
    print("\n🐍 Cleaning Python cache...")
    app_pycache = Path("app") / "__pycache__"
    core_pycache = Path("app") / "core" / "__pycache__"
    
    for cache_dir in [app_pycache, core_pycache]:
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                print(f"  ✅ Deleted: {cache_dir}")
                deleted_folders += 1
            except Exception as e:
                print(f"  ❌ Failed to delete {cache_dir}: {e}")
    
    # Summary
    print(f"\n📊 Cleanup Summary:")
    print(f"  🗑️  Files deleted: {deleted_files}")
    print(f"  📁 Folders deleted: {deleted_folders}")
    
    # Show remaining important files
    print(f"\n📋 Important files kept:")
    important_files = [
        "app/enhanced_ui.py",
        "app/core/improved_chatbot.py", 
        "comprehensive_technical_scraper.py",
        "complete_technical_pipeline.py",
        "technical_embedder.py",
        "comprehensive_itential_docs.jsonl",
        "requirements.txt",
        "compatibility_imports.py"
    ]
    
    for file_name in important_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} (missing - this might be a problem)")
    
    print(f"\n🎉 Cleanup complete!")
    print(f"💡 Your project is now cleaner and ready for the technical pipeline.")

def show_what_will_be_deleted():
    """Show what will be deleted without actually deleting."""
    print("🔍 Preview: Files and folders that will be deleted")
    print("=" * 50)
    
    files_to_delete = [
        "scraper_diagnostic_tool.py",
        "database_inspector.py", 
        "setup_enhanced.py",
        "migration_script.py",
        "run_enhanced_pipeline.py",
        "rag_diagnosis.py",
        "retrieval_diagnostics.py",
        "qa_preprocessing_pipeline.py",
        "embedder_only.py",
        "create_qa_vector_database.py",
        "test_embeddings.py",
        "create_chroma_db.py",
        "enhanced_itential_docs.jsonl",
        "itential_qa_pairs.jsonl",
        "data.txt",
        "thumbnail_small.png"
    ]
    
    folders_to_delete = [
        "enhanced_pipeline",
        "logs", 
        "scripts",
        "__pycache__"
    ]
    
    print("\n📄 Files to delete:")
    for file_name in files_to_delete:
        if Path(file_name).exists():
            print(f"  🗑️  {file_name}")
        else:
            print(f"  ⏭️  {file_name} (not found)")
    
    print("\n📁 Folders to delete:")
    for folder_name in folders_to_delete:
        if Path(folder_name).exists():
            print(f"  🗑️  {folder_name}/")
        else:
            print(f"  ⏭️  {folder_name}/ (not found)")
    
    print("\n📂 Folders to ask about:")
    print("  ❓ backups/ (will ask before deleting)")
    
    print(f"\n📋 Files that will be KEPT:")
    keep_files = [
        "app/enhanced_ui.py",
        "app/core/improved_chatbot.py",
        "comprehensive_technical_scraper.py", 
        "complete_technical_pipeline.py",
        "technical_embedder.py",
        "comprehensive_itential_docs.jsonl",
        "requirements.txt",
        ".env",
        "compatibility_imports.py"
    ]
    
    for file_name in keep_files:
        print(f"  ✅ {file_name}")

if __name__ == "__main__":
    print("Itential RAG Project Cleanup Tool")
    print("=" * 40)
    print("This script will clean up old/unnecessary files from your project.")
    print()
    
    choice = input("What would you like to do?\n1. Preview what will be deleted\n2. Run cleanup\n3. Exit\nChoice (1/2/3): ").strip()
    
    if choice == "1":
        show_what_will_be_deleted()
    elif choice == "2":
        print("\n⚠️  WARNING: This will permanently delete files!")
        confirm = input("Are you sure you want to proceed? (y/N): ").strip().lower()
        if confirm in ['y', 'yes']:
            cleanup_project()
        else:
            print("❌ Cleanup cancelled.")
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Exiting.")