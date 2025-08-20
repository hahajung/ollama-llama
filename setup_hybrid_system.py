#!/usr/bin/env python3
"""
Hybrid System Setup Script - No Emojis Version
Integrates the hybrid approach with your existing setup.
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess

def check_prerequisites():
    """Check if all prerequisites are met."""
    print("Checking Prerequisites...")
    
    # Check if scraper output exists
    scraper_output = Path("complete_technical_docs.jsonl")
    if not scraper_output.exists():
        print("ERROR: complete_technical_docs.jsonl not found")
        print("   Run: python comprehensive_technical_scraper.py")
        return False
    else:
        print("OK: Scraper output found")
    
    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("OK: Ollama is running")
        else:
            print("ERROR: Ollama not responding correctly")
            return False
    except Exception:
        print("ERROR: Ollama not accessible")
        print("   Start with: ollama serve")
        return False
    
    # Check required models
    try:
        models_response = requests.get("http://localhost:11434/api/tags").json()
        available_models = [model['name'] for model in models_response.get('models', [])]
        
        required_models = ['nomic-embed-text:latest', 'mistral:7b']
        for model in required_models:
            if model in available_models:
                print(f"OK: Model available: {model}")
            else:
                print(f"ERROR: Model missing: {model}")
                print(f"   Pull with: ollama pull {model.replace(':latest', '')}")
                return False
    except Exception as e:
        print(f"WARNING: Could not check models: {e}")
    
    return True

def backup_existing_files():
    """Backup existing files before updating."""
    print("\nBacking up existing files...")
    
    files_to_backup = [
        "robust_technical_embedder.py",
        "app/enhanced_ui.py", 
        "app/core/improved_chatbot.py"
    ]
    
    backup_dir = Path("backup_" + str(int(__import__('time').time())))
    backup_dir.mkdir(exist_ok=True)
    
    for file_path in files_to_backup:
        file_path = Path(file_path)
        if file_path.exists():
            backup_path = backup_dir / file_path.name
            shutil.copy2(file_path, backup_path)
            print(f"OK: Backed up: {file_path} -> {backup_path}")
        else:
            print(f"INFO: File not found: {file_path}")
    
    print(f"Backup directory: {backup_dir}")
    return backup_dir

def install_updated_files():
    """Provide instructions for installing updated files."""
    print("\nInstallation Instructions:")
    print("=" * 50)
    
    print("1. Replace these files with the updated versions:")
    print("   robust_technical_embedder.py (updated with hybrid storage)")
    print("   app/enhanced_ui.py (updated with hybrid router)")
    print("   app/core/improved_chatbot.py (updated with hybrid integration)")
    
    print("\n2. The updated files include:")
    print("   Hybrid storage system (Q&A + Structured + Complete Pages)")
    print("   Intelligent query routing")
    print("   Enhanced accuracy for version-specific queries")
    print("   Maintains your existing performance optimizations")

def run_hybrid_embedder():
    """Run the updated embedder to create hybrid storage."""
    print("\nCreating Hybrid Storage System...")
    
    try:
        # Run the updated embedder
        result = subprocess.run([
            sys.executable, "robust_technical_embedder.py"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("SUCCESS: Hybrid storage system created successfully!")
            
            # Check which databases were created
            db_paths = [
                "./technical_optimized_chroma_db",
                "./technical_structured_db", 
                "./technical_complete_pages_db"
            ]
            
            for db_path in db_paths:
                if Path(db_path).exists():
                    print(f"OK: Created: {db_path}")
                else:
                    print(f"INFO: Optional: {db_path} (not created)")
            
        else:
            print("ERROR: Error creating hybrid storage system:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("TIMEOUT: Embedder timed out (this can happen with large datasets)")
        print("   Check if databases were created manually")
    except Exception as e:
        print(f"ERROR: Error running embedder: {e}")
        return False
    
    return True

def test_hybrid_system():
    """Test the hybrid system."""
    print("\nTesting Hybrid System...")
    
    try:
        # Import and test the updated chatbot
        sys.path.append('app/core')
        from improved_chatbot import HybridImprovedChatbot
        
        chatbot = HybridImprovedChatbot()
        
        # Test critical query
        test_query = "What version of RabbitMQ is required for IAP 2023.2?"
        print(f"Testing: {test_query}")
        
        result = chatbot.generate_hybrid_response(test_query)
        
        print(f"Layer used: {result['layer_used']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Response time: {result['response_time']:.2f}s")
        print(f"Answer preview: {result['answer'][:100]}...")
        
        # Check if it correctly identifies RabbitMQ as NOT_REQUIRED
        answer_lower = result['answer'].lower()
        if 'not required' in answer_lower or 'not needed' in answer_lower or 'n/a' in answer_lower:
            print("SUCCESS: Identified RabbitMQ as not required for IAP 2023.2")
            return True
        else:
            print("WARNING: Answer may not be accurate - review the response")
            return False
            
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        return False

def show_usage_instructions():
    """Show how to use the hybrid system."""
    print("\nUsage Instructions:")
    print("=" * 50)
    
    print("Start your enhanced chatbot:")
    print("   streamlit run app/enhanced_ui.py")
    
    print("\nTest these queries for accuracy:")
    print("   'What version of RabbitMQ is required for IAP 2023.2?'")
    print("     Expected: 'RabbitMQ is NOT required for IAP 2023.2'")
    print("   'What Node.js version is required for IAP 2023.2?'")  
    print("   'What are the system requirements for IAP 2023.2?'")
    print("   'Show me the dependency table for IAP versions'")
    
    print("\nThe hybrid system provides:")
    print("   Precise Q&A for version-specific queries")
    print("   Intact tables for dependency matrices")
    print("   Complete pages for procedural questions")
    print("   Intelligent routing for optimal accuracy")
    
    print("\nMonitoring:")
    print("   Check 'Show Debug Info' in the UI to see routing decisions")
    print("   Monitor which storage layer is used for each query")
    print("   Confidence scores indicate answer reliability")

def main():
    """Main setup function."""
    print("HYBRID ACCURACY SYSTEM SETUP")
    print("=" * 60)
    print("This will enhance your existing setup with hybrid storage")
    print("for Claude-level accuracy while keeping your optimizations.")
    print("=" * 60)
    
    # Step 1: Prerequisites
    if not check_prerequisites():
        print("\nERROR: Prerequisites not met. Please fix the issues above.")
        sys.exit(1)
    
    # Step 2: Backup
    backup_dir = backup_existing_files()
    
    # Step 3: Installation instructions
    install_updated_files()
    
    print(f"\nPAUSE: Please update your files now.")
    print("Copy the updated file contents from the artifacts above.")
    input("Press Enter when you've updated the files...")
    
    # Step 4: Create hybrid storage
    if not run_hybrid_embedder():
        print("\nERROR: Failed to create hybrid storage system.")
        print(f"ROLLBACK: Your original files are backed up in: {backup_dir}")
        sys.exit(1)
    
    # Step 5: Test system
    if test_hybrid_system():
        print("\nSUCCESS: HYBRID SYSTEM SETUP COMPLETE!")
        print("=" * 60)
        show_usage_instructions()
    else:
        print("\nWARNING: Setup completed but test failed.")
        print("The system may still work - check manually.")
        show_usage_instructions()

if __name__ == "__main__":
    main()