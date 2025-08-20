#!/usr/bin/env python3
"""
Enhanced chatbot wrapper with correct database configuration
"""

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
    print("\nChatbot ready! Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            
            response = chatbot.chat(query)
            print(f"\nAssistant: {response}\n")
            
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
    
    print("\nOptimized AI System ready! Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            
            result = ai_system.query_with_tiered_models(query)
            print(f"\nAssistant: {result['response']}\n")
            print(f"[Model: {result['model_used']} | Time: {result['response_time']:.2f}s]\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

print("\nGoodbye!")
