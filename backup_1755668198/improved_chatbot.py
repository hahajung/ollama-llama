#!/usr/bin/env python3
"""
Fixed and improved chatbot - no hardcoding, just smart retrieval
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fix_existing_system import FixedChatbot

class EnhancedTechnicalChatbot:
    """Enhanced chatbot using fixed retrieval"""
    
    def __init__(self, **kwargs):
        self.chatbot = FixedChatbot()
        self.chatbot.initialize()
    
    def chat(self, query: str) -> str:
        return self.chatbot.answer(query)
    
    def process_query(self, query: str) -> str:
        return self.chat(query)

# For testing
if __name__ == "__main__":
    chatbot = EnhancedTechnicalChatbot()
    
    # Test queries
    test_queries = [
        "Were there any maintenance patches related to eval tasks?",
        "What Node.js version is required for IAP 2023.1?",
        "List all bug fixes for evaluation"
    ]
    
    for query in test_queries:
        print(f"\nQ: {query}")
        print(f"A: {chatbot.chat(query)[:500]}...")
