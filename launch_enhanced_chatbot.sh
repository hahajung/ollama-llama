#!/bin/bash
# Enhanced Itential Chatbot Launcher

echo "Starting Enhanced Itential Chatbot..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Ollama not running. Please start with: ollama serve"
    exit 1
fi

echo "Ollama is running"

# Launch Streamlit app
echo "Starting Streamlit app..."
streamlit run app/enhanced_ui.py --server.port 8501 --server.headless true
