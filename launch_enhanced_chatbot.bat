@echo off
REM Enhanced Itential Chatbot Launcher

echo Starting Enhanced Itential Chatbot...

REM Check if Ollama is running
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo Ollama not running. Please start with: ollama serve
    pause
    exit /b 1
)

echo Ollama is running

REM Launch Streamlit app
echo Starting Streamlit app...
streamlit run app/enhanced_ui.py --server.port 8501 --server.headless true

pause
