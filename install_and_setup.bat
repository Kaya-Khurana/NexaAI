@echo off
echo ==========================================
echo   NexaAI Universal Engine - Setup
echo ==========================================
echo.
echo [1/3] Creating virtual environment...
python -m venv venv

echo [2/3] Installing dependencies...
call venv\Scripts\activate
pip install -r requirements.txt

echo [3/3] Creating configuration...
if not exist .env (
    echo OPENROUTER_API_KEY=your_key_here > .env
    echo.
    echo IMPORTANT: Open '.env' and replace 'your_key_here' with your NVIDIA/OpenRouter API Key.
)

echo.
echo ==========================================
echo   SETUP COMPLETE!
echo   1. Add your API key to the .env file.
echo   2. Double-click 'Launch_NexaAI.bat' to start.
echo ==========================================
pause
