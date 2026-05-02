@echo off
echo ==========================================
echo   NexaAI Universal Engine - Starting...
echo ==========================================
echo.
echo Launching browser...
timeout /t 2 >nul
start http://127.0.0.1:5000

echo Starting Server...
call venv\Scripts\activate
python app.py
pause
