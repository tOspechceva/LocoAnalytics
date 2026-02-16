@echo off
chcp 65001 > nul
echo ==========================================
echo    Locomotive Analytics - Windows Launcher
echo ==========================================

:: 1. Check Python
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found! Please install Python 3.9+ and add it to PATH.
    pause
    exit /b
)

:: 2. Setup venv
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: 3. Install dependencies
echo Installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt

:: 4. Run App
echo Starting Application...
streamlit run app.py

pause
