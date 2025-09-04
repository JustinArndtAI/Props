@echo off
echo Starting NFL Props Predictor Web App...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat 2>nul

REM Check if venv exists
if %errorlevel% neq 0 (
    echo Virtual environment not found!
    echo Please run setup_venv.bat first
    pause
    exit /b 1
)

REM Run streamlit app
echo Opening web browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run main.py

pause