@echo off
echo Starting Round Robin Calculator...
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

REM Run the calculator
python utils\round_robin_calculator.py

echo.
pause