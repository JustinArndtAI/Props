@echo off
echo ========================================
echo NFL Props Predictor Environment Setup
echo Using Python venv (no conda required)
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found! Please install Python 3.10 or higher.
    echo Download from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found:
python --version

echo.
echo Creating virtual environment...
python -m venv venv

echo.
echo Activating environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing PyTorch (CPU version for compatibility)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installing core dependencies...
pip install pandas numpy scikit-learn xgboost matplotlib seaborn

echo.
echo Installing additional packages...
pip install streamlit requests beautifulsoup4 sqlalchemy openpyxl tqdm python-dotenv pytest joblib

echo.
echo Installing NFL data package...
pip install nfl-data-py

echo.
echo Verifying installation...
python -c "import torch; print(f'PyTorch installed: {torch.__version__}')"
python -c "import xgboost; print(f'XGBoost installed: {xgboost.__version__}')"
python -c "import streamlit; print(f'Streamlit installed: {streamlit.__version__}')"

echo.
echo ========================================
echo Setup complete!
echo.
echo To activate the environment in the future, run:
echo    venv\Scripts\activate
echo.
echo To test the Round Robin Calculator, run:
echo    python utils\round_robin_calculator.py
echo ========================================
pause