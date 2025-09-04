@echo off
echo Setting up Conda Environment for Props Predictor
echo ==================================================
echo.

REM Accept Terms of Service
echo Accepting Conda Terms of Service...
C:\miniconda3\Scripts\conda.exe tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
C:\miniconda3\Scripts\conda.exe tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
C:\miniconda3\Scripts\conda.exe tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2

echo.
echo Creating props_predictor environment with Python 3.10...
C:\miniconda3\Scripts\conda.exe create -n props_predictor python=3.10 -y

echo.
echo Environment created! Listing all environments:
C:\miniconda3\Scripts\conda.exe env list

echo.
echo ==================================================
echo Setup complete!
echo.
echo To activate the environment:
echo   1. Open Anaconda Prompt
echo   2. Run: conda activate props_predictor
echo ==================================================
pause