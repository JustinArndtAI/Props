@echo off
echo Testing Miniconda installation...
echo.

REM Test conda from miniconda3 path
C:\miniconda3\Scripts\conda.exe --version

echo.
echo Creating environment...
C:\miniconda3\Scripts\conda.exe create -n props_predictor python=3.10 -y

echo.
echo Listing environments...
C:\miniconda3\Scripts\conda.exe env list

echo.
echo Complete!
pause