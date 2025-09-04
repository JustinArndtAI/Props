@echo off
echo Running Enhanced Predictor with wandb
echo =====================================
echo.

REM Activate conda environment
call C:\miniconda3\Scripts\activate.bat props_predictor

REM Set API key
set WANDB_API_KEY=1e2979700d05e29de10f2857b8465805cbd5ffe7

REM Run the enhanced script
python scripts/predict_props_enhanced.py

pause