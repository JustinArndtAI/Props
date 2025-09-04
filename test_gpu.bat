@echo off
echo Testing GPU Support...
call C:\miniconda3\Scripts\activate.bat props_predictor
python test_gpu.py
pause