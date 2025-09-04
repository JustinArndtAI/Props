@echo off
echo ========================================
echo NFL Props Predictor Environment Setup
echo ========================================
echo.

REM Check if conda is installed
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo Conda not found! Please install Miniconda first.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo Creating conda environment...
conda create -n props_predictor python=3.10 -y

echo.
echo Activating environment...
call conda activate props_predictor

echo.
echo Installing PyTorch with CUDA support...
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo.
echo Installing other dependencies...
pip install -r requirements.txt

echo.
echo Verifying GPU support...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else None}')"

echo.
echo ========================================
echo Setup complete!
echo To activate the environment, run:
echo    conda activate props_predictor
echo ========================================
pause