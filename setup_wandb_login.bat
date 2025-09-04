@echo off
echo Setting up Weights & Biases (wandb) with API key...
echo =====================================================
echo.

REM Activate conda environment
call C:\miniconda3\Scripts\activate.bat props_predictor

REM Set wandb API key
set WANDB_API_KEY=1e2979700d05e29de10f2857b8465805cbd5ffe7

echo Setting WANDB_API_KEY environment variable...
setx WANDB_API_KEY "1e2979700d05e29de10f2857b8465805cbd5ffe7"

REM Login to wandb
echo.
echo Logging into wandb...
python -c "import wandb; wandb.login(key='1e2979700d05e29de10f2857b8465805cbd5ffe7')"

REM Test wandb
echo.
echo Testing wandb setup...
python -c "import wandb; print(f'wandb version: {wandb.__version__}'); print('wandb login successful!')"

echo.
echo =====================================================
echo wandb setup complete!
echo API key has been saved to your environment
echo =====================================================
pause