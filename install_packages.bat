@echo off
echo =====================================================
echo Step 3: Installing Core Packages with GPU Support
echo =====================================================
echo.

REM Activate environment
echo Activating props_predictor environment...
call C:\miniconda3\Scripts\activate.bat props_predictor

echo.
echo Installing PyTorch with CUDA 12.1 support...
call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo.
echo Installing core ML packages...
call pip install pandas numpy scikit-learn xgboost tensorflow transformers

echo.
echo Installing visualization and utilities...
call pip install matplotlib seaborn jupyterlab streamlit

echo.
echo Installing optimization packages...
call pip install optuna scipy pytest

echo.
echo Installing explainability and tracking tools...
call pip install shap lime wandb

echo.
echo Installing advanced ML tools...
call pip install pyro-ppl networkx ray[tune]

echo.
echo Installing NFL data packages...
call pip install requests beautifulsoup4 nfl_data_py

echo.
echo =====================================================
echo Package installation complete!
echo =====================================================
pause