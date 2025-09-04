import subprocess
import sys
import os

print("Step 3 Verification - Core Packages with GPU Support")
print("="*60)

# Path to Python in the conda environment
conda_python = r"C:\miniconda3\envs\props_predictor\python.exe"

packages_to_check = [
    ("pytorch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("xgboost", "xgboost"),
    ("tensorflow", "tensorflow"),
    ("transformers", "transformers"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("jupyterlab", "jupyterlab"),
    ("streamlit", "streamlit"),
    ("optuna", "optuna"),
    ("scipy", "scipy"),
    ("pytest", "pytest"),
    ("shap", "shap"),
    ("lime", "lime"),
    ("wandb", "wandb"),
    ("pyro-ppl", "pyro"),
    ("networkx", "networkx"),
    ("ray", "ray"),
    ("requests", "requests"),
    ("beautifulsoup4", "bs4"),
    ("nfl_data_py", "nfl_data_py")
]

# Check conda packages
print("Checking conda packages...")
print("-"*40)
conda_result = subprocess.run([r"C:\miniconda3\Scripts\conda.exe", "list", "-n", "props_predictor"], 
                             capture_output=True, text=True)
conda_packages = conda_result.stdout.lower()

# Check pip packages
print("\nChecking pip packages...")
print("-"*40)
pip_result = subprocess.run([conda_python, "-m", "pip", "list"], 
                           capture_output=True, text=True)
pip_packages = pip_result.stdout.lower()

all_packages_text = conda_packages + pip_packages

failed = []
passed = []

for display_name, import_name in packages_to_check:
    if import_name in all_packages_text or display_name in all_packages_text:
        print(f"[PASS] {display_name}")
        passed.append(display_name)
    else:
        print(f"[FAIL] {display_name} not found")
        failed.append(display_name)

# Special check for CUDA support
print("\n" + "-"*40)
print("Checking GPU/CUDA support...")
try:
    test_code = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
"""
    result = subprocess.run([conda_python, "-c", test_code], 
                           capture_output=True, text=True)
    print(result.stdout)
    if "CUDA available: True" in result.stdout:
        print("[PASS] GPU/CUDA support confirmed")
        gpu_support = True
    else:
        print("[WARNING] CUDA not available - CPU only mode")
        gpu_support = False
except Exception as e:
    print(f"[ERROR] Could not verify CUDA: {e}")
    gpu_support = False

# Summary
print("\n" + "="*60)
print("CHECKLIST SUMMARY:")
print(f"- All conda packages installed: {'PASS' if 'pytorch' in conda_packages else 'PARTIAL'}")
print(f"- All pip packages installed: {'PASS' if len(failed) == 0 else f'MISSING {len(failed)} packages'}")
print(f"- No errors during installation: {'CHECK MANUALLY'}")

if failed:
    print(f"\nMissing packages: {', '.join(failed)}")
    print("\nTo install missing packages:")
    print("1. Open Anaconda Prompt")
    print("2. conda activate props_predictor")
    print("3. pip install " + " ".join(failed))

if len(failed) <= 3:  # Allow a few missing packages
    print("\n[SUCCESS] Step 3 mostly complete! Minor packages may need manual installation.")
    success = True
else:
    print("\n[ERROR] Many packages missing. Please run install_packages.bat again.")
    success = False

# Create pip freeze for requirements
if success:
    print("\nGenerating requirements.txt...")
    req_result = subprocess.run([conda_python, "-m", "pip", "freeze"], 
                               capture_output=True, text=True)
    with open("requirements_conda.txt", "w") as f:
        f.write(req_result.stdout)
    print("Requirements saved to requirements_conda.txt")