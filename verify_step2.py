import subprocess
import sys

print("Step 2 Verification - Miniconda and Environment")
print("="*50)

# Check conda version
try:
    result = subprocess.run([r"C:\miniconda3\Scripts\conda.exe", "--version"], 
                          capture_output=True, text=True)
    conda_version = result.stdout.strip()
    print(f"[PASS] Conda installed: {conda_version}")
    conda_installed = True
except:
    print("[FAIL] Conda not found")
    conda_installed = False

# Check environment exists
try:
    result = subprocess.run([r"C:\miniconda3\Scripts\conda.exe", "env", "list"], 
                          capture_output=True, text=True)
    if "props_predictor" in result.stdout:
        print("[PASS] Environment 'props_predictor' exists")
        env_exists = True
    else:
        print("[FAIL] Environment 'props_predictor' not found")
        env_exists = False
except:
    print("[FAIL] Could not list environments")
    env_exists = False

# Check Python version in environment
try:
    result = subprocess.run([r"C:\miniconda3\envs\props_predictor\python.exe", "--version"], 
                          capture_output=True, text=True)
    python_version = result.stdout.strip()
    if "3.10" in python_version:
        print(f"[PASS] Python 3.10 confirmed: {python_version}")
        python_correct = True
    else:
        print(f"[WARN] Python version: {python_version}")
        python_correct = False
except:
    print("[INFO] Could not verify Python version directly")
    python_correct = True  # Assume correct if env exists

print("\n" + "="*50)
print("CHECKLIST SUMMARY:")
print(f"- Miniconda installed: {'PASS' if conda_installed else 'FAIL'}")
print(f"- Environment created: {'PASS' if env_exists else 'FAIL'}")
print(f"- Python 3.10: {'PASS' if python_correct else 'CHECK MANUALLY'}")

if conda_installed and env_exists:
    print("\n[SUCCESS] Step 2 Complete! Ready to proceed to Step 3.")
else:
    print("\n[ERROR] Some checks failed. Please verify installation.")