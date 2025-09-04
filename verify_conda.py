import subprocess
import sys
import os

print("Verifying Miniconda Installation...")
print("="*50)

# Test conda
try:
    result = subprocess.run([r"C:\miniconda3\Scripts\conda.exe", "--version"], 
                          capture_output=True, text=True)
    print(f"[PASS] Conda Version: {result.stdout.strip()}")
except Exception as e:
    print(f"[FAIL] Conda not found: {e}")
    sys.exit(1)

# Check if environment exists
try:
    result = subprocess.run([r"C:\miniconda3\Scripts\conda.exe", "env", "list"], 
                          capture_output=True, text=True)
    if "props_predictor" in result.stdout:
        print("[PASS] Environment 'props_predictor' exists")
    else:
        print("[INFO] Environment 'props_predictor' not found. Creating...")
        # Create environment
        result = subprocess.run([r"C:\miniconda3\Scripts\conda.exe", "create", 
                               "-n", "props_predictor", "python=3.10", "-y"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("[PASS] Environment created successfully")
        else:
            print(f"[FAIL] Failed to create environment: {result.stderr}")
except Exception as e:
    print(f"[FAIL] Error checking environments: {e}")

print("\n" + "="*50)
print("Next steps:")
print("1. Open Anaconda Prompt (from Start Menu)")
print("2. Run: conda activate props_predictor")
print("3. Run: python --version")
print("4. Verify Python 3.10 is active")