import os
from pathlib import Path

print("="*60)
print("Step 5: Verifying Project Structure")
print("="*60)

base_path = Path("C:/Props_Project")

# Required directories
required_dirs = ["data", "models", "scripts", "notebooks", "tests", "logs", "utils", "config"]

# Check and create directories
for dir_name in required_dirs:
    dir_path = base_path / dir_name
    if dir_path.exists():
        print(f"[PASS] {dir_name}/ exists")
    else:
        dir_path.mkdir(exist_ok=True)
        print(f"[CREATED] {dir_name}/ directory")

# Additional subdirectories
subdirs = {
    "models": ["trained", "checkpoints", "configs"],
    "data": ["raw", "processed", "features"],
    "notebooks": ["exploration", "experiments"],
    "logs": ["training", "predictions"]
}

print("\nCreating subdirectories...")
for parent, subs in subdirs.items():
    for sub in subs:
        sub_path = base_path / parent / sub
        if not sub_path.exists():
            sub_path.mkdir(parents=True, exist_ok=True)
            print(f"  [CREATED] {parent}/{sub}/")
        else:
            print(f"  [EXISTS] {parent}/{sub}/")

# Check Git setup
git_dir = base_path / ".git"
if git_dir.exists():
    print("\n[PASS] Git repository initialized")
    # Check remote
    import subprocess
    result = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True, cwd=str(base_path))
    if "JustinArndtAI/Props" in result.stdout:
        print("[PASS] Remote repository configured")
else:
    print("\n[FAIL] Git not initialized")

print("\n" + "="*60)
print("Project Structure Summary:")
print("="*60)
print("""
C:\\Props_Project\\
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── models/
│   ├── trained/
│   ├── checkpoints/
│   └── configs/
├── scripts/
├── notebooks/
│   ├── exploration/
│   └── experiments/
├── tests/
├── logs/
│   ├── training/
│   └── predictions/
├── utils/
└── config/
""")

print("Project structure ready!")