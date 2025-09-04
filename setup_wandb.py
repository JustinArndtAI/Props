"""
Setup Weights & Biases for experiment tracking
"""

print("="*60)
print("Weights & Biases (wandb) Setup")
print("="*60)
print()

print("wandb is installed and ready for use!")
print()
print("To login and start using wandb:")
print("1. Get your API key from: https://wandb.ai/authorize")
print("2. Run in Anaconda Prompt:")
print("   conda activate props_predictor")
print("   wandb login")
print("3. Paste your API key when prompted")
print()
print("For this project, wandb will be used to track:")
print("- Model training metrics")
print("- Hyperparameter optimization")
print("- Prediction accuracy over time")
print("- GPU usage and performance")
print()

# Test import
try:
    import wandb
    print(f"[PASS] wandb version: {wandb.__version__}")
except ImportError:
    print("[FAIL] wandb not installed properly")

print()
print("="*60)