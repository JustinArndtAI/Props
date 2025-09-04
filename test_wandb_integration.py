"""
Test wandb integration with the API key
"""

import os
import wandb

# Set API key
os.environ['WANDB_API_KEY'] = '1e2979700d05e29de10f2857b8465805cbd5ffe7'

print("Testing wandb Integration")
print("="*60)

try:
    # Login with API key
    wandb.login(key='1e2979700d05e29de10f2857b8465805cbd5ffe7')
    print("[PASS] wandb login successful")
    
    # Initialize a test run
    run = wandb.init(
        project="nfl-props-test",
        name="test-run",
        config={
            "test": True,
            "gpu": "RTX 4060",
            "framework": "pytorch"
        }
    )
    
    print(f"[PASS] wandb run initialized: {run.name}")
    
    # Log some test metrics
    wandb.log({"test_metric": 0.95, "test_loss": 0.05})
    print("[PASS] Metrics logged")
    
    # Finish run
    wandb.finish()
    print("[PASS] wandb run completed")
    
    print("\n" + "="*60)
    print("wandb integration successful!")
    print("You can view your runs at: https://wandb.ai/")
    
except Exception as e:
    print(f"[FAIL] wandb error: {e}")
    print("\nPlease run: setup_wandb_login.bat")