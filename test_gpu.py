import torch
import tensorflow as tf
import sys
import os

print("="*60)
print("GPU/CUDA Verification Test")
print("="*60)

# PyTorch GPU Test
print("\n1. PyTorch GPU Test:")
print("-"*40)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Test tensor operations on GPU
    try:
        x = torch.rand(5, 3).cuda()
        y = torch.rand(5, 3).cuda()
        z = x + y
        print(f"GPU tensor operation test: PASS")
    except Exception as e:
        print(f"GPU tensor operation test: FAIL - {e}")
else:
    print("WARNING: CUDA not available for PyTorch")

# TensorFlow GPU Test
print("\n2. TensorFlow GPU Test:")
print("-"*40)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

if tf.config.list_physical_devices('GPU'):
    gpu_devices = tf.config.list_physical_devices('GPU')
    for device in gpu_devices:
        print(f"GPU device: {device}")
    
    # Test tensor operations on GPU
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
        print(f"GPU tensor operation test: PASS")
    except Exception as e:
        print(f"GPU tensor operation test: FAIL - {e}")
else:
    print("WARNING: No GPUs found for TensorFlow")

print("\n" + "="*60)
print("GPU Verification Complete")
print("="*60)