#!/usr/bin/env python3
"""
Copy all necessary Python files to K5 server directory for FPGA training
"""
import shutil
import os

# Files that need to be copied to K5 server
files_to_copy = [
    'train.py',
    'model.py', 
    'config.py',
    'dataset.py',
    'k5_fpga_accelerator.py',
    'tokenizers_en.json',
    'tokenizers_he.json',
    'FPGA_TRAINING_INTEGRATION.md',
    'training_cycle_analysis.md'
]

print("📋 Files to copy to K5 server:")
for file in files_to_copy:
    if os.path.exists(file):
        print(f"   ✅ {file}")
    else:
        print(f"   ❌ {file} (missing)")

print("\n🔧 Instructions for K5 Server Setup:")
print("1. Copy these files to your K5 server directory:")
print("   ~/Desktop/computer_engineering/Final_Project/k5_xbox_env/k5_xbox_fpga_win/sw/apps/de10_lite_matrix_multiplier/")
print()
print("2. On K5 server, install required packages:")
print("   pip install torch numpy tqdm datasets tokenizers")
print()
print("3. Create Helsinki-NLP directory and download dataset:")
print("   mkdir -p Helsinki-NLP")
print("   # Dataset will be auto-downloaded on first run")
print()
print("4. Run the FPGA training:")
print("   Terminal 1: launch_k5_app de10_lite_matrix_multiplier -ccd1 XON")
print("   Terminal 2: python3 train.py")
print()
print("🎯 This will give you real FPGA cycle measurements vs CPU training!")