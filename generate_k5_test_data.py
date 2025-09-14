#!/usr/bin/env python3
"""
Generate matrix test data files for K5 system
Creates files in the format expected by the K5 framework
"""

import numpy as np
import struct
import os
import sys

def generate_k5_matrix_test():
    """Generate matrix multiplication test case for K5 system"""
    
    print("🔧 Generating K5 Matrix Test Data")
    print("="*40)
    
    # Small matrices for initial testing
    rows_a = 4
    cols_a = 4  
    cols_b = 4
    
    print(f"Matrix dimensions: {rows_a}x{cols_a} * {cols_a}x{cols_b}")
    
    # Generate random matrices with small integers for easier debugging
    np.random.seed(42)  # Reproducible results
    matrix_a = np.random.randint(1, 8, (rows_a, cols_a), dtype=np.int16)
    matrix_b = np.random.randint(1, 8, (cols_a, cols_b), dtype=np.int16)
    
    # Calculate expected result
    matrix_c_expected = np.matmul(matrix_a, matrix_b).astype(np.int16)
    
    print("\nMatrix A:")
    print(matrix_a)
    print("\nMatrix B:")  
    print(matrix_b)
    print("\nExpected Result C = A * B:")
    print(matrix_c_expected)
    
    # Calculate matrix sizes in bytes
    matrix_a_size = rows_a * cols_a * 2  # 2 bytes per int16
    matrix_b_size = cols_a * cols_b * 2
    matrix_c_size = rows_a * cols_b * 2
    
    # Allocate memory addresses (simulated for K5)
    matrix_a_addr = 0x10000000
    matrix_b_addr = matrix_a_addr + matrix_a_size
    matrix_c_addr = matrix_b_addr + matrix_b_size
    
    print(f"\nMemory Layout:")
    print(f"  Matrix A: 0x{matrix_a_addr:08x} ({matrix_a_size} bytes)")
    print(f"  Matrix B: 0x{matrix_b_addr:08x} ({matrix_b_size} bytes)")
    print(f"  Matrix C: 0x{matrix_c_addr:08x} ({matrix_c_size} bytes)")
    
    # Create configuration structure matching C struct (matrix_config_t)
    config_data = struct.pack('<IIIIIIIII', 
                             rows_a,              # int rows_a
                             cols_a,              # int cols_a  
                             cols_b,              # int cols_b
                             matrix_a_addr,       # short * matrix_a_addr
                             matrix_b_addr,       # short * matrix_b_addr
                             matrix_c_addr,       # short * matrix_c_addr
                             matrix_a_size,       # int matrix_a_size
                             matrix_b_size,       # int matrix_b_size
                             matrix_c_size)       # int matrix_c_size
    
    # Create output files
    files_created = []
    
    # 1. Configuration file (binary format for C program)
    config_file = 'matrix_test_config.txt'
    with open(config_file, 'wb') as f:
        f.write(config_data)
    files_created.append(f"{config_file} ({len(config_data)} bytes)")
    
    # 2. Input data file (Matrix A followed by Matrix B, binary format)
    matrix_data = np.concatenate([matrix_a.flatten(), matrix_b.flatten()])
    input_file = 'matrix_test_in.txt'
    with open(input_file, 'wb') as f:
        f.write(matrix_data.tobytes())
    files_created.append(f"{input_file} ({len(matrix_data) * 2} bytes)")
    
    # 3. Expected result (for verification)
    expected_file = 'matrix_expected.txt'
    with open(expected_file, 'wb') as f:
        f.write(matrix_c_expected.tobytes())
    files_created.append(f"{expected_file} ({matrix_c_size} bytes)")
    
    # 4. Debug info (human readable)
    debug_file = 'matrix_test_debug.txt'
    with open(debug_file, 'w') as f:
        f.write(f"K5 Matrix Multiplication Test Case\n")
        f.write(f"==================================\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Matrix A: {rows_a} x {cols_a}\n")
        f.write(f"  Matrix B: {cols_a} x {cols_b}\n")
        f.write(f"  Result C: {rows_a} x {cols_b}\n\n")
        f.write(f"Memory Layout:\n")
        f.write(f"  A: 0x{matrix_a_addr:08x} ({matrix_a_size} bytes)\n")
        f.write(f"  B: 0x{matrix_b_addr:08x} ({matrix_b_size} bytes)\n")
        f.write(f"  C: 0x{matrix_c_addr:08x} ({matrix_c_size} bytes)\n\n")
        f.write(f"Matrix A:\n{matrix_a}\n\n")
        f.write(f"Matrix B:\n{matrix_b}\n\n")
        f.write(f"Expected Result C:\n{matrix_c_expected}\n")
    files_created.append(f"{debug_file} (debug info)")
    
    # 5. Create app_src_dir and python scripts
    os.makedirs("app_src_dir", exist_ok=True)
    
    # Copy the Python scripts to local directory as well
    gen_script_content = open('/Users/shalevdeutsch/Documents/claude_trial/app_src_dir/gen_matrix_test.py', 'r').read()
    with open('app_src_dir/gen_matrix_test.py', 'w') as f:
        f.write(gen_script_content)
    
    check_script_content = open('/Users/shalevdeutsch/Documents/claude_trial/app_src_dir/check_matrix_result.py', 'r').read()
    with open('app_src_dir/check_matrix_result.py', 'w') as f:
        f.write(check_script_content)
    
    files_created.append("app_src_dir/gen_matrix_test.py")
    files_created.append("app_src_dir/check_matrix_result.py")
    
    print(f"\n✅ Successfully generated test files:")
    for file_info in files_created:
        print(f"  📁 {file_info}")
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Copy these files to your K5 runspace directory:")
    print(f"   C:/Users/user/Desktop/computer_engineering/Final_Project/k5_xbox_env/k5_xbox_fpga_win/threads_runspace/t0/")
    print(f"2. Run your C program again")
    print(f"3. The program should now find the test files and execute!")

if __name__ == "__main__":
    generate_k5_matrix_test()