#!/usr/bin/env python3
"""
Matrix multiplication test case generator for K5 FPGA accelerator
Generates random test matrices and expected results for verification
"""

import numpy as np
import struct
import os

def generate_matrix_test():
    """Generate matrix multiplication test case"""
    
    # Small matrices for initial testing (can be scaled up)
    rows_a = 4
    cols_a = 4  
    cols_b = 4
    
    print(f"Generating {rows_a}x{cols_a} * {cols_a}x{cols_b} matrix test case")
    
    # Generate random matrices with small integers for easier debugging
    np.random.seed(42)  # Reproducible results
    matrix_a = np.random.randint(1, 8, (rows_a, cols_a), dtype=np.int16)
    matrix_b = np.random.randint(1, 8, (cols_a, cols_b), dtype=np.int16)
    
    # Calculate expected result
    matrix_c_expected = np.matmul(matrix_a, matrix_b).astype(np.int16)
    
    print("Matrix A:")
    print(matrix_a)
    print("\nMatrix B:")  
    print(matrix_b)
    print("\nExpected Result C = A * B:")
    print(matrix_c_expected)
    
    # Calculate matrix sizes in bytes
    matrix_a_size = rows_a * cols_a * 2  # 2 bytes per int16
    matrix_b_size = cols_a * cols_b * 2
    matrix_c_size = rows_a * cols_b * 2
    
    # Allocate memory addresses (simulated)
    matrix_a_addr = 0x10000000
    matrix_b_addr = matrix_a_addr + matrix_a_size
    matrix_c_addr = matrix_b_addr + matrix_b_size
    
    print(f"\nMatrix A address: 0x{matrix_a_addr:08x} ({matrix_a_size} bytes)")
    print(f"Matrix B address: 0x{matrix_b_addr:08x} ({matrix_b_size} bytes)")
    print(f"Matrix C address: 0x{matrix_c_addr:08x} ({matrix_c_size} bytes)")
    
    # Create configuration structure matching C struct
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
    
    # Write configuration file (binary format for C program)
    with open('matrix_test_config.txt', 'wb') as f:
        f.write(config_data)
    
    # Write matrix data file (Matrix A followed by Matrix B)
    matrix_data = np.concatenate([matrix_a.flatten(), matrix_b.flatten()])
    with open('matrix_test_in.txt', 'wb') as f:
        f.write(matrix_data.tobytes())
    
    # Save expected result for verification
    with open('matrix_expected.txt', 'wb') as f:
        f.write(matrix_c_expected.tobytes())
    
    # Also create human-readable versions
    with open('matrix_test_debug.txt', 'w') as f:
        f.write(f"Test Configuration:\n")
        f.write(f"Matrix A ({rows_a}x{cols_a}):\n{matrix_a}\n\n")
        f.write(f"Matrix B ({cols_a}x{cols_b}):\n{matrix_b}\n\n") 
        f.write(f"Expected Result C ({rows_a}x{cols_b}):\n{matrix_c_expected}\n\n")
        f.write(f"Memory Layout:\n")
        f.write(f"A: 0x{matrix_a_addr:08x} ({matrix_a_size} bytes)\n")
        f.write(f"B: 0x{matrix_b_addr:08x} ({matrix_b_size} bytes)\n")
        f.write(f"C: 0x{matrix_c_addr:08x} ({matrix_c_size} bytes)\n")
    
    print(f"\nGenerated files:")
    print(f"  matrix_test_config.txt - Configuration data ({len(config_data)} bytes)")
    print(f"  matrix_test_in.txt - Input matrices ({len(matrix_data) * 2} bytes)")
    print(f"  matrix_expected.txt - Expected result ({matrix_c_size} bytes)")
    print(f"  matrix_test_debug.txt - Human readable debug info")

if __name__ == "__main__":
    # Create app_src_dir if it doesn't exist
    os.makedirs("app_src_dir", exist_ok=True)
    generate_matrix_test()