#!/usr/bin/env python3
"""
Generate matrix test data files in K5 hex format
K5 system expects hex text files, not binary files
"""

import numpy as np
import struct

def generate_k5_hex_matrix_test():
    """Generate matrix test data in K5 hex format"""
    
    print("🔧 Generating K5 Matrix Test Data (HEX FORMAT)")
    print("=" * 50)
    
    # Small matrices for testing
    rows_a = 4
    cols_a = 4  
    cols_b = 4
    
    print(f"Matrix dimensions: {rows_a}x{cols_a} * {cols_a}x{cols_b}")
    
    # Generate test matrices
    np.random.seed(42)
    matrix_a = np.random.randint(1, 8, (rows_a, cols_a), dtype=np.int16)
    matrix_b = np.random.randint(1, 8, (cols_a, cols_b), dtype=np.int16)
    
    # Calculate expected result
    matrix_c_expected = np.matmul(matrix_a, matrix_b).astype(np.int16)
    
    print("\nMatrix A:")
    print(matrix_a)
    print("\nMatrix B:")  
    print(matrix_b)
    print("\nExpected Result C:")
    print(matrix_c_expected)
    
    # Memory addresses for K5
    matrix_a_size = rows_a * cols_a * 2
    matrix_b_size = cols_a * cols_b * 2
    matrix_c_size = rows_a * cols_b * 2
    
    matrix_a_addr = 0x10000000
    matrix_b_addr = matrix_a_addr + matrix_a_size
    matrix_c_addr = matrix_b_addr + matrix_b_size
    
    # 1. Generate configuration file in HEX FORMAT
    config_data = struct.pack('<IIIIIIIII', 
                             rows_a, cols_a, cols_b,
                             matrix_a_addr, matrix_b_addr, matrix_c_addr,
                             matrix_a_size, matrix_b_size, matrix_c_size)
    
    print(f"\n📁 Creating matrix_test_config.txt (HEX format)")
    with open('matrix_test_config.txt', 'w') as f:
        hex_bytes = []
        for byte in config_data:
            hex_bytes.append(f"{byte:02x}")
        # K5 expects hex bytes, typically 16 per line
        for i in range(0, len(hex_bytes), 16):
            line = ' '.join(hex_bytes[i:i+16])
            f.write(line + '\n')
    
    # 2. Generate matrix input data in HEX FORMAT  
    matrix_data = np.concatenate([matrix_a.flatten(), matrix_b.flatten()])
    matrix_bytes = matrix_data.tobytes()
    
    print(f"📁 Creating matrix_test_in.txt (HEX format)")
    with open('matrix_test_in.txt', 'w') as f:
        hex_bytes = []
        for byte in matrix_bytes:
            hex_bytes.append(f"{byte:02x}")
        # K5 expects hex bytes, typically 16 per line
        for i in range(0, len(hex_bytes), 16):
            line = ' '.join(hex_bytes[i:i+16])
            f.write(line + '\n')
    
    # 3. Create expected result for verification
    result_bytes = matrix_c_expected.tobytes()
    print(f"📁 Creating matrix_expected.txt (HEX format)")
    with open('matrix_expected.txt', 'w') as f:
        hex_bytes = []
        for byte in result_bytes:
            hex_bytes.append(f"{byte:02x}")
        for i in range(0, len(hex_bytes), 16):
            line = ' '.join(hex_bytes[i:i+16])
            f.write(line + '\n')
    
    # 4. Create debug info
    with open('matrix_test_debug.txt', 'w') as f:
        f.write(f"K5 Matrix Test (HEX Format)\n")
        f.write(f"============================\n\n")
        f.write(f"Dimensions: {rows_a}x{cols_a} * {cols_a}x{cols_b} = {rows_a}x{cols_b}\n")
        f.write(f"Config size: {len(config_data)} bytes\n")
        f.write(f"Input size: {len(matrix_bytes)} bytes\n")  
        f.write(f"Result size: {len(result_bytes)} bytes\n\n")
        f.write(f"Matrix A:\n{matrix_a}\n\n")
        f.write(f"Matrix B:\n{matrix_b}\n\n")
        f.write(f"Expected C:\n{matrix_c_expected}\n")
    
    print(f"\n✅ Generated K5 hex format files:")
    print(f"  📄 matrix_test_config.txt ({len(config_data)} bytes -> hex)")
    print(f"  📄 matrix_test_in.txt ({len(matrix_bytes)} bytes -> hex)")
    print(f"  📄 matrix_expected.txt ({len(result_bytes)} bytes -> hex)")
    print(f"  📄 matrix_test_debug.txt (debug info)")
    
    print(f"\n🎯 Copy these files to your K5 directory:")
    print(f"C:/Users/user/Desktop/computer_engineering/Final_Project/k5_xbox_env/k5_xbox_fpga_win/threads_runspace/t0/")

if __name__ == "__main__":
    generate_k5_hex_matrix_test()